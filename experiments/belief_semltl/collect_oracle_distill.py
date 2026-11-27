"""Collect on-policy GRU+DAgger trajectories with privileged oracle-action targets.

The frozen actor is queried twice on the SAME physical grid: once with the
estimated state (for rollout) and once with the true state (teacher label).
Only epsilon-free LetterWorld tasks are supported. Ground truth never enters
the rollout action selection or the residual adapter's input.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.eval import EvalResetOptions
from jaxolotl.eval.utils import load_batched_actor_critic
from experiments.belief_semltl.collect_dagger import EXPERIMENT, sha256
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper
from experiments.belief_semltl.run_learned_estimator import DEFAULT_FORMULAS, load_estimator
from experiments.belief_semltl.oracle_distill_math import physical_action_probs, validate_dataset

LOG = logging.getLogger("oracle_distill_collect")


class Traces(NamedTuple):
    active: jax.Array
    grid: jax.Array
    hidden: jax.Array
    embedding: jax.Array
    teacher_probs: jax.Array


class Carry(NamedTuple):
    agent: object
    state: object
    observation: object
    completed: jax.Array
    traces: Traces
    idx: jax.Array
    key: jax.Array


@eqx.filter_jit
def collect_rollout(agent, env, params, task, key, num_episodes: int,
                    max_steps: int, deterministic: bool):
    key, reset_key = jax.random.split(key)
    keys = jax.random.split(reset_key, num_episodes)
    state, obs = env.reset(keys, None, params, EvalResetOptions(task=task))
    propositions = eqx.filter_vmap(env.compute_propositions)(state, params)
    agent = agent.update(obs, state, propositions, env)
    grid = obs.features.grid
    hidden = state.estimator_hidden
    embedding = obs.embedding
    trace = Traces(
        jnp.zeros((max_steps, num_episodes), dtype=jnp.bool_),
        jnp.zeros((max_steps,) + grid.shape, dtype=grid.dtype),
        jnp.zeros((max_steps,) + hidden.shape, dtype=hidden.dtype),
        jnp.zeros((max_steps,) + embedding.shape, dtype=embedding.dtype),
        jnp.zeros((max_steps, num_episodes, 4), dtype=jnp.float32),
    )

    def cond(c):
        return (~jnp.all(c.completed)) & (c.idx < max_steps)

    def step(c):
        agent, state, obs, completed, traces, idx, rng = c
        # Privileged teacher is a *training target only*; it cannot affect action selection.
        q_true = state.ldba_state
        true_embedding = jnp.take_along_axis(
            state.ldba.embeddings, q_true[:, None, None], axis=1,
        )[:, 0, :]
        oracle_obs = eqx.tree_at(lambda o: o.embedding, obs, true_embedding)
        teacher_dist = agent.get_action(oracle_obs)
        n_eps = state.ldba.epsilon_transitions.shape[-1]
        targets = physical_action_probs(teacher_dist, num_episodes, n_eps)

        current_dist = agent.get_action(obs)
        if deterministic:
            action = current_dist.mode()
        else:
            rng, action_key = jax.random.split(rng)
            action = current_dist.sample(seed=action_key)
        rng, step_key = jax.random.split(rng)
        transition = env.step(jax.random.split(step_key, num_episodes), state, action, params)
        propositions = eqx.filter_vmap(env.compute_propositions)(transition.state, params)
        agent = agent.update(transition.observation, transition.state, propositions, env)
        active = ~completed
        new_traces = Traces(
            traces.active.at[idx].set(active),
            traces.grid.at[idx].set(obs.features.grid),
            traces.hidden.at[idx].set(state.estimator_hidden),
            traces.embedding.at[idx].set(obs.embedding),
            traces.teacher_probs.at[idx].set(targets),
        )
        return Carry(agent, transition.state, transition.observation,
                     completed | transition.done, new_traces, idx + 1, rng)

    final = eqx_utils.filter_while_loop(
        cond, step,
        Carry(agent, state, obs, jnp.zeros(num_episodes, dtype=jnp.bool_),
              trace, jnp.array(0, dtype=jnp.int32), key),
    )
    return final.traces


def flatten_active(traces, limit: int, random_key: int) -> dict:
    active = np.asarray(traces.active).T.reshape(-1)
    if not active.any():
        raise RuntimeError("No active distillation timesteps collected")
    result = {}
    for name in ("grid", "hidden", "embedding", "teacher_probs"):
        raw = np.asarray(getattr(traces, name))
        raw = raw.swapaxes(0, 1).reshape((-1,) + raw.shape[2:])
        result[name] = raw[active]
    n = len(result["grid"])
    if limit and n > limit:
        selection = np.sort(np.random.default_rng(random_key).choice(n, limit, replace=False))
        result = {k: v[selection] for k, v in result.items()}
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--examples-per-condition", type=int, default=128)
    p.add_argument("--model-seeds", type=int, nargs="+", default=list(range(7)))
    p.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.10, 0.30])
    a = p.parse_args()
    split = "smoke" if a.smoke else "full"
    if a.smoke:
        a.episodes = 4
        a.examples_per_condition = 48
        a.model_seeds = [0, 1]
        a.formula_indices = [0, 3]
        a.noise_levels = [0.30]
    a.model_path = a.model_path or EXPERIMENT / "models" / "dagger" / split / "learned_estimator.eqx"
    a.output_dir = a.output_dir or EXPERIMENT / "data" / "distillation" / split
    if not a.model_seeds or len(set(a.model_seeds)) != len(a.model_seeds) or not set(a.model_seeds).issubset(set(range(7))):
        p.error("Only unique training policy seeds 0..6 are allowed")
    if not a.formula_indices or len(set(a.formula_indices)) != len(a.formula_indices) or 5 in a.formula_indices:
        p.error("Use unique epsilon-free formula indices")
    if a.episodes < 1 or a.examples_per_condition < 1:
        p.error("episodes and examples-per-condition must be positive")
    if not a.noise_levels or any(not (0 < x <= 1) for x in a.noise_levels):
        p.error("Use nonzero noise probabilities in (0, 1]")
    return a


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data_path = args.output_dir / "train.npz"
    meta_path = args.output_dir / "metadata.json"
    if data_path.exists() or meta_path.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)

    with initialize_config_dir(version_base="1.3", config_dir=str((ROOT / "conf").resolve())):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])
    originals = list(cfg.formulas)
    if any(i < 0 or i >= len(originals) for i in args.formula_indices):
        raise ValueError("Formula indices out of range")
    cfg.formulas = [originals[i] for i in args.formula_indices]
    base_env, params = jaxolotl.make(cfg.env.name, reset_source="test", **cfg.get("env_params", {}))
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Distillation only supports epsilon-free formulas")

    estimator = load_estimator(args.model_path)
    models, available = load_batched_actor_critic(cfg, base_env, params, key=jax.random.PRNGKey(42))
    available = list(map(int, available))
    if not set(args.model_seeds).issubset(available):
        raise ValueError(f"Missing pretrained policy seeds, available={available}")
    parts = {k: [] for k in ("grid", "hidden", "embedding", "teacher_probs", "model_seeds", "noise_probs")}
    count = 0
    for seed in args.model_seeds:
        index = available.index(seed)
        single = jax.tree.map(
            lambda x: x[index] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == len(available) else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, single)
        for f, formula_index in enumerate(args.formula_indices):
            task = jax.tree.map(lambda x: x[f], formulas)
            for noise in args.noise_levels:
                key = jax.random.fold_in(jax.random.key(5124), seed)
                key = jax.random.fold_in(key, formula_index)
                key = jax.random.fold_in(key, int(round(noise * 1_000_000)))
                wrapper = VectorizeWrapper(LearnedTrackerWrapper(
                    base_env, estimator=estimator, noise_prob=noise, overwrite_finite=True,
                ))
                traces = collect_rollout(agent, wrapper, params, task, key,
                                         args.episodes, int(params.max_steps_in_episode),
                                         bool(cfg.eval.deterministic))
                sampled = flatten_active(traces, args.examples_per_condition,
                                         23 + seed * 1000 + formula_index * 10 + int(noise * 100))
                n = len(sampled["grid"])
                for name in ("grid", "hidden", "embedding", "teacher_probs"):
                    parts[name].append(sampled[name])
                parts["model_seeds"].append(np.full(n, seed, np.int32))
                parts["noise_probs"].append(np.full(n, noise, np.float32))
                count += n
                LOG.info("seed=%d formula=%d p=%.2f examples=%d", seed, formula_index, noise, n)
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    validate_dataset(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with data_path.open("xb") as f:
        np.savez_compressed(f, **data)
    with meta_path.open("x", encoding="utf-8") as f:
        json.dump({
            "method": "privileged oracle-policy targets on DAgger-GRU-MAP rollouts",
            "source_dagger_model": str(args.model_path),
            "source_dagger_sha256": sha256(args.model_path),
            "model_seeds": args.model_seeds, "formula_indices": args.formula_indices,
            "noise_levels": args.noise_levels, "episodes_per_condition": args.episodes,
            "examples_per_condition": args.examples_per_condition, "total_examples": count,
            "teacher": "same grid as student; true LDBA embedding only to form supervision label",
            "student": "GRU hidden and estimated-state embedding; no true state",
            "no_epsilon_transitions": True, "base_random_key": 5124,
            "original_checkpoint_selection_caveat": "original GRU was selected on seeds 7..9",
        }, f, indent=2)
    LOG.info("Saved %d supervised action targets to %s", count, data_path)


if __name__ == "__main__":
    main()
