"""Generate trajectory datasets for training and testing learned task-state estimators.

Collects rollouts using the pretrained frozen SemLTL policies under noisy proposition
observations, recording:
- Inputs: noisy_assignment (int32), prev_action (int32), hard_state (int32)
- Labels: true_state (int32), true_assignment (int32); optional for structured filter
- Metadata: active_mask (bool), num_states (int32)

Usage:
    MPLCONFIGDIR=/tmp JAX_PLATFORMS=cpu .pixi/envs/default/bin/python experiments/belief_semltl/collect_trajectories.py --smoke
    MPLCONFIGDIR=/tmp JAX_PLATFORMS=cpu .pixi/envs/default/bin/python experiments/belief_semltl/collect_trajectories.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import hydra
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.eval import EvalResetOptions
from jaxolotl.eval.utils import load_batched_actor_critic

LOG = logging.getLogger("collect_trajectories")
DEFAULT_FORMULAS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
DEFAULT_NOISE = (0.0, 0.1, 0.3)


class StepTraces(NamedTuple):
    active: jax.Array
    noisy_assignment: jax.Array
    prev_action: jax.Array
    hard_state: jax.Array
    true_state: jax.Array
    true_assignment: jax.Array


class Carry(NamedTuple):
    agent: object
    env_state: object
    observation: object
    completed: jax.Array
    traces: StepTraces
    index: jax.Array
    key: jax.Array


def empty_step_traces(t: int, n: int) -> StepTraces:
    shape = (t, n)
    z = jnp.zeros(shape, dtype=jnp.int32)
    b = jnp.zeros(shape, dtype=jnp.bool_)
    return StepTraces(
        active=b,
        noisy_assignment=z,
        prev_action=z,
        hard_state=z,
        true_state=z,
        true_assignment=z,
    )


@eqx.filter_jit
def rollout_collector(
    agent,
    env,
    base_env,
    env_params,
    task,
    key,
    num_episodes: int,
    max_steps: int,
    deterministic: bool,
):
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_episodes)
    env_state, obsv = env.reset(reset_keys, None, env_params, EvalResetOptions(task=task))
    props = eqx.filter_vmap(env.compute_propositions)(env_state, env_params)
    agent = agent.update(obsv, env_state, props, env)

    init_completed = jnp.zeros((num_episodes,), dtype=jnp.bool_)

    def cond(carry: Carry):
        return jnp.logical_and(
            jnp.logical_not(jnp.all(carry.completed)),
            carry.index < max_steps,
        )

    def step(carry: Carry):
        agent, env_state, obsv, completed, traces, idx, key = carry
        pi = agent.get_action(obsv)
        if deterministic:
            action = pi.mode()
        else:
            key, sample_key = jax.random.split(key)
            action = pi.sample(seed=sample_key)

        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_episodes)
        transition = env.step(step_keys, env_state, action, env_params)
        agent = agent.update(transition.observation, transition.state, transition.propositions, env)

        observed = jax.vmap(base_env.map_assignment_to_index)(
            transition.state.perceived_propositions
        )
        active = ~completed
        true_q = transition.state.ldba_state
        true_assignment = jax.vmap(base_env.map_assignment_to_index)(transition.propositions)
        hard_q = transition.state.perceived_ldba_state

        # action is tuple: (physical_action, epsilon_action)
        phys_action = action[0] if isinstance(action, tuple) else action

        next_traces = StepTraces(
            active=traces.active.at[idx].set(active),
            noisy_assignment=traces.noisy_assignment.at[idx].set(jnp.where(active, observed, 0)),
            prev_action=traces.prev_action.at[idx].set(jnp.where(active, phys_action, 4)),
            hard_state=traces.hard_state.at[idx].set(jnp.where(active, hard_q, 0)),
            true_state=traces.true_state.at[idx].set(jnp.where(active, true_q, 0)),
            true_assignment=traces.true_assignment.at[idx].set(jnp.where(active, true_assignment, 0)),
        )

        next_completed = completed | transition.done
        return Carry(
            agent,
            transition.state,
            transition.observation,
            next_completed,
            next_traces,
            idx + 1,
            key,
        )

    initial_carry = Carry(
        agent,
        env_state,
        obsv,
        init_completed,
        empty_step_traces(max_steps, num_episodes),
        jnp.array(0, dtype=jnp.int32),
        key,
    )
    result = eqx_utils.filter_while_loop(cond, step, initial_carry)
    return result.traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run")
    parser.add_argument("--num-train-episodes", type=int, default=32)
    parser.add_argument("--num-test-episodes", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.smoke:
        seeds_train = [0, 1]
        seeds_test = [2]
        formulas_eval = [0, 3]
        noise_levels = [0.0, 0.3]
        episodes_per_cond = 4
        out_name = "smoke"
    else:
        # Train on seeds 0..6, test on seeds 7..9
        seeds_train = [0, 1, 2, 3, 4, 5, 6]
        seeds_test = [7, 8, 9]
        formulas_eval = list(DEFAULT_FORMULAS)
        noise_levels = list(DEFAULT_NOISE)
        episodes_per_cond = None  # per split below
        out_name = "full"

    out_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "data" / "trajectories" / out_name
    )
    for name in ("train.npz", "test.npz"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to replace existing dataset: {out_dir / name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dir = str((REPO_ROOT / "conf").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])

    base_env, env_params = jaxolotl.make(cfg.env.name, reset_source="test", **cfg.get("env_params", {}))
    base_env = TimeLimitWrapper(base_env)
    formulas_all = preprocess_formulas(cfg.formulas, base_env)

    model_key = jax.random.PRNGKey(42)
    all_models, model_seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    nseeds = len(model_seeds)
    max_steps = int(env_params.max_steps_in_episode)

    rng = jax.random.PRNGKey(1000)

    for split_name, seed_list in [("train", seeds_train), ("test", seeds_test)]:
        if not args.smoke:
            episodes_per_cond = (args.num_train_episodes if split_name == "train"
                                 else args.num_test_episodes)
        LOG.info("Collecting %s split (%d seeds)...", split_name, len(seed_list))
        all_y = []
        all_u = []
        all_q_hard = []
        all_q_true = []
        all_a_true = []
        all_mask = []
        all_num_states = []
        all_model_seeds = []
        all_formula_indices = []
        all_noise_probs = []

        for model_seed in seed_list:
            s_idx = int(model_seed)
            single_model = jax.tree.map(
                lambda x: x[s_idx] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == nseeds else x,
                all_models,
            )
            agent = hydra.utils.instantiate(cfg.alg.agent, single_model)

            for f_idx in formulas_eval:
                task = jax.tree.map(lambda x: x[f_idx], formulas_all)
                num_s = int(formulas_all.num_states[f_idx])

                for p in noise_levels:
                    rng, subkey = jax.random.split(rng)
                    raw_wrapper = NoisySemanticLDBAWrapper(base_env, noise_prob=p, overwrite_finite=True)
                    env = VectorizeWrapper(raw_wrapper)

                    traces = rollout_collector(
                        agent, env, base_env, env_params, task, subkey,
                        num_episodes=episodes_per_cond, max_steps=max_steps,
                        deterministic=bool(cfg.eval.deterministic),
                    )
                    # traces shape: (max_steps, num_episodes) -> transpose to (num_episodes, max_steps)
                    all_y.append(np.asarray(jnp.transpose(traces.noisy_assignment)))
                    all_u.append(np.asarray(jnp.transpose(traces.prev_action)))
                    all_q_hard.append(np.asarray(jnp.transpose(traces.hard_state)))
                    all_q_true.append(np.asarray(jnp.transpose(traces.true_state)))
                    all_a_true.append(np.asarray(jnp.transpose(traces.true_assignment)))
                    all_mask.append(np.asarray(jnp.transpose(traces.active)))
                    all_num_states.append(np.full((episodes_per_cond,), num_s, dtype=np.int32))
                    all_model_seeds.append(np.full((episodes_per_cond,), model_seed, dtype=np.int32))
                    all_formula_indices.append(np.full((episodes_per_cond,), f_idx, dtype=np.int32))
                    all_noise_probs.append(np.full((episodes_per_cond,), p, dtype=np.float32))

        data = {
            "noisy_assignments": np.concatenate(all_y, axis=0),
            "prev_actions": np.concatenate(all_u, axis=0),
            "hard_states": np.concatenate(all_q_hard, axis=0),
            "true_states": np.concatenate(all_q_true, axis=0),
            "true_assignments": np.concatenate(all_a_true, axis=0),
            "active_mask": np.concatenate(all_mask, axis=0),
            "num_states": np.concatenate(all_num_states, axis=0),
            "model_seeds": np.concatenate(all_model_seeds, axis=0),
            "formula_indices": np.concatenate(all_formula_indices, axis=0),
            "noise_probs": np.concatenate(all_noise_probs, axis=0),
        }
        out_file = out_dir / f"{split_name}.npz"
        if out_file.exists():
            raise FileExistsError(f"Refusing to replace existing dataset: {out_file}")
        # np.savez does not itself offer exclusive creation: create the file first.
        with out_file.open("xb") as stream:
            np.savez_compressed(stream, **data)
        LOG.info("Saved %s to %s (total trajectories: %d)", split_name, out_file, data["noisy_assignments"].shape[0])


if __name__ == "__main__":
    main()
