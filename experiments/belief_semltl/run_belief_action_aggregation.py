"""Compare hard, frozen-GRU MAP, and GRU belief-weighted action aggregation.

No training. Runs only the existing epsilon-free LetterWorld benchmark with the
same experiment seeds, formula split, reset key and estimator checkpoint as
run_learned_estimator.py. Never replaces existing results.

Run:
  JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_belief_action_aggregation.py --smoke
  JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_belief_action_aggregation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import NoisySemanticLDBAWrapper
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.eval import EvalResetOptions
from jaxolotl.eval.utils import load_batched_actor_critic

from experiments.belief_semltl.belief_action_aggregation import aggregate_physical_action
from experiments.belief_semltl.learned_stats import assert_zero_noise_parity
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper
from experiments.belief_semltl.run_learned_estimator import (
    Carry, EpisodeValues, StepTraces, DEFAULT_FORMULAS, DEFAULT_NOISE,
    MODEL_PATH, current_commit, empty_episode_values, empty_step_traces,
    load_estimator, rollout_hard, rollout_learned, save_csv,
)
from experiments.belief_semltl.pilot_stats import seed_ci

LOG = logging.getLogger("belief_action_aggregation")
METHODS = ("hard", "map", "aggregation")
METRICS = ("success", "violation", "step_accuracy", "steps")


@eqx.filter_jit
def rollout_aggregated(
    agent, env, env_params, task, key, estimator,
    num_episodes: int, max_steps: int, deterministic: bool,
    top_k: int, alpha: float, zero_noise: bool,
):
    """Same rollout/scoring as rollout_learned, changing *only* actor action selection."""
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_episodes)
    env_state, obsv = env.reset(reset_keys, None, env_params, EvalResetOptions(task=task))
    props = eqx.filter_vmap(env.compute_propositions)(env_state, env_params)
    agent = agent.update(obsv, env_state, props, env)

    def cond(carry: Carry):
        return jnp.logical_and(~jnp.all(carry.episodes.completed), carry.index < max_steps)

    def step(carry: Carry):
        agent, env_state, obsv, episodes, traces, idx, key = carry
        if deterministic:
            action_key = key  # no random draw for deterministic mode
        else:
            key, action_key = jax.random.split(key)
        action = aggregate_physical_action(
            agent, obsv, env_state, estimator, action_key,
            step_index=idx, top_k=top_k, alpha=alpha,
            deterministic=deterministic, zero_noise=zero_noise,
        )
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_episodes)
        transition = env.step(step_keys, env_state, action, env_params)
        props = eqx.filter_vmap(env.compute_propositions)(transition.state, env_params)
        agent = agent.update(transition.observation, transition.state, props, env)
        active = ~episodes.completed
        active_f = active.astype(jnp.float32)
        true_q = transition.state.ldba_state
        hard_q = transition.state.perceived_ldba_state
        learned_q = transition.state.estimated_ldba_state
        hard_correct = (hard_q == true_q).astype(jnp.float32)
        learned_correct = (learned_q == true_q).astype(jnp.float32)
        is_sink = transition.info["is_sink"].astype(jnp.bool_)
        next_episodes = EpisodeValues(
            steps=episodes.steps + active_f,
            hard_correct_steps=episodes.hard_correct_steps + active_f * hard_correct,
            learned_correct_steps=episodes.learned_correct_steps + active_f * learned_correct,
            returns=episodes.returns + active_f * jax.nn.relu(transition.reward),
            violations=episodes.violations | (active & is_sink),
            completed=episodes.completed | transition.done,
            hard_any_disagreement=episodes.hard_any_disagreement | (active & (hard_q != true_q)),
            learned_any_disagreement=episodes.learned_any_disagreement | (active & (learned_q != true_q)),
        )
        next_traces = StepTraces(
            active=traces.active.at[idx].set(active),
            hard_correct=traces.hard_correct.at[idx].set(active_f * hard_correct),
            learned_correct=traces.learned_correct.at[idx].set(active_f * learned_correct),
            true_state=traces.true_state.at[idx].set(jnp.where(active, true_q, -1)),
            hard_state=traces.hard_state.at[idx].set(jnp.where(active, hard_q, -1)),
            learned_state=traces.learned_state.at[idx].set(jnp.where(active, learned_q, -1)),
        )
        return Carry(agent, transition.state, transition.observation, next_episodes, next_traces, idx + 1, key)

    initial = Carry(
        agent, env_state, obsv,
        empty_episode_values(num_episodes),
        empty_step_traces(max_steps, num_episodes),
        jnp.asarray(0, dtype=jnp.int32), key,
    )
    result = eqx_utils.filter_while_loop(cond, step, initial)
    return result.episodes, result.traces


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 seeds, 2 formulas, 4 episodes; p=0,0.3")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE))
    parser.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    parser.add_argument("--top-k", type=int, default=3, help="Number of posterior states mixed (default: 3)")
    parser.add_argument("--alpha", type=float, default=1.0, help="0=MAP, 1=posterior mixture")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.model_seeds = [7, 8]
        args.num_episodes = 4
        args.noise_levels = [0.0, 0.3]
        args.formula_indices = [0, 3]
    args.noise_levels = sorted(set(args.noise_levels))
    if args.num_episodes < 1 or not args.model_seeds or len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("Positive episode count and unique model seeds required")
    if not args.noise_levels or 0.0 not in args.noise_levels or any(not 0 <= p <= 1 for p in args.noise_levels):
        parser.error("Include p=0 and only valid noise probabilities")
    if not args.formula_indices or len(set(args.formula_indices)) != len(args.formula_indices) or 5 in args.formula_indices:
        parser.error("Use distinct epsilon-free formula indices; formula 5 is excluded")
    if not (1 <= args.top_k <= 14):
        parser.error("top-k must be between 1 and the tracker checkpoint max_states (14)")
    if not (0 <= args.alpha <= 1):
        parser.error("alpha must be between 0 and 1")
    args.output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "results" / "learned_aggregation" /
        ("smoke" if args.smoke else "full")
    )
    return args


def _episode_metrics(ep, index: int, method: str) -> dict[str, float]:
    n_steps = max(float(ep.steps[index]), 1.0)
    correct = ep.hard_correct_steps if method == "hard" else ep.learned_correct_steps
    return {
        "success": float(ep.returns[index] > 0),
        "violation": float(ep.violations[index]),
        "step_accuracy": float(correct[index] / n_steps),
        "steps": float(ep.steps[index]),
    }


def summarize(rows: list[dict], noise_levels: list[float], formulas: list[int]):
    """Aggregate episodes/formulas within each pretrained seed; t CI over seeds."""
    output, effects = [], []
    for p in noise_levels:
        for formula_idx in (None, *formulas):
            subset = [r for r in rows if r["noise_prob"] == p and
                      (formula_idx is None or r["formula_idx"] == formula_idx)]
            seeds = sorted({r["model_seed"] for r in subset})
            scope = "overall" if formula_idx is None else "per_formula"
            f = -1 if formula_idx is None else formula_idx
            for metric in METRICS:
                seed_values = {
                    method: np.asarray([
                        np.mean([r[f"{method}_{metric}"] for r in subset if r["model_seed"] == seed])
                        for seed in seeds
                    ])
                    for method in METHODS
                }
                for method in METHODS:
                    ci = seed_ci(seed_values[method])
                    output.append({
                        "noise_prob": p, "scope": scope, "formula_idx": f,
                        "method": method, "metric": metric, "mean": ci.mean,
                        "ci_lower": ci.lower, "ci_upper": ci.upper, "n_seeds": ci.n_seeds,
                    })
                for method1, method2 in (("hard", "map"), ("hard", "aggregation"), ("map", "aggregation")):
                    ci = seed_ci(seed_values[method2] - seed_values[method1])
                    effects.append({
                        "noise_prob": p, "scope": scope, "formula_idx": f,
                        "comparison": f"{method2}_minus_{method1}", "metric": metric,
                        "mean_delta": ci.mean, "ci_lower": ci.lower,
                        "ci_upper": ci.upper, "n_seeds": ci.n_seeds,
                    })
    return output, effects


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    output_names = ("episodes.csv", "summary.csv", "paired_effects.csv", "metadata.json")
    conflicts = [args.output_dir / name for name in output_names if (args.output_dir / name).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite old results: {conflicts}. Select --output-dir")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Missing GRU checkpoint: {args.model_path}")

    with initialize(version_base="1.3", config_path="../../conf"):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            f"eval.num_episodes={args.num_episodes}",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])
    original_formulas = list(cfg.formulas)
    if any(i < 0 or i >= len(original_formulas) for i in args.formula_indices):
        raise ValueError("Formula index outside available formula list")
    cfg.formulas = [original_formulas[i] for i in args.formula_indices]
    base_env, env_params = jaxolotl.make(cfg.env.name, reset_source="test", **cfg.get("env_params", {}))
    base_env = TimeLimitWrapper(base_env)
    LOG.info("Preprocessing %d LTL formulas", len(cfg.formulas))
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Only epsilon-free formulas are supported")
    max_steps = int(env_params.max_steps_in_episode)
    estimator = load_estimator(args.model_path, hidden_dim=args.hidden_dim)
    if args.top_k > estimator.max_states:
        raise ValueError("top_k exceeds estimator output dimension")

    _, model_key = jax.random.split(jax.random.key(0))
    models, all_seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    all_seeds = [int(s) for s in all_seeds]
    if set(args.model_seeds) - set(all_seeds):
        raise ValueError(f"Missing pretrained model seeds: {sorted(set(args.model_seeds)-set(all_seeds))}")
    selection = [all_seeds.index(s) for s in args.model_seeds]
    n_all = len(all_seeds)
    models = jax.tree.map(
        lambda x: x[jnp.asarray(selection)] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_all else x,
        models,
    )
    deterministic = bool(cfg.eval.deterministic)
    LOG.info("Selected model seeds: %s; deterministic=%s; top_k=%d alpha=%g",
             args.model_seeds, deterministic, args.top_k, args.alpha)
    rows = []
    for idx, seed in enumerate(args.model_seeds):
        model = jax.tree.map(
            lambda x: x[idx] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == len(args.model_seeds) else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, model)
        for j, formula_idx in enumerate(args.formula_indices):
            task = jax.tree.map(lambda x: x[j], formulas)
            for p in args.noise_levels:
                run_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(200), int(seed)), int(formula_idx)
                )
                run_key = jax.random.fold_in(run_key, int(round(p * 1_000_000)))
                hard_env = VectorizeWrapper(NoisySemanticLDBAWrapper(
                    base_env, noise_prob=p, overwrite_finite=True
                ))
                tracker_env = VectorizeWrapper(LearnedTrackerWrapper(
                    base_env, estimator=estimator, noise_prob=p, overwrite_finite=True
                ))
                hard_ep, _ = rollout_hard(agent, hard_env, env_params, task, run_key,
                                          args.num_episodes, max_steps, deterministic)
                map_ep, _ = rollout_learned(agent, tracker_env, env_params, task, run_key,
                                           args.num_episodes, max_steps, deterministic)
                agg_ep, _ = rollout_aggregated(
                    agent, tracker_env, env_params, task, run_key, estimator,
                    args.num_episodes, max_steps, deterministic,
                    args.top_k, args.alpha, bool(p == 0),
                )
                hard_ep, map_ep, agg_ep = [jax.tree.map(np.asarray, x) for x in (hard_ep, map_ep, agg_ep)]
                if any(not np.all(ep.completed) for ep in (hard_ep, map_ep, agg_ep)):
                    raise RuntimeError("Some rollouts exceeded max_steps")
                if p == 0:
                    assert_zero_noise_parity(hard_ep, map_ep, seed=int(seed), formula_idx=int(formula_idx))
                    assert_zero_noise_parity(map_ep, agg_ep, seed=int(seed), formula_idx=int(formula_idx))
                if args.alpha == 0:
                    assert_zero_noise_parity(map_ep, agg_ep, seed=int(seed), formula_idx=int(formula_idx))
                if args.top_k == 1 and deterministic:
                    assert_zero_noise_parity(map_ep, agg_ep, seed=int(seed), formula_idx=int(formula_idx))
                for episode in range(args.num_episodes):
                    row = {
                        "noise_prob": p, "model_seed": int(seed), "formula_idx": formula_idx,
                        "formula": str(cfg.formulas[j]), "episode": episode,
                    }
                    for method, ep in (("hard", hard_ep), ("map", map_ep), ("aggregation", agg_ep)):
                        for metric, value in _episode_metrics(ep, episode, method).items():
                            row[f"{method}_{metric}"] = value
                    rows.append(row)
                LOG.info("seed=%d formula=%d p=%.2f | hard=%.3f MAP=%.3f agg=%.3f",
                         seed, formula_idx, p,
                         np.mean(hard_ep.returns > 0), np.mean(map_ep.returns > 0),
                         np.mean(agg_ep.returns > 0))

    summary, effects = summarize(rows, args.noise_levels, args.formula_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "episodes.csv", rows)
    save_csv(args.output_dir / "summary.csv", summary)
    save_csv(args.output_dir / "paired_effects.csv", effects)
    metadata = {
        "method": "GRU MAP versus weighted frozen actor action probabilities",
        "git_revision_at_run": current_commit(),
        "checkpoint": str(args.model_path), "top_k": args.top_k, "alpha": args.alpha,
        "model_seeds": args.model_seeds, "formula_indices": args.formula_indices,
        "noise_probabilities": args.noise_levels, "num_episodes_per_condition": args.num_episodes,
        "deterministic": deterministic,
        "run_key": "200 folded with pretrained model seed, formula index, noise * 1000000",
        "leakage": "true LDBA state only for scoring, no true propositions in GRU or aggregation",
        "limitations": [
            "Only epsilon-free LetterWorld with four discrete physical actions is supported.",
            "Original frozen actor sees unmasked LetterWorld spatial grid, as before.",
            "Checkpoint was selected using archived 7..9 validation seeds, not an independent test.",
            "Different policy actions yield different closed-loop trajectories despite shared reset keys.",
            "Top-k renormalizes posterior mass, and GRU probabilities are not calibrated by this study.",
            "95% intervals use pretrained policy seeds, not individual episodes, as replicates.",
        ],
    }
    with (args.output_dir / "metadata.json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print("\nCLOSED-LOOP SUCCESS (seed-level means):")
    print(f"{'noise':>7} {'hard':>9} {'GRU MAP':>9} {'agg':>9} {'agg-MAP':>10}")
    for p in args.noise_levels:
        values = {r["method"]: r["mean"] for r in summary
                  if r["noise_prob"] == p and r["scope"] == "overall" and r["metric"] == "success"}
        print(f"{p:7.2f} {values['hard']:9.4f} {values['map']:9.4f} "
              f"{values['aggregation']:9.4f} {values['aggregation']-values['map']:+10.4f}")
    print("\nPaired effects and CIs:", args.output_dir / "paired_effects.csv")
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
