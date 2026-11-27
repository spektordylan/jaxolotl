"""Evaluate the learned task-state estimator in closed-loop against the hard baseline.

The frozen pretrained SemLTL policy is conditioned on the LEARNED estimator's
predicted automaton state embedding, not the hard (noisy) perceived state.

Comparisons collected per episode:
  - success rate (from ground-truth automaton state)
  - violation rate (from ground-truth automaton state)
  - per-timestep state accuracy of:
      * hard monitor (NoisySemanticLDBAWrapper baseline)
      * learned estimator (LearnedTrackerWrapper)

Design constraints respected:
  - True propositions / ldba_state never feed the policy.
  - Ground truth is accessed only for scoring after the step.
  - The estimator receives only: noisy_assignment, prev_action, hard_state.

Usage:
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py --smoke
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

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
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.eval import EvalResetOptions
from jaxolotl.eval.utils import load_batched_actor_critic
from experiments.belief_semltl.learned_stats import paired_effect_rows, assert_zero_noise_parity
from experiments.belief_semltl.learned_tracker import LearnedStateEstimator
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper

LOG = logging.getLogger("learned_estimator")
DEFAULT_FORMULAS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
DEFAULT_NOISE = (0.0, 0.1, 0.3)
MODEL_PATH = (
    REPO_ROOT / "experiments" / "belief_semltl" / "models" / "full" / "learned_estimator.eqx"
)


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------

class EpisodeValues(NamedTuple):
    steps: jax.Array
    hard_correct_steps: jax.Array
    learned_correct_steps: jax.Array
    returns: jax.Array
    violations: jax.Array
    completed: jax.Array
    hard_any_disagreement: jax.Array
    learned_any_disagreement: jax.Array


class StepTraces(NamedTuple):
    active: jax.Array           # (T, N) bool
    hard_correct: jax.Array     # (T, N) float32
    learned_correct: jax.Array  # (T, N) float32
    true_state: jax.Array       # (T, N) int32
    hard_state: jax.Array       # (T, N) int32
    learned_state: jax.Array    # (T, N) int32


class Carry(NamedTuple):
    agent: object
    env_state: object
    observation: object
    episodes: EpisodeValues
    traces: StepTraces
    index: jax.Array
    key: jax.Array


def empty_episode_values(n: int) -> EpisodeValues:
    z = jnp.zeros((n,), dtype=jnp.float32)
    false = jnp.zeros((n,), dtype=jnp.bool_)
    return EpisodeValues(z, z, z, z, false, false, false, false)


def empty_step_traces(t: int, n: int) -> StepTraces:
    shape = (t, n)
    z = jnp.zeros(shape, dtype=jnp.float32)
    false = jnp.zeros(shape, dtype=jnp.bool_)
    q = jnp.full(shape, -1, dtype=jnp.int32)
    return StepTraces(false, z, z, q, q, q)


# ---------------------------------------------------------------------------
# Closed-loop rollout — LEARNED estimator (LearnedTrackerWrapper)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def rollout_learned(
    agent,
    env,
    env_params,
    task,
    key,
    num_episodes: int,
    max_steps: int,
    deterministic: bool,
):
    """Closed-loop rollout where the policy is conditioned on the learned estimator's state.

    LearnedTrackerWrapper info keys: is_sink, is_accepting, disagreement,
    true_ldba_state, estimated_ldba_state.
    """
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_episodes)
    env_state, obsv = env.reset(reset_keys, None, env_params, EvalResetOptions(task=task))
    props = eqx.filter_vmap(env.compute_propositions)(env_state, env_params)
    agent = agent.update(obsv, env_state, props, env)

    def cond(carry: Carry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(jnp.all(carry.episodes.completed)),
            carry.index < max_steps,
        )

    def step(carry: Carry) -> Carry:
        agent, env_state, obsv, episodes, traces, idx, key = carry
        pi = agent.get_action(obsv)
        if deterministic:
            action = pi.mode()
        else:
            key, sample_key = jax.random.split(key)
            action = pi.sample(seed=sample_key)
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_episodes)
        transition = env.step(step_keys, env_state, action, env_params)
        props = eqx.filter_vmap(env.compute_propositions)(transition.state, env_params)
        agent = agent.update(transition.observation, transition.state, props, env)

        active = ~episodes.completed
        active_f = active.astype(jnp.float32)

        # State comparisons — for evaluation diagnostics only (never fed to policy)
        true_q = transition.state.ldba_state            # ground truth
        hard_q = transition.state.perceived_ldba_state  # hard monitor
        learned_q = transition.state.estimated_ldba_state  # learned estimator

        hard_correct = (hard_q == true_q).astype(jnp.float32)
        learned_correct = (learned_q == true_q).astype(jnp.float32)

        # LearnedTrackerWrapper: key is "is_sink", "is_accepting"
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

    initial_carry = Carry(
        agent, env_state, obsv,
        empty_episode_values(num_episodes),
        empty_step_traces(max_steps, num_episodes),
        jnp.array(0, dtype=jnp.int32),
        key,
    )
    result = eqx_utils.filter_while_loop(cond, step, initial_carry)
    return result.episodes, result.traces


# ---------------------------------------------------------------------------
# Closed-loop rollout — HARD monitor baseline (NoisySemanticLDBAWrapper)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def rollout_hard(
    agent,
    env,
    env_params,
    task,
    key,
    num_episodes: int,
    max_steps: int,
    deterministic: bool,
):
    """Standard hard-monitor rollout. Tracks true vs hard perceived state for accuracy.

    NoisySemanticLDBAWrapper info keys: is_sink, true_is_accepting,
    perceived_is_accepting, perceived_is_sink.
    """
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_episodes)
    env_state, obsv = env.reset(reset_keys, None, env_params, EvalResetOptions(task=task))
    props = eqx.filter_vmap(env.compute_propositions)(env_state, env_params)
    agent = agent.update(obsv, env_state, props, env)

    def cond(carry: Carry) -> jax.Array:
        return jnp.logical_and(
            jnp.logical_not(jnp.all(carry.episodes.completed)),
            carry.index < max_steps,
        )

    def step(carry: Carry) -> Carry:
        agent, env_state, obsv, episodes, traces, idx, key = carry
        pi = agent.get_action(obsv)
        if deterministic:
            action = pi.mode()
        else:
            key, sample_key = jax.random.split(key)
            action = pi.sample(seed=sample_key)
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_episodes)
        transition = env.step(step_keys, env_state, action, env_params)
        props = eqx.filter_vmap(env.compute_propositions)(transition.state, env_params)
        agent = agent.update(transition.observation, transition.state, props, env)

        active = ~episodes.completed
        active_f = active.astype(jnp.float32)

        true_q = transition.state.ldba_state
        hard_q = transition.state.perceived_ldba_state
        hard_correct = (hard_q == true_q).astype(jnp.float32)

        # NoisySemanticLDBAWrapper: key is "is_sink" for true sink
        is_sink = transition.info["is_sink"].astype(jnp.bool_)

        next_episodes = EpisodeValues(
            steps=episodes.steps + active_f,
            hard_correct_steps=episodes.hard_correct_steps + active_f * hard_correct,
            learned_correct_steps=episodes.learned_correct_steps + active_f * hard_correct,  # mirrored
            returns=episodes.returns + active_f * jax.nn.relu(transition.reward),
            violations=episodes.violations | (active & is_sink),
            completed=episodes.completed | transition.done,
            hard_any_disagreement=episodes.hard_any_disagreement | (active & (hard_q != true_q)),
            learned_any_disagreement=episodes.learned_any_disagreement | (active & (hard_q != true_q)),
        )
        next_traces = StepTraces(
            active=traces.active.at[idx].set(active),
            hard_correct=traces.hard_correct.at[idx].set(active_f * hard_correct),
            learned_correct=traces.learned_correct.at[idx].set(active_f * hard_correct),
            true_state=traces.true_state.at[idx].set(jnp.where(active, true_q, -1)),
            hard_state=traces.hard_state.at[idx].set(jnp.where(active, hard_q, -1)),
            learned_state=traces.learned_state.at[idx].set(jnp.where(active, hard_q, -1)),
        )
        return Carry(agent, transition.state, transition.observation, next_episodes, next_traces, idx + 1, key)

    initial_carry = Carry(
        agent, env_state, obsv,
        empty_episode_values(num_episodes),
        empty_step_traces(max_steps, num_episodes),
        jnp.array(0, dtype=jnp.int32),
        key,
    )
    result = eqx_utils.filter_while_loop(cond, step, initial_carry)
    return result.episodes, result.traces


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="2 seeds × 2 formulas × 4 episodes; p=0,0.3 only")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[7, 8, 9],
                        help="Seeds excluded from gradient training (0..6); note 7..9 were used for checkpoint selection")
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE))
    parser.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GRU dimension used for this checkpoint")
    args = parser.parse_args()

    if args.smoke:
        args.model_seeds = [7, 8]
        args.num_episodes = 4
        args.formula_indices = [0, 3]
        args.noise_levels = [0.0, 0.3]

    if not args.model_seeds or len(args.model_seeds) != len(set(args.model_seeds)):
        parser.error("model-seeds must be nonempty and unique")
    args.noise_levels = sorted(set(args.noise_levels))
    if not args.noise_levels or any(not 0 <= p <= 1 for p in args.noise_levels):
        parser.error("noise-levels must be nonempty probabilities in [0,1]")
    if 0.0 not in args.noise_levels:
        parser.error("Include p=0 as a correctness gate")
    if not args.formula_indices or len(args.formula_indices) != len(set(args.formula_indices)):
        parser.error("formula-indices must be nonempty and unique")
    if 5 in args.formula_indices:
        parser.error("Formula index 5 contains epsilon transitions; excluded from this study")

    args.output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "results"
        / "learned_estimator" / ("smoke" if args.smoke else "full")
    )
    return args


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty CSV")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_estimator(model_path: Path, hidden_dim: int = 64) -> LearnedStateEstimator:
    template = LearnedStateEstimator(key=jax.random.PRNGKey(0), hidden_dim=hidden_dim)
    return eqx.tree_deserialise_leaves(model_path, template)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _summarize(episode_rows: list[dict], noise_levels, formula_indices) -> list[dict]:
    """Mean ± stderr per noise level, aggregated to seed level first."""
    metrics = [
        "hard_success", "learned_success",
        "hard_violation", "learned_violation",
        "hard_step_accuracy", "learned_step_accuracy",
        "hard_any_disagreement", "learned_any_disagreement",
    ]
    rows = []

    def _agg(group: list[dict], scope: str, noise_prob: float, formula_idx: int | None) -> list[dict]:
        seeds = sorted(set(r["model_seed"] for r in group))
        out = []
        for metric in metrics:
            seed_means = [
                float(np.mean([r[metric] for r in group if r["model_seed"] == seed]))
                for seed in seeds
            ]
            n = len(seed_means)
            mean = float(np.mean(seed_means)) if n > 0 else float("nan")
            stderr = float(np.std(seed_means, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            out.append({
                "scope": scope,
                "noise_prob": noise_prob,
                "formula_idx": formula_idx,
                "metric": metric,
                "mean": mean,
                "stderr": stderr,
                "n_seeds": n,
            })
        return out

    for p in noise_levels:
        prows = [r for r in episode_rows if r["noise_prob"] == p]
        rows.extend(_agg(prows, "overall", p, None))
        for fi in formula_indices:
            frows = [r for r in prows if r["formula_idx"] == fi]
            if frows:
                rows.extend(_agg(frows, "per_formula", p, fi))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Never overwrite any existing learned-evaluation results.
    outputs = [args.output_dir / n for n in ("episodes.csv", "summary.csv", "paired_effects.csv", "metadata.json")]
    conflicts = [path for path in outputs if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite existing results: {conflicts}; use --output-dir")

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Trained estimator not found at {args.model_path}. "
            "Run train_estimator.py first."
        )

    with initialize(version_base="1.3", config_path="../../conf"):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            f"eval.num_episodes={args.num_episodes}",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])

    original_formulas = list(cfg.formulas)
    if any(i < 0 or i >= len(original_formulas) for i in args.formula_indices):
        raise ValueError("Formula index out of range")
    cfg.formulas = [original_formulas[i] for i in args.formula_indices]
    formula_strings = [str(f) for f in cfg.formulas]

    LOG.info("Preprocessing %d LTL formulas", len(cfg.formulas))
    base_env, env_params = jaxolotl.make(
        cfg.env.name, reset_source="test", **cfg.get("env_params", {})
    )
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)

    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Selected formulas unexpectedly contain epsilon transitions")

    max_steps = int(env_params.max_steps_in_episode)

    LOG.info("Loading estimator from %s", args.model_path)
    estimator = load_estimator(args.model_path, hidden_dim=args.hidden_dim)

    key = jax.random.key(0)
    _, model_key = jax.random.split(key)
    models, all_seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    all_seeds = [int(s) for s in all_seeds]
    missing = set(args.model_seeds).difference(all_seeds)
    if missing:
        raise ValueError(f"Requested model seeds {sorted(missing)} absent from pretrained models {all_seeds}")
    selected_indices = [all_seeds.index(s) for s in args.model_seeds]
    n_all = len(all_seeds)
    models = jax.tree.map(
        lambda x: x[jnp.asarray(selected_indices)] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_all else x,
        models,
    )
    seeds = list(args.model_seeds)
    nseeds = len(seeds)
    LOG.info("Selected model seeds: %s", seeds)

    episode_rows: list[dict] = []
    deterministic = bool(cfg.eval.deterministic)

    for s, model_seed in enumerate(seeds):
        single_model = jax.tree.map(
            lambda x: x[s] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == nseeds else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, single_model)

        for f, formula_idx in enumerate(args.formula_indices):
            task = jax.tree.map(lambda x: x[f], formulas)

            for p in args.noise_levels:
                # Deterministic run_key — different base (200) from other experiments
                run_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(200), int(model_seed)),
                    int(formula_idx),
                )
                run_key = jax.random.fold_in(run_key, int(round(p * 1_000_000)))

                # --- Hard-monitor rollout ---
                hard_env = VectorizeWrapper(
                    NoisySemanticLDBAWrapper(base_env, noise_prob=p, overwrite_finite=True)
                )
                hard_ep, _ = rollout_hard(
                    agent, hard_env, env_params, task, run_key,
                    args.num_episodes, max_steps, deterministic,
                )
                hard_ep = jax.tree.map(np.asarray, hard_ep)
                if not np.all(hard_ep.completed):
                    raise RuntimeError("Hard-monitor rollout: some episodes exceeded max_steps")

                # --- Learned-estimator rollout ---
                learned_env = VectorizeWrapper(
                    LearnedTrackerWrapper(base_env, estimator=estimator, noise_prob=p, overwrite_finite=True)
                )
                learned_ep, _ = rollout_learned(
                    agent, learned_env, env_params, task, run_key,
                    args.num_episodes, max_steps, deterministic,
                )
                learned_ep = jax.tree.map(np.asarray, learned_ep)
                if not np.all(learned_ep.completed):
                    raise RuntimeError("Learned-estimator rollout: some episodes exceeded max_steps")

                # p=0 sanity check: at zero noise the estimator should behave identically to hard
                if p == 0.0:
                    assert_zero_noise_parity(
                        hard_ep, learned_ep, seed=int(model_seed), formula_idx=int(formula_idx)
                    )

                for ep in range(args.num_episodes):
                    n_hard = max(int(hard_ep.steps[ep]), 1)
                    n_learned = max(int(learned_ep.steps[ep]), 1)
                    row = {
                        "noise_prob": p,
                        "model_seed": int(model_seed),
                        "formula_idx": int(formula_idx),
                        "formula": formula_strings[f],
                        "episode": ep,
                        # Hard-monitor metrics
                        "hard_steps": n_hard,
                        "hard_success": int(hard_ep.returns[ep] > 0),
                        "hard_violation": int(hard_ep.violations[ep]),
                        "hard_correct_steps": int(hard_ep.hard_correct_steps[ep]),
                        "hard_step_accuracy": float(hard_ep.hard_correct_steps[ep] / n_hard),
                        "hard_any_disagreement": int(hard_ep.hard_any_disagreement[ep]),
                        # Learned estimator metrics
                        "learned_steps": n_learned,
                        "learned_success": int(learned_ep.returns[ep] > 0),
                        "learned_violation": int(learned_ep.violations[ep]),
                        "learned_correct_steps": int(learned_ep.learned_correct_steps[ep]),
                        "learned_step_accuracy": float(
                            learned_ep.learned_correct_steps[ep] / n_learned
                        ),
                        "learned_any_disagreement": int(learned_ep.learned_any_disagreement[ep]),
                    }
                    episode_rows.append(row)

                h_acc = float(
                    hard_ep.hard_correct_steps.sum() / max(hard_ep.steps.sum(), 1)
                )
                l_acc = float(
                    learned_ep.learned_correct_steps.sum() / max(learned_ep.steps.sum(), 1)
                )
                h_succ = float(np.mean(hard_ep.returns > 0))
                l_succ = float(np.mean(learned_ep.returns > 0))
                LOG.info(
                    "seed=%-4s formula=%d p=%.2f | hard acc=%.3f succ=%.3f | "
                    "learned acc=%.3f succ=%.3f",
                    model_seed, formula_idx, p, h_acc, h_succ, l_acc, l_succ,
                )

    summary_rows = _summarize(episode_rows, args.noise_levels, args.formula_indices)
    effect_rows = paired_effect_rows(episode_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "episodes.csv", episode_rows)
    save_csv(args.output_dir / "summary.csv", summary_rows)
    save_csv(args.output_dir / "paired_effects.csv", effect_rows)

    metadata = {
        "git_revision_at_run": current_commit(),
        "method": "learned GRU estimator in closed-loop vs hard-monitor baseline",
        "estimator_model_path": str(args.model_path),
        "noise_probabilities": args.noise_levels,
        "formula_indices": args.formula_indices,
        "model_seeds": [int(s) for s in seeds],
        "excluded_from_estimator_gradient_training": all(s in (7, 8, 9) for s in seeds),
        "checkpoint_training_policy_seeds": list(range(7)),
        "checkpoint_selection_policy_seeds": [7, 8, 9],
        "num_episodes_per_condition": args.num_episodes,
        "run_key_base": "key(200) folded with model_seed, formula_idx, integer noise prob",
        "deterministic_policy": deterministic,
        "true_proposition_access": (
            "ground-truth for scoring only; "
            "estimator receives only noisy obs, prev action, hard state"
        ),
        "limitations": [
            "Estimator trained on hard-policy trajectories; distribution shift when "
            "learned estimator controls policy (trajectory distribution differs from training)",
            "Independent uniform proposition prior is not LetterWorld dynamics",
            "No epsilon transitions evaluated",
            "Original frozen actor sees LetterWorld's unmasked raw grid; only the learned estimator is proposition-channel restricted",
            "Results are matched by environment run key, not the same physical trajectories after policies diverge",
            "This run is not a same-trajectory test of state inference; use offline checkpoint diagnostics separately",
            "The provided checkpoint was selected using the 7..9 validation split: 7..9 are not an untouched final test set",
        ],
    }
    with (args.output_dir / "metadata.json").open("x", encoding="utf-8") as meta_file:
        meta_file.write(json.dumps(metadata, indent=2) + "\n")

    # Print summary table
    print(
        "\nLEARNED ESTIMATOR vs HARD BASELINE "
        "(closed-loop, mean across seeds × formulas × episodes):"
    )
    print(
        f"{'noise':>7} {'hard_succ':>10} {'lrn_succ':>10} {'Δsucc':>8} "
        f"{'hard_acc':>10} {'lrn_acc':>10} {'Δacc':>8}"
    )
    for p in args.noise_levels:
        prows = [r for r in summary_rows if r["noise_prob"] == p and r["scope"] == "overall"]
        vals = {r["metric"]: r["mean"] for r in prows}
        h_s = vals.get("hard_success", float("nan"))
        l_s = vals.get("learned_success", float("nan"))
        h_a = vals.get("hard_step_accuracy", float("nan"))
        l_a = vals.get("learned_step_accuracy", float("nan"))
        print(
            f"{p:7.2f} {h_s:10.4f} {l_s:10.4f} {l_s - h_s:+8.4f} "
            f"{h_a:10.4f} {l_a:10.4f} {l_a - h_a:+8.4f}"
        )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
