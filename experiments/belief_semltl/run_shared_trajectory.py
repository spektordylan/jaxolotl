"""Diagnose hard versus Bayesian task-state estimates on IDENTICAL trajectories.

One frozen hard SemLTL policy drives each LetterWorld episode. The hard monitor
and a passive Bayesian filter read the exact same corrupted proposition labels;
only the scorer may read the ground-truth automaton state. Neither MAP nor
weighted embeddings control the environment in this experiment.

From repository root:
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_shared_trajectory.py --smoke
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_shared_trajectory.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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
from jaxolotl.alg.sem_ltl.wrappers.belief_semantic_ldba_wrapper import (
    compute_belief_next,
    compute_posterior_a,
)
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.eval import EvalResetOptions, Evaluator
from jaxolotl.eval.utils import load_batched_actor_critic
from experiments.belief_semltl.run_pilot import select_model_seeds
from experiments.belief_semltl.shared_trajectory_stats import (
    summarize_seed_records,
    summarize_timestep_records,
)

LOG = logging.getLogger("shared_trajectory")
DEFAULT_FORMULAS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
DEFAULT_NOISE = (0.0, 0.1, 0.3)


class EpisodeValues(NamedTuple):
    steps: jax.Array
    hard_correct_steps: jax.Array
    map_correct_steps: jax.Array
    prob_true_sum: jax.Array
    nll_sum: jax.Array
    entropy_sum: jax.Array
    hard_any_disagreement: jax.Array
    map_any_disagreement: jax.Array
    returns: jax.Array
    violations: jax.Array
    completed: jax.Array


class StepTraces(NamedTuple):
    active: jax.Array
    hard_correct: jax.Array
    map_correct: jax.Array
    true_prob: jax.Array
    neg_log_true_prob: jax.Array
    entropy: jax.Array
    true_state: jax.Array
    hard_state: jax.Array
    map_state: jax.Array


class Carry(NamedTuple):
    agent: object
    env_state: object
    observation: object
    belief: jax.Array
    episodes: EpisodeValues
    traces: StepTraces
    index: jax.Array
    key: jax.Array


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 seeds x 2 formulas x 4 episodes; p=0,0.3")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE))
    parser.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-parity-check", action="store_true", help="Not recommended; only for isolating evaluator incompatibility")
    args = parser.parse_args()
    if args.smoke:
        args.num_seeds = 2
        args.num_episodes = 4
        args.formula_indices = [0, 3]
        args.noise_levels = [0.0, 0.3]
    if args.num_seeds < 1 or args.num_episodes < 1:
        parser.error("num-seeds and num-episodes must be positive")
    args.noise_levels = sorted(set(args.noise_levels))
    if not args.noise_levels or any(not 0 <= p <= 1 for p in args.noise_levels):
        parser.error("noise-levels must be nonempty probabilities in [0,1]")
    if 0.0 not in args.noise_levels:
        parser.error("Include p=0 as a deterministic correctness gate")
    if not args.formula_indices or len(args.formula_indices) != len(set(args.formula_indices)):
        parser.error("Formula indices must be nonempty and unique")
    if 5 in args.formula_indices:
        parser.error("Formula index 5 contains epsilon transitions; exclude from this diagnostic")
    args.output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "results"
        / "shared_trajectory" / ("smoke" if args.smoke else "full")
    )
    return args


def empty_episode_values(n: int) -> EpisodeValues:
    z = jnp.zeros((n,), dtype=jnp.float32)
    false = jnp.zeros((n,), dtype=jnp.bool_)
    return EpisodeValues(z, z, z, z, z, z, false, false, z, false, false)


def empty_step_traces(t: int, n: int) -> StepTraces:
    shape = (t, n)
    z = jnp.zeros(shape, dtype=jnp.float32)
    false = jnp.zeros(shape, dtype=jnp.bool_)
    q = jnp.full(shape, -1, dtype=jnp.int32)
    return StepTraces(false, z, z, z, z, z, q, q, q)


@eqx.filter_jit
def rollout_hard_with_passive_belief(
    agent,
    env,
    base_env,
    env_params,
    task,
    key,
    num_episodes: int,
    max_steps: int,
    noise_prob: float,
    deterministic: bool,
):
    """A matched hard-policy rollout with a passive, never-controlling belief filter.

    Mirrors Evaluator.eval's reset, action-sampling, and step-key handling.
    Completed episodes are masked out of all diagnostic metrics. No extra PRNG
    calls are introduced, since the passive filter is deterministic given y.
    """
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_episodes)
    env_state, obsv = env.reset(reset_keys, None, env_params, EvalResetOptions(task=task))
    props = eqx.filter_vmap(env.compute_propositions)(env_state, env_params)
    agent = agent.update(obsv, env_state, props, env)

    initial = jax.nn.one_hot(task.initial_state, task.transitions.shape[0], dtype=jnp.float32)
    belief = jnp.broadcast_to(initial, (num_episodes, initial.shape[0]))

    def cond(carry):
        return jnp.logical_and(
            jnp.logical_not(jnp.all(carry.episodes.completed)),
            carry.index < max_steps,
        )

    def step(carry):
        agent, env_state, obsv, belief, episode, traces, idx, key = carry
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

        # This is the *only* proposition input to the Bayesian filter.
        observed = jax.vmap(base_env.map_assignment_to_index)(
            transition.state.perceived_propositions
        )
        posterior = jax.vmap(
            lambda y: compute_posterior_a(
                y, task.transitions.shape[1], jnp.asarray(noise_prob, dtype=jnp.float32)
            )
        )(observed)
        belief_candidate = jax.vmap(compute_belief_next, in_axes=(0, None, 0))(
            belief, task.transitions, posterior
        )
        active = ~episode.completed
        next_belief = jnp.where(active[:, None], belief_candidate, belief)

        # Ground truth is referenced here ONLY AFTER the belief update, for scoring.
        true_q = transition.state.ldba_state
        hard_q = transition.state.perceived_ldba_state
        map_q = jnp.argmax(next_belief, axis=1)
        hard_correct = hard_q == true_q
        map_correct = map_q == true_q
        prob_true = jnp.take_along_axis(next_belief, true_q[:, None], axis=1)[:, 0]
        nll = -jnp.log(jnp.clip(prob_true, 1e-12, 1.0))
        entropy = -jnp.sum(
            jnp.where(next_belief > 0, next_belief * jnp.log(jnp.maximum(next_belief, 1e-12)), 0),
            axis=1,
        )
        active_f = active.astype(jnp.float32)
        next_episode = EpisodeValues(
            episode.steps + active_f,
            episode.hard_correct_steps + active_f * hard_correct,
            episode.map_correct_steps + active_f * map_correct,
            episode.prob_true_sum + active_f * prob_true,
            episode.nll_sum + active_f * nll,
            episode.entropy_sum + active_f * entropy,
            episode.hard_any_disagreement | (active & ~hard_correct),
            episode.map_any_disagreement | (active & ~map_correct),
            episode.returns + active_f * jax.nn.relu(transition.reward),
            episode.violations | (active & transition.info["is_sink"].astype(bool)),
            episode.completed | transition.done,
        )
        next_traces = StepTraces(
            traces.active.at[idx].set(active),
            traces.hard_correct.at[idx].set(active_f * hard_correct),
            traces.map_correct.at[idx].set(active_f * map_correct),
            traces.true_prob.at[idx].set(active_f * prob_true),
            traces.neg_log_true_prob.at[idx].set(active_f * nll),
            traces.entropy.at[idx].set(active_f * entropy),
            traces.true_state.at[idx].set(jnp.where(active, true_q, -1)),
            traces.hard_state.at[idx].set(jnp.where(active, hard_q, -1)),
            traces.map_state.at[idx].set(jnp.where(active, map_q, -1)),
        )
        return Carry(agent, transition.state, transition.observation, next_belief,
                     next_episode, next_traces, idx + 1, key)

    initial_carry = Carry(
        agent, env_state, obsv, belief, empty_episode_values(num_episodes),
        empty_step_traces(max_steps, num_episodes),
        jnp.array(0, dtype=jnp.int32), key,
    )
    result = eqx_utils.filter_while_loop(cond, step, initial_carry)
    return result.episodes, result.traces


def reference_parity_gate(agent, env, env_params, task, key, episodes, diagnostic, deterministic):
    """Check success, violations, lengths, and hard disagreement versus Jaxolotl's evaluator."""
    standard = Evaluator(
        num_episodes=episodes, discount=1.0,
        return_trajs=False, return_disagreement=True,
    )
    returns, _, lengths, violations, disagreement, _ = standard.eval(
        agent, deterministic, env, env_params, task, key
    )
    observed = [
        np.asarray(diagnostic.returns),
        np.asarray(diagnostic.steps),
        np.asarray(diagnostic.violations),
        np.asarray(diagnostic.hard_any_disagreement),
    ]
    expected = [np.asarray(returns), np.asarray(lengths),
                np.asarray(violations), np.asarray(disagreement)]
    for name, a, b in zip(("returns", "lengths", "violations", "hard disagreement"), observed, expected):
        if not np.array_equal(a, b):
            raise AssertionError(
                f"Diagnostic diverged from Jaxolotl Evaluator on {name}. "
                f"first mismatching episode={int(np.flatnonzero(a != b)[0])}; "
                "do not interpret results until resolved"
            )
    LOG.info("PASS direct evaluator parity: returns, lengths, violations, disagreement")


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Refusing to write an empty diagnostic CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def maybe_plot(summary: list[dict], by_step: list[dict], output_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, metric in (("Hard", "hard_step_accuracy"), ("MAP", "map_step_accuracy")):
        rows = sorted((r for r in summary if r["scope"] == "overall" and r["metric"] == metric),
                      key=lambda r: r["noise_prob"])
        ax.plot([r["noise_prob"] for r in rows], [r["mean"] for r in rows],
                marker="o", label=mode)
    ax.set(xlabel="Categorical proposition noise probability",
           ylabel="Accuracy of automaton state at each active timestep",
           title="Shared hard-policy trajectories: state-estimation accuracy", ylim=(0, 1.03))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "step_accuracy.png", dpi=220)
    fig.savefig(output_dir / "step_accuracy.pdf")
    plt.close(fig)

    # Entropy over time for three prespecified diagnostic specifications.
    chosen_p = max(r["noise_prob"] for r in by_step)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for formula_idx in (3, 4, 7):
        records = sorted(
            (r for r in by_step if r["noise_prob"] == chosen_p
             and r["formula_idx"] == formula_idx and r["num_active"] >= 3),
            key=lambda r: r["timestep"],
        )
        if records:
            ax.plot([r["timestep"] for r in records],
                    [r["mean_belief_entropy"] for r in records],
                    label=f"Formula {formula_idx}")
    ax.set(xlabel="Physical environment step (active episodes only)",
           ylabel="Mean entropy of automaton-state belief (nats)",
           title=f"Belief uncertainty during hard-policy rollouts (p={chosen_p:.2f})")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_by_step.png", dpi=220)
    fig.savefig(output_dir / "entropy_by_step.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with initialize(version_base="1.3", config_path="../../conf"):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            f"eval.num_episodes={args.num_episodes}",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
            f"+eval.num_seeds={args.num_seeds}",
        ])
    original = list(cfg.formulas)
    if any(i < 0 or i >= len(original) for i in args.formula_indices):
        raise ValueError("Selected formula index outside configured finite task set")
    cfg.formulas = [original[i] for i in args.formula_indices]
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

    key = jax.random.key(0)
    _, model_key = jax.random.split(key)
    models, seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    models, seeds = select_model_seeds(models, seeds, args.num_seeds)
    nseeds = len(seeds)
    LOG.info("Selected model seeds %s", seeds)

    episode_rows = []
    timestep_rows = []
    performed_reference_gate = False

    for s, model_seed in enumerate(seeds):
        # Models from the standard loader are stacked along their leading axis.
        single_model = jax.tree.map(
            lambda x: x[s] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == nseeds else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, single_model)
        for f, formula_idx in enumerate(args.formula_indices):
            task = jax.tree.map(lambda x: x[f], formulas)
            for p in args.noise_levels:
                env = VectorizeWrapper(
                    NoisySemanticLDBAWrapper(base_env, noise_prob=p, overwrite_finite=True)
                )
                # Different seed/formula/noise draws. Reproducible across reruns;
                # exactly one trajectory is observed by both estimators.
                run_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(100), int(model_seed)),
                    int(formula_idx),
                )
                run_key = jax.random.fold_in(run_key, int(round(p * 1_000_000)))
                episode, traces = rollout_hard_with_passive_belief(
                    agent, env, base_env, env_params, task, run_key,
                    args.num_episodes, max_steps, p, bool(cfg.eval.deterministic),
                )
                episode = jax.tree.map(np.asarray, episode)
                traces = jax.tree.map(np.asarray, traces)
                if not np.all(episode.completed):
                    raise RuntimeError("At least one episode exceeded max_steps without terminating")
                if not performed_reference_gate and not args.skip_parity_check:
                    reference_parity_gate(
                        agent, env, env_params, task, run_key, args.num_episodes, episode,
                        bool(cfg.eval.deterministic)
                    )
                    performed_reference_gate = True
                if p == 0.0:
                    active = traces.active.astype(bool)
                    if not np.array_equal(traces.hard_state[active], traces.map_state[active]):
                        raise AssertionError(f"p=0 hard/MAP state mismatch at seed {model_seed}, formula {formula_idx}")
                    if not np.allclose(traces.true_prob[active], 1.0, atol=1e-5):
                        raise AssertionError("p=0 belief is not concentrated on the true automaton state")
                for ep in range(args.num_episodes):
                    n = int(episode.steps[ep])
                    row = {
                        "noise_prob": p, "model_seed": int(model_seed),
                        "formula_idx": int(formula_idx), "formula": formula_strings[f],
                        "episode": ep, "steps": n,
                        "success": int(episode.returns[ep] > 0),
                        "violation": int(episode.violations[ep]),
                        "hard_correct_steps": int(episode.hard_correct_steps[ep]),
                        "map_correct_steps": int(episode.map_correct_steps[ep]),
                        "hard_step_accuracy": float(episode.hard_correct_steps[ep] / n),
                        "map_step_accuracy": float(episode.map_correct_steps[ep] / n),
                        "prob_true_sum": float(episode.prob_true_sum[ep]),
                        "nll_sum": float(episode.nll_sum[ep]),
                        "entropy_sum": float(episode.entropy_sum[ep]),
                        "mean_true_state_probability": float(episode.prob_true_sum[ep] / n),
                        "mean_nll": float(episode.nll_sum[ep] / n),
                        "mean_entropy": float(episode.entropy_sum[ep] / n),
                        "hard_any_disagreement": int(episode.hard_any_disagreement[ep]),
                        "map_any_disagreement": int(episode.map_any_disagreement[ep]),
                    }
                    episode_rows.append(row)
                    for t in range(n):
                        timestep_rows.append({
                            "noise_prob": p, "model_seed": int(model_seed),
                            "formula_idx": int(formula_idx), "episode": ep,
                            "timestep": t + 1,
                            "true_state": int(traces.true_state[t, ep]),
                            "hard_state": int(traces.hard_state[t, ep]),
                            "map_state": int(traces.map_state[t, ep]),
                            "hard_correct": int(traces.hard_correct[t, ep]),
                            "map_correct": int(traces.map_correct[t, ep]),
                            "true_state_probability": float(traces.true_prob[t, ep]),
                            "negative_log_true_state_probability": float(traces.neg_log_true_prob[t, ep]),
                            "belief_entropy": float(traces.entropy[t, ep]),
                        })
                LOG.info("seed=%s formula=%d p=%.2f: hard=%.3f MAP=%.3f (timestep accuracy)",
                         model_seed, formula_idx, p,
                         float(episode.hard_correct_steps.sum() / episode.steps.sum()),
                         float(episode.map_correct_steps.sum() / episode.steps.sum()))

    summary = summarize_seed_records(episode_rows)
    by_step = summarize_timestep_records(timestep_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "episodes.csv", episode_rows)
    save_csv(args.output_dir / "steps.csv", timestep_rows)
    save_csv(args.output_dir / "summary.csv", summary)
    save_csv(args.output_dir / "by_step.csv", by_step)
    metadata = {
        "git_revision_at_run": current_commit(),
        "method": "hard-policy closed-loop rollout with passive Bayesian replay on identical noisy observations",
        "no_active_map_or_weighted_policy": True,
        "noise_probabilities": args.noise_levels,
        "formula_indices": args.formula_indices,
        "model_seeds": [int(s) for s in seeds],
        "num_episodes_per_seed_formula_condition": args.num_episodes,
        "randomness": "key(100) folded with model seed, original formula index, integer noise probability",
        "uniform_assignment_prior": True,
        "deterministic_policy": bool(cfg.eval.deterministic),
        "true_proposition_access": "ground-truth monitor and scoring only; passive filter receives perceived labels",
        "initial_observation": "initial LDBA state remains one-hot; score after each physical action",
        "confidence_intervals": "two-sided Student t across pretrained model seeds, fixed formula set",
        "reference_evaluator_parity": ("skipped by flag" if args.skip_parity_check else "PASS"),
        "limitations": [
            "Policy-induced trajectory distribution from hard baseline only",
            "Uniform independent proposition prior is not LetterWorld's physical dynamics",
            "Probability of true state and entropy measured on hard-policy trajectories only",
            "No epsilon transitions evaluated",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    maybe_plot(summary, by_step, args.output_dir)
    print("\nSHARED-TRAJECTORY state-estimation diagnostic (hard policy controls actions):")
    print(f"{'noise':>7} {'hard_acc':>11} {'MAP_acc':>11} {'MAP-hard':>11} {'P(true)':>11} {'NLL':>9} {'entropy':>9}")
    for p in args.noise_levels:
        vals = {r["metric"]: r["mean"] for r in summary if r["scope"] == "overall" and r["noise_prob"] == p}
        print(f"{p:7.2f} {vals['hard_step_accuracy']:11.4f} {vals['map_step_accuracy']:11.4f} "
              f"{vals['accuracy_delta']:+11.4f} {vals['true_state_probability']:11.4f} "
              f"{vals['negative_log_true_state_probability']:9.4f} {vals['belief_entropy']:9.4f}")
    print(f"\nSaved: {args.output_dir}")
    print("Inspect summary.csv for seed-level CIs and by_step.csv for time dependence.")


if __name__ == "__main__":
    main()
