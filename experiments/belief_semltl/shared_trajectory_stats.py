"""Pure NumPy summaries for the shared-trajectory SemLTL diagnostic.

The unit for confidence intervals is the pretrained model seed. Episodes and
formulas are fixed evaluation conditions, not independent training replicates.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from experiments.belief_semltl.pilot_stats import seed_ci

METRICS = (
    "hard_step_accuracy",
    "map_step_accuracy",
    "accuracy_delta",
    "true_state_probability",
    "negative_log_true_state_probability",
    "belief_entropy",
    "hard_episode_disagreement",
    "map_episode_disagreement",
)


def safe_neg_log_prob(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Finite log score, with epsilon clipping explicitly documented."""
    arr = np.asarray(prob, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
        raise ValueError("Probability must be finite and in [0,1]")
    if not 0 < eps < 1:
        raise ValueError("eps must be in (0,1)")
    return -np.log(np.clip(arr, eps, 1.0))


def summarize_seed_records(rows: Iterable[dict]) -> list[dict]:
    """Pool active timesteps *within* each seed, then construct seed-level CIs.

    Assumes input rows contain per-episode integer counts/sums, rather than
    already rounded means, to avoid equal-weighting 2-step and 100-step episodes.
    """
    data = list(rows)
    if not data:
        raise ValueError("No episode data supplied")
    results: list[dict] = []
    noises = sorted(set(float(row["noise_prob"]) for row in data))
    formulas = sorted(set(int(row["formula_idx"]) for row in data))
    seeds = sorted(set(int(row["model_seed"]) for row in data))

    for p in noises:
        for scope, formula in [("overall", -1)] + [("formula", f) for f in formulas]:
            selected = [r for r in data if float(r["noise_prob"]) == p and
                        (formula == -1 or int(r["formula_idx"]) == formula)]
            by_seed = {s: [r for r in selected if int(r["model_seed"]) == s] for s in seeds}
            if not all(by_seed.values()):
                raise ValueError(f"Missing seed in noise={p}, formula={formula}")
            totals = {}
            for seed, group in by_seed.items():
                nsteps = sum(int(r["steps"]) for r in group)
                nepisodes = len(group)
                if nsteps <= 0:
                    raise ValueError("Encountered seed with zero active steps")
                hard_count = sum(float(r["hard_correct_steps"]) for r in group)
                map_count = sum(float(r["map_correct_steps"]) for r in group)
                totals[seed] = {
                    "hard_step_accuracy": hard_count / nsteps,
                    "map_step_accuracy": map_count / nsteps,
                    "accuracy_delta": (map_count - hard_count) / nsteps,
                    "true_state_probability": sum(float(r["prob_true_sum"]) for r in group) / nsteps,
                    "negative_log_true_state_probability": sum(float(r["nll_sum"]) for r in group) / nsteps,
                    "belief_entropy": sum(float(r["entropy_sum"]) for r in group) / nsteps,
                    "hard_episode_disagreement": sum(float(r["hard_any_disagreement"]) for r in group) / nepisodes,
                    "map_episode_disagreement": sum(float(r["map_any_disagreement"]) for r in group) / nepisodes,
                }
            for metric in METRICS:
                values = np.array([totals[seed][metric] for seed in seeds], dtype=float)
                ci = seed_ci(values)
                results.append({
                    "noise_prob": p, "scope": scope, "formula_idx": formula,
                    "metric": metric, "mean": ci.mean, "ci_lower": ci.lower,
                    "ci_upper": ci.upper, "num_seeds": ci.n_seeds,
                    "num_episodes": len(selected),
                })
    return results


def summarize_timestep_records(rows: Iterable[dict]) -> list[dict]:
    """Aggregate simultaneously observed hard/MAP metrics at each step index.

    Each record is one *active* (seed, formula, episode, timestep) combination.
    There are no fabricated post-termination observations.
    """
    groups: dict[tuple[float, int, int], list[dict]] = {}
    for row in rows:
        key = (float(row["noise_prob"]), int(row["formula_idx"]), int(row["timestep"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (p, f, t), group in sorted(groups.items()):
        if not group:
            continue
        result.append({
            "noise_prob": p, "formula_idx": f, "timestep": t,
            "num_active": len(group),
            "hard_step_accuracy": float(np.mean([r["hard_correct"] for r in group])),
            "map_step_accuracy": float(np.mean([r["map_correct"] for r in group])),
            "mean_true_state_probability": float(np.mean([r["true_state_probability"] for r in group])),
            "mean_negative_log_probability": float(np.mean([r["negative_log_true_state_probability"] for r in group])),
            "mean_belief_entropy": float(np.mean([r["belief_entropy"] for r in group])),
        })
    return result
