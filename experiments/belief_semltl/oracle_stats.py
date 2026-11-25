"""Pure NumPy reporting and correctness gates for oracle-monitor ablation.

Inference treats model seeds (not the 16 episodes) as independent replicates.
Specifications are a fixed evaluation set. All intervals are descriptive
Student t intervals across pretrained model seeds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.belief_semltl.pilot_stats import paired_effect, seed_ci

METRICS = ("success", "violations", "lengths", "disagreements")
PARITY_METRICS = ("success", "returns", "violations", "lengths", "disagreements")


def assert_equal_episodes(left: dict, right: dict, label: str) -> None:
    """Exact paired episode equality, including positive reward returns."""
    for metric in PARITY_METRICS:
        a, b = np.asarray(left[metric]), np.asarray(right[metric])
        if a.shape != b.shape:
            raise AssertionError(f"{label}: {metric} shape mismatch {a.shape} != {b.shape}")
        unequal = np.argwhere(a != b)
        if unequal.size:
            seed, formula, episode = map(int, unequal[0])
            raise AssertionError(
                f"{label}: {metric} mismatch at seed_pos={seed} "
                f"formula_pos={formula} episode={episode}: {a[seed,formula,episode]} "
                f"!= {b[seed,formula,episode]}"
            )


def assert_oracle_disagreement_zero(oracle: dict) -> None:
    """Do not mislabel the unused latent corrupted monitor as oracle error."""
    arr = np.asarray(oracle["disagreements"])
    if np.any(arr):
        raise AssertionError(
            "Oracle reported monitor disagreement. Check oracle wrapper info: "
            "the oracle semantic state equals ground truth at every step."
        )


def summarize(results: dict[float, dict[str, dict]], seeds, formula_indices):
    """Create reproducible episode/summary/effect tables from evaluator arrays."""
    episodes, summaries, effects = [], [], []
    seed_list = [int(s) for s in seeds]
    for p in sorted(results):
        conditions = results[p]
        hard = conditions["hard"]
        oracle = conditions["oracle"]
        for mode in ("hard", "oracle"):
            record = conditions[mode]
            nseeds, nformulas, nepisodes = record["success"].shape
            if (nseeds, nformulas) != (len(seed_list), len(formula_indices)):
                raise ValueError("Inconsistent evaluator array shape")
            for s, seed in enumerate(seed_list):
                for f, idx in enumerate(formula_indices):
                    for e in range(nepisodes):
                        episodes.append({
                            "noise_prob": p,
                            "method": mode,
                            "model_seed": seed,
                            "formula_idx": idx,
                            "episode": e,
                            "success": int(record["success"][s, f, e]),
                            "return": float(record["returns"][s, f, e]),
                            "violation": int(record["violations"][s, f, e]),
                            "length": int(record["lengths"][s, f, e]),
                            "monitor_disagreement": int(record["disagreements"][s, f, e]),
                        })
            for scope, pos in [("overall", None)] + [
                ("formula", f) for f in range(nformulas)
            ]:
                formula_idx = -1 if pos is None else formula_indices[pos]
                for name in METRICS:
                    arr = record[name]
                    seed_means = arr.mean(axis=(1, 2)) if pos is None else arr[:, pos, :].mean(axis=1)
                    est = seed_ci(seed_means)
                    summaries.append({
                        "noise_prob": p,
                        "method": mode,
                        "scope": scope,
                        "formula_idx": formula_idx,
                        "metric": name,
                        "mean": est.mean,
                        "ci_lower": est.lower,
                        "ci_upper": est.upper,
                        "num_seeds": est.n_seeds,
                        "num_episodes_per_seed_formula": nepisodes,
                    })
        # Paired effects across seeds and matched reset/sampling key schedules;
        # post-intervention paths are not required to be identical.
        for scope, pos in [("overall", None)] + [
            ("formula", f) for f in range(hard["success"].shape[1])
        ]:
            a = oracle["success"] if pos is None else oracle["success"][:, pos:pos+1, :]
            b = hard["success"] if pos is None else hard["success"][:, pos:pos+1, :]
            est = paired_effect(a, b)
            effects.append({
                "noise_prob": p,
                "scope": scope,
                "formula_idx": -1 if pos is None else formula_indices[pos],
                "oracle_minus_hard": est.mean,
                "ci_lower": est.lower,
                "ci_upper": est.upper,
                "num_seeds": est.n_seeds,
            })
    return episodes, summaries, effects


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
