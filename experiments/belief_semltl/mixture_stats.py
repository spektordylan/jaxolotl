"""Statistics and stable CSV schema for policy-mixture ablations.

Uncertainty intervals use pretrained model *seeds* as replications, treating
nine formulas as a fixed evaluation set. Results are exploratory and should
not be interpreted as held-out formula generalization.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.belief_semltl.pilot_stats import paired_effect, seed_ci


def method_name(alpha: float) -> str:
    return f"mix_{alpha:.2f}"


def assert_same_episodes(a, b, label: str):
    for field in ("success", "returns", "violations", "lengths", "disagreements"):
        left = np.asarray(a[field])
        right = np.asarray(b[field])
        if left.shape != right.shape:
            raise AssertionError(f"{label}: {field}: shape mismatch")
        mismatches = np.argwhere(left != right)
        if mismatches.size:
            s, f, e = map(int, mismatches[0])
            raise AssertionError(
                f"{label}: {field} differs at seed={s}, formula={f}, episode={e}: "
                f"reference={left[s,f,e]}, treatment={right[s,f,e]}"
            )


def summarize(results, seeds, formula_indices, formula_strings):
    """Inputs results[p][method][metric] with shape (seed,formula,episode)."""
    episodes, summaries, effects = [], [], []
    for p, methods in sorted(results.items()):
        baseline = methods["hard"]["success"]
        for method, data in methods.items():
            n_seeds, n_formulas, n_episodes = data["success"].shape
            for s in range(n_seeds):
                for f in range(n_formulas):
                    for e in range(n_episodes):
                        episodes.append({
                            "noise_prob": p,
                            "method": method,
                            "model_seed": int(seeds[s]),
                            "formula_idx": int(formula_indices[f]),
                            "formula": formula_strings[f],
                            "episode": e,
                            "success": int(data["success"][s, f, e]),
                            "return": float(data["returns"][s, f, e]),
                            "violation": int(data["violations"][s, f, e]),
                            "length": int(data["lengths"][s, f, e]),
                            "disagreement": int(data["disagreements"][s, f, e]),
                        })
            for metric, field in (
                ("success", "success"),
                ("violation", "violations"),
                ("disagreement", "disagreements"),
            ):
                arr = np.asarray(data[field])
                for f in [None, *range(n_formulas)]:
                    per_seed = arr.mean(axis=(1, 2)) if f is None else arr[:, f, :].mean(axis=1)
                    estimate = seed_ci(per_seed)
                    summaries.append({
                        "noise_prob": p, "method": method,
                        "scope": "overall" if f is None else "formula",
                        "formula_idx": -1 if f is None else int(formula_indices[f]),
                        "metric": metric,
                        "mean": estimate.mean,
                        "ci_lower": estimate.lower,
                        "ci_upper": estimate.upper,
                        "num_seeds": estimate.n_seeds,
                        "num_formulas": n_formulas if f is None else 1,
                        "num_episodes": n_episodes,
                    })
            if method != "hard":
                for f in [None, *range(n_formulas)]:
                    treatment = data["success"] if f is None else data["success"][:, f:f + 1, :]
                    control = baseline if f is None else baseline[:, f:f + 1, :]
                    est = paired_effect(treatment, control)
                    effects.append({
                        "noise_prob": p, "method": method,
                        "scope": "overall" if f is None else "formula",
                        "formula_idx": -1 if f is None else int(formula_indices[f]),
                        "mean_delta": est.mean, "ci_lower": est.lower,
                        "ci_upper": est.upper, "num_seeds": est.n_seeds,
                    })
    return episodes, summaries, effects


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No data to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
