"""Paired original-GRU vs one-round DAgger analysis from standard evaluator CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from experiments.belief_semltl.pilot_stats import seed_ci
except ModuleNotFoundError:
    # Script can also run from its directory for a local statistical check.
    from pilot_stats import seed_ci


def read_episodes(path: Path) -> dict[tuple[float, int, int, int], dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (float(row["noise_prob"]), int(row["model_seed"]),
                   int(row["formula_idx"]), int(row["episode"]))
            if key in rows:
                raise ValueError(f"Duplicate evaluation key {key} in {path}")
            rows[key] = row
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def compare_rows(original: dict, finetuned: dict) -> list[dict]:
    """Aggregate per-seed paired differences, not independent episode-level CIs."""
    if set(original) != set(finetuned):
        raise ValueError("Original and DAgger evaluations contain different condition keys")
    for key in original:
        a, b = original[key], finetuned[key]
        # Existing runner compares against an identical hard policy on both runs.
        for field in ("hard_success", "hard_violation", "hard_steps"):
            if a[field] != b[field]:
                raise AssertionError(f"Hard reference changed at {key} ({field})")
        # At zero noise learned policy bypasses the GRU completely.
        if key[0] == 0.0:
            for field in ("learned_success", "learned_violation", "learned_steps"):
                if a[field] != b[field]:
                    raise AssertionError(f"Zero-noise DAgger parity failure at {key} ({field})")

    results = []
    noise_levels = sorted({k[0] for k in original})
    for noise in noise_levels:
        for f_idx in (None, *sorted({k[2] for k in original if k[0] == noise})):
            subset = [key for key in original if key[0] == noise and (f_idx is None or key[2] == f_idx)]
            seeds = sorted({key[1] for key in subset})
            for metric in ("success", "violation", "step_accuracy"):
                field = "learned_" + metric
                per_seed = []
                original_seed = []
                finetuned_seed = []
                for seed in seeds:
                    keys = [key for key in subset if key[1] == seed]
                    baseline = np.array([float(original[key][field]) for key in keys])
                    treatment = np.array([float(finetuned[key][field]) for key in keys])
                    original_seed.append(float(np.mean(baseline)))
                    finetuned_seed.append(float(np.mean(treatment)))
                    per_seed.append(float(np.mean(treatment - baseline)))
                ci = seed_ci(np.asarray(per_seed))
                results.append({
                    "noise_prob": noise, "scope": "overall" if f_idx is None else "per_formula",
                    "formula_idx": -1 if f_idx is None else f_idx, "metric": metric,
                    "original_mean": float(np.mean(original_seed)),
                    "dagger_mean": float(np.mean(finetuned_seed)),
                    "mean_delta": ci.mean, "ci_lower": ci.lower, "ci_upper": ci.upper,
                    "n_seeds": ci.n_seeds, "num_pairs_per_seed": len(subset) // len(seeds),
                })
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path, help="Original run_learned_estimator episodes.csv")
    p.add_argument("--dagger", required=True, type=Path, help="Finetuned run_learned_estimator episodes.csv")
    p.add_argument("--output", required=True, type=Path, help="New CSV; refuses to overwrite")
    args = p.parse_args()
    baseline = read_episodes(args.baseline)
    dagger = read_episodes(args.dagger)
    rows = compare_rows(baseline, dagger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nORIGINAL GRU vs ONE-ROUND DAgger (seed-level means):")
    print(f"{'noise':>7} {'original':>10} {'DAgger':>10} {'delta':>9} {'95% CI (pp)':>23}")
    for row in rows:
        if row["scope"] == "overall" and row["metric"] == "success":
            lo = 100 * row["ci_lower"]
            hi = 100 * row["ci_upper"]
            print(f"{row['noise_prob']:7.2f} {row['original_mean']:10.4f} "
                  f"{row['dagger_mean']:10.4f} {row['mean_delta']:+9.4f} "
                  f"[{lo:+.2f}, {hi:+.2f}]")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
