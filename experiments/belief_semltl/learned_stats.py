"""Paired seed-level statistics and mandatory zero-noise sanity gate.

Pairs are matched by initial random key (seed, formula, episode). They do NOT
share a physical trajectory once the two policies take different actions.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from experiments.belief_semltl.pilot_stats import seed_ci


def assert_zero_noise_parity(hard, learned, *, seed: int, formula_idx: int) -> None:
    """Stop immediately if the new wrapper changes p=0 reference behavior."""
    for name in (
        "steps", "returns", "violations", "completed", "hard_correct_steps",
        "hard_any_disagreement",
    ):
        before = np.asarray(getattr(hard, name))
        after = np.asarray(getattr(learned, name))
        if before.shape != after.shape or not np.array_equal(before, after):
            raise AssertionError(
                f"p=0 parity failure: seed={seed}, formula={formula_idx}, {name}; "
                f"hard={before}, learned={after}"
            )
    if not np.array_equal(
        np.asarray(learned.learned_correct_steps), np.asarray(hard.hard_correct_steps)
    ):
        raise AssertionError(f"p=0 estimator-state parity failed: seed={seed}, formula={formula_idx}")


def paired_effect_rows(episode_rows: list[dict]) -> list[dict]:
    """Mean paired differences and 95% t CI across independent policy seeds.

    One mean is computed within each pretrained seed. Formulas are a fixed set.
    """
    if not episode_rows:
        raise ValueError("Empty episode records")
    groups = defaultdict(list)
    for row in episode_rows:
        groups[(float(row["noise_prob"]), int(row["formula_idx"]))].append(row)

    ps = sorted({key[0] for key in groups})
    formulas = sorted({key[1] for key in groups})
    names = (
        ("success", "hard_success", "learned_success"),
        ("violation", "hard_violation", "learned_violation"),
        ("episode_step_accuracy", "hard_step_accuracy", "learned_step_accuracy"),
        ("any_disagreement", "hard_any_disagreement", "learned_any_disagreement"),
    )
    result = []
    for p in ps:
        for f_idx in (None, *formulas):
            selected = [row for (noise, formula), rows in groups.items()
                        if noise == p and (f_idx is None or f_idx == formula)
                        for row in rows]
            if not selected:
                continue
            seeds = sorted({int(row["model_seed"]) for row in selected})
            for label, baseline, treatment in names:
                effects = []
                for seed in seeds:
                    paired = [row for row in selected if int(row["model_seed"]) == seed]
                    pairs = [(int(row["formula_idx"]), int(row["episode"])) for row in paired]
                    if len(set(pairs)) != len(pairs):
                        raise ValueError("Duplicate seed/formula/episode pairing")
                    effects.append(np.mean([float(row[treatment]) - float(row[baseline])
                                            for row in paired]))
                ci = seed_ci(np.asarray(effects))
                result.append({
                    "noise_prob": p,
                    "scope": "overall" if f_idx is None else "per_formula",
                    "formula_idx": -1 if f_idx is None else f_idx,
                    "metric": label,
                    "mean_delta": ci.mean,
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "n_seeds": ci.n_seeds,
                    "num_pairs_per_seed": len(selected) // len(seeds),
                })
    return result
