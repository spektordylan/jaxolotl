"""Seed-level statistics for the frozen-policy belief-aware SemLTL pilot.

The unit of replication is a pretrained model seed, not an episode. Formulas
are treated as the fixed, prespecified evaluation set.  Intervals are
exploratory Student-t intervals, not a claim of population-wide generality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float
    n_seeds: int


def seed_ci(seed_values: np.ndarray, confidence: float = 0.95) -> Estimate:
    """Mean and two-sided t interval for one value per independent model seed."""
    values = np.asarray(seed_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty finite 1D array of seed values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    n = values.size
    mu = float(np.mean(values))
    if n == 1:
        # A point estimate is valid, but a seed-level uncertainty interval is not.
        return Estimate(mu, float("nan"), float("nan"), n)
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    width = float(student_t.ppf((1 + confidence) / 2, n - 1) * se)
    return Estimate(mu, mu - width, mu + width, n)


def check_evaluation_arrays(
    returns: np.ndarray,
    violations: np.ndarray,
    lengths: np.ndarray,
    disagreements: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    """Reject malformed/nonfinite rollout output rather than reporting it."""
    for name, arr in {
        "returns": returns,
        "violations": violations,
        "lengths": lengths,
        "disagreements": disagreements,
    }.items():
        if np.shape(arr) != expected_shape:
            raise ValueError(f"{name} shape {np.shape(arr)} != {expected_shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains nonfinite values")
    if np.any(lengths < 0):
        raise ValueError("Negative episode lengths")
    if np.any((violations < 0) | (violations > 1)):
        raise ValueError("Invalid violation indicator")
    if np.any((disagreements < 0) | (disagreements > 1)):
        raise ValueError("Invalid disagreement indicator")


def assert_zero_noise_parity(
    reference: dict[str, np.ndarray],
    other: dict[str, np.ndarray],
    method: str,
) -> None:
    """At p=0, all observable per-episode outcomes must match hard exactly."""
    for metric in ("success", "violations", "lengths", "disagreements"):
        a = np.asarray(reference[metric])
        b = np.asarray(other[metric])
        if a.shape != b.shape:
            raise AssertionError(f"p=0 {method} {metric}: shape {b.shape} != {a.shape}")
        mismatch = np.argwhere(a != b)
        if mismatch.size:
            s, f, e = (int(x) for x in mismatch[0])
            raise AssertionError(
                f"p=0 parity failed for {method}, {metric} at "
                f"seed_index={s}, selected_formula_index={f}, episode={e}: "
                f"hard={a[s, f, e]}, {method}={b[s, f, e]}. "
                "Do not interpret nonzero-noise comparisons until resolved."
            )


def paired_effect(
    treatment: np.ndarray, baseline: np.ndarray
) -> Estimate:
    """Paired treatment-minus-baseline success effect across model seeds.

    Shapes: (seeds, formulas, episodes). First average formulas and episodes
    within each seed, THEN compute the t interval across seed differences.
    """
    treatment = np.asarray(treatment, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if treatment.shape != baseline.shape or treatment.ndim != 3:
        raise ValueError("Paired arrays must share (seed, formula, episode) shape")
    per_seed_delta = (treatment - baseline).mean(axis=(1, 2))
    return seed_ci(per_seed_delta)
