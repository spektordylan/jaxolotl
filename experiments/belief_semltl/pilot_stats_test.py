"""Pure statistical/integrity tests. Run without Jaxolotl dependencies."""

import numpy as np
import pytest

from experiments.belief_semltl.pilot_stats import (
    assert_zero_noise_parity,
    check_evaluation_arrays,
    paired_effect,
    seed_ci,
)


def test_seed_ci_identical_seeds():
    result = seed_ci(np.array([0.25, 0.25, 0.25]))
    assert result.n_seeds == 3
    assert result.mean == pytest.approx(0.25)
    assert result.lower == pytest.approx(0.25)
    assert result.upper == pytest.approx(0.25)


def test_seed_ci_one_seed_has_no_interval():
    result = seed_ci(np.array([0.5]))
    assert result.mean == 0.5
    assert np.isnan(result.lower) and np.isnan(result.upper)


def test_paired_effect_uses_seed_level_differences():
    hard = np.zeros((3, 2, 4))
    map_returns = hard.copy()
    map_returns[0] = 1.0
    map_returns[1, 0, :2] = 1.0
    # Per-seed differences: 1.0, 0.25, 0.0 -> mean=5/12.
    estimate = paired_effect(map_returns, hard)
    assert estimate.mean == pytest.approx(5 / 12)
    assert estimate.n_seeds == 3


def test_paired_effect_requires_equal_shapes():
    with pytest.raises(ValueError):
        paired_effect(np.zeros((2, 3, 4)), np.zeros((2, 2, 4)))


def test_zero_noise_parity_detects_single_differing_episode():
    reference = {
        name: np.zeros((2, 3, 4))
        for name in ("success", "violations", "lengths", "disagreements")
    }
    other = {name: arr.copy() for name, arr in reference.items()}
    assert_zero_noise_parity(reference, other, "map")
    other["success"][1, 2, 3] = 1
    with pytest.raises(AssertionError, match="seed_index=1"):
        assert_zero_noise_parity(reference, other, "map")


def test_nonfinite_rollouts_rejected():
    r = np.zeros((2, 3, 4))
    check_evaluation_arrays(r, r, r, r, r.shape)
    bad = r.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        check_evaluation_arrays(bad, r, r, r, r.shape)
