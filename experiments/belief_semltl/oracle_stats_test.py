"""Standalone regression tests for oracle ablation data integrity/statistics."""

import numpy as np
import pytest

from experiments.belief_semltl.oracle_stats import (
    assert_equal_episodes,
    assert_oracle_disagreement_zero,
    summarize,
)


def condition(success, disagreements=None):
    success = np.asarray(success, dtype=float)
    if disagreements is None:
        disagreements = np.zeros_like(success)
    return {
        "success": success,
        "returns": success.copy(),
        "violations": 1.0 - success,
        "lengths": np.full_like(success, 12),
        "disagreements": np.asarray(disagreements, dtype=float),
    }


def test_exact_parity_accepts_identical_arrays():
    a = condition(np.ones((2, 3, 4)))
    assert_equal_episodes(a, condition(np.ones((2, 3, 4))), "sanity")


def test_exact_parity_catches_single_episode_error():
    a = condition(np.ones((2, 3, 4)))
    b = condition(np.ones((2, 3, 4)))
    b["lengths"][1, 2, 3] += 1
    with pytest.raises(AssertionError, match="formula_pos=2 episode=3"):
        assert_equal_episodes(a, b, "p=0")


def test_exact_parity_includes_returns_and_shapes():
    a = condition(np.zeros((2, 2, 2)))
    b = condition(np.zeros((2, 2, 2)))
    b["returns"][0, 0, 1] = 0.1
    with pytest.raises(AssertionError, match="returns"):
        assert_equal_episodes(a, b, "test")
    c = condition(np.zeros((2, 2, 3)))
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_equal_episodes(a, c, "test")


def test_oracle_disagreement_is_zero():
    arr = np.zeros((3, 2, 4))
    assert_oracle_disagreement_zero(condition(arr))
    arr[2, 1, 0] = 1.0
    with pytest.raises(AssertionError, match="Oracle reported monitor disagreement"):
        assert_oracle_disagreement_zero(condition(np.zeros_like(arr), arr))


def test_summary_paired_seed_unit_not_episode_unit():
    hard = condition(np.zeros((3, 2, 4)))
    oracle_success = np.zeros((3, 2, 4))
    oracle_success[0] = 1.0
    oracle_success[1, 0, :2] = 1.0
    oracle = condition(oracle_success)
    episodes, stats, effects = summarize({0.3: {"hard": hard, "oracle": oracle}}, [3, 7, 9], [0, 6])
    assert len(episodes) == 3 * 2 * 4 * 2
    overall = next(row for row in effects if row["scope"] == "overall")
    assert overall["oracle_minus_hard"] == pytest.approx(5 / 12)
    assert overall["num_seeds"] == 3
    oracle_stat = next(row for row in stats if row["scope"] == "overall" and row["method"] == "oracle" and row["metric"] == "success")
    assert oracle_stat["num_seeds"] == 3
    assert oracle_stat["mean"] == pytest.approx(5 / 12)


def test_summary_rejects_mismatched_selected_dimensions():
    a = condition(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="Inconsistent evaluator array shape"):
        summarize({0.0: {"hard": a, "oracle": a}}, [0], [0, 1, 2])
