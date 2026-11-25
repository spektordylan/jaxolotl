"""Data-integrity tests for mixture experiment's paired statistics and CSVs."""

import numpy as np
import pytest

from experiments.belief_semltl.mixture_stats import (
    assert_same_episodes,
    method_name,
    summarize,
    write_csv,
)


def synthetic(success):
    success = np.asarray(success, dtype=float)
    return {
        "success": success,
        "returns": success.copy(),
        "lengths": np.full_like(success, 10),
        "violations": np.zeros_like(success),
        "disagreements": np.zeros_like(success),
    }


def test_zero_difference_parity():
    baseline = synthetic(np.ones((2, 2, 3)))
    assert_same_episodes(baseline, synthetic(np.ones((2, 2, 3))), "identical")


def test_parity_fails_on_non_success_metric():
    baseline = synthetic(np.ones((2, 2, 3)))
    other = synthetic(np.ones((2, 2, 3)))
    other["lengths"][0, 1, 2] = 14
    with pytest.raises(AssertionError, match="lengths"):
        assert_same_episodes(baseline, other, "test")


def test_summarize_seed_paired_success_effect():
    h = synthetic(np.array([[[1., 0.]], [[0., 0.]]]))
    m = synthetic(np.array([[[1., 1.]], [[1., 0.]]]))
    _, summary, effects = summarize({0.3: {"hard": h, "mix_1.00": m}},
                                    seeds=[0, 1], formula_indices=[3],
                                    formula_strings=["F a"])
    est = next(x for x in effects if x["scope"] == "overall")
    assert est["mean_delta"] == pytest.approx(0.5)
    assert est["num_seeds"] == 2
    assert any(x["metric"] == "success" for x in summary)


def test_csv_round_trip(tmp_path):
    p = tmp_path / "rows.csv"
    write_csv(p, [{"method": "mix_0.50", "success": 1}])
    assert "mix_0.50" in p.read_text()
    assert method_name(0.25) == "mix_0.25"
