"""DAgger regression checks that run without a Jaxolotl environment."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.belief_semltl.summarize_dagger import compare_rows


def row(*, success=0, hard=1, steps=10):
    return {"hard_success": str(hard), "hard_violation": "0", "hard_steps": str(steps),
            "learned_success": str(success), "learned_violation": "0", "learned_steps": str(steps),
            "learned_step_accuracy": "0.75"}


def test_paired_success_delta_across_seeds():
    original, new = {}, {}
    for seed in (7, 8, 9):
        for episode in (0, 1):
            key = (.3, seed, 0, episode)
            original[key] = row(success=0)
            new[key] = row(success=int(episode == 0))
    stats = compare_rows(original, new)
    overall = next(s for s in stats if s["scope"] == "overall" and s["metric"] == "success")
    assert overall["mean_delta"] == pytest.approx(0.5)
    assert overall["n_seeds"] == 3
    assert overall["num_pairs_per_seed"] == 2


def test_rejects_missing_pair():
    with pytest.raises(ValueError, match="different condition"):
        compare_rows({(.3, 7, 0, 0): row()}, {})


def test_rejects_changed_hard_reference():
    with pytest.raises(AssertionError, match="Hard reference"):
        compare_rows({(.3, 7, 0, 0): row(hard=1)}, {(.3, 7, 0, 0): row(hard=0)})


def test_rejects_zero_noise_parity_failure():
    with pytest.raises(AssertionError, match="Zero-noise"):
        compare_rows({(0.0, 7, 0, 0): row(success=1)}, {(0.0, 7, 0, 0): row(success=0)})


def test_success_effect_zero_for_identical_runs():
    data = {(0.0, s, 0, e): row(success=e) for s in (7, 8, 9) for e in (0, 1)}
    result = compare_rows(data, data)
    assert all(r["mean_delta"] == 0 for r in result)
