"""Pure-statistics tests; require only numpy/scipy/pytest, not Jaxolotl."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.belief_semltl.learned_stats import (
    assert_zero_noise_parity,
    paired_effect_rows,
)


def _episodes():
    return SimpleNamespace(
        steps=np.array([12, 20]), returns=np.array([1, 0]),
        violations=np.array([False, True]), completed=np.array([True, True]),
        hard_correct_steps=np.array([12, 20]),
        learned_correct_steps=np.array([12, 20]),
        hard_any_disagreement=np.array([False, False]),
    )


def test_zero_noise_parity():
    hard = _episodes()
    learned = _episodes()
    assert_zero_noise_parity(hard, learned, seed=7, formula_idx=0)
    learned.returns[1] = 1
    with pytest.raises(AssertionError, match="returns"):
        assert_zero_noise_parity(hard, learned, seed=7, formula_idx=0)


def test_paired_aggregation_uses_seed_not_episode_as_replication():
    data = []
    for seed, delta in [(7, 0), (8, 1), (9, 0)]:
        for formula in (0, 1):
            for ep in range(4):
                data.append({
                    "noise_prob": .3, "model_seed": seed,
                    "formula_idx": formula, "episode": ep,
                    "hard_success": 0, "learned_success": delta,
                    "hard_violation": 0, "learned_violation": 0,
                    "hard_step_accuracy": .5, "learned_step_accuracy": .75,
                    "hard_any_disagreement": 0, "learned_any_disagreement": 0,
                })
    rows = paired_effect_rows(data)
    success = next(x for x in rows if x["scope"] == "overall" and x["metric"] == "success")
    assert success["mean_delta"] == pytest.approx(1/3)
    assert success["n_seeds"] == 3
    assert success["num_pairs_per_seed"] == 8
    accuracy = next(x for x in rows if x["scope"] == "overall" and x["metric"] == "episode_step_accuracy")
    assert accuracy["mean_delta"] == pytest.approx(.25)
    data.append(dict(data[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        paired_effect_rows(data)
