"""CPU-only, repository-independent checks for diagnostic statistics."""

import numpy as np
import pytest

from experiments.belief_semltl.shared_trajectory_stats import (
    safe_neg_log_prob,
    summarize_seed_records,
    summarize_timestep_records,
)


def _row(seed, steps, hard, map_, formula=0, p=0.3):
    return {
        "model_seed": seed, "noise_prob": p, "formula_idx": formula,
        "steps": steps, "hard_correct_steps": hard, "map_correct_steps": map_,
        "prob_true_sum": 0.5 * steps, "nll_sum": 0.7 * steps,
        "entropy_sum": 0.4 * steps,
        "hard_any_disagreement": int(hard != steps),
        "map_any_disagreement": int(map_ != steps),
    }


def test_probability_clipping():
    scores = safe_neg_log_prob(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(scores))
    assert scores[2] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(np.log(2))


def test_probability_input_validation():
    with pytest.raises(ValueError):
        safe_neg_log_prob(np.array([1.2]))


def test_seed_aggregation_weights_timesteps_not_episodes():
    rows = [_row(0, 1, 0, 1), _row(0, 9, 9, 0),
            _row(1, 1, 0, 1), _row(1, 9, 9, 0)]
    summary = summarize_seed_records(rows)
    overall = {(r["scope"], r["metric"]): r for r in summary}
    assert overall[("overall", "hard_step_accuracy")]["mean"] == pytest.approx(0.9)
    assert overall[("overall", "map_step_accuracy")]["mean"] == pytest.approx(0.1)
    assert overall[("overall", "accuracy_delta")]["mean"] == pytest.approx(-0.8)
    assert overall[("overall", "accuracy_delta")]["num_seeds"] == 2


def test_timestep_aggregation_only_active_records():
    records = [
        {"noise_prob": 0.1, "formula_idx": 0, "timestep": 1,
         "hard_correct": 1, "map_correct": 0,
         "true_state_probability": 0.2,
         "negative_log_true_state_probability": 1.609,
         "belief_entropy": 0.5},
        {"noise_prob": 0.1, "formula_idx": 0, "timestep": 1,
         "hard_correct": 0, "map_correct": 1,
         "true_state_probability": 0.8,
         "negative_log_true_state_probability": 0.223,
         "belief_entropy": 0.3},
    ]
    summary = summarize_timestep_records(records)
    assert len(summary) == 1
    assert summary[0]["num_active"] == 2
    assert summary[0]["hard_step_accuracy"] == pytest.approx(0.5)
    assert summary[0]["mean_true_state_probability"] == pytest.approx(0.5)
