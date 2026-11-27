"""Tests of posterior masking and exact probability (not logit) mixing."""
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_semltl.belief_aggregation_math import (
    valid_state_posterior, mix_action_probabilities,
)


def test_masked_posterior_zeroes_padded_states():
    logits = jnp.array([[1.0, 2.0, 100.0], [0.0, 0.0, 0.0]])
    probs = valid_state_posterior(
        logits, jnp.array([2, 3]), jnp.array([1, 0]),
        initial_step=jnp.array(False), zero_noise=False,
    )
    assert float(probs[0, 2]) == 0.0
    np.testing.assert_allclose(np.array(probs.sum(axis=-1)), 1, atol=1e-6)


def test_initial_and_zero_noise_always_use_known_state():
    logits = jnp.zeros((2, 4))
    map_state = jnp.array([2, 1])
    for initial, zero in [(True, False), (False, True)]:
        probs = valid_state_posterior(
            logits, jnp.array([3, 3]), map_state,
            initial_step=jnp.array(initial), zero_noise=zero,
        )
        np.testing.assert_allclose(np.array(probs), np.eye(4)[[2, 1]])


def test_exact_action_mixture_and_map_anchor():
    pi = jnp.array([[[0.9, 0.1], [0.2, 0.8]]])
    weights = jnp.array([[0.6, 0.4]])
    np.testing.assert_allclose(
        np.array(mix_action_probabilities(pi, weights, alpha=1.0)),
        [[0.62, 0.38]], atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(mix_action_probabilities(pi, weights, alpha=0.0)),
        [[0.9, 0.1]], atol=1e-6,
    )


def test_single_state_matches_map_and_zero_weight_state_ignored():
    pi = jnp.array([[[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]])
    for weight in ([0.8, 0.0], [1.0, 0.0]):
        out = mix_action_probabilities(pi, jnp.array([weight]))
        np.testing.assert_allclose(np.array(out), [[0.2, 0.3, 0.5]], atol=1e-6)


def test_wrong_alpha_rejected():
    with pytest.raises(ValueError):
        mix_action_probabilities(jnp.ones((1, 1, 4)) / 4, jnp.ones((1, 1)), alpha=1.1)
