"""Tests the exact-mixture math and key-preserving state selector."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_semltl.mixture_sampling import (
    choose_policy_embeddings,
    marginal_mixture,
)


def test_zero_alpha_returns_hard_embedding_bitwise():
    hard = jnp.asarray([[42.0, 99.0], [30.0, 50.0]])
    all_embeddings = jnp.asarray([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    beliefs = jnp.asarray([[0.3, 0.7], [0.99, 0.01]])
    got = choose_policy_embeddings(jax.random.key(11), hard, all_embeddings, beliefs, 0.0)
    np.testing.assert_array_equal(got, hard)


def test_one_hot_state_any_alpha_returns_same_hard_embedding():
    all_embeddings = jnp.asarray([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    hard = jnp.asarray([[3., 4.], [5., 6.]])
    beliefs = jnp.asarray([[0., 1.], [1., 0.]])
    for a in [0.0, 0.2, 0.5, 1.0]:
        got = choose_policy_embeddings(jax.random.key(12), hard, all_embeddings, beliefs, a)
        np.testing.assert_array_equal(got, hard)


def test_stochastic_state_selection_matches_belief_frequency():
    # Repeated keys simulate independent stochastic decisions; this test is
    # about selector marginals, NOT full neural action distributions.
    n = 5000
    hard = jnp.zeros((n, 1))
    embeddings = jnp.broadcast_to(jnp.array([[[1.], [2.]]]), (n, 2, 1))
    beliefs = jnp.broadcast_to(jnp.array([[0.2, 0.8]]), (n, 2))
    chosen = choose_policy_embeddings(jax.random.key(3), hard, embeddings, beliefs, 1.0)
    frac = float(jnp.mean(chosen[:, 0] == 2.))
    assert 0.77 < frac < 0.83


def test_analytical_mixture_and_probability_mass():
    p_states = jnp.asarray([[[0.8, 0.2], [0.1, 0.9]]])
    belief = jnp.asarray([[0.25, 0.75]])
    hard = jnp.asarray([[0.4, 0.6]])
    expected_belief_pi = np.array([[0.275, 0.725]])
    np.testing.assert_allclose(
        marginal_mixture(p_states, belief, 1.0, hard), expected_belief_pi, atol=1e-6
    )
    np.testing.assert_allclose(
        marginal_mixture(p_states, belief, 0.5, hard),
        0.5 * expected_belief_pi + 0.5 * np.array(hard), atol=1e-6,
    )
    np.testing.assert_allclose(
        marginal_mixture(p_states, belief, 0.5, hard).sum(axis=-1), [1.], atol=1e-6
    )


def test_invalid_alpha_or_shapes_rejected():
    with pytest.raises(ValueError):
        choose_policy_embeddings(jax.random.key(0), jnp.ones((2, 1)),
                                 jnp.ones((2, 2, 1)), jnp.ones((2, 2)), 1.1)
    with pytest.raises(ValueError):
        choose_policy_embeddings(jax.random.key(0), jnp.ones((2, 1)),
                                 jnp.ones((3, 2, 1)), jnp.ones((2, 2)), 0.5)


def test_jitted_selector_matches_unjitted():
    hard = jnp.asarray([[2., 3.]])
    embeddings = jnp.asarray([[[2., 3.], [4., 5.]]])
    belief = jnp.asarray([[0., 1.]])
    fn = jax.jit(choose_policy_embeddings, static_argnums=4)
    direct = choose_policy_embeddings(jax.random.key(10), hard, embeddings, belief, 1.0)
    compiled = fn(jax.random.key(10), hard, embeddings, belief, 1.0)
    np.testing.assert_array_equal(direct, compiled)
