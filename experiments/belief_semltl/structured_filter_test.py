"""Mathematical and archive-layout regression tests (no full Jaxolotl needed)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_semltl.prepare_empirical_prior import estimate_prior
from experiments.belief_semltl.structured_filter_math import (
    assignment_posterior, propagate_belief,
)


def test_bayes_matches_closed_form_and_p0():
    prior = jnp.asarray([.8, .1, .1])
    p = assignment_posterior(jnp.array(1), jnp.array(.3), prior)
    expected = np.array([.8 * .15, .1 * .7, .1 * .15])
    expected /= expected.sum()
    np.testing.assert_allclose(p, expected, atol=1e-6)
    np.testing.assert_array_equal(assignment_posterior(1, 0., prior), [0, 1, 0])


def test_exact_ldba_pushforward_handles_rejecting_sink():
    # Three valid states, one padded. Under assignment 1, q0 -> q1,
    # q1 -> q2 (rejecting sink), q2 remains q2. Padding has -1 transitions.
    trans = jnp.asarray([[0, 1], [1, 2], [2, 2], [-1, -1]])
    belief = jnp.asarray([.3, .5, .2, 0.])
    posterior = jnp.asarray([.2, .8])
    out = propagate_belief(belief, trans, posterior, jnp.array(3))
    # q0 .3 -> q0 .06, q1 .24; q1 .5 -> q1 .10,q2 .40; q2 .2->q2 .2
    np.testing.assert_allclose(out, [.06, .34, .60, 0.], atol=1e-6)
    np.testing.assert_allclose(out.sum(), 1., atol=1e-6)
    out_jit = jax.jit(propagate_belief)(belief, trans, posterior, jnp.array(3))
    np.testing.assert_allclose(out_jit, out, atol=1e-6)


def test_empirical_prior_never_counts_corrupt_training_labels():
    n = 7 * 9 * 3 * 32
    y = np.zeros((n, 2), dtype='int32')
    active = np.ones_like(y, dtype=bool)
    block = np.arange(n) // 32
    p_zero = (block % 3) == 0
    y[p_zero] = 12
    y[~p_zero] = 1
    prior, counts, source = estimate_prior({
        'noisy_assignments': y, 'active_mask': active,
    })
    assert counts.sum() == 7 * 9 * 32 * 2
    assert counts[12] == counts.sum()
    assert counts[1] == 0
    assert 'ASSUMPTION' in source
    with pytest.raises(ValueError, match='Unknown archive'):
        estimate_prior({'noisy_assignments': y[:-1], 'active_mask': active[:-1]})


def test_ground_truth_preferred_if_logged():
    y = np.zeros((3, 2), dtype='int32')
    true = np.full((3, 2), 12, dtype='int32')
    active = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    prior, counts, _ = estimate_prior({
        'noisy_assignments': y, 'true_assignments': true, 'active_mask': active,
    })
    assert counts[12] == 3
    assert counts[0] == 0
    assert np.isclose(prior.sum(), 1)
