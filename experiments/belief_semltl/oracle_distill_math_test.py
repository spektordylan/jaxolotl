"""Pure JAX tests; don't require a pretrained checkpoint or SemML."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from experiments.belief_semltl.oracle_distill_math import (
    NUM_PHYSICAL_ACTIONS, physical_action_probs, teacher_kl, validate_dataset,
)


class ToyPhysicalDistribution:
    def log_prob(self, action):
        physical, epsilon = action
        logits = jnp.array([0.0, 2.0, -1.0, 1.0])
        return jnp.where(epsilon == 1, logits[physical], -1e6)


def test_distribution_uses_physical_epsilon_sentinel():
    probs = physical_action_probs(ToyPhysicalDistribution(), n=5, epsilon_token=1)
    expected = jax.nn.softmax(jnp.array([0., 2., -1., 1.]))
    assert probs.shape == (5, NUM_PHYSICAL_ACTIONS)
    np.testing.assert_allclose(probs, np.broadcast_to(np.asarray(expected), (5, 4)), atol=1e-6)


def test_forward_kl_zero_for_identical_distributions():
    prob = jax.nn.softmax(jnp.array([[2., 3., -1., 0.], [1., 1., 1., 1.]]))
    assert abs(float(teacher_kl(prob, prob))) < 1e-6


def test_forward_kl_is_positive_and_has_gradient():
    teacher = jnp.array([[0.7, 0.1, 0.1, 0.1]])
    logits = jnp.array([[0., 0., 0., 0.]])
    loss = lambda x: teacher_kl(teacher, jax.nn.softmax(x, axis=-1))
    assert float(loss(logits)) > 0
    grad = jax.grad(loss)(logits)
    assert jnp.all(jnp.isfinite(grad))
    assert abs(float(jnp.sum(grad))) < 1e-6


def example_dataset(seed=0):
    return dict(
        grid=np.zeros((3, 7, 7, 13), np.float32),
        hidden=np.zeros((3, 64), np.float32),
        embedding=np.ones((3, 100), np.float32),
        teacher_probs=np.tile(np.array([[.25, .25, .25, .25]], np.float32), (3, 1)),
        model_seeds=np.full(3, seed, dtype=np.int32),
        noise_probs=np.full(3, .30, dtype=np.float32),
    )


def test_valid_dataset_passes():
    validate_dataset(example_dataset())


def test_evaluation_seed_leak_rejected():
    with pytest.raises(ValueError, match="Evaluation policy seeds"):
        validate_dataset(example_dataset(seed=7))


def test_corrupt_teacher_distribution_rejected():
    data = example_dataset()
    data["teacher_probs"][0] = [.3, .3, .3, .3]
    with pytest.raises(ValueError, match="Invalid normalized"):
        validate_dataset(data)


def test_zero_noise_training_data_rejected():
    data = example_dataset()
    data["noise_probs"][0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        validate_dataset(data)
