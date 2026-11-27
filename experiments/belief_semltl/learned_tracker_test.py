import jax
import jax.numpy as jnp
import pytest
from experiments.belief_semltl.learned_tracker import LearnedStateEstimator


def test_learned_state_estimator_step():
    key = jax.random.PRNGKey(0)
    model = LearnedStateEstimator(key=key, max_states=14)
    h0 = model.initial_hidden()

    y = jnp.array(3, dtype=jnp.int32)
    u = jnp.array(1, dtype=jnp.int32)
    q = jnp.array(0, dtype=jnp.int32)

    h1, logits = model.step(h0, y, u, q)
    assert h1.shape == (model.hidden_dim,)
    assert logits.shape == (14,)


def test_learned_state_estimator_unroll():
    key = jax.random.PRNGKey(42)
    model = LearnedStateEstimator(key=key, max_states=14)
    T = 20

    ys = jax.random.randint(key, (T,), 0, 13)
    us = jax.random.randint(key, (T,), 0, 5)
    qs = jax.random.randint(key, (T,), 0, 14)

    logits = model.unroll(ys, us, qs)
    assert logits.shape == (T, 14)


def test_learned_state_estimator_gradient():
    key = jax.random.PRNGKey(123)
    model = LearnedStateEstimator(key=key, max_states=14)
    T = 10

    ys = jax.random.randint(key, (T,), 0, 13)
    us = jax.random.randint(key, (T,), 0, 5)
    qs = jax.random.randint(key, (T,), 0, 14)
    targets = jax.random.randint(key, (T,), 0, 14)

    def loss_fn(m):
        logits = m.unroll(ys, us, qs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        return -jnp.mean(target_log_probs)

    grads = jax.grad(loss_fn)(model)
    assert jnp.all(jnp.isfinite(grads.head.weight))
    assert jnp.all(jnp.isfinite(grads.gru_cell.weight_hh))


def test_state_mask_uses_per_formula_state_counts():
    from experiments.belief_semltl.learned_tracker import mask_invalid_states

    logits = jnp.array([[[1., 0., 100., 100.]], [[1., 0., 2., 100.]]])
    masked = mask_invalid_states(logits, jnp.array([2, 3]))
    assert jnp.array_equal(jnp.argmax(masked, axis=-1), jnp.array([[0], [2]]))
    assert jnp.all(masked[0, :, 2:] == -1e9)
    assert jnp.all(masked[1, :, 3:] == -1e9)
