import jax
import jax.numpy as jnp
import pytest
from experiments.belief_semltl.learned_estimator import RecurrentStateEstimator


def test_recurrent_state_estimator_step():
    key = jax.random.PRNGKey(0)
    model = RecurrentStateEstimator(key=key, max_states=14)
    h0 = model.initial_hidden()

    y = jnp.array(3, dtype=jnp.int32)
    u = jnp.array(1, dtype=jnp.int32)
    q = jnp.array(0, dtype=jnp.int32)
    num_states = jnp.array(6, dtype=jnp.int32)

    h1, logits = model.step(h0, y, u, q, num_states)
    assert h1.shape == (model.hidden_dim,)
    assert logits.shape == (14,)
    # Check that invalid states are masked with -1e9
    assert jnp.all(logits[6:] == -1e9)
    assert jnp.all(logits[:6] > -1e9)


def test_recurrent_state_estimator_sequence():
    key = jax.random.PRNGKey(42)
    model = RecurrentStateEstimator(key=key, max_states=14)
    T = 20

    ys = jax.random.randint(key, (T,), 0, 13)
    us = jax.random.randint(key, (T,), 0, 5)
    qs = jax.random.randint(key, (T,), 0, 6)
    num_states = jnp.array(6, dtype=jnp.int32)

    hiddens, logits = model(ys, us, qs, num_states)
    assert hiddens.shape == (T, model.hidden_dim)
    assert logits.shape == (T, 14)
    assert jnp.all(logits[:, 6:] == -1e9)


def test_recurrent_state_estimator_gradient():
    key = jax.random.PRNGKey(123)
    model = RecurrentStateEstimator(key=key, max_states=14)
    T = 10

    ys = jax.random.randint(key, (T,), 0, 13)
    us = jax.random.randint(key, (T,), 0, 5)
    qs = jax.random.randint(key, (T,), 0, 6)
    targets = jax.random.randint(key, (T,), 0, 6)
    num_states = jnp.array(6, dtype=jnp.int32)

    def loss_fn(m):
        _, logits = m(ys, us, qs, num_states)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        return -jnp.mean(target_log_probs)

    grads = jax.grad(loss_fn)(model)
    # Ensure gradients are finite
    assert jnp.all(jnp.isfinite(grads.head.weight))
    assert jnp.all(jnp.isfinite(grads.gru_cell.weight_hh))
