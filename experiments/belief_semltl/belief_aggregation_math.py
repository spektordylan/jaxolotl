"""JAX-only operations for mixing frozen-policy action distributions.

These routines never average semantic embeddings or logits. They form a
mixture of *probabilities* over actions, using the GRU state posterior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def valid_state_posterior(
    logits: jax.Array,
    num_states: jax.Array,
    current_map: jax.Array,
    *,
    initial_step: jax.Array,
    zero_noise: bool,
) -> jax.Array:
    """Return (N,S) posterior; reset and zero-noise are known one-hot states.

    The trained tracker was *not* advanced on reset. Evaluating its head on
    h_0 before the first environment transition would be meaningless.
    """
    n, max_states = logits.shape
    num = jnp.asarray(num_states)
    if num.ndim == 0:
        num = jnp.broadcast_to(num, (n,))
    valid = jnp.arange(max_states)[None, :] < num[:, None]
    masked = jnp.where(valid, logits, -jnp.inf)
    posterior = jax.nn.softmax(masked, axis=-1)
    known = jax.nn.one_hot(current_map, max_states, dtype=posterior.dtype)
    return jnp.where(initial_step | zero_noise, known, posterior)


def mix_action_probabilities(
    per_state_action_probs: jax.Array,
    posterior_weights: jax.Array,
    *,
    alpha: float = 1.0,
) -> jax.Array:
    """Mix (N,K,A) action probabilities using (N,K) normalized weights.

    The first candidate must be the MAP state, so alpha=0 is MAP.
    alpha=1 is full belief-weighted action aggregation over the chosen K.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    weights = posterior_weights / jnp.maximum(
        posterior_weights.sum(axis=-1, keepdims=True), 1e-30
    )
    expected = jnp.sum(per_state_action_probs * weights[:, :, None], axis=1)
    result = (1.0 - alpha) * per_state_action_probs[:, 0, :] + alpha * expected
    return result / jnp.maximum(result.sum(axis=-1, keepdims=True), 1e-30)
