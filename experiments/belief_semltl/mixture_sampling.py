"""Pure JAX sampling utilities for exact hierarchical mixtures of policies.

These select an *existing* semantic-state-conditioned policy before sampling
its action.  For stochastic policies this realizes, exactly in distribution,
  (1-alpha) pi(.|e(q_hard)) + alpha sum_q b(q) pi(.|e(q)).
No privileged true state is used. The caller retains the original action RNG
key and derives *separate* selection randomness with fold_in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def choose_policy_embeddings(
    actor_key: jax.Array,
    hard_embeddings: jax.Array,
    state_embeddings: jax.Array,
    beliefs: jax.Array,
    alpha: float,
) -> jax.Array:
    """Select conditioning embeddings in a vectorized batch.

    Shapes:
      hard_embeddings (episodes, embedding_dim)
      state_embeddings (episodes, states, embedding_dim)
      beliefs (episodes, states)
    State selection is *not* action sampling. Caller subsequently invokes the
    original pretrained actor's distribution.sample(seed=actor_key).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0,1]")
    if hard_embeddings.ndim != 2 or state_embeddings.ndim != 3 or beliefs.ndim != 2:
        raise ValueError("Expected batched 2D hard/belief and 3D state embeddings")
    n, d = hard_embeddings.shape
    if state_embeddings.shape != (n, beliefs.shape[1], d):
        raise ValueError("Embedding and belief dimensions are inconsistent")

    # Consume no new actor key: the action head receives exactly actor_key.
    # fold_in is deterministic and does not mutate the provided key.
    state_key = jax.random.fold_in(actor_key, 294)
    gate_key = jax.random.fold_in(actor_key, 295)

    # jax.random.categorical supports batched logit rows. Normalize defensively.
    probs = jnp.maximum(beliefs, 0.0)
    probs = probs / jnp.maximum(probs.sum(axis=-1, keepdims=True), 1e-30)
    logits = jnp.where(probs > 0, jnp.log(jnp.maximum(probs, 1e-30)), -jnp.inf)
    chosen_q = jax.random.categorical(state_key, logits, axis=-1)
    chosen_embedding = jnp.take_along_axis(
        state_embeddings, chosen_q[:, None, None], axis=1
    )[:, 0, :]
    use_belief = jax.random.bernoulli(gate_key, p=alpha, shape=(n,))
    return jnp.where(use_belief[:, None], chosen_embedding, hard_embeddings)


def marginal_mixture(action_probabilities: jax.Array, beliefs: jax.Array, alpha: float,
                     hard_action_probabilities: jax.Array) -> jax.Array:
    """Analytical reference, useful for tests but not needed in production.

    pi_states: (batch, states, actions), belief: (batch, states),
    hard_pi: (batch, actions). Returns (batch, actions).
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha outside [0,1]")
    if action_probabilities.ndim != 3 or beliefs.ndim != 2:
        raise ValueError("Invalid action probability dimensions")
    return ((1 - alpha) * hard_action_probabilities
            + alpha * jnp.sum(beliefs[:, :, None] * action_probabilities, axis=1))
