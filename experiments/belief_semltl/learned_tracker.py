"""History-aware recurrent task-state estimator for SemLTL under proposition noise.

Predicts a distribution over automaton states given:
- Current noisy proposition observation y_t (13 classes)
- Previous action u_{t-1} (4 physical actions + start indicator = 5 classes)
- Hard monitor state q^hard_t (0 .. max_states-1)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class LearnedStateEstimator(eqx.Module):
    """GRU-based task-state estimator."""

    noisy_embed: eqx.nn.Embedding
    action_embed: eqx.nn.Embedding
    hard_state_embed: eqx.nn.Embedding
    gru_cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)
    max_states: int = eqx.field(static=True)

    def __init__(
        self,
        num_assignments: int = 13,
        num_actions: int = 5,  # 0..3: physical actions, 4: start/init token
        max_states: int = 14,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        *,
        key: jax.Array,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden_dim = hidden_dim
        self.max_states = max_states
        self.noisy_embed = eqx.nn.Embedding(num_assignments, embed_dim, key=k1)
        self.action_embed = eqx.nn.Embedding(num_actions, embed_dim, key=k2)
        self.hard_state_embed = eqx.nn.Embedding(max_states, embed_dim, key=k3)

        in_features = embed_dim * 3
        self.gru_cell = eqx.nn.GRUCell(input_size=in_features, hidden_size=hidden_dim, key=k4)
        self.head = eqx.nn.Linear(hidden_dim, max_states, key=k5)

    def initial_hidden(self) -> jax.Array:
        return jnp.zeros(self.hidden_dim, dtype=jnp.float32)

    def step(
        self,
        hidden: jax.Array,
        noisy_assignment: jax.Array,
        prev_action: jax.Array,
        hard_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step update.

        Args:
            hidden: (hidden_dim,)
            noisy_assignment: scalar int32 (0..12)
            prev_action: scalar int32 (0..4)
            hard_state: scalar int32 (0..max_states-1)

        Returns:
            next_hidden: (hidden_dim,)
            logits: (max_states,)
        """
        # Clamp inputs to valid ranges
        y_safe = jnp.clip(noisy_assignment, 0, 12)
        u_safe = jnp.clip(prev_action, 0, 4)
        q_safe = jnp.clip(hard_state, 0, self.max_states - 1)

        y_emb = self.noisy_embed(y_safe)
        u_emb = self.action_embed(u_safe)
        q_emb = self.hard_state_embed(q_safe)

        x = jnp.concatenate([y_emb, u_emb, q_emb], axis=-1)
        next_hidden = self.gru_cell(x, hidden)
        logits = self.head(next_hidden)
        return next_hidden, logits

    def unroll(
        self,
        noisy_assignments: jax.Array,  # (T,)
        prev_actions: jax.Array,       # (T,)
        hard_states: jax.Array,        # (T,)
        init_hidden: jax.Array | None = None,
    ) -> jax.Array:
        """Sequence scan over a single episode.

        Returns:
            all_logits: (T, max_states)
        """
        if init_hidden is None:
            init_hidden = self.initial_hidden()

        def scan_fn(h, inputs):
            y, u, q = inputs
            next_h, logits = self.step(h, y, u, q)
            return next_h, logits

        inputs = (noisy_assignments, prev_actions, hard_states)
        _, all_logits = jax.lax.scan(scan_fn, init_hidden, inputs)
        return all_logits


def mask_invalid_states(logits: jax.Array, num_states: jax.Array) -> jax.Array:
    """Remove padding states before softmax/argmax (last dimension is Q).

    ``num_states`` is scalar or a vector of shape (B,) for batched logits.
    The checkpoint architecture is unchanged; this is inference/training logic.
    """
    valid = jnp.arange(logits.shape[-1]) < jnp.asarray(num_states)[..., None]
    while valid.ndim < logits.ndim:
        valid = jnp.expand_dims(valid, -2)
    return jnp.where(valid, logits, -1e9)
