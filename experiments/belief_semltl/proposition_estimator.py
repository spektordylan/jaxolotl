"""Formula-agnostic recurrent predictor P(a_t | y_1:t, u_1:t-1)."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class PropositionEstimator(eqx.Module):
    noisy_embed: eqx.nn.Embedding
    action_embed: eqx.nn.Embedding
    gru_cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int = 64, embed_dim: int = 16,
                 *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.hidden_dim = hidden_dim
        self.noisy_embed = eqx.nn.Embedding(13, embed_dim, key=k1)
        self.action_embed = eqx.nn.Embedding(5, embed_dim, key=k2)
        self.gru_cell = eqx.nn.GRUCell(embed_dim * 2, hidden_dim, key=k3)
        self.head = eqx.nn.Linear(hidden_dim, 13, key=k4)

    def initial_hidden(self):
        return jnp.zeros((self.hidden_dim,), dtype=jnp.float32)

    def step(self, hidden, noisy_assignment, prev_action):
        x = jnp.concatenate((
            self.noisy_embed(jnp.clip(noisy_assignment, 0, 12)),
            self.action_embed(jnp.clip(prev_action, 0, 4)),
        ))
        next_hidden = self.gru_cell(x, hidden)
        return next_hidden, self.head(next_hidden)

    def unroll(self, noisy_assignments, prev_actions):
        def scan(h, pair):
            next_hidden, logits = self.step(h, pair[0], pair[1])
            return next_hidden, logits
        _, logits = jax.lax.scan(scan, self.initial_hidden(),
                                  (noisy_assignments, prev_actions))
        return logits
