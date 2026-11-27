"""Residual semantic adapter for a frozen SemLTL policy and frozen GRU-MAP tracker.

Because the SemLTL semantic projection is linear, perturbing its input by r(h)
induces a residual W r(h) in the projected semantic feature space. Keeping
this intervention at the semantic observation boundary avoids changing the
pretrained actor's implementation and enables gradients through get_action().
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper


class SemanticResidualAdapter(eqx.Module):
    linear: eqx.nn.Linear
    max_delta: float = eqx.field(static=True)

    def __init__(self, hidden_dim: int, semantic_dim: int, *, key: jax.Array,
                 max_delta: float = 1.0):
        if hidden_dim < 1 or semantic_dim < 1 or max_delta <= 0:
            raise ValueError("Invalid adapter dimensions or max_delta")
        linear = eqx.nn.Linear(hidden_dim, semantic_dim, key=key)
        # Exact identity at initialization: essential for a controlled baseline.
        self.linear = eqx.tree_at(lambda l: (l.weight, l.bias), linear,
                                  (jnp.zeros_like(linear.weight),
                                   jnp.zeros_like(linear.bias)))
        self.max_delta = float(max_delta)

    def __call__(self, hidden: jax.Array) -> jax.Array:
        return jnp.clip(self.linear(hidden), -self.max_delta, self.max_delta)


def adapted_observation(obs, hidden, adapter: SemanticResidualAdapter, noise_prob):
    """Apply residual only in noisy conditions; p=0 is exact GRU-MAP parity."""
    residual = adapter(hidden)
    residual = jnp.where(jnp.asarray(noise_prob) == 0.0,
                         jnp.zeros_like(residual), residual)
    return eqx.tree_at(lambda o: o.embedding, obs, obs.embedding + residual)


class DistilledTrackerWrapper(LearnedTrackerWrapper):
    """Uses LearnedTrackerWrapper for ALL dynamics, scoring and hidden-state updates.

    Only its output observation's semantic embedding is changed. The true state
    is never accessed by the residual adapter at deployment.
    """
    adapter: SemanticResidualAdapter

    def __init__(self, env, estimator, adapter: SemanticResidualAdapter,
                 noise_prob=0.0, overwrite_finite=False):
        super().__init__(env, estimator=estimator, noise_prob=noise_prob,
                         overwrite_finite=overwrite_finite)
        self.adapter = adapter

    @eqx.filter_jit
    def reset(self, key, state, params, options=None):
        next_state, observation = super().reset(key, state, params, options)
        observation = adapted_observation(observation, next_state.estimator_hidden,
                                          self.adapter, self.noise_prob)
        return next_state, observation

    @eqx.filter_jit
    def step(self, key, state, action, params):
        transition = super().step(key, state, action, params)
        next_observation = adapted_observation(
            transition.observation, transition.state.estimator_hidden,
            self.adapter, self.noise_prob,
        )
        terminal_observation = adapted_observation(
            transition.terminal_observation, transition.state.estimator_hidden,
            self.adapter, self.noise_prob,
        )
        return eqx.tree_at(
            lambda t: (t.observation, t.terminal_observation),
            transition, (next_observation, terminal_observation),
        )
