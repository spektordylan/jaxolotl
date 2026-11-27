"""Exact finite-state automaton propagation with approximate proposition evidence.

The assignment posterior is a *marginal* P(a_t | history), not an exact
P(a_t | q_(t-1), history). Consequently the belief update is approximate.
Only epsilon-free tasks are evaluated.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def assignment_posterior(observed: jax.Array, noise_prob: jax.Array,
                         prior: jax.Array) -> jax.Array:
    """Bayes' rule for the known symmetric categorical corruption channel.

    Channel: P(y=a | a)=1-p; P(y!=a | a)=p/(A-1).
    For p=0 the result is exactly one-hot regardless of the prior.
    """
    prior = jnp.asarray(prior, dtype=jnp.float32)
    p = jnp.asarray(noise_prob, dtype=jnp.float32)
    n = prior.shape[0]
    likelihood = jnp.where(jnp.arange(n) == observed, 1.0 - p, p / (n - 1))
    weights = prior * likelihood
    return weights / jnp.maximum(jnp.sum(weights), 1e-30)


def propagate_belief(belief: jax.Array, transitions: jax.Array,
                     assignment_probs: jax.Array, num_states: jax.Array) -> jax.Array:
    """Push state mass through LDBA delta(q,a), including sink states.

    transitions is (max_states, assignments), with -1 in padded entries.
    Every valid (q,a) transition must be defined. No truncation of sink mass.
    """
    qmax = transitions.shape[0]
    active = jnp.arange(qmax) < num_states
    belief = jnp.where(active, belief, 0.0)
    next_onehot = jax.nn.one_hot(transitions, qmax, dtype=jnp.float32)
    output = jnp.einsum('q,a,qar->r', belief, assignment_probs, next_onehot)
    output = jnp.where(active, output, 0.0)
    return output / jnp.maximum(jnp.sum(output), 1e-30)
