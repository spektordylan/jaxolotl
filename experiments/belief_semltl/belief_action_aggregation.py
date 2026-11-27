"""Frozen SemLTL policy-probability aggregation with a trained GRU belief.

Scope: 9 epsilon-free LetterWorld formulas; discrete physical actions only.
The frozen actor still gets its ordinary spatial observation. The GRU receives
only noisy detector history/actions/hard monitor state, as in Phase 2.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from experiments.belief_semltl.belief_aggregation_math import (
    mix_action_probabilities,
    valid_state_posterior,
)


NUM_LETTERWORLD_ACTIONS = 4


def aggregate_physical_action(
    agent,
    observation,
    tracker_state,
    estimator,
    key: jax.Array,
    *,
    step_index: jax.Array,
    top_k: int,
    alpha: float,
    deterministic: bool,
    zero_noise: bool,
):
    """Select one batch of physical actions from exact mixed probabilities.

    No changes are made to the frozen actor, GRU, or precomputed LDBA labels.
    An epsilon-free task is required: epsilon indices are always the physical
    action sentinel, and no epsilon-choice probabilities are mixed.
    """
    # At p=0, preserve the exact MAP-policy sampling path, not just its
    # action probabilities. jax.random.categorical and the actor's sample()
    # can generate different actions from the same key even when the mixture
    # posterior has all its mass on the known automaton state.
    if alpha == 0.0 or zero_noise:
        pi = agent.get_action(observation)
        return pi.mode() if deterministic else pi.sample(seed=key)

    ldba = tracker_state.ldba
    n = observation.embedding.shape[0]
    n_states = estimator.max_states
    if top_k < 1 or top_k > n_states:
        raise ValueError(f"top_k must be in [1,{n_states}]")

    # The GRU's linear head applied to its current recurrent hidden state is
    # precisely the online logits. Avoid recomputing/replaying history.
    logits = jax.vmap(estimator.head)(tracker_state.estimator_hidden)
    posterior = valid_state_posterior(
        logits, ldba.num_states, tracker_state.estimated_ldba_state,
        initial_step=(step_index == 0), zero_noise=zero_noise,
    )
    weights, candidates = jax.lax.top_k(posterior, top_k)  # (N,K)

    # For the chosen formula the LDBA has padded state embeddings (N,S,D).
    # Only *individual valid embeddings* enter the frozen actor. In a padded
    # candidate slot with exactly zero weight, the embedding is immaterial.
    embeddings = jnp.take_along_axis(
        ldba.embeddings, candidates[..., None], axis=1
    )  # (N,K,D)
    num_eps = ldba.epsilon_transitions.shape[-1]
    physical_epsilon_token = jnp.full((n,), num_eps, dtype=jnp.int32)

    def conditioned_action_probs(embedding_batch):
        conditioned_obs = eqx.tree_at(
            lambda obs: obs.embedding, observation, embedding_batch
        )
        dist = agent.get_action(conditioned_obs)

        def prob_of_action(a):
            physical_action = jnp.full((n,), a, dtype=jnp.int32)
            # The actor's joint action is (physical_action, epsilon_action).
            # The epsilon sentinel denotes a normal environment step.
            return jnp.exp(dist.log_prob((physical_action, physical_epsilon_token)))

        probs = jax.vmap(prob_of_action)(
            jnp.arange(NUM_LETTERWORLD_ACTIONS, dtype=jnp.int32)
        ).T  # (N,4)
        return probs / jnp.maximum(probs.sum(axis=-1, keepdims=True), 1e-30)

    per_state_probs = jax.vmap(conditioned_action_probs)(
        jnp.swapaxes(embeddings, 0, 1)
    )  # (K,N,4)
    per_state_probs = jnp.swapaxes(per_state_probs, 0, 1)  # (N,K,4)
    mixed = mix_action_probabilities(per_state_probs, weights, alpha=alpha)

    if deterministic:
        physical_action = jnp.argmax(mixed, axis=-1).astype(jnp.int32)
    else:
        physical_action = jax.random.categorical(
            key, jnp.log(jnp.maximum(mixed, 1e-30)), axis=-1
        ).astype(jnp.int32)
    return physical_action, physical_epsilon_token
