"""Belief-tracking environment adapter for a training-free SemLTL policy mixture.

Wraps the *existing* NoisySemanticLDBAWrapper without reimplementing physical
transitions, reward, termination, or the hard perceived automaton. It additionally
propagates a passive approximate belief using only corrupted propositions.

The new agent (policy_mixture_agent.py) samples the policy branch: either the
hard semantic embedding, or a categorical state drawn from this belief. It then
samples from the ORIGINAL pretrained policy conditioned on that valid embedding.
For stochastic policies this is an exact action-distribution mixture in law.

Scope: EPSILON-FREE formulas only. The wrapper does not define a belief update
for epsilon actions; the experiment runner enforces this at compile time.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.sem_ltl.wrappers.belief_semantic_ldba_wrapper import (
    compute_belief_next,
    compute_posterior_a,
)
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAObservation
from jaxolotl.environments.environment import EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState


class PolicyMixtureObservation(eqx.Module):
    """Neural observation plus nonprivileged state-belief side channel."""
    base: SemanticLDBAObservation
    belief: jax.Array
    state_embeddings: jax.Array


class PolicyMixtureState(WrapperState):
    """WrapperState.state is the intact existing hard noisy-wrapper state."""
    belief: jax.Array


class PolicyMixtureWrapper(EnvWrapper):
    """Extra belief tracker *outside* the tested hard/true environment wrapper."""

    def __init__(self, env: Any, noise_prob: float, overwrite_finite: bool = True):
        super().__init__(
            NoisySemanticLDBAWrapper(
                env, noise_prob=noise_prob, overwrite_finite=overwrite_finite
            )
        )

    @staticmethod
    def _observation(inner_obs, belief, ldba):
        return PolicyMixtureObservation(
            base=inner_obs,
            belief=belief,
            state_embeddings=ldba.embeddings,
        )

    @eqx.filter_jit
    def reset(self, key, state, params, options=None):
        # EnvWrapper.reset unwraps one WrapperState level internally. Pass
        # OUR outer state, not state.state, to avoid dropping the noisy monitor.
        inner_state, inner_obs = super().reset(key, state, params, options)
        num_states = inner_state.ldba.transitions.shape[0]
        belief = jax.nn.one_hot(
            inner_state.ldba.initial_state, num_states, dtype=jnp.float32
        )
        outer_state = PolicyMixtureState(state=inner_state, belief=belief)
        return outer_state, self._observation(inner_obs, belief, inner_state.ldba)

    @eqx.filter_jit
    def step(self, key, state: PolicyMixtureState, action, params) -> EnvTransition:
        # EnvWrapper.step already dispatches self._env.step(key, state.state, ...).
        # Passing state.state here would unwrap twice and send the physical
        # LetterWorld EnvState to NoisySemanticLDBAWrapper.step().
        inner = super().step(key, state, action, params)
        ldba = inner.state.ldba
        num_assignments = ldba.transitions.shape[1]
        # CRITICAL: the filter sees only perceived (corrupted) propositions.
        # The inner noisy wrapper alone maintains ground-truth labels and rewards.
        observed_assignment = self._env._env.map_assignment_to_index(
            inner.state.perceived_propositions
        )
        posterior = compute_posterior_a(
            observed_assignment, num_assignments, self._env.noise_prob
        )
        updated = compute_belief_next(state.belief, ldba.transitions, posterior)
        # There are no epsilon actions on the nine selected tasks. This guard
        # keeps semantics well-defined if a no-op epsilon is nevertheless sent.
        physical_step = action[1] == ldba.epsilon_transitions.shape[1]
        updated = jnp.where(physical_step, updated, state.belief)
        outer_state = PolicyMixtureState(state=inner.state, belief=updated)
        return EnvTransition(
            state=outer_state,
            observation=self._observation(inner.observation, updated, ldba),
            reward=inner.reward,
            terminated=inner.terminated,
            truncated=inner.truncated,
            terminal_observation=self._observation(
                inner.terminal_observation, updated, ldba
            ),
            propositions=inner.propositions,
            info=inner.info,
        )
