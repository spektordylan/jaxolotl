"""Structured Bayesian/neuro-symbolic task-state tracking (epsilon-free evaluation).

The policy sees the embedding of argmax of a state belief propagated through
exact LDBA transitions. Observation evidence is either empirical-prior Bayes or
a formula-agnostic proposition GRU. Ground-truth states remain scoring-only.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import corrupt_assignment
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAObservation
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.ltl.automata.jax_semantic_ldba import JaxSemanticLDBA
from experiments.belief_semltl.proposition_estimator import PropositionEstimator
from experiments.belief_semltl.structured_filter_math import assignment_posterior, propagate_belief


class StructuredFilterState[TObsFeatures: NamedTuple](WrapperState):
    """Wrapper state tracking true, hard, and structured estimated automaton states."""

    ldba: JaxSemanticLDBA
    ldba_state: jax.Array            # Ground-truth state (rewards, termination, metrics)
    perceived_ldba_state: jax.Array  # Hard perceived state
    estimated_ldba_state: jax.Array  # Learned estimator perceived state (fed to policy)
    estimator_hidden: jax.Array      # Proposition GRU hidden state, if present
    belief: jax.Array                # Probability vector over valid LDBA states
    prev_action: jax.Array           # Last physical action (0..4)

    obs: EnvObservation[TObsFeatures]
    propositions: jax.Array
    perceived_propositions: jax.Array
    has_disagreed: jax.Array


class StructuredFilterWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """Condition a frozen SemLTL actor on a propagated LDBA-state belief."""

    estimator: PropositionEstimator | None
    prior: jax.Array
    noise_prob: jax.Array
    overwrite_finite: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
        estimator: PropositionEstimator | None,
        prior: jax.Array,
        noise_prob: float | jax.Array = 0.0,
        overwrite_finite: bool = False,
    ):
        super().__init__(env)
        self.estimator = estimator
        self.prior = jnp.asarray(prior, dtype=jnp.float32)
        self.noise_prob = jnp.array(noise_prob, dtype=jnp.float32)
        self.overwrite_finite = overwrite_finite

    def _epsilon_enabled(
        self, ldba: JaxSemanticLDBA, state: jax.Array, assignment: jax.Array
    ) -> jax.Array:
        """Mask epsilon actions using the same rule as SemanticLDBAWrapper.

        The mask is a vector of length max_eps_transitions, not a scalar.
        An epsilon transition is allowed only when its successor exists and
        the subsequent regular transition would not enter a sink.
        """
        epsilon_states = ldba.get_next_epsilon_states(state)
        exists = epsilon_states >= 0
        # JAX clamps negative indices; mask them back out with `exists`.
        safe_states = jnp.maximum(epsilon_states, 0)
        successor_after_assignment = ldba.transitions[safe_states, assignment]
        safe = ~ldba.is_sink_state(successor_after_assignment)
        return exists & safe

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: StructuredFilterState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[StructuredFilterState, SemanticLDBAObservation[TObsFeatures]]:
        assert options is not None, "CurriculumResetOptions must be provided to reset."
        re_state, obs = super().reset(key, state, params, options)
        propositions = self._env.compute_propositions(re_state, params)
        true_assignment = self._env.map_assignment_to_index(propositions)

        noise_key = jax.random.fold_in(key, 0)
        perceived_assignment = corrupt_assignment(
            noise_key,
            true_assignment,
            len(self._env.assignments()),
            self.noise_prob,
        )
        perceived_propositions = self._env.assignments_array[perceived_assignment]

        initial_state = options.task.initial_state
        init_hidden = (self.estimator.initial_hidden() if self.estimator is not None
                       else jnp.zeros((1,), dtype=jnp.float32))
        init_prev_action = jnp.array(4, dtype=jnp.int32)  # 4 = start token

        # The training dataset starts at the FIRST PHYSICAL TRANSITION; it does
        # not contain a reset-time detector sample. Keep the RNN unadvanced at
        # reset or deployment observes a different history from training.
        h_0 = init_hidden
        estimated_state = initial_state

        wrapper_state = StructuredFilterState(
            state=re_state,
            ldba=options.task,
            ldba_state=initial_state,
            perceived_ldba_state=initial_state,
            estimated_ldba_state=estimated_state,
            estimator_hidden=h_0,
            belief=jax.nn.one_hot(initial_state, options.task.transitions.shape[0]),
            prev_action=init_prev_action,
            obs=obs,
            propositions=propositions,
            perceived_propositions=perceived_propositions,
            has_disagreed=jnp.array(False, dtype=jnp.bool_),
        )

        embedding = wrapper_state.ldba.get_embedding(wrapper_state.estimated_ldba_state)
        eps_embeddings = wrapper_state.ldba.get_epsilon_embeddings(wrapper_state.estimated_ldba_state)
        epsilon_enabled = self._epsilon_enabled(
            wrapper_state.ldba, wrapper_state.estimated_ldba_state, perceived_assignment
        )
        return wrapper_state, SemanticLDBAObservation.from_obs(
            obs, embedding, eps_embeddings, epsilon_enabled
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: StructuredFilterState,
        action: tuple[jax.Array, jax.Array] | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[StructuredFilterState, TObsFeatures]:
        # SemLTL uses epsilon_action == max_eps as its *physical-action* token.
        # In the Phase 2 study all formulas are epsilon-free, but preserve the
        # normal action convention so the masked actor can choose correctly.
        max_eps = state.ldba.epsilon_transitions.shape[1]
        if isinstance(action, tuple):
            phys_action, eps_action = action
        else:
            phys_action = action
            eps_action = jnp.asarray(max_eps, dtype=jnp.int32)
        execute_epsilon = eps_action != max_eps
        safe_eps_action = jnp.minimum(eps_action, max_eps - 1)

        step_key, noise_key = jax.random.split(key)
        physical_transition = super().step(step_key, state, phys_action, params)

        # An epsilon action advances only the automaton, not the environment.
        # Use a JAX conditional to preserve matching pytrees under jit/vmap.
        transition = jax.lax.cond(
            execute_epsilon,
            lambda: EnvTransition(
                state=state.state,
                observation=state.obs,
                reward=jnp.zeros_like(physical_transition.reward),
                terminated=jnp.zeros_like(physical_transition.terminated),
                truncated=jnp.zeros_like(physical_transition.truncated),
                terminal_observation=state.obs,
                propositions=state.propositions,
                info=physical_transition.info,
            ),
            lambda: physical_transition,
        )

        propositions = transition.propositions
        true_assignment = self._env.map_assignment_to_index(propositions)
        perceived_assignment = corrupt_assignment(
            noise_key,
            true_assignment,
            len(self._env.assignments()),
            self.noise_prob,
        )
        perceived_propositions = self._env.assignments_array[perceived_assignment]

        # The true monitor determines rewards and termination; the hard and
        # estimated monitors are never used as substitutes for ground truth.
        true_regular_next, regular_accepting = state.ldba.get_next_state(
            state.ldba_state, true_assignment
        )
        true_epsilon_next = state.ldba.get_next_epsilon_state(
            state.ldba_state, safe_eps_action
        )
        next_true_state = jnp.where(
            execute_epsilon, true_epsilon_next, true_regular_next
        )
        is_accepting = jnp.where(execute_epsilon, False, regular_accepting)

        perceived_regular_next, _ = state.ldba.get_next_state(
            state.perceived_ldba_state, perceived_assignment
        )
        perceived_epsilon_next = state.ldba.get_next_epsilon_state(
            state.perceived_ldba_state, safe_eps_action
        )
        next_perceived_state = jnp.where(
            execute_epsilon, perceived_epsilon_next, perceived_regular_next
        )

        # Evidence must use only the corrupted detector and previous action.
        # Never feed true_assignment or true automaton state to inference.
        if self.estimator is None:
            next_hidden = state.estimator_hidden
            assignment_probs = assignment_posterior(
                perceived_assignment, self.noise_prob, self.prior
            )
        else:
            next_hidden, logits = self.estimator.step(
                state.estimator_hidden, perceived_assignment, phys_action
            )
            assignment_probs = jax.nn.softmax(logits)

        regular_belief = propagate_belief(
            state.belief, state.ldba.transitions,
            assignment_probs, state.ldba.num_states,
        )
        # Explicit epsilon progression of a *state belief*, not of a letter.
        eps_successors = state.ldba.epsilon_transitions[:, safe_eps_action]
        eps_successors = jnp.where(
            eps_successors >= 0, eps_successors,
            jnp.arange(state.ldba.transitions.shape[0]),
        )
        eps_belief = jnp.einsum(
            'q,qr->r', state.belief,
            jax.nn.one_hot(eps_successors, state.ldba.transitions.shape[0]),
        )
        next_belief = jnp.where(execute_epsilon, eps_belief, regular_belief)
        # Preserve bit-for-bit p=0 reference behavior (also for stochastic actor).
        next_belief = jnp.where(
            self.noise_prob == 0.0,
            jax.nn.one_hot(next_perceived_state, state.ldba.transitions.shape[0]),
            next_belief,
        )
        next_estimated_state = jnp.argmax(next_belief).astype(jnp.int32)

        is_sink = state.ldba.is_sink_state(next_true_state)
        reward = jnp.where(is_accepting, 1.0, jnp.where(is_sink, -1.0, 0.0))
        finite = jnp.logical_or(state.ldba.finite, self.overwrite_finite)
        semantic_termination = jnp.logical_or(
            is_sink, jnp.logical_and(is_accepting, finite)
        )

        has_disagreed = state.has_disagreed | (
            next_estimated_state != next_true_state
        )
        next_wrapper_state = StructuredFilterState(
            state=transition.state,
            ldba=state.ldba,
            ldba_state=next_true_state,
            perceived_ldba_state=next_perceived_state,
            estimated_ldba_state=next_estimated_state,
            estimator_hidden=next_hidden,
            belief=next_belief,
            prev_action=phys_action,
            obs=transition.observation,
            propositions=propositions,
            perceived_propositions=perceived_propositions,
            has_disagreed=has_disagreed,
        )

        embedding = state.ldba.get_embedding(next_estimated_state)
        eps_embeddings = state.ldba.get_epsilon_embeddings(next_estimated_state)
        epsilon_mask = self._epsilon_enabled(
            state.ldba, next_estimated_state, perceived_assignment
        )
        next_obs = SemanticLDBAObservation.from_obs(
            transition.observation, embedding, eps_embeddings, epsilon_mask
        )
        terminal_obs = SemanticLDBAObservation.from_obs(
            transition.terminal_observation, embedding, eps_embeddings, epsilon_mask
        )
        info = transition.info | {
            "is_sink": is_sink,
            "is_accepting": is_accepting,
            "disagreement": has_disagreed,
            "true_ldba_state": next_true_state,
            "estimated_ldba_state": next_estimated_state,
        }
        return EnvTransition(
            state=next_wrapper_state,
            observation=next_obs,
            reward=reward,
            terminated=transition.terminated | semantic_termination,
            truncated=transition.truncated,
            terminal_observation=terminal_obs,
            propositions=propositions,
            info=info,
        )
