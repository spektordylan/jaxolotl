"""SemLTL LDBA wrapper handling noisy proposition observations while maintaining
strict separation between ground-truth and perceived automaton states.
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAObservation
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.ltl.automata.jax_semantic_ldba import JaxSemanticLDBA


def corrupt_assignment(
    key: jax.Array,
    true_assignment: jax.Array,
    num_assignments: int,
    noise_prob: float | jax.Array,
) -> jax.Array:
    """Independent categorical observation-corruption model.

    With probability 1 - p: report the true assignment.
    With probability p: replace with a different valid assignment selected uniformly
    from the remaining (num_assignments - 1) valid assignments.

    Args:
        key: JAX PRNG key.
        true_assignment: Ground-truth assignment index (scalar int32).
        num_assignments: Total number of valid assignments in the environment.
        noise_prob: Corruption probability p in [0, 1].

    Returns:
        Corrupted or exact assignment index (scalar int32).
    """
    key_bernoulli, key_choice = jax.random.split(key)
    # Sample uniformly from {0, ..., num_assignments - 2}
    offset = jax.random.randint(
        key_choice,
        shape=(),
        minval=0,
        maxval=num_assignments - 1,
    )
    # Shift by offset + 1 modulo num_assignments to exclude true_assignment
    alt_assignment = (true_assignment + 1 + offset) % num_assignments
    corrupt = jax.random.bernoulli(key_bernoulli, p=noise_prob)
    return jnp.where(corrupt, alt_assignment, true_assignment)


def _resolve_true_epsilon_successor(
    current_true_state: jax.Array,
    regular_successor: jax.Array,
    epsilon_successor: jax.Array,
    execute_epsilon: jax.Array,
) -> jax.Array:
    """Use an epsilon successor only if present; otherwise epsilon is a no-op.

    In particular, an unavailable epsilon action must NOT consume the current
    proposition by falling through to ``regular_successor``.
    The automaton uses -1 to mark an unavailable epsilon transition.
    """
    epsilon_result = jnp.where(
        epsilon_successor != -1, epsilon_successor, current_true_state
    )
    return jnp.where(execute_epsilon, epsilon_result, regular_successor)


class NoisySemanticLDBAWrapperState[TObsFeatures: NamedTuple](WrapperState):
    """Wrapper state tracking both true and perceived automaton states."""

    ldba: JaxSemanticLDBA  # Current LDBA
    ldba_state: jax.Array  # Ground-truth LDBA state (evaluator, reward, termination)
    perceived_ldba_state: jax.Array  # Perceived LDBA state (policy embedding & mask)

    # Epsilon-related state
    obs: EnvObservation[TObsFeatures]  # Last observation
    propositions: jax.Array  # Last true propositions
    perceived_propositions: jax.Array  # Last perceived propositions
    has_disagreed: jax.Array  # Boolean scalar: whether perceived != true at any point


class NoisySemanticLDBAWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper that simulates noisy proposition observations for SemLTL policies.

    The wrapper maintains two concurrent automaton states:
      1. `ldba_state` (ground truth): updated with the true environment proposition,
         used strictly for computing reward, task termination, sink-state detection,
         and evaluation metrics.
      2. `perceived_ldba_state` (agent belief): updated with corrupted observations,
         used to generate semantic embeddings and epsilon-action masks for the policy.
    """

    noise_prob: jax.Array
    overwrite_finite: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
        noise_prob: float | jax.Array = 0.0,
        overwrite_finite: bool = False,
    ):
        super().__init__(env)
        self.noise_prob = jnp.array(noise_prob, dtype=jnp.float32)
        self.overwrite_finite = overwrite_finite

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: NoisySemanticLDBAWrapperState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[NoisySemanticLDBAWrapperState, SemanticLDBAObservation[TObsFeatures]]:
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
        state = NoisySemanticLDBAWrapperState(
            state=re_state,
            ldba=options.task,
            ldba_state=initial_state,
            perceived_ldba_state=initial_state,
            obs=obs,
            propositions=propositions,
            perceived_propositions=perceived_propositions,
            has_disagreed=jnp.array(False, dtype=jnp.bool_),
        )
        embedding = state.ldba.get_embedding(state.perceived_ldba_state)
        eps_embeddings = state.ldba.get_epsilon_embeddings(state.perceived_ldba_state)
        epsilon_enabled = self._epsilon_enabled(
            state.ldba, state.perceived_ldba_state, perceived_assignment
        )
        return state, SemanticLDBAObservation.from_obs(
            obs, embedding, eps_embeddings, epsilon_enabled
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: NoisySemanticLDBAWrapperState,
        action: tuple[jax.Array, jax.Array],  # (env_action, epsilon (int32))
        params: TEnvParams,
    ) -> EnvTransition[NoisySemanticLDBAWrapperState, TObsFeatures]:
        env_action, epsilon_action = action
        max_eps_transitions = state.ldba.epsilon_transitions.shape[1]
        execute_eps = epsilon_action != max_eps_transitions

        env_transition = super().step(key, state, env_action, params)
        env_transition.info["is_sink"] = jnp.zeros(
            (), dtype=jnp.bool_
        )  # placeholder, updated below
        transition: EnvTransition = jax.lax.cond(
            execute_eps,
            lambda: self._epsilon_step(state, env_transition.info),
            lambda: env_transition,
        )

        # 1. Ground-truth and perceived proposition assignments
        true_assignment = self._env.map_assignment_to_index(transition.propositions)

        noise_key = jax.random.fold_in(key, 1)
        perceived_assignment_step = corrupt_assignment(
            noise_key,
            true_assignment,
            len(self._env.assignments()),
            self.noise_prob,
        )
        # On epsilon step, environment propositions do not advance
        perceived_assignment = jnp.where(
            execute_eps,
            self._env.map_assignment_to_index(state.perceived_propositions),
            perceived_assignment_step,
        )
        perceived_propositions = self._env.assignments_array[perceived_assignment]

        # 2. Ground-truth LDBA transition
        next_true_ldba_state, true_is_accepting = state.ldba.get_next_state(
            state.ldba_state, true_assignment
        )
        next_true_epsilon_state = state.ldba.get_next_epsilon_state(
            state.ldba_state, epsilon_action
        )
        next_true_ldba_state = _resolve_true_epsilon_successor(
            current_true_state=state.ldba_state,
            regular_successor=next_true_ldba_state,
            epsilon_successor=next_true_epsilon_state,
            execute_epsilon=execute_eps,
        )
        true_is_accepting = jnp.where(execute_eps, False, true_is_accepting)
        true_is_sink = state.ldba.is_sink_state(next_true_ldba_state)

        # 3. Perceived LDBA transition
        next_perceived_ldba_state, perceived_is_accepting = state.ldba.get_next_state(
            state.perceived_ldba_state, perceived_assignment
        )
        next_perceived_epsilon_state = state.ldba.get_next_epsilon_state(
            state.perceived_ldba_state, epsilon_action
        )
        next_perceived_ldba_state = jnp.where(
            execute_eps, next_perceived_epsilon_state, next_perceived_ldba_state
        )
        perceived_is_accepting = jnp.where(execute_eps, False, perceived_is_accepting)
        perceived_is_sink = state.ldba.is_sink_state(next_perceived_ldba_state)

        # 4. True task satisfaction and termination logic
        reward = jax.lax.cond(
            true_is_accepting,
            lambda: 1.0,
            lambda: jax.lax.cond(true_is_sink, lambda: -1.0, lambda: 0.0),
        )
        finite = jnp.logical_or(state.ldba.finite, self.overwrite_finite)
        terminated = jnp.logical_and(true_is_accepting, finite)
        terminated = jnp.logical_or(terminated, true_is_sink)

        # Disagreement tracking
        disagreed_now = next_true_ldba_state != next_perceived_ldba_state
        has_disagreed = jnp.logical_or(state.has_disagreed, disagreed_now)

        info = transition.info | {
            "is_sink": true_is_sink,
            "true_is_accepting": true_is_accepting,
            "perceived_is_accepting": perceived_is_accepting,
            "perceived_is_sink": perceived_is_sink,
            "disagreement": disagreed_now,
            "has_disagreed": has_disagreed,
        }

        new_state = NoisySemanticLDBAWrapperState(
            state=transition.state,
            ldba=state.ldba,
            ldba_state=next_true_ldba_state,
            perceived_ldba_state=next_perceived_ldba_state,
            obs=transition.observation,
            propositions=transition.propositions,
            perceived_propositions=perceived_propositions,
            has_disagreed=has_disagreed,
        )

        # 5. Policy observations (semantic embeddings & masks) from perceived state
        embedding = new_state.ldba.get_embedding(new_state.perceived_ldba_state)
        eps_embeddings = new_state.ldba.get_epsilon_embeddings(
            new_state.perceived_ldba_state
        )
        epsilon_enabled = self._epsilon_enabled(
            new_state.ldba, new_state.perceived_ldba_state, perceived_assignment
        )
        next_obs = SemanticLDBAObservation.from_obs(
            transition.observation, embedding, eps_embeddings, epsilon_enabled
        )

        return EnvTransition(
            state=new_state,
            observation=next_obs,
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=SemanticLDBAObservation.from_obs(
                transition.terminal_observation,
                embedding,
                eps_embeddings,
                epsilon_enabled,
            ),
            propositions=transition.propositions,
            info=info,
        )

    def _epsilon_step(
        self, state: NoisySemanticLDBAWrapperState, info: dict
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        """Transition corresponding to an epsilon action."""
        return EnvTransition(
            state=state.state,
            observation=state.obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=jnp.bool_),
            truncated=jnp.zeros((), dtype=jnp.bool_),
            terminal_observation=state.obs,
            propositions=state.propositions,
            info=info,
        )

    def _epsilon_enabled(
        self, ldba: JaxSemanticLDBA, ldba_state: jax.Array, assignment_index: jax.Array
    ) -> jax.Array:
        """Returns a boolean array indicating which epsilon actions can be taken. An
        epsilon action can only be taken if doing so would not immediately lead to a
        sink state given the current assignment.

        Returns:
            epsilon_enabled: (max_eps_transitions,), bool
        """
        next_epsilon_states = ldba.get_next_epsilon_states(ldba_state)
        enabled = next_epsilon_states != -1

        # State after taking epsilon then the env transition
        next_states = jnp.where(
            enabled, ldba.transitions[next_epsilon_states, assignment_index], ldba_state
        )  # (max_eps_transitions,)
        is_valid = jnp.logical_not(ldba.is_sink_state(next_states))

        return jnp.logical_and(enabled, is_valid)
