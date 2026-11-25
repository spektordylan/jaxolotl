"""Privileged task-state oracle for the frozen SemLTL policy (diagnostic only).

Only the *policy-facing semantic automaton features* are replaced with those
of the true LDBA state. The base NoisySemanticLDBAWrapper still evolves the
physical environment, true and noisy monitors, rewards, and termination using
its existing code and PRNG schedule. The agent must not use this at deployment:
true propositions and the true LDBA state are privileged information.

The intended experiment excludes epsilon-containing LTL formulas. We still
rebuild epsilon embeddings/masks from the true monitor for completeness.
"""

import equinox as eqx
import jax.numpy as jnp

from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
    NoisySemanticLDBAWrapperState,
)
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAObservation
from jaxolotl.environments.environment import EnvTransition


class OracleSemanticLDBAWrapper(NoisySemanticLDBAWrapper):
    """Noisy underlying world, privileged true-state semantic policy input.

    At p>0 a *latent* noisy-state tracker still runs for experimental parity.
    Its disagreement is not an oracle-monitor error and is not presented as
    such: evaluator's ``disagreement`` field is set to false for the oracle.
    """

    def _oracle_features(self, state: NoisySemanticLDBAWrapperState):
        q = state.ldba_state
        true_assignment = self._env.map_assignment_to_index(state.propositions)
        return (
            state.ldba.get_embedding(q),
            state.ldba.get_epsilon_embeddings(q),
            self._epsilon_enabled(state.ldba, q, true_assignment),
        )

    def _oracle_semantic_observation(self, state: NoisySemanticLDBAWrapperState):
        embedding, epsilon_embeddings, epsilon_mask = self._oracle_features(state)
        return SemanticLDBAObservation.from_obs(
            state.obs, embedding, epsilon_embeddings, epsilon_mask
        )

    @staticmethod
    def _oracle_terminal_observation(existing, embedding, eps_embeddings, eps_mask):
        """Keep the parent's raw terminal observation, replace semantic fields.

        ``SemanticLDBAObservation`` is a NamedTuple. Its epsilon embedding
        field can vary across upstream versions; access it via its static field
        names to avoid replacing/losing the raw terminal features.
        """
        updates = {"embedding": embedding, "epsilon_mask": eps_mask}
        fields = getattr(existing, "_fields", ())
        for name in ("epsilon_embeddings", "eps_embeddings"):
            if name in fields or (not fields and hasattr(existing, name)):
                updates[name] = eps_embeddings
        if hasattr(existing, "_replace"):
            return existing._replace(**updates)
        # Equinox dataclass-style observation fallback. The field structure is
        # static at tracing time; no traced value controls the Python branch.
        names = tuple(updates)
        return eqx.tree_at(
            lambda obs: tuple(getattr(obs, name) for name in names),
            existing,
            tuple(updates[name] for name in names),
        )

    @eqx.filter_jit
    def reset(self, key, state, params, options=None):
        next_state, _ = super().reset(key, state, params, options)
        return next_state, self._oracle_semantic_observation(next_state)

    @eqx.filter_jit
    def step(self, key, state, action, params):
        transition = super().step(key, state, action, params)
        embedding, eps_embeddings, eps_mask = self._oracle_features(transition.state)
        oracle_obs = self._oracle_semantic_observation(transition.state)
        oracle_terminal_obs = self._oracle_terminal_observation(
            transition.terminal_observation, embedding, eps_embeddings, eps_mask
        )
        # This is *not* a failure of the privileged oracle state. Preserve the
        # latent noisy disagreement separately for debugging, but report zero
        # oracle-monitor disagreement to standard evaluators.
        info = transition.info | {
            "latent_noisy_disagreement": transition.info["disagreement"],
            "disagreement": jnp.zeros((), dtype=jnp.bool_),
            "has_disagreed": jnp.zeros((), dtype=jnp.bool_),
        }
        return EnvTransition(
            state=transition.state,
            observation=oracle_obs,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            terminal_observation=oracle_terminal_obs,
            propositions=transition.propositions,
            info=info,
        )
