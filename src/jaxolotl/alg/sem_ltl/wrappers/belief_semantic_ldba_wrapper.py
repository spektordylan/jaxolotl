"""Belief-aware SemLTL LDBA wrapper.

Maintains a Bayesian belief b_t(q) = P(q_t = q | y_{1:t}) over automaton
states, where y_t is the noisy (corrupted) proposition assignment.

Two belief modes are supported:
- ``map``      : feed the embedding of argmax(belief) to the frozen policy.
- ``weighted`` : feed the probability-weighted average of embeddings.

The existing ``NoisySemanticLDBAWrapper`` (``hard`` mode) is imported and
used directly by the pilot script; this module does not re-implement it.

Design constraints
------------------
* Rewards, termination, sink detection, and violation flags use ``ldba_state``
  (ground truth) exactly as ``NoisySemanticLDBAWrapper`` does.
* The belief filter only reads the corrupted assignment ``y``, never the
  ground-truth assignment ``a``.
* At p=0 the belief stays a delta on the deterministic state, so the MAP
  embedding equals the standard wrapper's embedding exactly.
* Epsilon semantics are identical to ``NoisySemanticLDBAWrapper``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    _resolve_true_epsilon_successor,
    corrupt_assignment,
)
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAObservation
from jaxolotl.environments.environment import EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.ltl.automata.jax_semantic_ldba import JaxSemanticLDBA

# ---------------------------------------------------------------------------
# Belief update
# ---------------------------------------------------------------------------

def compute_belief_next(
    belief: jax.Array,
    transitions: jax.Array,
    posterior_a: jax.Array,
) -> jax.Array:
    """Single-step belief propagation (pure function, JIT-friendly).

    b_{t+1}(q') = Σ_q b_t(q) Σ_a P(a|y_t) · 1[δ(q,a) = q']

    Args:
        belief:      Current belief, shape (num_states,), sums to 1.
        transitions: LDBA transition table, shape (num_states, num_assignments),
                     dtype int32. Entry [q, a] is the next state index.
        posterior_a: Posterior P(a|y) over true assignments given the observed
                     corrupted assignment y. Shape (num_assignments,), sums to 1.

    Returns:
        belief_next: Updated and normalised belief, shape (num_states,).
    """
    num_states = transitions.shape[0]

    # weight[q, a] = b(q) * P(a|y)  shape (num_states, num_assignments)
    weight = belief[:, None] * posterior_a[None, :]

    # Scatter-add into next-state bins.
    # Flatten: each of the (num_states * num_assignments) pairs contributes
    # weight[q, a] to belief_next[transitions[q, a]].
    flat_targets = transitions.ravel()          # (num_states * num_assignments,)
    flat_weights = weight.ravel()               # (num_states * num_assignments,)
    belief_next = (
        jnp.zeros(num_states, dtype=jnp.float32)
        .at[flat_targets]
        .add(flat_weights)
    )

    # Normalise (numerical safety; should already sum to 1).
    total = belief_next.sum()
    belief_next = jnp.where(total > 0, belief_next / total, belief_next)
    return belief_next


def compute_posterior_a(
    observed_assignment: jax.Array,
    num_assignments: int,
    noise_prob: jax.Array,
) -> jax.Array:
    """P(a | y) under uniform prior and symmetric corruption model.

    Args:
        observed_assignment: The noisy assignment index y, scalar int32.
        num_assignments:     Total number of assignments (e.g. 13).
        noise_prob:          Corruption probability p ∈ [0, 1].

    Returns:
        posterior_a: Shape (num_assignments,), sums to 1.
    """
    # P(y|a) = (1-p) if a==y else p/(K-1) where K = num_assignments
    p = noise_prob.astype(jnp.float32)
    arange = jnp.arange(num_assignments, dtype=jnp.int32)
    likelihood = jnp.where(arange == observed_assignment, 1.0 - p, p / (num_assignments - 1))
    # Under uniform prior P(a) = 1/K, the posterior is proportional to P(y|a).
    # Renormalise for numerical safety.
    posterior = likelihood / likelihood.sum()
    return posterior


# ---------------------------------------------------------------------------
# Wrapper state
# ---------------------------------------------------------------------------

class BeliefSemanticLDBAWrapperState(WrapperState):
    """Wrapper state for belief-aware SemLTL.

    ``ldba_state`` is the ground-truth automaton state; ``belief`` is the
    normalised distribution over all automaton states maintained using
    only the corrupted observation.
    """

    ldba: JaxSemanticLDBA
    ldba_state: jax.Array          # ground-truth LDBA state (rewards / termination)
    belief: jax.Array              # shape (num_states,) – belief over LDBA states
    obs: EnvObservation            # last environment observation (before LDBA aug.)
    propositions: jax.Array        # ground-truth propositions
    perceived_propositions: jax.Array  # noisy propositions (for epsilon mask)
    has_disagreed: jax.Array       # bool: MAP state ≠ true state at any point


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class BeliefSemanticLDBAWrapper(EnvWrapper):
    """Bayesian belief tracker over LDBA states for frozen SemLTL policies.

    Parameters
    ----------
    env : Environment | EnvWrapper
        Underlying environment (should be the base LetterWorld, not yet
        wrapped with SemanticLDBAWrapper).
    noise_prob : float
        Proposition corruption probability p ∈ [0, 1].
    belief_mode : str
        ``"map"`` or ``"weighted"``.
    overwrite_finite : bool
        Whether to treat all tasks as finite (passed through for termination
        logic, same as the other wrappers).
    """

    noise_prob: jax.Array
    belief_mode: str
    overwrite_finite: bool

    def __init__(
        self,
        env: Any,
        noise_prob: float | jax.Array = 0.0,
        belief_mode: str = "map",
        overwrite_finite: bool = False,
    ):
        if belief_mode not in {"map", "weighted"}:
            raise ValueError(
                f"BeliefSemanticLDBAWrapper belief_mode must be 'map' or 'weighted', "
                f"got '{belief_mode}'. For hard mode use NoisySemanticLDBAWrapper."
            )
        super().__init__(env)
        self.noise_prob = jnp.array(noise_prob, dtype=jnp.float32)
        self.belief_mode = belief_mode
        self.overwrite_finite = overwrite_finite

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: BeliefSemanticLDBAWrapperState | None,
        params: Any,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[BeliefSemanticLDBAWrapperState, SemanticLDBAObservation]:
        assert options is not None, "CurriculumResetOptions must be provided."
        re_state, obs = super().reset(key, state, params, options)

        # Ground-truth propositions at reset.
        propositions = self._env.compute_propositions(re_state, params)
        true_assignment = self._env.map_assignment_to_index(propositions)

        # Corrupt for the perceived initial observation.
        noise_key = jax.random.fold_in(key, 0)
        perceived_assignment = corrupt_assignment(
            noise_key,
            true_assignment,
            len(self._env.assignments()),
            self.noise_prob,
        )
        perceived_propositions = self._env.assignments_array[perceived_assignment]

        # Belief: one-hot on the initial LDBA state.
        initial_state = options.task.initial_state
        num_states = options.task.transitions.shape[0]
        belief0 = jax.nn.one_hot(initial_state, num_states, dtype=jnp.float32)

        wrapper_state = BeliefSemanticLDBAWrapperState(
            state=re_state,
            ldba=options.task,
            ldba_state=initial_state,
            belief=belief0,
            obs=obs,
            propositions=propositions,
            perceived_propositions=perceived_propositions,
            has_disagreed=jnp.array(False, dtype=jnp.bool_),
        )

        # Build initial policy observation.
        embedding, eps_embeddings, epsilon_enabled = self._policy_obs(
            wrapper_state.ldba, belief0, perceived_assignment
        )
        return wrapper_state, SemanticLDBAObservation.from_obs(
            obs, embedding, eps_embeddings, epsilon_enabled
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: BeliefSemanticLDBAWrapperState,
        action: tuple[jax.Array, jax.Array],
        params: Any,
    ) -> EnvTransition:
        env_action, epsilon_action = action
        max_eps_transitions = state.ldba.epsilon_transitions.shape[1]
        execute_eps = epsilon_action != max_eps_transitions

        # Step the underlying environment (or no-op for epsilon).
        env_transition = super().step(key, state, env_action, params)
        env_transition.info["is_sink"] = jnp.zeros((), dtype=jnp.bool_)
        transition: EnvTransition = jax.lax.cond(
            execute_eps,
            lambda: self._epsilon_step(state, env_transition.info),
            lambda: env_transition,
        )

        # --- Ground-truth proposition and LDBA transition ---
        true_assignment = self._env.map_assignment_to_index(transition.propositions)

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

        # --- Reward and termination (ground-truth only) ---
        reward = jax.lax.cond(
            true_is_accepting,
            lambda: 1.0,
            lambda: jax.lax.cond(true_is_sink, lambda: -1.0, lambda: 0.0),
        )
        finite = jnp.logical_or(state.ldba.finite, self.overwrite_finite)
        terminated = jnp.logical_and(true_is_accepting, finite)
        terminated = jnp.logical_or(terminated, true_is_sink)

        # --- Corrupted observation (only y is observed, not a) ---
        noise_key = jax.random.fold_in(key, 1)
        perceived_assignment_step = corrupt_assignment(
            noise_key,
            true_assignment,
            len(self._env.assignments()),
            self.noise_prob,
        )
        # On epsilon step, don't advance the proposition.
        perceived_assignment = jnp.where(
            execute_eps,
            self._env.map_assignment_to_index(state.perceived_propositions),
            perceived_assignment_step,
        )
        perceived_propositions = self._env.assignments_array[perceived_assignment]

        # --- Belief update (uses only perceived_assignment, never true_assignment) ---
        num_assignments = state.ldba.transitions.shape[1]
        posterior_a = compute_posterior_a(
            perceived_assignment, num_assignments, self.noise_prob
        )
        # On epsilon steps, don't update belief (no new proposition observed).
        belief_next_prop = compute_belief_next(
            state.belief, state.ldba.transitions, posterior_a
        )
        belief_next = jnp.where(execute_eps, state.belief, belief_next_prop)

        # --- Disagreement: MAP state vs ground truth ---
        map_state = jnp.argmax(belief_next)
        disagree_now = map_state != next_true_ldba_state
        has_disagreed = jnp.logical_or(state.has_disagreed, disagree_now)

        new_state = BeliefSemanticLDBAWrapperState(
            state=transition.state,
            ldba=state.ldba,
            ldba_state=next_true_ldba_state,
            belief=belief_next,
            obs=transition.observation,
            propositions=transition.propositions,
            perceived_propositions=perceived_propositions,
            has_disagreed=has_disagreed,
        )

        # --- Policy observation (embedding from belief, mask from MAP state) ---
        embedding, eps_embeddings, epsilon_enabled = self._policy_obs(
            state.ldba, belief_next, perceived_assignment
        )
        next_obs = SemanticLDBAObservation.from_obs(
            transition.observation, embedding, eps_embeddings, epsilon_enabled
        )

        info = transition.info | {
            "is_sink": true_is_sink,
            "true_is_accepting": true_is_accepting,
            "disagreement": disagree_now,
            "has_disagreed": has_disagreed,
        }

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

    # ------------------------------------------------------------------
    # Policy observation helpers
    # ------------------------------------------------------------------

    def _policy_obs(
        self,
        ldba: JaxSemanticLDBA,
        belief: jax.Array,
        perceived_assignment: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute embedding, epsilon embeddings, and epsilon mask from belief.

        For ``map`` mode:   use the state with highest belief probability.
        For ``weighted`` mode: use belief-weighted sum of state embeddings.

        The epsilon mask is always computed from the MAP state (we don't have
        a canonical way to weight boolean masks).
        """
        map_state = jnp.argmax(belief).astype(jnp.int32)

        if self.belief_mode == "map":
            embedding = ldba.get_embedding(map_state)
        else:  # weighted
            # embeddings shape: (num_states, embed_dim)
            embedding = jnp.tensordot(belief, ldba.embeddings, axes=[[0], [0]])

        eps_embeddings = ldba.get_epsilon_embeddings(map_state)
        epsilon_enabled = self._epsilon_enabled(ldba, map_state, perceived_assignment)
        return embedding, eps_embeddings, epsilon_enabled

    # ------------------------------------------------------------------
    # Epsilon helpers (copied verbatim from NoisySemanticLDBAWrapper)
    # ------------------------------------------------------------------

    def _epsilon_step(
        self, state: BeliefSemanticLDBAWrapperState, info: dict
    ) -> EnvTransition:
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
        self,
        ldba: JaxSemanticLDBA,
        ldba_state: jax.Array,
        assignment_index: jax.Array,
    ) -> jax.Array:
        next_epsilon_states = ldba.get_next_epsilon_states(ldba_state)
        enabled = next_epsilon_states != -1
        next_states = jnp.where(
            enabled,
            ldba.transitions[next_epsilon_states, assignment_index],
            ldba_state,
        )
        is_valid = jnp.logical_not(ldba.is_sink_state(next_states))
        return jnp.logical_and(enabled, is_valid)
