"""Training-free stochastic policy aggregation for frozen SemLTL agents.

The evaluator calls Agent.get_action(observation).sample(seed=key). Instead of
manually extracting an actor's logits or approximating action distributions,
we draw a valid semantic state from the belief then sample from the pretrained
agent's action distribution at that state. By marginalization, this is exactly
pi_mix(a) = (1-alpha) pi_hard(a) + alpha sum_q b(q) pi(a|e(q)).

No neural weights are changed; no out-of-distribution interpolated embedding is
used. The same sample key reaches the original actor for CRN parity checks.
Do not interpret .mode(): this is a stochastic-policy experiment only.
"""

from __future__ import annotations

import equinox as eqx

from experiments.belief_semltl.mixture_sampling import choose_policy_embeddings
from jaxolotl.alg.sem_ltl.wrappers.policy_mixture_wrapper import PolicyMixtureObservation


class HierarchicalPolicyMixture(eqx.Module):
    base_agent: object
    observation: PolicyMixtureObservation
    alpha: float = eqx.field(static=True)

    def sample(self, seed):
        obs = self.observation
        conditioning = choose_policy_embeddings(
            seed,
            obs.base.embedding,
            obs.state_embeddings,
            obs.belief,
            self.alpha,
        )
        # Preserve every other field, including navigational observation and
        # epsilon features. Candidate tasks are guaranteed epsilon-free.
        conditioned_obs = eqx.tree_at(
            lambda o: o.embedding,
            obs.base,
            conditioning,
        )
        # IMPORTANT: call pretrained actor sample with *unchanged* seed.
        return self.base_agent.get_action(conditioned_obs).sample(seed=seed)

    def mode(self):
        raise RuntimeError(
            "Hierarchical sampling is an exact STOCHASTIC policy mixture, "
            "not a deterministic argmax of mixture probabilities. The runner "
            "must set eval.deterministic=false."
        )


class PolicyMixtureAgent(eqx.Module):
    base_agent: object
    alpha: float = eqx.field(static=True)

    def __init__(self, base_agent, alpha: float):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie in [0,1]")
        self.base_agent = base_agent
        self.alpha = float(alpha)

    def get_action(self, observation: PolicyMixtureObservation):
        return HierarchicalPolicyMixture(self.base_agent, observation, self.alpha)

    def update(self, observation, state, propositions, env):
        # The Jaxolotl evaluator calls this at reset and after transitions.
        # Preserve any updates from the wrapped agent, while passing its native
        # observation object (the extra belief is not a pretrained input).
        new_base = self.base_agent.update(
            observation.base, state.state, propositions, env
        )
        return PolicyMixtureAgent(new_base, self.alpha)
