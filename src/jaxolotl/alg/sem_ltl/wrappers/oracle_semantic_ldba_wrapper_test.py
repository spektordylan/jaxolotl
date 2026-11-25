"""Integration tests for privileged semantic policy input (requires Pixi/Jaxolotl).

Run under JAX_PLATFORMS=cpu. Formula is epsilon-free by construction.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import NoisySemanticLDBAWrapper
from jaxolotl.alg.sem_ltl.wrappers.oracle_semantic_ldba_wrapper import OracleSemanticLDBAWrapper
from jaxolotl.environments.registration import make


@pytest.fixture(scope="module")
def world():
    env, params = make("LetterWorld")
    task = jax.tree.map(lambda x: x[0], preprocess_formulas(["F a & F b"], env))
    return env, params, task


def test_oracle_reset_matches_hard_at_zero_noise(world):
    env, params, task = world
    hard = NoisySemanticLDBAWrapper(env, noise_prob=0.0, overwrite_finite=True)
    oracle = OracleSemanticLDBAWrapper(env, noise_prob=0.0, overwrite_finite=True)
    key = jax.random.key(2026)
    opts = CurriculumResetOptions(task=task)
    hard_state, hard_obs = hard.reset(key, None, params, opts)
    oracle_state, oracle_obs = oracle.reset(key, None, params, opts)
    assert int(hard_state.ldba_state) == int(oracle_state.ldba_state)
    assert jnp.array_equal(hard_obs.embedding, oracle_obs.embedding)
    assert jnp.array_equal(hard_obs.epsilon_mask, oracle_obs.epsilon_mask)


def test_oracle_follows_true_ldba_and_keeps_reward_termination(world):
    env, params, task = world
    hard = NoisySemanticLDBAWrapper(env, noise_prob=1.0, overwrite_finite=True)
    oracle = OracleSemanticLDBAWrapper(env, noise_prob=1.0, overwrite_finite=True)
    opts = CurriculumResetOptions(task=task)
    hs, _ = hard.reset(jax.random.key(17), None, params, opts)
    os, _ = oracle.reset(jax.random.key(17), None, params, opts)
    no_eps_index = task.epsilon_transitions.shape[1]
    for t in range(8):
        action = (jnp.asarray(t % 4, dtype=jnp.int32), jnp.asarray(no_eps_index, dtype=jnp.int32))
        key = jax.random.fold_in(jax.random.key(44), t)
        hard_tr = hard.step(key, hs, action, params)
        oracle_tr = oracle.step(key, os, action, params)
        assert jnp.array_equal(hard_tr.reward, oracle_tr.reward)
        assert jnp.array_equal(hard_tr.terminated, oracle_tr.terminated)
        assert jnp.array_equal(hard_tr.truncated, oracle_tr.truncated)
        assert jnp.array_equal(hard_tr.state.ldba_state, oracle_tr.state.ldba_state)
        assert jnp.array_equal(hard_tr.state.perceived_ldba_state, oracle_tr.state.perceived_ldba_state)
        true_embedding = task.get_embedding(oracle_tr.state.ldba_state)
        assert jnp.allclose(oracle_tr.observation.embedding, true_embedding)
        assert jnp.allclose(oracle_tr.terminal_observation.embedding, true_embedding)
        assert not bool(oracle_tr.info["disagreement"])
        assert jnp.array_equal(oracle_tr.info["latent_noisy_disagreement"], hard_tr.info["disagreement"])
        hs, os = hard_tr.state, oracle_tr.state


def test_oracle_observation_uses_true_not_perceived_state(world):
    env, params, task = world
    oracle = OracleSemanticLDBAWrapper(env, noise_prob=0.3, overwrite_finite=True)
    opts = CurriculumResetOptions(task=task)
    state, _ = oracle.reset(jax.random.key(23), None, params, opts)
    assert task.transitions.shape[0] >= 2
    true_q = jnp.asarray(1, dtype=jnp.int32)
    wrong_q = jnp.asarray(0, dtype=jnp.int32)
    # Wrapper state is an Equinox module, not a NamedTuple. Construct a new
    # immutable state with distinct true and perceived automaton states.
    constructed = type(state)(
        state=state.state,
        ldba=state.ldba,
        ldba_state=true_q,
        perceived_ldba_state=wrong_q,
        obs=state.obs,
        propositions=state.propositions,
        perceived_propositions=state.perceived_propositions,
        has_disagreed=state.has_disagreed,
    )
    oracle_obs = oracle._oracle_semantic_observation(constructed)
    assert jnp.allclose(oracle_obs.embedding, task.get_embedding(true_q))
    assert jnp.array_equal(
        oracle_obs.epsilon_mask,
        oracle._epsilon_enabled(task, true_q, env.map_assignment_to_index(constructed.propositions)),
    )


def test_oracle_policy_features_noise_invariance_with_fixed_actions(world):
    env, params, task = world
    opts = CurriculumResetOptions(task=task)
    low = OracleSemanticLDBAWrapper(env, noise_prob=0.0, overwrite_finite=True)
    high = OracleSemanticLDBAWrapper(env, noise_prob=0.3, overwrite_finite=True)
    ls, lo = low.reset(jax.random.key(66), None, params, opts)
    hs, ho = high.reset(jax.random.key(66), None, params, opts)
    assert jnp.array_equal(lo.embedding, ho.embedding)
    no_eps_index = task.epsilon_transitions.shape[1]
    for t in range(8):
        action = (jnp.asarray(t % 4, dtype=jnp.int32), jnp.asarray(no_eps_index, dtype=jnp.int32))
        key = jax.random.fold_in(jax.random.key(77), t)
        lt = low.step(key, ls, action, params)
        ht = high.step(key, hs, action, params)
        assert jnp.array_equal(lt.observation.embedding, ht.observation.embedding)
        assert jnp.array_equal(lt.observation.epsilon_mask, ht.observation.epsilon_mask)
        assert jnp.array_equal(lt.reward, ht.reward)
        assert jnp.array_equal(lt.terminated, ht.terminated)
        ls, hs = lt.state, ht.state
