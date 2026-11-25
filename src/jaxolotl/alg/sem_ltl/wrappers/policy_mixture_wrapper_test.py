"""Integration tests for wrapper delegation and strictly nonprivileged beliefs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import NoisySemanticLDBAWrapper
from jaxolotl.alg.sem_ltl.wrappers.policy_mixture_wrapper import PolicyMixtureWrapper
from jaxolotl.environments.registration import make


@pytest.fixture(scope="module")
def setup_world():
    env, params = make("LetterWorld")
    batch = preprocess_formulas(["F a & F b"], env)
    task = jax.tree.map(lambda x: x[0], batch)
    return env, params, task


def test_initial_observation_unchanged_and_belief_delta(setup_world):
    env, params, task = setup_world
    base = NoisySemanticLDBAWrapper(env, noise_prob=0.3, overwrite_finite=True)
    mixture = PolicyMixtureWrapper(env, noise_prob=0.3, overwrite_finite=True)
    options = CurriculumResetOptions(task=task)
    key = jax.random.key(23)
    hard_s, hard_obs = base.reset(key, None, params, options)
    mix_s, mix_obs = mixture.reset(key, None, params, options)
    np.testing.assert_array_equal(hard_obs.embedding, mix_obs.base.embedding)
    np.testing.assert_array_equal(hard_s.ldba_state, mix_s.state.ldba_state)
    np.testing.assert_array_equal(hard_s.perceived_ldba_state, mix_s.state.perceived_ldba_state)
    np.testing.assert_allclose(mix_s.belief.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(mix_s.belief[int(task.initial_state)], 1.0, atol=1e-6)
    np.testing.assert_array_equal(task.embeddings, mix_obs.state_embeddings)


def test_forced_actions_delegate_ground_truth_and_hard_monitor(setup_world):
    env, params, task = setup_world
    base = NoisySemanticLDBAWrapper(env, noise_prob=0.3, overwrite_finite=True)
    mix = PolicyMixtureWrapper(env, noise_prob=0.3, overwrite_finite=True)
    options = CurriculumResetOptions(task=task)
    hard_s, _ = base.reset(jax.random.key(55), None, params, options)
    mix_s, _ = mix.reset(jax.random.key(55), None, params, options)
    max_eps = task.epsilon_transitions.shape[1]
    key = jax.random.key(12)
    for t in range(5):
        key, action_key, step_key = jax.random.split(key, 3)
        move = jax.random.randint(action_key, (), minval=0, maxval=4)
        action = (move, jnp.asarray(max_eps, dtype=jnp.int32))
        ha = base.step(step_key, hard_s, action, params)
        ma = mix.step(step_key, mix_s, action, params)
        np.testing.assert_array_equal(ha.state.ldba_state, ma.state.state.ldba_state)
        np.testing.assert_array_equal(ha.state.perceived_ldba_state, ma.state.state.perceived_ldba_state)
        np.testing.assert_allclose(ha.reward, ma.reward)
        np.testing.assert_array_equal(ha.terminated, ma.terminated)
        np.testing.assert_array_equal(ha.info["is_sink"], ma.info["is_sink"])
        np.testing.assert_array_equal(ha.observation.embedding, ma.observation.base.embedding)
        assert abs(float(ma.state.belief.sum()) - 1.0) < 1e-5
        hard_s, mix_s = ha.state, ma.state


def test_p_zero_belief_matches_hard_progression(setup_world):
    env, params, task = setup_world
    mix = PolicyMixtureWrapper(env, noise_prob=0.0, overwrite_finite=True)
    options = CurriculumResetOptions(task=task)
    state, observation = mix.reset(jax.random.key(4), None, params, options)
    max_eps = task.epsilon_transitions.shape[1]
    for i in range(5):
        action = (jnp.asarray(i % 4, dtype=jnp.int32), jnp.asarray(max_eps, dtype=jnp.int32))
        trans = mix.step(jax.random.fold_in(jax.random.key(6), i), state, action, params)
        q = int(trans.state.state.perceived_ldba_state)
        assert int(jnp.argmax(trans.state.belief)) == q
        assert float(trans.state.belief[q]) == pytest.approx(1.0, abs=1e-5)
        state = trans.state


def test_reset_with_existing_outer_state_preserves_wrapper_nesting(setup_world):
    """Regression: a repeated reset must not unwrap the noisy monitor twice."""
    env, params, task = setup_world
    mix = PolicyMixtureWrapper(env, noise_prob=0.3, overwrite_finite=True)
    options = CurriculumResetOptions(task=task)
    key = jax.random.key(201)
    initial, _ = mix.reset(key, None, params, options)
    repeated, obs = mix.reset(key, initial, params, options)
    np.testing.assert_array_equal(initial.state.ldba_state, repeated.state.ldba_state)
    np.testing.assert_array_equal(
        initial.state.perceived_ldba_state, repeated.state.perceived_ldba_state
    )
    np.testing.assert_allclose(repeated.belief.sum(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        repeated.state.ldba.get_embedding(repeated.state.perceived_ldba_state),
        obs.base.embedding,
    )
