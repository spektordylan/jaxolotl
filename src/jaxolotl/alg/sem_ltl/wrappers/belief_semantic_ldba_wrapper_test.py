"""Tests for BeliefSemanticLDBAWrapper and belief update utilities.

Coverage:
1. compute_posterior_a: correct probabilities at p=0 and p>0.
2. compute_belief_next: sum-to-one, non-negativity, deterministic at p=0.
3. p=0 MAP wrapper matches NoisySemanticLDBAWrapper exactly (state, reward,
   termination, embedding) over 10 steps with matched PRNG keys.
4. Belief normalization after several steps.
5. Reward/termination use ground-truth state, not belief.
6. Weighted embedding equals probability-weighted sum.
7. MAP and weighted wrappers produce same reward/termination.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.belief_semantic_ldba_wrapper import (
    BeliefSemanticLDBAWrapper,
    compute_belief_next,
    compute_posterior_a,
)
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.environments.registration import make

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_task_params():
    env, params = make("LetterWorld")
    formulas = ["F a & F b"]
    ldbas = preprocess_formulas(formulas, env)
    task = jax.tree.map(lambda x: x[0], ldbas)
    return env, task, params


@pytest.fixture(scope="module")
def options(env_task_params):
    _, task, _ = env_task_params
    return CurriculumResetOptions(task=task)


# ---------------------------------------------------------------------------
# Unit tests: compute_posterior_a
# ---------------------------------------------------------------------------

def test_posterior_a_zero_noise():
    """At p=0, P(a|y) is a delta on y."""
    num_assignments = 13
    for y in range(num_assignments):
        post = compute_posterior_a(
            jnp.array(y, dtype=jnp.int32),
            num_assignments,
            jnp.array(0.0),
        )
        assert float(post[y]) == pytest.approx(1.0, abs=1e-6)
        for a in range(num_assignments):
            if a != y:
                assert float(post[a]) == pytest.approx(0.0, abs=1e-6)


def test_posterior_a_sums_to_one():
    """P(a|y) must sum to 1 for any y and p."""
    for p in [0.0, 0.1, 0.5, 1.0]:
        for y in [0, 5, 12]:
            post = compute_posterior_a(
                jnp.array(y, dtype=jnp.int32),
                13,
                jnp.array(p),
            )
            assert float(post.sum()) == pytest.approx(1.0, abs=1e-5)


def test_posterior_a_nonzero_noise():
    """At p=0.3, P(a=y|y) > P(a≠y|y) and uniform over others."""
    p = 0.3
    num = 13
    post = compute_posterior_a(jnp.array(5, dtype=jnp.int32), num, jnp.array(p))
    # P(a=y|y) must be highest
    assert float(post[5]) > float(post[0])
    # Alternatives must be equal
    alts = [float(post[i]) for i in range(num) if i != 5]
    assert max(alts) - min(alts) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: compute_belief_next
# ---------------------------------------------------------------------------

def test_belief_next_sums_to_one(env_task_params):
    env, task, _ = env_task_params
    num_states = task.transitions.shape[0]
    num_assignments = task.transitions.shape[1]
    belief = jnp.ones(num_states) / num_states
    for a in range(num_assignments):
        post = compute_posterior_a(jnp.array(a, dtype=jnp.int32), num_assignments, jnp.array(0.1))
        belief_next = compute_belief_next(belief, task.transitions, post)
        assert float(belief_next.sum()) == pytest.approx(1.0, abs=1e-5)
        assert jnp.all(belief_next >= 0)


def test_belief_next_zero_noise_matches_deterministic(env_task_params):
    """At p=0, belief stays a delta on the deterministic next state."""
    env, task, _ = env_task_params
    num_states = task.transitions.shape[0]
    num_assignments = task.transitions.shape[1]
    # Start from initial state
    q0 = int(task.initial_state)
    belief = jax.nn.one_hot(q0, num_states, dtype=jnp.float32)
    for a in range(num_assignments):
        post = compute_posterior_a(jnp.array(a, dtype=jnp.int32), num_assignments, jnp.array(0.0))
        belief_next = compute_belief_next(belief, task.transitions, post)
        # Should be a delta on the deterministic successor
        expected_q = int(task.transitions[q0, a])
        assert float(belief_next[expected_q]) == pytest.approx(1.0, abs=1e-5)
        for q in range(num_states):
            if q != expected_q:
                assert float(belief_next[q]) < 1e-5


def test_belief_next_nonnegative(env_task_params):
    env, task, _ = env_task_params
    num_states = task.transitions.shape[0]
    num_assignments = task.transitions.shape[1]
    key = jax.random.key(0)
    belief = jax.nn.softmax(jax.random.normal(key, (num_states,)))
    post = compute_posterior_a(jnp.array(3, dtype=jnp.int32), num_assignments, jnp.array(0.2))
    belief_next = compute_belief_next(belief, task.transitions, post)
    assert jnp.all(belief_next >= 0)


# ---------------------------------------------------------------------------
# Integration test: p=0 MAP belief wrapper == NoisySemanticLDBAWrapper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("belief_mode", ["map", "weighted"])
def test_belief_wrapper_zero_noise_matches_noisy_wrapper(env_task_params, options, belief_mode):
    """At p=0, both MAP and weighted modes reproduce the original wrapper's rewards,
    terminations, is_sink flags, and embeddings as NoisySemanticLDBAWrapper(p=0).
    """
    env, task, params = env_task_params

    noisy_wrapper = NoisySemanticLDBAWrapper(env, noise_prob=0.0, overwrite_finite=True)
    belief_wrapper = BeliefSemanticLDBAWrapper(env, noise_prob=0.0, belief_mode=belief_mode, overwrite_finite=True)

    key = jax.random.key(42)
    reset_key, step_key = jax.random.split(key)

    noisy_state, noisy_obs = noisy_wrapper.reset(reset_key, None, params, options)
    belief_state, belief_obs = belief_wrapper.reset(reset_key, None, params, options)

    # Initial LDBA states match.
    assert jnp.array_equal(noisy_state.ldba_state, belief_state.ldba_state)
    # Belief is a delta on the initial state.
    assert float(belief_state.belief[int(belief_state.ldba_state)]) == pytest.approx(1.0, abs=1e-5)
    # Initial embeddings match.
    assert jnp.allclose(noisy_obs.embedding, belief_obs.embedding, atol=1e-5)

    max_eps = task.epsilon_transitions.shape[1]
    act_key = jax.random.key(123)

    for step_i in range(10):
        act_key, subkey = jax.random.split(act_key)
        env_act = jax.random.randint(subkey, (), 0, 4)
        action = (env_act, jnp.array(max_eps, dtype=jnp.int32))

        noisy_trans = noisy_wrapper.step(step_key, noisy_state, action, params)
        belief_trans = belief_wrapper.step(step_key, belief_state, action, params)

        # Ground-truth states must match.
        assert jnp.array_equal(noisy_trans.state.ldba_state, belief_trans.state.ldba_state), (
            f"Step {step_i}: ldba_state mismatch: "
            f"{noisy_trans.state.ldba_state} vs {belief_trans.state.ldba_state}"
        )
        # Reward matches.
        assert jnp.allclose(noisy_trans.reward, belief_trans.reward, atol=1e-6), (
            f"Step {step_i}: reward mismatch: {noisy_trans.reward} vs {belief_trans.reward}"
        )
        # Termination matches.
        assert jnp.array_equal(noisy_trans.terminated, belief_trans.terminated), (
            f"Step {step_i}: terminated mismatch"
        )
        # is_sink matches.
        assert jnp.array_equal(
            noisy_trans.info["is_sink"], belief_trans.info["is_sink"]
        ), f"Step {step_i}: is_sink mismatch"
        # Embedding matches (belief is delta so MAP == deterministic state).
        assert jnp.allclose(noisy_trans.observation.embedding, belief_trans.observation.embedding, atol=1e-5), (
            f"Step {step_i}: embedding mismatch"
        )

        noisy_state = noisy_trans.state
        belief_state = belief_trans.state


# ---------------------------------------------------------------------------
# Test: weighted embedding equals belief-weighted sum
# ---------------------------------------------------------------------------

def test_weighted_embedding_matches_manual(env_task_params, options):
    """Verify that the weighted wrapper computes jnp.tensordot(belief, embeddings)."""
    env, task, params = env_task_params
    wrapper = BeliefSemanticLDBAWrapper(env, noise_prob=0.1, belief_mode="weighted", overwrite_finite=True)
    key = jax.random.key(5)
    state, obs = wrapper.reset(key, None, params, options)
    max_eps = task.epsilon_transitions.shape[1]

    # Take one step.
    action = (jnp.array(2, dtype=jnp.int32), jnp.array(max_eps, dtype=jnp.int32))
    trans = wrapper.step(key, state, action, params)

    # Manually compute expected embedding from returned belief.
    belief_next = trans.state.belief
    expected_emb = jnp.tensordot(belief_next, task.embeddings, axes=[[0], [0]])
    assert jnp.allclose(trans.observation.embedding, expected_emb, atol=1e-5), (
        "Weighted embedding does not match manual computation"
    )


# ---------------------------------------------------------------------------
# Test: rewards/termination use ground truth, not belief
# ---------------------------------------------------------------------------

def test_reward_uses_ground_truth():
    """Reward and termination must follow ldba_state (ground truth), not belief.

    Uses 'F a & G !b' which is known to have a sink state.
    """
    env, params = make("LetterWorld")
    # Use a formula with a known sink state
    ldbas = preprocess_formulas(["F a & G !b"], env)
    task = jax.tree.map(lambda x: x[0], ldbas)
    opts = CurriculumResetOptions(task=task)

    wrapper = BeliefSemanticLDBAWrapper(env, noise_prob=1.0, belief_mode="map", overwrite_finite=True)
    key = jax.random.key(99)
    state, _ = wrapper.reset(key, None, params, opts)
    max_eps = task.epsilon_transitions.shape[1]

    # Find a sink state.
    sink_candidates = jnp.nonzero(task.sink_states, size=1)[0]
    assert int(task.sink_states.sum()) > 0, "Formula must have sink states"
    sink_q = sink_candidates[0].astype(jnp.int32)
    initial_q = state.ldba_state

    # Inject a divergent state: true = initial (safe), belief concentrated on sink.
    num_states = task.transitions.shape[0]
    sink_belief = jax.nn.one_hot(sink_q, num_states, dtype=jnp.float32)
    divergent_state = eqx.tree_at(
        lambda s: (s.ldba_state, s.belief),
        state,
        (initial_q, sink_belief),
    )
    # MAP state is now sink_q, but ground-truth is initial_q (not sink).
    assert not bool(task.is_sink_state(initial_q)), "initial state should not be sink"
    assert bool(task.is_sink_state(sink_q)), "sink_q should be a sink"
    action = (jnp.array(0, dtype=jnp.int32), jnp.array(max_eps, dtype=jnp.int32))
    trans = wrapper.step(key, divergent_state, action, params)

    # Ground truth is not sink → no negative reward yet.
    assert float(trans.reward) >= 0.0, "Reward should not be -1 (sink) when true state is safe"
    assert not bool(trans.info["is_sink"]), "is_sink should reflect ground truth"


# ---------------------------------------------------------------------------
# Test: belief stays non-negative and sums to one over many steps
# ---------------------------------------------------------------------------

def test_belief_normalization_over_many_steps(env_task_params, options):
    env, task, params = env_task_params
    wrapper = BeliefSemanticLDBAWrapper(env, noise_prob=0.3, belief_mode="map", overwrite_finite=True)
    key = jax.random.key(7)
    state, _ = wrapper.reset(key, None, params, options)
    max_eps = task.epsilon_transitions.shape[1]

    for i in range(20):
        key, subkey = jax.random.split(key)
        action = (jnp.array(i % 4, dtype=jnp.int32), jnp.array(max_eps, dtype=jnp.int32))
        trans = wrapper.step(subkey, state, action, params)
        belief = trans.state.belief
        assert float(belief.sum()) == pytest.approx(1.0, abs=1e-4), f"Step {i}: belief does not sum to 1"
        assert jnp.all(belief >= 0), f"Step {i}: negative belief entry"
        if trans.terminated:
            break
        state = trans.state
