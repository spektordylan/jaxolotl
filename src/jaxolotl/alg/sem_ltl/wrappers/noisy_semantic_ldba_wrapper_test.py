"""Tests for NoisySemanticLDBAWrapper and observation corruption."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
    _resolve_true_epsilon_successor,
    corrupt_assignment,
)
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAWrapper
from jaxolotl.environments.registration import make


def test_corrupt_assignment_zero_noise():
    """At p=0, corruption model must always return the true assignment."""
    key = jax.random.key(0)
    keys = jax.random.split(key, 1000)
    true_assignment = jnp.array(5, dtype=jnp.int32)
    corrupted = jax.vmap(
        lambda k: corrupt_assignment(
            k, true_assignment, num_assignments=13, noise_prob=0.0
        )
    )(keys)
    assert jnp.all(corrupted == true_assignment)


def test_corrupt_assignment_full_noise():
    """At p=1.0, corruption model must NEVER return the true assignment and must be valid."""
    key = jax.random.key(1)
    keys = jax.random.split(key, 5000)
    true_assignment = jnp.array(3, dtype=jnp.int32)
    corrupted = jax.vmap(
        lambda k: corrupt_assignment(
            k, true_assignment, num_assignments=13, noise_prob=1.0
        )
    )(keys)
    assert not jnp.any(corrupted == true_assignment)
    assert jnp.all((corrupted >= 0) & (corrupted < 13))

    # All 12 alternative assignments must be represented
    unique_vals = set(np.array(corrupted))
    expected_vals = set(range(13)) - {3}
    assert unique_vals == expected_vals


def test_corrupt_assignment_marginal_distribution():
    """Test categorical distribution matches p and uniform alternative distribution."""
    key = jax.random.key(2)
    n_samples = 100_000
    keys = jax.random.split(key, n_samples)
    true_assignment = jnp.array(7, dtype=jnp.int32)
    p = 0.20
    k_total = 13

    corrupted = jax.vmap(
        lambda k: corrupt_assignment(
            k, true_assignment, num_assignments=k_total, noise_prob=p
        )
    )(keys)

    corrupt_rate = float(jnp.mean(corrupted != true_assignment))
    assert abs(corrupt_rate - p) < 0.005  # Within 0.5%

    counts = np.bincount(np.array(corrupted), minlength=k_total)
    expected_alt_rate = p / (k_total - 1)
    for idx in range(k_total):
        if idx == 7:
            assert abs(counts[idx] / n_samples - (1 - p)) < 0.005
        else:
            assert abs(counts[idx] / n_samples - expected_alt_rate) < 0.005


def test_zero_noise_matches_standard_wrapper():
    """Verify that NoisySemanticLDBAWrapper with p=0.0 behaves identically to SemanticLDBAWrapper."""
    env, params = make("LetterWorld")
    formulas = ["F a & F b"]
    ldbas = preprocess_formulas(formulas, env)
    task = jax.tree.map(lambda x: x[0], ldbas)
    options = CurriculumResetOptions(task=task)

    std_wrapper = SemanticLDBAWrapper(env, overwrite_finite=True)
    noisy_wrapper = NoisySemanticLDBAWrapper(env, noise_prob=0.0, overwrite_finite=True)

    key = jax.random.key(42)
    key, reset_key = jax.random.split(key)

    std_state, std_obs = std_wrapper.reset(reset_key, None, params, options)
    noisy_state, noisy_obs = noisy_wrapper.reset(reset_key, None, params, options)

    assert jnp.array_equal(std_state.ldba_state, noisy_state.ldba_state)
    assert jnp.array_equal(noisy_state.ldba_state, noisy_state.perceived_ldba_state)
    assert jnp.allclose(std_obs.embedding, noisy_obs.embedding)
    assert jnp.array_equal(std_obs.epsilon_mask, noisy_obs.epsilon_mask)

    # Step through 10 actions
    act_key = jax.random.key(123)
    max_eps = task.epsilon_transitions.shape[1]
    for _ in range(10):
        key, step_key, act_key = jax.random.split(act_key, 3)
        action = (
            jax.random.randint(step_key, (), 0, 4),
            jnp.array(max_eps, dtype=jnp.int32),
        )

        std_trans = std_wrapper.step(step_key, std_state, action, params)
        noisy_trans = noisy_wrapper.step(step_key, noisy_state, action, params)

        assert jnp.array_equal(std_trans.state.ldba_state, noisy_trans.state.ldba_state)
        assert jnp.array_equal(
            noisy_trans.state.ldba_state, noisy_trans.state.perceived_ldba_state
        )
        assert jnp.array_equal(std_trans.reward, noisy_trans.reward)
        assert jnp.array_equal(std_trans.terminated, noisy_trans.terminated)
        assert jnp.array_equal(std_trans.info["is_sink"], noisy_trans.info["is_sink"])
        assert jnp.allclose(
            std_trans.observation.embedding, noisy_trans.observation.embedding
        )

        std_state = std_trans.state
        noisy_state = noisy_trans.state


def test_ground_truth_vs_perceived_separation():
    """Verify that rewards and terminations depend exclusively on ground truth, not perceived state."""
    env, params = make("LetterWorld")
    formulas = ["F a & G !b"]
    ldbas = preprocess_formulas(formulas, env)
    task = jax.tree.map(lambda x: x[0], ldbas)
    options = CurriculumResetOptions(task=task)

    # With high noise (p=1.0), perceived state will rapidly diverge from true state
    wrapper = NoisySemanticLDBAWrapper(env, noise_prob=1.0, overwrite_finite=True)

    key = jax.random.key(99)
    state, obs = wrapper.reset(key, None, params, options)
    max_eps = task.epsilon_transitions.shape[1]

    # Run for 20 steps
    for step_i in range(20):
        key, subkey = jax.random.split(key)
        action = (
            jnp.array(step_i % 4, dtype=jnp.int32),
            jnp.array(max_eps, dtype=jnp.int32),
        )
        trans = wrapper.step(subkey, state, action, params)

        # Scientific safeguard checks:
        true_sink = task.is_sink_state(trans.state.ldba_state)
        assert trans.info["is_sink"] == true_sink

        # Policy observation must reflect perceived state
        expected_emb = task.get_embedding(trans.state.perceived_ldba_state)
        assert jnp.allclose(trans.observation.embedding, expected_emb)

        # Policy observation must NOT be forced to match true state if they diverge
        if trans.info["disagreement"]:
            # State disagreed: check that observation embedding is for perceived, not true
            true_emb = task.get_embedding(trans.state.ldba_state)
            if not jnp.allclose(expected_emb, true_emb):
                assert not jnp.allclose(trans.observation.embedding, true_emb)

        state = trans.state


def test_ground_truth_reward_and_termination_independence():
    """Explicitly verify that reward, termination, and sink flags depend strictly on q_true."""
    env, params = make("LetterWorld")
    # Formula requiring reaching 'a' while avoiding 'b'
    formulas = ["F a & G !b"]
    ldbas = preprocess_formulas(formulas, env)
    task = jax.tree.map(lambda x: x[0], ldbas)
    options = CurriculumResetOptions(task=task)

    # Use p=0 wrapper as base to manually step with controlled perceived vs true
    wrapper = NoisySemanticLDBAWrapper(env, noise_prob=0.0, overwrite_finite=True)
    key = jax.random.key(7)
    state, obs = wrapper.reset(key, None, params, options)
    max_eps = task.epsilon_transitions.shape[1]

    # Verify initial state agreement
    assert jnp.array_equal(state.ldba_state, state.perceived_ldba_state)
    initial_q = state.ldba_state

    # Construct a state where perceived_ldba_state is sink (e.g. visited 'b'),
    # but true ldba_state is still safe (initial_q).
    sink_candidates = jnp.nonzero(task.sink_states, size=1)[0]
    sink_q = sink_candidates[0].astype(jnp.int32)
    assert bool(task.is_sink_state(sink_q))
    assert not bool(task.is_sink_state(initial_q))

    divergent_state = eqx.tree_at(
        lambda s: (s.ldba_state, s.perceived_ldba_state),
        state,
        (initial_q, sink_q),
    )

    # Step in an empty environment transition (no proposition triggered, stay in initial_q)
    empty_action = (jnp.array(0, dtype=jnp.int32), jnp.array(max_eps, dtype=jnp.int32))
    trans = wrapper.step(key, divergent_state, empty_action, params)

    # True state is NOT sink, so reward must be 0.0 (or non-negative), is_sink must be False, terminated must be False
    assert float(trans.reward) == 0.0
    assert not bool(trans.info["is_sink"])
    assert bool(trans.info["perceived_is_sink"])
    assert not bool(trans.terminated)

    # Policy observation must use perceived state (which is sink_q)
    sink_emb = task.get_embedding(sink_q)
    assert jnp.allclose(trans.observation.embedding, sink_emb)

    # Now construct the converse: true state is sink, but perceived state is safe
    divergent_state_2 = eqx.tree_at(
        lambda s: (s.ldba_state, s.perceived_ldba_state),
        state,
        (sink_q, initial_q),
    )
    trans_2 = wrapper.step(key, divergent_state_2, empty_action, params)

    # True state IS sink, so reward must be -1.0, is_sink must be True, terminated must be True
    assert float(trans_2.reward) == -1.0
    assert bool(trans_2.info["is_sink"])
    assert not bool(trans_2.info["perceived_is_sink"])
    assert bool(trans_2.terminated)

    # Policy observation must STILL use perceived state (initial_q), NOT true state (sink_q)
    initial_emb = task.get_embedding(initial_q)
    assert jnp.allclose(trans_2.observation.embedding, initial_emb)



def test_unavailable_true_epsilon_does_not_consume_proposition():
    """An epsilon action valid only for the perceived state leaves q_true unchanged.

    Crucially, falling back to the ordinary symbol transition is also wrong:
    an epsilon action must not consume a proposition when its true edge is missing.
    """
    true_q = jnp.array(2, dtype=jnp.int32)
    perceived_q = jnp.array(3, dtype=jnp.int32)
    perceived_epsilon_successor = jnp.array(5, dtype=jnp.int32)
    unavailable_true_epsilon = jnp.array(-1, dtype=jnp.int32)
    regular_true_successor = jnp.array(4, dtype=jnp.int32)

    # This action changes the perceived state but has no true-state epsilon edge.
    assert int(perceived_epsilon_successor) != int(perceived_q)
    next_true_q = _resolve_true_epsilon_successor(
        true_q, regular_true_successor, unavailable_true_epsilon, jnp.array(True)
    )
    assert int(next_true_q) == int(true_q)
    assert int(next_true_q) != int(regular_true_successor)
    assert int(next_true_q) != -1

    # If a true-state epsilon edge exists, it is taken.
    valid_true_epsilon = jnp.array(6, dtype=jnp.int32)
    assert int(
        _resolve_true_epsilon_successor(
            true_q, regular_true_successor, valid_true_epsilon, jnp.array(True)
        )
    ) == 6

    # A regular environment action must still use the proposition transition.
    assert int(
        _resolve_true_epsilon_successor(
            true_q, regular_true_successor, unavailable_true_epsilon, jnp.array(False)
        )
    ) == int(regular_true_successor)
