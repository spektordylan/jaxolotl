"""Teacher-forced comparison of structured and GRU-MAP trackers on saved rollouts.

The archived test.npz was used to select the legacy GRU checkpoint, so this is
validation/diagnostic, not independent final evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

import jaxolotl
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from experiments.belief_semltl.learned_tracker import mask_invalid_states
from experiments.belief_semltl.proposition_estimator import PropositionEstimator
from experiments.belief_semltl.structured_filter_math import assignment_posterior, propagate_belief
from experiments.belief_semltl.run_learned_estimator import DEFAULT_FORMULAS, DEFAULT_NOISE, MODEL_PATH, load_estimator
from experiments.belief_semltl.run_structured_filter import DEFAULT_PRIOR, DEFAULT_PROPOSITION_GRU


def get_metadata(d, dataset):
    count = d['noisy_assignments'].shape[0]
    if 'formula_indices' in d and 'noise_probs' in d:
        return np.asarray(d['formula_indices']), np.asarray(d['noise_probs'])
    # Only the original archived full dataset may use implicit record order.
    if dataset.name != 'test.npz' or count != 3 * 9 * 3 * 16:
        raise ValueError('Missing formula/noise metadata; recollect trajectories')
    blocks = np.arange(count) // 16
    formulas = np.asarray(DEFAULT_FORMULAS)[(blocks // 3) % 9]
    noise = np.asarray(DEFAULT_NOISE)[blocks % 3]
    return formulas, noise


def predict_episode(transitions, num_states, initial_state, assignments, actions,
                    mask, p, prior, model):
    if model is None:
        posteriors = jax.vmap(lambda y: assignment_posterior(y, p, prior))(assignments)
    else:
        posteriors = jax.nn.softmax(model.unroll(assignments, actions))

    def step(belief, inputs):
        posterior, active = inputs
        next_belief = propagate_belief(belief, transitions, posterior, num_states)
        belief = jnp.where(active, next_belief, belief)
        return belief, jnp.argmax(belief)

    initial = jax.nn.one_hot(initial_state, transitions.shape[0])
    _, q_pred = jax.lax.scan(step, initial, (posteriors, mask))
    return q_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['empirical', 'proposition-gru'], required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'experiments/belief_semltl/data/trajectories/full/test.npz')
    parser.add_argument('--prior-file', type=Path, default=DEFAULT_PRIOR)
    parser.add_argument('--proposition-model', type=Path, default=DEFAULT_PROPOSITION_GRU)
    parser.add_argument('--gru-model', type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    with np.load(args.dataset) as d:
        formulas_index, noise_levels = get_metadata(d, args.dataset)
        data = {k: np.asarray(d[k]) for k in ('noisy_assignments', 'prev_actions', 'hard_states',
                                              'true_states', 'active_mask', 'num_states')}
    with args.prior_file.open() as f:
        prior = jnp.asarray(json.load(f)['assignment_probabilities'], dtype=jnp.float32)
    estimator = None
    if args.method == 'proposition-gru':
        template = PropositionEstimator(key=jax.random.key(0))
        estimator = eqx.tree_deserialise_leaves(args.proposition_model, template)
    baseline = load_estimator(args.gru_model)
    with initialize(version_base='1.3', config_path='../../conf'):
        cfg = compose(config_name='eval', overrides=['alg=sem_ltl', 'env=letter_world',
            'task_set=finite', 'run=pretrained'])
    base_env, _ = jaxolotl.make(cfg.env.name, reset_source='test', **cfg.get('env_params', {}))
    formulas = preprocess_formulas(cfg.formulas, TimeLimitWrapper(base_env))

    print('Teacher-forced state accuracy (diagnostic; not independent test):')
    for p in sorted(set(float(x) for x in noise_levels)):
        counts = np.zeros(3, dtype=np.float64)
        correct = np.zeros(3, dtype=np.float64)
        for formula in sorted(set(formulas_index)):
            selection = np.where((formulas_index == formula) & np.isclose(noise_levels, p))[0]
            if not len(selection):
                continue
            task = jax.tree.map(lambda x: x[int(formula)], formulas)
            if np.any(np.asarray(task.epsilon_transitions) >= 0):
                raise ValueError('Offline experiment supports epsilon-free formulas only')
            y = jnp.asarray(data['noisy_assignments'][selection])
            u = jnp.asarray(data['prev_actions'][selection])
            h = jnp.asarray(data['hard_states'][selection])
            mask = jnp.asarray(data['active_mask'][selection])
            true_q = np.asarray(data['true_states'][selection])
            if not np.all(data['num_states'][selection] == int(task.num_states)):
                raise ValueError('Formula num_states mismatch; check archived ordering')
            raw_gru = jax.vmap(baseline.unroll)(y, u, h)
            raw_gru = mask_invalid_states(raw_gru, task.num_states)
            q_map = np.asarray(jnp.argmax(raw_gru, axis=-1))
            q_struct = np.asarray(jax.vmap(
                lambda labels, actions, active: predict_episode(
                    task.transitions, task.num_states, task.initial_state,
                    labels, actions, active, jnp.asarray(p), prior, estimator),
            )(y, u, mask))
            if p == 0.0:
                q_map = np.asarray(h)
                q_struct = np.asarray(h)
            active = np.asarray(mask, dtype=bool)
            for idx, pred in enumerate((np.asarray(h), q_map, q_struct)):
                counts[idx] += active.sum()
                correct[idx] += ((pred == true_q) & active).sum()
        accuracies = correct / np.maximum(counts, 1)
        print(f'p={p:.2f} | hard {accuracies[0]:.2%} | GRU-MAP {accuracies[1]:.2%}'
              f' | structured {accuracies[2]:.2%} | struct-MAP {accuracies[2]-accuracies[1]:+.2%}')

if __name__ == '__main__':
    main()
