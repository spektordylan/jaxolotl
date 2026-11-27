"""Stage 2a/2b: closed-loop hard vs GRU-MAP vs structured LDBA filtering.

Stage 2a: --method empirical     (Bayes assignment posterior from training prior)
Stage 2b: --method proposition-gru (learned assignment posterior, exact LDBA delta)

Uses identical run keys and episode counts to run_learned_estimator.py.
All outputs are in separate, exclusive-create directories.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

import jaxolotl
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import NoisySemanticLDBAWrapper
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic

from experiments.belief_semltl.learned_stats import assert_zero_noise_parity
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper
from experiments.belief_semltl.proposition_estimator import PropositionEstimator
from experiments.belief_semltl.structured_filter_wrapper import StructuredFilterWrapper
from experiments.belief_semltl.run_learned_estimator import (
    DEFAULT_FORMULAS, DEFAULT_NOISE, MODEL_PATH, current_commit,
    load_estimator, rollout_hard, rollout_learned, save_csv,
)
from experiments.belief_semltl.pilot_stats import seed_ci

LOG = logging.getLogger('structured_filter')
DEFAULT_PRIOR = ROOT / 'experiments/belief_semltl/models/structured_filter/empirical_prior.json'
DEFAULT_PROPOSITION_GRU = ROOT / 'experiments/belief_semltl/models/structured_filter/proposition_gru/proposition_estimator.eqx'
METHODS = ('hard', 'map', 'structured')
METRICS = ('success', 'violation', 'step_accuracy', 'any_disagreement')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['empirical', 'proposition-gru'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model-seeds', nargs='+', type=int, default=[7, 8, 9])
    parser.add_argument('--num-episodes', type=int, default=16)
    parser.add_argument('--formula-indices', nargs='+', type=int, default=list(DEFAULT_FORMULAS))
    parser.add_argument('--noise-levels', nargs='+', type=float, default=list(DEFAULT_NOISE))
    parser.add_argument('--prior-file', type=Path, default=DEFAULT_PRIOR)
    parser.add_argument('--proposition-model', type=Path, default=DEFAULT_PROPOSITION_GRU)
    parser.add_argument('--gru-model', type=Path, default=MODEL_PATH)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.model_seeds = [7, 8]
        args.num_episodes = 4
        args.formula_indices = [0, 3]
        args.noise_levels = [0., .3]
    if (not args.model_seeds or len(set(args.model_seeds)) != len(args.model_seeds)
        or not args.formula_indices or len(set(args.formula_indices)) != len(args.formula_indices)
        or 5 in args.formula_indices
        or args.num_episodes < 1):
        parser.error('Invalid model/formula/episode selections or epsilon-bearing formula 5')
    args.noise_levels = sorted(set(args.noise_levels))
    if not args.noise_levels or args.noise_levels[0] != 0 or any(not 0 <= p <= 1 for p in args.noise_levels):
        parser.error('Noise levels must include p=0 and lie in [0,1]')
    args.output_dir = args.output_dir or (
        ROOT / 'experiments/belief_semltl/results/structured_filter' /
        args.method / ('smoke' if args.smoke else 'full')
    )
    return args


def episode_metrics(ep, episode):
    nsteps = max(1, int(ep.steps[episode]))
    return {
        'success': int(ep.returns[episode] > 0),
        'violation': int(ep.violations[episode]),
        'step_accuracy': float(ep.learned_correct_steps[episode] / nsteps),
        'any_disagreement': int(ep.learned_any_disagreement[episode]),
        'steps': nsteps,
    }


def summarize(rows, noise_levels, formulas):
    summary, effects = [], []
    for p in noise_levels:
        for f in [None, *formulas]:
            sample = [r for r in rows if r['noise_prob'] == p and
                      (f is None or r['formula_idx'] == f)]
            if not sample:
                continue
            seeds = sorted(set(r['model_seed'] for r in sample))
            for metric in METRICS:
                means = {}
                for method in METHODS:
                    values = np.array([
                        np.mean([r[f'{method}_{metric}'] for r in sample
                                 if r['model_seed'] == seed]) for seed in seeds
                    ])
                    means[method] = values
                    ci = seed_ci(values)
                    summary.append({
                        'noise_prob': p, 'scope': 'overall' if f is None else 'per_formula',
                        'formula_idx': -1 if f is None else f,
                        'method': method, 'metric': metric,
                        'mean': ci.mean, 'ci_lower': ci.lower, 'ci_upper': ci.upper,
                        'n_seeds': ci.n_seeds,
                    })
                for baseline in ('hard', 'map'):
                    ci = seed_ci(means['structured'] - means[baseline])
                    effects.append({
                        'noise_prob': p, 'scope': 'overall' if f is None else 'per_formula',
                        'formula_idx': -1 if f is None else f,
                        'metric': metric, 'treatment': 'structured', 'baseline': baseline,
                        'mean_delta': ci.mean, 'ci_lower': ci.lower,
                        'ci_upper': ci.upper, 'n_seeds': ci.n_seeds,
                    })
    return summary, effects


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    output_names = ('episodes.csv', 'summary.csv', 'paired_effects.csv', 'metadata.json')
    conflicts = [str(args.output_dir / name) for name in output_names if (args.output_dir / name).exists()]
    if conflicts:
        raise FileExistsError(f'Existing results: {conflicts}; select another --output-dir')
    if not args.gru_model.exists():
        raise FileNotFoundError(f'Missing Phase 2 GRU baseline: {args.gru_model}')
    with args.prior_file.open('r') as f:
        prior_info = json.load(f)
    prior = np.asarray(prior_info['assignment_probabilities'], dtype=np.float32)
    if prior.shape != (13,) or not np.all(prior > 0) or not np.isclose(prior.sum(), 1):
        raise ValueError('Invalid 13-class empirical prior')

    proposition_estimator = None
    if args.method == 'proposition-gru':
        if not args.proposition_model.exists():
            raise FileNotFoundError(f'Missing proposition GRU: {args.proposition_model}')
        template = PropositionEstimator(hidden_dim=args.hidden_dim, key=jax.random.key(0))
        proposition_estimator = eqx.tree_deserialise_leaves(args.proposition_model, template)

    with initialize(version_base='1.3', config_path='../../conf'):
        cfg = compose(config_name='eval', overrides=[
            'alg=sem_ltl', 'env=letter_world', 'task_set=finite', 'run=pretrained',
            f'eval.num_episodes={args.num_episodes}',
            'eval.models_per_batch=1', 'eval.formulas_per_batch=1',
        ])
    all_formulas = list(cfg.formulas)
    if any(i < 0 or i >= len(all_formulas) for i in args.formula_indices):
        raise ValueError('Formula index out of bounds')
    cfg.formulas = [all_formulas[i] for i in args.formula_indices]
    base_env, env_params = jaxolotl.make(cfg.env.name, reset_source='test', **cfg.get('env_params', {}))
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError('Selected formulas must be epsilon-free')
    model_key = jax.random.split(jax.random.key(0))[1]
    models, model_seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    all_seeds = [int(s) for s in model_seeds]
    if not set(args.model_seeds).issubset(all_seeds):
        raise ValueError(f'Missing pretrained seeds: {set(args.model_seeds) - set(all_seeds)}')
    selection = [all_seeds.index(seed) for seed in args.model_seeds]
    models = jax.tree.map(
        lambda x: x[jnp.asarray(selection)] if eqx.is_array(x) and x.ndim >= 1 and
        x.shape[0] == len(all_seeds) else x, models,
    )
    original_gru = load_estimator(args.gru_model)
    max_steps = int(env_params.max_steps_in_episode)
    deterministic = bool(cfg.eval.deterministic)
    rows = []
    LOG.info('method=%s, model seeds=%s', args.method, args.model_seeds)

    for model_idx, seed in enumerate(args.model_seeds):
        model = jax.tree.map(
            lambda x: x[model_idx] if eqx.is_array(x) and x.ndim >= 1 and
            x.shape[0] == len(args.model_seeds) else x, models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, model)
        for formula_pos, formula_idx in enumerate(args.formula_indices):
            task = jax.tree.map(lambda x: x[formula_pos], formulas)
            for p in args.noise_levels:
                run_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(200), int(seed)), int(formula_idx))
                run_key = jax.random.fold_in(run_key, int(round(p * 1_000_000)))
                hard_env = VectorizeWrapper(NoisySemanticLDBAWrapper(
                    base_env, noise_prob=p, overwrite_finite=True))
                map_env = VectorizeWrapper(LearnedTrackerWrapper(
                    base_env, estimator=original_gru, noise_prob=p, overwrite_finite=True))
                structured_env = VectorizeWrapper(StructuredFilterWrapper(
                    base_env, estimator=proposition_estimator, prior=jnp.asarray(prior),
                    noise_prob=p, overwrite_finite=True))
                hard_ep, _ = rollout_hard(agent, hard_env, env_params, task, run_key,
                                         args.num_episodes, max_steps, deterministic)
                map_ep, _ = rollout_learned(agent, map_env, env_params, task, run_key,
                                            args.num_episodes, max_steps, deterministic)
                structured_ep, _ = rollout_learned(agent, structured_env, env_params, task, run_key,
                                                   args.num_episodes, max_steps, deterministic)
                hard_ep, map_ep, structured_ep = (
                    jax.tree.map(np.asarray, e) for e in (hard_ep, map_ep, structured_ep)
                )
                if any(not np.all(e.completed) for e in (hard_ep, map_ep, structured_ep)):
                    raise RuntimeError('One or more rollouts did not complete')
                if p == 0:
                    assert_zero_noise_parity(hard_ep, map_ep, seed=int(seed), formula_idx=int(formula_idx))
                    assert_zero_noise_parity(hard_ep, structured_ep, seed=int(seed), formula_idx=int(formula_idx))
                for episode in range(args.num_episodes):
                    record = {'noise_prob': p, 'model_seed': seed,
                              'formula_idx': formula_idx, 'episode': episode}
                    for method, ep in zip(METHODS, (hard_ep, map_ep, structured_ep)):
                        # Hard rollout carries a mirrored learned_correct_steps metric.
                        record.update({f'{method}_{k}': v for k, v in episode_metrics(ep, episode).items()})
                    rows.append(record)
                LOG.info('seed=%d formula=%d p=%.2f success: hard=%.3f map=%.3f structured=%.3f',
                         seed, formula_idx, p, np.mean(hard_ep.returns > 0),
                         np.mean(map_ep.returns > 0), np.mean(structured_ep.returns > 0))

    summary, effects = summarize(rows, args.noise_levels, args.formula_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / 'episodes.csv', rows)
    save_csv(args.output_dir / 'summary.csv', summary)
    save_csv(args.output_dir / 'paired_effects.csv', effects)
    metadata = {
        'method': args.method, 'git_revision': current_commit(),
        'prior_file': str(args.prior_file), 'prior_provenance': prior_info.get('provenance'),
        'proposition_checkpoint': str(args.proposition_model) if proposition_estimator is not None else None,
        'gru_map_checkpoint': str(args.gru_model),
        'model_seeds': args.model_seeds, 'formula_indices': args.formula_indices,
        'noise_levels': args.noise_levels, 'episodes_per_condition': args.num_episodes,
        'deterministic_policy': deterministic,
        'paired_keys': '200 folded with seed, original formula index, scaled noise',
        'evaluation_note': 'Seed-level inference only; checkpoint selection on archived validation seeds 7..9',
        'limitations': [
            'State belief update assumes conditional factorization of proposition evidence.',
            'Independent proposition marginal ignores latent position and state correlations.',
            'Only epsilon-free formulas are evaluated.',
            'The frozen actor still receives the unmodified spatial grid.',
            'Policy-induced trajectories differ after action divergence.',
        ],
    }
    with (args.output_dir / 'metadata.json').open('x') as target:
        json.dump(metadata, target, indent=2)
        target.write('\n')
    print('\nCLOSED-LOOP SUCCESS (seed-level means):')
    print(f"{'noise':>7} {'hard':>9} {'GRU MAP':>9} {'structured':>11} {'struct-MAP':>12}")
    for p in args.noise_levels:
        means = {r['method']: r['mean'] for r in summary if r['noise_prob'] == p
                 and r['scope'] == 'overall' and r['metric'] == 'success'}
        print(f"{p:7.2f} {means['hard']:9.4f} {means['map']:9.4f} {means['structured']:11.4f} "
              f"{means['structured']-means['map']:+12.4f}")
    print('Results:', args.output_dir)

if __name__ == '__main__':
    main()
