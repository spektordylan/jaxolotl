"""Estimate a proposition prior from training data only; never touch test data.

The archived full/train.npz lacks true assignments and noise metadata. Its
collector iterated model_seed, formula, noise=(0,.1,.3), episode, so the p=0
slices contain uncorrupted assignments. This fallback is restricted to the
exact archived (7 seeds * 9 formulas * 3 noise levels * 32 episodes) layout.
Recollected files with true_assignments use those labels directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'experiments/belief_semltl/data/trajectories/full/train.npz'
OUTPUT = ROOT / 'experiments/belief_semltl/models/structured_filter/empirical_prior.json'


def estimate_prior(data: dict, pseudocount: float = 1.0):
    if pseudocount <= 0:
        raise ValueError('Pseudocount must be positive')
    observed = data['noisy_assignments']
    mask = data['active_mask'].astype(bool)
    if 'true_assignments' in data:
        true = data['true_assignments']
        selected = np.ones(observed.shape[0], dtype=bool)
        provenance = 'all active ground-truth proposition labels from training rollouts'
    elif 'noise_probs' in data:
        selected = np.isclose(data['noise_probs'], 0)
        if not np.any(selected):
            raise ValueError('No zero-noise training trajectories')
        true = observed
        provenance = 'explicitly labeled p=0 training rollouts'
    else:
        expected = (7, 9, 3, 32)
        if observed.shape[0] != int(np.prod(expected)):
            raise ValueError('Unknown archive layout: recollect true_assignments or noise metadata')
        episode_in_block = np.arange(observed.shape[0]) // expected[3]
        selected = (episode_in_block % expected[2]) == 0
        true = observed
        provenance = 'ARCHIVED ORDER ASSUMPTION: seed(7), formula(9), noise(3), episodes(32); p=0 blocks'
    values = true[selected][mask[selected]]
    if values.size == 0 or np.any((values < 0) | (values >= 13)):
        raise ValueError('Invalid or empty selected proposition labels')
    counts = np.bincount(values.astype(np.int64), minlength=13)
    prior = (counts + pseudocount) / (counts.sum() + 13 * pseudocount)
    return prior, counts, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--pseudocount', type=float, default=1.0)
    args = parser.parse_args()
    with np.load(args.train_data) as source:
        prior, counts, provenance = estimate_prior(source, args.pseudocount)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'assignment_probabilities': prior.tolist(),
               'counts': counts.tolist(), 'n_training_steps': int(counts.sum()),
               'pseudocount': args.pseudocount, 'provenance': provenance,
               'source': str(args.train_data)}
    with args.output.open('x', encoding='utf8') as target:
        json.dump(payload, target, indent=2)
    print('Prior written to', args.output)
    print('P(empty, index 12) =', round(float(prior[12]), 4))
    print('Selected training steps =', int(counts.sum()))

if __name__ == '__main__':
    main()
