"""Train a formula-agnostic GRU to predict true 13-way LetterWorld propositions.

Requires a *freshly collected* train/val dataset containing true_assignments.
Never loads a held-out test split to choose model weights. The existing
collector uses model seeds 0..6 vs 7..9; its 'test.npz' is a validation split.
The final closed-loop evaluation should use fresh independent environment runs.
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
import optax

from experiments.belief_semltl.proposition_estimator import PropositionEstimator

DEFAULT_DATA = ROOT / 'experiments/belief_semltl/data/structured_full'
DEFAULT_OUT = ROOT / 'experiments/belief_semltl/models/structured_filter/proposition_gru'


def load_data(path):
    with np.load(path) as d:
        required = ('noisy_assignments', 'prev_actions', 'true_assignments', 'active_mask')
        missing = [name for name in required if name not in d.files]
        if missing:
            raise ValueError(f'{path} missing labels {missing}; recollect with updated collect_trajectories.py')
        values = {k: jnp.asarray(d[k]) for k in required}
        if 'formula_indices' in d.files:
            values['formula_indices'] = jnp.asarray(d['formula_indices'])
        return values


def loss(model, batch):
    logits = jax.vmap(model.unroll)(batch['noisy_assignments'], batch['prev_actions'])
    labels = batch['true_assignments']
    mask = batch['active_mask'].astype(jnp.float32)
    per_step_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.sum(per_step_loss * mask) / jnp.maximum(jnp.sum(mask), 1.)


def accuracy(model, batch):
    logits = jax.vmap(model.unroll)(batch['noisy_assignments'], batch['prev_actions'])
    pred = jnp.argmax(logits, axis=-1)
    mask = batch['active_mask']
    denom = jnp.maximum(jnp.sum(mask), 1)
    return float(jnp.sum((pred == batch['true_assignments']) & mask) / denom)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--holdout-formula-index', type=int, default=None,
                        help='Exclude formula from training AND checkpoint-selection validation')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    data_dir = args.data_dir or (ROOT / 'experiments/belief_semltl/data' /
                                  ('structured_smoke' if args.smoke else 'structured_full'))
    output = args.output_dir or (DEFAULT_OUT.parent / ('proposition_gru_smoke' if args.smoke else 'proposition_gru'))
    checkpoint = output / 'proposition_estimator.eqx'
    metrics_file = output / 'metrics.json'
    if checkpoint.exists() or metrics_file.exists():
        raise FileExistsError(f'Existing model artifacts in {output}; pass a new --output-dir')
    train = load_data(data_dir / 'train.npz')
    val = load_data(data_dir / 'test.npz')  # validation, not independent test
    if args.holdout_formula_index is not None:
        for split_name, split in [('train', train), ('validation', val)]:
            if 'formula_indices' not in split:
                raise ValueError(f'{split_name} lacks formula_indices; recollect data')
            selected = np.asarray(split['formula_indices']) != args.holdout_formula_index
            if not selected.any():
                raise ValueError(f'No {split_name} data remain after excluding held-out formula')
            filtered = {k: v[selected] for k, v in split.items()}
            if split_name == 'train':
                train = filtered
            else:
                val = filtered
    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    model = PropositionEstimator(hidden_dim=args.hidden_dim, key=init_key)
    opt = optax.adamw(args.learning_rate, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_batch(m, state, batch):
        l, grad = eqx.filter_value_and_grad(loss)(m, batch)
        updates, next_state = opt.update(grad, state, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), next_state, l

    n = int(train['noisy_assignments'].shape[0])
    batch_size = min(args.batch_size, n)
    best = -1.0
    best_model = model
    history = []
    for epoch in range(1, (3 if args.smoke else args.epochs) + 1):
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, n))
        for start in range(0, n, batch_size):
            ids = order[start:start + batch_size]
            batch = {k: v[ids] for k, v in train.items() if k != 'formula_indices'}
            model, opt_state, _ = train_batch(model, opt_state, batch)
        train_acc = accuracy(model, train)
        val_acc = accuracy(model, val)
        history.append({'epoch': epoch, 'train_accuracy': train_acc, 'validation_accuracy': val_acc})
        print(f'epoch {epoch:02d}: train={train_acc:.4f}, validation={val_acc:.4f}')
        if val_acc > best:
            best = val_acc
            best_model = model

    output.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint, best_model)
    with metrics_file.open('x') as f:
        json.dump({'best_validation_accuracy': best, 'history': history,
                   'train_data': str(data_dir),
                   'held_out_formula_index': args.holdout_formula_index,
                   'note': 'test.npz used for checkpoint selection; this is validation, not final test'}, f, indent=2)
    print('Checkpoint:', checkpoint)

if __name__ == '__main__':
    main()
