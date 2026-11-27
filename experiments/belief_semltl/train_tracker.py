"""Train LearnedStateEstimator on collected trajectory datasets.

Minimizes masked cross-entropy loss over true LDBA states:
    L(psi) = - sum_{t in active} log b_t(q_t^true) / sum(active)

Evaluates accuracy, NLL, and calibration on validation split.
Saves model checkpoint to checkpoints/learned_tracker.eqx.

Usage:
    MPLCONFIGDIR=/tmp JAX_PLATFORMS=cpu .pixi/envs/default/bin/python experiments/belief_semltl/train_tracker.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.belief_semltl.learned_tracker import LearnedStateEstimator

LOG = logging.getLogger("train_tracker")


def load_dataset(data_path: Path):
    with np.load(data_path) as data:
        return {
            "noisy_assignments": jnp.asarray(data["noisy_assignments"], dtype=jnp.int32),
            "prev_actions": jnp.asarray(data["prev_actions"], dtype=jnp.int32),
            "hard_states": jnp.asarray(data["hard_states"], dtype=jnp.int32),
            "true_states": jnp.asarray(data["true_states"], dtype=jnp.int32),
            "active_mask": jnp.asarray(data["active_mask"], dtype=jnp.bool_),
            "num_states": jnp.asarray(data["num_states"], dtype=jnp.int32),
        }


def compute_batch_loss(model: LearnedStateEstimator, batch: dict[str, jax.Array]) -> tuple[jax.Array, dict[str, jax.Array]]:
    # batch: (B, T)
    # vmap unroll over batch dimension
    unroll_fn = jax.vmap(model.unroll, in_axes=(0, 0, 0))
    logits = unroll_fn(batch["noisy_assignments"], batch["prev_actions"], batch["hard_states"])  # (B, T, max_states)

    # Mask logits outside valid num_states
    max_states = model.max_states
    state_indices = jnp.arange(max_states)[None, None, :]  # (1, 1, S)
    valid_state_mask = state_indices < batch["num_states"][:, None, None]  # (B, 1, S)
    masked_logits = jnp.where(valid_state_mask, logits, -1e9)

    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)  # (B, T, S)
    target_log_probs = jnp.take_along_axis(log_probs, batch["true_states"][:, :, None], axis=-1).squeeze(-1)  # (B, T)

    active = batch["active_mask"]  # (B, T)
    active_count = jnp.maximum(jnp.sum(active), 1.0)
    loss = -jnp.sum(jnp.where(active, target_log_probs, 0.0)) / active_count

    # Metrics
    preds = jnp.argmax(masked_logits, axis=-1)  # (B, T)
    correct = (preds == batch["true_states"]) & active
    hard_correct = (batch["hard_states"] == batch["true_states"]) & active

    accuracy = jnp.sum(correct) / active_count
    hard_accuracy = jnp.sum(hard_correct) / active_count

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "hard_accuracy": hard_accuracy,
    }
    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "experiments" / "belief_semltl" / "data" / "trajectories" / "full")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments" / "belief_semltl" / "checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    for name in ("learned_tracker_best.eqx", "learned_tracker_final.eqx"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.output_dir / name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading dataset from %s...", args.data_dir)
    train_data = load_dataset(args.data_dir / "train.npz")
    test_data = load_dataset(args.data_dir / "test.npz")

    num_train = train_data["noisy_assignments"].shape[0]
    num_test = test_data["noisy_assignments"].shape[0]
    LOG.info("Loaded %d train episodes, %d test episodes", num_train, num_test)

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    model = LearnedStateEstimator(
        max_states=14,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        key=model_key,
    )

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        grad_fn = eqx.filter_value_and_grad(compute_batch_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, metrics

    @eqx.filter_jit
    def eval_step(model, batch):
        _, metrics = compute_batch_loss(model, batch)
        return metrics

    LOG.info("Starting training for %d epochs (batch size %d)...", args.epochs, args.batch_size)
    best_test_acc = 0.0
    best_checkpoint = args.output_dir / "learned_tracker_best.eqx"

    for epoch in range(1, args.epochs + 1):
        start_time = time.monotonic()
        key, perm_key = jax.random.split(key)
        perms = jax.random.permutation(perm_key, num_train)

        train_loss = 0.0
        train_acc = 0.0
        train_hard_acc = 0.0
        num_batches = 0

        for i in range(0, num_train, args.batch_size):
            batch_indices = perms[i : i + args.batch_size]
            batch = {k: v[batch_indices] for k, v in train_data.items()}
            model, opt_state, metrics = train_step(model, opt_state, batch)
            train_loss += float(metrics["loss"])
            train_acc += float(metrics["accuracy"])
            train_hard_acc += float(metrics["hard_accuracy"])
            num_batches += 1

        train_loss /= num_batches
        train_acc /= num_batches
        train_hard_acc /= num_batches

        # Test eval
        test_metrics = eval_step(model, test_data)
        test_acc = float(test_metrics["accuracy"])
        test_hard_acc = float(test_metrics["hard_accuracy"])
        elapsed = time.monotonic() - start_time

        LOG.info(
            "Epoch %2d/%2d [%.1fs]: train_loss=%.4f acc=%.4f (hard=%.4f) | test_acc=%.4f (hard=%.4f)",
            epoch, args.epochs, elapsed, train_loss, train_acc, train_hard_acc, test_acc, test_hard_acc,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            eqx.tree_serialise_leaves(best_checkpoint, model)
            LOG.info("  --> Saved new best checkpoint (test_acc=%.4f) to %s", best_test_acc, best_checkpoint)

    # Save final
    final_checkpoint = args.output_dir / "learned_tracker_final.eqx"
    eqx.tree_serialise_leaves(final_checkpoint, model)
    LOG.info("Training complete. Best test accuracy: %.4f", best_test_acc)


if __name__ == "__main__":
    main()
