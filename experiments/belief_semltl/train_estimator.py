"""Train the LearnedStateEstimator on collected trajectory datasets.

Minimizes masked cross-entropy loss over true LDBA states.
Evaluates accuracy, probability assigned to true state, and hard baseline comparison on test split.
Saves the trained model weights to experiments/belief_semltl/models/.

Usage:
    MPLCONFIGDIR=/tmp JAX_PLATFORMS=cpu .pixi/envs/default/bin/python experiments/belief_semltl/train_estimator.py --smoke
    MPLCONFIGDIR=/tmp JAX_PLATFORMS=cpu .pixi/envs/default/bin/python experiments/belief_semltl/train_estimator.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.belief_semltl.learned_tracker import LearnedStateEstimator, mask_invalid_states

LOG = logging.getLogger("train_estimator")


def load_dataset(npz_path: Path):
    data = np.load(npz_path)
    return {
        "noisy_assignments": jnp.asarray(data["noisy_assignments"], dtype=jnp.int32),
        "prev_actions": jnp.asarray(data["prev_actions"], dtype=jnp.int32),
        "hard_states": jnp.asarray(data["hard_states"], dtype=jnp.int32),
        "true_states": jnp.asarray(data["true_states"], dtype=jnp.int32),
        "active_mask": jnp.asarray(data["active_mask"], dtype=jnp.bool_),
        "num_states": jnp.asarray(data["num_states"], dtype=jnp.int32),
    }


def compute_metrics(model: LearnedStateEstimator, batch: dict) -> dict[str, float]:
    y = batch["noisy_assignments"]
    u = batch["prev_actions"]
    q_hard = batch["hard_states"]
    q_true = batch["true_states"]
    mask = batch["active_mask"]  # (N, T)

    # vmap unroll over batch dimension
    logits = jax.vmap(model.unroll)(y, u, q_hard)  # (N, T, max_states)
    logits = mask_invalid_states(logits, batch["num_states"])
    preds = jnp.argmax(logits, axis=-1)  # (N, T)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    true_log_probs = jnp.take_along_axis(log_probs, q_true[..., None], axis=-1).squeeze(-1)
    probs = jnp.exp(true_log_probs)

    mask_f = mask.astype(jnp.float32)
    total_active = jnp.maximum(mask_f.sum(), 1.0)

    loss = -(true_log_probs * mask_f).sum() / total_active
    acc = ((preds == q_true) * mask_f).sum() / total_active
    hard_acc = ((q_hard == q_true) * mask_f).sum() / total_active
    mean_prob_true = (probs * mask_f).sum() / total_active

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "hard_accuracy": float(hard_acc),
        "prob_true": float(mean_prob_true),
    }


def loss_fn(model: LearnedStateEstimator, batch: dict):
    y = batch["noisy_assignments"]
    u = batch["prev_actions"]
    q_hard = batch["hard_states"]
    q_true = batch["true_states"]
    mask = batch["active_mask"]

    logits = jax.vmap(model.unroll)(y, u, q_hard)
    logits = mask_invalid_states(logits, batch["num_states"])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    true_log_probs = jnp.take_along_axis(log_probs, q_true[..., None], axis=-1).squeeze(-1)

    mask_f = mask.astype(jnp.float32)
    total_active = jnp.maximum(mask_f.sum(), 1.0)
    return -jnp.where(mask, true_log_probs, 0.0).sum() / total_active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 5 epochs on smoke data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None, help="New directory for checkpoints; existing checkpoints are never overwritten")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory with train.npz and test.npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    split_name = "smoke" if args.smoke else "full"
    data_dir = args.data_dir or (REPO_ROOT / "experiments" / "belief_semltl" / "data" / "trajectories" / split_name)
    train_data = load_dataset(data_dir / "train.npz")
    test_data = load_dataset(data_dir / "test.npz")

    out_model_dir = args.output_dir or (REPO_ROOT / "experiments" / "belief_semltl" / "models" / split_name)
    if (out_model_dir / "learned_estimator.eqx").exists() or (out_model_dir / "metrics.json").exists():
        raise FileExistsError(f"Model artifacts already exist in {out_model_dir}; use --output-dir for a new run")

    num_train = train_data["noisy_assignments"].shape[0]
    num_test = test_data["noisy_assignments"].shape[0]
    LOG.info("Loaded datasets: %d train trajectories, %d test trajectories", num_train, num_test)

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    model = LearnedStateEstimator(
        num_assignments=13,
        num_actions=5,
        max_states=14,
        embed_dim=16,
        hidden_dim=args.hidden_dim,
        key=init_key,
    )

    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    epochs = 5 if args.smoke else args.epochs
    batch_size = min(args.batch_size, num_train)

    best_test_acc = -1.0
    best_model = model

    LOG.info("Starting training for %d epochs (batch_size=%d)...", epochs, batch_size)
    for epoch in range(1, epochs + 1):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, num_train)
        epoch_losses = []

        for start_idx in range(0, num_train, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            batch = {k: v[batch_indices] for k, v in train_data.items()}
            model, opt_state, loss_val = train_step(model, opt_state, batch)
            epoch_losses.append(float(loss_val))

        train_metrics = compute_metrics(model, train_data)
        test_metrics = compute_metrics(model, test_data)

        LOG.info(
            "Epoch %02d | Train loss=%.4f acc=%.4f (hard=%.4f) | Test loss=%.4f acc=%.4f (hard=%.4f, P(true)=%.4f)",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["hard_accuracy"],
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["hard_accuracy"],
            test_metrics["prob_true"],
        )

        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = test_metrics["accuracy"]
            best_model = model

    out_model_dir = args.output_dir or (REPO_ROOT / "experiments" / "belief_semltl" / "models" / split_name)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_model_dir / "learned_estimator.eqx"
    eqx.tree_serialise_leaves(out_path, best_model)

    final_metrics = compute_metrics(best_model, test_data)
    with (out_model_dir / "metrics.json").open("x") as f:
        json.dump(final_metrics, f, indent=2)

    LOG.info("Best model saved to %s (Test acc=%.4f vs Hard acc=%.4f)", out_path, final_metrics["accuracy"], final_metrics["hard_accuracy"])


if __name__ == "__main__":
    main()
