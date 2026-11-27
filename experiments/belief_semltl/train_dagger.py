"""One fixed-budget supervised fine-tuning pass on old + GRU-MAP DAgger trajectories.

No checkpoint selection on archived test.npz, which already influenced original
checkpoint selection. Evaluation is performed separately on policy seeds 7..9.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.belief_semltl.collect_dagger import EXPERIMENT, sha256
from experiments.belief_semltl.run_learned_estimator import MODEL_PATH, load_estimator
from experiments.belief_semltl.train_estimator import load_dataset, loss_fn

LOG = logging.getLogger("dagger_train")
DATA_KEYS = ("noisy_assignments", "prev_actions", "hard_states", "true_states", "active_mask", "num_states")


def merge_datasets(original: dict, on_policy: dict) -> tuple[dict, int, int]:
    """Concatenate complete episodes, preserving masking and sequence boundaries."""
    old_n = int(original["active_mask"].shape[0])
    new_n = int(on_policy["active_mask"].shape[0])
    if old_n == 0 or new_n == 0:
        raise ValueError("Both original and on-policy datasets must be nonempty")
    merged = {}
    for name in DATA_KEYS:
        a, b = np.asarray(original[name]), np.asarray(on_policy[name])
        if a.ndim != b.ndim or a.shape[1:] != b.shape[1:]:
            raise ValueError(f"Incompatible shapes for {name}: {a.shape}, {b.shape}")
        merged[name] = jnp.asarray(np.concatenate((a, b), axis=0))
    if not np.all(np.asarray(merged["active_mask"]).any(axis=1)):
        raise ValueError("Episode with no active transitions")
    if np.max(np.asarray(merged["num_states"])) > 14:
        raise ValueError("Checkpoint supports at most 14 LDBA states")
    return merged, old_n, new_n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--original-model", type=Path, default=MODEL_PATH)
    p.add_argument("--original-data", type=Path)
    p.add_argument("--dagger-data", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--epochs", type=int, default=3, help="Fixed number of epochs, no test-set model selection")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=102)
    args = p.parse_args()
    split = "smoke" if args.smoke else "full"
    args.original_data = args.original_data or EXPERIMENT / "data" / "trajectories" / split / "train.npz"
    args.dagger_data = args.dagger_data or EXPERIMENT / "data" / "dagger" / split / "train.npz"
    args.output_dir = args.output_dir or EXPERIMENT / "models" / "dagger" / split
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 0.01:
        p.error("epochs, batch-size must be positive; lr in (0, 0.01)")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    target = args.output_dir / "learned_estimator.eqx"
    meta_target = args.output_dir / "metadata.json"
    if target.exists() or meta_target.exists():
        raise FileExistsError(f"Will not overwrite checkpoints in {args.output_dir}")
    for p in (args.original_model, args.original_data, args.dagger_data):
        if not p.is_file():
            raise FileNotFoundError(p)
    with np.load(args.dagger_data) as contents:
        if "model_seeds" in contents:
            seeds = set(int(x) for x in np.unique(contents["model_seeds"]))
            if not seeds.issubset(set(range(7))):
                raise ValueError(f"DAgger dataset contains evaluation policy seeds: {seeds}")
    provenance_path = args.dagger_data.parent / "metadata.json"
    if provenance_path.is_file():
        with provenance_path.open(encoding="utf-8") as file:
            provenance = json.load(file)
        if provenance.get("original_model_sha256") != sha256(args.original_model):
            raise ValueError("On-policy data was collected under a different GRU checkpoint")
    old_data = load_dataset(args.original_data)
    dagger_data = load_dataset(args.dagger_data)
    train_data, old_n, new_n = merge_datasets(old_data, dagger_data)
    LOG.info("Training on %d old + %d on-policy trajectories (%.1f%% new)",
             old_n, new_n, 100 * new_n / (old_n + new_n))

    model = load_estimator(args.original_model)
    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    key = jax.random.PRNGKey(args.seed)
    num = old_n + new_n
    history = []
    for epoch in range(1, args.epochs + 1):
        key, perm_key = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(perm_key, num))
        batch_losses = []
        for start in range(0, num, args.batch_size):
            indices = perm[start:start + args.batch_size]
            batch = {k: v[indices] for k, v in train_data.items()}
            model, opt_state, loss = train_step(model, opt_state, batch)
            batch_losses.append((float(loss), len(indices)))
        average = sum(loss * n for loss, n in batch_losses) / num
        history.append({"epoch": epoch, "train_loss": average})
        LOG.info("epoch %d/%d: aggregated training loss %.4f", epoch, args.epochs, average)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(target, model)
    metadata = {
        "method": "one DAgger data aggregation iteration; fixed-epoch finetuning",
        "original_model": str(args.original_model),
        "original_model_sha256": sha256(args.original_model),
        "original_data": str(args.original_data), "original_data_sha256": sha256(args.original_data),
        "dagger_data": str(args.dagger_data), "dagger_data_sha256": sha256(args.dagger_data),
        "old_trajectories": old_n, "on_policy_trajectories": new_n,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "optimizer": "new AdamW state; frozen actor unchanged; all GRU weights fine-tuned",
        "checkpoint_selection": "fixed final epoch; no evaluation-set selection",
        "seed": args.seed, "train_loss_history": history,
    }
    with meta_target.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    LOG.info("Saved new checkpoint to %s (original unchanged)", target)


if __name__ == "__main__":
    main()
