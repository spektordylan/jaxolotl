"""Evaluate the frozen learned tracker on exactly the same hard-policy trajectories.

By default this reads the recorded test.npz and reports state accuracy/NLL.
The supplied original dataset predates per-episode noise metadata; use
--legacy-full-order ONLY for the unmodified dataset collected by the archived
collect_trajectories.py loop (seeds 7..9, formulas 0,1,2,3,4,6,7,8,9,
noise 0,.1,.3, 32 episodes/condition). That dataset was used to SELECT the
supplied checkpoint, so results on it are VALIDATION diagnostics, not final
held-out test estimates.

    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/offline_tracker_diagnostic.py --legacy-full-order
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_semltl.learned_tracker import (
    LearnedStateEstimator, mask_invalid_states,
)

ROOT = REPO_ROOT / "experiments" / "belief_semltl"


def legacy_full_noise_labels(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Explicit provenance-based recovery. Never guess noise from correctness."""
    n = len(dataset["num_states"])
    num_seeds, num_formulas, num_noise = 3, 9, 3
    total_conditions = num_seeds * num_formulas * num_noise
    if n != 2592:  # original test: 3*9*3*32
        raise ValueError(f"Legacy layout expects 2592 trajectories; received {n}")
    levels = np.tile(np.repeat([0.0, 0.1, 0.3], 32), num_seeds * num_formulas)
    active = dataset["active_mask"]
    hard_acc_zero = np.sum((dataset["hard_states"][levels == 0] ==
                            dataset["true_states"][levels == 0]) & active[levels == 0]) / active[levels == 0].sum()
    if hard_acc_zero != 1.0:
        raise ValueError("Legacy ordering failed p=0 parity check; do not use inferred noise labels")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/trajectories/full/test.npz")
    parser.add_argument("--model", type=Path, default=ROOT / "models/full/learned_estimator.eqx")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--legacy-full-order", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    with np.load(args.dataset) as archive:
        data = {name: archive[name] for name in archive.files}
    if "noise_probs" in data:
        noise = data["noise_probs"]
    elif args.legacy_full_order:
        noise = legacy_full_noise_labels(data)
    else:
        noise = None
        print("No per-episode noise metadata. Add --legacy-full-order only for the archived full/test.npz")

    template = LearnedStateEstimator(hidden_dim=args.hidden_dim, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.model, template)

    # Compute in batches to bound CPU RAM. This matches the teacher-forced
    # history used for training: the first GRU update is on physical step 1.
    n = len(data["num_states"])
    predictions = np.empty_like(data["true_states"])
    for start in range(0, n, args.batch_size):
        stop = min(n, start + args.batch_size)
        logits = jax.vmap(model.unroll)(
            jnp.asarray(data["noisy_assignments"][start:stop]),
            jnp.asarray(data["prev_actions"][start:stop]),
            jnp.asarray(data["hard_states"][start:stop]),
        )
        logits = mask_invalid_states(logits, jnp.asarray(data["num_states"][start:stop]))
        predictions[start:stop] = np.asarray(jnp.argmax(logits, axis=-1))

    mask = data["active_mask"]
    hard = data["hard_states"]
    true = data["true_states"]
    def summary(label, rows):
        active = mask[rows]
        total = int(active.sum())
        if total == 0:
            raise ValueError(f"Empty stratum {label}")
        hard_acc = float(((hard[rows] == true[rows]) & active).sum()/total)
        learned_acc = float(((predictions[rows] == true[rows]) & active).sum()/total)
        print(f"{label:>16} | active steps {total:7d} | hard {hard_acc:7.2%} | raw GRU {learned_acc:7.2%} | delta {learned_acc-hard_acc:+7.2%}")
        return hard_acc, learned_acc

    print("Teacher-forced *raw-estimator* diagnostics (no p=0 hard passthrough):")
    summary("all", np.ones(n, dtype=bool))
    if noise is not None:
        for p in sorted(set(map(float, noise))):
            h, m = summary(f"p={p:.2f}", np.isclose(noise, p))
            if np.isclose(p, .3):
                print("p=.30 same-trajectory inference gate:", "PASS" if m > h else "FAIL")
    print("Note: online p=0 parity is enforced by bypassing the tracker, not by raw-GRU accuracy.")
    print("The archived test.npz was used to select the provided checkpoint; this is validation, not independent test.")


if __name__ == "__main__":
    main()
