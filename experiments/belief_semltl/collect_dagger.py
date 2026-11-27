"""One-round DAgger data: collect labeled trajectories under the frozen GRU-MAP policy.

The policy receives only estimated-state embeddings. True states are copied into
supervised labels *after* environment transitions; never into the policy or GRU.
Existing trajectory datasets, checkpoints and evaluation outputs are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
from hydra import compose, initialize_config_dir
import hydra
import jax
import jax.numpy as jnp
import jaxolotl
import numpy as np
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic

from experiments.belief_semltl.collect_trajectories import rollout_collector
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper
from experiments.belief_semltl.run_learned_estimator import DEFAULT_FORMULAS, MODEL_PATH, load_estimator

LOG = logging.getLogger("dagger_collect")
EXPERIMENT = ROOT / "experiments" / "belief_semltl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def traces_to_arrays(traces):
    """Preserve exact post-transition order expected by LearnedStateEstimator.unroll."""
    return {
        "noisy_assignments": np.asarray(traces.noisy_assignment).T,
        "prev_actions": np.asarray(traces.prev_action).T,
        "hard_states": np.asarray(traces.hard_state).T,
        "true_states": np.asarray(traces.true_state).T,
        "active_mask": np.asarray(traces.active).T,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=16, help="Episodes per seed/formula/noise condition")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.10, 0.30])
    args = parser.parse_args()
    if args.smoke:
        args.episodes = 4
        args.model_seeds = [0, 1]
        args.formula_indices = [0, 3]
        args.noise_levels = [0.30]
    if not args.model_seeds or len(set(args.model_seeds)) != len(args.model_seeds) or not set(args.model_seeds).issubset(set(range(7))):
        parser.error("DAgger training must use unique policy seeds from 0..6 only")
    if not args.formula_indices or len(set(args.formula_indices)) != len(args.formula_indices) or 5 in args.formula_indices:
        parser.error("Use distinct epsilon-free formula indices (formula 5 excluded)")
    if args.episodes < 1 or not args.noise_levels or any(not 0 < p <= 1 for p in args.noise_levels):
        parser.error("Require positive episodes and nonzero noise probabilities in (0,1]")
    if len(set(args.noise_levels)) != len(args.noise_levels):
        parser.error("Noise probabilities must be distinct")
    args.output_dir = args.output_dir or (EXPERIMENT / "data" / "dagger" / ("smoke" if args.smoke else "full"))
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    for name in ("train.npz", "metadata.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"Will not overwrite {args.output_dir / name}")

    with initialize_config_dir(version_base="1.3", config_dir=str((ROOT / "conf").resolve())):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])
    all_formula_strings = list(cfg.formulas)
    if any(i < 0 or i >= len(all_formula_strings) for i in args.formula_indices):
        raise ValueError("Formula index out of bounds")
    cfg.formulas = [all_formula_strings[i] for i in args.formula_indices]

    base_env, params = jaxolotl.make(cfg.env.name, reset_source="test", **cfg.get("env_params", {}))
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("DAgger collector currently supports epsilon-free formulas only")
    estimator = load_estimator(args.model_path)
    models, all_seeds = load_batched_actor_critic(cfg, base_env, params, key=jax.random.PRNGKey(42))
    all_seeds = [int(x) for x in all_seeds]
    if not set(args.model_seeds).issubset(set(all_seeds)):
        raise ValueError(f"Missing policy seed(s). Available seeds: {all_seeds}")
    n_all = len(all_seeds)

    accumulator = {k: [] for k in (
        "noisy_assignments", "prev_actions", "hard_states", "true_states", "active_mask",
        "num_states", "model_seeds", "formula_indices", "noise_probs",
    )}
    for model_seed in args.model_seeds:
        index = all_seeds.index(model_seed)
        single_model = jax.tree.map(
            lambda x: x[index] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_all else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, single_model)
        for f, f_idx in enumerate(args.formula_indices):
            task = jax.tree.map(lambda x: x[f], formulas)
            count_states = int(np.asarray(formulas.num_states[f]))
            for noise in args.noise_levels:
                run_key = jax.random.fold_in(jax.random.key(4100), model_seed)
                run_key = jax.random.fold_in(run_key, f_idx)
                run_key = jax.random.fold_in(run_key, int(round(noise * 1_000_000)))
                env = VectorizeWrapper(
                    LearnedTrackerWrapper(base_env, estimator=estimator,
                                          noise_prob=noise, overwrite_finite=True)
                )
                traces = rollout_collector(
                    agent, env, base_env, params, task, run_key,
                    num_episodes=args.episodes, max_steps=int(params.max_steps_in_episode),
                    deterministic=bool(cfg.eval.deterministic),
                )
                batch = traces_to_arrays(traces)
                if not np.all(batch["active_mask"].any(axis=1)):
                    raise RuntimeError("An episode contains no active transitions")
                for name, arr in batch.items():
                    accumulator[name].append(arr)
                n = args.episodes
                accumulator["num_states"].append(np.full(n, count_states, np.int32))
                accumulator["model_seeds"].append(np.full(n, model_seed, np.int32))
                accumulator["formula_indices"].append(np.full(n, f_idx, np.int32))
                accumulator["noise_probs"].append(np.full(n, noise, np.float32))
                LOG.info("collected seed=%d formula=%d noise=%.2f episodes=%d", model_seed, f_idx, noise, n)

    data = {k: np.concatenate(v, axis=0) for k, v in accumulator.items()}
    n = data["active_mask"].shape[0]
    for key in ("noisy_assignments", "prev_actions", "hard_states", "true_states"):
        if data[key].shape != data["active_mask"].shape:
            raise RuntimeError(f"Mismatch between {key} and active_mask")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation: no silent clobber of archived data.
    with (args.output_dir / "train.npz").open("xb") as file:
        np.savez_compressed(file, **data)
    metadata = {
        "method": "one-round DAgger data collected under original GRU-MAP policy",
        "original_model": str(args.model_path), "original_model_sha256": sha256(args.model_path),
        "model_seeds": args.model_seeds, "formula_indices": args.formula_indices,
        "noise_levels": args.noise_levels, "episodes_per_condition": args.episodes,
        "num_trajectories": n, "active_transitions": int(data["active_mask"].sum()),
        "random_key_base": 4100, "deterministic_policy": bool(cfg.eval.deterministic),
        "true_state_usage": "label after transition only; no privileged model inputs",
        "train_only": "evaluation seeds 7..9 not used for gradient updates",
    }
    with (args.output_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    LOG.info("Saved %d DAgger trajectories to %s", n, args.output_dir / "train.npz")


if __name__ == "__main__":
    main()
