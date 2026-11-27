"""Paired closed-loop evaluation: frozen DAgger GRU-MAP vs oracle-distilled residual.

Both treatments use the exact same trained DAgger estimator, original SemLTL
policy weights, run keys, and scoring. The only changed input is the semantic
embedding supplied to the actor. p=0 parity is a strict gate.
"""
from __future__ import annotations

import argparse
import json
import logging
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
from hydra import compose, initialize_config_dir

import jaxolotl
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic
from experiments.belief_semltl.collect_dagger import EXPERIMENT, sha256
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper
from experiments.belief_semltl.oracle_distill_adapter import (
    SemanticResidualAdapter, DistilledTrackerWrapper,
)
from experiments.belief_semltl.run_learned_estimator import (
    DEFAULT_FORMULAS, DEFAULT_NOISE, load_estimator, rollout_learned,
    assert_zero_noise_parity, save_csv, current_commit,
)
from experiments.belief_semltl.pilot_stats import seed_ci

LOG = logging.getLogger("oracle_distill_eval")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dagger-model", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--model-seeds", type=int, nargs="+", default=[7, 8, 9])
    p.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    p.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE))
    p.add_argument("--num-episodes", type=int, default=16)
    a = p.parse_args()
    split = "smoke" if a.smoke else "full"
    if a.smoke:
        a.model_seeds = [7, 8]
        a.formula_indices = [0, 3]
        a.noise_levels = [0.0, 0.30]
        a.num_episodes = 4
    a.dagger_model = a.dagger_model or EXPERIMENT / "models" / "dagger" / split / "learned_estimator.eqx"
    a.adapter = a.adapter or EXPERIMENT / "models" / "distillation" / split / "residual.eqx"
    a.output_dir = a.output_dir or EXPERIMENT / "results" / "distillation" / split
    if not a.model_seeds or len(set(a.model_seeds)) != len(a.model_seeds) or not set(a.model_seeds).issubset({7,8,9}):
        p.error("Use distinct evaluation policy seeds from 7..9")
    if not a.formula_indices or len(set(a.formula_indices)) != len(a.formula_indices) or 5 in a.formula_indices:
        p.error("Use distinct epsilon-free formula indices")
    if 0.0 not in a.noise_levels or any(not (0 <= q <= 1) for q in a.noise_levels):
        p.error("Noise levels must include p=0 and lie within [0,1]")
    if a.num_episodes < 1:
        p.error("num-episodes must be positive")
    return a


def make_summary(rows, noise_levels, seeds):
    summary = []
    for p in noise_levels:
        at_p = [r for r in rows if r["noise_prob"] == p]
        baseline = np.mean([r["baseline_success"] for r in at_p])
        distilled = np.mean([r["distilled_success"] for r in at_p])
        seed_effects = []
        for seed in seeds:
            ss = [r for r in at_p if r["model_seed"] == seed]
            if not ss:
                raise ValueError(f"Missing seed {seed} at p={p}")
            seed_effects.append(np.mean([
                r["distilled_success"] - r["baseline_success"] for r in ss
            ]))
        ci = seed_ci(np.asarray(seed_effects))
        summary.append({
            "noise_prob": p, "dagger_success": float(baseline),
            "distilled_success": float(distilled), "delta": float(ci.mean),
            "ci_lower": ci.lower, "ci_upper": ci.upper, "n_policy_seeds": ci.n_seeds,
        })
    return summary


def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if any((a.output_dir / name).exists() for name in ("episodes.csv", "paired_effects.csv", "metadata.json")):
        raise FileExistsError(f"Will not overwrite existing evaluation outputs: {a.output_dir}")
    for path in (a.dagger_model, a.adapter):
        if not path.is_file():
            raise FileNotFoundError(path)
    with initialize_config_dir(version_base="1.3", config_dir=str((ROOT / "conf").resolve())):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            f"eval.num_episodes={a.num_episodes}",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])
    originals = list(cfg.formulas)
    if any(i < 0 or i >= len(originals) for i in a.formula_indices):
        raise ValueError("Formula index out of range")
    cfg.formulas = [originals[i] for i in a.formula_indices]
    base_env, params = jaxolotl.make(cfg.env.name, reset_source="test", **cfg.get("env_params", {}))
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Unexpected epsilon-transition formula")
    estimator = load_estimator(a.dagger_model)
    # Adapter dimensions are derived from the exact pretrained task encodings.
    meta_path = a.adapter.parent / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing adapter metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dagger_model_sha256") != sha256(a.dagger_model):
        raise ValueError("Adapter was trained on a different DAgger estimator")
    semantic_dim = int(formulas.embeddings.shape[-1])
    if semantic_dim != int(meta["semantic_dim"]):
        raise ValueError("Semantic embedding dimension changed since training")
    adapter_template = SemanticResidualAdapter(
        int(meta["hidden_dim"]), semantic_dim,
        key=jax.random.PRNGKey(int(meta["seed"])),
        max_delta=float(meta["max_delta"]),
    )
    adapter = eqx.tree_deserialise_leaves(a.adapter, adapter_template)

    key = jax.random.key(0)
    _, model_key = jax.random.split(key)
    models, all_seeds = load_batched_actor_critic(cfg, base_env, params, key=model_key)
    all_seeds = list(map(int, all_seeds))
    if not set(a.model_seeds).issubset(all_seeds):
        raise ValueError(f"Missing evaluation policy models. Available: {all_seeds}")
    rows = []
    for seed in a.model_seeds:
        i = all_seeds.index(seed)
        single = jax.tree.map(
            lambda x: x[i] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == len(all_seeds) else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, single)
        for f, formula_index in enumerate(a.formula_indices):
            task = jax.tree.map(lambda x: x[f], formulas)
            for noise in a.noise_levels:
                run_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(200), seed), formula_index,
                )
                run_key = jax.random.fold_in(run_key, int(round(noise * 1_000_000)))
                baseline_env = VectorizeWrapper(LearnedTrackerWrapper(
                    base_env, estimator=estimator, noise_prob=noise, overwrite_finite=True,
                ))
                distilled_env = VectorizeWrapper(DistilledTrackerWrapper(
                    base_env, estimator=estimator, adapter=adapter,
                    noise_prob=noise, overwrite_finite=True,
                ))
                base_ep, _ = rollout_learned(
                    agent, baseline_env, params, task, run_key, a.num_episodes,
                    int(params.max_steps_in_episode), bool(cfg.eval.deterministic),
                )
                dist_ep, _ = rollout_learned(
                    agent, distilled_env, params, task, run_key, a.num_episodes,
                    int(params.max_steps_in_episode), bool(cfg.eval.deterministic),
                )
                base_ep, dist_ep = jax.tree.map(np.asarray, (base_ep, dist_ep))
                if not np.all(base_ep.completed) or not np.all(dist_ep.completed):
                    raise RuntimeError("An episode exceeded max_steps before termination")
                if noise == 0.0:
                    assert_zero_noise_parity(base_ep, dist_ep, seed=seed, formula_idx=formula_index)
                for e in range(a.num_episodes):
                    rows.append({
                        "model_seed": seed, "formula_idx": formula_index,
                        "noise_prob": noise, "episode": e,
                        "baseline_success": int(base_ep.returns[e] > 0),
                        "distilled_success": int(dist_ep.returns[e] > 0),
                        "baseline_violation": int(base_ep.violations[e]),
                        "distilled_violation": int(dist_ep.violations[e]),
                        "baseline_steps": int(base_ep.steps[e]),
                        "distilled_steps": int(dist_ep.steps[e]),
                        "baseline_state_accuracy": float(base_ep.learned_correct_steps[e] / max(base_ep.steps[e], 1)),
                        "distilled_state_accuracy": float(dist_ep.learned_correct_steps[e] / max(dist_ep.steps[e], 1)),
                    })
                LOG.info("seed=%d formula=%d p=%.2f DAgger=%.3f distilled=%.3f",
                         seed, formula_index, noise,
                         np.mean(base_ep.returns > 0), np.mean(dist_ep.returns > 0))
    summary = make_summary(rows, a.noise_levels, a.model_seeds)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(a.output_dir / "episodes.csv", rows)
    save_csv(a.output_dir / "paired_effects.csv", summary)
    with (a.output_dir / "metadata.json").open("x", encoding="utf-8") as f:
        json.dump({
            "method": "paired GRU+DAgger vs frozen-actor oracle distilled residual",
            "dagger_model": str(a.dagger_model), "dagger_model_sha256": sha256(a.dagger_model),
            "adapter": str(a.adapter), "adapter_sha256": sha256(a.adapter),
            "git_revision": current_commit(), "noise_levels": a.noise_levels,
            "formula_indices": a.formula_indices, "model_seeds": a.model_seeds,
            "episodes_per_condition": a.num_episodes,
            "run_key": "200 folded with seed, formula_idx, round(p*1e6)",
            "limitations": "3 evaluation policy seeds; original GRU checkpoint selected on 7..9; exploratory CI",
        }, f, indent=2)
    print("\nDAgger GRU-MAP vs ORACLE-DISTILLED RESIDUAL (paired seed-level means):")
    print(f"{'noise':>7} {'DAgger':>10} {'distilled':>10} {'delta':>10} {'95% CI (pp)':>26}")
    for r in summary:
        print(f"{r['noise_prob']:7.2f} {r['dagger_success']:10.4f} "
              f"{r['distilled_success']:10.4f} {r['delta']:+10.4f} "
              f"[{r['ci_lower']*100:+.2f}, {r['ci_upper']*100:+.2f}]")
    print("Saved to:", a.output_dir)


if __name__ == "__main__":
    main()
