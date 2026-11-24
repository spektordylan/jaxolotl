"""Compare hard, MAP, and weighted frozen SemLTL policies in LetterWorld.

Run from the repository root:
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_pilot.py --smoke
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_pilot.py

The p=0 episode-level equality gate must pass before any nonzero-noise result
is written. Formula index 5 is excluded; no epsilon actions are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# The baseline runner uses the same repository-root import convention.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import hydra
import jax
import numpy as np
from hydra import compose, initialize

import jaxolotl
from experiments.belief_semltl.pilot_stats import (
    assert_zero_noise_parity,
    check_evaluation_arrays,
    paired_effect,
    seed_ci,
)
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.belief_semantic_ldba_wrapper import (
    BeliefSemanticLDBAWrapper,
)
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic, make_eval_fn

LOG = logging.getLogger("belief_pilot")
MODES = ("hard", "map", "weighted")
DEFAULT_FORMULAS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
DEFAULT_NOISE = (0.0, 0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 seeds x 2 formulas x 4 episodes; p=0,0.1")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=list(DEFAULT_NOISE))
    parser.add_argument("--formula-indices", nargs="+", type=int, default=list(DEFAULT_FORMULAS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip comparison plot")
    args = parser.parse_args()
    if args.smoke:
        args.num_seeds = 2
        args.num_episodes = 4
        args.noise_levels = [0.0, 0.1]
        args.formula_indices = [0, 3]
    args.noise_levels = sorted(set(args.noise_levels))
    if args.num_seeds < 1 or args.num_episodes < 1:
        parser.error("num-seeds and num-episodes must be positive")
    if not args.noise_levels or args.noise_levels[0] != 0.0:
        parser.error("Always include p=0 for the equivalence gate")
    if any(p < 0 or p > 1 for p in args.noise_levels):
        parser.error("Noise probabilities must be in [0,1]")
    if not args.formula_indices or len(set(args.formula_indices)) != len(args.formula_indices):
        parser.error("Provide nonempty, unique formula indices")
    if 5 in args.formula_indices:
        parser.error("Formula index 5 has epsilon transitions and is excluded from this pilot")
    args.output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "results" / "belief_pilot"
        / ("smoke" if args.smoke else "full")
    )
    return args


def build_wrapper(mode: str, env, p: float):
    if mode == "hard":
        return NoisySemanticLDBAWrapper(env, noise_prob=p, overwrite_finite=True)
    return BeliefSemanticLDBAWrapper(
        env, noise_prob=p, belief_mode=mode, overwrite_finite=True
    )


def select_model_seeds(models, seeds, count: int):
    """Subselect the leading model batch dimension if loader returns all seeds."""
    total = len(seeds)
    if total < count:
        raise ValueError(f"Requested {count} model seeds; loader returned only {total}")
    if total == count:
        return models, list(seeds)
    LOG.info("Loader returned %d model seeds; taking first %d for this run", total, count)

    def take_first(x):
        if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == total:
            return x[:count]
        return x

    return jax.tree.map(take_first, models), list(seeds[:count])


def eval_condition(eval_fn, agents, base_env, env_params, formulas, key, mode: str, p: float,
                   expected_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    wrapped = VectorizeWrapper(build_wrapper(mode, base_env, p))
    start = time.monotonic()
    values = eval_fn(agents, wrapped, env_params, formulas, key)
    if len(values) != 6:
        raise RuntimeError(
            "Expected make_eval_fn(... return_disagreement=True) to return 6 values; "
            f"received {len(values)}. Check src/jaxolotl/eval/utils.py."
        )
    returns, _, lengths, violations, disagreements, _ = values
    jax.block_until_ready(returns)
    result = {
        "success": (np.asarray(returns) > 0).astype(np.float64),
        "returns": np.asarray(returns, dtype=np.float64),
        "lengths": np.asarray(lengths, dtype=np.float64),
        "violations": np.asarray(violations, dtype=np.float64),
        "disagreements": np.asarray(disagreements, dtype=np.float64),
    }
    check_evaluation_arrays(
        result["returns"], result["violations"], result["lengths"],
        result["disagreements"], expected_shape,
    )
    LOG.info("p=%.2f mode=%-8s success=%.4f violation=%.4f [%.1fs]",
             p, mode, result["success"].mean(), result["violations"].mean(),
             time.monotonic() - start)
    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_results(results, seeds, formula_indices, formula_strings, output_dir: Path):
    """Write granular results, seed-level CIs, and paired success differences."""
    rows: list[dict] = []
    summary: list[dict] = []
    effects: list[dict] = []
    for p in sorted(results):
        hard_success = results[p]["hard"]["success"]
        for mode in MODES:
            data = results[p][mode]
            s_count, f_count, e_count = data["success"].shape
            for s in range(s_count):
                for f in range(f_count):
                    for ep in range(e_count):
                        rows.append({
                            "noise_prob": p,
                            "method": mode,
                            "model_seed": int(seeds[s]),
                            "formula_idx": formula_indices[f],
                            "formula": formula_strings[f],
                            "episode": ep,
                            "success": int(data["success"][s, f, ep]),
                            "return": float(data["returns"][s, f, ep]),
                            "violation": int(data["violations"][s, f, ep]),
                            "length": int(data["lengths"][s, f, ep]),
                            "disagreement": int(data["disagreements"][s, f, ep]),
                        })
            for metric, key in (("success", "success"), ("violation", "violations"),
                                ("disagreement", "disagreements")):
                arr = data[key]
                for scope, formula_pos in [("overall", None)] + [
                    ("formula", f) for f in range(f_count)
                ]:
                    values = (arr.mean(axis=(1, 2)) if formula_pos is None else
                              arr[:, formula_pos, :].mean(axis=1))
                    estimate = seed_ci(values)
                    summary.append({
                        "noise_prob": p, "method": mode, "scope": scope,
                        "formula_idx": -1 if formula_pos is None else formula_indices[formula_pos],
                        "metric": metric,
                        "mean": estimate.mean, "ci_lower": estimate.lower,
                        "ci_upper": estimate.upper, "num_seeds": estimate.n_seeds,
                        "num_formulas": f_count if formula_pos is None else 1,
                        "num_episodes": e_count,
                    })
            if mode != "hard":
                for scope, pos in [("overall", None)] + [
                    ("formula", f) for f in range(f_count)
                ]:
                    treatment = data["success"] if pos is None else data["success"][:, pos:pos + 1, :]
                    baseline = hard_success if pos is None else hard_success[:, pos:pos + 1, :]
                    est = paired_effect(treatment, baseline)
                    effects.append({
                        "noise_prob": p, "method": mode, "scope": scope,
                        "formula_idx": -1 if pos is None else formula_indices[pos],
                        "mean_delta": est.mean,
                        "ci_lower": est.lower, "ci_upper": est.upper,
                        "num_seeds": est.n_seeds,
                    })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "episodes.csv", rows, [
        "noise_prob", "method", "model_seed", "formula_idx", "formula", "episode",
        "success", "return", "violation", "length", "disagreement",
    ])
    write_csv(output_dir / "summary.csv", summary, [
        "noise_prob", "method", "scope", "formula_idx", "metric", "mean",
        "ci_lower", "ci_upper", "num_seeds", "num_formulas", "num_episodes",
    ])
    write_csv(output_dir / "paired_effects.csv", effects, [
        "noise_prob", "method", "scope", "formula_idx", "mean_delta",
        "ci_lower", "ci_upper", "num_seeds",
    ])
    return summary, effects


def plot_success(summary: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    for mode in MODES:
        rec = sorted((r for r in summary if r["method"] == mode and
                      r["scope"] == "overall" and r["metric"] == "success"),
                     key=lambda r: r["noise_prob"])
        x = np.array([r["noise_prob"] for r in rec])
        y = np.array([r["mean"] for r in rec])
        lo = np.array([r["ci_lower"] for r in rec])
        hi = np.array([r["ci_upper"] for r in rec])
        line, = ax.plot(x, y, marker="o", label=mode.upper() if mode == "map" else mode.title())
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1),
                            alpha=0.13, color=line.get_color())
    ax.set_xlabel("Categorical proposition corruption probability")
    ax.set_ylabel("True task success / acceptance rate")
    ax.set_title("Frozen SemLTL: hard vs. belief-aware task monitoring")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "success_comparison.png", dpi=220)
    fig.savefig(output_dir / "success_comparison.pdf")
    plt.close(fig)


def current_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with initialize(version_base="1.3", config_path="../../conf"):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            f"eval.num_episodes={args.num_episodes}",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
            f"+eval.num_seeds={args.num_seeds}",
        ])

    original_formulas = list(cfg.formulas)
    if any(i < 0 or i >= len(original_formulas) for i in args.formula_indices):
        raise ValueError(f"Requested formula index outside 0..{len(original_formulas)-1}")
    selected = [original_formulas[i] for i in args.formula_indices]
    cfg.formulas = selected  # Keep evaluator config and passed formula PyTree consistent.
    formula_strings = [str(f) for f in selected]

    LOG.info("Preparing LetterWorld and %d selected finite-LTL formulas", len(selected))
    base_env, env_params = jaxolotl.make(
        cfg.env.name, reset_source="test", **cfg.get("env_params", {})
    )
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)

    # Guard that no epsilon-enabled automaton entered this first pilot.
    epsilon_edges = np.asarray(formulas.epsilon_transitions)
    if np.any(epsilon_edges >= 0):
        raise ValueError(
            "Selected formulas contain epsilon transitions. This pilot supports "
            "only epsilon-free formulas; remove the affected indices."
        )

    key = jax.random.key(0)
    _, model_key = jax.random.split(key)
    models, seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    models, seeds = select_model_seeds(models, seeds, args.num_seeds)
    agents = hydra.utils.instantiate(cfg.alg.agent, models)
    num_models = len(seeds)
    num_formulas = len(selected)
    eval_fn = make_eval_fn(
        cfg, num_models=num_models, num_formulas=num_formulas,
        return_trajs=False, return_disagreement=True,
    )
    expected_shape = (num_models, num_formulas, args.num_episodes)

    # Match the original noise sweep's evaluation-key convention. The same
    # key is used for every method and noise probability; policy differences
    # can still induce divergent environment trajectories at p>0.
    eval_key = jax.random.key(100)
    results = {}
    for p in args.noise_levels:
        results[p] = {}
        for mode in MODES:
            results[p][mode] = eval_condition(
                eval_fn, agents, base_env, env_params, formulas, eval_key,
                mode, p, expected_shape,
            )
        if p == 0.0:
            for mode in ("map", "weighted"):
                assert_zero_noise_parity(results[p]["hard"], results[p][mode], mode)
            LOG.info("PASS p=0: hard, map, weighted have identical episode outcomes")

    summary, effects = export_results(
        results, seeds, args.formula_indices, formula_strings, args.output_dir
    )
    metadata = {
        "git_revision_at_run": current_revision(),
        "status": "exploratory frozen-policy pilot",
        "formula_indices": args.formula_indices,
        "formula_strings": formula_strings,
        "epsilon_formula_5_excluded": True,
        "methods": list(MODES),
        "noise_probabilities": args.noise_levels,
        "model_seeds": [int(s) for s in seeds],
        "num_episodes_per_seed_formula_condition": args.num_episodes,
        "evaluation_key": 100,
        "belief_model": "uniform independent prior on 13 true assignments per step",
        "zero_noise_episode_parity": "PASS",
        "confidence_intervals": "two-sided Student t over pretrained model seed means",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_plot:
        plot_success(summary, args.output_dir)

    print("\nFrozen-policy pilot (all reported success rates are exploratory):")
    print(f"{'p':>5}  {'hard':>8}  {'map':>8}  {'weighted':>8}  {'map-hard':>10}  {'weighted-hard':>14}")
    for p in args.noise_levels:
        def mean(mode):
            return results[p][mode]["success"].mean()
        print(f"{p:5.2f}  {mean('hard'):8.4f}  {mean('map'):8.4f}  "
              f"{mean('weighted'):8.4f}  {mean('map')-mean('hard'):+10.4f}  "
              f"{mean('weighted')-mean('hard'):+14.4f}")
    print(f"\nSaved: {args.output_dir}")
    print("Review paired_effects.csv for seed-paired uncertainty intervals.")


if __name__ == "__main__":
    main()
