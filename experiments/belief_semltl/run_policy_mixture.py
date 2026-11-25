"""Training-free SemLTL belief-weighted OUTPUT policy-mixture experiment.

The actor sees ONLY valid single-state embeddings, never their average. To
sample actions from the mixture, draw a latent state (or hard anchor) then
sample the frozen actor conditioned on that state. This is EXACT in law for
stochastic action distributions and needs only one actor forward pass per step.

JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_policy_mixture.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_policy_mixture.py

This is a stochastic policy experiment: the runner explicitly sets
cfg.eval.deterministic=False for ALL conditions. The hard baseline is
reevaluated under this same setting rather than copied from earlier CSVs.
Epsilon formula 5 is excluded. No ground-truth propositions enter the belief.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

import jaxolotl
from experiments.belief_semltl.mixture_stats import (
    assert_same_episodes,
    method_name,
    summarize,
    write_csv,
)
from experiments.belief_semltl.policy_mixture_agent import PolicyMixtureAgent
from experiments.belief_semltl.run_pilot import (
    DEFAULT_FORMULAS,
    eval_condition,
    select_model_seeds,
)
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.policy_mixture_wrapper import PolicyMixtureWrapper
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic, make_eval_fn

LOG = logging.getLogger("policy_mixture")
DEFAULT_NOISE = (0.0, 0.1, 0.3)
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 seeds x 2 formulas x 4 episodes; p=0,0.3")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=list(DEFAULT_NOISE))
    parser.add_argument("--formula-indices", nargs="+", type=int, default=list(DEFAULT_FORMULAS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.num_seeds = 2
        args.num_episodes = 4
        args.noise_levels = [0.0, 0.3]
        args.formula_indices = [0, 3]
        args.alphas = [0.0, 0.5, 1.0]
    args.noise_levels = sorted(set(args.noise_levels))
    args.alphas = sorted(set(args.alphas))
    if args.num_seeds < 1 or args.num_episodes < 1:
        parser.error("--num-seeds and --num-episodes must be positive")
    if not args.noise_levels or args.noise_levels[0] != 0.0 or any(
        not (0 <= p <= 1) for p in args.noise_levels
    ):
        parser.error("Always include p=0 and noise probabilities in [0,1]")
    if not args.alphas or args.alphas[0] != 0.0 or 1.0 not in args.alphas or any(
        not (0 <= a <= 1) for a in args.alphas
    ):
        parser.error("Always include alpha=0 and alpha=1, with alphas in [0,1]")
    if not args.formula_indices or len(args.formula_indices) != len(set(args.formula_indices)):
        parser.error("Provide a nonempty set of distinct formula indices")
    if 5 in args.formula_indices:
        parser.error("Formula 5 has epsilon transitions and is excluded")
    args.output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "results" / "policy_mixture"
        / ("smoke" if args.smoke else "full")
    )
    return args


def eval_mixture(eval_fn, base_agents, base_env, params, formulas, key,
                 p: float, alpha: float, expected_shape):
    wrapped_env = VectorizeWrapper(PolicyMixtureWrapper(
        base_env, noise_prob=p, overwrite_finite=True
    ))
    agents = PolicyMixtureAgent(base_agents, alpha=alpha)
    start = time.monotonic()
    values = eval_fn(agents, wrapped_env, params, formulas, key)
    if len(values) != 6:
        raise RuntimeError("Expected make_eval_fn(return_disagreement=True) to return 6 values")
    returns, _, lengths, violations, disagreements, _ = values
    jax.block_until_ready(returns)
    data = {
        "success": (np.asarray(returns) > 0).astype(np.float64),
        "returns": np.asarray(returns, dtype=np.float64),
        "lengths": np.asarray(lengths, dtype=np.float64),
        "violations": np.asarray(violations, dtype=np.float64),
        "disagreements": np.asarray(disagreements, dtype=np.float64),
    }
    from experiments.belief_semltl.pilot_stats import check_evaluation_arrays
    check_evaluation_arrays(
        data["returns"], data["violations"], data["lengths"],
        data["disagreements"], expected_shape,
    )
    LOG.info("p=%.2f alpha=%-4.2f success=%.4f violation=%.4f [%.1fs]",
             p, alpha, data["success"].mean(), data["violations"].mean(),
             time.monotonic() - start)
    return data


def plot_success(rows, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    methods = list(dict.fromkeys(r["method"] for r in rows if r["scope"] == "overall"))
    for method in methods:
        vals = sorted((r for r in rows if r["scope"] == "overall" and
                       r["metric"] == "success" and r["method"] == method),
                      key=lambda r: r["noise_prob"])
        if not vals:
            continue
        x = [v["noise_prob"] for v in vals]
        y = [v["mean"] for v in vals]
        line, = ax.plot(x, y, marker="o", label=method)
        if all(np.isfinite(v["ci_lower"]) and np.isfinite(v["ci_upper"]) for v in vals):
            ax.fill_between(
                x,
                [max(0, v["ci_lower"]) for v in vals],
                [min(1, v["ci_upper"]) for v in vals],
                color=line.get_color(), alpha=0.12,
            )
    ax.set(xlabel="Proposition corruption probability", ylabel="True task success rate",
           title="Frozen SemLTL: stochastic policy-output mixtures", ylim=(0, 1.03))
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mixture_success.png", dpi=220)
    fig.savefig(out_dir / "mixture_success.pdf")
    plt.close(fig)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
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
    # IMPORTANT: this method requires stochastic categorical sampling.
    # Force the *same* stochastic setting for hard and mixture conditions.
    OmegaConf.update(cfg, "eval.deterministic", False, force_add=True)
    assert not bool(cfg.eval.deterministic)

    original = list(cfg.formulas)
    if any(i < 0 or i >= len(original) for i in args.formula_indices):
        raise ValueError(f"Formula index outside [0,{len(original)-1}]")
    cfg.formulas = [original[i] for i in args.formula_indices]
    formula_strings = [str(f) for f in cfg.formulas]

    LOG.info("Preprocessing %d epsilon-free formulas", len(cfg.formulas))
    base_env, params = jaxolotl.make(
        cfg.env.name, reset_source="test", **cfg.get("env_params", {})
    )
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Compiled epsilon transitions detected. Exclude these formulas.")

    model_key = jax.random.split(jax.random.key(0))[1]
    models, seeds = load_batched_actor_critic(cfg, base_env, params, key=model_key)
    models, seeds = select_model_seeds(models, seeds, args.num_seeds)
    agents = hydra.utils.instantiate(cfg.alg.agent, models)
    expected_shape = (len(seeds), len(cfg.formulas), args.num_episodes)
    eval_fn = make_eval_fn(
        cfg, num_models=len(seeds), num_formulas=len(cfg.formulas),
        return_trajs=False, return_disagreement=True,
    )
    eval_key = jax.random.key(100)

    results = {}
    for p in args.noise_levels:
        baseline = eval_condition(
            eval_fn, agents, base_env, params, formulas, eval_key,
            "hard", p, expected_shape,
        )
        results[p] = {"hard": baseline}
        for alpha in args.alphas:
            name = method_name(alpha)
            data = eval_mixture(
                eval_fn, agents, base_env, params, formulas, eval_key,
                p, alpha, expected_shape,
            )
            results[p][name] = data
            if alpha == 0.0:
                assert_same_episodes(baseline, data, f"alpha=0 vs hard (p={p})")
                LOG.info("PASS alpha=0 matches hard, p=%.2f (all metrics)", p)
            if p == 0.0:
                assert_same_episodes(baseline, data, f"p=0 alpha={alpha} vs hard")
                LOG.info("PASS p=0 alpha=%.2f matches hard (all metrics)", alpha)

    episode_rows, summary_rows, effect_rows = summarize(
        results, seeds, args.formula_indices, formula_strings
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", episode_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_effects.csv", effect_rows)
    metadata = {
        "git_revision": git_revision(),
        "noise_probabilities": args.noise_levels,
        "alphas": args.alphas,
        "formula_indices": args.formula_indices,
        "model_seeds": [int(s) for s in seeds],
        "num_episodes_per_seed_formula_condition": args.num_episodes,
        "key": "jax.random.key(100) reused across conditions",
        "policy_training": "frozen pretrained; stochastic action selection",
        "aggregation": "hierarchical state-selection then original actor sampling; exact policy probability mixture in distribution",
        "belief_model": "uniform-independent categorical proposition prior; approximate, same as belief pilot",
        "information_boundary": "only corrupt propositions update belief; true states determine reward/termination only",
        "correctness_gates": "alpha=0 equals hard for every p, all alpha match hard at p=0",
        "epsilon_formula_5": "excluded",
        "multiplicity": "exploratory predefined alphas; no held-out LTL generalization",
        "note": "Action-distribution JS divergence not measured by this patch; hierarchical sampling does not expose per-state logits",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_plot:
        plot_success(summary_rows, args.output_dir)

    print("\nPOLICY-OUTPUT MIXTURE (stochastic frozen SemLTL):")
    cols = ["hard", *(method_name(a) for a in args.alphas)]
    print("  noise " + " ".join(f"{c:>11}" for c in cols))
    for p in args.noise_levels:
        print(f"  {p:5.2f} " + " ".join(
            f"{results[p][c]['success'].mean():11.4f}" for c in cols
        ))
    print(f"\nSaved: {args.output_dir}")
    print("Paired seed-level 95% CIs: paired_effects.csv; outcomes: episodes.csv")


if __name__ == "__main__":
    main()
