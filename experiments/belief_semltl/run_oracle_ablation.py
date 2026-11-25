"""Frozen SemLTL oracle task-monitor ablation; not a deployable policy.

The oracle changes ONLY the semantic task-state information consumed by the
same pretrained policy: it supplies true LDBA state rather than noisy state.
Ground-truth transitions, reward, termination, physical LetterWorld and policy
weights are otherwise identical to the hard baseline. Epsilon formula 5 is
excluded, as in the original 9-formula frozen-policy pilot.

From the repository root:
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_oracle_ablation.py --smoke
    JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_oracle_ablation.py

Gates: p=0 oracle == hard for every episode and metric; oracle == its p=0
outcomes for every nonzero noise level; oracle disagreement is identically 0.
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

import jaxolotl
from experiments.belief_semltl.oracle_stats import (
    assert_equal_episodes,
    assert_oracle_disagreement_zero,
    summarize,
    write_csv,
)
from experiments.belief_semltl.pilot_stats import check_evaluation_arrays
from experiments.belief_semltl.run_pilot import (
    DEFAULT_FORMULAS,
    eval_condition,
    select_model_seeds,
)
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.oracle_semantic_ldba_wrapper import (
    OracleSemanticLDBAWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic, make_eval_fn

LOG = logging.getLogger("oracle_ablation")
DEFAULT_NOISE = (0.0, 0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 seeds, 2 formulas, 4 episodes, p=0 and p=0.30")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE))
    parser.add_argument("--formula-indices", type=int, nargs="+", default=list(DEFAULT_FORMULAS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.num_seeds, args.num_episodes = 2, 4
        args.formula_indices = [0, 3]
        args.noise_levels = [0.0, 0.3]
    if args.num_seeds < 1 or args.num_episodes < 1:
        parser.error("--num-seeds and --num-episodes must be positive")
    if not args.formula_indices or len(args.formula_indices) != len(set(args.formula_indices)):
        parser.error("Select nonempty, unique formula indices")
    if 5 in args.formula_indices:
        parser.error("Formula index 5 contains epsilon transitions; exclude it")
    args.noise_levels = sorted(set(args.noise_levels))
    if not args.noise_levels or args.noise_levels[0] != 0.0 or any(
        not (0.0 <= p <= 1.0) for p in args.noise_levels
    ):
        parser.error("Include p=0 and noise levels in [0,1]")
    args.output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "belief_semltl" / "results" / "oracle_ablation"
        / ("smoke" if args.smoke else "full")
    )
    return args


def eval_oracle(eval_fn, agents, base_env, params, formulas, key, p, expected_shape):
    env = VectorizeWrapper(
        OracleSemanticLDBAWrapper(base_env, noise_prob=p, overwrite_finite=True)
    )
    start = time.monotonic()
    values = eval_fn(agents, env, params, formulas, key)
    if len(values) != 6:
        raise RuntimeError("Expected six evaluator outputs with return_disagreement=True")
    returns, _, lengths, violations, disagreements, _ = values
    jax.block_until_ready(returns)
    data = {
        "success": (np.asarray(returns) > 0).astype(np.float64),
        "returns": np.asarray(returns, dtype=np.float64),
        "lengths": np.asarray(lengths, dtype=np.float64),
        "violations": np.asarray(violations, dtype=np.float64),
        "disagreements": np.asarray(disagreements, dtype=np.float64),
    }
    check_evaluation_arrays(
        data["returns"], data["violations"], data["lengths"],
        data["disagreements"], expected_shape,
    )
    assert_oracle_disagreement_zero(data)
    LOG.info("p=%.2f mode=oracle   success=%.4f violation=%.4f [%.1fs]",
             p, data["success"].mean(), data["violations"].mean(),
             time.monotonic() - start)
    return data


def maybe_plot(summary: list[dict], output_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for mode in ("hard", "oracle"):
        rows = sorted(
            (r for r in summary if r["scope"] == "overall" and
             r["method"] == mode and r["metric"] == "success"),
            key=lambda r: r["noise_prob"],
        )
        x = np.asarray([r["noise_prob"] for r in rows])
        y = np.asarray([r["mean"] for r in rows])
        lower = np.asarray([r["ci_lower"] for r in rows])
        upper = np.asarray([r["ci_upper"] for r in rows])
        line, = ax.plot(x, y, marker="o", label="Oracle true-state" if mode == "oracle" else "Hard noisy-state")
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            ax.fill_between(x, np.clip(lower, 0, 1), np.clip(upper, 0, 1),
                            alpha=0.14, color=line.get_color())
    ax.set(
        xlabel="Categorical proposition corruption probability",
        ylabel="True task success rate",
        title="Frozen SemLTL: oracle-monitor ablation (privileged reference)",
        ylim=(0, 1.02),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_vs_hard_success.png", dpi=220)
    fig.savefig(output_dir / "oracle_vs_hard_success.pdf")
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
    all_formulas = list(cfg.formulas)
    if any(i < 0 or i >= len(all_formulas) for i in args.formula_indices):
        raise ValueError(f"Formula index outside 0..{len(all_formulas)-1}")
    cfg.formulas = [all_formulas[i] for i in args.formula_indices]
    formula_strings = [str(f) for f in cfg.formulas]
    LOG.info("Preprocessing %d epsilon-free LTL formulas", len(cfg.formulas))
    base_env, params = jaxolotl.make(
        cfg.env.name, reset_source="test", **cfg.get("env_params", {})
    )
    base_env = TimeLimitWrapper(base_env)
    formulas = preprocess_formulas(cfg.formulas, base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Epsilon transitions detected; use epsilon-free formula subset")

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
        results[p] = {
            "hard": eval_condition(
                eval_fn, agents, base_env, params, formulas, eval_key,
                "hard", p, expected_shape,
            ),
            "oracle": eval_oracle(
                eval_fn, agents, base_env, params, formulas, eval_key,
                p, expected_shape,
            ),
        }
        if p == 0.0:
            assert_equal_episodes(results[p]["hard"], results[p]["oracle"], "p=0 hard/oracle parity")
            LOG.info("PASS p=0 hard/oracle episode parity (all five metrics)")
        else:
            assert_equal_episodes(
                results[0.0]["oracle"], results[p]["oracle"],
                f"oracle invariance p=0 vs p={p}",
            )
            LOG.info("PASS oracle noise invariance at p=%.2f (all five metrics)", p)

    episode_rows, summary_rows, effect_rows = summarize(
        results, seeds, args.formula_indices,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", episode_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_effects.csv", effect_rows)
    metadata = {
        "git_revision_at_run": git_revision(),
        "status": "privileged oracle diagnostic; NOT a deployable monitor or learned policy",
        "formula_indices": args.formula_indices,
        "formula_strings": formula_strings,
        "excluded_epsilon_formula_index": 5,
        "model_seeds": [int(s) for s in seeds],
        "episodes_per_seed_formula_condition": args.num_episodes,
        "noise_probabilities": args.noise_levels,
        "methods": ["hard", "oracle"],
        "oracle_intervention": "replace only policy-facing semantic embedding and epsilon features with ground-truth LDBA state; retain noisy wrapper reward, termination and physical dynamics",
        "oracle_privileged_information": "true LDBA state and true proposition assignment used in policy-facing monitor",
        "noise_model": "symmetric categorical proposition label corruption sampled by NoisySemanticLDBAWrapper",
        "model_weights": "frozen pretrained SemLTL checkpoints, identical for both conditions",
        "configuration": "alg=sem_ltl, env=letter_world, task_set=finite, run=pretrained, reset_source=test",
        "eval_key": 100,
        "matched_conditions": "same model seeds/formulas/episodes/reset and action sampling RNG keys; trajectories may diverge after action choices differ",
        "zero_noise_hard_oracle_parity": "PASS",
        "oracle_noise_invariance": "PASS",
        "oracle_monitor_disagreement": "identically zero by construction; latent unused noisy monitor may disagree but is not an oracle error",
        "uncertainty": "two-sided Student t confidence interval across 10 (or selected) pretrained seeds; fixed task set",
        "limitations": [
            "oracle is privileged and inaccessible in a realistic noisy-observation deployment",
            "oracle is an informative reference for the existing frozen network, not a mathematical upper bound on all controllers",
            "same random schedule does not imply identical post-intervention trajectories",
            "no epsilon-containing specifications evaluated",
            "oracle does not train or adapt the network; only semantic monitoring is modified",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_plot:
        maybe_plot(summary_rows, args.output_dir)

    print("\nORACLE-MONITOR ABLATION (privileged state, frozen policy):")
    print(f"{'p':>5} {'hard':>9} {'oracle':>9} {'gain pp':>10} {'gain 95% CI':>22}")
    for p in args.noise_levels:
        est = next(e for e in effect_rows if e["noise_prob"] == p and e["scope"] == "overall")
        h = float(np.mean(results[p]["hard"]["success"]))
        o = float(np.mean(results[p]["oracle"]["success"]))
        print(f"{p:5.2f} {h:9.4f} {o:9.4f} {100*(o-h):+10.2f} "
              f"[{est['ci_lower']*100:+.2f}, {est['ci_upper']*100:+.2f}] pp")
    print(f"\nSaved: {args.output_dir}")
    print("Interpret gain as privileged-monitor headroom, NOT as feasible policy improvement.")


if __name__ == "__main__":
    main()
