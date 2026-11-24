"""Run comprehensive evaluation sweep of frozen SemLTL policies across noise levels.

Evaluates 10 pretrained model checkpoints across 10 finite-LTL evaluation tasks
in LetterWorld for noise levels p in {0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30}.
Records granular episode-level outcomes and aggregated confidence intervals.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

# Ensure repo root and experiment package are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig

import jaxolotl
from experiments.belief_semltl.plot_results import generate_all_figures
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.alg.sem_ltl.wrappers.noisy_semantic_ldba_wrapper import (
    NoisySemanticLDBAWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_batched_actor_critic, make_eval_fn
from jaxolotl.utils.stats import bootstrap_task_set, student_t_task_set

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("noise_sweep")

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def parse_args():
    parser = argparse.ArgumentParser(description="SemLTL Observation Noise Sweep")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=16,
        help="Number of evaluation episodes per model seed and formula (default: 16)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of pretrained model seeds to evaluate (default: 10)",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=NOISE_LEVELS,
        help="List of proposition corruption probabilities",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/belief_semltl/results",
        help="Directory to save raw and summary CSVs",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate publication figures immediately after sweep",
    )
    return parser.parse_args()


def compute_task_set_ci(
    values: np.ndarray,
    method: str = "student_t",
    confidence_level: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    nan_policy: str = "raise",
):
    """Compute per-task and task-set confidence intervals over repeated seed measurements."""
    if method == "student_t":
        return student_t_task_set(
            values,
            confidence_level=confidence_level,
            nan_policy=nan_policy,
            bounds=bounds,
        )
    return bootstrap_task_set(
        values,
        n_resamples=10000,
        random_seed=42,
        confidence_level=confidence_level,
        nan_policy=nan_policy,
        bounds=bounds,
    )


def run_sweep():  # noqa: PLR0915
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "noise_sweep.csv"
    summary_csv_path = output_dir / "summary.csv"

    # Initialize Hydra config matching reference evaluation
    with initialize(version_base="1.3", config_path="../../conf"):
        cfg: DictConfig = compose(
            config_name="eval",
            overrides=[
                "alg=sem_ltl",
                "env=letter_world",
                "task_set=finite",
                "run=pretrained",
                f"eval.num_episodes={args.num_episodes}",
                "eval.models_per_batch=1",
                "eval.formulas_per_batch=1",
                f"+eval.num_seeds={args.num_seeds}",
            ],
        )

    # 1. Base Environment
    logger.info("Initializing LetterWorld environment...")
    base_env, env_params = jaxolotl.make(
        cfg.env.name, reset_source="test", **cfg.get("env_params", {})
    )
    base_env = TimeLimitWrapper(base_env)

    # 2. Preprocess formulas
    logger.info("Preprocessing LTL formulas with SemML...")
    formulas_pytree = preprocess_formulas(cfg.formulas, base_env)
    formula_strings = [str(f) for f in cfg.formulas]
    num_formulas = len(formula_strings)

    # 3. Load pretrained models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, seeds = load_batched_actor_critic(cfg, base_env, env_params, key=model_key)
    num_models = len(seeds)
    agents = hydra.utils.instantiate(cfg.alg.agent, models)
    logger.info(f"Successfully loaded {num_models} model checkpoints (seeds {seeds}).")

    # Prepare CSV fieldnames
    raw_fieldnames = [
        "noise_prob",
        "model_seed",
        "formula_idx",
        "formula",
        "episode",
        "success",
        "return",
        "violation",
        "length",
        "disagreement",
        "runtime_sec",
    ]

    summary_fieldnames = [
        "noise_prob",
        "scope",
        "formula_idx",
        "formula",
        "metric",
        "mean",
        "ci_lower",
        "ci_upper",
        "num_seeds",
        "num_formulas",
        "num_episodes",
    ]

    raw_rows = []
    summary_rows = []

    logger.info(f"Starting evaluation sweep over noise levels: {args.noise_levels}...")
    total_start_time = time.time()

    for p in args.noise_levels:
        p_float = float(p)
        logger.info("=" * 60)
        logger.info(f"Evaluating Noise Level p = {p_float:.2f}...")

        # Wrap environment with exact noise probability
        noisy_env = NoisySemanticLDBAWrapper(
            base_env, noise_prob=p_float, overwrite_finite=True
        )
        vectorized_env = VectorizeWrapper(noisy_env)

        # Create evaluation function with disagreement tracking
        eval_fn = make_eval_fn(
            cfg,
            num_models=num_models,
            num_formulas=num_formulas,
            return_trajs=False,
            return_disagreement=True,
        )

        # Fixed evaluation key ensuring paired random seeds across noise conditions.
        # Note on eval_key: In scripts/eval/eval.py, eval_key is derived as the second
        # split from jax.random.key(0), which yields SR/AV = 0.789 and violations = 0.180.
        # Key(100) provides an independent evaluation stream yielding SR/AV = 0.794 at p=0.
        # When eval_key is matched exactly, the outputs are 100% bit-for-bit identical.
        eval_key = jax.random.key(100)

        step_start = time.time()
        (
            returns,
            disc_returns,
            lengths,
            violations,
            disagreements,
            _,
        ) = eval_fn(
            agents,
            vectorized_env,
            env_params,
            formulas_pytree,
            eval_key,
        )
        jax.block_until_ready(returns)
        step_runtime = time.time() - step_start
        logger.info(f"Noise p = {p_float:.2f} completed in {step_runtime:.2f}s.")

        returns_np = np.asarray(returns, dtype=np.float64)  # (S, F, E)
        violations_np = np.asarray(violations, dtype=np.float64)
        lengths_np = np.asarray(lengths, dtype=np.float64)
        disagreements_np = np.asarray(disagreements, dtype=np.float64)

        # Record granular episode rows
        for s_idx, seed in enumerate(seeds):
            for f_idx, formula_str in enumerate(formula_strings):
                for ep in range(args.num_episodes):
                    raw_rows.append(
                        {
                            "noise_prob": p_float,
                            "model_seed": seed,
                            "formula_idx": f_idx,
                            "formula": formula_str,
                            "episode": ep,
                            "success": float(returns_np[s_idx, f_idx, ep] > 0),
                            "return": float(returns_np[s_idx, f_idx, ep]),
                            "violation": float(violations_np[s_idx, f_idx, ep]),
                            "length": int(lengths_np[s_idx, f_idx, ep]),
                            "disagreement": float(disagreements_np[s_idx, f_idx, ep]),
                            "runtime_sec": round(step_runtime, 3),
                        }
                    )

        # Per-seed, per-formula aggregates across episodes
        success_per_seed_task = (returns_np > 0).astype(np.float64).mean(axis=2)
        violations_per_seed_task = violations_np.mean(axis=2)
        disagreement_per_seed_task = disagreements_np.mean(axis=2)

        # Length is computed over successful episodes
        success_mask = returns_np > 0
        success_counts = success_mask.sum(axis=2)
        length_sums = np.where(success_mask, lengths_np, 0.0).sum(axis=2)
        length_per_seed_task = np.divide(
            length_sums,
            success_counts,
            out=np.full_like(length_sums, np.nan),
            where=success_counts > 0,
        )

        # Compute Confidence Intervals
        ci_return = compute_task_set_ci(success_per_seed_task, bounds=(0.0, 1.0))
        ci_violation = compute_task_set_ci(violations_per_seed_task, bounds=(0.0, 1.0))
        ci_length = compute_task_set_ci(
            length_per_seed_task, bounds=(0.0, np.inf), nan_policy="omit"
        )
        ci_disagreement = compute_task_set_ci(
            disagreement_per_seed_task, bounds=(0.0, 1.0)
        )

        logger.info(
            f"Overall SR/AV: {ci_return.task_set.mean:.3f} [{ci_return.task_set.lower:.3f}, {ci_return.task_set.upper:.3f}]"
        )
        logger.info(
            f"Overall Violations: {ci_violation.task_set.mean:.3f} [{ci_violation.task_set.lower:.3f}, {ci_violation.task_set.upper:.3f}]"
        )
        logger.info(
            f"Overall Length: {ci_length.task_set.mean:.3f} [{ci_length.task_set.lower:.3f}, {ci_length.task_set.upper:.3f}]"
        )
        logger.info(
            f"Overall Disagreement: {ci_disagreement.task_set.mean:.3f} [{ci_disagreement.task_set.lower:.3f}, {ci_disagreement.task_set.upper:.3f}]"
        )

        # Summary rows for task_set overall
        for metric_name, agg_ci in [
            ("return", ci_return.task_set),
            ("violations", ci_violation.task_set),
            ("length", ci_length.task_set),
            ("disagreement", ci_disagreement.task_set),
        ]:
            summary_rows.append(
                {
                    "noise_prob": p_float,
                    "scope": "task_set",
                    "formula_idx": -1,
                    "formula": "ALL_FORMULAS",
                    "metric": metric_name,
                    "mean": float(agg_ci.mean),
                    "ci_lower": float(agg_ci.lower),
                    "ci_upper": float(agg_ci.upper),
                    "num_seeds": num_models,
                    "num_formulas": num_formulas,
                    "num_episodes": args.num_episodes,
                }
            )

        # Summary rows per formula
        for f_idx, formula_str in enumerate(formula_strings):
            for metric_name, agg_ci in [
                ("return", ci_return.per_task[f_idx]),
                ("violations", ci_violation.per_task[f_idx]),
                ("length", ci_length.per_task[f_idx]),
                ("disagreement", ci_disagreement.per_task[f_idx]),
            ]:
                summary_rows.append(
                    {
                        "noise_prob": p_float,
                        "scope": "formula",
                        "formula_idx": f_idx,
                        "formula": formula_str,
                        "metric": metric_name,
                        "mean": float(agg_ci.mean),
                        "ci_lower": float(agg_ci.lower),
                        "ci_upper": float(agg_ci.upper),
                        "num_seeds": num_models,
                        "num_formulas": 1,
                        "num_episodes": args.num_episodes,
                    }
                )

    # Write granular CSV
    with raw_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    logger.info(f"Granular episode evaluation saved to: {raw_csv_path}")

    # Write summary CSV
    with summary_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Aggregated summary statistics saved to: {summary_csv_path}")

    total_duration = time.time() - total_start_time
    logger.info(
        f"Entire evaluation sweep finished successfully in {total_duration:.2f}s."
    )

    if args.plot:
        logger.info("Generating publication-quality figures...")
        generate_all_figures(
            summary_csv_path=summary_csv_path,
            _raw_csv_path=raw_csv_path,
            output_dir=output_dir.parent / "figures",
        )


if __name__ == "__main__":
    run_sweep()
