"""Train a zero-initialized semantic residual to mimic oracle actions.

Frozen: pretrained SemLTL actor and fine-tuned DAgger GRU. Trainable: only a
linear map from the frozen GRU hidden state into semantic input coordinates.
Gradients pass THROUGH the frozen SemLTL actor into the adapter parameters.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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
import optax
from hydra import compose, initialize_config_dir

import jaxolotl
from jaxolotl.alg.sem_ltl.batching import preprocess_formulas
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.eval import EvalResetOptions
from jaxolotl.eval.utils import load_batched_actor_critic
from experiments.belief_semltl.collect_dagger import EXPERIMENT, sha256
from experiments.belief_semltl.learned_tracker_wrapper import LearnedTrackerWrapper
from experiments.belief_semltl.run_learned_estimator import load_estimator
from experiments.belief_semltl.oracle_distill_adapter import SemanticResidualAdapter
from experiments.belief_semltl.oracle_distill_math import (
    physical_action_probs, teacher_kl, validate_dataset,
)

LOG = logging.getLogger("oracle_distill_train")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--dagger-model", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--l2-residual", type=float, default=1e-4)
    p.add_argument("--max-delta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=112)
    a = p.parse_args()
    split = "smoke" if a.smoke else "full"
    a.dataset = a.dataset or EXPERIMENT / "data" / "distillation" / split / "train.npz"
    a.dagger_model = a.dagger_model or EXPERIMENT / "models" / "dagger" / split / "learned_estimator.eqx"
    a.output_dir = a.output_dir or EXPERIMENT / "models" / "distillation" / split
    if a.smoke:
        a.epochs = 1
        a.batch_size = 16
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.l2_residual < 0 or a.max_delta <= 0:
        p.error("Invalid training hyperparameters")
    return a


def make_training_observation(env, params, task, estimator, agent, batch_size):
    """Provide the actor a genuine SemLTL observation template at minibatch size."""
    vector = VectorizeWrapper(LearnedTrackerWrapper(
        env, estimator=estimator, noise_prob=0.3, overwrite_finite=True,
    ))
    keys = jax.random.split(jax.random.PRNGKey(941), batch_size)
    state, obs = vector.reset(keys, None, params, EvalResetOptions(task=task))
    props = eqx.filter_vmap(vector.compute_propositions)(state, params)
    agent = agent.update(obs, state, props, vector)
    return agent, obs


def make_train_step(optimizer, l2_residual):
    @eqx.filter_jit
    def train_step(adapter, opt_state, agent, observation, batch):
        def loss_fn(adapter):
            hidden, embedding, grid, teacher = batch
            correction = jax.vmap(adapter)(hidden)
            corrected_obs = eqx.tree_at(
                lambda o: (o.features.grid, o.embedding),
                observation, (grid, embedding + correction),
            )
            dist = agent.get_action(corrected_obs)
            epsilon_token = corrected_obs.epsilon_embeddings.shape[1]
            student_probs = physical_action_probs(dist, grid.shape[0], epsilon_token)
            kl = teacher_kl(teacher, student_probs)
            reg = l2_residual * jnp.mean(jnp.sum(correction ** 2, axis=-1))
            return kl + reg, (kl, reg)

        (loss, (kl, reg)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(adapter)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(adapter, eqx.is_array))
        return eqx.apply_updates(adapter, updates), opt_state, loss, kl, reg

    return train_step


def batch_indices(indices: np.ndarray, batch_size: int, generator: np.random.Generator):
    """Visit all examples once; pad only final short batch to fixed JAX shape."""
    shuffled = generator.permutation(indices)
    for start in range(0, len(shuffled), batch_size):
        current = shuffled[start:start + batch_size]
        if len(current) < batch_size:
            current = np.concatenate([
                current, generator.choice(indices, batch_size - len(current), replace=True)
            ])
        yield current


def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ckpt = a.output_dir / "residual.eqx"
    meta_path = a.output_dir / "metadata.json"
    if ckpt.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite distillation output: {a.output_dir}")
    for path in (a.dataset, a.dagger_model):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(a.dataset) as f:
        data = {k: f[k] for k in f.files}
    validate_dataset(data)
    dataset_meta = a.dataset.parent / "metadata.json"
    if dataset_meta.is_file():
        provenance = json.loads(dataset_meta.read_text(encoding="utf-8"))
        if provenance.get("source_dagger_sha256") != sha256(a.dagger_model):
            raise ValueError("Collector and trainer use different DAgger checkpoints")
    seeds = sorted(set(map(int, data["model_seeds"])))
    embed_dim = data["embedding"].shape[-1]
    hidden_dim = data["hidden"].shape[-1]
    estimator = load_estimator(a.dagger_model, hidden_dim=hidden_dim)
    adapter = SemanticResidualAdapter(hidden_dim, embed_dim,
                                      key=jax.random.PRNGKey(a.seed), max_delta=a.max_delta)
    optimizer = optax.adamw(a.lr, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(adapter, eqx.is_array))
    step = make_train_step(optimizer, a.l2_residual)

    with initialize_config_dir(version_base="1.3", config_dir=str((ROOT / "conf").resolve())):
        cfg = compose(config_name="eval", overrides=[
            "alg=sem_ltl", "env=letter_world", "task_set=finite", "run=pretrained",
            "eval.models_per_batch=1", "eval.formulas_per_batch=1",
        ])
    base_env, params = jaxolotl.make(cfg.env.name, reset_source="test", **cfg.get("env_params", {}))
    base_env = TimeLimitWrapper(base_env)
    # The action head only needs one valid epsilon-free LDBA to construct its
    # properly shaped observation template; all dataset formulas have no eps.
    formulas = preprocess_formulas([cfg.formulas[0]], base_env)
    if np.any(np.asarray(formulas.epsilon_transitions) >= 0):
        raise ValueError("Template formula should be epsilon-free")
    task = jax.tree.map(lambda x: x[0], formulas)
    models, all_seeds = load_batched_actor_critic(cfg, base_env, params, key=jax.random.PRNGKey(42))
    all_seeds = list(map(int, all_seeds))
    if not set(seeds).issubset(all_seeds):
        raise ValueError(f"Policy seeds unavailable: {seeds}, available: {all_seeds}")
    trained_agents = {}
    for seed in seeds:
        i = all_seeds.index(seed)
        model = jax.tree.map(
            lambda x: x[i] if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == len(all_seeds) else x,
            models,
        )
        agent = hydra.utils.instantiate(cfg.alg.agent, model)
        trained_agents[seed] = make_training_observation(
            base_env, params, task, estimator, agent, a.batch_size,
        )

    rng = np.random.default_rng(a.seed)
    history = []
    for epoch in range(1, a.epochs + 1):
        losses = []
        for seed in seeds:
            agent, template_obs = trained_agents[seed]
            choices = np.flatnonzero(np.asarray(data["model_seeds"]) == seed)
            for idx in batch_indices(choices, a.batch_size, rng):
                batch = tuple(jnp.asarray(data[k][idx]) for k in
                              ("hidden", "embedding", "grid", "teacher_probs"))
                adapter, opt_state, loss, kl, reg = step(
                    adapter, opt_state, agent, template_obs, batch,
                )
                loss_float, kl_float, reg_float = float(loss), float(kl), float(reg)
                if not np.isfinite(loss_float):
                    raise FloatingPointError("Nonfinite oracle distillation loss")
                losses.append((loss_float, kl_float, reg_float))
        means = np.asarray(losses).mean(axis=0).tolist()
        history.append({"epoch": epoch, "total_loss": means[0],
                        "teacher_KL": means[1], "residual_penalty": means[2]})
        LOG.info("epoch=%d/%d loss=%.5f teacher_KL=%.5f residual_penalty=%.5f",
                 epoch, a.epochs, *means)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt, adapter)
    with meta_path.open("x", encoding="utf-8") as f:
        json.dump({
            "method": "frozen-actor oracle policy distillation with zero-init residual",
            "dataset": str(a.dataset), "dataset_sha256": sha256(a.dataset),
            "dagger_model": str(a.dagger_model), "dagger_model_sha256": sha256(a.dagger_model),
            "hidden_dim": hidden_dim, "semantic_dim": embed_dim, "max_delta": a.max_delta,
            "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
            "l2_residual": a.l2_residual, "seed": a.seed, "policy_seeds": seeds,
            "loss_history": history, "model_selection": "fixed final epoch, no evaluation tuning",
            "frozen": "pretrained SemLTL actors and DAgger GRU; only adapter updated",
            "intervention": "z = projection(e(q_map)+r(hidden)) = projection(e(q_map))+W*r(hidden)",
            "limitations": [
                "Teacher acts with true automaton state and ordinary LetterWorld raw grid",
                "No direct closed-loop reward optimization; objective is teacher action KL",
                "Original GRU checkpoint was previously selected using seeds 7..9",
                "No epsilon-transition formulas supported",
            ],
        }, f, indent=2)
    LOG.info("Saved adapter to %s", ckpt)


if __name__ == "__main__":
    main()
