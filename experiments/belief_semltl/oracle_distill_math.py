"""Small, differentiable probability utilities for oracle-policy distillation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

NUM_PHYSICAL_ACTIONS = 4  # epsilon-free LetterWorld only


def physical_action_probs(dist, n: int, epsilon_token: int):
    """Extract normalized probabilities of the 4 valid physical joint actions."""
    eps = jnp.full((n,), epsilon_token, dtype=jnp.int32)
    def get_log_prob(a):
        phys = jnp.full((n,), a, dtype=jnp.int32)
        return dist.log_prob((phys, eps))
    logp = jax.vmap(get_log_prob)(jnp.arange(NUM_PHYSICAL_ACTIONS)).T
    return jax.nn.softmax(logp, axis=-1)


def teacher_kl(teacher_probs, student_probs):
    """Forward KL D_KL(teacher || student), averaged across batch."""
    p = jnp.maximum(teacher_probs, 1e-30)
    q = jnp.maximum(student_probs, 1e-30)
    return jnp.mean(jnp.sum(teacher_probs * (jnp.log(p) - jnp.log(q)), axis=-1))


def validate_dataset(data):
    """Validate cached privileged targets and prevent evaluation-seed leakage."""
    required = ("grid", "hidden", "embedding", "teacher_probs", "model_seeds", "noise_probs")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    count = len(data["grid"])
    if count < 1 or any(len(data[k]) != count for k in required):
        raise ValueError("Dataset lengths inconsistent or empty")
    probs = np.asarray(data["teacher_probs"])
    if probs.shape != (count, 4) or not np.all(np.isfinite(probs)):
        raise ValueError("Teacher probabilities must be finite [N, 4]")
    if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Invalid normalized teacher probabilities")
    if not np.all(np.isfinite(data["grid"])) or not np.all(np.isfinite(data["hidden"])):
        raise ValueError("Nonfinite input data")
    seeds = set(np.unique(np.asarray(data["model_seeds"]).astype(int)))
    if not seeds.issubset(set(range(7))):
        raise ValueError(f"Evaluation policy seeds appear in training data: {seeds}")
    if np.any(np.asarray(data["noise_probs"]) <= 0):
        raise ValueError("Distillation data must use nonzero observation noise")
