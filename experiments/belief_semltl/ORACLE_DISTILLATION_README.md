# Residual oracle-policy distillation (one fixed-budget intervention)

**Question:** Can a decision-aware semantic residual improve on GRU+DAgger's 66.20% closed-loop success at proposition noise p=0.30, without retraining the frozen SemLTL actor or the state tracker?

## Intervention

- Base: `models/dagger/{smoke,full}/learned_estimator.eqx`, unchanged.
- At time t, compute the existing GRU hidden state `h_t` and MAP embedding `e(q_hat_t)`.
- Learn a small **zero-initialized** linear adapter `r(h_t)` and replace the semantic input with `e(q_hat_t) + r(h_t)`. The frozen SemLTL model uses a linear semantic projection, so the actual projected representation is `projection(e(q_hat_t)) + W r(h_t)` (a residual in the projection space). No actor weights change.
- Training targets are distributions over the **four physical actions** from the same frozen actor with the **true automaton state's embedding** and identical LetterWorld spatial observation. The teacher's true state is a privileged *label*, never a student input.
- Loss is `KL(oracle || adapted_actor) + 1e-4 ||r(h_t)||^2`. Gradients pass through the frozen actor to **only** the adapter. No policy-gradient RL or fine-tuning of the GRU.
- At p=0, the wrapper explicitly disables the residual; the existing hard/GRU parity holds exactly. All formulas must be epsilon-free.

The adapter is an *input-space* correction that induces a projected-space residual by linearity; it does **not** modify the frozen model's projection parameters. Unlike changing the projection network directly, it can be implemented entirely at Jaxolotl's existing observation interface.

## Setup

Apply the `.patch` **once**, from the Jaxolotl repository root. All new sources live under `experiments/belief_semltl`. There is no replacement ZIP. Run with the existing Pixi environment and preinstalled pretrained policies. The DAgger data and checkpoint from the prior experiment must be present.

Unit test and syntax check:

```bash
JAX_PLATFORMS=cpu pixi run pytest -q experiments/belief_semltl/oracle_distill_math_test.py
```

## Smoke (distinct outputs, do not overwrite)

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/collect_oracle_distill.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/train_oracle_distill.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_oracle_distill.py --smoke
```

This collects on-policy samples under the **smoke DAgger checkpoint** using pretrained policy seeds 0,1 (2 formulas, 4 episodes each), trains the adapter, then compares original DAgger against adapted DAgger on policy seeds 7,8 (2 formulas, 4 episodes each). Smoke outcomes are an integration check, not evidence of a reliable performance gain.

## Full run (fixed protocol)

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/collect_oracle_distill.py
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/train_oracle_distill.py
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_oracle_distill.py
```

Defaults: 7 collection policy seeds (0–6), 9 epsilon-free formulas, p in {0.10,0.30}, 8 episodes per condition, up to 128 labeled timesteps per condition, 2 training epochs, batch size 64, AdamW lr=3e-4, and no hyperparameter sweep. Test uses policy seeds 7–9, all 9 formulas, p in {0,0.10,0.30}, 16 episodes per condition, with paired run keys matching the existing `run_learned_estimator.py` evaluator. The unit of replication for confidence intervals is the pretrained policy seed.

Outputs (all use exclusive file creation / no intentional overwrites):

- `data/distillation/{smoke,full}/train.npz` and `metadata.json`
- `models/distillation/{smoke,full}/residual.eqx` and `metadata.json`
- `results/distillation/{smoke,full}/episodes.csv`, `paired_effects.csv`, `metadata.json`

To rerun, specify a fresh `--output-dir`. When using `--output-dir` for evaluation, point to an unused directory; when using a different adapter, supply `--adapter` and `--dagger-model` together. For custom collection data paths, use `--dataset` during training.

## Gates and interpretation

1. Collection and training finish without shape/JIT errors; training KL is finite.
2. The **zero-noise parity gate** passes, including sampled episode outcomes.
3. The full evaluator prints DAgger and distillation success at p=0.30 plus a paired seed-level 95% t interval.
4. If there is no clear improvement over the 66.20% DAgger baseline, **stop** and write up the experiment; don't tune the adapter to the validation seeds.

Limitations: the original GRU checkpoint was selected using seeds 7–9, so this is not an untouched test split. The adapter only sees GRU hidden state; the pretrained actor still sees the ordinary raw LetterWorld grid including the current-cell letter channel. Distilling the oracle's action distribution is not equivalent to optimizing long-horizon expected return. Our experiment deliberately avoids epsilon-transition formulas and cannot establish generality to them.
