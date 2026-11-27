# One-round DAgger-style trajectory aggregation for GRU-MAP

**Scope:** Adapt the existing supervised task-state GRU to the histories it induces when it controls a frozen SemLTL actor. The actor is not retrained. The intervention remains GRU-MAP, with a fine-tuned estimator checkpoint.

This is *DAgger-style dataset aggregation*, not classic imitation-learning DAgger with an expert action oracle. The privileged label is the true automaton state; no expert action is queried.

## Protocol

- Collect a single batch of **on-policy trajectories from the ORIGINAL GRU-MAP checkpoint** on policy seeds 0..6, epsilon-free formulas 0,1,2,3,4,6,7,8,9 and proposition noise p=0.10 and p=0.30.
- Inputs and labels are recorded immediately **after** each environment step, exactly matching the original trajectory data and the GRU's deployment ordering: `(noisy_assignment_t, executed_physical_action_t, hard_monitor_state_t)` -> `true_automaton_state_t`. This alignment matters. Ground-truth automaton states and propositions are *labels only*, never estimator/policy features.
- Concatenate complete original hard-policy training trajectories with these new on-policy trajectories (not with original validation/test trajectories). Retrain all GRU weights for **three fixed epochs** at learning rate `1e-4`, starting from a deserialized copy of the original checkpoint and a **fresh** AdamW optimizer state. Do not select checkpoints using `test.npz` or final evaluation outcomes.
- Compare the original and fine-tuned checkpoints with the **same** `run_learned_estimator.py` evaluator, model seeds 7..9, and the same environment random keys. A separate analysis computes per-seed paired differences and 95% t intervals. The evaluation seed split is separate from new DAgger gradient data, but the *original* checkpoint had previously been selected on seeds 7..9, so this does not constitute a completely untouched holdout.
- Pretrained actor still sees its standard LetterWorld grid (including current-cell letters) just as in previous experiments; *only the GRU* is input-restricted. The original `p=0` bypass remains in effect.

## Requirements

Run from the **Jaxolotl repository root** with its working Pixi environment, after applying the previous learned-tracker patches (including API/state-chain repairs). These scripts reuse the existing collector and closed-loop evaluator. Do not extract an overlay ZIP in addition to applying the patch.

## Smoke test (small training set and evaluation; does not touch existing results)

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/collect_dagger.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/train_dagger.py --smoke

JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py --smoke \
  --output-dir experiments/belief_semltl/results/dagger/smoke/original
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py --smoke \
  --model-path experiments/belief_semltl/models/dagger/smoke/learned_estimator.eqx \
  --output-dir experiments/belief_semltl/results/dagger/smoke/finetuned
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/summarize_dagger.py \
  --baseline experiments/belief_semltl/results/dagger/smoke/original/episodes.csv \
  --dagger experiments/belief_semltl/results/dagger/smoke/finetuned/episodes.csv \
  --output experiments/belief_semltl/results/dagger/smoke/paired_comparison.csv
```

## Full experiment

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/collect_dagger.py
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/train_dagger.py

JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py \
  --output-dir experiments/belief_semltl/results/dagger/full/original
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py \
  --model-path experiments/belief_semltl/models/dagger/full/learned_estimator.eqx \
  --output-dir experiments/belief_semltl/results/dagger/full/finetuned
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/summarize_dagger.py \
  --baseline experiments/belief_semltl/results/dagger/full/original/episodes.csv \
  --dagger experiments/belief_semltl/results/dagger/full/finetuned/episodes.csv \
  --output experiments/belief_semltl/results/dagger/full/paired_comparison.csv
```

The summary prints success, delta and seed-paired confidence intervals for each noise level. The detailed CSV includes success, violation, and episode-averaged state accuracy. Compare `p=.30` DAgger success directly to the original GRU-MAP (previously 0.6366 in full evaluation). Both original and fine-tuned runs must use the exact same evaluator configuration for a valid paired comparison. The summary rejects mismatched episode conditions, diverging hard-policy references, and `p=0` parity failures.

## Output safety and limitations

- Existing `data/trajectories/*`, `models/full/learned_estimator.eqx`, and `results/learned_estimator/*` remain unchanged. Every new artifact is in `data/dagger`, `models/dagger`, or `results/dagger`, and scripts refuse to overwrite their own outputs. To repeat a run, supply **new** `--output-dir` paths. For training, `--dagger-data` and `--original-data` allow explicit input paths. Evaluation's `--model-path` picks the checkpoint.
- Don't infer distribution shift from offline accuracy versus online accuracy alone. A DAgger improvement would support the practical benefit of on-policy adaptation, but not isolate a unique causal mechanism without an ablation controlling dataset size and training budget.
- New collection is performed under the **original** checkpoint, once. There is no recursive multi-iteration DAgger and no policy-gradient training.
- The frozen policy gets the raw grid; downstream action choices may still access true position-level semantic information through that channel. Interpret conclusions within the existing setup.
- Full Jaxolotl runtime tests require the original repository and environment, neither of which was included in the archived experiment ZIP. A smoke run is mandatory before trusting the full experiment.
