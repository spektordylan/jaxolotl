# Phase 2: learned task-state tracking (checkpoint-preserving handoff)

## Objective and design

Test whether a lightweight supervised task-state estimator can improve an **unchanged, pretrained** SemLTL actor under noisy proposition detections. The tracker is a 64-unit Equinox GRU; inputs at physical step `t` are only `(noisy_assignment_t, previous_physical_action_t, hard_monitor_state_t)`. Its output is masked to the current LDBA's number of valid states, then `argmax` selects a real automaton state whose **precomputed semantic embedding** conditions the frozen actor.

The true automaton state is retained only for environment reward, termination and diagnostic scoring. The tracker receives no raw LetterWorld grid and thus cannot trivially read the true current-cell letter. The **frozen actor itself still receives the standard unmasked grid**, as in prior experiments; do not claim the entire policy is restricted to the noisy detection channel.

The archive contains two partially overlapping model/trainer variants (`learned_tracker.py` + `train_tracker.py` and `learned_estimator.py` + `train_estimator.py`). The **active** end-to-end path is `learned_tracker.py`, `train_estimator.py`, `learned_tracker_wrapper.py`, and `run_learned_estimator.py`. The other files remain for compatibility, not because both paths have been independently validated.

## What was repaired

1. **Reset/sequence alignment:** The original collector records input/output starting at the first **physical step**, not the reset observation. The online wrapper previously advanced the GRU during reset, creating a history shift. It now begins at zero hidden state and the known initial task state; the first GRU update occurs after physical action 1. The initial detector observation is intentionally not consumed by the GRU, consistent with the saved checkpoint's training data.
2. **Valid-state masking:** Training, offline diagnostics and deployment now mask automaton padding states consistently (`num_states` differs between formulas). This does not change the checkpoint architecture, but the archived checkpoint was *trained* before this loss correction. Retraining requires a new output path.
3. **Evaluation:** At `p=0` the learned wrapper must exactly reproduce hard-policy per-episode lengths, returns, violations, and hard-state correctness; the runner now raises instead of merely warning. At `p>0` different controllers may diverge; their reported comparisons are matched by random initialization, **not** same-trajectory comparisons.
4. **Statistical analysis:** `paired_effects.csv` includes mean learned-minus-hard effects and 95% Student-t confidence intervals over pretrained policy seeds, aggregating episodes and the fixed set of formulas within each seed.
5. **Output safety:** Collectors, trainers, and evaluation refuse to replace existing output artifacts. All archived experiment results, NPZ datasets, and checkpoints remain intact.
6. **Collector provenance:** New datasets include `model_seeds`, `formula_indices`, and `noise_probs` metadata. The old test collector accidentally used `--num-train-episodes` for the test split; the `--num-test-episodes` flag is now respected in future collections. Original dataset is unchanged.

## Existing evidence (not a new closed-loop result)

The archive contains the existing `models/full/metrics.json`, reporting raw held-out-across-gradient-seeds accuracy `0.7202` compared with hard-state accuracy `0.6230`, pooled across noise conditions. This is **not** new evidence produced by the patch. In addition, `train_estimator.py` selected the archived checkpoint using `test.npz` (policy seeds 7..9): this is a **validation** set, not an untouched final holdout. The difference is descriptive and does not imply closed-loop improvement. Raw estimator `p=0` accuracy must not be conflated with the *forced* zero-noise hard-state passthrough used by the deployment wrapper.

The archived checkpoint and original collector also use the same semantic task distribution; no held-out-formula generalization is claimed.

## Running in the full Jaxolotl repository

Apply the patch **from the repository root**, where `experiments/belief_semltl/` and the original `jaxolotl/`, `conf/`, and pretrained model directories exist. The supplied ZIP is only the experiment subtree, not the full runnable upstream repository.

```bash
# Pure statistics tests and Equinox model tests (in original pixi environment)
JAX_PLATFORMS=cpu pixi run pytest experiments/belief_semltl/learned_stats_test.py experiments/belief_semltl/learned_tracker_test.py -q

# Inspect original frozen checkpoint on the recorded, matched hard-policy histories.
# --legacy-full-order is ONLY for the original 2592-episode full/test.npz.
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/offline_tracker_diagnostic.py --legacy-full-order

# Mandatory closed-loop smoke test (policy seeds 7,8; formulas 0,3; 4 episodes; p=0,.3)
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py --smoke

# Only after zero-noise parity holds and offline p=.3 state accuracy is promising:
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_learned_estimator.py
```

The evaluator defaults to seed identifiers `7,8,9` rather than `0..9` because `0..6` generated gradient-training trajectories. **Caveat:** the original checkpoint was selected using `7..9`, so these are *not* independent final-model-selection test seeds. A rigorous final benchmark requires separate checkpoint-selection data and new unseen evaluation layouts/seeds.

The smoke/full evaluator writes **new** artifacts in `results/learned_estimator/{smoke,full}/`: `episodes.csv`, `summary.csv`, `paired_effects.csv`, `metadata.json`. Choose a different `--output-dir` for another run. No earlier results will be overwritten.

To collect *new* trajectories or train from scratch, use **new** directory paths:

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/collect_trajectories.py --output-dir experiments/belief_semltl/data/trajectories/recollection_v2
# CAUTION: train_estimator.py currently chooses its best model on "test.npz".
# Treat that data as validation; reserve a distinct fresh split for final conclusions.
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/train_estimator.py --data-dir experiments/belief_semltl/data/trajectories/recollection_v2 --output-dir experiments/belief_semltl/models/retrain_v2
```

`--data-dir` selects newly collected datasets; if omitted, training uses the original `data/trajectories/{smoke,full}`. **Never overwrite the archived datasets.** For the immediate handoff the provided checkpoint is the default evaluator input; retraining is optional, not required to exercise the code. Retrained checkpoints must also be explicitly selected with the evaluator's `--model-path`.

## Validation status / scientific gates

- **Implemented, locally checked:** pure statistical tests, Python parse/compile, output overwrite safeguards. Model tests require the user's Pixi environment.
- **Not verified in this ZIP-only environment:** Equinox model execution, Jaxolotl wrapper compatibility, complete p=0 episode parity, offline `p=.30` estimator advantage, and closed-loop success. This environment does not include `jaxolotl`, `equinox`, `hydra`, or `optax`, and the ZIP did not include the original source tree.
- **Stop before interpreting `p>0` results** if the exact `p=0` parity gate raises. Do not claim the learned estimator helps until the same-trajectory accuracy and closed-loop paired success results are actually observed.
- **Next methodological improvement:** retrain with a separate **training / model-selection / final evaluation** split (and ideally new environment layouts), then compare `p=.30` success with the hard baseline and oracle headroom; report both successes and failures.
