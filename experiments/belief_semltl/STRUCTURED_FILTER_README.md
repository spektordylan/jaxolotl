# Stage 2: Structured Hybrid LDBA Filters

Two interventions sharing the **exact LDBA transition table**, the original frozen
SemLTL actor, and the previously used 9 epsilon-free formulas.

- **Stage 2a (`empirical`):** estimate the 13-way assignment prior from *training*
  trajectories. Compute `P(true_assignment | noisy_assignment)` using the known
  symmetric noise channel and propagate the state belief through exact automaton
  transitions. No neural retraining.
- **Stage 2b (`proposition-gru`):** train a formula-agnostic 13-way recurrent
  proposition predictor from noisy assignments and actions, then propagate its
  predicted assignment probabilities through the exact automaton transitions.
  True propositions are recorded for supervised *training only*.

The inferred LDBA state is `argmax` of the propagated belief, which supplies a
**valid existing semantic embedding** to the frozen policy. True LDBA state and
true proposition are only used to determine outcomes and training labels.
The existing policy still sees the unchanged original spatial grid.

## Stage 2a — use existing archived `train.npz`

```bash
cd ~/Desktop/SemLTL/jaxolotl
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/prepare_empirical_prior.py
JAX_PLATFORMS=cpu pixi run pytest -q \
  experiments/belief_semltl/structured_filter_test.py
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_structured_filter.py --method empirical --smoke
```

After smoke parity passes:

```bash
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_structured_filter.py --method empirical
```

Optional teacher-forced validation (the archived `test.npz` is **not independent**):

```bash
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/offline_structured_diagnostic.py --method empirical
```

**Archived dataset caveat:** original `train.npz` contains no `true_assignments`
or `noise_probs`. The prior builder recognizes the exact legacy 6048-trajectory
collector order (7 seeds × 9 formulas × 3 noise levels × 32 episodes), using
only the `p=0` blocks as clean labels. It refuses unknown layouts. If the
collector layout has been changed, **do not use this fallback**; recollect
with true proposition labels, then explicitly provide that new train file.

## Stage 2b — fresh data collection and supervised training

Existing `.npz` files are left untouched. The updated collector now logs
`true_assignments`, `formula_indices`, and `noise_probs` in newly written files.

```bash
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/collect_trajectories.py --smoke \
  --output-dir experiments/belief_semltl/data/structured_smoke
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/train_proposition_estimator.py --smoke
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_structured_filter.py \
  --method proposition-gru --smoke \
  --proposition-model experiments/belief_semltl/models/structured_filter/proposition_gru_smoke/proposition_estimator.eqx
```

If smoke works, collect and train a *fresh full* dataset:

```bash
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/collect_trajectories.py \
  --output-dir experiments/belief_semltl/data/structured_full
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/train_proposition_estimator.py
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/offline_structured_diagnostic.py \
  --method proposition-gru \
  --dataset experiments/belief_semltl/data/structured_full/test.npz
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_structured_filter.py --method proposition-gru
```

For a **held-out formula** probe, train with e.g.
`--holdout-formula-index 0 --output-dir experiments/belief_semltl/models/structured_filter/holdout_0`.
Both training and validation exclude formula 0. Evaluate only formula 0 via
`run_structured_filter.py --method proposition-gru --formula-indices 0
--proposition-model experiments/belief_semltl/models/structured_filter/holdout_0/proposition_estimator.eqx
--output-dir experiments/belief_semltl/results/structured_filter/holdout_0`.
This tests architectural transfer, but a single formula is insufficient to
establish broad zero-shot generalization.

## Comparison and statistical interpretation

Every evaluation prints hard, original GRU-MAP, structured MAP, and structured
minus GRU-MAP closed-loop success for p=0, 0.1, 0.3. Each method uses the same
random reset keys; different decisions may lead to different trajectories.
`paired_effects.csv` reports seed-aggregated differences, not episode-level
pseudo-replication. The historical GRU checkpoint was selected using seeds
7–9, so these are **validation-style results**, not untouched final tests.

Stored under `results/structured_filter/{empirical,proposition-gru}/{smoke,full}`.
An existing output path causes an error rather than overwrite. `models` and
`data` output paths also refuse replacement.

## Important conceptual limitations

1. The global marginal prior is estimated from policy-induced visits under
   **zero noise**, not the true conditional letter distribution at every
   physical location under the new policy.
2. The neural proposition posterior summarizes *history* and is then multiplied
   into a recursively maintained state belief that also summarizes history.
   The update omits correlations between previous LDBA state and current
   proposition. It is a **factored approximation**, not a calibrated exact
   Bayesian POMDP filter; exact symbolic transitions alone do not fix this.
3. The exact transition table guarantees that the filter makes no impossible
   **individual state transitions**, but it can assign substantial probability
   to the wrong *reachable* state and can enter the wrong absorbing state.
4. Formula-agnostic model inputs permit testing transfer to unseen formulas;
   transfer performance or a 10× data-efficiency gain is **not guaranteed**.
5. The rollout machinery was adapted from the validated learned-estimator
   wrapper, but the complete original Jaxolotl/Pixi environment is absent from
   this handoff. The first `--smoke` run remains the runtime integration gate.
