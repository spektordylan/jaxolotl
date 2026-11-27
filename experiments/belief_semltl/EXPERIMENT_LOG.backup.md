# SemLTL under uncertain propositions — chronological experiment log

**Created:** 2026-11-24. **Status:** Research log, not a final claims section. Update this file as new experiments finish; do not replace negative results with later positive results. This log is intended to support the December 4 project report and presentation.

## Research question and controlled environment

Can agents trained for LTL-guided navigation with reliable atomic propositions remain successful when their **task-monitor proposition observations** are noisy, and can uncertainty-aware monitoring recover performance without retraining? The environment is LetterWorld in Jaxolotl, with the same frozen pretrained SemLTL actor-critic network checkpoints across conditions. LetterWorld has 12 singleton letter propositions and the empty assignment (13 categorical possible values).

For a true assignment `a`, the reported label `y` is correct with probability `1-p`; otherwise it is drawn uniformly from the other 12 assignments (probability `p/12` each). This noise affects the task-monitor input; the standard egocentric environment observation remains available. The true LDBA state, updated with the true environment proposition, **always** defines reward, sink/violation, and termination. Only the representation of task progress presented to the policy varies by method.

### Milestone 1 — Noisy-observation hard baseline (completed)

- **Question:** How does the original frozen SemLTL controller perform when its monitor treats corrupted propositions as certain?
- **Method:** `NoisySemanticLDBAWrapper`: one perceived LDBA state advanced with corrupted labels, plus an independent true LDBA state for scoring. Noise levels `0,.02,.05,.10,.15,.20,.30` across 10 seeds, 10 formulas, 16 episodes = 11,200 episodes.
- **Result, original 10-formula sweep:** success decreased from **0.794** at `p=0` to **0.536** at `p=.30`; reported episode-level disagreement rose to approximately **0.806** at `p=.30`.
- **Caveat:** Formula index 5 is the only epsilon-containing formula; invalid true epsilon actions originally required a no-op correction. Thus original ten-formula aggregates including formula 5 are **provisional** until a corrected rerun. Nine epsilon-free formulas are the principal comparison set below. The original reference `.789` vs sweep `.794` at no noise used different evaluation random keys.
- **Code:** `run_noise_sweep.py`, `noisy_semantic_ldba_wrapper.py`, `README.md`.
- **Motivation for next step:** Deterministically committing to a possibly wrong proposition loses explicit uncertainty. Test a distribution over automaton states.

### Milestone 2 — Bayesian filter + frozen-policy MAP/weighted pilot (completed)

- **Question:** Can a simple probabilistic task monitor improve the frozen policy without modifying network weights?
- **Belief approximation:** Start from a point mass on the initial automaton state. At each physical step estimate `P(a|y)` with a *uniform independent prior* on all 13 true assignments, then propagate state mass via `q' = delta(q,a)`. This is **not** an exact posterior for LetterWorld's spatial dynamics or its full observations.
- **Methods:** `hard` uses the perceived deterministic LDBA; `map` uses the semantic embedding of `argmax_q b(q)`; `weighted` uses `sum_q b(q)e(q)` in place of the ordinary single-state embedding. All share frozen network weights and matched evaluation keys; post-intervention paths may diverge.
- **Validation:** 17 tests passed in the user environment. At `p=0` all three yielded exactly the same episode-level success/violations/lengths/disagreement.
- **Conditions:** 10 pretrained seeds, 9 epsilon-free formulas `[0,1,2,3,4,6,7,8,9]`, 16 episodes, `p=0,.10,.30`, 3 modes = 12,960 policy-evaluation episodes.

| Noise p | Hard success | MAP success | Weighted success |
|---:|---:|---:|---:|
| 0.00 | 0.8458 | 0.8458 | 0.8458 |
| 0.10 | 0.7236 | 0.7188 | 0.6396 |
| 0.30 | 0.5708 | 0.5215 | 0.4715 |

- **Finding:** Both belief-aware policy-input schemes underperformed the hard baseline in aggregate. At p=.30, MAP lost **4.93 percentage points** and weighted embeddings lost **9.93 points** relative to hard. These negative results are retained as experimental evidence, **not** interpreted as proof that Bayesian filtering is useless in general. Per-formula performance was heterogeneous, with some small local improvements that require separate evaluation.
- **Possible explanations:** A misspecified prior yields worse task-state estimates; averaging semantic embeddings produces out-of-distribution conditioning; MAP loses temporal history/uncertainty information.
- **Code:** `run_pilot.py`, `pilot_stats.py`, `PILOT_README.md`, `belief_semantic_ldba_wrapper.py`; results under `results/belief_pilot/full/`.
- **Motivation for next step:** Closed-loop performance combines inference quality and downstream policy behavior. Separate those using matched observation histories.

### Milestone 3 — Shared-trajectory estimator diagnostic (completed)

- **Question:** Does MAP actually infer the *correct* automaton state more accurately than hard on exactly the same input sequence?
- **Method:** Hard policy alone controls an episode; both its own perceived monitor and a passive Bayesian filter consume exactly the same corrupted labels on that one trajectory. Ground-truth LDBA state is consulted only for scoring. The test runner reproduced the standard evaluator's returns/lengths/violations/disagreement on its parity check and enforced hard/MAP equality at zero noise.
- **Conditions:** 10 seeds, 9 epsilon-free formulas, 16 episodes, p=`0,.10,.30`.

| Noise p | Hard timestep state accuracy | MAP timestep state accuracy | MAP minus hard |
|---:|---:|---:|---:|
| 0.00 | 1.0000 | 1.0000 | 0.00 pp |
| 0.10 | 0.6672 | 0.6627 | -0.45 pp |
| 0.30 | 0.3788 | 0.3435 | -3.53 pp |

- **Finding:** The simple approximate MAP monitor has an **estimation deficit**, even holding the trajectory and noisy observations fixed. This supports the filter-misspecification hypothesis as at least one failure mechanism, but does not establish that an improved filter would produce higher task success. At p=.30 the mean mass on the true state was ~0.2704, mean true-state negative log-likelihood ~2.6041, mean belief entropy ~0.9951 nats. These summary values **do not** alone establish probability calibration.
- **Probable modeling weakness:** The filter assigns a uniform prior to 13 propositions at every step, independently of position, movement history and the egocentric grid. It does not implement a full joint spatial/automaton Bayes filter. Sink-state probability can accumulate under implausible hypothetical proposition sequences; this is a mechanism to test, not yet measured as the cause.
- **Code:** `run_shared_trajectory.py`, `shared_trajectory_stats.py`, `SHARED_TRAJECTORY_README.md`; results under `results/shared_trajectory/full/`.
- **Motivation for next step:** Determine the *potential headroom* of the existing frozen network if its monitor knew task progress perfectly. This informs whether additional filter research is warranted.

### Milestone 4 — Oracle-monitor ablation (implemented; outcome pending)

- **Question:** How much closed-loop success can the existing frozen policy achieve when task-progress inference is made perfect, without changing the policy weights?
- **Method:** Introduce `OracleSemanticLDBAWrapper`, which retains the *same* noisy underlying wrapper and true LDBA reward/termination but replaces the semantic embedding and epsilon features fed to the neural policy with those from the **true** LDBA state. Compare to hard with matched seeds, formulas, episode/reset/action-sampling RNG schedule, and the same evaluator. Only the true-state semantic input is the intervention. No retraining.
- **Conditions:** 10 seeds, same 9 epsilon-free formulas, 16 episodes, `p=0,.10,.30`, 2 modes = **8,640 evaluated episodes**. Formula 5 remains excluded.
- **Required correctness gates:** (1) hard and oracle episode outcomes must match exactly at p=0; (2) oracle's episode outcomes must be identical at every tested noise level given the same random key because the oracle disregards the noisy proposition channel; (3) the oracle monitor's reported disagreement must be zero, rather than reflecting the latent unused noisy tracker. Stop if any gate fails.
- **Status:** Source and tests packaged. **The Jaxolotl/Pixi integration has not yet been run at the time this entry was written. No results are claimed.** Add actual outcomes, seed-paired uncertainty, and interpretation after the user runs it.
- **Outputs:** `results/oracle_ablation/{smoke,full}/episodes.csv`, `summary.csv`, `paired_effects.csv`, `metadata.json`, `oracle_vs_hard_success.png/.pdf`.
- **Interpretation plan:** Large oracle-minus-hard gap at p>0 implies potential room for better task-state inference or alternative use of inference with this frozen network. Small gap suggests little practical headroom from monitor correction alone. The oracle is *privileged* and *not deployable* under noisy sensing, and it is **not a mathematical upper bound on all retrained policies**.

## Future decisions and guardrails

1. If oracle performance shows meaningful headroom, allow **at most one targeted better-prior filter trial** (e.g., transition/location-informed proposition predictions), with strict separation between observables and ground truth. Check whether estimation accuracy and policy success both improve.
2. If oracle does not yield meaningful headroom, or the one improved-filter trial fails, pivot to a distinct approach: e.g., explicit uncertainty-aware action selection or modest noise-aware training that fits the project's available CPU compute.
3. Formula-dependent/hybrid selection is a plausible *later* idea given heterogeneous results, but do not fit a selector to the same nine formulas and claim generalization. It requires held-out specifications and a component that actually beats hard on a justified subset.
4. Retain both positive and negative results in their original chronological order. Include PRNG/method differences, epsilon caveats and uncertainty intervals in all final report tables. Do not combine the original 10-formula and later 9-formula averages as though they are directly comparable.

## Working documents

- `README.md`: original noisy hard-baseline methods/results (historical, may contain provisional epsilon claims).
- `PILOT_README.md`: MAP and weighted frozen-policy comparison.
- `SHARED_TRAJECTORY_README.md`: passive-filter estimator diagnostic.
- `ORACLE_README.md`: ablation implementation, validation and usage.
- **This file** is the project-wide chronological narrative. Update this log when a new experiment completes and keep runnable scripts/outputs in their respective directories.
