# SemLTL under uncertain propositions — chronological experiment log

**Created:** 2026-11-24. **Updated:** 2026-11-27 (UTC). **Status:** Phase 1 complete; GRU-MAP, learned-belief aggregation, empirical-prior filtering, and one-round DAgger evaluated. Residual oracle distillation is documented separately on its experiment branch. Research log, not a final claims section. Update this file as new experiments finish; do not replace negative results with later positive results. This log is intended to support the December 4 project report and presentation.

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

### Milestone 4 — Oracle-monitor ablation (completed)

- **Question:** How much closed-loop success can the existing frozen policy achieve when task-progress inference is made perfect, without changing the policy weights?
- **Method:** Introduce `OracleSemanticLDBAWrapper`, which retains the *same* noisy underlying wrapper and true LDBA reward/termination but replaces the semantic embedding and epsilon features fed to the neural policy with those from the **true** LDBA state. Compare to hard with matched seeds, formulas, episode/reset/action-sampling RNG schedule, and the same evaluator. Only the true-state semantic input is the intervention. No retraining.
- **Conditions:** 10 seeds, same 9 epsilon-free formulas, 16 episodes, `p=0,.10,.30`, 2 modes = **8,640 evaluated episodes**. Formula 5 remains excluded.
- **Required correctness gates:** (1) hard and oracle episode outcomes must match exactly at p=0; (2) oracle's episode outcomes must be identical at every tested noise level given the same random key because the oracle disregards the noisy proposition channel; (3) the oracle monitor's reported disagreement must be zero, rather than reflecting the latent unused noisy tracker. Stop if any gate fails.
- **Result:** The user completed and reported the full 10-seed, nine-formula oracle ablation. The oracle success rate remained **0.8458** at p=`0,.10,.30`; hard progression fell from **0.8458** to **0.7236** and **0.5708**. Oracle-minus-hard gains were **+12.22 pp** (seed-paired 95% CI `[+10.29,+14.16]` pp) at p=.10 and **+27.50 pp** (95% CI `[+22.93,+32.07]` pp) at p=.30. This is a strong privileged-monitor reference, not evidence that accurate state inference is possible from the noisy detectors. The oracle invariance with p is expected from the construction, not an independent surprise.
- **Outputs:** `results/oracle_ablation/{smoke,full}/episodes.csv`, `summary.csv`, `paired_effects.csv`, `metadata.json`, `oracle_vs_hard_success.png/.pdf`.
- **Interpretation plan:** Large oracle-minus-hard gap at p>0 implies potential room for better task-state inference or alternative use of inference with this frozen network. Small gap suggests little practical headroom from monitor correction alone. The oracle is *privileged* and *not deployable* under noisy sensing, and it is **not a mathematical upper bound on all retrained policies**.

### Milestone 5 — Policy-output mixture with hard anchoring (completed; negative result)

- **Question:** Can the same imperfect state belief help the *frozen* SemLTL policy if combined at the **stochastic action-distribution level** rather than by averaging semantic embeddings or selecting MAP?
- **Method:** Hierarchical marginal policy sampling: with probability `1-alpha`, condition the pretrained actor on the hard monitor's valid state embedding; otherwise sample an automaton state `q ~ b(q)` and condition the actor on that valid embedding. Sample the action with the existing actor. Marginally this realizes `(1-alpha)*pi_hard + alpha*sum_q b(q)*pi_q` for stochastic actors, while requiring one actor evaluation per step. No policy weights are changed. The same approximate uniform-prior filter from milestone 2 is reused; **this is not a new belief estimator**.
- **Conditions:** Ten pretrained seeds, nine epsilon-free formulas `[0,1,2,3,4,6,7,8,9]`, 16 episodes per seed/formula/noise/method, `p=0,.10,.30`; methods `hard`, `mix_0.00`, `mix_0.25`, `mix_0.50`, `mix_1.00`. Total **21,600 policy-evaluation episodes**. All methods use the same stochastic action-sampling setting, model checkpoints, formulas, and evaluation key schedule.

| Noise p | Hard success | Mix 0.00 | Mix 0.25 | Mix 0.50 | Mix 1.00 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8458 | 0.8458 | 0.8458 | 0.8458 | 0.8458 |
| 0.10 | 0.7236 | 0.7236 | 0.7056 | 0.7007 | 0.6639 |
| 0.30 | 0.5708 | 0.5708 | 0.5500 | 0.5514 | 0.5153 |

- **Effect relative to the concurrently re-evaluated hard baseline:** At p=.10, differences for alpha `0.25 / 0.50 / 1.00` are **-1.80 / -2.29 / -5.97 percentage points**. At p=.30, they are **-2.08 / -1.94 / -5.55 points**. These are descriptive pooled differences; inspect `paired_effects.csv` for seed-paired uncertainty intervals before making inferential or per-formula claims.
- **Validation and integration history:** An initial test detected incorrect wrapper-state unwrapping on delegation to `NoisySemanticLDBAWrapper`; the wrapper implementation and test were corrected in the separate mixture-wrapper fix patch. The user subsequently completed the full evaluation and reported the table above. The full runner enforces `mix_0.00`/hard episode parity at every noise level and zero-noise parity of all alpha values; completion without a reported gate failure is consistent with those checks having passed. No fresh full test log was supplied with the results, so do not claim a particular total of passing integration tests for the corrected version.
- **Finding:** All nonzero belief-mixture weights **underperformed** the hard baseline in aggregate at p=.10 and p=.30. Therefore simply avoiding weighted-embedding interpolation does **not** rescue this particular, approximate belief tracker. This does **not** show that policy-output aggregation or belief-aware control is ineffective when supplied with a better belief distribution, and does not settle which failure mechanism dominates.
- **Code:** `run_policy_mixture.py`, `policy_mixture_agent.py`, `mixture_sampling.py`, `policy_mixture_wrapper.py` (corrected), `mixture_stats.py`, `MIXTURE_README.md`; results under `results/policy_mixture/full/`. Commit the corrected wrapper/tests, not the original broken wrapper from the first patch.
- **Motivation for next step:** The frozen-policy exploration has tested deterministic hard tracking, MAP, weighted embeddings and stochastic output mixtures without finding a practical gain from the uniform-prior belief. A learned uncertainty-aware adapter or carefully scoped noise-aware training is a more distinct intervention than further mixture-weight tuning.

### Frozen-policy exploration — phase conclusion (historical snapshot, 2026-11-24)

1. **Noisy monitoring creates a meaningful robustness problem.** In the common nine-formula comparison, hard success falls from **0.8458** (p=0) to **0.5708** (p=.30).
2. **Privileged perfect monitoring reveals headroom:** the oracle retains **0.8458** success at p=.30, a **+27.50 pp** gap over hard (seed-paired 95% CI **[+22.93, +32.07] pp**). The oracle sees the true automaton state, so this is **not** an achievable noisy-detector result and not a universal performance ceiling.
3. **The approximate filter has an empirical estimation deficit:** on identical hard-policy trajectories at p=.30, hard and MAP timestep accuracies were **0.3788** and **0.3435**, respectively. The uniform independent-proposition prior ignores temporal/spatial correlations.
4. **Three ways of using the same approximate belief failed to outperform hard in closed-loop control:** MAP-state embedding, probability-weighted embedding, and stochastic state-conditioned policy-output mixture (including hard-anchored variants). This is evidence against *these implementations under this observation model*, not against Bayesian filtering or uncertainty-aware policies in general.
5. **Future work / next project phase:** Favor a small learned adapter (possibly trained by privileged oracle-policy distillation) or a carefully defined noise-aware policy intervention. Freeze or preserve current checkpoints and results. Enforce the sensing boundary: ground-truth propositions may be used as training supervision or evaluator targets, but not as deployment-time adapter input. Establish train/evaluation separation and include held-out specifications or layouts when claiming generalization.

**Presentation/report narrative:** Begin with the robustness problem, then document the unsuccessful Bayesian embedding interventions, shared-trajectory diagnosis, oracle headroom, and unsuccessful policy-output aggregation. Explain why the sequence motivated a switch to learning, rather than hiding the negative results. For every table identify the nine-formula versus provisional ten-formula comparison and avoid treating seed-level CIs as generalization guarantees.

## Next-phase decisions and guardrails (historical plan from 2026-11-24)

1. **Close the uniform-prior frozen-policy intervention series.** Do not keep tuning mixture weights or adding minor embedding heuristics to the same filter absent new evidence; the prespecified nonzero alpha values all lost in aggregate. The oracle gap alone does not imply a noisy sensor can recover the true automaton state.
2. **Investigate a genuinely different method.** The leading candidate is a small trainable, uncertainty-aware adapter or oracle-policy distillation protocol with the pretrained SemLTL actor frozen. Design permitted inputs, loss, compute budget, and held-out evaluation before implementation. No adapter results exist yet.
3. **Keep sensing assumptions explicit.** Proposition detector outputs are corrupted; the ordinary egocentric navigation observation is unchanged. Do not use ground-truth propositions, automaton state, or a privileged letter lookup as deployment-time inputs. Audit the observation boundary if a proposed solution exploits grid features that could directly disclose true letters.
4. **Formula-dependent/hybrid selection is still exploratory.** A selector based on formula syntax or compiled automaton structure must be evaluated on held-out specifications and needs a demonstrably beneficial candidate method; the current results do not justify fitting one to these same nine formulas.
5. **Preserve statistical and methodological limits.** The original ten-formula sweep contains provisional epsilon-formula results; later experiments exclude formula 5. Seed-paired confidence intervals reflect pretrained-seed variation on this benchmark, not guaranteed robustness across environments. Re-evaluated hard controls must be compared within each experiment's action-sampling setting.
6. **Retain chronology and negative results.** Keep original scripts and outputs immutable where possible. Update this log when a learned experiment is designed or completed rather than rewriting earlier findings retrospectively.

## Working documents

- `README.md`: original noisy hard-baseline methods/results (historical, may contain provisional epsilon claims).
- `PILOT_README.md`: MAP and weighted frozen-policy comparison.
- `SHARED_TRAJECTORY_README.md`: passive-filter estimator diagnostic.
- `ORACLE_README.md`: ablation implementation, validation and usage.
- `MIXTURE_README.md`: completed policy-output mixture and hard-anchored variants, results, validation, and exact stochastic-mixture construction.
- **This file** is the project-wide chronological narrative. Update this log when a new experiment completes and keep runnable scripts/outputs in their respective directories.

---

## Phase 2 implementation handoff (historical snapshot, 2026-11-27; subsequently executed)

The partial learned-estimator branch in the exported ZIP included a collector, recurrent GRU model, a `full` checkpoint, and a draft closed-loop evaluator. A corrective patch was prepared without changing any archived results or checkpoints. It aligns online GRU initialization with the existing trajectory dataset (first update **after** the first physical action), masks invalid formula-dependent state logits in training and inference, enforces exact zero-noise parity, adds seed-paired outcome intervals, and safeguards output artifacts against overwriting. New collector datasets also record provenance and use the requested number of test episodes.

The original checkpoint metrics (`models/full/metrics.json`, 72.02% raw state accuracy versus 62.30% hard on the archived split) are **pre-patch records**, not independently reproduced results. The archived `test.npz` was used for best-checkpoint selection, and therefore serves as a validation set. In particular, the 72.02% number does **not** establish a causal success benefit or independent generalization. Neither Jaxolotl parity nor closed-loop policy outcomes were executable in the patch-preparation environment because only the experiment subtree was supplied and the Jaxolotl/Pixi dependencies are absent. Follow `LEARNED_TRACKER_README.md` for required local validation gates and reproduction commands.


## Phase 2 — completed interventions and results (late November 2026)

The following results were **reported from full closed-loop runs in the user's Jaxolotl/Pixi environment** after the initial handoff and wrapper fixes. The comparison set is 9 epsilon-free formulas `[0,1,2,3,4,6,7,8,9]`, 16 episodes per seed/formula/noise condition, and policy seeds **7–9** for evaluation. These runs are **not directly interchangeable** with the 10-seed Phase 1 hard/oracle aggregates (`0.8458`/`0.5708`), because the later evaluation uses a different seed cohort. The full-run hard control in this phase is `0.8449` at p=0 and `0.5602` at p=.30.

### Milestone 6 — learned task-state GRU with frozen policy (completed)

- **Intervention:** A 64-hidden-unit recurrent estimator takes `(noisy proposition, previous physical action, hard monitor state)` histories, predicts a valid LDBA state via masked MAP, and substitutes that state's existing semantic embedding into the **unchanged** SemLTL policy. True automaton states are supervised training labels and evaluator ground truth, not deployment inputs. The policy still sees its ordinary spatial grid. At p=0 the intervention explicitly bypasses the estimator, ensuring exact hard-policy parity.
- **Offline diagnostic on archived `test.npz`:** raw GRU accuracy was `99.01%` versus hard `100.00%` at p=0; `73.92%` versus `64.89%` at p=.10; and `53.53%` versus `36.51%` at p=.30 (GRU advantage `+17.02 pp` at p=.30). These teacher-forced histories were used for checkpoint selection, making this a **validation**, not untouched final-test, result.
- **Closed-loop result:**

| Noise | Hard success | GRU-MAP success | GRU minus hard | Hard step accuracy | GRU step accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8449 | 0.8449 | +0.00 pp | 1.0000 | 1.0000 |
| 0.10 | 0.7593 | 0.7801 | +2.08 pp | 0.8348 | 0.8496 |
| 0.30 | 0.5602 | 0.6366 | +7.64 pp | 0.5419 | 0.5823 |

- **Interpretation:** Learned recurrent state estimation improves **observed** task success under noise, without policy retraining. Offline and online state-accuracy gains are measured on different trajectory distributions and cannot by themselves establish covariate shift as the cause of the difference. The archived checkpoint was previously selected using policy seeds 7–9, so this is an exploratory model-selection/evaluation split, not an independent holdout.
- **Artifacts:** `LEARNED_TRACKER_README.md`; `models/full/learned_estimator.eqx`; `results/learned_estimator/full/`.

### Milestone 7 — top-3 GRU belief-weighted action aggregation (completed)

- **Intervention:** For the three most probable **valid** LDBA states from the same GRU, evaluate the frozen actor under each state's real embedding, weight **action probabilities** by their renormalized posterior weights, and sample from the resulting policy mixture (`top_k=3`, `alpha=1`). No retraining. Exact zero-noise parity was verified after using the original actor sampler for the p=0 path.

| Noise | Hard success | GRU-MAP | Belief aggregation | Aggregation minus GRU-MAP |
|---:|---:|---:|---:|---:|
| 0.00 | 0.8449 | 0.8449 | 0.8449 | +0.00 pp |
| 0.10 | 0.7593 | 0.7801 | 0.7662 | -1.39 pp |
| 0.30 | 0.5602 | 0.6366 | 0.6389 | +0.23 pp |

- **Interpretation:** No material incremental benefit relative to GRU-MAP in this full evaluation. This rejects the *tested top-3 policy-probability mixture*, not belief-aware planning generally. The initial 16-episode smoke result of `1.0000` at p=.30 did **not** generalize to the full run.
- **Artifacts:** `BELIEF_ACTION_AGGREGATION_README.md`; `results/learned_aggregation/full/`.

### Milestone 8 — structured filter, Stage 2a empirical proposition prior (completed); Stage 2b not evaluated

- **Intervention:** Replace the Phase 1 uniform 13-assignment prior with a marginal estimated from the archived **zero-noise training** trajectories, then propagate automaton-state probabilities through the exact LDBA transition table and provide the MAP embedding to the frozen policy. The empirical frequency of `empty` was about **47.45%**, with individual letters roughly **2.7–6.2%**; the commonly posited 85% empty-cell prior is **not** the prior measured in this dataset.

| Noise | Hard success | GRU-MAP | Empirical-prior structured MAP | Structured minus GRU-MAP |
|---:|---:|---:|---:|---:|
| 0.00 | 0.8449 | 0.8449 | 0.8449 | +0.00 pp |
| 0.10 | 0.7593 | 0.7801 | 0.7616 | -1.85 pp |
| 0.30 | 0.5602 | 0.6366 | 0.5162 | -12.04 pp |

- **Interpretation:** The empirical marginal prior alone is insufficient and is worse even than hard progression at p=.30 (`-4.40 pp`). Neither base-rate suppression nor the exact cause of recursive belief errors was directly isolated by these outcomes. Exact LDBA transitions enforce legal **transitions**, not correct inference; the recursive filter remains an approximation.
- **Stage 2b:** A formula-agnostic proposition-GRU collector/trainer and symbolic propagation pipeline were implemented, but **no trained Stage 2b full-run result was reported**. Do not claim this method was evaluated or that it generalizes to held-out formulas.
- **Artifacts:** `STRUCTURED_FILTER_README.md`; `results/structured_filter/empirical/full/`.

### Milestone 9 — one-round DAgger-style GRU fine-tuning (completed)

- **Intervention:** Collect one round of trajectories **controlled by the original GRU-MAP** on policy seeds 0–6, concatenate these with original hard-policy training trajectories, and fine-tune the existing GRU for **three epochs at lr=1e-4**. The frozen actor is unchanged. This uses **true task-state labels**, not expert action labels, so it is DAgger-*style* on-policy dataset aggregation rather than canonical action-imitation DAgger. The original model/checkpoint remains intact.
- **Full closed-loop comparison, same evaluation keys (policy seeds 7–9):**

| Noise | Hard success | Original GRU-MAP | DAgger GRU-MAP | DAgger minus original, paired 95% seed CI |
|---:|---:|---:|---:|---:|
| 0.00 | 0.8449 | 0.8449 | 0.8449 | +0.00 pp [0.00, 0.00] |
| 0.10 | 0.7593 | 0.7801 | 0.7870 | +0.69 pp [-3.87, +5.26] |
| 0.30 | 0.5602 | 0.6366 | 0.6620 | +2.55 pp [-6.95, +12.05] |

- **State accuracy:** At p=.30, GRU-MAP increased from `0.5823` to `0.5929` (about `+1.06 pp`). Relative to the concurrently evaluated hard success `0.5602`, DAgger success `0.6620` is an **observed +10.18 pp** (`+10.19 pp` in rounded runner output), equivalent to roughly **35.8% of the p=0 to p=.30 hard-success decrease**. This percentage is a descriptive recovery fraction, not an oracle/headroom benchmark or an inferential guarantee.
- **Interpretation:** Fine-tuning was directionally positive, but its *incremental* benefit over GRU-MAP is statistically inconclusive; the paired CI includes zero. The offline-versus-online difference alone does not prove the cause was covariate shift, and one-round fine-tuning does not isolate dataset size from on-policy adaptation.
- **Artifacts:** `DAGGER_README.md`; `models/dagger/full/learned_estimator.eqx`; `results/dagger/full/original/`, `results/dagger/full/finetuned/`, `results/dagger/full/paired_comparison.csv`.

### Phase 2 interpretation and reporting guardrails (current through Milestone 9)

1. On the common three-evaluation-seed protocol, **GRU-MAP** improves success at p=.30 from `56.02%` to `63.66%`; **GRU+DAgger** has the best observed result so far at `66.20%`. These are descriptive means, not claims of statistical significance against hard without their own seed-paired intervals.
2. Weighted GRU action aggregation (`63.89%`), the empirical-prior structured filter (`51.62%`), and one-round fine-tuning (`66.20%`) should all be reported, including negative and uncertain findings. Do not select the 100% smoke-test success as the reported aggregation result.
3. Phase 1's oracle success of `84.58%` and gap of `27.50 pp` came from a different **ten-seed** experiment. Do **not** combine it quantitatively with this three-seed Phase 2 hard baseline without a matched oracle rerun.
4. The GRU checkpoint was chosen using the same seed cohort later used for evaluation, so this is **not** an independently held-out test; only nine epsilon-free formulas were evaluated; the frozen actor itself still receives the ordinary grid (which includes true current-cell letter channels). At p=0, equality is imposed via the tracker bypass rather than learned perfect inference. With only three policy-seed units, seed-level CIs are wide and do not establish transfer across formulas/layouts.
5. The original Phase 1 plan and the Phase 2 implementation handoff above are **historical snapshots**; their statements about future learned interventions/no runtime results are superseded by these completed records. Results are preserved as reported in the CLI; numeric tables here were not independently reloaded from the user's latest local CSV files.
