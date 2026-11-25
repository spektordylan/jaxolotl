# SemLTL under uncertain propositions — chronological experiment log

**Created:** 2026-11-24. **Updated:** 2026-11-24. **Status:** Training-free frozen-policy exploration closed; future learned interventions not yet implemented. Research log, not a final claims section. Update this file as new experiments finish; do not replace negative results with later positive results. This log is intended to support the December 4 project report and presentation.

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

### Frozen-policy exploration — phase conclusion (2026-11-24)

1. **Noisy monitoring creates a meaningful robustness problem.** In the common nine-formula comparison, hard success falls from **0.8458** (p=0) to **0.5708** (p=.30).
2. **Privileged perfect monitoring reveals headroom:** the oracle retains **0.8458** success at p=.30, a **+27.50 pp** gap over hard (seed-paired 95% CI **[+22.93, +32.07] pp**). The oracle sees the true automaton state, so this is **not** an achievable noisy-detector result and not a universal performance ceiling.
3. **The approximate filter has an empirical estimation deficit:** on identical hard-policy trajectories at p=.30, hard and MAP timestep accuracies were **0.3788** and **0.3435**, respectively. The uniform independent-proposition prior ignores temporal/spatial correlations.
4. **Three ways of using the same approximate belief failed to outperform hard in closed-loop control:** MAP-state embedding, probability-weighted embedding, and stochastic state-conditioned policy-output mixture (including hard-anchored variants). This is evidence against *these implementations under this observation model*, not against Bayesian filtering or uncertainty-aware policies in general.
5. **Future work / next project phase:** Favor a small learned adapter (possibly trained by privileged oracle-policy distillation) or a carefully defined noise-aware policy intervention. Freeze or preserve current checkpoints and results. Enforce the sensing boundary: ground-truth propositions may be used as training supervision or evaluator targets, but not as deployment-time adapter input. Establish train/evaluation separation and include held-out specifications or layouts when claiming generalization.

**Presentation/report narrative:** Begin with the robustness problem, then document the unsuccessful Bayesian embedding interventions, shared-trajectory diagnosis, oracle headroom, and unsuccessful policy-output aggregation. Explain why the sequence motivated a switch to learning, rather than hiding the negative results. For every table identify the nine-formula versus provisional ten-formula comparison and avoid treating seed-level CIs as generalization guarantees.

## Next-phase decisions and guardrails

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
