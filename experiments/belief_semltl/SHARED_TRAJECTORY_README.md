# Shared-trajectory belief diagnostic

**Purpose:** Determine whether the uniform-prior Bayesian monitor accurately tracks temporal-task state when it receives *exactly the same noisy proposition sequence* as the original hard monitor. This is **not** a fourth RL policy or an evaluation of a belief-aware controller.

## Research progression

1. **Noisy hard-progression baseline.** A frozen SemLTL policy conditions on one automaton state progressed using corrupted categorical propositions. Original full ten-formula evaluation showed decreasing true task success with greater corruption.
2. **Belief-aware frozen-policy pilot.** A uniform-prior state filter was implemented. MAP-state embeddings and weighted-average embeddings were tested on nine epsilon-free formulas. Neither improved on hard progression in the initial closed-loop pilot; at p=0.30 the reported success rates were hard **0.5708**, MAP **0.5215**, weighted **0.4715**. The pilot was correctly controlled for the same initial environment distribution, but alternative policies visited different locations after their actions diverged.
3. **This shared-trajectory diagnostic.** Run *only* the hard-policy controller. Feed the resulting same corruption sequence to the hard and passive Bayesian monitors. Compare estimates against ground-truth automaton state on every active timestep. This distinguishes monitor inference quality from consequences of different policies taking different actions.

The chronological experiments, including unsuccessful interventions, should be retained for the eventual report and presentation. Existing results are left unchanged.

## Method

For each pretrained model seed, selected LTL formula, and corruption probability, the hard policy controls LetterWorld. The original noisy wrapper produces the true state `ldba_state`, the hard perceived state `perceived_ldba_state`, and the corrupted proposition labels `perceived_propositions`.

A passive filter, initialized as a one-hot distribution on the initial automaton state, computes

\[
P(a\mid y) = \begin{cases}1-p &a=y,\\p/(K-1)&a\ne y,\end{cases}
\qquad
b_{t+1}(q')=\sum_{q,a}b_t(q)P(a\mid y_t)\mathbf 1[\delta(q,a)=q'].
\]

The filter receives **only** `perceived_propositions`, its previous belief, the known corruption probability, and the automaton transition table. **Ground-truth state is consulted only afterward for evaluation**, never for the belief update or policy actions.

The agent continues receiving the *hard* monitor's original observation. No active MAP or weighted controller is evaluated. This ensures both estimators are scored against identical trajectories and observation sequences, without forcing a MAP policy to move against its own decisions.

## Measures

- **Hard timestep accuracy:** fraction of active physical steps for which `perceived_ldba_state == ldba_state`.
- **MAP timestep accuracy:** fraction of those same steps for which `argmax(belief) == ldba_state`.
- **MAP minus hard accuracy:** paired difference measured on exactly the same steps.
- **True-state probability:** mean `b_t(q_true)`; measures how much posterior mass the filter assigns to the actual task stage.
- **Negative log true-state probability (NLL):** mean `-log(max(b_t(q_true),1e-12))`; the clip is a finite numerical convention.
- **Entropy:** mean `-sum_q b_t(q)log b_t(q)` in nats.
- **Episode disagreement:** whether an estimator is wrong at least once in an episode. This is **not** percent of steps wrong.

Step-level aggregates pool active timesteps **within each model seed**; 95% Student's t intervals are constructed across model seed means (10 seeds in full mode). LTL formulas are treated as a *fixed evaluation set*, not a random sample of all LTL formulas.

`by_step.csv` conditions each time-index average on episodes still active at that step; changing episode composition causes survivorship effects. The figures are descriptive, not causal mechanistic analyses.

## Commands (from repository root)

```bash
JAX_PLATFORMS=cpu pixi run pytest experiments/belief_semltl/shared_trajectory_stats_test.py -q
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_shared_trajectory.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_shared_trajectory.py
```

The smoke command uses 2 pretrained model seeds, 2 formulas, 4 episodes, and p=0,0.3. The full command uses 10 seeds, 9 epsilon-free formulas, 16 episodes, and p=0,0.1,0.3.

The runner checks at p=0 that hard and MAP states match at **every active step**, and that the passive filter assigns essentially unit probability to the true state. By default it also compares its first hard rollout against the existing `Evaluator.eval` on the same key, checking returns, lengths, violations, and episode disagreement. If the first gate fails, **stop and diagnose rather than interpreting results**.

Note: the runner hasn't been executed in this patch's build environment against the full Jaxolotl dependencies; local Pixi execution is required.

## Output files

Stored separately from prior results in `experiments/belief_semltl/results/shared_trajectory/{smoke,full}/`:

- `episodes.csv`: one record per episode, including monitor accuracy counts/sums and success/violation indicators.
- `steps.csv`: active timestep records, including ground-truth, hard and MAP state indexes, posterior mass on truth, log score, entropy. Potentially large; exclude from Git unless desired.
- `summary.csv`: per-formula and overall seed-level means/CIs.
- `by_step.csv`: time-indexed trajectory diagnostics (descriptive).
- `metadata.json`: run specification, seed set, prior assumptions, and limitations.
- `step_accuracy.png/.pdf`, `entropy_by_step.png/.pdf`.

## Interpretation and limitations

The filter uses an **independent uniform prior** over all 13 possible proposition assignments at each step; this is not the actual physical dynamics or an exact posterior given LetterWorld's egocentric grid view. The hard policy's observations may reveal the true current proposition independently of this monitor.

This diagnostic measures estimator quality *conditional on trajectories produced by the hard policy*. It does not establish that a state estimator would perform equally on trajectories generated by MAP or weighted policies, or that improved state accuracy necessarily improves closed-loop task success.

A MAP deficit would motivate improving proposition prediction or accounting for temporal/spatial correlations **before** modifying policy architecture. A MAP advantage paired with worse task success would suggest investigating how the frozen policy consumes inferred task-stage information (potentially policy mixtures). Neither outcome alone establishes a novel research contribution.

**Excluded:** formula index 5, the only epsilon-containing specification in the original ten-formula evaluation, so the epsilon semantics do not confound this diagnostic.
