# Milestone 5: Training-free policy-output mixtures (experimental)

**Central chronological narrative:** [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). The older `PILOT_README.md`, `SHARED_TRAJECTORY_README.md` and `ORACLE_README.md` remain valid historical records. This experiment **does not modify** those previous source scripts or outcomes.

## Why this experiment follows the previous results

1. **Noisy hard monitor:** ~79.4% success at zero noise, ~53.6% at p=.30 on the original **ten-formula** set. This aggregate includes epsilon formula 5 and is provisional pending the epsilon correction.
2. **Nine-formula frozen-policy pilot:** Hard/MAP/weighted success at `p=.30` was `0.5708 / 0.5215 / 0.4715`, respectively, when the policy received a single hard state embedding, a MAP embedding, or an averaged embedding. MAP/weighted failed to improve performance. At `p=0` all three were `0.8458`.
3. **Shared-trajectory diagnostic:** At p=.30, hard timestep accuracy `0.3788`; Bayesian MAP `0.3435`. The approximate independent uniform-proposition prior is one plausible cause of inaccurate beliefs.
4. **Oracle-monitor ablation:** With the same **frozen policy** but privileged correct automaton-state input, success `0.8458` at p=`0, .10, .30` versus hard `0.8458, 0.7236, 0.5708`; oracle-minus-hard `+12.22` pp (95% CI `[+10.29,+14.16]`) at p=.10, and `+27.50` pp (95% CI `[+22.93,+32.07]`) at p=.30. This demonstrates headroom under privileged monitoring but no practical way to obtain the true state.
5. **This experiment:** We retain the unchanged, imperfect filter but change **how the frozen actor consumes it**. The previous weighted-embedding input may have been out-of-distribution. This experiment NEVER feeds an averaged embedding to the neural network. Instead, it samples a valid state-conditioned policy and samples its action.

## Exact mathematical intervention

Let `q_hard` be the perceived deterministic automaton state, `b(q)` the approximate Bayesian state probabilities from corrupted labels only, `o` the existing navigation observation, `e(q)` a *valid* SemLTL state embedding, and `pi_theta(a|o,e(q))` the unchanged pretrained **stochastic** action distribution.

For fixed blend weight `alpha`:

```
pi_alpha(a | o, b, q_hard)
  = (1-alpha) * pi_theta(a | o, e(q_hard))
    + alpha * sum_q b(q) * pi_theta(a | o, e(q)).
```

Implementation is **hierarchical**, not explicit logit extraction: at every timestep, with probability `1-alpha`, choose `q_hard`; otherwise draw `q ~ Categorical(b)`. Then draw `a ~ pi_theta(. | o, e(q))`. Marginalizing over the chosen state gives **exactly** the equation above for each stochastic action. This takes **one actor evaluation per step**, unlike evaluating the entire policy for every state, and works with the actor's existing joint environment/epsilon action distribution (although epsilon-containing formulas are excluded because the belief tracker does not model epsilon transitions).

**This is NOT the same as averaging logits, averaging embeddings, averaging greedy actions, or taking argmax over the mixture.** The evaluator is forced into **stochastic sampling mode for ALL methods**, including the hard baseline. As a result the new hard success numbers should be compared with the new hard numbers in the **same run**; any difference from older experiments under different deterministic settings must be reported, not silently combined.

### Methods and prespecified weights

- `hard`: untouched baseline/noisy semantic wrapper, evaluated again for matched stochastic controls.
- `mix_0.00`: alpha=0: identical to hard but using the outer wrapper and new agent; **strict parity control** at each noise value.
- `mix_0.25`: 25% belief mixture, 75% hard anchor.
- `mix_0.50`: equal blend of hard and belief mixture.
- `mix_1.00`: fully belief-weighted policy output mixture.

The weights are defined before seeing results. These are not optimized on the nine test formulas; avoid presenting the best-performing one as unbiased evidence of generalization. A subsequent experiment could evaluate a chosen alpha on held-out tasks.

## Environment, sensing and reward boundary

`PolicyMixtureWrapper` wraps the existing `NoisySemanticLDBAWrapper`, rather than reimplementing true progression or reward. The wrapper's own `belief` starts as a delta on the initial LDBA state and updates only from **corrupted perceived propositions** using the existing `compute_posterior_a` and `compute_belief_next` functions. It does not inspect true propositions or true automaton state when choosing an embedding. Original ground-truth automaton remains responsible for reward, termination and violations. The egocentric navigational observation is unchanged from previous experiments.

The belief prior remains **uniform, independent over 13 assignment labels each step** and is not a calibrated LetterWorld generative model. A better action mixture need not solve the state-estimation deficit, and a negative mixture result does not invalidate belief-based control generally.

## Apply and run

Run from Jaxolotl repository root after extracting the ZIP:

```bash
# Standalone pure JAX and statistical tests, then Jaxolotl integration:
JAX_PLATFORMS=cpu pixi run pytest \
  experiments/belief_semltl/mixture_sampling_test.py \
  experiments/belief_semltl/mixture_stats_test.py \
  src/jaxolotl/alg/sem_ltl/wrappers/policy_mixture_wrapper_test.py \
  -x -q

JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_policy_mixture.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_policy_mixture.py
```

Smoke: two pretrained seeds, formulas `[0,3]`, four episodes per seed/formula, p=`[0,.3]`, alpha=`[0,.5,1]`. Full: ten seeds, nine epsilon-free formulas `[0,1,2,3,4,6,7,8,9]`, 16 episodes, p=`[0,.1,.3]`, alpha=`[0,.25,.5,1]`. This is `10*9*16*3*5=21,600` evaluation episodes (counting hard and mix_0 control).

**Strict correctness gates:**

1. At p=0, ALL mixture methods must match hard on **each episode's** success, return, violation, length, and disagreement.
2. At every p, `mix_0.00` must match hard on those same metrics, episode by episode.
3. Formulas with epsilon transitions (including index 5) are excluded, checked after compilation.
4. All methods share `jax.random.key(100)` and actor sample keys. Branch selection uses a separate derived key, preserving the actor's original action-sampling key for zero-mix and zero-noise parity.
5. Any failure invalidates nonzero-noise comparisons: stop and supply the first traceback; do not publish the partial run.

The methods may follow different trajectories after an action divergence, even with the same reset and sampling key. This is appropriate for **closed-loop** success comparison; it is not a matched-trajectory estimator test.

## Outputs

`experiments/belief_semltl/results/policy_mixture/{smoke,full}/`:

- `episodes.csv`: per-seed/formula/episode binary success, returns, lengths, violations, disagreement.
- `summary.csv`: per-method/per-formula and pooled means and Student's t intervals **across 10 model seeds**.
- `paired_effects.csv`: mixture-minus-hard paired success-rate differences and 95% seed-level intervals (other methods are compared with the same new hard control).
- `mixture_success.png`, `mixture_success.pdf`: overall success curves.
- `metadata.json`: settings, validation design, git revision, known limitations.

**Not measured in this patch:** per-step action-probability Jensen–Shannon disagreement. The hierarchical sampler intentionally avoids exposing or evaluating per-state action logits. We can add a focused, optionally expensive diagnostic later if a mixture helps; do not mislabel state entropy or sampled state differences as action-distribution disagreement.

## Completed results (2026-11-24)

The user completed the full nine-formula, ten-seed, 16-episode-per-cell evaluation after applying the wrapper delegation correction. The runner returned the following aggregate task-success rates:

| Noise p | Hard | Mix 0.00 | Mix 0.25 | Mix 0.50 | Mix 1.00 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8458 | 0.8458 | 0.8458 | 0.8458 | 0.8458 |
| 0.10 | 0.7236 | 0.7236 | 0.7056 | 0.7007 | 0.6639 |
| 0.30 | 0.5708 | 0.5708 | 0.5500 | 0.5514 | 0.5153 |

At p=.30, the differences from hard were **-2.08 pp** for alpha=.25, **-1.94 pp** for alpha=.50, and **-5.55 pp** for alpha=1.00. At p=.10, they were **-1.80, -2.29, and -5.97 pp**, respectively. **These are descriptive differences, not confidence-interval conclusions**; see `results/policy_mixture/full/paired_effects.csv` for paired uncertainty intervals and `summary.csv` for per-formula effects.

The identical p=0 values and `mix_0.00` control provide expected parity checks. The runner enforces stricter episode-level parity, and the full run completed without a reported failure. An earlier implementation bug (double-unwrapping the nested environment state before delegating to the noisy wrapper) was fixed in the separate wrapper-fix patch before the completed run. The corrected Jaxolotl test suite's final test count was not supplied alongside the aggregate results.

**Interpretation:** Every nonzero mixture weight lost to hard progression in aggregate at the tested nonzero noise levels. Valid single-state embeddings and policy-output aggregation did not rescue the existing misspecified belief distribution. This does not rule out a better state estimator, alternate policy-selection rule, or learned adaptation. The frozen-policy series is now closed and the next candidate is a trained uncertainty-aware adapter rather than additional alpha sweeps.

## Research chronology

The complete sequence, including the original noisy baseline, MAP/weighted embedding failures, matched-trajectory estimation deficit, oracle-monitor headroom, and this negative output-mixture result, is in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Retain this chronology when writing the final report and presentation; do not overwrite negative experiments with only the eventual best method.
