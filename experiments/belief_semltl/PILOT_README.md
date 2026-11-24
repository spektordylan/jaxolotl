# Frozen-policy belief-aware SemLTL pilot

This experimental extension compares a deterministic noisy-label SemLTL monitor
with two belief-conditioned ways of using the **same pretrained frozen policy**.
It does not retrain the model or change original baseline CSVs.

## Conditions

- `hard`: existing `NoisySemanticLDBAWrapper`, which advances a single perceived
  LDBA state with corrupted categorical proposition assignments.
- `map`: `BeliefSemanticLDBAWrapper`, using the embedding of the most-probable
  state in its approximate belief distribution.
- `weighted`: the same filter, using the probability-weighted average of state
  embeddings. This is not equivalent to a mixture of policy outputs and may be
  out of distribution for the pretrained policy.

**Inference assumption.** At each physical step the filter assumes a uniform,
independent prior over the 13 possible true LetterWorld proposition assignments.
With corruption probability `p`, its posterior over assignments given an observed
label `y` is `P(a=y|y) = 1-p` and `P(a!=y|y) = p/12`. Belief propagation is
`b_next(q') = sum_{q,a} b(q) P(a|y) 1[delta(q,a)=q']`.
This is an **approximate filter**, not the true posterior given LetterWorld's
spatial dynamics. The policy still receives the usual egocentric spatial map.

The pilot deliberately **excludes formula index 5**, the only formula reported
to contain epsilon transitions. The remaining selected formulas are checked
against the generated automata for epsilon edges before evaluation. The epsilon
handling of belief-aware actions is therefore **not evaluated** here.

## Commands (repository root)

Run the standalone statistical tests and JAX wrapper integration tests:

```bash
JAX_PLATFORMS=cpu pixi run pytest \
  experiments/belief_semltl/pilot_stats_test.py \
  src/jaxolotl/alg/sem_ltl/wrappers/belief_semantic_ldba_wrapper_test.py -q
```

Small end-to-end smoke test (2 seeds, formula indices 0 and 3, 4 episodes,
noise 0 and 0.1):

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_pilot.py --smoke
```

Full pilot (10 pretrained model seeds, 9 epsilon-free finite-LTL formulas,
16 episodes, noise probabilities 0, 0.1, and 0.3):

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_pilot.py
```

Each condition shares the same JAX evaluation key (`100`), matching the
original noise sweep. This pairs the reset/action-randomness initialization,
although different policies can visit different states after their actions
diverge. At **p=0** the script enforces exact episode-level equality of
success, violation, length, and disagreement across all three methods. Any
mismatch fails immediately before running p>0 or writing results.

## Outputs

Smoke outputs are placed in `experiments/belief_semltl/results/belief_pilot/smoke/`;
full results are in `experiments/belief_semltl/results/belief_pilot/full/`:

- `episodes.csv`: one row for each seed, formula, episode, method and noise level.
- `summary.csv`: seed-level Student-t confidence intervals for success,
  violation and episode disagreement, overall and by formula.
- `paired_effects.csv`: per-seed paired treatment-minus-`hard` differences
  in success, overall and by formula; intervals use the number of pretrained
  seeds, **not** the number of episodes, as the sample size.
- `success_comparison.png` and `.pdf`: overall success by noise level and method.
- `metadata.json`: seed identifiers, evaluation key, formulas, and run context.

**Interpretation caution.** Differences from hard progression are preliminary.
The observation model is misspecified relative to LetterWorld dynamics, and
averaged embeddings can be outside the training distribution. Formula-level
results are descriptive because the nine formulas are fixed and heterogeneous.

The scripts have no license to edit prior `noise_sweep.csv` or `summary.csv`.
Do not report the result of a smoke test as the full experiment.
