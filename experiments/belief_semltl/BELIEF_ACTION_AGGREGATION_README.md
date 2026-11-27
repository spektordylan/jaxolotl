# Phase 2 extension: GRU belief-weighted frozen-policy action probabilities

## Hypothesis

The current learned state tracker feeds the frozen SemLTL actor the **single**
most likely LDBA state. To test whether it helps to account for state uncertainty,
we instead compute the actor's physical-action distribution for the top `k`
valid LDBA states and weight those **probabilities** by the GRU posterior.

\[
\pi_{\mathrm{agg}}(a\mid o,h) = \sum_{q\in\mathrm{TopK}(\hat b(h))}
\frac{\hat b(q)}{\sum_{r\in\mathrm{TopK}}\hat b(r)}\;
\pi_\theta(a\mid o,e(q)).
\]

`alpha` interpolates *action probabilities*, not logits or embeddings:
`(1-alpha) * pi_MAP + alpha * pi_agg`. Default is `alpha=1`, `top-k=3`.
This is a policy mixture, **not** exact POMDP planning or QMDP.

## Implementation details

- Reuses `models/full/learned_estimator.eqx` with **no retraining**.
- The hidden state is already maintained by `LearnedTrackerWrapper`; its linear
  head recovers the current GRU state logits. Invalid/padded state probabilities
  are masked before softmax.
- On the initial step, where the tracker has not consumed a detector reading,
  the known initial automaton state is used as a one-hot distribution.
- At `p=0`, bypasses the GRU posterior with the known hard state; exact
  episode parity is enforced against both hard progression and GRU-MAP.
- For each candidate state, `eqx.tree_at` substitutes a **valid LDBA semantic
  embedding** into the frozen policy observation, while preserving the same
  original spatial observation and epsilon features.
- The actor's `log_prob((physical_action, epsilon_noop))` is evaluated for all
  4 LetterWorld physical actions. Converted probabilities are normalized and
  mixed, then an argmax (deterministic) or a categorical draw (stochastic) is
  used. This is *not* a mixture of logits and does *not* sample a latent LDBA
  state per step.
- No true letter, true LDBA state, or current-cell proposition feature is given
  to the **tracker/aggregator**; the frozen actor still sees its ordinary grid.
- Only epsilon-free formulas are supported; formula 5 is excluded.

## Commands (from the Jaxolotl repository root)

```bash
# Commit the previously working GRU experiment first, after reviewing staged files.
git status --short

# Cheap pure JAX test (does not need SemML compilation):
JAX_PLATFORMS=cpu pixi run pytest -q \
  experiments/belief_semltl/belief_aggregation_math_test.py

# End-to-end integration smoke test: 2 seeds x 2 formulas x 4 episodes
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_belief_action_aggregation.py --smoke

# Full evaluation (defaults to the existing checkpoint's validation seeds 7,8,9):
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_belief_action_aggregation.py
```

### Sanity check: top-1 reduces to GRU-MAP

Run this **after** the first smoke test, with a new output path:

```bash
JAX_PLATFORMS=cpu pixi run python \
  experiments/belief_semltl/run_belief_action_aggregation.py \
  --smoke --top-k 1 \
  --output-dir experiments/belief_semltl/results/learned_aggregation/smoke_top1
```

The runner asserts episode-level MAP parity for deterministic top-1 trials.
It also asserts hard/MAP/aggregation parity at `p=0` and refuses to overwrite
prior artifacts. If a check fails, **do not trust** subsequent results.

## Files saved

New directory: `experiments/belief_semltl/results/learned_aggregation/{smoke,full}`

- `episodes.csv`: matched initialization keys, success, violation, per-episode
  state accuracy and length for **hard, GRU-MAP, aggregation**.
- `summary.csv`: mean and 95% Student-t interval over pretrained model seeds.
- `paired_effects.csv`: paired seed-level effects (MAP-hard, aggregation-hard,
  aggregation-MAP) with 95% intervals.
- `metadata.json`: configuration, checkpoint and limitations.

Use `--output-dir ...` to preserve earlier results when changing `top-k`,
`alpha`, or rerunning. No existing checkpoint, dataset or results are changed.

## Scientific caveats

A learned GRU softmax is not guaranteed to be a calibrated belief, and truncating
to top-k changes its posterior. Mixing state-conditioned policies minimizes no
particular Bellman objective. It can help, hurt, or have no effect. The saved
checkpoint was selected on the same seed group (7..9) used in the previous
closed-loop benchmark, so report these results as exploratory/validation,
not an untouched external test. Seed-level intervals with only 3 policy seeds
are necessarily uncertain.

### Runtime-compatibility note

This patch was compiled and unit-tested in an environment without the complete
Jaxolotl/Pixi/Equinox runtime. It relies on the frozen SemLTL policy's action
object supporting `.log_prob((physical_action, epsilon_action))`. Verify this
with the `--smoke` command; if your local Jaxolotl revision exposes a different
joint-distribution API, stop and inspect the traceback before a full run.
