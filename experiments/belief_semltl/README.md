# Uncertainty-Aware SemLTL: Noisy-Observation Baseline

This directory contains experiments evaluating **frozen, pretrained SemLTL policies** under noisy atomic-proposition observations in **LetterWorld**.

**Validation status (November 24, 2026):** The noisy-observation baseline and initial 11,200-episode sweep are complete. A narrow correction to invalid epsilon-action handling has been prepared in `noisy_semantic_ldba_wrapper.py`, along with a regression test. **The corrected code must be tested in the local Jaxolotl environment; the affected evaluations have not yet been rerun.** The results below are the original, preliminary results.

---

## 1. Research Context & Motivation

SemLTL (Semantically Labelled Automata for LTL-Guided Reinforcement Learning) uses limit-deterministic Büchi automata (LDBAs), semantic state embeddings, and a goal-conditioned neural policy. The baseline assumes exact proposition valuations when progressing the task automaton. Here, we investigate how a frozen policy behaves when its temporal task monitor receives corrupted proposition observations, before considering explicit belief-state representations.

## 2. Experimental Design

### 2.1 Categorical observation corruption

LetterWorld has 12 atomic propositions (`a`–`l`). Each grid cell has either a single letter or no letter, yielding 13 valid assignments (12 singleton assignments and the empty assignment).

For a true assignment index $a^*\in\{0,\ldots,12\}$, the observed assignment follows

$$
P(a_{\mathrm{obs}}=a'\mid a^*,p)=
\begin{cases}
1-p,&a'=a^*,\\
p/12,&a'\neq a^*.
\end{cases}
$$

An alternative assignment is drawn as $a'=(a^*+1+r)\bmod 13$ with $r$ uniform on $\{0,\ldots,11\}$. JAX PRNG keys make the experiment reproducible. Evaluated corruption probabilities are $p\in\{0,0.02,0.05,0.10,0.15,0.20,0.30\}$.

### 2.2 Ground-truth versus perceived automaton state

The wrapper tracks two automaton states:

- **$q_{\mathrm{true}}$** progresses with true propositions and determines acceptance, sink violations, reward, and termination.
- **$q_{\mathrm{perceived}}$** progresses with corrupted propositions and determines the semantic state embedding and epsilon-action mask provided to the frozen policy.

The evaluator uses rewards and sink flags from the true-state transition. The stateless SemLTL `Agent.update()` discards its `state` and `props` parameters rather than storing the ground-truth state. This is **task-monitor corruption**, not a full partially observable environment: the agent still receives the ordinary egocentric grid observation.

### 2.3 Epsilon transitions: intended semantics and correction

An LDBA **epsilon transition** is an internal automaton action that does not consume a new proposition observation or advance the physical environment. With noise, the perceived and true states can disagree. An epsilon action selected using the perceived-state mask may be unavailable from the true state.

For this experiment, the intended **state-local, indexed no-op convention** is:

1. Apply the selected epsilon transition to $q_{\mathrm{perceived}}$ (assuming the policy selected a permitted action).
2. If the corresponding epsilon successor exists at the same index from $q_{\mathrm{true}}$, use that successor.
3. If no true-state epsilon successor exists (the automaton returns `-1`), **keep $q_{\mathrm{true}}$ unchanged**. Crucially, do **not** fall back to the ordinary proposition-driven transition: an epsilon step consumes no proposition.
4. Keep the physical environment unchanged during the epsilon action.

The original wrapper used `get_next_epsilon_state()` directly for $q_{\mathrm{true}}$, which could assign the invalid `-1` sentinel when an epsilon edge did not exist. The prepared fix introduces `_resolve_true_epsilon_successor()` and tests that an unavailable true epsilon is a no-op, a valid epsilon advances, and a non-epsilon action still uses ordinary proposition progression.

This is an explicit modelling choice rather than a general semantics for belief-state LDBAs. When two different states both admit a given epsilon index, the associated commitments may still differ; this issue requires attention in future belief-aware work.

**Scope of provisional results:** Among the ten evaluated finite-LTL specifications, the initial automaton inspection found epsilon transitions only for **$\varphi_5$** (the **sixth** formula, using zero-based indexing). Its original results, and aggregates that include it, remain provisional until the corrected run is verified. The other nine formulas do not directly use this epsilon logic.

### 2.4 Evaluation and uncertainty

The sweep uses ten pretrained policy seeds, ten finite-LTL formulas, and 16 episodes per seed/formula/noise level: **11,200 episodes total**. Reported 95% confidence intervals are two-sided Student's $t$ intervals over seed-level summaries. `disagreement` is an **episode-level indicator**: it is one if $q_{\mathrm{true}}\ne q_{\mathrm{perceived}}$ at any step during an episode, rather than a timestep-wise disagreement fraction or a Bayesian belief-divergence measure.

The standalone original evaluation and the sweep use different PRNG evaluation keys; the initial zero-noise sweep reports success 0.794, while the original evaluator's separate run reported 0.789. This should not be interpreted as a treatment effect.

---

## 3. Original Results (pre-correction; provisional aggregates)

| Noise probability $p$ | Task success / acceptance | True violation rate | Mean successful episode length | Episode state disagreement |
|---:|---:|---:|---:|---:|
| 0.00 | 0.794 [0.770, 0.819] | 0.174 [0.148, 0.199] | 20.94 [20.10, 21.77] | 0.000 [0.000, 0.000] |
| 0.02 | 0.767 [0.748, 0.787] | 0.191 [0.169, 0.213] | 20.97 [20.02, 21.92] | 0.124 [0.108, 0.141] |
| 0.05 | 0.734 [0.708, 0.760] | 0.207 [0.187, 0.228] | 21.03 [20.12, 21.95] | 0.264 [0.243, 0.285] |
| 0.10 | 0.676 [0.644, 0.708] | 0.233 [0.206, 0.259] | 20.45 [19.54, 21.37] | 0.452 [0.419, 0.484] |
| 0.15 | 0.631 [0.601, 0.660] | 0.250 [0.223, 0.277] | 20.90 [19.81, 21.98] | 0.573 [0.542, 0.604] |
| 0.20 | 0.596 [0.574, 0.619] | 0.259 [0.230, 0.289] | 21.38 [20.24, 22.52] | 0.663 [0.637, 0.689] |
| 0.30 | 0.536 [0.516, 0.557] | 0.281 [0.245, 0.318] | 22.83 [21.20, 24.46] | 0.806 [0.791, 0.821] |

The original success rate falls by 25.8 percentage points from $p=0$ to $p=0.30$. As a sensitivity check, averaging the previously published **rounded** per-formula results excluding $\varphi_5$ gives approximately 84.2% at $p=0$ and 59.0% at $p=0.30$: a 25.2-point decrease. This suggests the aggregate degradation is not solely an epsilon-transition artifact, but these are not rerun or re-estimated confidence intervals.

### Interpretation, not yet causally established

- Global safety and long sequential formulas exhibit relatively large declines. Misperceived task progress is a plausible mechanism, not yet verified by conditional trajectory analysis.
- Branching formulas are more robust in these ten evaluated tasks, but formula complexity and baseline difficulty are confounded.
- At $p=0.30$, 80.6% of episodes experience at least one perceived/true automaton-state disagreement. This does not constitute a posterior belief measurement.

---

## 4. Files

```text
experiments/belief_semltl/
├── run_noise_sweep.py       # Frozen-policy evaluation sweep
├── plot_results.py          # PNG/PDF figure generation
├── README.md                # This document
├── results/
│   ├── noise_sweep.csv      # Episode-level results from original sweep
│   └── summary.csv          # Aggregates from original sweep
└── figures/
    ├── success_vs_noise.png / .pdf
    ├── violations_vs_noise.png / .pdf
    ├── per_formula_heatmap.png / .pdf
    ├── normalized_degradation.png / .pdf
    └── disagreement_vs_noise.png / .pdf
```

Additional code:

- `src/jaxolotl/alg/sem_ltl/wrappers/noisy_semantic_ldba_wrapper.py`
- `src/jaxolotl/alg/sem_ltl/wrappers/noisy_semantic_ldba_wrapper_test.py`

## 5. Reproduction and remaining validation

Run wrapper tests in the project's own Pixi environment:

```bash
JAX_PLATFORMS=cpu pixi run pytest \
  src/jaxolotl/alg/sem_ltl/wrappers/noisy_semantic_ldba_wrapper_test.py -q
```

Regenerate figures from **existing** CSVs without rerunning episodes:

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/plot_results.py
```

Rerun the full sweep if the epsilon fix changes affected outcomes or if updated aggregate results are required:

```bash
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_noise_sweep.py
```

**Before treating the tables as finalized:** (1) pass the updated regression test in the repository environment; (2) exercise the $\varphi_5$ epsilon path in an integration run; (3) recompute affected results and aggregate figures as needed. No retraining of the pretrained policy is required.

## 6. Next research direction

Evaluate explicit posterior beliefs over automaton states under the same observation-corruption model. Initial frozen-policy comparisons can include maximum-posterior state selection, probability-weighted semantic embeddings, and belief-weighted action distributions. None of these belief-aware methods is implemented or evaluated by the present baseline.
