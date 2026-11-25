# Oracle-monitor ablation (privileged reference)

**Research chronology:** See [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for the complete sequence: noisy hard baseline → frozen-policy MAP/weighted comparison (negative) → shared-trajectory state-estimation diagnostic (MAP deficit) → oracle ablation. Prior `PILOT_README.md` and `SHARED_TRAJECTORY_README.md` also document their individual experiments. This patch intentionally does not overwrite the existing baseline `README.md`.

## Rationale and hypothesis

The uniform-prior Bayesian filter was less accurate than the deterministic monitor on shared trajectories, and both MAP and weighted embeddings reduced closed-loop task success under noise. However, those findings alone do not establish whether accurate progress tracking could help the *frozen* SemLTL actor. This experiment provides a **privileged reference** for what this same fixed network can do when supplied the true automaton state at every decision point.

**Hard:** same pretrained policy receives `embedding(q_perceived)` from the noisy observed proposition monitor. **Oracle:** *identical* pretrained policy receives `embedding(q_true)` (plus true-state epsilon features). Both run the same underlying LetterWorld with the same true-state rewards, violation and termination. The true LDBA state is known to the oracle by construction; this is *not* a practical inference method. Environment grid observations are unchanged. The latent noisy monitor still runs under the oracle wrapper to keep noise sampling and physical logic consistent; its disagreement is not an oracle error.

Only nine epsilon-free finite formulas are evaluated: indices `[0,1,2,3,4,6,7,8,9]`. Formula index 5, the only epsilon-containing formula, is excluded to avoid ambiguity in automaton epsilon semantics and match the previously validated pilot.

## Apply and run

From the local Jaxolotl repository root:

```bash
unzip -l ~/Downloads/semltl_oracle_monitor_ablation_patch.zip
unzip -o ~/Downloads/semltl_oracle_monitor_ablation_patch.zip -d .

JAX_PLATFORMS=cpu pixi run pytest \
  experiments/belief_semltl/oracle_stats_test.py \
  src/jaxolotl/alg/sem_ltl/wrappers/oracle_semantic_ldba_wrapper_test.py -q

JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_oracle_ablation.py --smoke
JAX_PLATFORMS=cpu pixi run python experiments/belief_semltl/run_oracle_ablation.py
```

The smoke run uses two pretrained seeds, two formulas (`0,3`), four episodes per condition, `p=0,.30`. The full run uses ten seeds, nine formulas, 16 episodes per seed/formula/condition, noise probabilities `0,.10,.30`, two modes, for **8,640 evaluation episodes**. Output goes to `experiments/belief_semltl/results/oracle_ablation/{smoke,full}/`, never to the older pilot's directory. You can pass `--no-plot` or `--output-dir PATH`. You must include `p=0` to run the parity gate.

## Correctness gates

1. **No-noise parity:** when `p=0`, oracle and hard must have identical per-episode success, reward/return, length, true sink-violation and monitor-disagreement indicators. This checks that the oracle isn't changing dynamics, rewards or other unintended inputs.
2. **Oracle noise invariance:** with the same evaluation random-key schedule and a perfect task-progress input, oracle per-episode outcomes must match its `p=0` outcomes for every tested `p>0`. The oracle must not be influenced by the sampled noisy proposition labels.
3. **Oracle disagreement:** The oracle monitor's state is the true state by construction, so it never disagrees. The unused latent corrupted monitor's disagreement is stored in `transition.info['latent_noisy_disagreement']` but **not** as evaluator `disagreement`.
4. **Epsilon exclusion:** If any selected compiled LDBA has epsilon edges, stop.

If any gate fails, do **not** interpret the ablation as a successful or valid experiment. Send the first traceback/failed assertion. The wrapper's standalone integration tests also inspect whether the semantic embedding follows the true rather than perceived state, and that rewards/terminations match the base noisy wrapper for forced identical actions.

## Outputs and statistics

- `episodes.csv`: success, return, violation, episode length and oracle/hard monitor disagreement for every `noise x method x seed x formula x episode`.
- `summary.csv`: success, violations, lengths, disagreement means and 95% Student-t intervals across **pretrained model seeds**, overall and per formula. Do not treat all episodes as independent samples.
- `paired_effects.csv`: oracle-minus-hard success difference with 95% *seed-paired* intervals, overall and per formula.
- `metadata.json`: formulas, seeds, evaluation key, git commit, validation outcomes, information-leakage boundary and limitations.
- `oracle_vs_hard_success.png/.pdf`: graphical comparison.

The same `jax.random.key(100)` is provided to both policies for each evaluation condition, following `run_pilot.py`. It matches initial reset and action-sampling key *schedules*, not necessarily the physical trajectories once actions diverge. A noisy-monitor error can cause different navigation behavior, which is the causal mechanism being evaluated.

## Interpretation guardrails

- **Large oracle advantage** indicates potential *headroom* if task-state inference could be made more accurate or exploited better by the frozen network. It does not prove that any feasible Bayesian filter can recover that headroom.
- **Small oracle advantage** suggests correcting only semantic task progress may be insufficient to substantially improve this frozen policy (though learned or modified policies might still succeed).
- **Oracle should not decline with p** under this task-monitor-only corruption setup. If it does, investigate an unintended noise-dependent input or a reproducibility failure before using numbers.
- This is neither a new trainable algorithm nor an upper bound over all possible controllers. It intentionally gives the policy hidden ground-truth task information and cannot be presented as a deployable robustness solution.
- Original baseline 10-formula aggregate includes provisional epsilon formula 5; the fair direct comparison for this ablation is the matched **nine-formula** hard baseline, not the 10-formula result.

**Validation status at packaging:** Python source syntax verified and standalone statistics tested; full integration must run locally under the user's Pixi/Jaxolotl installation. The ablation results are not yet known and the research log must be updated from actual results after running.
