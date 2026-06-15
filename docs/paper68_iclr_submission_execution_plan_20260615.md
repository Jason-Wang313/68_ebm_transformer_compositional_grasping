# Paper 68 ICLR-Main Execution Plan

Date: 2026-06-15

Paper: `68_ebm_transformer_compositional_grasping`

Goal: verify whether the current real MuJoCo EBM/Transformer compositional-grasping evidence can honestly support an ICLR-main-target submission, or whether the paper must remain `KILL_ARCHIVE` as a falsified negative result.

## Execution Gates

1. Reproducibility gate:
   - Compile `src/run_experiment.py`.
   - Confirm training, main, seed, pairwise, ablation, stress-sweep, negative-case, and compatibility CSV outputs exist.
   - Confirm all CSV outputs are non-empty and finite.
   - Rebuild the PDF from `paper/main.tex` with BibTeX.

2. Evidence gate:
   - Confirm the benchmark uses real MuJoCo parallel-jaw grasp rollouts rather than synthetic probability tables.
   - Confirm rollout-labeled training data, five seeds, seven grasping splits, nine main methods, confidence intervals, pairwise comparisons, ablations, stress sweeps, and negative cases.
   - Confirm baselines include random grasp, antipodal geometry, force closure, CEM grasp search, MLP energy, Transformer policy ranking, ensemble uncertainty, and oracle MuJoCo grid search.

3. Negative-claim gate:
   - Compare `ebm_transformer_compositional` against force closure, CEM, MLP energy, Transformer ranking, ensemble uncertainty, and oracle grid search under the combined composition shift.
   - Check whether internal ablations show any useful energy-component signal.
   - Keep the decision honest if internal ablation signal exists but external baseline performance is too weak for ICLR-main readiness.
   - Fix stale documentation that still presents the archive reason as synthetic-only evidence rather than the current real MuJoCo falsification.

4. Artifact gate:
   - Rebuild `paper/main.pdf`.
   - Copy only `C:/Users/wangz/Downloads/68.pdf`.
   - Confirm `C:/Users/wangz/Desktop/68.pdf` is absent.
   - Confirm the GitHub repository is public, clean, and pushed.

## Decision Rule

Upgrade only if the EBM/Transformer composition clearly beats analytic and learned non-oracle baselines under held-out composition shift and its ablations show the mechanism is necessary. If it is matched or beaten by force closure, CEM, MLP energy, Transformer ranking, or ensemble uncertainty, keep the terminal decision as `KILL_ARCHIVE`. If ablations show partial internal signal, document that signal as insufficient rather than erasing it.
