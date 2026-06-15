# Paper 68 Terminal Audit

Date: 2026-06-15

Paper: `68_ebm_transformer_compositional_grasping`

Decision: `KILL_ARCHIVE`

ICLR-main ready: no

## Commands Executed

- `python -m py_compile src\run_experiment.py`
- CSV finite/schema audit over `results/training_rollouts.csv`, `results/training_summary.csv`, `results/ebm_grasping_raw.csv`, `results/ebm_grasping_metrics.csv`, `results/ebm_grasping_pairwise.csv`, `results/ebm_grasping_ablation.csv`, `results/ebm_grasping_ablation_raw.csv`, `results/raw_seed_metrics.csv`, `results/negative_cases.csv`, compatibility CSVs, and `results/stress_sweep.csv`.
- `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper`
- `Copy-Item paper\main.pdf C:\Users\wangz\Downloads\68.pdf -Force`

## Verified Evidence

- Real MuJoCo parallel-jaw grasp rollouts are implemented in `src/run_experiment.py`.
- Training evidence contains 720 rollout-labeled grasp candidates with a `0.3944` success rate and 31-dimensional features.
- Main evidence contains 3,150 MuJoCo rollouts: 7 splits, 5 seeds, 10 episodes per seed/split/method, and 9 methods.
- Ablation evidence contains 400 combined-shift rollouts.
- Stress-sweep evidence contains 1,200 rollouts.
- Baselines include random grasp, antipodal geometry, force closure, CEM grasp search, MLP energy, Transformer policy ranking, ensemble uncertainty, and oracle MuJoCo grid search.
- CSV outputs are present, non-empty, and finite.
- BibTeX warnings from missing prior-work sort keys were fixed without inventing authors.
- The wide main-results table was tightened to remove the large overfull table warning.
- The rebuilt PDF is `C:/Users/wangz/Downloads/68.pdf`.
- `C:/Users/wangz/Desktop/68.pdf` is absent.

## Fatal Results

The EBM/Transformer compositional grasping claim is falsified as an ICLR-main submission:

- Combined composition shift: `ebm_transformer_compositional` reaches `0.080 +/- 0.076` success.
- Combined composition shift: `force_closure_score` reaches `0.140 +/- 0.097` success.
- Combined composition shift: `mlp_energy_model` reaches `0.140 +/- 0.097` success.
- Combined composition shift: `cem_grasp_search` reaches `0.100 +/- 0.084` success.
- Combined composition shift: `ensemble_uncertainty_ranker` reaches `0.100 +/- 0.084` success.
- The oracle MuJoCo grid reaches `0.200 +/- 0.112`, showing room for better selection but not by the proposed method.
- Pairwise seed comparisons show no reliable advantage over analytic or learned non-oracle baselines.

## Internal Signal

The result is not a pure implementation null. In the combined-shift ablation suite, `full_ebm_transformer_compositional` reaches `0.180 +/- 0.108`, while `monolithic_scalar_energy_only` reaches `0.040 +/- 0.055` and several energy removals are weaker. This means the energy terms are not meaningless internally.

That internal signal is still insufficient for ICLR-main readiness because the main external gate requires beating force closure, CEM, MLP energy, Transformer ranking, and ensemble uncertainty on the held-out composition shift. The method does not clear that gate.

## Gate Decision

This paper satisfies the local evidence-package requirements for a real negative result: high-fidelity simulator evidence, rollout-labeled training data, learned and analytic baselines, ablations, stress tests, uncertainty, negative cases, rebuilt PDF, corrected BibTeX metadata, corrected table formatting, corrected hostile-review documentation, and public repository.

It does not satisfy `STRONG_REVISE` because the mechanism remains externally non-competitive despite partial internal ablation signal. The correct terminal state remains `KILL_ARCHIVE`.

Required revival work:

- improve candidate search and learned scoring enough to beat force closure, CEM, MLP energy, Transformer ranking, and ensemble uncertainty;
- prove the composed energy mechanism is necessary on the same external benchmark used for baselines;
- validate on hardware or a public grasping benchmark;
- add raw perception, tactile/material sensing, or richer scene geometry rather than abstract-only features;
- perform a manual full-paper related-work synthesis.
