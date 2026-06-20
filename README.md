# 68 EBM Transformer Compositional Grasping

Submission-hardening version: v5 expanded frozen MuJoCo archive.

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository is retained as a negative submission-readiness archive for a falsified compositional grasping mechanism. The v5 rebuild expands the original paper into a 35-page ICLR-style manuscript with bright boxed clickable citations, real MuJoCo rollout evidence, rollout-labeled training data, analytic and learned baselines, hostile ablations, stress sweeps, seed-level statistics, explicit theory claims, and a pre-specified terminal decision rule.

The proposed `ebm_transformer_compositional_v5` ranker does not survive the ICLR-main gate. On the frozen aggregate benchmark it reaches 0.667 success, while the strongest non-oracle aggregate baselines `ensemble_uncertainty_ranker` and `gradient_boosted_energy_model` reach 0.747. On the combined-composition shift, v5 reaches 0.067 while `calibrated_stacked_ranker` and `ensemble_uncertainty_ranker` reach 0.075. The hostile ablation suite also falsifies the mechanism claim because `monolithic_scalar_energy_only` beats full v5 and `no_cem_fallback` matches full v5.

## Frozen Evidence

- Seeds: 8
- Training rollouts: 1,152
- Main evaluation rows: 18,480
- Ablation rows: 4,680
- Stress rows: 4,320
- Negative cases: 4
- Final PDF: `C:/Users/wangz/Downloads/68.pdf`
- PDF pages: 35
- PDF SHA256: `53ED77ED5ACDEDC571539647ADB67844C0D9AFC0CDAB38579876C1391924F2AE`
- Validation: `python scripts\validate_submission_artifacts.py` passes counts, figures, TeX links, Downloads PDF, and Desktop hygiene.

## Reproduce Frozen Evidence

```powershell
python src\run_experiment.py --seeds 8 --episodes 15 --ablation-episodes 15 --stress-episodes 10 --train-scenes 96 --train-candidates 12 --main-candidates 22 --oracle-candidates 7 --workers 4
```

The run writes training rollouts, main raw rollouts, seed metrics, pairwise comparisons, aggregate metrics, ablations, stress sweeps, negative cases, training summaries, and figures into `results/` and `figures/`.

## Rebuild Archive PDF

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
```

The build renders generated LaTeX tables from the frozen CSV artifacts, compiles the manuscript, and writes the canonical numbered PDF to `C:/Users/wangz/Downloads/68.pdf`. It does not copy the PDF to Desktop.
