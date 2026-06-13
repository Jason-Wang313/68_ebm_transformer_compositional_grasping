# 68 EBM Transformer Compositional Grasping

Submission-hardening version: v4 real MuJoCo rebuild

Terminal decision: KILL_ARCHIVE for ICLR main conference.

The repository is retained as an archive of a falsified compositional grasping mechanism. The v4 rebuild replaces the synthetic probability scaffold with a MuJoCo parallel-jaw grasping benchmark, real rollout-labeled training data, learned rankers, analytic grasp baselines, ablations, stress sweeps, and a non-submission oracle grid.

The proposed EBM/Transformer compositional ranker does not survive the ICLR-main gate. On the combined composition shift it reaches 0.080 success, while force-closure and MLP energy baselines reach 0.140. Pairwise seed comparisons show no reliable advantage over analytic or learned baselines.

## Reproduce Real Evidence

```powershell
python src\run_experiment.py
```

The run writes MuJoCo training rollouts, main raw rollouts, seed metrics, pairwise comparisons, ablations, stress sweeps, negative cases, training summaries, and figures into `results/` and `figures/`.

## Rebuild Archive PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/68.pdf`
