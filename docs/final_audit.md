# Final Audit

1. Chosen thesis: EBM Transformer Compositional Grasping explores `Compose grasp energies over objects, contacts, and task constraints without collapsing them into one policy token.` for energy-based transformers for grasping.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4 real MuJoCo rebuild.
4. Reason: real MuJoCo grasping evidence falsifies the mechanism. On combined composition shift, the proposed method reaches 0.080 success while force closure and MLP energy reach 0.140; pairwise comparisons show no reliable advantage over strong non-oracle baselines.
5. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
6. Reproducibility: `python src\run_experiment.py` reproduces the MuJoCo training set, main rollouts, learned rankers, CSVs, figures, ablations, pairwise stats, stress sweep, and negative cases.
7. Claim-validity status: main-conference claims killed by direct empirical evidence; archive retained as a negative result.
8. 2026-06-15 continuation audit: code compilation, CSV finite/schema checks, BibTeX/PDF rebuild, table-width cleanup, Downloads-only PDF placement, and hostile-review documentation checks passed; terminal decision remains `KILL_ARCHIVE`.
9. Exact Downloads PDF path: `C:/Users/wangz/Downloads/68.pdf`
10. GitHub URL: https://github.com/Jason-Wang313/68_ebm_transformer_compositional_grasping
11. Confirmation: no visible Desktop copy was requested or made.
