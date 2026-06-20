# Paper 68 Expanded Submission Plan

Date: 2026-06-20

Paper: EBM Transformer Compositional Grasping

Target venue style: ICLR main-conference submission package

Terminal policy: optimize for hostile-review survival, not attractive numbers. If the expanded evidence fails the frozen gates, the terminal state remains KILL_ARCHIVE.

## Objective

Rebuild Paper 68 from a short v4 negative MuJoCo note into a 25+ page ICLR-style evidence package. The package must contain real CPU-only experiments, strong analytic and learned baselines, hostile ablations, stress sweeps, paired statistics, generated tables, verified references, bright boxed clickable citations, a Downloads-only numbered PDF, a public GitHub repository, and updated root ledgers.

## Current Problem

The v4 paper is honest but too small for submission readiness. It uses only 5 seeds, 10 episodes per seed, 7 splits, 9 methods, 3,150 main rows, 400 ablation rows, and 1,200 stress rows. It shows that the compositional EBM/Transformer ranker loses to force closure and MLP energy on combined composition shift, but it does not yet provide enough scale, theory, ablation pressure, or reference quality to survive serious review.

## Expanded Evidence Plan

1. Increase main evidence scale while keeping RAM light.

- Use CPU-only MuJoCo rollouts.
- Use process parallelism capped at a small worker count.
- Add CLI controls for seeds, episodes, splits, stress levels, output directories, and workers.
- Target at least 18,000 main rows, 4,000 ablation rows, and 4,000 stress rows for the frozen run.

2. Expand the benchmark beyond the current seven splits.

- Retain seen_simple, unseen_dimensions, unseen_shape_family, slippery_contact, clutter_collision, task_constraint_shift, and combined_composition_shift.
- Add long_bar_high_aspect, thin_handle_shift, adversarial_clutter_gap, and material_shift_proxy.
- Preserve simulator realism boundaries instead of pretending this is hardware evidence.

3. Strengthen baselines.

- Retain random_grasp, antipodal_geometry, force_closure_score, cem_grasp_search, mlp_energy_model, transformer_policy_ranker, ensemble_uncertainty_ranker, ebm_transformer_compositional, and oracle_mujoco_grid.
- Add gradient_boosted_energy_model using the already trained HGB classifier.
- Add calibrated_stacked_ranker combining MLP, transformer, HGB, and analytic CEM signals.
- Add robust_perturbation_ranker using worst-case candidate scores under center, angle, jaw-width, and friction perturbations.
- Add collision_aware_force_closure as a stronger analytic baseline.
- Add ebm_transformer_compositional_v5 as the repaired proposed method.

4. Improve the proposed method only before protocol freeze.

- Repair obvious v4 weaknesses: collision underweighting, lack of robust perturbation scoring, weak fallback when every candidate is collision-prone, and brittle learned-context dominance.
- Do not tune after the protocol freeze.
- Keep all development runs separate from frozen outputs.

5. Add hostile ablations.

- Test full v5 against removals of object, contact, task, collision, feasibility, transformer context, robust perturbation, calibrated stack, fallback, and monolithic scalar variants.
- Run ablations on combined_composition_shift, adversarial_clutter_gap, and material_shift_proxy.
- The ablation gate fails if any removed-component variant matches or beats full v5 within the predefined tolerance.

6. Add stress testing.

- Sweep friction, clutter clearance, aspect ratio, sensor noise, and task constraint strength.
- Compare v5 against force closure, CEM, MLP, HGB, calibrated stack, robust perturbation, ensemble uncertainty, and oracle upper bound.
- The stress gate fails if v5 is not on the non-oracle success/safety frontier at maximum stress.

7. Add theory without pretending it proves hardware transfer.

- Formalize the compositional ranking objective as a product-of-experts or additive-energy selection rule.
- Prove a narrow sufficient condition: if each energy upper-bounds its failure factor and the weighted sum is calibrated, minimizing the sum controls a union-bound failure certificate.
- Prove the negative limitation: miscalibrated or redundant energies can be dominated by a scalar learned ranker or robust analytic baseline.
- Tie theory directly to ablation and calibration tests.

8. Build the submission artifact.

- Generate all tables from CSVs.
- Use bright boxed clickable citation links in LaTeX.
- Replace placeholder references with real grasping, energy-based modeling, Transformer, Dex-Net/grasp-quality, domain/generalization, and MuJoCo references.
- Build `C:/Users/wangz/Downloads/68.pdf` only.
- Verify no visible Desktop PDF exists.

## Frozen Decision Gates

The paper can be at most STRONG_REVISE, never ICLR-main-ready, unless external robot or recognized public benchmark evidence is added. Under the CPU-only local MuJoCo scope, the final decision is:

- KILL_ARCHIVE if v5 fails to beat the strongest non-oracle aggregate baseline by at least 0.030 success.
- KILL_ARCHIVE if v5 fails to beat the strongest non-oracle combined-composition baseline by at least 0.030 success.
- KILL_ARCHIVE if v5 does not improve safety failures relative to the strongest non-oracle baseline within 0.020.
- KILL_ARCHIVE if any removed-component ablation matches or beats full v5 within 0.020 success.
- KILL_ARCHIVE if v5 loses the maximum-stress gate to any non-oracle baseline.
- STRONG_REVISE only if all local gates pass; still not submission-ready without external validation.

## Deliverables

- Expanded runner with CPU-only/RAM-light controls.
- Development log and frozen protocol document.
- Frozen results CSVs, seed metrics, pairwise statistics, ablations, stress sweeps, diagnostics, negative cases, and figures.
- 25+ page ICLR-style PDF in Downloads only.
- Validation script for row counts, figures, TeX link settings, PDF page count, and Desktop hygiene.
- Updated README, child status, submission decision, root ledgers, and public GitHub repository.
