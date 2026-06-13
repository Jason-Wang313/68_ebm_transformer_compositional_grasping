# Paper 68 Rebuild Plan: EBM Transformer Compositional Grasping

## Terminal Objective
Rebuild `68_ebm_transformer_compositional_grasping` into a real evidence package. The paper may be submission-ready only if an energy-based compositional model improves grasp selection under object/contact/task composition beyond strong geometric, learned, uncertainty, and sampling baselines. If the EBM/Transformer mechanism is matched by simpler energy models, heuristics, or robust sampling, archive it.

## Central Claim Under Test
A grasp should not be represented as one collapsed policy token. The claimed mechanism is to compose separate energies for object geometry, contact stability, task constraint, collision/clutter, and gripper feasibility, then search for grasps that jointly satisfy those factors. The test is whether this explicit compositional energy improves closed-loop grasp success under held-out object and task compositions.

## High-Fidelity Benchmark
- Build a MuJoCo parallel-jaw grasping benchmark with real contact rollouts.
- Candidate grasps are parameterized by planar pose, approach angle, jaw width, closing force, and lift direction.
- Object families:
  - boxes / cuboids
  - cylinders
  - thin bars
  - asymmetric L/T composites
  - slippery objects
  - clutter/near-collision scenes
  - held-out combined compositions
- Task constraints:
  - lift-only
  - reorient before lift
  - avoid forbidden contact face
  - grasp handle/long side
  - clutter-safe extraction
- Each episode samples candidates, evaluates selected grasp by executing close-and-lift in MuJoCo, and records success, slip, drop, collision, force, and energy.

## Methods And Baselines
- `random_grasp`: lower bound.
- `antipodal_geometry`: analytic antipodal/contact-normal heuristic.
- `force_closure_score`: analytic force-closure/friction-cone score.
- `cem_grasp_search`: cross-entropy method over grasp candidates using analytic quality.
- `mlp_energy_model`: learned scalar energy over flattened grasp/object/task features.
- `transformer_policy_ranker`: learned transformer ranker without explicit compositional energy terms.
- `ensemble_uncertainty_ranker`: ensemble of learned rankers with variance penalty.
- `ebm_transformer_compositional`: proposed model with object/contact/task/collision/feasibility energy terms.
- `oracle_mujoco_grid`: non-submission upper bound that evaluates many candidate grasps in MuJoCo.

## Required Experiments
- Generate a compact training set from MuJoCo candidate rollouts, with train/test split by object family and task composition.
- Main benchmark: at least 5 seeds, 10-12 episodes per seed, 7 splits, all main methods, and real MuJoCo execution for chosen grasps.
- Splits:
  - seen simple objects
  - unseen dimensions
  - unseen shape family
  - slippery contact
  - clutter collision
  - task constraint shift
  - combined composition shift
- Ablations:
  - no object energy
  - no contact energy
  - no task energy
  - no collision energy
  - no feasibility energy
  - no transformer/context encoder
  - monolithic scalar energy only
- Pairwise seed comparisons against `mlp_energy_model`, `transformer_policy_ranker`, `ensemble_uncertainty_ranker`, `cem_grasp_search`, and `force_closure_score`.
- Stress sweep over friction, clutter clearance, object aspect ratio, and task constraint strength.
- Negative cases: deformable object, occluded geometry, adversarial low-friction coating, and semantic task ambiguity.

## Submission-Readiness Gate
To be ICLR-main ready, the proposed method must:
- beat every non-oracle baseline on combined composition shift and at least four of six non-nominal splits
- show that compositional energy terms matter through ablations
- avoid trading success for unsafe force, collision, or excessive sampling budget
- show uncertainty across seeds and pairwise statistics against strong learned baselines
- include honest hostile prior-work discussion and limitations

## Terminal Decision Rules
- `SUBMISSION_READY_CANDIDATE`: only if the EBM/Transformer clears all empirical gates and the paper can support a strong contribution.
- `STRONG_REVISE`: if compositional energy helps but lacks hardware/public benchmark breadth, manual related-work depth, or enough learned-model rigor.
- `KILL_ARCHIVE`: if MLP energy, transformer ranker, CEM, force-closure, ensemble uncertainty, or monolithic-energy ablations match the proposed method.

## Resource Discipline
Keep RAM light with a compact MuJoCo model, small learned models, cached candidate features, compact CSVs, and worker caps. Do not reduce rigor: preserve real rollouts, seeds, strong baselines, ablations, stress sweeps, uncertainty, and terminal-failure analysis.

## Deliverables
- Rewritten `src/run_experiment.py` with MuJoCo grasp rollouts, training/evaluation pipeline, and implemented baselines.
- Updated requirements, README, child status, claims, gate, readiness, audit, and terminal evidence docs.
- CSV results, pairwise comparisons, ablations, stress sweep, negative cases, figures, and learned-model summaries.
- Rewritten paper and compiled `C:/Users/wangz/Downloads/68.pdf` only.
- Public GitHub repo pushed with final commit.
- Root reports updated before Paper 69 starts.
