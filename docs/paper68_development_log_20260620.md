# Paper 68 Development Log

Date: 2026-06-20

Policy: Development runs may repair obvious implementation defects before freeze. No result-chasing is allowed after the protocol freeze.

## Starting Point

The v4 archive is clean and reproducible, but too small for the expanded-standard submission hardening pass. Existing frozen-looking v4 evidence:

- 5 seeds.
- 10 episodes per seed.
- 7 main splits.
- 9 methods.
- 3,150 main MuJoCo rows.
- 400 ablation rows.
- 1,200 stress rows.
- Terminal decision: KILL_ARCHIVE.
- Primary reason: the EBM compositional ranker is matched or beaten by non-oracle baselines on combined composition shift.

## Planned Development Changes

- Add CLI configurability and output isolation.
- Add stronger baselines: HGB, calibrated stack, robust perturbation, collision-aware force closure.
- Add `ebm_transformer_compositional_v5`.
- Add hostile splits and multi-split ablations.
- Add aggregate metrics and all-split paired statistics.
- Add generated table and validation scripts after the frozen run.

## Development Runs

### Dev1: hard robust fallback

Command scale: 2 seeds, 2 main episodes, 2 ablation episodes, 2 stress episodes, 12 training scenes, 6 training candidates, 10 main candidates, 4 oracle candidates, 3 main splits, 2 ablation splits, and 2 stress levels.

Output: `results/dev_20260620_2201`

Result:

- Main rows: 168.
- Ablation rows: 104.
- Stress rows: 72.
- Terminal decision: KILL_ARCHIVE.
- v5 aggregate success: 0.4167.
- Strongest non-oracle aggregate baseline: `cem_grasp_search`, 0.4167.
- Combined-shift v5 success: 0.2500.
- Strongest combined-shift baselines: `cem_grasp_search`, `collision_aware_force_closure`, and `robust_perturbation_ranker`, 0.2500.
- Observed issue: hard fallback made many ablations identical to full v5, which is a mechanism-identifiability weakness.

### Dev2: blended fallback

Command scale: same as Dev1.

Output: `results/dev_20260620_2220`

Result:

- Main rows: 168.
- Ablation rows: 104.
- Stress rows: 72.
- Terminal decision: KILL_ARCHIVE.
- v5 aggregate success fell to 0.3333.
- Combined-shift v5 success fell to 0.0000.
- Safety failures worsened relative to the strongest baseline.
- The blended fallback preserved more scoring variation in principle but removed the only useful hard-split recovery behavior.

### Pre-freeze choice

The frozen implementation uses the Dev1 hard robust fallback. This is not because it makes the paper pass; it does not. It is the strongest pre-freeze implementation observed under the development checks. The fact that the fallback masks several component ablations is retained as a hostile-review failure mode rather than tuned away after the fact.
