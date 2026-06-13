# Paper 68 Terminal Evidence

Decision: `KILL_ARCHIVE`

## Real-Evidence Rebuild
The v4 rebuild replaces the synthetic scaffold with a MuJoCo parallel-jaw grasping benchmark. It trains compact rankers on rollout-labeled MuJoCo grasp candidates, then executes selected grasps in MuJoCo across object, contact, task, clutter, and combined composition shifts.

Run command:

```powershell
python src\run_experiment.py
```

Generated evidence:
- 720 MuJoCo training rollouts.
- 3,150 main MuJoCo evaluation rows.
- 400 ablation rows.
- 1,200 stress-sweep rows.
- 5 seeds, 10 main episodes per seed, 7 splits, 9 main methods.
- CSVs: training rollouts, training summary, raw main rollouts, metrics, seed metrics, pairwise comparisons, ablations, stress sweep, negative cases.
- Figures: success by split, ablation success, stress sweep, safety failures.

## Combined Composition-Shift Results

| Method | Success | CI95 | Slip | Drop | Collision | Energy |
|---|---:|---:|---:|---:|---:|---:|
| `random_grasp` | 0.020 | 0.039 | 0.114 | 0.800 | 0.940 | 1.749 |
| `antipodal_geometry` | 0.080 | 0.076 | 0.094 | 0.560 | 0.840 | 3.188 |
| `force_closure_score` | 0.140 | 0.097 | 0.086 | 0.360 | 0.780 | 3.545 |
| `cem_grasp_search` | 0.100 | 0.084 | 0.081 | 0.360 | 0.820 | 3.960 |
| `mlp_energy_model` | 0.140 | 0.097 | 0.075 | 0.400 | 0.840 | 3.625 |
| `transformer_policy_ranker` | 0.060 | 0.067 | 0.076 | 0.420 | 0.900 | 2.737 |
| `ensemble_uncertainty_ranker` | 0.100 | 0.084 | 0.083 | 0.440 | 0.820 | 3.564 |
| `ebm_transformer_compositional` | 0.080 | 0.076 | 0.085 | 0.420 | 0.820 | 3.574 |
| `oracle_mujoco_grid` | 0.200 | 0.112 | 0.050 | 0.120 | 0.740 | 4.068 |

Pairwise combined-shift comparisons show no reliable advantage for the EBM method over analytic or learned baselines.

## Ablation Results

| Ablation | Success | CI95 | Collision |
|---|---:|---:|---:|
| `full_ebm_transformer_compositional` | 0.180 | 0.108 | 0.740 |
| `no_transformer_context` | 0.100 | 0.084 | 0.840 |
| `no_collision_energy` | 0.080 | 0.076 | 0.900 |
| `no_contact_energy` | 0.080 | 0.076 | 0.840 |
| `no_object_energy` | 0.080 | 0.076 | 0.820 |
| `no_task_energy` | 0.080 | 0.076 | 0.840 |
| `no_feasibility_energy` | 0.060 | 0.067 | 0.880 |
| `monolithic_scalar_energy_only` | 0.040 | 0.055 | 0.920 |

## Terminal Rationale
The central claim requires the compositional EBM/Transformer to beat strong non-oracle analytic and learned grasp rankers under held-out object/task composition. It does not. The honest action is `KILL_ARCHIVE`.
