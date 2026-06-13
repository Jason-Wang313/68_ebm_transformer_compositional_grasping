# Claims

- Mechanism claim under test: composing object, contact, task, collision, and feasibility energies with a transformer context should improve grasp ranking under held-out object/task compositions.
- Real-evidence result: the v4 MuJoCo grasping benchmark falsifies the claim as implemented. On combined composition shift, `ebm_transformer_compositional` reaches 0.080 success, below `force_closure_score` and `mlp_energy_model` at 0.140.
- Ablation result: removing several energy terms does not create a decisive pattern, and the full method does not beat strong non-oracle baselines in the main gate.
- Scope claim: results support archiving this specific EBM/Transformer mechanism, not deployment.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance.
