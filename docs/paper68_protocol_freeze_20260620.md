# Paper 68 Protocol Freeze

Date: 2026-06-20

Freeze status: frozen before full run.

## Frozen Implementation

The frozen implementation is `src/run_experiment.py` after the Dev1 hard-fallback repair and the Dev2 fallback-blend rejection recorded in `docs/paper68_development_log_20260620.md`.

No further method tuning is allowed after this document. Recoverable failures may be fixed only if they are execution, serialization, plotting, or PDF-generation defects that do not alter the method or decision gates.

## Frozen Full Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 15 --ablation-episodes 15 --stress-episodes 10 --train-scenes 96 --train-candidates 12 --main-candidates 22 --oracle-candidates 7 --splits seen_simple unseen_dimensions unseen_shape_family slippery_contact clutter_collision task_constraint_shift combined_composition_shift long_bar_high_aspect thin_handle_shift adversarial_clutter_gap material_shift_proxy --ablation-splits combined_composition_shift adversarial_clutter_gap material_shift_proxy --stress-levels 0.0 0.2 0.4 0.6 0.8 1.0 --workers 4 --results-dir results --figures-dir figures
```

## Frozen Evidence Scale

- Training rows: 1,152.
- Main CSV rows: 18,480.
- Ablation CSV rows: 4,680.
- Stress CSV rows: 4,320.
- Main methods: 14.
- Main splits: 11.
- Ablation methods: 13.
- Ablation splits: 3.
- Stress methods: 9.
- Stress levels: 6.

The oracle rows internally evaluate up to seven MuJoCo candidates, so the simulator rollout count is larger than the CSV row count.

## Frozen Decision Gates

- KILL_ARCHIVE if v5 fails to beat the strongest non-oracle aggregate baseline by at least 0.030 success.
- KILL_ARCHIVE if v5 fails to beat the strongest non-oracle combined-composition baseline by at least 0.030 success.
- KILL_ARCHIVE if v5 safety failures exceed the strongest non-oracle aggregate baseline by more than 0.020.
- KILL_ARCHIVE if any removed-component ablation matches or beats full v5 within 0.020 success.
- KILL_ARCHIVE if v5 loses the maximum-stress gate to any non-oracle baseline.
- STRONG_REVISE only if all local gates pass; ICLR-main readiness still requires real robot or recognized public benchmark validation.

## Artifact Rules

- Write the final numbered PDF only to `C:/Users/wangz/Downloads/68.pdf`.
- Do not copy any PDF to the visible Desktop.
- Preserve frozen CSVs, figures, generated tables, validation script, and public GitHub repository state.
