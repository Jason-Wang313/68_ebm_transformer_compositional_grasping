# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v5 expands the paper into a 35-page ICLR-style frozen MuJoCo archive with 1,152 training rollouts, 18,480 main evaluation rows, 4,680 ablation rows, 4,320 stress rows, seed-level statistics, explicit theory claims, and bright boxed clickable citations. The evidence is still negative. The proposed `ebm_transformer_compositional_v5` ranker reaches 0.667 aggregate success, below the strongest non-oracle aggregate baselines `ensemble_uncertainty_ranker` and `gradient_boosted_energy_model` at 0.747. On the combined-composition shift, v5 reaches 0.067, below `calibrated_stacked_ranker` and `ensemble_uncertainty_ranker` at 0.075. The ablation gate fails because `monolithic_scalar_energy_only` beats full v5, and `no_cem_fallback` plus `no_collision_energy` match or nearly match full v5 within the pre-registered tolerance. The maximum-stress gate also fails against `ensemble_uncertainty_ranker`.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: invent and test a substantially stronger compositional grasp model that clears learned, analytic, robust, stacked, and oracle-bounded baselines on public grasp benchmarks or hardware, with richer perception/tactile inputs, a jointly optimized candidate generator, and a separately reported fallback operating regime.
