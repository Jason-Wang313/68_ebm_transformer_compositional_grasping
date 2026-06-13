# Novelty Boundary Map

## Crowded Territory
- Bigger data/model scaling.
- New benchmark only.
- Generic active learning or uncertainty.
- Combining a planner with a learned policy without a new state/action object.

## Claimed Boundary
Ebm transformer compositional grasping keeps action-critical alternatives explicit until a physical observation collapses them.

## What Would Falsify The Claim
If observed-only baselines match the adverse-mode coverage and closed-loop success of the proposed branch-aware mechanism, the paper should be revised or killed.

## v4 Falsification
The real MuJoCo rebuild falsifies the current claim. On combined composition shift, `ebm_transformer_compositional` reaches 0.080 success, while `force_closure_score` and `mlp_energy_model` reach 0.140. The mechanism does not clear analytic or learned baselines.
