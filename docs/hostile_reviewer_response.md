# Hostile Reviewer Response

Paper: 68 EBM Transformer Compositional Grasping

## Strongest Technical Threats

- FViT-Grasp: Grasping Objects With Using Fast Vision Transformers (2023)
- OVAL-Grasp: Open-Vocabulary Affordance Localization for Task Oriented Grasping (2025)
- GAT-Grasp: Gesture-Driven Affordance Transfer for Task-Aware Robotic Grasping (2025)
- HMT-Grasp: A Hybrid Mamba-Transformer Approach for Robot Grasping in Cluttered Environments (2024)
- FunGrasp: Functional Grasping for Diverse Dexterous Hands (2024)
- Kinematic Synthesis of Minimally Actuated Multi-Loop Planar Linkages With Second Order Motion Constraints for Object Grasping (2013)
- Oracle-grasp: zero-shot affordance-aligned robotic grasping using large multimodal models (2026)
- Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter (2023)

## ICLR Main Response

A hostile ICLR reviewer would be correct to reject this as a main-conference submission. The v4 rebuild contains a real MuJoCo grasping benchmark, rollout-labeled training data, learned baselines, analytic grasp baselines, ablations, stress sweeps, uncertainty, and negative cases. That stronger evidence does not rescue the paper: the EBM/Transformer compositional ranker is matched or beaten by force closure, CEM, MLP energy, Transformer ranking, and ensemble uncertainty on the hardest composition split.

## Honest Action

The paper is marked `KILL_ARCHIVE`. The internal ablation signal is worth preserving, but it is not enough to convert the paper into an ICLR-main claim.

## What Would Be Needed To Revive

- A stronger candidate-search and scoring mechanism that clears force closure, CEM, MLP energy, Transformer ranking, and ensemble uncertainty.
- External-baseline ablations showing the composed energy terms are necessary on the same held-out composition split.
- Real robot or public high-fidelity grasping benchmark experiments.
- Richer perception, tactile/material sensing, or more realistic scene geometry.
- Manual full-paper related-work audit.
