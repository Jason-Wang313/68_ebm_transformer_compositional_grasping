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
        A hostile ICLR reviewer would be correct to reject this as a main-conference submission. The v4 rebuild now contains a real MuJoCo grasping benchmark and learned baselines, but the compositional EBM/Transformer ranker is matched or beaten by force-closure and MLP energy baselines on the hardest composition split.

        ## Honest Action
        The paper is marked `KILL_ARCHIVE`. This avoids converting a falsified mechanism into an overstated main-conference claim.

        ## What Would Be Needed To Revive
        - A substantially stronger compositional grasp model that clears analytic and learned baselines.
        - Real robot or public grasp benchmark experiments.
        - Manual full-paper related-work audit.
        - Evidence that the core mechanism is learned and useful under object/task composition shift.
