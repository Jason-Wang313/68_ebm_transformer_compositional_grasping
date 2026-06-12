        # Literature Map

        Paper: 68 ebm_transformer_compositional_grasping

        Field box: energy-based transformers for grasping

        Thesis: EBM Transformer Compositional Grasping turns the seed bet into a mechanism: Compose grasp energies over objects, contacts, and task constraints without collapsing them into one policy token.

        ## Landscape Sweep Summary
        The selector ranked records from the shared 500,000-record pool. The closest visible clusters are:
        - FViT-Grasp: Grasping Objects With Using Fast Vision Transformers (2023)
- Dexterous Grasp Transformer (2024)
- OVAL-Grasp: Open-Vocabulary Affordance Localization for Task Oriented Grasping (2025)
- HMT-Grasp: A Hybrid Mamba-Transformer Approach for Robot Grasping in Cluttered Environments (2024)
- Vision-based Robotic Grasping from Object Localization, Pose Estimation, Grasp Detection to Motion Planning: A Review. (2019)
- Compositional Diffusion-Based Continuous Constraint Solvers (2023)
- MetaGraspNetV2: All-in-One Dataset Enabling Fast and Reliable Robotic Bin Picking via Object Relationship Reasoning and Dexterous Grasping (2023)
- Learning Task Models for Robotic Manipulation of Nonrigid Objects (2017)
- Grasp Stability and Design Analysis of a Flexure-Jointed Gripper Mechanism via Efficient Energy-Based Modeling (2022)
- Transformers Discover Molecular Structure Without Graph Priors (2025)
- Learning Generalizable Vision-Tactile Robotic Grasping Strategy for Deformable Objects via Transformer (2024)
- Improving Object Grasp Performance via Transformer-Based Sparse Shape Completion (2022)

        ## Hidden Assumptions
        - The executed trajectory is a sufficient training target.
- Unobserved physical alternatives can be averaged into uncertainty.
- Task labels capture the mechanism that caused failure.
- A planner only needs nominal feasibility.
- Embodiment-specific contact effects are nuisance variation.

        ## Boundary
        The project avoids weak moves such as bigger models, generic uncertainty, or a benchmark-only contribution. It centers a mechanism-level change: Ebm transformer compositional grasping keeps action-critical alternatives explicit until a physical observation collapses them.
