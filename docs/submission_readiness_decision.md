# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v4 adds a real MuJoCo parallel-jaw grasping benchmark with rollout-labeled learned baselines, but the evidence is negative. The proposed compositional EBM/Transformer ranker is matched or beaten by force-closure, MLP energy, CEM, and ensemble baselines on the combined composition shift.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: invent and test a substantially stronger compositional grasp model that clears learned and analytic baselines on public grasp benchmarks or hardware, with manual related-work depth and richer perception/tactile inputs.
