# Model Cards

## `artifacts/s2p2e_g50/checkpoints/joint_best.pt`

- Purpose: main S2P2E joint checkpoint
- Training data: local ShapeNet-style point clouds, expanded Objaverse subset, synthetic dynamics/control supervision
- Main role: Layer I + Layer II + Layer III joint stack

## `artifacts/s2p2e_full_pipeline/policy_bridge_best.pt`

- Purpose: full-pipeline policy bridge checkpoint
- Training data: converted DROID subset transitions
- Main role: PPE-informed Actor-Critic policy bridge

## `artifacts/actor_critic_droid/actor_critic_droid_best.pt`

- Purpose: standalone DROID-subset Actor-Critic checkpoint
- Training data: converted DROID subset transitions
- Main role: validate policy-side training loop on real public robot data

## `artifacts/s2p2e_small/` and `artifacts/s2p2e_joint/`

- Purpose: smoke and intermediate ablation runs
- Main role: debugging, validation, and staged reproduction
