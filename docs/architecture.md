# S2P2E System Architecture

## Layer I: 3D-Physics RAG

### Inputs

- Point cloud: `B x N x 6`
- Token ids: `B x L`
- Geometric descriptor: `B x 12`
- Operational descriptor: `B x 12`

### Internal Tensors

- Scene embedding: `B x 512`
- Semantic embedding: `B x 512`
- Geometric embedding: `B x 512`
- Operational embedding: `B x 512`
- Retrieval memory: `B x 3 x 512`
- Candidate poses: `B x K x 8`

### Pose Format

`[x, y, z, qw, qx, qy, qz, g]`

- Position: 3
- Quaternion: 4
- Gripper state/logit: 1

## Layer II: Physics Prior Encoder

### Inputs

- Joint state: `B x T x 21` for `q, qd, qdd`
- Embodiment vector: `B x 21`

### Internal Tensors

- Flattened state: `(B*T) x 21`
- PPE hidden feature: `(B*T) x 512`
- Torque prediction: `(B*T) x 7`
- Feasibility score: `(B*T) x 1`

## Layer III: Hierarchical Residual Control

### Inputs

- Velocity history: `B x H x 7`
- Torque history: `B x H x 7`
- Pose history: `B x H x 7`
- Force signal: `B x 6`
- Current pose: `B x 7`
- Task phase: `B x 5`
- PID torque: `B x 7`

### Outputs

- Friction residual: `B x 7`
- Load residual: `B x 7`
- Noise residual: `B x 7`
- Gate alpha: `B x 1`
- Clipped residual: `B x 7`
- Final torque command: `B x 7`

## Training Protocol

1. Train Layer I on joint ShapeNet and Objaverse object corpora.
2. Train PPE on synthetic inverse-dynamics trajectories.
3. Train residual controller on synthetic contact-disturbance signals.
4. Optionally run joint fine-tuning with Layer I + PPE consistency.
