# Data Sources

## Local 3D Data

### Objaverse

- Current local location: `objaverse/`
- Used for: retrieval memory construction, geometric descriptors, grasp-pose proposal pretraining
- Current local status: expanded subset stored as `.glb`; mesh files are tracked with Git LFS

### ShapeNet / Point-Cloud Text Data

- Current local location: `shapenet/models/`
- Used for: point-cloud encoder training and KB construction
- Current repository status: local subset included directly, excluding caches and partial downloads

## Public Robot Data

### DROID

- Official page: [DROID Dataset](https://droid-dataset.github.io/droid/the-droid-dataset)
- Current local location: `public_data/droid_100/1.0.0/`
- Current usage: TFRecord subset converted to `jsonl` transitions for Actor-Critic training
- Current conversion output: `public_data/droid_subset_jsonl/`
- Current repository status: local `droid_100` subset included; 31 shards are converted into 6,235 lightweight policy transitions

## Project-Native Generated Data

### S2P2E-RAG

- Current local location: `public_data/s2p2e_rag_dataset/`
- Generated from: local 3D object subset, S2P2E task vocabulary, and semantic/geometric/operational tier construction
- Current size: 7,405 retrieval entries from 1,492 objects across five task families
- Entry schema: raw model vectors plus object priors, task phase labels, geometry summaries, operational summaries, physics proxies, quality flags, and provenance
- Intended usage: Layer I RAG training, KB-tier ablations, open retrieval dataset release, and manuscript source-data support

### RH20T

- Official page: [RH20T](https://rh20t.github.io/)
- Intended usage: joint-level trajectory, torque, and dynamics supervision
- Current repository status: manifest prepared, direct dataset integration pending download

### RoboSet

- Official page: [TFDS RoboSet](https://www.tensorflow.org/datasets/catalog/robo_set)
- Intended usage: additional policy-side trajectory supervision
- Current repository status: data-source manifest prepared, direct integration pending

## Important Notes

- Each external dataset keeps its own license and access conditions.
- This repository does not assume redistribution rights for third-party datasets.
- Git LFS is required to fetch large Objaverse `.glb` files after cloning.
- See `docs/DATA_RELEASE.md` for the exact local data-release boundary.
