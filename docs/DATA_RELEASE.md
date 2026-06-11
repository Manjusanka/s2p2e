# Data Release Notes

This repository includes the local release package prepared for review and reproducibility.

## Included

- `shapenet/`: local ShapeNet/ModelNet-style point-cloud subset used by the code paths and smoke tests.
- `objaverse/`: local Objaverse subset; `.glb` files are tracked with Git LFS because several assets exceed GitHub's normal 100 MB object limit.
- `public_data/droid_100/`: local DROID TFRecord subset used for Actor-Critic ingestion and conversion.
- `public_data/droid_subset_jsonl/`: converted DROID transition files used by lightweight policy tests.
- `public_data/s2p2e_rag_dataset/`: generated S2P2E-RAG retrieval dataset with semantic, geometric, and operational tiers.
- `paper/source_data/`: manuscript source-data CSV exports matching the submitted figures and tables.
- `artifacts/`: selected trained checkpoints and metric JSON files used by the manuscript package.

## Excluded

- Download caches such as `.cache/`.
- Partial downloads such as `*.part`.
- Generated retrieval caches such as `*.faiss`, `*.npy`, and `*.npz`.

## Reconstructing Larger Training Inputs

Use the scripts in `scripts/` and the manifests in `public_data/` to rebuild larger local datasets under the paths used by the configs:

- `objaverse/models/`
- `shapenet/models/`
- `public_data/droid_100/1.0.0/`

The included local subset is sufficient for smoke tests and review-scale reproduction. Larger training runs should reconstruct or mount the full upstream datasets under the same paths.

## Generated RAG Dataset

`public_data/s2p2e_rag_dataset/` is generated from the local 3D object subset and the S2P2E task vocabulary. It is project-native data rather than a mirrored upstream dataset, so it can be cited as the open S2P2E-RAG release in the manuscript. The current release contains 7,405 retrieval entries, deterministic train/val/test splits, category counts, task counts, and a dataset card.
