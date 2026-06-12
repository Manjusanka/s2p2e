# S2P2E

This repository contains the code, configurations, checkpoints, source-data exports, and release data samples for **S2P2E: Physical Grounding for Language-Driven Robotic Manipulation**.

The repository name is intentionally `s2p2e`, matching the manuscript availability statement and the planned GitHub repository:

```text
https://github.com/Manjusanka/s2p2e
```

## What Is Included

- Layer I: retrieval-augmented pose generation with semantic, geometric, and operational tiers.
- Layer II: Physics Prior Encoder (PPE) with embodiment-conditioned inverse-dynamics modeling.
- Layer III: hierarchical residual control with `FrictionNet`, `LoadNet`, `NoiseNet`, and `GateNet`.
- Policy bridge: PPE feature injection into an actor-critic policy, including DROID conversion and cross-embodiment dynamics augmentation.
- Experiment stack: staged training, joint training, ablations, task-family analysis, KB-tier analysis, and end-to-end pipeline execution.
- Checkpoints: best checkpoints used by the released reproduction under `artifacts/`.
- Paper source data: machine-readable exports under `paper/source_data/`.
- Local data subset: ShapeNet/ModelNet-style point clouds, Objaverse meshes tracked with Git LFS, DROID TFRecord shards, and converted DROID JSONL transitions.
- S2P2E-RAG dataset: the generated v2 retrieval dataset under `public_data/s2p2e_rag_dataset/`, including semantic, geometric, operational, physics-prior, and quality-audit fields.
- S2P2E-RAG-QA dataset: the learned quality-audited and QA-D compressed retrieval subset under `public_data/s2p2e_rag_quality/`.

## Data Boundary

The release includes the local data subset that is present in this workspace. Large Objaverse `.glb` files are tracked with Git LFS. Download caches, partial downloads, and generated retrieval arrays are excluded so the repository remains reproducible rather than cache-dependent.

Use the scripts and manifests in `scripts/`, `configs/`, `docs/DATA_SOURCES.md`, and `public_data/dataset_manifest.json` to reconstruct larger assets under their original licenses.

## Repository Layout

```text
s2p2e/
  artifacts/       selected checkpoints and metric JSON files
  configs/         experiment configurations
  docs/            data, model-card, and reproducibility notes
  paper/           source-data exports and manuscript audit notes
  public_data/     DROID subset, S2P2E-RAG dataset, converted JSONL transitions, and manifests
  scripts/         download, convert, train, evaluate, and export utilities
  src/             S2P2E source code
  tests/           smoke and integration tests
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run smoke training:

```bash
python scripts/train_all.py --config configs/experiment_small.yaml
```

Run the larger joint pipeline:

```bash
python scripts/train_all.py --config configs/experiment_g50.yaml
```

Run the full pipeline:

```bash
python scripts/run_full_pipeline.py
```

Run the expanded GPU training suite after installing CUDA-enabled PyTorch:

```bash
python scripts/train_rag_layer1.py --config configs/experiment_rag_full.yaml
python scripts/train_all.py --config configs/experiment_physics_full.yaml
python scripts/train_actor_critic_droid.py --config configs/public_robot_data.yaml --out-dir artifacts/actor_critic_droid_full
python scripts/export_rag_dataset_release.py
```

Run tests:

```bash
pytest tests -q
```

## Main Artifacts

- S2P2E joint model: `artifacts/s2p2e_g50/checkpoints/joint_best.pt`
- Layer checkpoints: `artifacts/s2p2e_g50/checkpoints/layer*_best.pt`
- DROID policy bridge: `artifacts/actor_critic_droid/actor_critic_droid_best.pt`
- Full-pipeline policy bridge: `artifacts/s2p2e_full_pipeline/policy_bridge_best.pt`
- Expanded RAG model: `artifacts/s2p2e_rag_full/checkpoints/layer1_best.pt`
- Expanded full-stack model: `artifacts/s2p2e_physics_full/checkpoints/joint_best.pt`
- Expanded DROID policy bridge: `artifacts/actor_critic_droid_full/actor_critic_droid_best.pt`
- Experiment summaries: `artifacts/*/metrics/*.json`
- Paper source data: `paper/source_data/*.csv`
- S2P2E-RAG open dataset: `public_data/s2p2e_rag_dataset/`
- S2P2E-RAG-QA compressed dataset: `public_data/s2p2e_rag_quality/`
- Local 3D data subset: `shapenet/` and `objaverse/`
- DROID subset: `public_data/droid_100/` and `public_data/droid_subset_jsonl/`

## Data Notes

The committed DROID files are a local `droid_100` subset for interface testing and released reproduction. They are not a substitute for the complete DROID dataset.

The expanded policy-bridge run mixes 6,235 converted DROID transitions with 32,768 cross-embodiment synthetic dynamics transitions, producing 39,003 Actor-Critic training/evaluation samples with a held-out validation split.

Objaverse meshes require Git LFS after cloning. For full-scale training beyond the included subset, use the manifests and reconstruction scripts described in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) and [docs/DATA_RELEASE.md](docs/DATA_RELEASE.md).

The generated S2P2E-RAG dataset is the main project-native open data asset. It contains 68,960 enriched retrieval entries derived from 13,803 local 3D objects across five tabletop manipulation task families. The v2 schema keeps the raw semantic/geometric/operational supervision vectors and adds interpretable object priors, task phase sequences, geometry summaries, physics proxies, quality flags, and provenance fields for dataset-level auditing.

The generated S2P2E-RAG-QA release adds a learned quality auditor and QA-D compression step. It scores all 68,960 entries, then exports a 20,688-entry compressed subset selected by quality, MC-dropout uncertainty, task-floor coverage, and category/source diversity.

## Documentation

- [Architecture](docs/architecture.md)
- [Data sources](docs/DATA_SOURCES.md)
- [Data release notes](docs/DATA_RELEASE.md)
- [S2P2E-RAG dataset card](public_data/s2p2e_rag_dataset/DATASET_CARD.md)
- [S2P2E-RAG-QA dataset card](public_data/s2p2e_rag_quality/DATASET_CARD.md)
- [Model cards](docs/MODEL_CARDS.md)
- [Open-source checklist](docs/OPEN_SOURCE_CHECKLIST.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Paper source data](paper/source_data/README.md)

## License

This repository is released under the MIT License. External datasets and third-party model assets keep their own licenses and terms.
