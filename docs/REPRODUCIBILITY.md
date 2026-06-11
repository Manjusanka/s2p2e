# Reproducibility Guide

## Environment

- Python: `3.12`
- Main dependencies: `torch`, `faiss-cpu`, `trimesh`, `pytest`, `tfrecord`, `gsutil`

Install:

```bash
pip install -r requirements.txt
pip install gsutil tfrecord
```

## Basic Validation

Run tests:

```bash
pytest tests -q
```

## Training Commands

### Small smoke run

```bash
python scripts/train_all.py --config configs/experiment_small.yaml
```

### Larger joint run

```bash
python scripts/train_all.py --config configs/experiment_g50.yaml
```

### Full end-to-end pipeline

```bash
python scripts/run_full_pipeline.py
```

## Public Data Conversion

### DROID subset

```bash
python scripts/convert_droid_subset.py
python scripts/train_actor_critic_droid.py
```

## Main Outputs

- `artifacts/s2p2e_g50/metrics/summary.json`
- `artifacts/s2p2e_g50/checkpoints/joint_best.pt`
- `artifacts/s2p2e_full_pipeline/full_pipeline_summary.json`
- `artifacts/s2p2e_full_pipeline/policy_bridge_best.pt`

## Known Gaps

- RH20T and RoboSet direct ingestion are planned but not yet fully downloaded locally.
- Some physics and deployment assets are reconstructed because the original paper does not publicly release all of them.
