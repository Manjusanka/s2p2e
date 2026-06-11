# S2P2E-RAG Dataset

S2P2E-RAG is the retrieval knowledge-base dataset generated for the S2P2E manipulation stack. It pairs object-level 3D assets with five tabletop manipulation task families and stores three retrieval tiers for each entry: semantic text, geometric descriptors, and operational pose/action descriptors.

## Scope

- Objects: 1492
- Retrieval entries: 7405
- Categories: 40
- Tasks: insert, pick, place, pour, tool_use
- Splits: train 5924, val 740, test 741

## Intended Use

This dataset is intended for retrieval-augmented pose proposal, manipulation knowledge-base ablations, and reproducibility of the Layer I S2P2E experiments. It is not a full replacement for upstream Objaverse, ShapeNet/ModelNet, or DROID assets.

## Training Snapshot

- RAG best epoch: 21
- RAG validation pose MAE: 0.009294
- Full-stack validation loss: 0.008164
- DROID policy samples: 6235

## Files

- `entries.jsonl`: all retrieval entries.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: deterministic split files.
- `manifest.json`: dataset metadata.
- `category_counts.json`, `task_counts.json`: dataset statistics.
- `SCHEMA.md`: field-level description for the enriched v2 entry format.
