# S2P2E-RAG Dataset

S2P2E-RAG is the retrieval knowledge-base dataset generated for the S2P2E manipulation stack. It pairs object-level 3D assets with five tabletop manipulation task families and stores three retrieval tiers for each entry: semantic text, geometric descriptors, and operational pose/action descriptors.

## Scope

- Objects: 13803
- Retrieval entries: 68960
- Categories: 41
- Tasks: insert, pick, place, pour, tool_use
- Splits: train 55168, val 6896, test 6896

## Intended Use

This dataset is intended for retrieval-augmented pose proposal, manipulation knowledge-base ablations, and reproducibility of the Layer I S2P2E experiments. It is not a full replacement for upstream Objaverse, ShapeNet/ModelNet, or DROID assets.

## Training Snapshot

- RAG best epoch: 7
- RAG validation pose MAE: 0.007492
- Full-stack validation loss: 0.007597
- DROID policy samples: 6235

## Files

- `entries.jsonl`: all retrieval entries.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: deterministic split files.
- `manifest.json`: dataset metadata.
- `category_counts.json`, `task_counts.json`: dataset statistics.
- `SCHEMA.md`: field-level description for the enriched v2 entry format.
