# S2P2E-RAG Schema

Each JSONL row represents one `(object, task)` retrieval entry.

## Core Fields

- `entry_id`: stable release identifier.
- `schema_version`: current entry schema.
- `object`: object identity, source dataset, category, and manipulation priors.
- `instruction`: canonical task text plus task family and phase sequence.
- `geometric`: 12-dimensional vector used by the model.
- `operational`: 12-dimensional vector containing target pose and grasp/action descriptors.

## Enriched Fields

- `geometry_summary`: interpretable center, extents, curvature, approachability, volume proxy, and aspect ratio.
- `operational_summary`: target pose, grasp score proxy, task id, upright requirement, and contact-risk class.
- `physics_prior`: mass proxy, rigidity/material prior, and torque-sensitivity label.
- `quality_flags`: finite-value, extent, quaternion, and task-consistency checks.
- `provenance`: script, source KB artifact, and release date.
