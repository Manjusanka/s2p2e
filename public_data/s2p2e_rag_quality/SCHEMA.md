# S2P2E-RAG-QA Schema

Each row keeps the S2P2E-RAG v2 fields and adds a `quality_audit` object.

## `quality_audit`

- `semantic_grounding`: learned score for task/object/instruction consistency.
- `geometric_action_consistency`: learned score for geometry-to-action plausibility.
- `physics_feasibility`: learned score for finite, physically plausible manipulation descriptors.
- `retrieval_utility`: learned estimate of retrieval usefulness.
- `overall_quality`: aggregate learned quality score.
- `uncertainty`: MC-dropout predictive uncertainty.
- `quality_confidence`: `1 - uncertainty`, clipped to `[0, 1]`.
- `compression_score`: quality score after uncertainty penalty.
- `selection_score`: final QA-D score after diversity-aware selection.

## Selection Objective

The compressed split uses quality-audited diversity:

- Penalize entries with high predictive uncertainty.
- Enforce minimum task-family coverage.
- Reward underrepresented categories and source datasets during selection.
