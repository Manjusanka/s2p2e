# S2P2E-RAG-QA Audit Protocol

This protocol defines the manual review fields used to validate S2P2E-RAG-QA entries.

## Sampling

- Audit table: `paper/source_data/rag_quality_human_audit_template.csv`
- Current sample size: 500 entries
- Sampling rule: deterministic mixed sample from the full scored release and QA-D compressed subset.

## Review Dimensions

Each audited entry is scored on `[0, 1]`:

- `human_semantic_grounding_score`: object, task, and instruction consistency.
- `human_geometric_action_consistency_score`: compatibility between object geometry and action descriptor.
- `human_physics_feasibility_score`: plausibility of mass, contact risk, torque sensitivity, and finite descriptors.
- `human_retrieval_utility_score`: usefulness as a retrieval memory for pose proposal.

## Decision

- `accept`: entry is directly usable.
- `revise`: entry is partly useful but should be corrected, down-weighted, or filtered for some tasks.
- `reject`: entry should not be used for training or retrieval.

## Adjudication

If multiple reviewers disagree, keep the individual scores in a local reviewer sheet and write the reconciled decision in `human_overall_decision`.
