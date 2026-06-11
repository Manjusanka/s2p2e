# S2P2E-RAG-QA Dataset

S2P2E-RAG-QA is the quality-audited and compressed release derived from S2P2E-RAG. It adds learned quality scores for every retrieval entry and exports a compact high-quality subset selected by quality, uncertainty, and diversity criteria.

## Quality-Auditing Paradigm

The auditor predicts five scores:

- `semantic_grounding`: consistency between object category, task family, and instruction.
- `geometric_action_consistency`: consistency between object geometry and target pose/action descriptor.
- `physics_feasibility`: finite-value, mass-proxy, torque-sensitivity, and contact-risk plausibility.
- `retrieval_utility`: expected usefulness of the entry for retrieval-augmented pose proposal.
- `overall_quality`: aggregate quality score used for compression.

## QA-D Compression

The compressed set is selected with a quality-utility-diversity objective:

`selection_score = quality - uncertainty_penalty + diversity_bonus`

The current release uses MC-dropout uncertainty estimation, task-floor constrained diversity, and category/dataset diversity bonuses.

## Scope

- Full scored entries: 68,960
- Compressed entries: 20,688
- Compression ratio: 0.30
- Full mean quality: 0.8726
- Compressed mean quality: 0.9252
- Full mean uncertainty: 0.00476
- Compressed mean uncertainty: 0.00446
- Predicted compressed accept rate: 0.9422

## Files

- `scored_entries.jsonl`: all S2P2E-RAG entries with `quality_audit` fields.
- `compressed_entries.jsonl`: QA-D compressed high-quality subset.
- `quality_manifest.json`: compression statistics and task/category counts.
- `AUDIT_PROTOCOL.md`: manual validation protocol for replacing model-prefilled audit values with human review.
