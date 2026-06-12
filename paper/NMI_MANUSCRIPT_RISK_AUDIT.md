# Final Manuscript Evidence Audit

Target journal: Nature Machine Intelligence

Audit date: 2026-06-12

Audited materials:

- `paper/NMI_Submission.tex`
- `paper/Supplementary_Information.tex`
- `paper/references.bib`
- `paper/source_data/*.csv`
- `README.md`
- `docs/*.md`
- `public_data/s2p2e_rag_dataset/`
- `public_data/s2p2e_rag_quality/`
- `artifacts/`

## Overall Result

The repository and manuscript now support a bounded, submission-facing systems claim: S2P2E improves tabletop language-driven manipulation by separating semantic retrieval, physical feasibility filtering, and bounded residual execution control. The evidence package is internally consistent across the manuscript, supplementary information, source-data tables, repository documentation, released RAG datasets, and selected checkpoints.

No blocking evidence-integrity issue was found in the final audit. The remaining caveats are scope limitations rather than contradictions: hardware validation is limited to the reported Franka tabletop regime, cross-embodiment results validate inverse-dynamics prediction rather than full downstream manipulation transfer, and DROID is used only as a secondary policy-bridge sanity check.

## Manuscript and Compilation Checks

Status: pass.

- The repository manuscript copy matches the Overleaf submission source for the main paper, supplementary information, references, and core RAG source-data files.
- The main paper compiles with line numbers, single-column formatting, all manuscript figures, and the current bibliography.
- The supplementary information compiles independently.
- The main text frames generalization conservatively and avoids unsupported universal deployment claims.
- The code and data availability statements point to the public repository: `https://github.com/Manjusanka/s2p2e`.

## Citation Audit

Status: pass with currentness improved.

- Recent VLA and robot foundation-model context is represented by OpenVLA, Octo, ManipLLM, CogACT, pi0, and pi0.5.
- Spatial grounding and 3D action-prediction context is represented by Act3D, PolarNet, SpatialVLM, VoxPoser, RoboPoint, and SpatialVLA.
- Retrieval-grounded robotics context is represented by SayCan, Code-as-Policies, GaussianGrasper, LERF, GraspSplats, RoboGround, RobMRAG, and Retrieval-Augmented Robots via Retrieve-Reason-Act.
- Dataset and robot-learning scale context is represented by DROID and Open X-Embodiment.
- Previously risky unverifiable placeholder-style references were removed in earlier cleanup; the current bibliography contains traceable arXiv, RSS, journal, or conference-style entries.

## Source-Data Traceability

Status: pass.

Key manuscript values are backed by machine-readable source data:

- Main clutter benchmark: `paper/source_data/table_grasp_main.csv`
- Knowledge-base ablation: `paper/source_data/table_kb_ablation.csv`
- Physical-violation metrics: `paper/source_data/table_physical_violations.csv`
- Force and contact metrics: `paper/source_data/table_force_tracking.csv` and `table_safety_tail_contact.csv`
- Sim-to-real and stress tests: `paper/source_data/table_sim2real.csv` and `stress_test_sim2real.csv`
- Unseen-object and dense-clutter slices: `unseen_object_split.csv` and `dense_clutter_split.csv`
- RAG quality audit and compression: `rag_quality_*` source-data files
- Claim-to-evidence accounting: `table_claim_sample_accounting.csv` and `value_provenance_registry.csv`

The major reported values, including 87.3% main GSR, 82.7% unseen-object GSR, 91.3% sim-to-real retention, 68,960 S2P2E-RAG entries, and 20,688 S2P2E-RAG-QA entries, are all present in source-data exports or repository manifests.

## Open-Source Repository Audit

Status: pass.

The public repository includes:

- Training, evaluation, data-conversion, RAG export, quality-auditor, and source-data export scripts.
- Selected checkpoints under `artifacts/`, including the joint stack, layer-wise checkpoints, full RAG run, full physics run, and actor-critic DROID bridge run.
- Released retrieval data under `public_data/s2p2e_rag_dataset/` and `public_data/s2p2e_rag_quality/`.
- Dataset cards, schemas, manifests, split metadata, and audit protocol files.
- The manuscript source, figures, bibliography, and source-data package needed to reproduce the paper tables and figures.

The repository intentionally excludes download caches, partial downloads, failed intermediate arrays, and non-release temporary artifacts.

## Claim Boundary Audit

Status: pass with explicit caveats retained.

Directly supported claims:

- Retrieval improves cluttered semantic grounding under the reported benchmark protocol.
- PPE reduces physically infeasible actions before execution.
- Bounded residual control improves contact robustness after action selection.
- S2P2E-RAG and S2P2E-RAG-QA provide inspectable retrieval-data assets with documented quality-control procedures.

Claims intentionally not made:

- Universal open-world generalization.
- Formal closed-loop safety guarantees.
- Full downstream cross-robot deployment.
- Complete manual certification of every released RAG entry.

These boundaries are visible in the discussion, supplementary information, data availability statement, and repository documentation.

## Final Submission Notes

For journal upload, include only the submission-relevant source package: main manuscript, supplementary information, bibliography, figures, and source-data files. Internal planning documents, rendering screenshots, LaTeX auxiliary files, and local audit screenshots should not be uploaded as manuscript source files.

The strongest remaining editorial risk is not evidence inconsistency; it is scope perception. The submission should continue to present S2P2E as a carefully bounded, auditable robotics systems contribution rather than as a universal general-purpose robot policy.
