# Manuscript Evidence Audit

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

## Summary

The repository is organized to support the manuscript's bounded systems claim: retrieval-grounded pose proposal, inverse-dynamics feasibility filtering, and bounded residual execution control provide complementary improvements for tabletop language-driven manipulation. The evidence package includes source-data exports, training scripts, configurations, selected checkpoints, public-data conversion utilities, local dataset subsets, and two project-native retrieval-data releases.

## Manuscript Package

The manuscript copy in `paper/` mirrors the submission-oriented Overleaf source. It uses a single-column format with line numbers, includes the current Data Availability and Code Availability statements, and cites the repository as:

```text
https://github.com/Manjusanka/s2p2e
```

The reference file `paper/references.bib` is included so that the manuscript source can be compiled from the repository copy.

## Source Data

The `paper/source_data/` directory contains machine-readable CSV exports for the main manuscript tables, figures, robustness slices, source-data accounting, RAG quality curves, and human audit summary ranges. The directory README documents the role of each file and the convention used for provenance fields.

Key checks:

- Main benchmark, module, robustness, and compute tables have corresponding CSV exports.
- RAG quality compression and calibration curves have corresponding CSV exports.
- Human audit ranges are recorded in `rag_quality_human_audit_summary.csv`.
- Value provenance files are retained for traceability from manuscript values to source-data records.

## Open Data

The repository contains two project-native retrieval-data releases:

- `public_data/s2p2e_rag_dataset/`: S2P2E-RAG, containing 68,960 enriched retrieval entries with deterministic train/validation/test splits, manifest, schema, task counts, category counts, and dataset card.
- `public_data/s2p2e_rag_quality/`: S2P2E-RAG-QA, containing scored entries, a 20,688-entry QA-D compressed subset, quality manifest, audit protocol, schema, and dataset card.

The release also includes local public-data subsets and manifests for DROID, Objaverse, and ShapeNet/ModelNet-style assets. Large assets are documented as license-bound upstream data and are either included as local subsets or reconstructed through scripts and manifests.

## Code and Checkpoints

The repository includes scripts for staged training, full-pipeline training, public-data conversion, RAG dataset export, quality-auditor training, compression evaluation, experiment export, and smoke tests.

Selected checkpoints and run artifacts are present under `artifacts/`, including:

- `artifacts/s2p2e_g50/checkpoints/joint_best.pt`
- `artifacts/s2p2e_g50/checkpoints/layer1_best.pt`
- `artifacts/s2p2e_g50/checkpoints/layer2_best.pt`
- `artifacts/s2p2e_g50/checkpoints/layer3_best.pt`
- `artifacts/s2p2e_rag_full/checkpoints/layer1_best.pt`
- `artifacts/s2p2e_physics_full/checkpoints/joint_best.pt`
- `artifacts/actor_critic_droid/actor_critic_droid_best.pt`
- `artifacts/actor_critic_droid_full/actor_critic_droid_best.pt`

## Claim Boundaries

The manuscript is framed as a bounded tabletop manipulation study rather than a claim of unrestricted open-world deployment. The strongest directly supported claims are:

- Retrieval improves cluttered semantic grounding under the reported benchmark protocol.
- PPE reduces physically infeasible actions before execution.
- Bounded residual control improves contact robustness once an action is selected.
- S2P2E-RAG and S2P2E-RAG-QA provide inspectable, reusable retrieval-data assets with documented quality-control procedures.

## Remaining Reader-Facing Caveats

The evidence package intentionally keeps several boundaries explicit:

- Hardware evidence is limited to the reported Franka tabletop regime and stress-test conditions.
- Cross-embodiment evidence validates inverse-dynamics portability rather than full downstream manipulation transfer.
- The DROID policy-bridge experiment is supportive secondary evidence and is not used as the primary hardware benchmark.
- External datasets and 3D assets remain subject to their upstream licenses and redistribution terms.

These caveats are reflected in the manuscript text, data-release notes, and documentation.
