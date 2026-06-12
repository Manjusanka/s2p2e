# Source Data Package

This directory contains the machine-readable numerical exports that support the tables, figures, and claim-to-evidence mapping in the main manuscript and supplementary information.

## Scope

The source-data package is organized around final measured values, sample definitions, and provenance links. Each CSV is intended to make one table, figure, benchmark slice, or reporting-summary item independently auditable.

## Files

- `table_module_metrics.csv`: module-level validation metrics for Layer I, Layer II, Layer III, joint-stack validation, and the DROID policy-bridge check.
- `table_training_protocol_metrics.csv`: training configuration, model settings, and protocol-level metadata.
- `table_grasp_main.csv`: main cluttered-scene grasp results.
- `table_physical_violations.csv`: torque and acceleration violation results.
- `table_force_tracking.csv`: force-tracking and contact-stability metrics.
- `table_task_family_breakdown.csv`: task-family performance breakdown.
- `table_kb_ablation.csv`: knowledge-base tier ablation.
- `table_cross_embodiment.csv`: inverse-dynamics transfer across robot embodiments.
- `table_sim2real.csv`: simulation-to-hardware transfer results.
- `table_hyperparameter.csv`: hyperparameter sensitivity results.
- `table_compute.csv`: inference cost and parameter-count comparison.
- `table_claim_sample_accounting.csv`: mapping between claims, independent units, and sample sizes.
- `table_stress_test_protocol.csv`: stress-test definitions and measured sample sizes.
- `table_safety_tail_contact.csv`: safety-tail and contact-waveform metrics.
- `stress_test_sim2real.csv`: camera-shift, illumination-shift, and payload stress-test outcomes.
- `unseen_object_split.csv`: held-out object evaluation slice.
- `dense_clutter_split.csv`: dense-clutter evaluation slice.
- `value_provenance_registry.csv`: internal value-provenance audit trail retained for traceability.
- `historical_value_audit.csv`: internal historical value-tracking file retained for traceability.
- `training_expansion_summary.csv`: expanded GPU training and data-scale summary for S2P2E-RAG, PPE, residual control, joint S2P2E, and the DROID policy bridge.
- `rag_dataset_schema_summary.csv`: machine-readable summary of the enriched S2P2E-RAG v2 field groups used to document the open retrieval dataset contribution.
- `rag_quality_compression_summary.csv`: source-data summary for the learned S2P2E-RAG-QA auditor and QA-D compressed dataset.
- `rag_quality_compression_eval.csv`: split-level comparison of full versus QA-D compressed retrieval memory.
- `rag_quality_benchmark_curves.csv`: compression-ratio curves for predicted quality, MC-dropout uncertainty, and pose-memory proxy error used in Figure 11.
- `rag_quality_calibration_bins.csv`: quality-bin calibration table linking auditor uncertainty to estimated rejection risk used in Figure 11.
- `rag_quality_human_audit_template.csv`: deterministic 500-entry audit sheet with model-prefilled quality scores and empty human-review fields.
- `rag_quality_predicted_audit_summary.csv`: provisional model-estimated audit summary used to plan manual validation and replacement with final human-review values.

## Field Convention

Most table-level exports use the following columns:

- `table_or_figure_id`
- `row_id`
- `method`
- `condition`
- `metric`
- `mean`
- `sd`
- `n`
- `unit`
- `provenance`

Where a table requires a different schema, the column names are kept explicit and self-describing. The `provenance` field records the artifact, protocol, or split from which the reported value was derived.

## Release Note

The review-facing manuscript reports final measured values. Internal value-tracking files are retained only to preserve an audit trail from manuscript preparation to the final source-data package.
