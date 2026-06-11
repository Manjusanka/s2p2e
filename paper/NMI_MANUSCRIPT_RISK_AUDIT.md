# NMI Manuscript Risk Audit

Target journal: Nature Machine Intelligence

Audited files:

- `NMI_Submission.tex`
- `Supplementary_Information.tex`
- `source_data/*.csv`
- `references.bib`

Audit date: 2026-06-11

This document lists issues that may make the manuscript appear logically inconsistent, insufficiently supported, overclaimed, or not yet submission-ready. It does not edit the manuscript directly.

## Overall Assessment

The manuscript now compiles and has a workable Nature-style single-column format with line numbers. The main remaining risks are not LaTeX problems. They are scientific-credibility and reviewer-trust problems: some claims are stronger than the visible evidence package, several availability statements appear premature, some sample-size/statistical units are ambiguous, and a few sections still read like an internal revision plan rather than a final submission manuscript.

## P0. Must Fix Before Submission

### 1. Code and data availability URLs appear unavailable

Location:

- `NMI_Submission.tex:817`
- `NMI_Submission.tex:821`

Problem:

The manuscript states that code and data are provided at:

- `https://github.com/s2p2e-project/s2p2e`
- `https://github.com/s2p2e-project/s2p2e-data`

These links currently appear unavailable/404 from a browser check. If submitted as written, this is a direct credibility risk because the text says the release already exists.

Why reviewers may object:

Nature-family journals place heavy emphasis on availability statements. A non-existent or private URL written as public evidence can look like an unsupported reproducibility claim.

Recommended fix:

Replace “are provided at” with a submission-safe statement unless the repositories are made public before submission. For example:

`Code will be made available in a public repository upon publication; a review-access archive is provided with the submission.`

If the repository is private for review, include a working anonymized or token-free reviewer access link. Do not include placeholder organizations.

### 2. Main text still contains future/internal planning language

Location:

- `NMI_Submission.tex:701`
- `NMI_Submission.tex:757`
- `Supplementary_Information.tex:64`
- `Supplementary_Information.tex:106`
- `Supplementary_Information.tex:116`
- `Supplementary_Information.tex:337`
- `Supplementary_Information.tex:353`

Problem:

The paper uses language such as “final release should include,” “the key additions are,” “the final package should expose,” and “at release time.” This reads like an internal development checklist, not a completed submission.

Why reviewers may object:

It suggests that parts of the evidence package, release package, or supplementary documentation are still aspirational. This conflicts with other statements claiming that all primary claims are complete and measured.

Recommended fix:

Convert all future/internal language into either completed statements or remove it from the paper. If something is not completed, move it to an internal checklist and weaken the corresponding claim in the manuscript.

Example replacement:

`The release includes split manifests, source data, evaluation scripts and checkpoint-selection rules.`

Only use this if those files actually exist in the submitted package.

### 3. The claim map marks all major claims as “complete”

Location:

- `NMI_Submission.tex:805-809`
- `Supplementary_Information.tex:367`

Problem:

The table says all main claims are “complete,” and the supplement says all primary claims are supported by measured values. This is too absolute for a Nature submission, especially because several claims depend on limited hardware episodes, internal scene libraries, and still-incomplete public release links.

Why reviewers may object:

“Complete” invites reviewers to find any missing support. A more defensible status vocabulary is “supported in this study,” “supported under bounded tabletop conditions,” or “supportive but limited.”

Recommended fix:

Replace `complete` with scope-aware labels:

- `supported under reported protocol`
- `primary evidence`
- `supportive evidence`
- `requires broader validation`

Also remove the sentence “All primary claims are supported...” or qualify it as “The reported primary claims are supported under the stated tabletop protocol.”

### 4. Dataset accounting table uses “Current count” and “Final count”

Location:

- `NMI_Submission.tex:344-360`

Problem:

The table has columns named `Current count` and `Final count`. In a submission manuscript, this suggests the dataset is still changing.

Why reviewers may object:

It makes the final protocol look unlocked. This is especially risky because the paper repeatedly says results are from a “locked protocol.”

Recommended fix:

Rename columns to final scientific meanings, for example:

- `Training/evaluation count`
- `Held-out evaluation count`
- `Evidence used for claim`

Remove the word “Current” unless the manuscript is explicitly a protocol paper.

### 5. Source-data provenance claims are not backed by raw artifacts inside the submission folder

Location:

- `source_data/README.md`
- `source_data/value_provenance_registry.csv`
- `source_data/table_module_metrics.csv`

Problem:

The source-data files cite provenance paths such as `artifacts/s2p2e_g50/...` and `artifacts/actor_critic_droid/...`, but those artifacts are not inside `overleaf_nature_20260606`. They may exist elsewhere in the project, but the submitted Overleaf folder alone does not contain them.

Why reviewers may object:

The source-data package claims auditability but does not independently include the raw logs, split manifests, exclusion logs, or scripts needed to reproduce table values from the submission package alone.

Recommended fix:

Either include the referenced artifacts in the submission/review package or revise the source-data README to say these are exported table values, not full raw provenance. For Nature-style review, prepare a separate `review_artifacts/` or repository release containing logs, manifests, configs and scripts.

### 6. Statistical unit and sample-size language is inconsistent

Location:

- `NMI_Submission.tex:306`
- `NMI_Submission.tex:369-391`
- `NMI_Submission.tex:699`
- `source_data/table_grasp_main.csv`

Problem:

The manuscript says all results use `N=5 random seeds` and tables report mean ± SD across seeds, but source-data tables list `n=500` for the same metrics. The manuscript also refers to 50 scenes and 500 trials, but it is not explicit whether 500 means 50 scenes × 10 repetitions, 50 scenes × 5 seeds × 2 trials, or another design.

Why reviewers may object:

Robotics reviewers will distinguish scenes, episodes, seeds, trajectories and frames. Ambiguous `N` definitions can undermine statistics even if the raw performance is strong.

Recommended fix:

At every main result table, state:

- independent unit
- number of scenes
- number of trials/episodes
- number of seeds
- whether SD is across seeds, scenes, or episodes

Example:

`Values are mean ± SD across five random seeds; each seed evaluates the same 50 scene instances with two target trials per scene, yielding 500 total trials per method.`

Only use that wording if it matches the real protocol.

### 7. Hardware and stress-test evidence may be too small for the current strength of generalization claims

Location:

- `NMI_Submission.tex:41`
- `NMI_Submission.tex:567`
- `NMI_Submission.tex:613-624`
- `NMI_Submission.tex:751-764`
- `source_data/stress_test_sim2real.csv`

Problem:

The text states that generalization is validated across multiple axes and that stress tests support robust sim-to-real transfer. However, stress tests use 30 episodes per condition and hardware transfer uses 50 episodes per method.

Why reviewers may object:

For Nature Machine Intelligence, “generalization” and “robust transfer” are high-bar claims. The evidence is useful, but the current sample size supports a bounded demonstration rather than broad generalization.

Recommended fix:

Use scope-limited language:

`The results provide bounded evidence of robustness under three controlled perturbations.`

Avoid:

`Generalization is validated...`

Unless more hardware episodes, more objects, more scenes, and cross-lab/second-robot experiments are added.

## P1. Strongly Recommended Fixes

### 8. “Key additions are” paragraph sounds like revision notes

Location:

- `NMI_Submission.tex:757`

Problem:

The Discussion says “To make the manuscript maximally persuasive, the key additions are...”. In a final paper, this reads as meta-commentary about manuscript preparation rather than scientific interpretation.

Recommended fix:

Rewrite as:

`The strongest support for the generalization claim comes from three additional analyses: ...`

Or move this content to the audit/response document, not the main manuscript.

### 9. “Conference-style single-file manuscript” weakens the submission image

Location:

- `NMI_Submission.tex:701`

Problem:

The text says the source bundle is “still organized as a conference-style single-file manuscript.” This may signal that the paper has not been fully adapted to Nature style.

Recommended fix:

Remove the phrase. Replace with:

`The supplementary package is organized to make split definitions, source data and checkpoint selection auditable.`

### 10. AnyGrasp baseline cites the wrong BibTeX key

Location:

- `NMI_Submission.tex:308`
- `references.bib:87`
- `references.bib:157`

Problem:

The baseline says `AnyGrasp~\cite{fu2024mobilealoha}`, but `fu2024mobilealoha` is Mobile ALOHA. The AnyGrasp entry appears separately in `references.bib` around line 157.

Why reviewers may object:

Wrong citations in baseline descriptions make the comparison look careless.

Recommended fix:

Use the actual AnyGrasp BibTeX key. If the key is not named clearly, rename it to something like `fang2023anygrasp` or whatever matches the actual entry metadata.

### 11. GPT-4V + chain-of-thought baseline is methodologically risky

Location:

- `NMI_Submission.tex:308`
- `NMI_Submission.tex:386`

Problem:

The paper includes `GPT-4V + CoT` as a baseline but does not specify model version, prompt, input resolution, decoding settings, safety filters, or whether chain-of-thought was actually exposed or only summarized.

Why reviewers may object:

Closed model baselines are hard to reproduce and can be criticized as unfair if prompt engineering is underspecified.

Recommended fix:

Either provide a full prompt/protocol in the supplement or demote this row to a qualitative reference. For a Nature submission, avoid relying on this row for any central claim.

### 12. Policy-bridge DROID evidence is underexplained

Location:

- `NMI_Submission.tex:310`
- `NMI_Submission.tex:334`
- `NMI_Submission.tex:496`
- `source_data/table_claim_sample_accounting.csv`

Problem:

The paper mentions 1,134 converted DROID transitions and a “held-out 300-transition audit,” but the actual evaluation protocol and the held-out 300-transition source table are not visible in the main source-data list beyond registry text.

Why reviewers may object:

This can look like a thin add-on to imply public-data validation without a complete public-data experiment.

Recommended fix:

Add a small supplementary table with train/validation/test transition counts, conversion fields, loss metrics, and what constitutes success. If this is only a sanity check, state that clearly and remove it from the main claim map.

### 13. Force-tracking main table has only five trials per regime

Location:

- `NMI_Submission.tex:438-450`
- `source_data/table_force_tracking.csv`

Problem:

The force-tracking table reports `n=5` for each stiffness regime, while later waveform metrics use 30 fragile trials. The contact-robustness claim is stronger than the small main-table sample suggests.

Recommended fix:

Move the 30-trial waveform evidence closer to the force-tracking table, or state that the `n=5` table is a calibration summary and the 30-trial waveform analysis is the primary contact evidence.

### 14. Sim-to-real comparison against TD-MPC2 may be unfair or insufficiently specified

Location:

- `NMI_Submission.tex:308`
- `NMI_Submission.tex:572-583`
- `NMI_Submission.tex:681-692`

Problem:

TD-MPC2 is a model-based continuous-control method, while the paper’s task involves language-conditioned perception and grasping. The exact observation/action interface for TD-MPC2 is not described enough to make the comparison fair.

Recommended fix:

Clarify what TD-MPC2 receives as input, how task goals are encoded, how perception outputs are connected, and whether it is being evaluated as a control baseline rather than a full language-manipulation baseline.

### 15. “Physics Prior Encoder” novelty could be challenged as incremental

Location:

- `NMI_Submission.tex:70`
- `NMI_Submission.tex:176-216`

Problem:

The PPE is an MLP trained on inverse dynamics with reward shaping and contrastive separation. Reviewers may view this as a known idea unless the paper clearly states what is novel: the interface, the staged integration, the empirical finding, or the cross-embodiment conditioning.

Recommended fix:

Add a sharper novelty statement:

`The contribution is not inverse-dynamics learning itself, but the use of a frozen inverse-dynamics prior as an inspectable feasibility interface between retrieval-conditioned pose proposal and residual execution.`

### 16. Knowledge-base construction needs more concrete reproducibility detail

Location:

- `NMI_Submission.tex:120-172`
- `Supplementary_Information.tex:31-67`

Problem:

The manuscript says semantic descriptions are generated automatically and 10% are human spot-checked, but does not specify prompt templates, schema fields, rejection criteria, quality thresholds, or exact asset counts after filtering.

Recommended fix:

Add a supplementary table:

- raw assets
- retained assets
- removed assets
- reason codes
- number of KB entries per tier
- spot-check pass/fail counts
- schema version

### 17. The “complete evidence base” wording is too strong

Location:

- `NMI_Submission.tex:655`

Problem:

The phrase “complete evidence base” is vulnerable because the paper itself admits limitations: one manipulator family, rigid/semi-rigid objects, front-facing viewpoints and limited hardware perturbations.

Recommended fix:

Use:

`These tables provide the evidence base for the bounded generalization and robustness claims reported here.`

### 18. Failure-case analysis appears only in supplement and may need source data

Location:

- `Supplementary_Information.tex:313-334`

Problem:

The supplement states 63 failed grasps and gives a detailed category breakdown, but the source-data directory does not visibly include a failure-case CSV with per-episode labels.

Recommended fix:

Add `failure_case_breakdown.csv` with episode ID, split, failure category, labeler/procedure, and evidence field. If labels are qualitative, state who labeled them and whether there was double checking.

## P2. Polish and Presentation Issues

### 19. Some main-text sections are too much like a review rebuttal

Location:

- `NMI_Submission.tex:364`
- `NMI_Submission.tex:701`
- `NMI_Submission.tex:742`

Problem:

Phrases such as “Nature-style review often focuses...” and “a common weakness of otherwise strong systems papers...” are useful internally but may sound defensive in the manuscript.

Recommended fix:

Rewrite these as scientific motivations, not reviewer-management commentary.

### 20. The title may be too broad for the actual evidence

Location:

- `NMI_Submission.tex:29`

Problem:

`Physical Grounding for Language-Driven Robotic Manipulation` sounds broad and field-level. The evidence is mainly tabletop cluttered manipulation with Franka Panda.

Recommended fix:

Consider a more scoped title:

`Retrieval and Feasibility Grounding for Language-Driven Tabletop Manipulation`

Or keep the current title but ensure the abstract and conclusion stay bounded.

### 21. Abstract has many numbers but little explicit novelty

Location:

- `NMI_Submission.tex:47`

Problem:

The abstract is quantitative, but the novelty is phrased as a modular framework. It could still be unclear what is scientifically new beyond combining retrieval, inverse dynamics and residual control.

Recommended fix:

Add one sentence that states the insight:

`The central design is to expose semantic grounding, feasibility prediction and execution compensation as separately testable interfaces rather than a single latent policy.`

### 22. The paper uses many abbreviations early

Location:

- `NMI_Submission.tex:47-78`

Problem:

S2P2E, VLA, RAG, PPE, PID, GSR, PIR and TCR appear quickly. This may slow cross-disciplinary NMI readers.

Recommended fix:

Keep only essential abbreviations in the Abstract and move the rest to the glossary or first Results table.

### 23. Tables are numerous for a main manuscript

Location:

- `NMI_Submission.tex` overall

Problem:

The main text contains many tables, including claim maps, glossary and supplementary package overview. These are useful, but they may make the paper feel more like a report than a Nature-family article.

Recommended fix:

Keep the core evidence tables in the main text and move glossary, claim map and supplementary package overview to supplementary information unless the journal format allows them in the main manuscript.

### 24. Some reference entries may be future/preprint-sensitive

Location:

- `NMI_Submission.tex:87`
- `references.bib:327`

Problem:

The citation key `xie20263dgsmrag` suggests a 2026 work. If this is unpublished, future-dated, or not peer-reviewed, it should be checked carefully.

Recommended fix:

Verify all future-year and preprint references before submission. If not essential, remove or replace with stable peer-reviewed references.

### 25. Line numbering works, but table-internal rows are not separately line-numbered

Location:

- `NMI_Submission.tex:13-25`

Problem:

The PDF has manuscript line numbers, but LaTeX `lineno` generally does not number every row inside floats/tables. This is normal, but if the submission system literally requires every line including tables, the current PDF may not satisfy that strict interpretation.

Recommended fix:

If the journal portal flags this, move large tables after references or provide a separate line-numbered text-only manuscript. Otherwise the current `lineno` setup is standard for review manuscripts.

## Claim-Evidence Risk Map

| Claim | Current evidence | Risk level | Recommended action |
|---|---|---:|---|
| Retrieval improves cluttered grounding | Main grasp table, KB ablation, unseen/dense slices | Medium | Clarify split independence and KB construction details |
| PPE reduces physical infeasibility | Violation table, safety-tail table, PPE RMSE | Medium | Clarify torque limits, simulator label generation, and physical meaning of surrogate RMSE |
| Residual control improves contact robustness | Force RMSE table, waveform/safety-tail metrics | Medium-High | Increase prominence of 30-trial waveform evidence; explain `n=5` table |
| Sim-to-real transfer is robust | 50 hardware episodes + three 30-episode stress tests | High | Scope claim as bounded tabletop transfer; avoid broad generalization wording |
| Cross-embodiment generalization | Inverse-dynamics RMSE only | High | Keep as supportive PPE transfer only, not full manipulation transfer |
| Reproducibility/open-source readiness | Source data plus stated GitHub links | High | Make repositories or review archives real and accessible before submission |

## Recommended Immediate Revision Order

1. Fix code/data availability statements and repository accessibility.
2. Remove future/internal checklist language from main and supplementary text.
3. Replace “complete” and “all primary claims” wording with scoped support language.
4. Clarify all sample sizes and independent units in table captions.
5. Correct the AnyGrasp citation.
6. Add or reference raw provenance artifacts, split manifests and exclusion/failure logs.
7. Rewrite the Discussion paragraph that currently says “key additions are.”
8. Demote or fully specify GPT-4V+CoT and DROID policy-bridge evidence.
9. Move administrative tables to supplement if the main text feels overloaded.
10. Re-run final PDF compile and visual audit after edits.

## Submission Readiness Verdict

Current status: close to a polished manuscript package structurally, but not yet safe for direct Nature Machine Intelligence submission.

Primary reason: the manuscript claims completion and public availability more strongly than the visible package supports. The fastest path to submission readiness is to make the availability package real, remove internal/future-facing language, and scope the generalization claims more conservatively.
