# Clinical NLP Config Acceptance Criteria

This document defines how an implementation agent should validate
[configs/clinical_nlp_config.yaml](/d:/VS%20CODE%20Project/scRNA_seq/configs/clinical_nlp_config.yaml)
and
[configs/clinical_nlp_ontology.yaml](/d:/VS%20CODE%20Project/scRNA_seq/configs/clinical_nlp_ontology.yaml)
before using them in the TCGA pathology EMT pipeline.

## Validation Goals

The config passes only if it is:

1. syntactically valid YAML
2. internally consistent
3. aligned with the file and directory contracts of the planned pipeline
4. safe for the local Windows machine memory budget
5. usable by both the rule-based and transformer-based extraction stages

## Required Validation Order

1. Parse both YAML files without coercion errors.
2. Confirm all required top-level keys exist.
3. Validate value types and allowed enumerations.
4. Validate cross-key consistency inside each file.
5. Validate cross-file consistency between runtime config and ontology.
6. Validate output paths are relative workspace paths and do not point into `data/raw/`.
7. Validate the RAM guard is present and does not exceed 9 GB by default.

## Runtime Config Acceptance Criteria

### `project`

Acceptance criteria:

1. `name`, `owner`, `organism`, and `primary_use_case` must be non-empty strings.
2. `organism` must be `Homo sapiens` for this project.
3. `seed` must be an integer and equal to `42` unless a documented decision log entry explicitly changes it.

Failure conditions:

1. Missing any required key.
2. Non-string metadata values.
3. Seed missing or not an integer.

### `data_sources`

Acceptance criteria:

1. `gdc_api_base` must be an absolute HTTPS URL.
2. `gdc_files_endpoint` and `gdc_cases_endpoint` must start with `/`.
3. `projects` must contain at least `TCGA-BRCA` and `TCGA-COAD`.
4. `optional_projects` may be empty but if present must contain only TCGA-style project IDs.
5. `report_file_filters.data_category` must include `Clinical`.
6. `bulk_rna_filters.data_category` must include `Transcriptome Profiling`.
7. `bulk_rna_filters.data_type` must include `Gene Expression Quantification`.
8. `manifest_dir` and `cohort_dir` must be relative workspace paths.
9. Neither `manifest_dir` nor `cohort_dir` may resolve inside `data/raw/`.

Failure conditions:

1. Any GDC URL or endpoint malformed.
2. Missing BRCA or COAD.
3. Output paths targeting read-only or raw-data locations.

### `experimental_design`

Acceptance criteria:

1. `primary_cohorts` must exactly match or be a subset of `data_sources.projects`.
2. `matching_key_priority` must contain `submitter_id` before `case_id`.
3. `require_matched_bulk_rna` must be boolean.
4. `minimum_cases_per_cohort` must be a positive integer.

Failure conditions:

1. Primary cohorts not represented in `data_sources.projects`.
2. Matching keys absent or reversed without a documented override.

### `cohort_filters`

Acceptance criteria:

1. `allowed_sample_types` must include `Primary Tumor`.
2. `exclude_sample_types` must not contain `Primary Tumor`.
3. `report_selection_strategy` must be non-empty and machine-readable.
4. `minimum_report_characters` must be an integer greater than or equal to 100.
5. `keep_unmatched_reports_for_nlp_only_analysis` must be boolean.

Failure conditions:

1. Contradictory sample filters.
2. Minimum report length too low to support morphology extraction.

### `document_processing`

Acceptance criteria:

1. `preferred_input_order` must begin with `native_text`.
2. `ocr_enabled` must be boolean.
3. If `ocr_enabled` is true, `ocr_backend` must be non-empty and `ocr_max_pages` must be a positive integer.
4. `section_headers` must contain at least `diagnosis`, `microscopic description`, `comment`, and `synoptic report`.
5. `document_quality_thresholds.min_alpha_ratio` must be a float between 0 and 1.
6. `autoswitch.enable_memory_guard` must be true.
7. `autoswitch.max_ram_gb` must be a number greater than 0 and less than or equal to 9.
8. `autoswitch.fallback_on_low_quality` and `fallback_on_oom_risk` must be boolean.

Failure conditions:

1. Native text is not the default path.
2. Memory guard absent or above 9 GB.
3. Required section headers missing.

### `annotation_strategy`

Acceptance criteria:

1. `gold_set_target_reports` must be an integer between 50 and 500.
2. `weak_supervision_enabled` and `disagreement_sampling_enabled` must be boolean.
3. `review_queue_batch_size` must be a positive integer.

Failure conditions:

1. Gold set too small to stratify.
2. Review queue batch size zero or negative.

### `baseline_models`

Acceptance criteria:

1. `regex_extractor_enabled` must be boolean.
2. If `context_pipeline_enabled` is true, `context_window_tokens` and `negation_scope_tokens` must be positive integers.
3. If `tfidf_classifier.enabled` is true, `ngram_range` must be a two-element integer list and `classifier` must be one of the supported baseline classifiers.
4. `max_features` must be a positive integer.

Failure conditions:

1. Invalid n-gram range shape.
2. Missing classifier when TF-IDF is enabled.

### `transformer_model`

Acceptance criteria:

1. `enabled` must be boolean.
2. If enabled, `model_name` must be a non-empty Hugging Face model identifier.
3. `max_length`, `stride`, `batch_size`, and `epochs` must be positive integers.
4. `learning_rate` must be positive and less than 0.01.
5. `task_mode` must be one of the supported modes for the training script.
6. `calibration_enabled` must be boolean.

Failure conditions:

1. Enabled transformer without a model name.
2. Batch size or sequence lengths invalid.

### `emt_scoring`

Acceptance criteria:

1. `score_method` and `label_strategy` must be non-empty strings.
2. `high_quantile` and `low_quantile` must be floats between 0 and 1.
3. `high_quantile` must be greater than `low_quantile`.
4. `signature_sets` must contain at least two entries, including a project-derived EMT set and one external reference set.
5. `gene_id_type` must be `hgnc_symbol` unless the scoring script explicitly supports another identifier.
6. `discard_mid_band_for_binary_training` must be boolean.

Failure conditions:

1. Quantiles inverted.
2. Only one EMT signature source defined.

### `benchmarking`

Acceptance criteria:

1. `baseline_rows` must contain manual, rules-based, classical ML, and transformer rows.
2. `metrics` must include at least one extraction metric, one classification metric, one biology metric, one throughput metric, and one memory metric.
3. `primary_selection_metric` must exist in `metrics` or be derivable from them.

Failure conditions:

1. Benchmark matrix omits either operational or biological metrics.
2. Primary selection metric not declared.

### `dashboard`

Acceptance criteria:

1. `enabled` must be boolean.
2. `export_dir` must be a relative path under `analysis/clinical_nlp/`.
3. `pages` must include overview, report review, feature evidence, molecular concordance, and benchmark matrix.
4. `default_cohort` must exist in either `primary_cohorts` or `expansion_cohorts`.

Failure conditions:

1. Export path outside `analysis/clinical_nlp/`.
2. Default cohort not configured elsewhere.

### `output`

Acceptance criteria:

1. Every value must be a relative workspace path string.
2. No output path may resolve inside `data/raw/`.
3. Every path must point into a directory already declared by the project skeleton.
4. Output keys needed by downstream scripts must exist exactly once.

Failure conditions:

1. Absolute or external filesystem paths.
2. Outputs directed into read-only raw-data locations.

## Ontology Config Acceptance Criteria

### `metadata`

Acceptance criteria:

1. `version`, `domain`, and `last_reviewed` must be non-empty strings.
2. `target_cancers` must include `TCGA-BRCA` and `TCGA-COAD`.

### `global_rules`

Acceptance criteria:

1. `negations` and `uncertainty` must be non-empty lists of strings.
2. `historical_context` and `family_context` must be lists of strings.
3. `ignore_sections_for_primary_labels` entries must correspond to section aliases or section IDs defined under `sections`.

Failure conditions:

1. Negation list empty.
2. Ignored sections not represented elsewhere.

### `sections`

Acceptance criteria:

1. Each section must define `priority` and `aliases`.
2. Priorities must be positive integers.
3. Aliases must be non-empty string lists.
4. `diagnosis`, `synoptic_report`, and `microscopic_description` must exist.

Failure conditions:

1. Missing required analytical sections.
2. Duplicate aliases that create ambiguous mapping without intention.

### `feature_schema`

Acceptance criteria applied to every feature block:

1. Each feature must define `task`, `allowed_labels`, `synonyms`, and `evidence_sections`.
2. `allowed_labels` must be a non-empty list of unique strings.
3. `synonyms` must be a mapping whose keys are valid labels or justified label groups.
4. `evidence_sections` must reference section IDs defined in `sections`.
5. Every feature must include an `unknown`-style fallback label where appropriate.

Failure conditions:

1. Missing task type.
2. Evidence sections referencing undefined section names.
3. Synonym groups that map to labels not present in `allowed_labels` without documented reason.

### Feature-specific checks

1. `invasion_type` must include at least `ductal`, `lobular`, `adenocarcinoma`, and `unknown`.
2. `invasive_border` must include `infiltrative` and at least one non-infiltrative comparator label.
3. `tumor_budding_score` must support ordered severity, not just binary presence.
4. `stromal_reaction` must include `desmoplastic`.
5. `fibrosis_or_desmoplasia` must remain binary or binary plus unknown.
6. `grade` must support both differentiation-style and numeric grade language.
7. `lymphovascular_invasion` and `perineural_invasion` must each support `present`, `absent`, `suspicious`, and `unknown`.

### `normalization_rules`

Acceptance criteria:

1. `abbreviation_expansion` must include `idc`, `ilc`, `lvi`, and `pni`.
2. `section_priority_for_conflicts` must reference valid section IDs.
3. `section_priority_for_conflicts` should place `synoptic_report` and `diagnosis` ahead of `gross_description`.

Failure conditions:

1. Missing common pathology abbreviations.
2. Conflict priority lists unknown sections.

### `evidence_rules`

Acceptance criteria:

1. `sentence_window` and `token_window` must be positive integers.
2. `prefer_high_priority_section_evidence` must be boolean.
3. `negate_if_negation_term_precedes_within_tokens` must be a positive integer.
4. `require_label_specific_synonym_or_regex` must be true for the initial release.

Failure conditions:

1. Windows too large for explainable evidence tracing.
2. Label-specific matching disabled without stronger alternative safeguards.

### `review_rules`

Acceptance criteria:

1. `always_review_if` must include conflict, low-quality, disagreement, and low-confidence triggers.
2. `do_not_autolabel_if` must include historical-only and family-history-only contexts.

Failure conditions:

1. Review escalation rules too weak to support auditability.
2. Historical context allowed to generate final labels automatically.

## Cross-file Consistency Checks

1. Every section ID referenced in ontology evidence rules must be discoverable from the section headers configured in `document_processing.section_headers`.
2. Every dashboard page expected by `dashboard.pages` must have a corresponding exportable artifact category in the runtime config output plan.
3. `document_processing.autoswitch.max_ram_gb` must be compatible with local machine policy and remain less than or equal to 9.
4. `benchmarking.baseline_rows` must include rows that can actually be produced from the enabled baseline and transformer settings.
5. `project.organism` must be `Homo sapiens` and `emt_scoring.gene_id_type` must be consistent with human gene symbols.

## Minimum Agent Validation Procedure

An implementation agent should refuse to proceed if any of the following checks fail:

1. YAML parsing succeeds for both files.
2. All required top-level sections are present.
3. All required section-local keys are present.
4. No output path points into `data/raw/`.
5. `max_ram_gb` is present and less than or equal to 9.
6. BRCA and COAD are both present in runtime and ontology scope.
7. Ontology evidence sections and runtime section headers can be reconciled.

If all checks pass, the config is accepted for implementation.