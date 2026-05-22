"""Config validation helpers for the clinical NLP EMT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.clinical_nlp_io import load_yaml_config, resolve_workspace_path


REQUIRED_RUNTIME_KEYS = [
    "project",
    "data_sources",
    "experimental_design",
    "cohort_filters",
    "document_processing",
    "annotation_strategy",
    "baseline_models",
    "transformer_model",
    "emt_scoring",
    "benchmarking",
    "dashboard",
    "output",
]

REQUIRED_ONTOLOGY_KEYS = [
    "metadata",
    "global_rules",
    "sections",
    "feature_schema",
    "normalization_rules",
    "evidence_rules",
    "review_rules",
]


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping")
    return value


def _require_list(config: dict[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config key '{key}' must be a non-empty list")
    return value


def _assert_relative_output(path_text: str) -> None:
    path = Path(path_text)
    if path.is_absolute():
        raise ValueError(f"Output path must be relative: {path_text}")
    if "raw" in path.parts and path.parts[:2] == ("data", "raw"):
        raise ValueError(f"Output path may not target data/raw: {path_text}")


def validate_runtime_config(config: dict[str, Any]) -> list[str]:
    """Validate runtime config structure and return warnings."""
    warnings: list[str] = []
    for key in REQUIRED_RUNTIME_KEYS:
        _require_mapping(config, key)

    project = _require_mapping(config, "project")
    if project.get("organism") != "Homo sapiens":
        raise ValueError("project.organism must be 'Homo sapiens'")
    if not isinstance(project.get("seed"), int):
        raise ValueError("project.seed must be an integer")

    data_sources = _require_mapping(config, "data_sources")
    projects = _require_list(data_sources, "projects")
    if "TCGA-BRCA" not in projects or "TCGA-COAD" not in projects:
        raise ValueError("data_sources.projects must include TCGA-BRCA and TCGA-COAD")
    for key in ["manifest_dir", "cohort_dir"]:
        _assert_relative_output(str(data_sources[key]))

    experimental_design = _require_mapping(config, "experimental_design")
    matching_keys = _require_list(experimental_design, "matching_key_priority")
    if matching_keys[:2] != ["submitter_id", "case_id"]:
        raise ValueError("matching_key_priority must start with submitter_id, case_id")

    cohort_filters = _require_mapping(config, "cohort_filters")
    allowed_sample_types = _require_list(cohort_filters, "allowed_sample_types")
    if "Primary Tumor" not in allowed_sample_types:
        raise ValueError("cohort_filters.allowed_sample_types must include 'Primary Tumor'")

    document_processing = _require_mapping(config, "document_processing")
    preferred_inputs = _require_list(document_processing, "preferred_input_order")
    if preferred_inputs[0] != "native_text":
        raise ValueError("document_processing.preferred_input_order must start with native_text")
    autoswitch = _require_mapping(document_processing, "autoswitch")
    max_ram_gb = autoswitch.get("max_ram_gb")
    if not isinstance(max_ram_gb, (int, float)) or max_ram_gb <= 0 or max_ram_gb > 9:
        raise ValueError("document_processing.autoswitch.max_ram_gb must be > 0 and <= 9")

    annotation_strategy = _require_mapping(config, "annotation_strategy")
    gold_target = annotation_strategy.get("gold_set_target_reports")
    if not isinstance(gold_target, int) or not 50 <= gold_target <= 500:
        raise ValueError("annotation_strategy.gold_set_target_reports must be between 50 and 500")

    transformer_model = _require_mapping(config, "transformer_model")
    if transformer_model.get("enabled") and not transformer_model.get("model_name"):
        raise ValueError("transformer_model.model_name is required when enabled")

    emt_scoring = _require_mapping(config, "emt_scoring")
    high_quantile = emt_scoring.get("high_quantile")
    low_quantile = emt_scoring.get("low_quantile")
    if not isinstance(high_quantile, (int, float)) or not isinstance(low_quantile, (int, float)):
        raise ValueError("emt_scoring quantiles must be numeric")
    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError("emt_scoring.low_quantile must be < high_quantile and both between 0 and 1")

    dashboard = _require_mapping(config, "dashboard")
    if not str(dashboard.get("export_dir", "")).startswith("analysis/clinical_nlp/"):
        raise ValueError("dashboard.export_dir must be under analysis/clinical_nlp/")

    output = _require_mapping(config, "output")
    for path_text in output.values():
        _assert_relative_output(str(path_text))
    return warnings


def validate_ontology_config(config: dict[str, Any]) -> list[str]:
    """Validate ontology config structure and return warnings."""
    warnings: list[str] = []
    for key in REQUIRED_ONTOLOGY_KEYS:
        _require_mapping(config, key)

    metadata = _require_mapping(config, "metadata")
    target_cancers = _require_list(metadata, "target_cancers")
    if "TCGA-BRCA" not in target_cancers or "TCGA-COAD" not in target_cancers:
        raise ValueError("ontology metadata.target_cancers must include TCGA-BRCA and TCGA-COAD")

    sections = _require_mapping(config, "sections")
    for required in ["diagnosis", "synoptic_report", "microscopic_description"]:
        if required not in sections:
            raise ValueError(f"sections must include '{required}'")

    feature_schema = _require_mapping(config, "feature_schema")
    required_features = [
        "invasion_type",
        "invasive_border",
        "tumor_budding_score",
        "stromal_reaction",
        "fibrosis_or_desmoplasia",
        "grade",
        "lymphovascular_invasion",
        "perineural_invasion",
    ]
    for feature_name in required_features:
        feature = _require_mapping(feature_schema, feature_name)
        _require_list(feature, "allowed_labels")
        _require_mapping(feature, "synonyms")
        evidence_sections = _require_list(feature, "evidence_sections")
        for section_name in evidence_sections:
            if section_name not in sections:
                raise ValueError(f"Feature '{feature_name}' references unknown section '{section_name}'")
    return warnings


def load_and_validate_configs(config_path: Path, ontology_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Load runtime and ontology configs and validate both files."""
    runtime_config = load_yaml_config(config_path)
    ontology_config = load_yaml_config(ontology_path)

    warnings = []
    warnings.extend(validate_runtime_config(runtime_config))
    warnings.extend(validate_ontology_config(ontology_config))

    runtime_sections = {header.lower() for header in runtime_config["document_processing"]["section_headers"]}
    ontology_sections = set(ontology_config["sections"].keys())
    aliases = {
        alias.lower()
        for section in ontology_config["sections"].values()
        for alias in section.get("aliases", [])
    }
    for expected in ["diagnosis", "synoptic report", "microscopic description"]:
        if expected not in runtime_sections:
            raise ValueError(f"document_processing.section_headers must include '{expected}'")
    if not ontology_sections:
        raise ValueError("ontology.sections may not be empty")
    if not runtime_sections.intersection(aliases.union(ontology_sections)):
        raise ValueError("Runtime section headers and ontology section aliases do not overlap")

    return runtime_config, ontology_config, warnings
