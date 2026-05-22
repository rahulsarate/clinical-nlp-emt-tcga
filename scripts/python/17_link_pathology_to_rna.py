#!/usr/bin/env python3
"""Link pathology baseline features to TCGA bulk RNA EMT scores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from utils.clinical_nlp_config import load_and_validate_configs
from utils.clinical_nlp_io import build_empty_dataframe, ensure_parent_dir, write_csv, write_parquet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_config.yaml"
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_ontology.yaml"


def log_package_versions() -> None:
    """Log package versions used by the integration script."""
    logger.info("pandas==%s", pd.__version__)


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load baseline features, optional transformer predictions, and RNA EMT scores."""
    baseline_path = PROJECT_ROOT / config["output"]["baseline_features"]
    transformer_path = PROJECT_ROOT / config["output"]["transformer_predictions"]
    emt_path = PROJECT_ROOT / config["output"]["emt_scores"]

    baseline_df = pd.read_parquet(baseline_path) if baseline_path.exists() else build_empty_dataframe(["case_submitter_id", "file_id"])
    transformer_df = pd.read_parquet(transformer_path) if transformer_path.exists() else build_empty_dataframe(["file_id"])
    emt_df = pd.read_csv(emt_path) if emt_path.exists() else build_empty_dataframe(["project_id", "case_submitter_id", "emt_score", "emt_label"])
    return baseline_df, transformer_df, emt_df


def merge_report_level_predictions(baseline_df: pd.DataFrame, transformer_df: pd.DataFrame) -> pd.DataFrame:
    """Merge optional transformer predictions into the baseline feature table."""
    if baseline_df.empty or transformer_df.empty:
        return baseline_df
    return baseline_df.merge(transformer_df, on="file_id", how="left")


def select_representative_reports(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Select one representative pathology report per case for multimodal linkage."""
    if baseline_df.empty:
        return baseline_df.copy()

    report_df = baseline_df.copy()
    quality_rank = {"high": 0, "low": 1, "empty": 2}
    report_df["quality_rank"] = report_df["document_quality"].map(quality_rank).fillna(3)
    regex_columns = [column for column in report_df.columns if column.startswith("regex_") and column.endswith("_candidate_count")]
    if regex_columns:
        report_df["total_candidate_count"] = report_df[regex_columns].fillna(0).sum(axis=1)
    else:
        report_df["total_candidate_count"] = 0
    report_df["character_count"] = pd.to_numeric(report_df.get("character_count", 0), errors="coerce").fillna(0)

    report_df = report_df.sort_values(
        by=["case_submitter_id", "quality_rank", "total_candidate_count", "character_count"],
        ascending=[True, True, False, False],
    )
    selected_df = report_df.drop_duplicates(subset=["case_submitter_id"]).drop(columns=["quality_rank"])
    return selected_df


def build_patient_matrix(selected_reports_df: pd.DataFrame, emt_df: pd.DataFrame) -> pd.DataFrame:
    """Join representative pathology reports to EMT scores on case ID."""
    if selected_reports_df.empty or emt_df.empty:
        return build_empty_dataframe(["project_id", "case_submitter_id", "file_id", "emt_score", "emt_label"])

    patient_matrix = selected_reports_df.merge(
        emt_df,
        on="case_submitter_id",
        how="inner",
        suffixes=("_pathology", "_rna"),
    )
    preferred_columns = [
        "project_id",
        "case_submitter_id",
        "file_id",
        "file_name",
        "document_quality",
        "total_candidate_count",
        "project_humanized_emt_score",
        "hallmark_epithelial_mesenchymal_transition_score",
        "emt_score",
        "emt_label",
    ]
    available_columns = [column for column in preferred_columns if column in patient_matrix.columns]
    remaining_columns = [column for column in patient_matrix.columns if column not in available_columns]
    return patient_matrix[available_columns + remaining_columns]


def build_linkage_summary(selected_reports_df: pd.DataFrame, emt_df: pd.DataFrame, patient_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the pathology-to-RNA linkage coverage."""
    summary_rows = [
        {"metric": "representative_pathology_cases", "value": int(selected_reports_df["case_submitter_id"].nunique()) if not selected_reports_df.empty else 0},
        {"metric": "rna_emt_cases", "value": int(emt_df["case_submitter_id"].nunique()) if not emt_df.empty else 0},
        {"metric": "linked_cases", "value": int(patient_matrix_df["case_submitter_id"].nunique()) if not patient_matrix_df.empty else 0},
    ]
    return pd.DataFrame(summary_rows)


def main() -> None:
    """Create the patient-level pathology-to-RNA linkage matrix."""
    log_package_versions()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    baseline_df, transformer_df, emt_df = load_inputs(config)
    baseline_df = merge_report_level_predictions(baseline_df, transformer_df)
    selected_reports_df = select_representative_reports(baseline_df)
    patient_matrix_df = build_patient_matrix(selected_reports_df, emt_df)
    linkage_summary_df = build_linkage_summary(selected_reports_df, emt_df, patient_matrix_df)

    patient_matrix_path = PROJECT_ROOT / config["output"]["patient_matrix"]
    summary_path = PROJECT_ROOT / "analysis" / "clinical_nlp" / "integration" / "patient_linkage_summary.csv"
    ensure_parent_dir(patient_matrix_path)
    ensure_parent_dir(summary_path)
    write_parquet(patient_matrix_df, patient_matrix_path)
    write_csv(linkage_summary_df, summary_path)


if __name__ == "__main__":
    main()