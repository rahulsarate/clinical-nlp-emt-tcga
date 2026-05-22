#!/usr/bin/env python3
"""Update clinical NLP benchmark metrics with pathology-to-EMT concordance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.clinical_nlp_config import load_and_validate_configs
from utils.clinical_nlp_io import build_empty_dataframe, ensure_parent_dir, write_csv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_config.yaml"
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_ontology.yaml"
AGGRESSIVE_LABEL_SUFFIXES = [
    "high_grade",
    "poorly_differentiated",
    "present",
    "desmoplastic",
    "high",
    "intermediate",
    "infiltrative",
    "mucinous",
]


def log_package_versions() -> None:
    """Log package versions used by the benchmark update script."""
    logger.info("pandas==%s", pd.__version__)


def load_artifacts(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load linked patient matrix and available model summaries."""
    patient_matrix_path = PROJECT_ROOT / config["output"]["patient_matrix"]
    benchmark_path = PROJECT_ROOT / config["output"]["benchmark_matrix"]
    baseline_summary_path = PROJECT_ROOT / "analysis" / "clinical_nlp" / "models" / "baseline_model_summary.csv"
    transformer_summary_path = PROJECT_ROOT / "analysis" / "clinical_nlp" / "models" / "transformer_model_summary.csv"

    patient_matrix_df = pd.read_parquet(patient_matrix_path) if patient_matrix_path.exists() else build_empty_dataframe(["case_submitter_id", "emt_score", "emt_label"])
    benchmark_matrix_df = pd.read_csv(benchmark_path) if benchmark_path.exists() else build_empty_dataframe(["model_name"] + config["benchmarking"]["metrics"])
    baseline_summary_df = pd.read_csv(baseline_summary_path) if baseline_summary_path.exists() else build_empty_dataframe(["feature_name", "trained", "macro_f1"])
    transformer_summary_df = pd.read_csv(transformer_summary_path) if transformer_summary_path.exists() else build_empty_dataframe(["feature_name", "trained", "macro_f1"])
    return patient_matrix_df, benchmark_matrix_df, baseline_summary_df, transformer_summary_df


def average_macro_f1(summary_df: pd.DataFrame) -> float | pd.NA:
    """Compute the mean macro F1 across trained feature tasks."""
    if summary_df.empty or "trained" not in summary_df.columns:
        return pd.NA
    trained_df = summary_df[summary_df["trained"] == True].copy()
    if trained_df.empty:
        return pd.NA
    return float(pd.to_numeric(trained_df["macro_f1"], errors="coerce").dropna().mean())


def build_regex_score(patient_matrix_df: pd.DataFrame) -> pd.Series:
    """Build a rule-based pathology risk score from regex candidate counts."""
    regex_columns = [column for column in patient_matrix_df.columns if column.startswith("regex_") and column.endswith("_candidate_count")]
    if not regex_columns:
        return pd.Series(np.zeros(len(patient_matrix_df)), index=patient_matrix_df.index)
    return patient_matrix_df[regex_columns].fillna(0).sum(axis=1).astype(float)


def build_model_probability_score(patient_matrix_df: pd.DataFrame, prefix: str) -> pd.Series:
    """Build an EMT-oriented risk score from task probability columns for one model family."""
    probability_columns = [column for column in patient_matrix_df.columns if column.startswith(f"{prefix}_") and "_prob_" in column]
    selected_columns = [
        column for column in probability_columns if any(column.endswith(f"_{suffix}") for suffix in AGGRESSIVE_LABEL_SUFFIXES)
    ]
    if not selected_columns:
        return pd.Series(np.zeros(len(patient_matrix_df)), index=patient_matrix_df.index)
    probability_df = patient_matrix_df[selected_columns].apply(pd.to_numeric, errors="coerce")
    return probability_df.mean(axis=1, skipna=True)


def compute_concordance_metrics(patient_matrix_df: pd.DataFrame, score: pd.Series) -> dict[str, Any]:
    """Compute pathology-versus-EMT concordance metrics from a model score."""
    if patient_matrix_df.empty or score.empty:
        return {
            "emt_auc": pd.NA,
            "emt_auprc": pd.NA,
            "feature_emt_correlation": pd.NA,
            "calibration_brier": pd.NA,
        }

    aligned_df = patient_matrix_df.copy()
    aligned_df["pathology_score"] = pd.to_numeric(score, errors="coerce")
    aligned_df = aligned_df.dropna(subset=["pathology_score", "emt_score"]).copy()
    if aligned_df.empty:
        return {
            "emt_auc": pd.NA,
            "emt_auprc": pd.NA,
            "feature_emt_correlation": pd.NA,
            "calibration_brier": pd.NA,
        }

    if aligned_df["pathology_score"].nunique() < 2:
        return {
            "emt_auc": pd.NA,
            "emt_auprc": pd.NA,
            "feature_emt_correlation": pd.NA,
            "calibration_brier": pd.NA,
        }

    correlation_value = spearmanr(aligned_df["pathology_score"], aligned_df["emt_score"], nan_policy="omit").correlation

    binary_df = aligned_df[aligned_df["emt_label"].isin(["high", "low"])].copy()
    if binary_df.empty or binary_df["emt_label"].nunique() < 2:
        return {
            "emt_auc": pd.NA,
            "emt_auprc": pd.NA,
            "feature_emt_correlation": correlation_value,
            "calibration_brier": pd.NA,
        }

    binary_df["emt_binary"] = (binary_df["emt_label"] == "high").astype(int)
    auc_value = roc_auc_score(binary_df["emt_binary"], binary_df["pathology_score"])
    auprc_value = average_precision_score(binary_df["emt_binary"], binary_df["pathology_score"])

    score_min = float(binary_df["pathology_score"].min())
    score_max = float(binary_df["pathology_score"].max())
    if score_max > score_min:
        scaled_score = (binary_df["pathology_score"] - score_min) / (score_max - score_min)
        brier_value = float(((scaled_score - binary_df["emt_binary"]) ** 2).mean())
    else:
        brier_value = pd.NA

    return {
        "emt_auc": float(auc_value),
        "emt_auprc": float(auprc_value),
        "feature_emt_correlation": float(correlation_value) if correlation_value is not None else pd.NA,
        "calibration_brier": brier_value,
    }


def update_row(benchmark_df: pd.DataFrame, model_name: str, updates: dict[str, Any]) -> pd.DataFrame:
    """Apply metric updates to one benchmark row, creating it if missing."""
    if model_name not in benchmark_df["model_name"].tolist():
        benchmark_df = pd.concat([benchmark_df, pd.DataFrame([{column: pd.NA for column in benchmark_df.columns}])], ignore_index=True)
        benchmark_df.loc[benchmark_df.index[-1], "model_name"] = model_name
    row_mask = benchmark_df["model_name"] == model_name
    for column_name, value in updates.items():
        benchmark_df.loc[row_mask, column_name] = value
    return benchmark_df


def main() -> None:
    """Update benchmark metrics using linked pathology and EMT outputs."""
    log_package_versions()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    patient_matrix_df, benchmark_matrix_df, baseline_summary_df, transformer_summary_df = load_artifacts(config)
    if benchmark_matrix_df.empty:
        benchmark_matrix_df = pd.DataFrame(columns=["model_name", *config["benchmarking"]["metrics"]])
        for model_name in config["benchmarking"]["baseline_rows"]:
            benchmark_matrix_df = pd.concat(
                [benchmark_matrix_df, pd.DataFrame([{"model_name": model_name, **{metric: pd.NA for metric in config["benchmarking"]["metrics"]}}])],
                ignore_index=True,
            )

    regex_updates = compute_concordance_metrics(patient_matrix_df, build_regex_score(patient_matrix_df))
    tfidf_updates = compute_concordance_metrics(patient_matrix_df, build_model_probability_score(patient_matrix_df, "tfidf"))
    transformer_updates = compute_concordance_metrics(patient_matrix_df, build_model_probability_score(patient_matrix_df, "transformer"))

    tfidf_updates["extraction_macro_f1"] = average_macro_f1(baseline_summary_df)
    transformer_updates["extraction_macro_f1"] = average_macro_f1(transformer_summary_df)

    benchmark_matrix_df = update_row(benchmark_matrix_df, "regex_plus_context", regex_updates)
    benchmark_matrix_df = update_row(benchmark_matrix_df, "tfidf_logreg", tfidf_updates)
    benchmark_matrix_df = update_row(benchmark_matrix_df, "pubmedbert_multitask", transformer_updates)

    output_path = PROJECT_ROOT / config["output"]["benchmark_matrix"]
    ensure_parent_dir(output_path)
    write_csv(benchmark_matrix_df, output_path)


if __name__ == "__main__":
    main()