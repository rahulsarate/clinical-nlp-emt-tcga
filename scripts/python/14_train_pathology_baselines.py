#!/usr/bin/env python3
"""Train weakly supervised pathology baseline models and export document features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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
MIN_CLASS_EXAMPLES = 5
MIN_TOTAL_REPORTS = 20


def log_package_versions() -> None:
    """Log package versions used by the baseline training script."""
    import sklearn

    logger.info("pandas==%s, scikit-learn==%s", pd.__version__, sklearn.__version__)


def load_pipeline_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized reports and weak labels, returning empty frames when absent."""
    normalized_path = PROJECT_ROOT / config["output"]["normalized_reports"]
    weak_labels_path = PROJECT_ROOT / config["output"]["weak_labels"]

    if normalized_path.exists():
        normalized_df = pd.read_parquet(normalized_path)
    else:
        normalized_df = build_empty_dataframe(
            [
                "case_submitter_id",
                "file_id",
                "file_name",
                "document_type",
                "extraction_mode",
                "normalized_text",
                "section_count",
                "alpha_ratio",
                "ocr_noise_ratio",
                "document_quality",
            ]
        )

    if weak_labels_path.exists():
        weak_labels_df = pd.read_parquet(weak_labels_path)
    else:
        weak_labels_df = build_empty_dataframe(
            ["case_submitter_id", "file_id", "feature_name", "label", "matched_phrase", "confidence"]
        )

    return normalized_df, weak_labels_df


def prepare_documents(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one row per pathology report for baseline exports and scoring."""
    if normalized_df.empty:
        return normalized_df.copy()

    documents = normalized_df.drop_duplicates(subset=["file_id"]).copy()
    documents["normalized_text"] = documents["normalized_text"].fillna("").astype(str)
    documents["character_count"] = documents["normalized_text"].str.len()
    return documents


def select_training_documents(documents_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Filter documents to the subset suitable for baseline model training."""
    if documents_df.empty:
        return documents_df.copy()
    minimum_characters = int(config["cohort_filters"]["minimum_report_characters"])
    return documents_df[documents_df["character_count"] >= minimum_characters].copy()


def summarize_regex_hits(weak_labels_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate weak-label regex hits into document-level features and task targets."""
    if weak_labels_df.empty:
        regex_counts = build_empty_dataframe(["file_id"])
        targets = build_empty_dataframe(["file_id", "feature_name", "target_label", "target_support"])
        return regex_counts, targets

    feature_counts = (
        weak_labels_df.groupby(["file_id", "feature_name"]).size().unstack(fill_value=0).reset_index()
    )
    rename_map = {column: f"regex_{column}_candidate_count" for column in feature_counts.columns if column != "file_id"}
    regex_counts = feature_counts.rename(columns=rename_map)

    label_counts = (
        weak_labels_df.groupby(["file_id", "feature_name", "label"]).size().reset_index(name="label_count")
    )
    label_counts = label_counts.sort_values(
        by=["file_id", "feature_name", "label_count", "label"],
        ascending=[True, True, False, True],
    )
    targets = label_counts.drop_duplicates(subset=["file_id", "feature_name"]).rename(
        columns={"label": "target_label", "label_count": "target_support"}
    )
    return regex_counts, targets[["file_id", "feature_name", "target_label", "target_support"]]


def build_model_pipeline(config: dict[str, Any], random_state: int) -> Pipeline:
    """Build the TF-IDF plus logistic-regression pipeline from config."""
    tfidf_config = config["baseline_models"]["tfidf_classifier"]
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(tfidf_config["ngram_range"]),
                    min_df=int(tfidf_config["min_df"]),
                    max_features=int(tfidf_config["max_features"]),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=tfidf_config.get("class_weight"),
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_feature_models(
    documents_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one baseline classifier per feature and return predictions plus model metrics."""
    if documents_df.empty or targets_df.empty:
        empty_predictions = documents_df[["file_id"]].copy() if "file_id" in documents_df.columns else build_empty_dataframe(["file_id"])
        empty_metrics = build_empty_dataframe(
            [
                "feature_name",
                "trained",
                "n_reports",
                "n_classes",
                "macro_f1",
                "majority_macro_f1",
                "labels",
            ]
        )
        return empty_predictions, empty_metrics

    seed = int(config["project"]["seed"])
    prediction_tables: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for feature_name, feature_targets in targets_df.groupby("feature_name"):
        task_df = documents_df[["file_id", "normalized_text"]].merge(feature_targets, on="file_id", how="inner")
        if task_df.empty:
            continue

        label_counts = task_df["target_label"].value_counts()
        eligible_labels = label_counts[label_counts >= MIN_CLASS_EXAMPLES].index.tolist()
        task_df = task_df[task_df["target_label"].isin(eligible_labels)].copy()
        if len(task_df) < MIN_TOTAL_REPORTS or task_df["target_label"].nunique() < 2:
            metrics_rows.append(
                {
                    "feature_name": feature_name,
                    "trained": False,
                    "n_reports": int(len(task_df)),
                    "n_classes": int(task_df["target_label"].nunique()),
                    "macro_f1": pd.NA,
                    "majority_macro_f1": pd.NA,
                    "labels": "|".join(sorted(task_df["target_label"].unique())),
                }
            )
            continue

        test_size = 0.2 if len(task_df) >= 50 else 0.25
        train_df, test_df = train_test_split(
            task_df,
            test_size=test_size,
            random_state=seed,
            stratify=task_df["target_label"],
        )

        model = build_model_pipeline(config, seed)
        model.fit(train_df["normalized_text"], train_df["target_label"])
        test_predictions = model.predict(test_df["normalized_text"])

        majority_label = train_df["target_label"].mode().iloc[0]
        majority_predictions = pd.Series([majority_label] * len(test_df), index=test_df.index)
        macro_f1 = f1_score(test_df["target_label"], test_predictions, average="macro")
        majority_macro_f1 = f1_score(test_df["target_label"], majority_predictions, average="macro")

        probabilities = model.predict_proba(documents_df["normalized_text"])
        classes = list(model.named_steps["classifier"].classes_)
        prediction_df = pd.DataFrame({"file_id": documents_df["file_id"]})
        prediction_df[f"tfidf_{feature_name}_predicted_label"] = model.predict(documents_df["normalized_text"])
        for class_index, class_name in enumerate(classes):
            prediction_df[f"tfidf_{feature_name}_prob_{class_name}"] = probabilities[:, class_index]
        prediction_tables.append(prediction_df)

        metrics_rows.append(
            {
                "feature_name": feature_name,
                "trained": True,
                "n_reports": int(len(task_df)),
                "n_classes": int(len(classes)),
                "macro_f1": float(macro_f1),
                "majority_macro_f1": float(majority_macro_f1),
                "labels": "|".join(classes),
            }
        )

    if prediction_tables:
        merged_predictions = prediction_tables[0]
        for prediction_df in prediction_tables[1:]:
            merged_predictions = merged_predictions.merge(prediction_df, on="file_id", how="left")
    else:
        merged_predictions = documents_df[["file_id"]].copy()

    metrics_df = pd.DataFrame(metrics_rows)
    return merged_predictions, metrics_df


def build_benchmark_matrix(model_metrics_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build a benchmark matrix aligned to configured baseline rows."""
    metric_columns = config["benchmarking"]["metrics"]
    benchmark_rows: list[dict[str, Any]] = []
    trained_metrics = model_metrics_df[model_metrics_df["trained"] == True].copy() if not model_metrics_df.empty else build_empty_dataframe(["feature_name", "macro_f1"])
    average_macro_f1 = float(trained_metrics["macro_f1"].mean()) if not trained_metrics.empty else pd.NA

    for row_name in config["benchmarking"]["baseline_rows"]:
        row = {"model_name": row_name}
        for metric_name in metric_columns:
            row[metric_name] = pd.NA
        if row_name == "regex_only":
            row["extraction_macro_f1"] = pd.NA
        if row_name == "tfidf_logreg":
            row["extraction_macro_f1"] = average_macro_f1
        benchmark_rows.append(row)
    return pd.DataFrame(benchmark_rows, columns=["model_name", *metric_columns])


def main() -> None:
    """Train baseline pathology models and export document-level features."""
    log_package_versions()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    normalized_df, weak_labels_df = load_pipeline_inputs(config)
    documents_df = prepare_documents(normalized_df)
    training_documents_df = select_training_documents(documents_df, config)
    regex_counts_df, targets_df = summarize_regex_hits(weak_labels_df)
    model_predictions_df, model_metrics_df = train_feature_models(training_documents_df, targets_df, config)

    baseline_features_df = documents_df.merge(regex_counts_df, on="file_id", how="left")
    baseline_features_df = baseline_features_df.merge(model_predictions_df, on="file_id", how="left")
    benchmark_matrix_df = build_benchmark_matrix(model_metrics_df, config)

    baseline_output = PROJECT_ROOT / config["output"]["baseline_features"]
    benchmark_output = PROJECT_ROOT / config["output"]["benchmark_matrix"]
    model_summary_output = PROJECT_ROOT / "analysis" / "clinical_nlp" / "models" / "baseline_model_summary.csv"

    ensure_parent_dir(baseline_output)
    ensure_parent_dir(benchmark_output)
    ensure_parent_dir(model_summary_output)
    write_parquet(baseline_features_df, baseline_output)
    write_csv(benchmark_matrix_df, benchmark_output)
    write_csv(model_metrics_df, model_summary_output)


if __name__ == "__main__":
    main()