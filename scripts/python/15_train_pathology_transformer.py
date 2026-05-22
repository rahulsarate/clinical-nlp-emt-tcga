#!/usr/bin/env python3
"""Train biomedical-transformer pathology classifiers on weak labels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transformer training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-name", type=str, default=None, help="Restrict training to one feature task")
    parser.add_argument("--max-documents", type=int, default=None, help="Limit the number of documents encoded for smoke tests")
    return parser.parse_args()


def log_package_versions() -> None:
    """Log package versions used by the transformer training script."""
    import transformers

    logger.info(
        "pandas==%s, torch==%s, transformers==%s",
        pd.__version__,
        torch.__version__,
        transformers.__version__,
    )


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


def prepare_documents(normalized_df: pd.DataFrame, max_documents: int | None) -> pd.DataFrame:
    """Prepare one row per pathology report for transformer scoring."""
    if normalized_df.empty:
        return normalized_df.copy()

    documents = normalized_df.drop_duplicates(subset=["file_id"]).copy()
    documents["normalized_text"] = documents["normalized_text"].fillna("").astype(str)
    documents["character_count"] = documents["normalized_text"].str.len()
    if max_documents is not None:
        quality_rank = {"high": 0, "low": 1, "empty": 2}
        documents["quality_rank"] = documents["document_quality"].map(quality_rank).fillna(3)
        documents = documents.sort_values(by=["quality_rank", "character_count"], ascending=[True, False]).head(max_documents)
        documents = documents.drop(columns=["quality_rank"])
    return documents


def select_training_documents(documents_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Filter documents to the subset suitable for transformer-model training."""
    if documents_df.empty:
        return documents_df.copy()
    minimum_characters = int(config["cohort_filters"]["minimum_report_characters"])
    return documents_df[documents_df["character_count"] >= minimum_characters].copy()


def summarize_targets(weak_labels_df: pd.DataFrame, feature_name_filter: str | None) -> pd.DataFrame:
    """Build one target label per file and feature from weak-label counts."""
    if weak_labels_df.empty:
        return build_empty_dataframe(["file_id", "feature_name", "target_label", "target_support"])

    label_counts = (
        weak_labels_df.groupby(["file_id", "feature_name", "label"]).size().reset_index(name="label_count")
    )
    if feature_name_filter:
        label_counts = label_counts[label_counts["feature_name"] == feature_name_filter].copy()
    label_counts = label_counts.sort_values(
        by=["file_id", "feature_name", "label_count", "label"],
        ascending=[True, True, False, True],
    )
    return label_counts.drop_duplicates(subset=["file_id", "feature_name"]).rename(
        columns={"label": "target_label", "label_count": "target_support"}
    )[["file_id", "feature_name", "target_label", "target_support"]]


def restrict_documents_to_targets(
    documents_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    max_documents: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Restrict bounded runs to label-bearing documents, stratified by feature labels when possible."""
    if documents_df.empty or max_documents is None or targets_df.empty:
        return documents_df

    candidate_df = documents_df.merge(targets_df[["file_id", "feature_name", "target_label"]], on="file_id", how="inner")
    if candidate_df.empty:
        return documents_df.head(max_documents).copy()

    sampled_groups: list[pd.DataFrame] = []
    per_label_target = max(1, max_documents // max(1, candidate_df["target_label"].nunique()))
    for _, label_group in candidate_df.groupby("target_label"):
        sample_size = min(len(label_group), per_label_target)
        sampled_groups.append(label_group.sample(n=sample_size, random_state=random_state))

    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    sampled_df = sampled_df.sort_values(by=["document_quality", "character_count"], ascending=[True, False])
    sampled_df = sampled_df.drop_duplicates(subset=["file_id"]).head(max_documents)
    return documents_df[documents_df["file_id"].isin(sampled_df["file_id"])].copy()


def mean_pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply masked mean pooling to transformer token embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_documents(documents_df: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Encode report text with the configured biomedical transformer."""
    if documents_df.empty:
        return np.empty((0, 0), dtype=np.float32)

    model_name = config["transformer_model"]["model_name"]
    batch_size = int(config["transformer_model"]["batch_size"])
    max_length = int(config["transformer_model"]["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    texts = documents_df["normalized_text"].tolist()
    logger.info("Encoding %s documents with %s", len(texts), model_name)
    with torch.no_grad():
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index : start_index + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())
            logger.info("Completed transformer encoding for %s documents", len(texts))
    return np.vstack(embeddings)


def train_transformer_feature_models(
    documents_df: pd.DataFrame,
    training_documents_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    document_embeddings: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train logistic models over biomedical-transformer embeddings for each feature task."""
    if documents_df.empty or training_documents_df.empty or targets_df.empty or document_embeddings.size == 0:
        return documents_df[["file_id"]].copy() if "file_id" in documents_df.columns else build_empty_dataframe(["file_id"]), build_empty_dataframe(
            ["feature_name", "trained", "n_reports", "n_classes", "macro_f1", "majority_macro_f1", "labels"]
        )

    seed = int(config["project"]["seed"])
    embedding_lookup = {
        file_id: document_embeddings[index]
        for index, file_id in enumerate(documents_df["file_id"].tolist())
    }

    prediction_tables: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for feature_name, feature_targets in targets_df.groupby("feature_name"):
        logger.info("Training transformer classifier for feature '%s'", feature_name)
        task_df = training_documents_df[["file_id"]].merge(feature_targets, on="file_id", how="inner")
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

        task_df["embedding_index"] = task_df["file_id"].map(lambda file_id: file_id)
        feature_embeddings = np.vstack(task_df["file_id"].map(embedding_lookup).tolist())
        labels = task_df["target_label"].to_numpy()

        test_size = 0.2 if len(task_df) >= 50 else 0.25
        train_index, test_index = train_test_split(
            np.arange(len(task_df)),
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

        classifier = LogisticRegression(
            max_iter=1000,
            class_weight=config["baseline_models"]["tfidf_classifier"].get("class_weight"),
            random_state=seed,
        )
        classifier.fit(feature_embeddings[train_index], labels[train_index])
        test_predictions = classifier.predict(feature_embeddings[test_index])
        majority_label = pd.Series(labels[train_index]).mode().iloc[0]
        majority_predictions = np.array([majority_label] * len(test_index))
        macro_f1 = f1_score(labels[test_index], test_predictions, average="macro")
        majority_macro_f1 = f1_score(labels[test_index], majority_predictions, average="macro")

        full_probabilities = classifier.predict_proba(document_embeddings)
        classes = list(classifier.classes_)
        prediction_df = pd.DataFrame({"file_id": documents_df["file_id"]})
        prediction_df[f"transformer_{feature_name}_predicted_label"] = classifier.predict(document_embeddings)
        for class_index, class_name in enumerate(classes):
            prediction_df[f"transformer_{feature_name}_prob_{class_name}"] = full_probabilities[:, class_index]
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

    return merged_predictions, pd.DataFrame(metrics_rows)


def main() -> None:
    """Train biomedical-transformer pathology classifiers and export predictions."""
    args = parse_args()
    log_package_versions()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    normalized_df, weak_labels_df = load_pipeline_inputs(config)
    documents_df = prepare_documents(normalized_df, args.max_documents)
    targets_df = summarize_targets(weak_labels_df, args.feature_name)
    documents_df = restrict_documents_to_targets(documents_df, targets_df, args.max_documents, int(config["project"]["seed"]))
    training_documents_df = select_training_documents(documents_df, config)
    document_embeddings = encode_documents(documents_df, config)
    transformer_predictions_df, model_summary_df = train_transformer_feature_models(
        documents_df,
        training_documents_df,
        targets_df,
        document_embeddings,
        config,
    )

    output_path = PROJECT_ROOT / config["output"]["transformer_predictions"]
    summary_path = PROJECT_ROOT / "analysis" / "clinical_nlp" / "models" / "transformer_model_summary.csv"
    ensure_parent_dir(output_path)
    ensure_parent_dir(summary_path)
    write_parquet(transformer_predictions_df, output_path)
    write_csv(model_summary_df, summary_path)


if __name__ == "__main__":
    main()