#!/usr/bin/env python3
"""Prepare annotation candidates and weak labels for pathology report review."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from utils.clinical_nlp_config import load_and_validate_configs
from utils.clinical_nlp_io import build_empty_dataframe, write_csv, write_parquet
from utils.clinical_nlp_text import sentence_spans


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_config.yaml"
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_ontology.yaml"


def sample_reports_for_review(normalized_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select a review candidate subset from normalized reports."""
    target = min(len(normalized_df), config["annotation_strategy"]["gold_set_target_reports"])
    if target == 0:
        return normalized_df.head(0).copy()
    unique_reports = normalized_df.drop_duplicates(subset=["file_id"]).copy()
    quality_rank = {"high": 0, "low": 1, "empty": 2}
    unique_reports["quality_rank"] = unique_reports["document_quality"].map(quality_rank).fillna(3)
    ranked = unique_reports.sort_values(
        by=["quality_rank", "section_count", "alpha_ratio", "case_submitter_id"],
        ascending=[True, False, False, True],
    )
    return ranked.head(target).drop(columns=["quality_rank"])


def generate_weak_labels(normalized_df: pd.DataFrame, ontology_config: dict[str, Any]) -> pd.DataFrame:
    """Generate weak labels using ontology synonyms over normalized text."""
    rows: list[dict[str, Any]] = []
    for _, report in normalized_df.iterrows():
        text = str(report.get("normalized_text") or "").lower()
        for feature_name, feature in ontology_config["feature_schema"].items():
            for label, synonyms in feature.get("synonyms", {}).items():
                matched = [synonym for synonym in synonyms if synonym.lower() in text]
                if matched:
                    rows.append(
                        {
                            "case_submitter_id": report.get("case_submitter_id"),
                            "file_id": report.get("file_id"),
                            "feature_name": feature_name,
                            "label": label,
                            "matched_phrase": matched[0],
                            "confidence": 0.75,
                        }
                    )
    return pd.DataFrame(rows, columns=["case_submitter_id", "file_id", "feature_name", "label", "matched_phrase", "confidence"])


def build_annotation_queue(review_df: pd.DataFrame, weak_labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build a review queue with report-level context and weak-label summaries."""
    if review_df.empty:
        return build_empty_dataframe(
            [
                "case_submitter_id",
                "file_id",
                "document_quality",
                "candidate_feature_count",
                "preview_text",
            ]
        )
    counts = weak_labels_df.groupby("file_id").size().reset_index(name="candidate_feature_count") if not weak_labels_df.empty else pd.DataFrame(columns=["file_id", "candidate_feature_count"])
    queue = review_df[["case_submitter_id", "file_id", "document_quality", "normalized_text"]].copy()
    queue = queue.merge(counts, on="file_id", how="left")
    queue["candidate_feature_count"] = pd.to_numeric(queue["candidate_feature_count"], errors="coerce").fillna(0).astype(int)
    queue["preview_text"] = queue["normalized_text"].apply(lambda text: " ".join(sentence_spans(str(text))[:2])[:300])
    return queue.drop(columns=["normalized_text"])


def merge_existing_manual_labels(review_df: pd.DataFrame) -> pd.DataFrame:
    """Return review candidates with manual-label placeholder columns."""
    merged = review_df.copy()
    if merged.empty:
        return merged
    merged["review_status"] = "pending"
    merged["reviewer"] = ""
    merged["adjudication_notes"] = ""
    return merged


def main() -> None:
    """Create review candidates, weak labels, and annotation queue outputs."""
    config, ontology_config, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    normalized_path = PROJECT_ROOT / config["output"]["normalized_reports"]
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

    review_candidates = sample_reports_for_review(normalized_df, config)
    weak_labels = generate_weak_labels(normalized_df, ontology_config)
    review_queue = build_annotation_queue(review_candidates, weak_labels)
    gold_set_candidates = merge_existing_manual_labels(review_candidates)

    write_csv(gold_set_candidates, PROJECT_ROOT / "analysis" / "clinical_nlp" / "annotation" / "gold_set_candidates.csv")
    write_parquet(weak_labels, PROJECT_ROOT / config["output"]["weak_labels"])
    write_csv(review_queue, PROJECT_ROOT / "analysis" / "clinical_nlp" / "annotation" / "review_queue.csv")


if __name__ == "__main__":
    main()