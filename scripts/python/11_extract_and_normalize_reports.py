#!/usr/bin/env python3
"""Extract and normalize pathology report text into a structured table."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
from pypdf import PdfReader

from utils.clinical_nlp_config import load_and_validate_configs
from utils.clinical_nlp_io import build_empty_dataframe, ensure_parent_dir, write_csv, write_parquet
from utils.clinical_nlp_text import compute_alpha_ratio, estimate_ocr_noise_ratio, normalize_text, section_report_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_config.yaml"
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_ontology.yaml"

RAW_TEXT_COLUMNS = [
    "case_submitter_id",
    "file_id",
    "file_name",
    "source_path",
    "document_type",
    "extraction_mode",
    "raw_text",
]

NORMALIZED_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for report normalization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write empty scaffold outputs without reading local files")
    return parser.parse_args()


def detect_document_type(file_name: str) -> str:
    """Infer a document type from the file extension."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    if suffix == ".xml":
        return "xml"
    return "unknown"


def extract_native_text(source_path: Path) -> str:
    """Extract text from plain-text inputs using a lightweight fallback path."""
    suffix = source_path.suffix.lower()
    if not source_path.exists():
        return ""
    if suffix == ".txt":
        return source_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".xml":
        try:
            root = ET.fromstring(source_path.read_text(encoding="utf-8", errors="ignore"))
            return "\n".join(text.strip() for text in root.itertext() if text and text.strip())
        except ET.ParseError:
            return source_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(source_path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""
    return ""


def extract_ocr_text(_source_path: Path) -> str:
    """Return OCR text placeholder until OCR tooling is installed."""
    return ""


def route_extraction_mode(document_type: str, config: dict[str, Any]) -> str:
    """Route a document to the preferred extraction mode based on type and config."""
    if document_type in {"txt", "xml"}:
        return "native_text"
    if document_type == "pdf" and config["document_processing"]["ocr_enabled"]:
        return "ocr"
    return "regex_fallback"


def score_document_quality(text: str, section_count: int, config: dict[str, Any]) -> str:
    """Assign a coarse document-quality label from normalized text metrics."""
    thresholds = config["document_processing"]["document_quality_thresholds"]
    alpha_ratio = compute_alpha_ratio(text)
    noise_ratio = estimate_ocr_noise_ratio(text)
    if alpha_ratio >= thresholds["min_alpha_ratio"] and section_count >= thresholds["min_nonempty_sections"] and noise_ratio <= thresholds["max_ocr_noise_ratio"]:
        return "high"
    if text:
        return "low"
    return "empty"


def build_raw_text_table(manifest_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build the raw extracted text table from manifest rows."""
    rows: list[dict[str, Any]] = []
    report_dir = PROJECT_ROOT / config["data_sources"].get("report_download_dir", config["data_sources"]["manifest_dir"])
    for _, row in manifest_df.iterrows():
        file_name = str(row.get("file_name") or "")
        source_path = report_dir / file_name
        document_type = detect_document_type(file_name)
        raw_text = extract_native_text(source_path)
        if raw_text:
            extraction_mode = "native_text"
        else:
            extraction_mode = route_extraction_mode(document_type, config)
        if extraction_mode == "ocr" and not raw_text:
            raw_text = extract_ocr_text(source_path)
        rows.append(
            {
                "case_submitter_id": row.get("case_submitter_id"),
                "file_id": row.get("file_id"),
                "file_name": file_name,
                "source_path": str(source_path.relative_to(PROJECT_ROOT)),
                "document_type": document_type,
                "extraction_mode": extraction_mode,
                "raw_text": raw_text,
            }
        )
    return pd.DataFrame(rows, columns=RAW_TEXT_COLUMNS)


def build_normalized_table(raw_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create normalized report rows and a document-quality summary table."""
    normalized_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    section_headers = config["document_processing"]["section_headers"]
    for _, row in raw_df.iterrows():
        normalized = normalize_text(str(row.get("raw_text") or ""), normalize_unicode=config["document_processing"]["normalize_unicode"])
        sections = section_report_text(normalized, section_headers)
        alpha_ratio = compute_alpha_ratio(normalized)
        noise_ratio = estimate_ocr_noise_ratio(normalized)
        quality = score_document_quality(normalized, len(sections), config)
        normalized_rows.append(
            {
                "case_submitter_id": row.get("case_submitter_id"),
                "file_id": row.get("file_id"),
                "file_name": row.get("file_name"),
                "document_type": row.get("document_type"),
                "extraction_mode": row.get("extraction_mode"),
                "normalized_text": normalized,
                "section_count": len(sections),
                "alpha_ratio": alpha_ratio,
                "ocr_noise_ratio": noise_ratio,
                "document_quality": quality,
            }
        )
        quality_rows.append(
            {
                "case_submitter_id": row.get("case_submitter_id"),
                "file_id": row.get("file_id"),
                "file_name": row.get("file_name"),
                "document_quality": quality,
                "section_count": len(sections),
                "alpha_ratio": alpha_ratio,
                "ocr_noise_ratio": noise_ratio,
            }
        )
    return pd.DataFrame(normalized_rows, columns=NORMALIZED_COLUMNS), pd.DataFrame(quality_rows)


def main() -> None:
    """Normalize report text and write scaffold outputs for downstream steps."""
    args = parse_args()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    manifest_path = PROJECT_ROOT / config["data_sources"]["manifest_dir"] / "gdc_files_manifest.tsv"
    if manifest_path.exists() and not args.dry_run:
        manifest_df = pd.read_csv(manifest_path, sep="\t")
        manifest_df = manifest_df[manifest_df["data_type"] == "Pathology Report"].drop_duplicates(subset=["file_id"]).copy()
    else:
        manifest_df = build_empty_dataframe(["case_submitter_id", "file_id", "file_name"])

    raw_df = build_raw_text_table(manifest_df, config) if not manifest_df.empty else build_empty_dataframe(RAW_TEXT_COLUMNS)
    normalized_df, quality_df = build_normalized_table(raw_df, config) if not raw_df.empty else (
        build_empty_dataframe(NORMALIZED_COLUMNS),
        build_empty_dataframe(["case_submitter_id", "file_id", "file_name", "document_quality", "section_count", "alpha_ratio", "ocr_noise_ratio"]),
    )

    raw_output = PROJECT_ROOT / "analysis" / "clinical_nlp" / "source_text" / "raw_report_text.parquet"
    normalized_output = PROJECT_ROOT / config["output"]["normalized_reports"]
    quality_output = PROJECT_ROOT / "analysis" / "clinical_nlp" / "normalized" / "document_quality_summary.csv"
    ensure_parent_dir(raw_output)
    ensure_parent_dir(normalized_output)
    ensure_parent_dir(quality_output)
    write_parquet(raw_df, raw_output)
    write_parquet(normalized_df, normalized_output)
    write_csv(quality_df, quality_output)


if __name__ == "__main__":
    main()
