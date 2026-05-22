#!/usr/bin/env python3
"""Download and inventory TCGA pathology-report and bulk RNA metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

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

FILE_COLUMNS = [
    "project_id",
    "case_submitter_id",
    "case_id",
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "data_format",
    "workflow_type",
]

CASE_COLUMNS = [
    "project_id",
    "case_submitter_id",
    "case_id",
    "primary_site",
    "disease_type",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the metadata downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Skip remote calls and write empty scaffold outputs")
    parser.add_argument(
        "--download-pathology-files",
        action="store_true",
        help="Download pathology report files after writing the manifest",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Maximum number of pathology report files to download",
    )
    return parser.parse_args()


def build_gdc_filters(projects: list[str], data_category: str, data_type: str | None = None) -> dict[str, Any]:
    """Build a simple GDC filter object for project-scoped queries."""
    filters: dict[str, Any] = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
            {"op": "=", "content": {"field": "data_category", "value": data_category}},
        ],
    }
    if data_type:
        filters["content"].append({"op": "=", "content": {"field": "data_type", "value": data_type}})
    return filters


def query_gdc(endpoint_url: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute a POST request against the GDC API and return hit dictionaries."""
    encoded = json.dumps(payload).encode("utf-8")
    request = Request(endpoint_url, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=60) as response:
        response_json = json.loads(response.read().decode("utf-8"))
    return response_json.get("data", {}).get("hits", [])


def query_gdc_files(config: dict[str, Any], data_category: str, data_type: str | None = None) -> pd.DataFrame:
    """Query GDC files endpoint and return normalized file metadata."""
    data_sources = config["data_sources"]
    endpoint_url = f"{data_sources['gdc_api_base']}{data_sources['gdc_files_endpoint']}"
    payload = {
        "filters": build_gdc_filters(data_sources["projects"], data_category, data_type=data_type),
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "data_category",
                "data_type",
                "data_format",
                "analysis.workflow_type",
                "cases.case_id",
                "cases.submitter_id",
                "cases.project.project_id",
            ]
        ),
        "format": "JSON",
        "size": 2000,
    }
    hits = query_gdc(endpoint_url, payload)
    rows: list[dict[str, Any]] = []
    for hit in hits:
        cases = hit.get("cases", []) or [{}]
        for case in cases:
            rows.append(
                {
                    "project_id": case.get("project", {}).get("project_id"),
                    "case_submitter_id": case.get("submitter_id"),
                    "case_id": case.get("case_id"),
                    "file_id": hit.get("file_id"),
                    "file_name": hit.get("file_name"),
                    "data_category": hit.get("data_category"),
                    "data_type": hit.get("data_type"),
                    "data_format": hit.get("data_format"),
                    "workflow_type": hit.get("analysis", {}).get("workflow_type"),
                }
            )
    return pd.DataFrame(rows, columns=FILE_COLUMNS)


def query_gdc_cases(config: dict[str, Any]) -> pd.DataFrame:
    """Query GDC cases endpoint and return normalized case metadata."""
    data_sources = config["data_sources"]
    endpoint_url = f"{data_sources['gdc_api_base']}{data_sources['gdc_cases_endpoint']}"
    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "project.project_id", "value": data_sources["projects"]},
        },
        "fields": ",".join(["submitter_id", "case_id", "project.project_id", "primary_site", "disease_type"]),
        "format": "JSON",
        "size": 2000,
    }
    hits = query_gdc(endpoint_url, payload)
    rows = [
        {
            "project_id": hit.get("project", {}).get("project_id"),
            "case_submitter_id": hit.get("submitter_id"),
            "case_id": hit.get("case_id"),
            "primary_site": hit.get("primary_site"),
            "disease_type": hit.get("disease_type"),
        }
        for hit in hits
    ]
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def build_case_inventory(files_df: pd.DataFrame, cases_df: pd.DataFrame) -> pd.DataFrame:
    """Build a case-level inventory summarizing available pathology and RNA metadata."""
    inventory = cases_df.copy()
    if inventory.empty:
        inventory = build_empty_dataframe(
            CASE_COLUMNS + ["n_clinical_files", "n_rna_files", "has_pathology_candidate", "has_bulk_rna_candidate"]
        )
        return inventory

    grouped = files_df.groupby(["case_submitter_id", "data_category"], dropna=False).size().reset_index(name="n_files")
    clinical_counts = grouped[grouped["data_category"] == "Clinical"][["case_submitter_id", "n_files"]]
    rna_counts = grouped[grouped["data_category"] == "Transcriptome Profiling"][["case_submitter_id", "n_files"]]
    clinical_counts = clinical_counts.rename(columns={"n_files": "n_clinical_files"})
    rna_counts = rna_counts.rename(columns={"n_files": "n_rna_files"})
    inventory = inventory.merge(clinical_counts, on="case_submitter_id", how="left")
    inventory = inventory.merge(rna_counts, on="case_submitter_id", how="left")
    inventory["n_clinical_files"] = inventory["n_clinical_files"].fillna(0).astype(int)
    inventory["n_rna_files"] = inventory["n_rna_files"].fillna(0).astype(int)
    inventory["has_pathology_candidate"] = inventory["n_clinical_files"] > 0
    inventory["has_bulk_rna_candidate"] = inventory["n_rna_files"] > 0
    return inventory


def save_manifest_tables(config: dict[str, Any], files_df: pd.DataFrame, inventory_df: pd.DataFrame) -> None:
    """Persist file manifest and case inventory outputs."""
    manifest_path = PROJECT_ROOT / config["data_sources"]["manifest_dir"] / "gdc_files_manifest.tsv"
    inventory_path = PROJECT_ROOT / config["data_sources"]["cohort_dir"] / "tcga_case_inventory.csv"
    download_inventory_path = PROJECT_ROOT / "analysis" / "clinical_nlp" / "source_text" / "download_inventory.csv"
    ensure_parent_dir(manifest_path)
    files_df.to_csv(manifest_path, index=False, sep="\t")
    logger.info("Wrote %s rows to %s", len(files_df), manifest_path)
    write_csv(inventory_df, inventory_path)
    download_inventory = files_df[["project_id", "case_submitter_id", "file_id", "file_name", "data_format"]].copy()
    write_csv(download_inventory, download_inventory_path)


def download_pathology_files(config: dict[str, Any], pathology_df: pd.DataFrame, max_downloads: int | None) -> None:
    """Download pathology-report source files from the GDC data endpoint."""
    if pathology_df.empty:
        logger.info("No pathology report rows available for download")
        return

    data_endpoint = f"{config['data_sources']['gdc_api_base']}{config['data_sources']['gdc_data_endpoint']}"
    report_dir = PROJECT_ROOT / config["data_sources"]["report_download_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    download_limit = max_downloads or config["data_sources"].get("max_pathology_report_downloads", len(pathology_df))
    unique_reports = pathology_df.drop_duplicates(subset=["file_id"]).head(download_limit)
    for _, row in unique_reports.iterrows():
        file_id = str(row["file_id"])
        file_name = str(row["file_name"])
        destination = report_dir / file_name
        if destination.exists() and destination.stat().st_size > 0:
            continue
        request = Request(f"{data_endpoint}/{file_id}", method="GET")
        try:
            with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
                handle.write(response.read())
            logger.info("Downloaded %s", destination.name)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.warning("Failed to download %s: %s", file_name, exc)
            if destination.exists() and destination.stat().st_size == 0:
                destination.unlink()


def main() -> None:
    """Run metadata retrieval and write the initial cohort inventory."""
    args = parse_args()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    files_df = build_empty_dataframe(FILE_COLUMNS)
    cases_df = build_empty_dataframe(CASE_COLUMNS)
    if not args.dry_run:
        try:
            clinical_df = query_gdc_files(config, data_category="Clinical", data_type="Pathology Report")
            rna_df = query_gdc_files(
                config,
                data_category="Transcriptome Profiling",
                data_type="Gene Expression Quantification",
            )
            files_df = pd.concat([clinical_df, rna_df], ignore_index=True)
            cases_df = query_gdc_cases(config)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            logger.error("GDC query failed, writing empty scaffold outputs instead: %s", exc)

    inventory_df = build_case_inventory(files_df, cases_df)
    save_manifest_tables(config, files_df, inventory_df)
    if args.download_pathology_files or config["data_sources"].get("download_pathology_reports", False):
        pathology_df = files_df[files_df["data_type"] == "Pathology Report"].copy()
        download_pathology_files(config, pathology_df, args.max_downloads)


if __name__ == "__main__":
    main()
