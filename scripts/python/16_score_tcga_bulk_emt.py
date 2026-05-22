#!/usr/bin/env python3
"""Download TCGA bulk RNA counts and compute case-level EMT scores."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
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

EMT_SIGNATURES: dict[str, list[str]] = {
    "project_humanized_emt": [
        "VIM",
        "CDH2",
        "SNAI1",
        "SNAI2",
        "TWIST1",
        "ZEB1",
        "ZEB2",
        "FN1",
        "COL1A1",
        "COL3A1",
        "MMP2",
        "MMP9",
        "MMP14",
        "TIMP1",
        "LOX",
        "LOXL2",
        "RAC1",
        "CDC42",
        "ITGB1",
        "RHOA",
        "ROCK1",
        "ROCK2",
        "YAP1",
        "WWTR1",
        "PIEZO1",
        "TRPV4",
        "LATS1",
        "LATS2",
        "TGFB1",
        "TGFB2",
        "TGFB3",
        "PDGFA",
        "VEGFA",
        "WNT5A",
        "IL6",
        "TNF",
    ],
    "hallmark_epithelial_mesenchymal_transition": [
        "VIM",
        "CDH2",
        "FN1",
        "COL1A1",
        "COL3A1",
        "MMP2",
        "MMP9",
        "ITGB1",
        "LOX",
        "LOXL2",
        "SNAI2",
        "TWIST1",
        "ZEB1",
        "ZEB2",
        "WNT5A",
        "TGFB1",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for EMT scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-missing", action="store_true", help="Download missing RNA count files before scoring")
    parser.add_argument("--max-downloads", type=int, default=None, help="Maximum number of RNA files to download in this run")
    return parser.parse_args()


def log_package_versions() -> None:
    """Log package versions used by the EMT scoring script."""
    logger.info("pandas==%s", pd.__version__)


def load_rna_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Load RNA gene-count rows from the pathology manifest."""
    manifest_path = PROJECT_ROOT / config["data_sources"]["manifest_dir"] / "gdc_files_manifest.tsv"
    if not manifest_path.exists():
        return build_empty_dataframe(["project_id", "case_submitter_id", "file_id", "file_name", "data_type", "workflow_type"])
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    return manifest_df[
        (manifest_df["data_type"] == "Gene Expression Quantification")
        & (manifest_df["workflow_type"] == "STAR - Counts")
    ].drop_duplicates(subset=["file_id"]).copy()


def download_rna_files(config: dict[str, Any], rna_df: pd.DataFrame, max_downloads: int | None) -> None:
    """Download missing TCGA RNA count files from the GDC data endpoint."""
    if rna_df.empty:
        logger.info("No RNA count files available for download")
        return

    download_dir = PROJECT_ROOT / config["data_sources"]["bulk_rna_download_dir"]
    download_dir.mkdir(parents=True, exist_ok=True)
    data_endpoint = f"{config['data_sources']['gdc_api_base']}{config['data_sources']['gdc_data_endpoint']}"

    remaining_df = rna_df.copy()
    remaining_df["local_path"] = remaining_df["file_name"].map(lambda name: download_dir / str(name))
    remaining_df = remaining_df[~remaining_df["local_path"].map(lambda path: path.exists() and path.stat().st_size > 0)]
    if max_downloads is not None:
        remaining_df = remaining_df.head(max_downloads)

    for _, row in remaining_df.iterrows():
        destination = Path(row["local_path"])
        request = Request(f"{data_endpoint}/{row['file_id']}", method="GET")
        try:
            with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
                handle.write(response.read())
            logger.info("Downloaded %s", destination.name)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.warning("Failed to download %s: %s", destination.name, exc)
            if destination.exists() and destination.stat().st_size == 0:
                destination.unlink()


def choose_expression_column(expression_df: pd.DataFrame, config: dict[str, Any]) -> str:
    """Choose the preferred expression column from the configured priority order."""
    for column_name in config["emt_scoring"]["expression_column_priority"]:
        if column_name in expression_df.columns:
            return str(column_name)
    raise ValueError("No configured expression column was found in the RNA count file")


def extract_signature_expression(file_path: Path, signature_genes: set[str], config: dict[str, Any]) -> dict[str, float]:
    """Extract signature-gene expression values from one RNA count file."""
    expression_df = pd.read_csv(file_path, sep="\t", comment="#")
    column_name = choose_expression_column(expression_df, config)
    filtered_df = expression_df[expression_df["gene_name"].fillna("").astype(str).str.startswith("N_") == False].copy()
    filtered_df["gene_name"] = filtered_df["gene_name"].fillna("").astype(str).str.upper()
    filtered_df = filtered_df[filtered_df["gene_name"].isin(signature_genes)].copy()
    if filtered_df.empty:
        return {}
    filtered_df[column_name] = pd.to_numeric(filtered_df[column_name], errors="coerce").fillna(0.0)
    return filtered_df.groupby("gene_name")[column_name].max().to_dict()


def build_expression_matrix(rna_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build a sample-by-signature-gene matrix from locally available RNA count files."""
    if rna_df.empty:
        return build_empty_dataframe(["project_id", "case_submitter_id", "file_id", "file_name"])

    download_dir = PROJECT_ROOT / config["data_sources"]["bulk_rna_download_dir"]
    signature_genes = {gene for genes in EMT_SIGNATURES.values() for gene in genes}
    rows: list[dict[str, Any]] = []

    for _, row in rna_df.iterrows():
        file_path = download_dir / str(row["file_name"])
        if not file_path.exists() or file_path.stat().st_size == 0:
            continue
        expression_values = extract_signature_expression(file_path, signature_genes, config)
        rows.append(
            {
                "project_id": row["project_id"],
                "case_submitter_id": row["case_submitter_id"],
                "file_id": row["file_id"],
                "file_name": row["file_name"],
                **expression_values,
            }
        )

    if not rows:
        return build_empty_dataframe(["project_id", "case_submitter_id", "file_id", "file_name"])
    return pd.DataFrame(rows).fillna(0.0)


def compute_signature_scores(expression_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute EMT signature scores and quantile-based case labels."""
    if expression_df.empty:
        return build_empty_dataframe(
            [
                "project_id",
                "case_submitter_id",
                "file_ids",
                "n_files_aggregated",
                "project_humanized_emt_score",
                "hallmark_epithelial_mesenchymal_transition_score",
                "emt_score",
                "emt_label",
            ]
        )

    metadata_columns = ["project_id", "case_submitter_id", "file_id", "file_name"]
    gene_columns = [column for column in expression_df.columns if column not in metadata_columns]
    log_expression = expression_df[gene_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    log_expression = np.log1p(log_expression)

    zscore_expression = log_expression.copy()
    for column in gene_columns:
        std_value = float(zscore_expression[column].std(ddof=0))
        mean_value = float(zscore_expression[column].mean())
        if std_value > 0:
            zscore_expression[column] = (zscore_expression[column] - mean_value) / std_value
        else:
            zscore_expression[column] = 0.0

    scores_df = expression_df[metadata_columns].copy()
    for signature_name, genes in EMT_SIGNATURES.items():
        available_genes = [gene for gene in genes if gene in zscore_expression.columns]
        if not available_genes:
            scores_df[f"{signature_name}_score"] = pd.NA
            continue
        scores_df[f"{signature_name}_score"] = zscore_expression[available_genes].mean(axis=1)
        scores_df[f"{signature_name}_genes_detected"] = (log_expression[available_genes] > 0).sum(axis=1)

    score_columns = [column for column in scores_df.columns if column.endswith("_score")]
    scores_df["emt_score"] = scores_df[score_columns].mean(axis=1)

    case_level = (
        scores_df.groupby(["project_id", "case_submitter_id"], as_index=False)
        .agg(
            file_ids=("file_id", lambda values: "|".join(sorted(map(str, values)))),
            n_files_aggregated=("file_id", "nunique"),
            project_humanized_emt_score=("project_humanized_emt_score", "mean"),
            hallmark_epithelial_mesenchymal_transition_score=(
                "hallmark_epithelial_mesenchymal_transition_score",
                "mean",
            ),
            emt_score=("emt_score", "mean"),
        )
    )

    low_quantile = float(config["emt_scoring"]["low_quantile"])
    high_quantile = float(config["emt_scoring"]["high_quantile"])
    labeled_groups: list[pd.DataFrame] = []
    for project_id, project_df in case_level.groupby("project_id"):
        project_copy = project_df.copy()
        low_cutoff = project_copy["emt_score"].quantile(low_quantile)
        high_cutoff = project_copy["emt_score"].quantile(high_quantile)
        project_copy["emt_label"] = "mid"
        project_copy.loc[project_copy["emt_score"] <= low_cutoff, "emt_label"] = "low"
        project_copy.loc[project_copy["emt_score"] >= high_cutoff, "emt_label"] = "high"
        labeled_groups.append(project_copy)
        logger.info(
            "Scored %s cases for %s with EMT cutoffs low=%.4f high=%.4f",
            len(project_copy),
            project_id,
            low_cutoff,
            high_cutoff,
        )

    return pd.concat(labeled_groups, ignore_index=True) if labeled_groups else case_level


def main() -> None:
    """Download RNA counts if requested and compute TCGA bulk EMT scores."""
    args = parse_args()
    log_package_versions()
    config, _, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    rna_df = load_rna_manifest(config)
    if args.download_missing:
        download_rna_files(config, rna_df, args.max_downloads)

    expression_df = build_expression_matrix(rna_df, config)
    scores_df = compute_signature_scores(expression_df, config)

    output_path = PROJECT_ROOT / config["output"]["emt_scores"]
    ensure_parent_dir(output_path)
    write_csv(scores_df, output_path)


if __name__ == "__main__":
    main()