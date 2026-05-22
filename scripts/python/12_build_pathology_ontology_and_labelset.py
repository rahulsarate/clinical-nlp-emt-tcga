#!/usr/bin/env python3
"""Export pathology ontology config into machine-readable schema artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from utils.clinical_nlp_config import load_and_validate_configs
from utils.clinical_nlp_io import write_csv, write_json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_config.yaml"
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "clinical_nlp_ontology.yaml"


def flatten_feature_schema(ontology_config: dict[str, Any]) -> pd.DataFrame:
    """Flatten the ontology feature schema into a tabular export."""
    rows: list[dict[str, Any]] = []
    for feature_name, feature in ontology_config["feature_schema"].items():
        for label in feature["allowed_labels"]:
            rows.append(
                {
                    "feature_name": feature_name,
                    "task": feature["task"],
                    "label": label,
                    "evidence_sections": "|".join(feature["evidence_sections"]),
                    "synonyms": "|".join(feature.get("synonyms", {}).get(label, [])),
                }
            )
    return pd.DataFrame(rows)


def build_regex_seed_patterns(ontology_config: dict[str, Any]) -> dict[str, Any]:
    """Build regex seed patterns from ontology synonym lists."""
    regex_patterns: dict[str, Any] = {}
    for feature_name, feature in ontology_config["feature_schema"].items():
        regex_patterns[feature_name] = {}
        for label, synonyms in feature.get("synonyms", {}).items():
            regex_patterns[feature_name][label] = [rf"\b{synonym}\b" for synonym in synonyms]
    return regex_patterns


def export_label_definitions(ontology_config: dict[str, Any]) -> dict[str, Any]:
    """Build a compact label-definition export for downstream scripts."""
    return {
        "metadata": ontology_config["metadata"],
        "sections": ontology_config["sections"],
        "feature_schema": ontology_config["feature_schema"],
        "global_rules": ontology_config["global_rules"],
    }


def main() -> None:
    """Export ontology configuration into schema, regex, and label definition files."""
    _, ontology_config, warnings = load_and_validate_configs(CONFIG_PATH, ONTOLOGY_PATH)
    for warning in warnings:
        logger.warning(warning)

    schema_df = flatten_feature_schema(ontology_config)
    regex_patterns = build_regex_seed_patterns(ontology_config)
    label_definitions = export_label_definitions(ontology_config)

    schema_output = PROJECT_ROOT / "analysis" / "clinical_nlp" / "annotation" / "pathology_feature_schema.csv"
    regex_output = PROJECT_ROOT / "analysis" / "clinical_nlp" / "annotation" / "regex_seed_patterns.yaml"
    definitions_output = PROJECT_ROOT / "analysis" / "clinical_nlp" / "annotation" / "label_definitions.json"

    write_csv(schema_df, schema_output)
    regex_output.parent.mkdir(parents=True, exist_ok=True)
    regex_output.write_text(json.dumps(regex_patterns, indent=2), encoding="utf-8")
    logger.info("Wrote regex seed patterns to %s", regex_output)
    write_json(label_definitions, definitions_output)


if __name__ == "__main__":
    main()
