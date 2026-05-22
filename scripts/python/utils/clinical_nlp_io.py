"""I/O helpers for the clinical NLP EMT pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must parse to a mapping")
    return data


def resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a workspace-relative path from the project root."""
    return PROJECT_ROOT / Path(relative_path)


def ensure_parent_dir(path: Path) -> None:
    """Create a file parent directory if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write a dataframe to CSV after ensuring the parent directory exists."""
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_path)


def write_json(data: Any, output_path: Path) -> None:
    """Write JSON data with stable formatting."""
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    logger.info("Wrote JSON to %s", output_path)


def write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write a dataframe to parquet, requiring an available parquet engine."""
    ensure_parent_dir(output_path)
    try:
        df.to_parquet(output_path, index=False)
    except Exception as exc:  # pragma: no cover - engine availability varies
        raise RuntimeError(
            "Parquet output requires an available parquet engine such as pyarrow"
        ) from exc
    logger.info("Wrote %s rows to %s", len(df), output_path)


def build_empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Build an empty dataframe with a defined column order."""
    return pd.DataFrame(columns=columns)
