"""Text normalization helpers for the clinical NLP EMT pipeline."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(raw_text: str, normalize_unicode: bool = True) -> str:
    """Normalize whitespace and common artifacts in a pathology report."""
    text = raw_text or ""
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def section_report_text(text: str, section_headers: list[str]) -> list[dict[str, Any]]:
    """Split a report into heuristic sections using configured section headers."""
    if not text:
        return []

    pattern = "|".join(re.escape(header) for header in sorted(section_headers, key=len, reverse=True))
    if not pattern:
        return [{"section": "full_text", "text": text, "priority": None}]

    matches = list(re.finditer(rf"(?im)^\s*({pattern})\s*:?\s*$", text))
    if not matches:
        return [{"section": "full_text", "text": text, "priority": None}]

    sections: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"section": match.group(1).lower(), "text": content, "priority": None})
    return sections


def compute_alpha_ratio(text: str) -> float:
    """Compute the alphabetic character ratio for a report body."""
    if not text:
        return 0.0
    alpha_chars = sum(1 for char in text if char.isalpha())
    return alpha_chars / max(len(text), 1)


def estimate_ocr_noise_ratio(text: str) -> float:
    """Estimate OCR noise using non-word artifact density."""
    if not text:
        return 1.0
    noisy_tokens = re.findall(r"[^A-Za-z0-9\s.,;:()\-]", text)
    return len(noisy_tokens) / max(len(text), 1)


def sentence_spans(text: str) -> list[str]:
    """Split text into approximate sentence spans."""
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
