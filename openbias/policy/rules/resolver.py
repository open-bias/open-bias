"""Resolve user-authored rules from config and optional files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$")


def _segment_freeform_text(raw_text: str) -> list[str]:
    """Segment markdown/plaintext into normalized rule strings."""
    text = raw_text.replace("\r\n", "\n").strip()
    if not text:
        return []

    segments: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        joined = " ".join(part.strip() for part in paragraph if part.strip()).strip()
        paragraph.clear()
        if not joined:
            return
        if ";" in joined:
            parts = [piece.strip() for piece in joined.split(";") if piece.strip()]
            segments.extend(parts)
            return
        segments.append(joined)

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            heading = heading_match.group(1).strip()
            if heading:
                segments.append(heading)
            continue

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            flush_paragraph()
            bullet = bullet_match.group(1).strip()
            if bullet:
                segments.append(bullet)
            continue

        paragraph.append(stripped)

    flush_paragraph()
    return [item for item in segments if item]


def _normalize_inline_rules(value: Any) -> list[str]:
    """Normalize `rules` config value into rule strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return _segment_freeform_text(value)
    if isinstance(value, list):
        rules: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Each `rules` entry must be a string.")
            rules.extend(_segment_freeform_text(item))
        return rules
    raise ValueError("`rules` must be a string or list of strings.")


def _read_rules_file(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("`rules_file` must point to a .md or .txt file.")
    if not path.exists():
        raise ValueError(f"Rules file not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_rules_payload(
    evaluator_config: dict[str, Any],
    *,
    base_dir: Path | None = None,
    auto_discover_rules_md: bool = True,
) -> list[str]:
    """Resolve normalized rules from inline and file-based sources."""
    root = base_dir or Path.cwd()
    resolved: list[str] = []

    resolved.extend(_normalize_inline_rules(evaluator_config.get("rules")))

    rules_file = evaluator_config.get("rules_file")
    if rules_file is not None:
        if not isinstance(rules_file, str):
            raise ValueError("`rules_file` must be a string path.")
        resolved.extend(_segment_freeform_text(_read_rules_file(rules_file, root)))
    elif auto_discover_rules_md:
        autodiscovered = root / "rules.md"
        if autodiscovered.exists():
            resolved.extend(_segment_freeform_text(autodiscovered.read_text(encoding="utf-8")))

    seen: set[str] = set()
    unique_rules: list[str] = []
    for rule in resolved:
        normalized = rule.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_rules.append(normalized)

    return unique_rules
