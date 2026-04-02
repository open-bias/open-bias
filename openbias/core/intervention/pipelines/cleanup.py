"""Response cleanup stage for stripping internal intervention scaffolding."""

from __future__ import annotations

import re


class ResponseCleanupStage:
    """Remove hidden repair scaffolding from assistant-visible text."""

    def clean_text(self, text: str, cleanup_rules: list[str] | None) -> str:
        if not text:
            return text

        cleaned = text
        rules = cleanup_rules or []

        # Remove explicit wrapper blocks regardless of exact inner formatting.
        if "[REPAIR-INSTRUCTION]" in rules:
            cleaned = re.sub(
                r"\[REPAIR-INSTRUCTION\].*?\[END-REPAIR-INSTRUCTION\]",
                "",
                cleaned,
                flags=re.DOTALL,
            )
            cleaned = re.sub(
                r"\[REPAIR-INSTRUCTION\].*?(?:\n|$)",
                "",
                cleaned,
                flags=re.DOTALL,
            )

        # Strip single-line scaffolding markers that should never be user-visible.
        for marker in rules:
            if marker == "[REPAIR-INSTRUCTION]":
                continue
            cleaned = re.sub(
                rf"^\s*{re.escape(marker)}.*(?:\n|$)",
                "",
                cleaned,
                flags=re.MULTILINE,
            )

        # Normalize whitespace after removing scaffolding lines.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

