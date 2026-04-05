"""Judge runtime compiler for rules-first serve-time compilation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbias.policy.compiler.protocol import CompilationResult, PolicyCompiler
from openbias.policy.compiler.registry import register_compiler
from openbias.policy.rules import POLICY_FILENAME


@register_compiler("judge")
class JudgeRuntimeCompiler(PolicyCompiler):
    """Compile canonical ``RULES.md`` text into Judge runtime config."""

    @property
    def engine_type(self) -> str:
        return "judge"

    async def compile(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> CompilationResult:
        del context
        compiled_rules = [line.strip() for line in rules_text.splitlines() if line.strip()]
        if not compiled_rules:
            return CompilationResult.failure(
                errors=["Judge runtime compilation produced no rules."],
            )
        return CompilationResult(
            success=True,
            config={
                "_compiled_rules": compiled_rules,
                "_rules_source": POLICY_FILENAME,
            },
        )

    def export(self, result: CompilationResult, output_path: Path) -> None:
        del result, output_path
        return None
