"""
Protocol definitions for rules compilers.

Compilers convert normalized rules text into
engine-specific configurations (e.g., FSM workflow YAML, NeMo Colang).

This mirrors the PolicyEngine pattern, enabling pluggable compilers
for different policy engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class CompilationResult:
    """
    Result of compiling normalized rules text to engine config.

    Attributes:
        success: Whether compilation succeeded
        config: Engine-specific config (WorkflowDefinition, dict, etc.)
        warnings: Non-fatal issues encountered during compilation
        errors: Fatal issues that prevented compilation
        metadata: Additional info (LLM tokens used, model, etc.)
    """

    success: bool
    config: Any
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, errors: list[str], warnings: list[str] | None = None) -> "CompilationResult":
        """Create a failed compilation result."""
        return cls(
            success=False,
            config=None,
            errors=errors,
            warnings=warnings or [],
        )

class PolicyCompiler(ABC):
    """
    Base class for rules compilers.

    Each engine can have its own compiler that converts normalized rules text
    to engine-specific configuration.

    Mirrors PolicyEngine pattern - register with @register_compiler decorator.

    Example:
        ```python
        @register_compiler("fsm")
        class FSMCompiler(PolicyCompiler):
            @property
            def engine_type(self) -> str:
                return "fsm"

            async def compile(
                self,
                natural_language: str,
                context: dict[str, Any] | None = None,
            ) -> CompilationResult:
                # Use LLM to parse NL -> WorkflowDefinition
                ...

            def export(self, result: CompilationResult, output_path: Path) -> None:
                # Write YAML file
                ...
        ```
    """

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """
        Engine type this compiler produces config for.

        Returns:
            Engine type identifier (e.g., 'fsm', 'nemo')
        """
        ...

    @abstractmethod
    async def compile(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> CompilationResult:
        """
        Compile normalized rules text to engine-specific config.

        Args:
            rules_text: Normalized rules payload for this evaluator
            context: Optional hints for compilation:
                - domain: Application domain (e.g., "customer support")
                - existing_states: List of known state names
                - examples: Example conversations
                - tool_names: Available tool/function names

        Returns:
            CompilationResult with engine-specific config
        """
        ...

    @abstractmethod
    def export(self, result: CompilationResult, output_path: Path) -> None:
        """
        Export compiled config to file(s).

        Args:
            result: Successful compilation result
            output_path: Where to write (file or directory depending on engine)

        Raises:
            ValueError: If result was not successful
        """
        ...

    def validate_result(self, result: CompilationResult) -> list[str]:
        """
        Validate a compilation result.

        Override in subclasses to add engine-specific validation.

        Args:
            result: Compilation result to validate

        Returns:
            List of validation errors (empty if valid)
        """
        if not result.success:
            return ["Compilation was not successful"]
        if result.config is None:
            return ["No config generated"]
        return []
