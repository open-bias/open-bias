"""
Policy compiler system for natural language to engine config conversion.

This module provides infrastructure for compiling natural language policy
descriptions into engine-specific configurations:

- FSM: Natural language → workflow.yaml (WorkflowDefinition)
- NeMo: Natural language → Colang + config
- Judge: Natural language → rubric.yaml

Compilation happens automatically during ``openbias serve`` startup.
Rules defined in ``openbias.yaml`` (inline ``rules`` or ``rules_file``)
are resolved and compiled into engine-native config before engine init.

Programmatic usage:
    ```python
    from openbias.policy.compiler import PolicyCompilerRegistry

    # Get FSM compiler class and instantiate
    compiler_class = PolicyCompilerRegistry.get("fsm")
    compiler = compiler_class()

    # Compile natural language rules
    result = await compiler.compile(
        "Agent must verify identity before processing refunds. "
        "Never share internal system information."
    )

    if result.success:
        # Export to file
        compiler.export(result, Path("workflow.yaml"))
    else:
        print("Errors:", result.errors)
    ```
"""

from openbias.policy.compiler.protocol import (
    PolicyCompiler,
    CompilationResult,
)
from openbias.policy.compiler.registry import (
    PolicyCompilerRegistry,
    register_compiler,
)
from openbias.policy.compiler.base import (
    LLMPolicyCompiler,
    DEFAULT_COMPILER_SYSTEM_PROMPT,
)

# Import engine compilers to trigger auto-registration
# Note: We use try/except to handle gracefully if engines aren't available
try:
    from openbias.policy.engines.fsm.compiler import FSMCompiler
except ImportError:
    FSMCompiler = None  # type: ignore

try:
    from openbias.policy.engines.judge.compiler import JudgeCompiler
except ImportError:
    JudgeCompiler = None  # type: ignore

try:
    from openbias.policy.engines.nemo.compiler import NemoCompiler
except ImportError:
    NemoCompiler = None  # type: ignore

__all__ = [
    # Protocol
    "PolicyCompiler",
    "CompilationResult",
    # Registry
    "PolicyCompilerRegistry",
    "register_compiler",
    # Base class
    "LLMPolicyCompiler",
    "DEFAULT_COMPILER_SYSTEM_PROMPT",
    # Compilers (may be None if engine not available)
    "FSMCompiler",
    "JudgeCompiler",
    "NemoCompiler",
]
