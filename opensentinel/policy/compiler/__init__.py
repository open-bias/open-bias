"""
Policy compiler system for natural language to engine config conversion.

This module provides infrastructure for compiling natural language policy
descriptions into engine-specific configurations:

- FSM: Natural language → workflow.yaml (WorkflowDefinition)
- NeMo: Natural language → Colang + config
- Judge: Natural language → rubric.yaml

Usage:
    ```python
    from opensentinel.policy.compiler import PolicyCompilerRegistry

    # Get FSM compiler class and instantiate
    compiler_class = PolicyCompilerRegistry.get("fsm")
    compiler = compiler_class()

    # Compile natural language policy
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

CLI Usage:
    ```bash
    # Compile to FSM workflow
    osentinel compile "verify identity before refunds" --engine fsm -o workflow.yaml

    # Auto-detect best engine
    osentinel compile "..." --engine auto
    ```
"""

from opensentinel.policy.compiler.protocol import (
    PolicyCompiler,
    CompilationResult,
)
from opensentinel.policy.compiler.registry import (
    PolicyCompilerRegistry,
    register_compiler,
)
from opensentinel.policy.compiler.base import (
    LLMPolicyCompiler,
    DEFAULT_COMPILER_SYSTEM_PROMPT,
)

# Import engine compilers to trigger auto-registration
# Note: We use try/except to handle gracefully if engines aren't available
try:
    from opensentinel.policy.engines.fsm.compiler import FSMCompiler
except ImportError:
    FSMCompiler = None  # type: ignore

try:
    from opensentinel.policy.engines.judge.compiler import JudgeCompiler
except ImportError:
    JudgeCompiler = None  # type: ignore

try:
    from opensentinel.policy.engines.nemo.compiler import NemoCompiler
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
