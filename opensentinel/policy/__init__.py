"""
Open Sentinel policy engine system.

This module provides a pluggable infrastructure for policy evaluation,
supporting multiple policy mechanisms including:

- FSM (Finite State Machine): Workflow enforcement using states and transitions
- NeMo Guardrails: NVIDIA's guardrails for input/output filtering
- Judge: LLM-as-a-Judge evaluating responses against rubrics
- LLM: Semantic state tracking and drift detection
- Composite: Combine multiple engines together

Usage:
    ```python
    from opensentinel.policy import PolicyEngineRegistry, Decision

    # Create and initialize an FSM engine
    engine = PolicyEngineRegistry.create("fsm")
    await engine.initialize({"workflow_path": "./workflow.yaml"})

    # Evaluate a request
    result = await engine.evaluate_request(
        session_id="session-123",
        request_data={"messages": [...]},
    )

    if result.decision == Decision.BLOCK:
        print("Request blocked:", result.message)
    ```

For composite engines:
    ```python
    engine = PolicyEngineRegistry.create("composite")
    await engine.initialize({
        "engines": [
            {"type": "fsm", "config": {"workflow_path": "..."}},
            {"type": "nemo", "config": {"config_path": "..."}}
        ]
    })
    ```
"""

from opensentinel.policy.protocols import (
    PolicyEngine,
    StatefulPolicyEngine,
    Decision,
    EngineResult,
    StateClassificationResult,
    require_initialized,
)
from opensentinel.policy.registry import PolicyEngineRegistry, register_engine

# Compiler imports
from opensentinel.policy.compiler import (
    PolicyCompiler,
    CompilationResult,
    PolicyCompilerRegistry,
    register_compiler,
    LLMPolicyCompiler,
)

# Import engines to trigger auto-registration
# Note: We use try/except to handle optional dependencies gracefully
try:
    from opensentinel.policy.engines.fsm import FSMPolicyEngine
except ImportError:
    FSMPolicyEngine = None  # type: ignore

try:
    from opensentinel.policy.engines.nemo import NemoGuardrailsPolicyEngine
except ImportError:
    # NeMo is optional
    NemoGuardrailsPolicyEngine = None  # type: ignore

try:
    from opensentinel.policy.engines.composite import CompositePolicyEngine
except ImportError:
    CompositePolicyEngine = None  # type: ignore

try:
    from opensentinel.policy.engines.judge import JudgePolicyEngine
except ImportError:
    JudgePolicyEngine = None  # type: ignore

__all__ = [
    # Core protocols
    "PolicyEngine",
    "StatefulPolicyEngine",
    "require_initialized",

    # Result types
    "Decision",
    "EngineResult",
    "StateClassificationResult",
    # Engine registry
    "PolicyEngineRegistry",
    "register_engine",
    # Engines (may be None if not available)
    "FSMPolicyEngine",
    "NemoGuardrailsPolicyEngine",
    "CompositePolicyEngine",
    "JudgePolicyEngine",
    # Compiler protocol
    "PolicyCompiler",
    "CompilationResult",
    # Compiler registry
    "PolicyCompilerRegistry",
    "register_compiler",
    "LLMPolicyCompiler",
]
