"""
Open Bias policy engine system.

This module provides a pluggable infrastructure for policy evaluation,
supporting multiple policy mechanisms including:

- FSM (Finite State Machine): Workflow enforcement using states and transitions
- NeMo Guardrails: NVIDIA's guardrails for input/output filtering
- Judge: LLM-as-a-Judge evaluating responses against compiled rules
- LLM: Semantic state tracking and drift detection

Usage:
    ```python
    from openbias.policy import PolicyEngineRegistry, EvaluationStatus

    # Create and initialize an FSM engine
    engine = PolicyEngineRegistry.create("fsm")
    await engine.initialize({"workflow_path": "./workflow.yaml"})

    # Evaluate a request
    result = await engine.evaluate_request(
        session_id="session-123",
        request_data={"messages": [...]},
    )

    if result.status == EvaluationStatus.VIOLATION:
        print("Violation detected:", [v.reason for v in result.violations])
    ```
"""

# Compiler imports
from openbias.policy.compiler import (
    CompilationResult,
    LLMPolicyCompiler,
    PolicyCompiler,
    PolicyCompilerRegistry,
    register_compiler,
)

# Import engines to trigger auto-registration
from openbias.policy.engines.fsm import FSMPolicyEngine
from openbias.policy.engines.judge import JudgePolicyEngine
from openbias.policy.engines.nemo import NemoGuardrailsPolicyEngine
from openbias.policy.engines.stateful import (
    StateClassificationResult,
    StatefulPolicyEngine,
)
from openbias.policy.protocols import (
    Decision,
    EvaluationResult,
    EvaluationStatus,
    PolicyEngine,
    ViolationRecord,
    require_initialized,
)
from openbias.policy.registry import GenericRegistry, PolicyEngineRegistry, register_engine

__all__ = [
    # Core protocols
    "PolicyEngine",
    "StatefulPolicyEngine",
    "require_initialized",

    # Result types
    "Decision",
    "EvaluationResult",
    "EvaluationStatus",
    "ViolationRecord",
    "StateClassificationResult",
    # Registry
    "GenericRegistry",
    "PolicyEngineRegistry",
    "register_engine",
    # Engines
    "FSMPolicyEngine",
    "NemoGuardrailsPolicyEngine",
    "JudgePolicyEngine",
    # Compiler protocol
    "PolicyCompiler",
    "CompilationResult",
    # Compiler registry
    "PolicyCompilerRegistry",
    "register_compiler",
    "LLMPolicyCompiler",
]
