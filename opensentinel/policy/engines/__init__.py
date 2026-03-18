"""
Policy engine implementations.

This package contains all available policy engine implementations:

- fsm: Finite State Machine based workflow enforcement
- nemo: NVIDIA NeMo Guardrails integration
- judge: LLM-as-a-Judge rubric evaluation
- llm: LLM-based state classification and drift detection

Engines are auto-registered when imported.
"""

# Import engines to trigger registration
# These imports populate the PolicyEngineRegistry
from opensentinel.policy.engines import fsm
from opensentinel.policy.engines import nemo
from opensentinel.policy.engines import llm
from opensentinel.policy.engines import judge

__all__ = ["fsm", "nemo", "llm", "judge"]

