"""
Open Bias - Reliability layer for AI agents.

Monitors workflow adherence and intervenes when agents deviate from expected behavior.
Uses LiteLLM proxy approach - customers only need to change their base_url.

Supports multiple policy engines:
- FSM: Finite State Machine workflow enforcement
- Judge: LLM-as-judge policy evaluation
- NeMo: NVIDIA NeMo Guardrails integration
- LLM: LLM-based drift detection and intervention

Quick Start:
    Configure via `openbias.yaml`:
    ```yaml
    evaluators:
      - type: judge
    ```
    Then run:
    ```bash
    openbias serve
    ```

    Or programmatically:
    ```python
    from openbias import Proxy, Settings

    # Configure and start proxy
    settings = Settings()
    proxy = Proxy(settings)
    await proxy.start()
    ```

    Then point your LLM client to http://localhost:4000/v1

Using Policy Engines Directly:
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

__version__ = "0.4.0"

# Core configuration
from openbias.config.settings import Settings

# Proxy server
from openbias.proxy.server import Proxy, start_proxy
from openbias.traces import TraceCase, TraceDataset, TraceMetadata

# Policy engines
from openbias.policy import (
    PolicyEngine,
    PolicyEngineRegistry,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Settings",
    # Proxy
    "Proxy",
    "start_proxy",
    # Traces
    "TraceCase",
    "TraceDataset",
    "TraceMetadata",
    # Policy engines
    "PolicyEngine",
    "PolicyEngineRegistry",
    "EvaluationResult",
    "EvaluationStatus",
    "ViolationRecord",
]
