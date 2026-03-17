"""
Open Sentinel - Reliability layer for AI agents.

Monitors workflow adherence and intervenes when agents deviate from expected behavior.
Uses LiteLLM proxy approach - customers only need to change their base_url.

Supports multiple policy engines:
- FSM: Finite State Machine workflow enforcement
- NeMo: NVIDIA NeMo Guardrails integration
- Composite: Combine multiple engines together

Quick Start:
    Configure via `osentinel.yaml`:
    ```yaml
    engine: judge
    policy:
      - "Be helpful and professional"
    ```
    Then run:
    ```bash
    osentinel serve
    ```

    Or programmatically:
    ```python
    from opensentinel import SentinelProxy, SentinelSettings

    # Configure and start proxy
    settings = SentinelSettings()
    proxy = SentinelProxy(settings)
    await proxy.start()
    ```

    Then point your LLM client to http://localhost:4000/v1

Using Policy Engines Directly:
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
"""

__version__ = "0.2.1"

# Core configuration
from opensentinel.config.settings import SentinelSettings

# Proxy server
from opensentinel.proxy.server import SentinelProxy, start_proxy

# Workflow components (from FSM engine)
from opensentinel.policy.engines.fsm import (
    WorkflowDefinition,
    State,
    Transition,
    Constraint,
    ConstraintType,
    WorkflowParser,
    WorkflowStateMachine,

    StateClassifier,
)

# Policy engines
from opensentinel.policy import (
    PolicyEngine,
    PolicyEngineRegistry,
    Decision,
    EngineResult,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "SentinelSettings",
    # Proxy
    "SentinelProxy",
    "start_proxy",
    # Workflow
    "WorkflowDefinition",
    "State",
    "Transition",
    "Constraint",
    "ConstraintType",
    "WorkflowParser",
    "WorkflowStateMachine",
    # Monitoring

    "StateClassifier",
    # Policy engines
    "PolicyEngine",
    "PolicyEngineRegistry",
    "Decision",
    "EngineResult",
]
