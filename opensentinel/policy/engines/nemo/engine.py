"""
NeMo Guardrails policy engine implementation.

Integrates NVIDIA's NeMo Guardrails as a PolicyEngine for
input/output filtering, jailbreak detection, and content moderation.

NeMo Guardrails provides:
- Input rails: Filter/block malicious or inappropriate inputs
- Output rails: Filter/block unsafe or policy-violating outputs
- Dialog rails: Guide conversation flow
- Retrieval rails: Verify factual accuracy

Requires: pip install nemoguardrails
"""

from typing import Any, TYPE_CHECKING
import logging
import sys

if TYPE_CHECKING:
    from opensentinel.policy.compiler.protocol import PolicyCompiler

from opensentinel.core.utils import extract_response_content
from opensentinel.policy.protocols import (
    PolicyEngine,
    Decision,
    EngineResult,
    require_initialized,
)
from opensentinel.policy.registry import register_engine

logger = logging.getLogger(__name__)

# Markers that indicate NeMo blocked a request/response
BLOCKED_MARKERS = [
    "i cannot",
    "i'm not able to",
    "i am not able to",
    "refuse to",
    "[blocked]",
    "i can't help with",
    "i'm unable to",
    "sorry, but i can't",
]


@register_engine("nemo")
class NemoGuardrailsPolicyEngine(PolicyEngine):
    """
    NeMo Guardrails based policy engine.

    Uses NVIDIA's NeMo Guardrails for comprehensive safety filtering
    including jailbreak detection, content moderation, and policy enforcement.

    Configuration:
        - config_path: str - Path to NeMo config directory (with config.yml)
        OR
        - config: dict - RailsConfig parameters

    Optional configuration:
        - custom_actions: dict - Custom action functions to register
        - rails: list - Which rails to enable ["input", "output", "dialog"]
        - fail_closed: bool - If True, block on evaluation errors (default: False)

    Example:
        ```python
        engine = NemoGuardrailsPolicyEngine()
        await engine.initialize({
            "config_path": "./nemo_config/"
        })

        result = await engine.evaluate_request(
            session_id="session-123",
            request_data={"messages": [...]},
        )
        ```
    """

    def __init__(self):
        self._rails = None
        self._config = None
        self._initialized = False
        self._fail_closed = False
        self._enabled_rails = ["input", "output"]

    @property
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        return "nemo:guardrails"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "nemo"

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize with NeMo configuration.

        Args:
            config: Configuration dict with either:
                - config_path: str - Path to NeMo config directory
                - config: dict - Direct RailsConfig parameters

        Raises:
            ImportError: If nemoguardrails not installed
            ValueError: If invalid configuration
        """
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError:
            raise ImportError(
                "NeMo Guardrails not installed. "
                "Install with: pip install 'opensentinel[nemo]' "
                "or: pip install nemoguardrails"
            )
        except TypeError as e:
            # Check for known Python 3.14 incompatibility with LangChain/Pydantic
            # "TypeError: 'function' object is not subscriptable"
            if sys.version_info >= (3, 14) and "subscriptable" in str(e):
                raise ImportError(
                    "NeMo Guardrails (via LangChain) is not compatible with Python 3.14 yet. "
                    "Please downgrade to Python 3.10-3.13."
                ) from e
            raise

        # Load configuration
        if "config_path" in config:
            self._config = RailsConfig.from_path(config["config_path"])
        elif "config" in config:
            config_params = config["config"]
            if isinstance(config_params, dict):
                self._config = RailsConfig.from_content(**config_params)
            else:
                self._config = config_params
        else:
            raise ValueError(
                "NeMo engine requires 'config_path' or 'config' in configuration"
            )

        # Create LLMRails instance
        self._rails = LLMRails(self._config)

        # Register custom actions if provided
        if "custom_actions" in config:
            for action_name, action_fn in config["custom_actions"].items():
                self._rails.register_action(action_fn, action_name)
                logger.debug(f"Registered custom action: {action_name}")

        # Configure rails to use
        if "rails" in config:
            self._enabled_rails = config["rails"]

        # Configure failure behavior
        self._fail_closed = config.get("fail_closed", False)

        # Register Open Sentinel bridge actions
        self._register_sentinel_actions()

        self._initialized = True
        logger.info(
            f"NemoGuardrailsPolicyEngine initialized with rails: {self._enabled_rails}"
        )

    def _register_sentinel_actions(self):
        """Register Open Sentinel-specific actions with NeMo."""
        if not self._rails:
            return

        async def sentinel_log_violation(
            violation_name: str,
            severity: str = "error",
            message: str = "",
        ) -> dict[str, Any]:
            """
            Log a policy violation from NeMo context.

            Can be called from Colang flows to record violations
            that Open Sentinel should track.
            """
            logger.warning(
                f"NeMo violation: {violation_name} (severity={severity}): {message}"
            )
            return {
                "logged": True,
                "violation": violation_name,
                "severity": severity,
            }

        async def sentinel_request_intervention(
            intervention_name: str,
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """
            Request a Open Sentinel intervention from NeMo context.

            Allows Colang flows to trigger Open Sentinel interventions.
            """
            logger.info(f"NeMo requesting intervention: {intervention_name}")
            return {
                "intervention_requested": intervention_name,
                "context": context or {},
            }

        self._rails.register_action(sentinel_log_violation, "sentinel_log_violation")
        self._rails.register_action(
            sentinel_request_intervention, "sentinel_request_intervention"
        )

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """
        Evaluate request using NeMo input rails.

        NeMo processes the messages through input rails which may:
        - Allow the request unchanged
        - Modify the request (mask sensitive data, etc.)
        - Block the request (jailbreak detected, etc.)
        """

        if "input" not in self._enabled_rails:
            return EngineResult(
                decision=Decision.ALLOW,
                metadata={"rails_skipped": "input not enabled"},
            )

        messages = request_data.get("messages", [])
        if not messages:
            return EngineResult(decision=Decision.ALLOW)

        try:
            # Process through NeMo input rails
            result = await self._rails.generate_async(
                messages=messages,
                options={
                    "output_vars": True,
                    "log": {"activated_rails": True},
                }
            )

            # Check if the request was blocked
            response_content = extract_response_content(result)

            if self._is_blocked_response(response_content):
                return EngineResult(
                    decision=Decision.BLOCK,
                    message="Request blocked by NeMo input guardrails",
                    metadata={
                        "nemo_response": response_content,
                        "session_id": session_id,
                    },
                )

            # Check if messages were modified (NeMo may sanitize or modify input)
            modified = self._check_for_modifications(result, messages)
            if modified:
                return EngineResult(
                    decision=Decision.INTERVENE,
                    modified_messages=modified,
                    metadata={
                        "modification_type": "nemo_input_sanitization",
                    },
                )

            return EngineResult(
                decision=Decision.ALLOW,
                metadata={"nemo_processed": True},
            )

        except Exception as e:
            logger.error(f"NeMo input rail evaluation failed: {e}", exc_info=True)

            if self._fail_closed:
                return EngineResult(
                    decision=Decision.BLOCK,
                    message=f"NeMo evaluation failed: {str(e)}",
                )

            # Fail open
            return EngineResult(
                decision=Decision.ALLOW,
                metadata={"error": str(e)},
            )

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """
        Evaluate response using NeMo output rails.

        NeMo will check the response for:
        - Hallucination (if fact-checking enabled)
        - Unsafe content
        - Policy violations
        - PII leakage
        """

        if "output" not in self._enabled_rails:
            return EngineResult(
                decision=Decision.ALLOW,
                metadata={"rails_skipped": "output not enabled"},
            )

        # Extract response content
        content = extract_response_content(response_data)
        if not content:
            return EngineResult(decision=Decision.ALLOW)

        messages = request_data.get("messages", [])
        messages_with_response = messages + [
            {"role": "assistant", "content": content}
        ]

        try:
            # Process through NeMo output rails
            result = await self._rails.generate_async(
                messages=messages_with_response,
                options={
                    "output_vars": True,
                    "log": {"activated_rails": True},
                }
            )

            response_content = extract_response_content(result)

            if self._is_blocked_response(response_content):
                return EngineResult(
                    decision=Decision.BLOCK,
                    message="Response blocked by NeMo output guardrails",
                    metadata={
                        "original_response": content[:200],
                        "nemo_response": response_content[:200],
                        "nemo_blocked": True,
                    },
                )

            return EngineResult(
                decision=Decision.ALLOW,
                metadata={"nemo_output_verified": True},
            )

        except Exception as e:
            logger.error(f"NeMo output rail evaluation failed: {e}", exc_info=True)

            if self._fail_closed:
                return EngineResult(
                    decision=Decision.BLOCK,
                    message=f"NeMo output evaluation failed: {str(e)}",
                )

            return EngineResult(
                decision=Decision.ALLOW,
                metadata={"error": str(e)},
            )

    def _is_blocked_response(self, content: str) -> bool:
        """Check if NeMo blocked the request/response."""
        if not content:
            return False

        content_lower = content.lower()
        return any(marker in content_lower for marker in BLOCKED_MARKERS)

    def _check_for_modifications(
        self,
        result: Any,
        original_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """
        Check if NeMo modified the messages.

        Returns modified messages if changes detected, None otherwise.
        """
        # TODO: implement proper message-diff detection per NeMo Guardrails version
        # NeMo may return modified messages in the result
        if hasattr(result, "messages") and result.messages != original_messages:
            return result.messages

        return None

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get current session state for debugging/tracing."""
        return {
            "engine": "nemo",
            "session_id": session_id,
            "initialized": self._initialized,
            "enabled_rails": self._enabled_rails,
        }

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        logger.debug(f"NeMo session {session_id} reset")

    def get_compiler(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "PolicyCompiler | None":
        """Return a NemoCompiler instance."""
        from opensentinel.policy.engines.nemo.compiler import NemoCompiler
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return NemoCompiler(**kwargs)

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._rails = None
        self._config = None
        self._initialized = False
        logger.info("NemoGuardrailsPolicyEngine shutdown")
