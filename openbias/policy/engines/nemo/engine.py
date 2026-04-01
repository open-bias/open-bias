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
    from openbias.policy.compiler.protocol import PolicyCompiler

from openbias.core.utils import extract_response_content
from openbias.policy.protocols import (
    PolicyEngine,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
    require_initialized,
)
from openbias.policy.registry import register_engine

logger = logging.getLogger(__name__)


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
                "Install with: pip install 'openbias[nemo]' "
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

        # Register Open Bias bridge actions
        self._register_openbias_actions()

        self._initialized = True
        logger.info(
            f"NemoGuardrailsPolicyEngine initialized with rails: {self._enabled_rails}"
        )

    def _register_openbias_actions(self):
        """Register Open Bias-specific actions with NeMo."""
        if not self._rails:
            return

        async def openbias_log_violation(
            violation_name: str,
            severity: str = "error",
            message: str = "",
        ) -> dict[str, Any]:
            """
            Log a policy violation from NeMo context.

            Can be called from Colang flows to record violations
            that Open Bias should track.
            """
            logger.warning(
                f"NeMo violation: {violation_name} (severity={severity}): {message}"
            )
            return {
                "logged": True,
                "violation": violation_name,
                "severity": severity,
            }

        async def openbias_request_intervention(
            intervention_name: str,
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """
            Request a Open Bias intervention from NeMo context.

            Allows Colang flows to trigger Open Bias interventions.
            """
            logger.info(f"NeMo requesting intervention: {intervention_name}")
            return {
                "intervention_requested": intervention_name,
                "context": context or {},
            }

        self._rails.register_action(openbias_log_violation, "openbias_log_violation")
        self._rails.register_action(
            openbias_request_intervention, "openbias_request_intervention"
        )

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate request using NeMo input rails.

        NeMo processes the messages through input rails which may:
        - Allow the request unchanged
        - Modify the request (mask sensitive data, etc.)
        - Block the request (jailbreak detected, etc.)
        """

        if "input" not in self._enabled_rails:
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"rails_skipped": "input not enabled"},
            )
        messages = request_data.get("messages", [])
        if not messages:
            return EvaluationResult(status=EvaluationStatus.ALLOW)
        try:
            # Process through NeMo input rails
            result = await self._rails.generate_async(
                messages=messages,
                options={
                    "output_vars": True,
                    "log": {"activated_rails": True},
                }
            )

            # Check if any input rails were activated
            activations = self._check_rail_activations(result)
            if activations:
                violations = [
                    ViolationRecord(
                        rule_id=f"nemo_{a.get('type', 'input')}_blocked",
                        rule_name=f"nemo_{a.get('type', 'input')}_blocked",
                        reason=a.get("name", "NeMo input rail triggered"),
                        severity="error",
                        engine=self.name,
                    )
                    for a in activations
                ]
                return EvaluationResult(
                    status=EvaluationStatus.VIOLATION,
                    violations=violations,
                    metadata={"session_id": session_id},
                )
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"nemo_processed": True},
            )
        except Exception as e:
            logger.error(f"NeMo input rail evaluation failed: {e}", exc_info=True)

            if self._fail_closed:
                # External engine error with fail_closed — report as violation
                # with provider metadata; interceptor decides actual enforcement
                return EvaluationResult(
                    status=EvaluationStatus.VIOLATION,
                    violations=[ViolationRecord(
                        rule_id="nemo_evaluation_error",
                        rule_name="nemo_evaluation_error",
                        reason=f"NeMo evaluation failed: {e}",
                        severity="critical",
                        engine=self.name,
                        extra={"provider_decision": "block", "provider_reason": str(e)},
                    )],
                )
            # Fail open
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"error": str(e)},
            )
    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate response using NeMo output rails.

        NeMo will check the response for:
        - Hallucination (if fact-checking enabled)
        - Unsafe content
        - Policy violations
        - PII leakage
        """

        if "output" not in self._enabled_rails:
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"rails_skipped": "output not enabled"},
            )
        # Extract response content
        content = extract_response_content(response_data)
        if not content:
            return EvaluationResult(status=EvaluationStatus.ALLOW)
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

            # Check if any output rails were activated
            activations = self._check_rail_activations(result)
            if activations:
                violations = [
                    ViolationRecord(
                        rule_id=f"nemo_{a.get('type', 'output')}_blocked",
                        rule_name=f"nemo_{a.get('type', 'output')}_blocked",
                        reason=a.get("name", "NeMo output rail triggered"),
                        severity="error",
                        engine=self.name,
                    )
                    for a in activations
                ]
                return EvaluationResult(
                    status=EvaluationStatus.VIOLATION,
                    violations=violations,
                    metadata={"original_response": content[:200]},
                )
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"nemo_output_verified": True},
            )
        except Exception as e:
            logger.error(f"NeMo output rail evaluation failed: {e}", exc_info=True)

            if self._fail_closed:
                return EvaluationResult(
                    status=EvaluationStatus.VIOLATION,
                    violations=[ViolationRecord(
                        rule_id="nemo_output_evaluation_error",
                        rule_name="nemo_output_evaluation_error",
                        reason=f"NeMo output evaluation failed: {e}",
                        severity="critical",
                        engine=self.name,
                        extra={"provider_decision": "block", "provider_reason": str(e)},
                    )],
                )
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"error": str(e)},
            )
    def _check_rail_activations(self, result: Any) -> list[dict[str, Any]]:
        """
        Inspect NeMo result for activated rails.

        Returns a list of activation dicts from result.log.activated_rails.
        An empty list means no rails were triggered.

        Handles older NeMo versions that return plain strings (no .log attribute)
        by returning an empty list.
        """
        try:
            log = getattr(result, "log", None)
            if log is None:
                return []
            activated = getattr(log, "activated_rails", None)
            if not activated:
                return []
            return list(activated)
        except Exception:
            return []

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get current session state for debugging/tracing."""
        return None

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
        from openbias.policy.engines.nemo.compiler import NemoCompiler
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
