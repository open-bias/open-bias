"""
Intervention configuration for workflow corrections.

Loads and manages intervention configs from workflow definitions.
Actual application of interventions is handled by the interceptor.
"""

import logging
from typing import Optional, Dict, Any

from opensentinel.core.intervention.strategies import (
    InterventionConfig,
    StrategyType,
)
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)


class InterventionHandler:
    """
    Manages intervention configurations from workflow definitions.

    Loads and provides access to intervention configs. Actual application
    of interventions is handled by the interceptor layer.

    Example:
        ```python
        from opensentinel.policy.engines.fsm import InterventionHandler, WorkflowParser

        workflow = WorkflowParser.parse_file("workflow.yaml")
        handler = InterventionHandler(workflow)

        # Look up intervention config
        info = handler.get_intervention_info("prompt_verify_identity")
        ```
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        default_strategy: StrategyType = StrategyType.SYSTEM_PROMPT_APPEND,
        max_intervention_attempts: int = 3,
    ):
        self.workflow = workflow
        self.default_strategy = default_strategy
        self.max_intervention_attempts = max_intervention_attempts
        self._intervention_configs = self._load_configs()

        logger.debug(
            f"InterventionHandler initialized with {len(self._intervention_configs)} interventions"
        )

    def _load_configs(self) -> Dict[str, InterventionConfig]:
        """Load intervention configs from workflow."""
        configs = {}

        for name, template in self.workflow.interventions.items():
            # Parse strategy from template if specified
            # Format: "strategy:template" or just "template"
            strategy = self.default_strategy
            message = template

            if template.startswith("block:"):
                strategy = StrategyType.SYSTEM_PROMPT_APPEND
                message = template[6:].strip()
            elif template.startswith("inject:"):
                strategy = StrategyType.USER_MESSAGE_INJECT
                message = template[7:].strip()
            elif template.startswith("remind:"):
                strategy = StrategyType.SYSTEM_PROMPT_APPEND
                message = template[7:].strip()

            configs[name] = InterventionConfig(
                strategy_type=strategy,
                message_template=message,
                max_applications=self.max_intervention_attempts,
            )

        return configs

    def get_config(self, intervention_name: str) -> Optional[InterventionConfig]:
        """Get intervention config by name."""
        return self._intervention_configs.get(intervention_name)

    def get_intervention_info(self, intervention_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an intervention."""
        config = self._intervention_configs.get(intervention_name)
        if not config:
            return None

        return {
            "name": intervention_name,
            "strategy": config.strategy_type.value,
            "template": config.message_template,
            "max_applications": config.max_applications,
        }

    def list_interventions(self) -> list[str]:
        """List all available intervention names."""
        return list(self._intervention_configs.keys())


