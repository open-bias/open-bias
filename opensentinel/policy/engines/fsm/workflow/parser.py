"""
Workflow definition parser.

Loads and validates workflow definitions from YAML or JSON files.
"""

import logging
from pathlib import Path

import yaml

from opensentinel.policy.engines.fsm.compiler import compile_workflow
from opensentinel.policy.engines.fsm.workflow.schema import SimpleWorkflowConfig, WorkflowDefinition

logger = logging.getLogger(__name__)


class WorkflowParser:
    """
    Parse and validate workflow definitions.

    Supports:
    - YAML files (.yaml, .yml)
    - JSON files (.json)
    - Direct string parsing

    Example:
        ```python
        # From file
        workflow = WorkflowParser.parse_file("workflow.yaml")

        # From string
        yaml_content = '''
        name: my-workflow
        states:
          - name: start
            is_initial: true
        '''
        workflow = WorkflowParser.parse_string(yaml_content)
        ```
    """

    @staticmethod
    def parse_file(path: str | Path) -> WorkflowDefinition:
        """
        Parse workflow from file.

        Args:
            path: Path to workflow file (YAML or JSON)

        Returns:
            Validated WorkflowDefinition

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
            ValidationError: If workflow is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            return WorkflowParser.parse_string(content, format="yaml")
        elif path.suffix == ".json":
            return WorkflowParser.parse_string(content, format="json")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def parse_string(content: str, format: str = "yaml") -> WorkflowDefinition:
        """
        Parse workflow from string content.

        Args:
            content: YAML or JSON string
            format: "yaml" or "json"

        Returns:
            Validated WorkflowDefinition

        Raises:
            ValueError: If format is unsupported
            ValidationError: If workflow is invalid
        """
        if format == "yaml":
            data = yaml.safe_load(content)
        elif format == "json":
            import json

            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if data is None:
            raise ValueError("Empty workflow definition")

        return WorkflowParser._load(data)

    @staticmethod
    def parse_dict(data: dict) -> WorkflowDefinition:
        """
        Parse workflow from dictionary.

        Detects format automatically:
        - ``steps`` key → simple format, compiled to internal
        - ``states`` key → internal format, loaded directly

        Args:
            data: Workflow definition as dict

        Returns:
            Validated WorkflowDefinition
        """
        return WorkflowParser._load(data)

    @staticmethod
    def _load(data: dict) -> WorkflowDefinition:
        """Detect format and load accordingly."""
        if "steps" in data:
            if "states" in data:
                logger.warning(
                    "Workflow '%s' has both 'steps' and 'states'; "
                    "using 'steps' (simple format). Remove 'states' to silence this warning.",
                    data.get("name", "<unknown>"),
                )
            config = SimpleWorkflowConfig.model_validate(data)
            return compile_workflow(config)
        return WorkflowDefinition.model_validate(data)

    @staticmethod
    def validate_file(path: str | Path) -> tuple[bool, str]:
        """
        Validate a workflow file without fully loading it.

        Args:
            path: Path to workflow file

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            workflow = WorkflowParser.parse_file(path)
            return True, f"Valid workflow: {workflow.name}"
        except FileNotFoundError as e:
            return False, f"File not found: {e}"
        except ValueError as e:
            return False, f"Invalid format: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"


class WorkflowRegistry:
    """
    Store and retrieve workflow definitions.

    Manages multiple workflows and their association with sessions.
    """

    def __init__(self):
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._session_mappings: dict[str, str] = {}  # session_id -> workflow_name
        self._default_workflow: str | None = None

    def register(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    def register_from_file(self, path: str | Path) -> str:
        """
        Register workflow from file.

        Returns:
            Name of registered workflow
        """
        workflow = WorkflowParser.parse_file(path)
        self.register(workflow)
        return workflow.name

    def get(self, name: str) -> WorkflowDefinition | None:
        """Get workflow by name."""
        return self._workflows.get(name)

    def get_for_session(self, session_id: str) -> WorkflowDefinition | None:
        """Get workflow for a session."""
        # Check explicit mapping
        if session_id in self._session_mappings:
            workflow_name = self._session_mappings[session_id]
            return self._workflows.get(workflow_name)

        # Check prefix matches
        for prefix, workflow_name in self._session_mappings.items():
            if session_id.startswith(prefix):
                return self._workflows.get(workflow_name)

        # Return default
        if self._default_workflow:
            return self._workflows.get(self._default_workflow)

        # Return first registered workflow if only one exists
        if len(self._workflows) == 1:
            return next(iter(self._workflows.values()))

        return None

    def assign_to_session(self, session_id: str, workflow_name: str) -> None:
        """Assign a workflow to a session."""
        if workflow_name not in self._workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        self._session_mappings[session_id] = workflow_name

    def set_default(self, workflow_name: str) -> None:
        """Set the default workflow."""
        if workflow_name not in self._workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        self._default_workflow = workflow_name

    def list_workflows(self) -> list[str]:
        """List all registered workflow names."""
        return list(self._workflows.keys())
