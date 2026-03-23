"""
LLM-based constraint evaluator for the LLM Policy Engine.

Uses an LLM to evaluate soft constraints that cannot be checked
deterministically, with evidence memory for context accumulation.
"""

import logging
from typing import Any

from openbias.policy.engines.fsm.workflow.schema import (
    Constraint,
    ConstraintType,
    WorkflowDefinition,
)
from openbias.policy.engines.llm.llm_client import LLMClient, LLMClientError
from openbias.policy.engines.llm.models import (
    ConstraintEvaluation,
    SessionContext,
)
from openbias.policy.engines.llm.prompts import (
    CONSTRAINT_EVALUATION_SYSTEM,
    CONSTRAINT_EVALUATION_USER,
)

logger = logging.getLogger(__name__)

class LLMConstraintEvaluator:
    """LLM-based constraint evaluator.
    
    Evaluates soft constraints using an LLM, with batching for efficiency
    and evidence memory for accumulated context across turns.
    
    Example:
        evaluator = LLMConstraintEvaluator(llm_client, workflow)
        violations = await evaluator.evaluate(
            session,
            assistant_message="I'll help with your account",
            tool_calls=["get_account_info"]
        )
        for v in violations:
            if v.violated:
                print(f"Constraint {v.constraint_id} violated: {v.evidence}")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        workflow: WorkflowDefinition,
        max_constraints_per_batch: int = 5,
        evidence_threshold: float = 0.3,
    ):
        self.llm_client = llm_client
        self.workflow = workflow
        self.max_constraints_per_batch = max_constraints_per_batch
        self.evidence_threshold = evidence_threshold
        
        # Build constraint lookup
        self._constraints = {c.name: c for c in workflow.constraints}

    async def evaluate(
        self,
        session: SessionContext,
        assistant_message: str,
        tool_calls: list[str] | None = None,
    ) -> list[ConstraintEvaluation]:
        """Evaluate constraints for current turn.
        
        Args:
            session: Current session context
            assistant_message: The assistant's response text
            tool_calls: List of tool names called
            
        Returns:
            List of ConstraintEvaluation results
        """
        tool_calls = tool_calls or []
        
        # Select active constraints
        active_constraints = self._select_active_constraints(session)
        
        if not active_constraints:
            return []
        
        # Evaluate in batches
        all_evaluations = []
        
        for i in range(0, len(active_constraints), self.max_constraints_per_batch):
            batch = active_constraints[i:i + self.max_constraints_per_batch]
            
            try:
                evaluations = await self._evaluate_batch(
                    session,
                    batch,
                    assistant_message,
                    tool_calls,
                )
                all_evaluations.extend(evaluations)
            except LLMClientError as e:
                logger.error(f"Constraint evaluation batch failed: {e}")
                # Continue with other batches
        
        # Update evidence memory
        self._update_memory(session, all_evaluations)
        
        return all_evaluations

    def _select_active_constraints(
        self,
        session: SessionContext,
    ) -> list[Constraint]:
        """Select constraints that are active for evaluation.

        Constraint selection rules:
        - NEVER: Always active
        - PRECEDENCE: Active when trigger is current or recent
        - EVENTUALLY: Active if target not yet reached
        - RESPONSE: Active after trigger has occurred, until target is reached
        """
        active = []
        state_history = session.get_state_sequence()
        current_state = session.current_state

        for constraint in self.workflow.constraints:
            should_include = False

            if constraint.type == ConstraintType.NEVER:
                # Never constraints are always checked
                should_include = True

            elif constraint.type == ConstraintType.PRECEDENCE:
                # Check when trigger state is current or recent
                if constraint.trigger == current_state:
                    should_include = True
                elif constraint.trigger in state_history[-3:]:
                    should_include = True

            elif constraint.type == ConstraintType.EVENTUALLY:
                # Active if target not yet reached
                if constraint.target not in state_history:
                    should_include = True

            elif constraint.type == ConstraintType.RESPONSE:
                # Active after trigger has occurred
                if constraint.trigger in state_history:
                    # Only active if target not yet reached after the LAST trigger occurrence
                    trigger_idx = len(state_history) - 1 - state_history[::-1].index(constraint.trigger)
                    target_after = any(
                        s == constraint.target
                        for s in state_history[trigger_idx + 1:]
                    )
                    should_include = not target_after

            if should_include:
                active.append(constraint)

        logger.debug(f"Selected {len(active)} active constraints")
        return active

    async def _evaluate_batch(
        self,
        session: SessionContext,
        constraints: list[Constraint],
        assistant_message: str,
        tool_calls: list[str],
    ) -> list[ConstraintEvaluation]:
        """Evaluate a batch of constraints via LLM."""
        # Build prompts
        system_prompt = CONSTRAINT_EVALUATION_SYSTEM.format(
            workflow_name=self.workflow.name,
            current_state=session.current_state,
            state_history=" → ".join(session.get_state_sequence()[-10:]),
            constraints_block=self._build_constraints_block(constraints, session),
        )
        
        user_prompt = CONSTRAINT_EVALUATION_USER.format(
            assistant_message=assistant_message or "(no text content)",
            tool_calls_block=", ".join(tool_calls) if tool_calls else "(none)",
            conversation_block=self._format_conversation(session.turn_window),
            evidence_block=self._format_evidence(constraints, session),
        )
        
        # Call LLM
        response = await self.llm_client.complete_json(
            system_prompt, 
            user_prompt,
            session_id=session.session_id
        )
        
        # Parse evaluations
        return self._parse_evaluations(response, constraints)

    def _build_constraints_block(
        self,
        constraints: list[Constraint],
        session: SessionContext,
    ) -> str:
        """Format constraints for the prompt."""
        lines = []
        for c in constraints:
            lines.append(f"Constraint: {c.name}")
            lines.append(f"  Type: {c.type.value}")
            if c.description:
                lines.append(f"  Description: {c.description}")
            if c.trigger:
                lines.append(f"  Trigger: {c.trigger}")
            if c.target:
                lines.append(f"  Target: {c.target}")
            if c.message:
                lines.append(f"  Message: {c.message}")
            lines.append("")
        return "\n".join(lines)

    def _format_conversation(self, turn_window: list[dict]) -> str:
        """Format recent conversation for the prompt."""
        if not turn_window:
            return "(no previous turns)"
        
        lines = []
        for turn in turn_window[-5:]:
            role = turn.get("role", "unknown")
            message = turn.get("message", "")[:200]
            lines.append(f"{role}: {message}")
        return "\n".join(lines)

    def _format_evidence(
        self,
        constraints: list[Constraint],
        session: SessionContext,
    ) -> str:
        """Format previous evidence for constraints."""
        lines = []
        for c in constraints:
            evidence_list = session.constraint_memory.get(c.name, [])
            if evidence_list:
                lines.append(f"{c.name}:")
                for ev in evidence_list[-3:]:  # Last 3 pieces of evidence
                    lines.append(f"  - {ev}")
        return "\n".join(lines) if lines else "(no previous evidence)"

    def _parse_evaluations(
        self,
        response: Any,
        constraints: list[Constraint],
    ) -> list[ConstraintEvaluation]:
        """Parse LLM response into constraint evaluations."""
        evaluations = []
        
        # Handle list or dict response
        if isinstance(response, dict):
            response = response.get("evaluations", response.get("constraints", [response]))
        
        if not isinstance(response, list):
            response = [response]
        
        constraint_names = {c.name for c in constraints}
        
        for item in response:
            if isinstance(item, dict):
                constraint_id = item.get("constraint_id", item.get("name", ""))
                
                # Validate constraint exists and was requested
                if constraint_id in constraint_names:
                    evaluations.append(ConstraintEvaluation(
                        constraint_id=constraint_id,
                        violated=bool(item.get("violated", False)),
                        confidence=float(item.get("confidence", 0.0)),
                        evidence=str(item.get("evidence", "")),
                        severity=str(item.get("severity", "warning")),
                    ))
        
        return evaluations

    def _update_memory(
        self,
        session: SessionContext,
        evaluations: list[ConstraintEvaluation],
    ) -> None:
        """Update evidence memory for high-confidence evaluations."""
        for ev in evaluations:
            if ev.confidence >= self.evidence_threshold and ev.evidence:
                if ev.constraint_id not in session.constraint_memory:
                    session.constraint_memory[ev.constraint_id] = []
                
                # Append evidence
                session.constraint_memory[ev.constraint_id].append(ev.evidence)
                
                # Keep only last 5 pieces of evidence
                if len(session.constraint_memory[ev.constraint_id]) > 5:
                    session.constraint_memory[ev.constraint_id].pop(0)
