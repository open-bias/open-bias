"""
Deterministic intervention message templates for the LLM Policy Engine.

These templates are NOT LLM-generated. They use str.format() with context
variables, matching the pattern in strategies.py.

Template slots are filled with values from the session context, violation
details, and drift scores.
"""

# Default intervention templates
DEFAULT_TEMPLATES = {
    # Gentle nudge for minor drift
    "drift_warning": (
        "Remember: You are currently in the '{current_state}' state of the "
        "'{workflow_name}' workflow. Please stay focused on the expected behavior "
        "for this state."
    ),
    
    # More substantial context reminder
    "drift_intervention": (
        "Important context reminder:\n"
        "• Workflow: {workflow_name}\n"
        "• Current state: {current_state}\n"
        "• Active constraints: {active_constraints}\n\n"
        "Please ensure your responses align with the workflow requirements."
    ),
    
    # Hard block for critical drift
    "drift_critical": (
        "WORKFLOW DEVIATION DETECTED\n\n"
        "This conversation has significantly deviated from the expected workflow "
        "'{workflow_name}'. Drift score: {drift_score:.2f}\n\n"
        "Session {session_id} has been flagged for human review. "
        "The current request cannot be processed."
    ),
    
    # Constraint violation notification
    "constraint_violation": (
        "Policy violation detected:\n"
        "• Constraint: {constraint_name}\n"
        "• Severity: {severity}\n"
        "• Evidence: {evidence}\n\n"
        "Please adjust your response to comply with this constraint."
    ),
    
    # Multiple uncertain classifications
    "structural_drift": (
        "Warning: The past several responses have had uncertain state classifications. "
        "This may indicate the conversation is drifting from the expected workflow.\n\n"
        "Current workflow: {workflow_name}\n"
        "Expected behavior for '{current_state}': {state_description}"
    ),
    
    # Skipped required intermediate state
    "skip_violation": (
        "Important: You appear to have skipped a required step in the workflow.\n\n"
        "You cannot transition directly from '{from_state}' to '{to_state}'.\n"
        "The following intermediate state(s) must be completed first: {skipped_states}\n\n"
        "Please address the skipped steps before proceeding."
    ),
    
    # Generic policy violation
    "policy_violation": (
        "A policy violation has been detected:\n"
        "• Type: {violation_type}\n"
        "• Details: {violation_message}\n\n"
        "Please adjust your response to comply with the workflow requirements."
    ),
    
    # Intervention escalation
    "escalation": (
        "Multiple policy issues detected. This session is being escalated.\n\n"
        "• Violations: {violation_count}\n"
        "• Drift level: {drift_level}\n\n"
        "A human reviewer will be notified."
    ),
}


def get_template(name: str) -> str:
    """Get an intervention template by name.
    
    Args:
        name: Template name
        
    Returns:
        Template string with {slot} placeholders
    """
    return DEFAULT_TEMPLATES.get(name, DEFAULT_TEMPLATES["policy_violation"])


def format_template(name: str, context: dict) -> str:
    """Format a template with context values.
    
    Args:
        name: Template name
        context: Dict of values to fill template slots
        
    Returns:
        Formatted message string
    """
    template = get_template(name)
    try:
        return template.format(**context)
    except KeyError:
        # Return template with unfilled placeholders rather than failing
        return template
