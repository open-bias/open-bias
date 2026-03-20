"""
FSM Policy Compiler — Deterministic SimpleWorkflowConfig → WorkflowDefinition.

Transforms plain-English steps/rules into the internal FSM representation
without any LLM calls. The pipeline:

1. Parse steps → State objects
2. Infer transitions from step order + parenthetical hints
3. Parse rules → Constraint objects
4. Generate hidden states for NEVER conceptual targets
5. Map tools to states
6. Generate classification hints
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from opensentinel.policy.compiler.protocol import CompilationResult, PolicyCompiler
from opensentinel.policy.compiler.registry import register_compiler
from opensentinel.policy.engines.fsm.workflow.schema import (
    ClassificationHint,
    Constraint,
    ConstraintType,
    SimpleWorkflowConfig,
    State,
    Transition,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)

# Keywords that signal a terminal state
_TERMINAL_KEYWORDS = re.compile(
    r"\b(resolve|close|end|finish|complete|done|terminate|goodbye|farewell)\b",
    re.IGNORECASE,
)

# Pattern to extract parenthetical hints from step text
_PAREN_HINT = re.compile(r"\(([^)]+)\)")


def slugify(text: str) -> str:
    """Convert plain-English text to a snake_case identifier."""
    # Strip parentheticals first
    text = _PAREN_HINT.sub("", text).strip()
    # Lowercase, replace non-alnum with underscore, collapse runs
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug


def _strip_filler(text: str) -> str:
    """Remove leading filler words (any, the, a, an) from text."""
    return re.sub(r"^(any|the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)


def _resolve_state(
    slug: str, state_names: set[str], *, strict: bool = False
) -> str:
    """
    Resolve a slugified rule reference to a known state name.

    Tries (in order):
    1. Exact match
    2. slug is a substring of a state name
    3. A state name is a substring of slug
    4. (unless *strict*) Significant word overlap between slug and state names

    When *strict* is True, only exact/substring matches are tried. This is
    used for NEVER constraints, which should create hidden states for
    conceptual targets rather than fuzzy-matching to existing states.
    """
    if slug in state_names:
        return slug

    # Substring: slug contained in state name
    matches = [name for name in state_names if slug in name]
    if matches:
        return min(matches, key=len)

    # Reverse substring: state name contained in slug
    matches = [name for name in state_names if name in slug]
    if matches:
        return max(matches, key=len)  # prefer longest (most specific) match

    if strict:
        return slug

    # Word overlap: pick the state sharing the most words with slug
    slug_words = set(slug.split("_")) - {"and", "or", "the", "a", "an"}
    best: str | None = None
    best_overlap = 0
    for name in state_names:
        name_words = set(name.split("_")) - {"and", "or", "the", "a", "an"}
        overlap = len(slug_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best = name
    if best is not None and best_overlap > 0:
        return best

    return slug


def compile_workflow(config: SimpleWorkflowConfig) -> WorkflowDefinition:
    """
    Compile a SimpleWorkflowConfig into a WorkflowDefinition.

    Args:
        config: Human-authored simple config.

    Returns:
        Fully populated WorkflowDefinition.
    """
    states = _parse_steps(config.steps)
    transitions = _infer_transitions(states, config.steps)
    constraints, hidden_states = _parse_rules(config.rules, states)
    states.extend(hidden_states)
    _map_tools(states, config.tools, config.steps)
    _generate_hints(states)

    return WorkflowDefinition(
        name=config.name,
        description=f"Compiled from simple config: {config.name}",
        states=states,
        transitions=transitions,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Step 1: Parse steps → States
# ---------------------------------------------------------------------------


def _parse_steps(steps: list[str]) -> list[State]:
    """Convert step descriptions into State objects."""
    states: list[State] = []
    for i, step in enumerate(steps):
        name = slugify(step)
        is_initial = i == 0
        is_terminal = bool(_TERMINAL_KEYWORDS.search(step))
        states.append(
            State(
                name=name,
                description=step,
                is_initial=is_initial,
                is_terminal=is_terminal,
            )
        )
    return states


# ---------------------------------------------------------------------------
# Step 2: Infer transitions
# ---------------------------------------------------------------------------


def _infer_transitions(states: list[State], steps: list[str]) -> list[Transition]:
    """
    Build transitions from step ordering.

    Sequential by default. Parenthetical hints like "(if X needed)" create
    skip-ahead branches from the previous state to the state after.
    """
    transitions: list[Transition] = []
    for i in range(len(states) - 1):
        transitions.append(
            Transition(from_state=states[i].name, to_state=states[i + 1].name)
        )

    # Detect conditional steps via parenthetical hints and add skip transitions
    for i, step in enumerate(steps):
        match = _PAREN_HINT.search(step)
        if match and i > 0 and i < len(states) - 1:
            # Previous state can skip the conditional state
            skip = Transition(
                from_state=states[i - 1].name,
                to_state=states[i + 1].name,
                description=f"skip conditional: {step}",
            )
            # Avoid duplicate
            existing = {(t.from_state, t.to_state) for t in transitions}
            if (skip.from_state, skip.to_state) not in existing:
                transitions.append(skip)

    return transitions


# ---------------------------------------------------------------------------
# Step 3: Parse rules → Constraints
# ---------------------------------------------------------------------------


def _parse_rules(
    rules: list[str], states: list[State]
) -> tuple[list[Constraint], list[State]]:
    """
    Parse natural language rules into Constraint objects.

    Returns constraints and any hidden states generated for NEVER conceptual targets.
    """
    constraints: list[Constraint] = []
    hidden_states: list[State] = []
    state_names = {s.name for s in states}

    for rule in rules:
        result = _parse_single_rule(rule, state_names)
        if result is None:
            logger.warning("Could not parse rule: %s", rule)
            continue
        constraint, hidden = result
        constraints.append(constraint)
        if hidden is not None:
            hidden_states.append(hidden)
            state_names.add(hidden.name)

    return constraints, hidden_states


def _parse_single_rule(
    rule: str, state_names: set[str]
) -> tuple[Constraint, State | None] | None:
    """
    Parse a single rule string into a Constraint and optional hidden State.

    Patterns recognized:
    - "X before Y" → PRECEDENCE (target=X, trigger=Y)
    - "never X" → NEVER (target=X)
    - "must eventually X" / "must reach X" → EVENTUALLY (target=X)
    - "if X then Y" → RESPONSE (trigger=X, target=Y)
    """
    normalized = rule.strip().lower()
    hidden: State | None = None

    # --- RESPONSE: "if X then Y" ---
    if_then = re.match(r"if\s+(.+?)\s+then\s+(?:must\s+)?(.+)", normalized)
    if if_then:
        trigger_text = _strip_filler(if_then.group(1))
        target_text = _strip_filler(if_then.group(2))
        trigger_slug = _resolve_state(slugify(trigger_text), state_names)
        target_slug = _resolve_state(slugify(target_text), state_names)
        return (
            Constraint(
                name=f"if_{trigger_slug}_then_{target_slug}",
                description=rule,
                type=ConstraintType.RESPONSE,
                trigger=trigger_slug,
                target=target_slug,
                message=f"Policy: {rule}",
            ),
            None,
        )

    # --- PRECEDENCE: "X before Y" ---
    before_match = re.search(r"(.+?)\s+before\s+(.+)", normalized)
    if before_match:
        target_text = _strip_filler(before_match.group(1))
        trigger_text = _strip_filler(before_match.group(2))
        target_slug = _resolve_state(slugify(target_text), state_names)
        trigger_slug = _resolve_state(slugify(trigger_text), state_names)
        return (
            Constraint(
                name=f"{target_slug}_before_{trigger_slug}",
                description=rule,
                type=ConstraintType.PRECEDENCE,
                trigger=trigger_slug,
                target=target_slug,
                message=f"Policy: {rule}",
            ),
            None,
        )

    # --- NEVER: "never X" ---
    never_match = re.match(r"never\s+(.+)", normalized)
    if never_match:
        target_text = _strip_filler(never_match.group(1))
        target_slug = slugify(target_text)
        # For NEVER, only use exact/substring match — not word overlap.
        # NEVER targets are often conceptual states that should become hidden states.
        resolved = _resolve_state(target_slug, state_names, strict=True)

        # Generate hidden state only if target doesn't match any existing state
        if resolved == target_slug and target_slug not in state_names:
            hidden = State(
                name=target_slug,
                description=target_text,
                is_initial=False,
                is_terminal=False,
                is_error=True,
            )
        else:
            target_slug = resolved

        return (
            Constraint(
                name=f"never_{target_slug}",
                description=rule,
                type=ConstraintType.NEVER,
                target=target_slug,
                message=f"Policy: {rule}",
            ),
            hidden,
        )

    # --- EVENTUALLY: "must eventually X" / "must reach X" ---
    eventually_match = re.match(r"must\s+(?:eventually|reach)\s+(.+)", normalized)
    if eventually_match:
        target_text = _strip_filler(eventually_match.group(1))
        target_slug = _resolve_state(slugify(target_text), state_names)
        return (
            Constraint(
                name=f"must_{target_slug}",
                description=rule,
                type=ConstraintType.EVENTUALLY,
                target=target_slug,
                message=f"Policy: {rule}",
            ),
            None,
        )

    return None


# ---------------------------------------------------------------------------
# Step 4: Map tools to states
# ---------------------------------------------------------------------------


def _map_tools(
    states: list[State],
    tools: dict[str, list[str]] | None,
    steps: list[str],
) -> None:
    """
    Map tool keys to states via case-insensitive substring match.

    Matches tool map keys against step descriptions (with parentheticals stripped).
    """
    if not tools:
        return

    for tool_key, tool_names in tools.items():
        key_lower = tool_key.lower()
        for state, step in zip(states, steps):
            # Strip parentheticals from the step for matching
            step_clean = _PAREN_HINT.sub("", step).strip().lower()
            if key_lower in step_clean:
                if state.classification.tool_calls is None:
                    state.classification = ClassificationHint(
                        tool_calls=list(tool_names),
                        patterns=state.classification.patterns,
                        exemplars=state.classification.exemplars,
                        min_similarity=state.classification.min_similarity,
                    )
                else:
                    state.classification.tool_calls.extend(tool_names)


# ---------------------------------------------------------------------------
# Step 5: Generate classification hints
# ---------------------------------------------------------------------------


def _generate_hints(states: list[State]) -> None:
    """
    Generate regex patterns and exemplar strings from state descriptions.

    Populates classification.patterns and classification.exemplars for states
    that don't already have them.
    """
    for state in states:
        desc = state.description
        if not desc:
            continue

        # --- Patterns ---
        if state.classification.patterns is None:
            patterns = _generate_patterns(desc)
            if patterns:
                state.classification = ClassificationHint(
                    tool_calls=state.classification.tool_calls,
                    patterns=patterns,
                    exemplars=state.classification.exemplars,
                    min_similarity=state.classification.min_similarity,
                )

        # --- Exemplars ---
        if state.classification.exemplars is None:
            exemplars = _generate_exemplars(desc)
            if exemplars:
                state.classification = ClassificationHint(
                    tool_calls=state.classification.tool_calls,
                    patterns=state.classification.patterns,
                    exemplars=exemplars,
                    min_similarity=state.classification.min_similarity,
                )


def _generate_patterns(description: str) -> list[str]:
    """Generate regex patterns from a step/state description."""
    # Strip parentheticals
    clean = _PAREN_HINT.sub("", description).strip()
    # Extract key words (3+ chars, not stopwords)
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "their", "them"}
    words = [w for w in re.findall(r"[a-zA-Z]+", clean) if len(w) >= 3 and w.lower() not in stopwords]
    if not words:
        return []
    # Build a case-insensitive pattern from key phrases
    # Use the full phrase and individual significant words
    patterns: list[str] = []
    if len(words) >= 2:
        # Phrase pattern: key words joined by flexible whitespace
        phrase = r"\b" + r"\b.*?\b".join(re.escape(w) for w in words[:3]) + r"\b"
        patterns.append(f"(?i){phrase}")
    for w in words:
        if len(w) >= 5:  # Only longer words as standalone patterns
            patterns.append(f"(?i)\\b{re.escape(w)}\\b")
    return patterns


def _generate_exemplars(description: str) -> list[str]:
    """Generate exemplar phrases for embedding similarity."""
    # Strip parentheticals for the base exemplar
    clean = _PAREN_HINT.sub("", description).strip()
    exemplars = [clean]
    # Add a variant: "The agent should {description}"
    exemplars.append(f"The agent should {clean.lower()}")
    return exemplars


# ---------------------------------------------------------------------------
# FSMCompiler — PolicyCompiler wrapper for registry / engine integration
# ---------------------------------------------------------------------------


@register_compiler("fsm")
class FSMCompiler(PolicyCompiler):
    """
    Deterministic FSM compiler registered as a PolicyCompiler.

    Wraps :func:`compile_workflow` so the engine and compiler registry
    can use the standard PolicyCompiler interface.  No LLM calls are made.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        """Accept and ignore kwargs (model, api_key, base_url) for interface compat."""

    @property
    def engine_type(self) -> str:
        return "fsm"

    async def compile(
        self,
        natural_language: str,
        context: dict[str, Any] | None = None,
    ) -> CompilationResult:
        """
        Compile a simple config dict (or natural-language placeholder) to a WorkflowDefinition.

        If *context* contains a ``"simple_config"`` key whose value is a
        :class:`SimpleWorkflowConfig` (or a raw dict), the deterministic
        pipeline is used directly.  Otherwise the *natural_language* string
        is ignored and an error is returned — the new compiler does not call
        an LLM.
        """
        context = context or {}
        raw = context.get("simple_config")
        if raw is None:
            return CompilationResult.failure(
                errors=[
                    "FSMCompiler requires context['simple_config'] "
                    "(a SimpleWorkflowConfig or dict). LLM-based compilation "
                    "has been replaced by the deterministic pipeline."
                ]
            )

        try:
            if isinstance(raw, SimpleWorkflowConfig):
                config = raw
            else:
                config = SimpleWorkflowConfig(**raw)

            workflow = compile_workflow(config)
            return CompilationResult(
                success=True,
                config=workflow,
                metadata={
                    "source": config.name,
                    "state_count": len(workflow.states),
                    "constraint_count": len(workflow.constraints),
                },
            )
        except Exception as e:
            logger.exception("Deterministic compilation failed")
            return CompilationResult.failure(
                errors=[f"Compilation failed: {type(e).__name__}: {e}"]
            )

    def export(self, result: CompilationResult, output_path: Path) -> None:
        """Export WorkflowDefinition to YAML."""
        if not result.success:
            raise ValueError("Cannot export failed compilation result")

        workflow: WorkflowDefinition = result.config
        workflow_dict: dict[str, Any] = {
            "name": workflow.name,
        }
        if workflow.description:
            workflow_dict["description"] = workflow.description

        workflow_dict["states"] = []
        for state in workflow.states:
            sd: dict[str, Any] = {"name": state.name}
            if state.description:
                sd["description"] = state.description
            if state.is_initial:
                sd["is_initial"] = True
            if state.is_terminal:
                sd["is_terminal"] = True
            if state.is_error:
                sd["is_error"] = True

            cls: dict[str, Any] = {}
            if state.classification.tool_calls:
                cls["tool_calls"] = state.classification.tool_calls
            if state.classification.patterns:
                cls["patterns"] = state.classification.patterns
            if state.classification.exemplars:
                cls["exemplars"] = state.classification.exemplars
            if state.classification.min_similarity != 0.7:
                cls["min_similarity"] = state.classification.min_similarity
            if cls:
                sd["classification"] = cls
            workflow_dict["states"].append(sd)

        if workflow.transitions:
            workflow_dict["transitions"] = [
                {"from_state": t.from_state, "to_state": t.to_state}
                | ({"description": t.description} if t.description else {})
                for t in workflow.transitions
            ]

        if workflow.constraints:
            workflow_dict["constraints"] = []
            for c in workflow.constraints:
                cd: dict[str, Any] = {"name": c.name, "type": c.type.value}
                if c.description:
                    cd["description"] = c.description
                if c.trigger:
                    cd["trigger"] = c.trigger
                if c.target:
                    cd["target"] = c.target
                if c.message:
                    cd["message"] = c.message
                workflow_dict["constraints"].append(cd)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            yaml.dump(workflow_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Exported workflow to %s", output)
