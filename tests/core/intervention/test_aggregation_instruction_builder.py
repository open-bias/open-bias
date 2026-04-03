from openbias.core.intervention.pipelines.aggregation import ViolationAggregationStage
from openbias.core.intervention.pipelines.instruction_builder import (
    DeterministicRepairInstructionBuilder,
)


def test_aggregation_deduplicates_messages_and_preserves_provenance() -> None:
    stage = ViolationAggregationStage()
    records = [
        {
            "evaluator": "judge_a",
            "message": "Do not disclose private data",
            "metadata": {"violations": [{"severity": "error", "scope": "turn"}]},
        },
        {
            "evaluator": "judge_b",
            "message": "Do not disclose private data",
            "metadata": {"violations": [{"severity": "warning", "scope": "turn"}]},
        },
        {
            "evaluator": "judge_c",
            "message": "Remove identifying details",
            "metadata": {"violations": [{"severity": "error", "scope": "turn"}]},
        },
    ]

    aggregated = stage.aggregate(records=records, mode="sync")

    assert len(aggregated.source_violations) == 2
    assert "Do not disclose private data" in aggregated.merged_violation_summary
    assert "Remove identifying details" in aggregated.merged_violation_summary
    assert set(aggregated.evaluators) == {"judge_a", "judge_b", "judge_c"}


def test_deterministic_builder_returns_stable_payload() -> None:
    stage = ViolationAggregationStage()
    builder = DeterministicRepairInstructionBuilder()
    records = [
        {
            "evaluator": "judge_a",
            "message": "Stay grounded in provided facts",
            "metadata": {"violations": [{"severity": "error", "scope": "turn"}]},
        }
    ]
    aggregated = stage.aggregate(records=records, mode="async")

    payload = builder.build(aggregated=aggregated, recent_messages=["User asked for summary"])

    assert payload.mode == "async"
    assert payload.source_violations[0]["message"] == "Stay grounded in provided facts"
    assert "Stay grounded in provided facts" in payload.sync_repair_instruction
    assert payload.async_guidance is not None
    assert "The previous response violated policy." in payload.async_guidance
    assert "I think I made a mistake before; here's what I mean:" in payload.async_guidance
    assert "Here's the response to your current message:" in payload.async_guidance
    assert payload.cleanup_rules
