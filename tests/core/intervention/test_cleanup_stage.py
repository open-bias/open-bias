from openbias.core.intervention.pipelines.cleanup import ResponseCleanupStage


def test_cleanup_stage_strips_repair_instruction_block() -> None:
    stage = ResponseCleanupStage()
    text = (
        "Visible answer.\n"
        "[REPAIR-INSTRUCTION]\n"
        "internal-only guidance\n"
        "[END-REPAIR-INSTRUCTION]\n"
        "Final sentence."
    )

    cleaned = stage.clean_text(
        text,
        cleanup_rules=["[REPAIR-INSTRUCTION]", "[END-REPAIR-INSTRUCTION]"],
    )

    assert cleaned == "Visible answer.\n\nFinal sentence."


def test_cleanup_stage_strips_single_line_markers() -> None:
    stage = ResponseCleanupStage()
    text = (
        "Safe output.\n"
        "[System Note]: internal note\n"
        "[WORKFLOW GUIDANCE]: hidden\n"
        "Still visible."
    )

    cleaned = stage.clean_text(
        text,
        cleanup_rules=["[System Note]", "[WORKFLOW GUIDANCE]"],
    )

    assert cleaned == "Safe output.\nStill visible."

