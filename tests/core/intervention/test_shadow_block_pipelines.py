"""Direct contract tests for shadow/block action pipelines."""

from openbias.core.intervention.pipelines.base import (
    PostCallPipelineContext,
    PreCallPipelineContext,
)
from openbias.core.intervention.pipelines.block import BlockPipeline
from openbias.core.intervention.pipelines.shadow import ShadowPipeline


def _pre_context(message: str | None = "blocked") -> PreCallPipelineContext:
    return PreCallPipelineContext(
        session_id="session-1",
        evaluator_name="policy-a",
        message=message,
        mapped_metadata={"k": "v"},
        modified_data={"messages": [{"role": "user", "content": "hi"}]},
        all_metadata={"results": [{"evaluator": "policy-a", "decision": "block"}]},
        default_strategy="user_message_inject",
        apply_intervention=lambda request, msg, strategy: request,
    )


def _post_context(message: str | None = "blocked") -> PostCallPipelineContext:
    return PostCallPipelineContext(
        session_id="session-1",
        evaluator_name="policy-a",
        message=message,
        mapped_metadata={"k": "v"},
        modified_data={"answer": "bad"},
        all_metadata={"results": [{"evaluator": "policy-a", "decision": "block"}]},
    )


class TestShadowPipeline:
    def test_handle_pre_call_returns_none(self):
        pipeline = ShadowPipeline()
        result = pipeline.handle_pre_call(_pre_context())
        assert result is None

    def test_handle_post_call_returns_none(self):
        pipeline = ShadowPipeline()
        result = pipeline.handle_post_call(_post_context())
        assert result is None


class TestBlockPipeline:
    def test_handle_pre_call_blocks_with_message_and_metadata(self):
        pipeline = BlockPipeline()
        context = _pre_context("pre blocked")

        result = pipeline.handle_pre_call(context)

        assert result is not None
        assert result.allowed is False
        assert result.user_message == "pre blocked"
        assert result.internal_metadata == context.all_metadata

    def test_handle_post_call_blocks_with_message_and_metadata(self):
        pipeline = BlockPipeline()
        context = _post_context("post blocked")

        result = pipeline.handle_post_call(context)

        assert result is not None
        assert result.allowed is False
        assert result.user_message == "post blocked"
        assert result.internal_metadata == context.all_metadata
