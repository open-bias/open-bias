import pytest
from unittest.mock import AsyncMock, patch
from opensentinel.policy.compiler.base import LLMPolicyCompiler, CompilationResult

class ConcreteCompiler(LLMPolicyCompiler):
    @property
    def engine_type(self):
        return "test"

    def export(self, result, output_path):
        pass

    def _build_compilation_prompt(self, natural_language, context=None):
        return f"Prompt: {natural_language}"

    def _parse_compilation_response(self, response, natural_language):
        return CompilationResult(success=True, config=response)

@pytest.fixture
def compiler():
    return ConcreteCompiler(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="http://test.url"
    )

@patch("litellm.acompletion")
async def test_call_llm_uses_litellm(mock_acompletion, compiler):
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="Hello"))]
    mock_acompletion.return_value = mock_response

    # Call the method
    response = await compiler._call_llm("Hello world")

    # Verify result
    assert response == "Hello"

    # Verify litellm call
    mock_acompletion.assert_called_once()
    call_kwargs = mock_acompletion.call_args.kwargs
    
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": compiler.system_prompt},
        {"role": "user", "content": "Hello world"}
    ]
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["base_url"] == "http://test.url"

@patch("litellm.acompletion")
async def test_call_llm_wraps_exceptions(mock_acompletion, compiler):
    mock_acompletion.side_effect = Exception("API Error")

    with pytest.raises(Exception):
        await compiler._call_llm("Hello")
