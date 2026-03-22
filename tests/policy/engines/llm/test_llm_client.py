"""
Tests for LLMClient.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from opensentinel.policy.engines.llm.llm_client import LLMClient, LLMClientError


@pytest.fixture
def client():
    """Create an LLMClient."""
    return LLMClient(model="gpt-4o-mini", temperature=0.0)


def test_default_model_selection():
    """Test that LLMClient with model=None stores None (no auto-detect)."""
    client = LLMClient()
    assert client.model is None


class TestJSONParsing:
    """Tests for JSON parsing."""

    async def test_successful_json_parse(self, client):
        """Test successful JSON parsing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_response.usage = MagicMock(total_tokens=100)
        
        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            result = await client.complete_json("System", "User")
        
        assert result == {"key": "value"}

    async def test_markdown_fence_stripping(self, client):
        """Test that markdown fences are stripped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_response.usage = MagicMock(total_tokens=100)
        
        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            result = await client.complete_json("System", "User")
        
        assert result == {"key": "value"}

    async def test_plain_fence_stripping(self, client):
        """Test that plain code fences are stripped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```\n{"key": "value"}\n```'
        mock_response.usage = MagicMock(total_tokens=100)
        
        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            result = await client.complete_json("System", "User")
        
        assert result == {"key": "value"}

    async def test_array_response(self, client):
        """Test parsing JSON array response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"a": 1}, {"b": 2}]'
        mock_response.usage = MagicMock(total_tokens=100)
        
        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            result = await client.complete_json("System", "User")
        
        assert result == [{"a": 1}, {"b": 2}]


class TestErrorHandling:
    """Tests for error handling."""

    async def test_empty_response_error(self, client):
        """Test error after all retries on empty response."""
        client = LLMClient(model="gpt-4o-mini", max_retries=1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.usage = MagicMock(total_tokens=0)

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            with pytest.raises(LLMClientError, match="failed after 2 attempts"):
                await client.complete_json("System", "User")

    async def test_invalid_json_error(self, client):
        """Test error after all retries on invalid JSON."""
        client = LLMClient(model="gpt-4o-mini", max_retries=1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_response.usage = MagicMock(total_tokens=50)

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            with pytest.raises(LLMClientError, match="failed after 2 attempts"):
                await client.complete_json("System", "User")

    async def test_retry_on_empty_response(self, client):
        """Test that empty response triggers retry and succeeds on next attempt."""
        client = LLMClient(model="gpt-4o-mini", max_retries=2)

        empty_response = MagicMock()
        empty_response.choices = [MagicMock()]
        empty_response.choices[0].message.content = ""
        empty_response.usage = MagicMock(total_tokens=0)

        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message.content = '{"recovered": true}'
        good_response.usage = MagicMock(total_tokens=80)

        responses = iter([empty_response, good_response])

        async def side_effect(*args, **kwargs):
            return next(responses)

        with patch("litellm.acompletion", AsyncMock(side_effect=side_effect)):
            result = await client.complete_json("System", "User")

        assert result == {"recovered": True}

    async def test_retry_on_invalid_json(self, client):
        """Test that invalid JSON triggers retry and succeeds on next attempt."""
        client = LLMClient(model="gpt-4o-mini", max_retries=2)

        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "not valid json"
        bad_response.usage = MagicMock(total_tokens=50)

        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message.content = '{"recovered": true}'
        good_response.usage = MagicMock(total_tokens=80)

        responses = iter([bad_response, good_response])

        async def side_effect(*args, **kwargs):
            return next(responses)

        with patch("litellm.acompletion", AsyncMock(side_effect=side_effect)):
            result = await client.complete_json("System", "User")

        assert result == {"recovered": True}

    async def test_no_model_configured_skips_retry(self):
        """Test that missing model raises immediately without entering retry loop."""
        client = LLMClient(model=None, max_retries=2)
        with patch("litellm.acompletion", AsyncMock()) as mock_acompletion:
            with pytest.raises(LLMClientError, match="No LLM model configured"):
                await client.complete_json("System", "User")
        mock_acompletion.assert_not_called()

    async def test_retry_on_failure(self, client):
        """Test retries on transient failures."""
        client = LLMClient(model="gpt-4o-mini", max_retries=2)
        
        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"success": true}'
        mock_response.usage = MagicMock(total_tokens=100)
        
        call_count = 0
        
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Transient error")
            return mock_response
        
        with patch("litellm.acompletion", AsyncMock(side_effect=side_effect)):
            result = await client.complete_json("System", "User")
        
        assert result == {"success": True}
        assert call_count == 3

    async def test_exhausted_retries(self, client):
        """Test error after exhausting retries."""
        client = LLMClient(model="gpt-4o-mini", max_retries=1)
        
        with patch(
            "litellm.acompletion",
            AsyncMock(side_effect=Exception("Persistent error")),
        ):
            with pytest.raises(LLMClientError, match="failed after 2 attempts"):
                await client.complete_json("System", "User")


class TestTokenTracking:
    """Tests for token usage tracking."""

    async def test_token_tracking(self, client):
        """Test that tokens are tracked across calls."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_response.usage = MagicMock(total_tokens=150)
        
        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            await client.complete_json("System", "User1")
            await client.complete_json("System", "User2")
        
        assert client.total_tokens_used == 300

    def test_reset_token_count(self, client):
        """Test resetting token count."""
        client._total_tokens_used = 500
        client.reset_token_count()
        assert client.total_tokens_used == 0
