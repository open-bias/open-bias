"""
Tests for opensentinel.proxy.middleware — session extraction, workflow context,
and response transformation.

Covers the critical scenario where HTTP headers arrive embedded inside
the LiteLLM data dict rather than as a separate parameter.
"""

import uuid

import pytest

from opensentinel.proxy.middleware import (
    SessionExtractor,
    _get_header,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------
def _litellm_proxy_data(
    *,
    headers: dict | None = None,
    metadata: dict | None = None,
    user: str | None = None,
    thread_id: str | None = None,
    messages: list | None = None,
) -> dict:
    """Build a data dict that looks like what LiteLLM proxy passes to callbacks."""
    data: dict = {}
    if messages is not None:
        data["messages"] = messages
    if user is not None:
        data["user"] = user
    if thread_id is not None:
        data["thread_id"] = thread_id
    if metadata is not None:
        data["metadata"] = metadata
    # Simulate LiteLLM's proxy_server_request injection
    if headers is not None:
        data["proxy_server_request"] = {
            "url": "http://localhost:4000/chat/completions",
            "method": "POST",
            "headers": headers,
            "body": {},
        }
    return data


# ===========================================================================
# _get_header — case-insensitive lookup
# ===========================================================================
class TestGetHeader:
    def test_exact_match(self):
        assert _get_header({"x-session-id": "abc"}, "x-session-id") == "abc"

    def test_case_insensitive(self):
        assert _get_header({"X-Session-Id": "abc"}, "x-session-id") == "abc"

    def test_missing_header(self):
        assert _get_header({"other": "val"}, "x-session-id") is None

    def test_empty_value_returns_none(self):
        assert _get_header({"x-session-id": ""}, "x-session-id") is None

    def test_empty_dict(self):
        assert _get_header({}, "x-session-id") is None


# ===========================================================================
# SessionExtractor._resolve_headers
# ===========================================================================
class TestResolveHeaders:
    def test_explicit_headers_take_priority(self):
        explicit = {"x-sentinel-session-id": "explicit"}
        data = _litellm_proxy_data(headers={"x-sentinel-session-id": "embedded"})
        resolved = SessionExtractor._resolve_headers(data, explicit)
        assert resolved is explicit

    def test_proxy_server_request_headers(self):
        data = _litellm_proxy_data(headers={"x-sentinel-session-id": "from-proxy"})
        resolved = SessionExtractor._resolve_headers(data, None)
        assert resolved == {"x-sentinel-session-id": "from-proxy"}

    def test_metadata_headers_fallback(self):
        data = {"metadata": {"headers": {"x-sentinel-session-id": "from-meta"}}}
        resolved = SessionExtractor._resolve_headers(data, None)
        assert resolved == {"x-sentinel-session-id": "from-meta"}

    def test_litellm_params_metadata_headers(self):
        data = {
            "litellm_params": {
                "metadata": {
                    "headers": {"x-sentinel-session-id": "from-lp"}
                }
            }
        }
        resolved = SessionExtractor._resolve_headers(data, None)
        assert resolved == {"x-sentinel-session-id": "from-lp"}

    def test_no_headers_returns_none(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        assert SessionExtractor._resolve_headers(data, None) is None

    def test_empty_proxy_server_request_headers(self):
        data = {"proxy_server_request": {"headers": {}}}
        assert SessionExtractor._resolve_headers(data, None) is None

    def test_non_dict_proxy_server_request(self):
        data = {"proxy_server_request": "not-a-dict"}
        assert SessionExtractor._resolve_headers(data, None) is None


# ===========================================================================
# SessionExtractor.extract_session_id — header-based extraction
# ===========================================================================
class TestExtractSessionIdFromHeaders:
    def test_explicit_header(self):
        data = {"messages": []}
        headers = {"x-sentinel-session-id": "sess-123"}
        assert SessionExtractor.extract_session_id(data, headers) == "sess-123"

    def test_x_session_id_header(self):
        data = {"messages": []}
        headers = {"x-session-id": "sess-456"}
        assert SessionExtractor.extract_session_id(data, headers) == "sess-456"

    def test_sentinel_header_takes_priority_over_x_session_id(self):
        data = {"messages": []}
        headers = {
            "x-sentinel-session-id": "sentinel",
            "x-session-id": "generic",
        }
        assert SessionExtractor.extract_session_id(data, headers) == "sentinel"

    def test_case_insensitive_header(self):
        data = {"messages": []}
        headers = {"X-Sentinel-Session-Id": "mixed-case"}
        assert SessionExtractor.extract_session_id(data, headers) == "mixed-case"

    def test_litellm_embedded_headers(self):
        """Core OpenClaw scenario: headers embedded by LiteLLM proxy."""
        data = _litellm_proxy_data(
            headers={"x-sentinel-session-id": "openclaw-session-42"}
        )
        # No explicit headers param — must pick from data dict
        assert SessionExtractor.extract_session_id(data) == "openclaw-session-42"

    def test_litellm_metadata_headers(self):
        """Fallback: headers in data["metadata"]["headers"]."""
        data = {"metadata": {"headers": {"x-session-id": "meta-sess"}}}
        assert SessionExtractor.extract_session_id(data) == "meta-sess"


# ===========================================================================
# SessionExtractor.extract_session_id — metadata-based extraction
# ===========================================================================
class TestExtractSessionIdFromMetadata:
    def test_session_id_in_metadata(self):
        data = {"metadata": {"session_id": "meta-123"}}
        assert SessionExtractor.extract_session_id(data) == "meta-123"

    def test_sentinel_session_id_in_metadata(self):
        data = {"metadata": {"sentinel_session_id": "pan-456"}}
        assert SessionExtractor.extract_session_id(data) == "pan-456"

    def test_run_id_in_metadata(self):
        data = {"metadata": {"run_id": "langchain-run-789"}}
        assert SessionExtractor.extract_session_id(data) == "langchain-run-789"

    def test_session_id_takes_priority_over_run_id(self):
        data = {"metadata": {"session_id": "sess", "run_id": "run"}}
        assert SessionExtractor.extract_session_id(data) == "sess"

    def test_header_takes_priority_over_metadata(self):
        data = _litellm_proxy_data(
            headers={"x-sentinel-session-id": "from-header"},
            metadata={"session_id": "from-meta"},
        )
        assert SessionExtractor.extract_session_id(data) == "from-header"


# ===========================================================================
# SessionExtractor.extract_session_id — body field extraction
# ===========================================================================
class TestExtractSessionIdFromBodyFields:
    def test_user_field(self):
        data = {"user": "alice"}
        assert SessionExtractor.extract_session_id(data) == "user_alice"

    def test_thread_id_field(self):
        data = {"thread_id": "thread_abc"}
        assert SessionExtractor.extract_session_id(data) == "thread_abc"

    def test_user_takes_priority_over_thread_id(self):
        data = {"user": "bob", "thread_id": "thread_xyz"}
        assert SessionExtractor.extract_session_id(data) == "user_bob"

    def test_metadata_takes_priority_over_user(self):
        data = {"user": "bob", "metadata": {"session_id": "explicit"}}
        assert SessionExtractor.extract_session_id(data) == "explicit"


# ===========================================================================
# SessionExtractor.extract_session_id — fallback UUID
# ===========================================================================
class TestExtractSessionIdFallback:
    def test_hash_of_first_message_fallback(self):
        """When no explicit session ID, hash of first message is used."""
        data = {"messages": [{"role": "user", "content": "hi"}]}
        result = SessionExtractor.extract_session_id(data)
        assert result.startswith("msg_")
        assert len(result) == 4 + 16  # "msg_" + 16-char hex

    def test_hash_of_first_message_is_deterministic(self):
        """Same first message content produces same session ID."""
        data1 = {"messages": [{"role": "user", "content": "hello world"}]}
        data2 = {"messages": [{"role": "user", "content": "hello world"}]}
        assert SessionExtractor.extract_session_id(data1) == SessionExtractor.extract_session_id(data2)

    def test_multimodal_content_does_not_crash(self):
        """Multimodal (list) content should hash via JSON, not crash on .encode()."""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ]
        }
        result = SessionExtractor.extract_session_id(data)
        assert result.startswith("msg_")
        assert len(result) == 4 + 16

    def test_multimodal_content_is_deterministic(self):
        """Same multimodal content produces same session ID."""
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]
        data1 = {"messages": [{"role": "user", "content": content}]}
        data2 = {"messages": [{"role": "user", "content": content}]}
        assert SessionExtractor.extract_session_id(data1) == SessionExtractor.extract_session_id(data2)

    def test_generates_uuid_when_nothing_available(self):
        data: dict = {}
        result = SessionExtractor.extract_session_id(data)
        # Should be a valid UUID
        uuid.UUID(result)  # raises ValueError if not valid

    def test_different_calls_produce_different_uuids(self):
        data: dict = {}
        id1 = SessionExtractor.extract_session_id(data)
        id2 = SessionExtractor.extract_session_id(data)
        assert id1 != id2

    def test_uuid_fallback_logs_warning(self, caplog):
        """Ensure a warning is logged when falling back to UUID."""
        import logging

        data: dict = {}
        with caplog.at_level(logging.WARNING, logger="opensentinel.proxy.middleware"):
            SessionExtractor.extract_session_id(data)
        assert "No session ID found" in caplog.text
        assert "x-sentinel-session-id" in caplog.text


# ===========================================================================
# SessionExtractor — multi-agent isolation scenarios
# ===========================================================================
class TestMultiAgentIsolation:
    """
    Simulate multiple concurrent agent sessions to verify that session IDs
    are correctly isolated per agent.
    """

    def test_different_headers_produce_different_sessions(self):
        agent_a = _litellm_proxy_data(
            headers={"x-sentinel-session-id": "agent-A-session"},
            messages=[{"role": "user", "content": "Hello from A"}],
        )
        agent_b = _litellm_proxy_data(
            headers={"x-sentinel-session-id": "agent-B-session"},
            messages=[{"role": "user", "content": "Hello from B"}],
        )
        assert SessionExtractor.extract_session_id(agent_a) == "agent-A-session"
        assert SessionExtractor.extract_session_id(agent_b) == "agent-B-session"

    def test_different_metadata_produce_different_sessions(self):
        agent_a = {"metadata": {"session_id": "meta-A"}}
        agent_b = {"metadata": {"session_id": "meta-B"}}
        assert SessionExtractor.extract_session_id(agent_a) == "meta-A"
        assert SessionExtractor.extract_session_id(agent_b) == "meta-B"

    def test_mixed_sources_still_isolate(self):
        """Agent A uses header, Agent B uses metadata."""
        agent_a = _litellm_proxy_data(
            headers={"x-sentinel-session-id": "header-sess"},
        )
        agent_b = {"metadata": {"session_id": "meta-sess"}}
        assert SessionExtractor.extract_session_id(agent_a) == "header-sess"
        assert SessionExtractor.extract_session_id(agent_b) == "meta-sess"


