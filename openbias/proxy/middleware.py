"""
Middleware for extracting and propagating context through Open Bias.

Key responsibilities:
- Session ID extraction from various sources (headers, metadata, body fields)
- Workflow context extraction
- Request/response transformation

Session extraction is designed to work with:
- Direct HTTP headers (when called from FastAPI middleware)
- LiteLLM proxy callbacks (where HTTP headers are embedded in
  ``data["proxy_server_request"]["headers"]`` and ``data["metadata"]["headers"]``)
- OpenClaw and other agent frameworks that pass custom headers or metadata
"""

import hashlib
import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session ID validation — prevents log/OTEL injection via external input.
# Accepts only alphanumerics plus underscore, hyphen, and dot, 1–256 chars.
# ---------------------------------------------------------------------------
_VALID_SESSION_ID: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\-\.]{1,256}$")


def _validate_session_id(value: str) -> str | None:
    """Return *value* if it matches the allowed session ID pattern, else None.

    Invalid values are rejected with a warning to prevent log injection and
    OTEL span poisoning via newlines, null bytes, path traversal sequences,
    or oversized strings embedded in externally supplied session identifiers.
    """
    if _VALID_SESSION_ID.match(value):
        return value
    logger.warning(
        "Rejected session ID from external input — failed validation "
        "(contains disallowed characters or exceeds 256 chars): %r",
        value[:80],  # truncate to avoid log flooding on huge payloads
    )
    return None


# ---------------------------------------------------------------------------
# Header names checked for session identity, in priority order.
# Case-insensitive lookup is performed by _get_header().
# ---------------------------------------------------------------------------
_SESSION_HEADER_NAMES: list[str] = [
    "x-openbias-session-id",
    "x-session-id",
]

def _get_header(
    headers: dict[str, str],
    name: str,
) -> str | None:
    """Case-insensitive header lookup.

    HTTP headers are case-insensitive per RFC 7230.  LiteLLM sometimes
    stores them lower-cased, sometimes not — this helper normalises.
    """
    # Fast path: exact match (common when LiteLLM already lower-cased them)
    val = headers.get(name)
    if val is not None and val != "":
        return val
    # Slow path: case-insensitive scan
    name_lower = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lower and v is not None and v != "":
            return v
    return None

class SessionExtractor:
    """
    Extract session ID from LLM request data.

    Session IDs are used to:
    - Group related LLM calls together
    - Maintain workflow state across calls
    - Correlate traces in observability backends (Langfuse, Jaeger, etc.)

    Extraction priority (first match wins):
    1. Explicit ``headers`` parameter (direct HTTP header access)
    2. HTTP headers embedded by LiteLLM in the data dict:
       a. ``data["proxy_server_request"]["headers"]``
       b. ``data["metadata"]["headers"]``
    3. ``metadata.session_id`` / ``metadata.openbias_session_id``
    4. ``metadata.run_id`` (LangChain convention)
    5. ``user`` field (OpenAI convention)
    6. ``thread_id`` field (OpenAI Assistants convention)
    7. Hash of first message content (deterministic fallback)
    8. Random UUID (last resort — logged as warning)
    """

    @staticmethod
    def _resolve_headers(
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        """Resolve the best available HTTP headers dict.

        Priority:
        1. Explicitly passed ``headers`` (caller already has them).
        2. ``data["proxy_server_request"]["headers"]`` — set by LiteLLM proxy
           for every request through ``add_litellm_data_to_request()``.
        3. ``data["metadata"]["headers"]`` — also set by LiteLLM proxy
           (duplicate of #2 for guardrails access).

        Returns ``None`` if no headers can be found.
        """
        if headers:
            return headers

        # LiteLLM proxy embeds the original HTTP headers here:
        psr = data.get("proxy_server_request")
        if isinstance(psr, dict):
            psr_headers = psr.get("headers")
            if isinstance(psr_headers, dict) and psr_headers:
                return psr_headers

        # Fallback: metadata.headers (also set by LiteLLM)
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            meta_headers = metadata.get("headers")
            if isinstance(meta_headers, dict) and meta_headers:
                return meta_headers

        # LiteLLM library mode: litellm_params.metadata.headers
        litellm_params = data.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_meta = litellm_params.get("metadata")
            if isinstance(lp_meta, dict):
                lp_headers = lp_meta.get("headers")
                if isinstance(lp_headers, dict) and lp_headers:
                    return lp_headers

        return None

    @staticmethod
    def extract_session_id(
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> str:
        """
        Extract session ID from request data and/or headers.

        Works in all deployment modes:
        - **LiteLLM Proxy mode**: HTTP headers are automatically extracted
          from ``data["proxy_server_request"]["headers"]``.
        - **LiteLLM Library/Router mode**: Headers from
          ``data["litellm_params"]["metadata"]["headers"]``.
        - **Direct call**: Pass ``headers`` explicitly.
        - **OpenClaw / other frameworks**: Send
          ``x-openbias-session-id`` header or
          ``metadata.session_id`` in the request body.

        Args:
            data: Request data dict (messages, metadata, etc.)
            headers: Optional explicit HTTP headers (takes top priority)

        Returns:
            A deterministic session ID string.
        """
        resolved_headers = SessionExtractor._resolve_headers(data, headers)

        # 1. Check HTTP headers (case-insensitive)
        if resolved_headers:
            for header_name in _SESSION_HEADER_NAMES:
                session_id = _get_header(resolved_headers, header_name)
                if session_id is not None:
                    validated = _validate_session_id(session_id)
                    if validated is not None:
                        logger.debug("Session ID from header %s: %s", header_name, validated)
                        return validated

        # 1b. Check litellm_params metadata (Library mode)
        # Internal calls via LLMClient/litellm.acompletion pass context here
        litellm_params = data.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_meta = litellm_params.get("metadata")
            if isinstance(lp_meta, dict):
                sid = lp_meta.get("session_id")
                if sid is not None and str(sid) != "":
                    validated = _validate_session_id(str(sid))
                    if validated is not None:
                        logger.debug("Session ID from litellm_params metadata: %s", validated)
                        return validated

        # 2. Check metadata fields
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("session_id", "openbias_session_id", "run_id"):
                val = metadata.get(key)
                if val is not None and str(val) != "":
                    validated = _validate_session_id(str(val))
                    if validated is not None:
                        logger.debug("Session ID from metadata.%s: %s", key, validated)
                        return validated

        # 3. Check user field (OpenAI pattern)
        user = data.get("user")
        if user is not None and str(user) != "":
            validated = _validate_session_id(str(user))
            if validated is not None:
                logger.debug("Session ID from user field: user_%s", validated)
                return f"user_{validated}"

        # 4. Check for thread_id (OpenAI Assistants)
        thread_id = data.get("thread_id")
        if thread_id is not None and str(thread_id) != "":
            validated = _validate_session_id(str(thread_id))
            if validated is not None:
                logger.debug("Session ID from thread_id: %s", validated)
                return validated

        # 5. Hash of first message content (deterministic fallback)
        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                content = first_msg.get("content", "")
                if content:
                    if isinstance(content, str):
                        raw = content.encode()
                    else:
                        raw = json.dumps(content, sort_keys=True).encode()
                    msg_hash = hashlib.sha256(raw).hexdigest()[:16]
                    logger.debug("Session ID from message hash: msg_%s", msg_hash)
                    return f"msg_{msg_hash}"

        # 6. Last resort: random UUID
        generated = str(uuid.uuid4())
        logger.warning(
            "No session ID found in request headers or metadata. "
            "Generated fallback UUID: %s. "
            "Set 'x-openbias-session-id' header or 'metadata.session_id' "
            "in the request body for reliable session tracking.",
            generated,
        )
        return generated
