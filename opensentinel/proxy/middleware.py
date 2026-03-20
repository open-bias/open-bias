"""
Middleware for extracting and propagating context through Open Sentinel.

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

import logging
import uuid
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header names checked for session identity, in priority order.
# Case-insensitive lookup is performed by _get_header().
# ---------------------------------------------------------------------------
_SESSION_HEADER_NAMES: list[str] = [
    "x-sentinel-session-id",
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
    3. ``metadata.session_id`` / ``metadata.sentinel_session_id``
    4. ``metadata.run_id`` (LangChain convention)
    5. ``user`` field (OpenAI convention)
    6. ``thread_id`` field (OpenAI Assistants convention)
    7. Random UUID (last resort — logged as warning)
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
          ``x-sentinel-session-id`` header or
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
                    return session_id

        # 1b. Check litellm_params metadata (Library mode)
        # Internal calls via LLMClient/litellm.acompletion pass context here
        litellm_params = data.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_meta = litellm_params.get("metadata")
            if isinstance(lp_meta, dict):
                sid = lp_meta.get("session_id")
                if sid is not None and str(sid) != "":
                    return str(sid)

        # 2. Check metadata fields
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("session_id", "sentinel_session_id", "run_id"):
                val = metadata.get(key)
                if val is not None and str(val) != "":
                    return str(val)

        # 3. Check user field (OpenAI pattern)
        user = data.get("user")
        if user is not None and str(user) != "":
            return f"user_{user}"

        # 4. Check for thread_id (OpenAI Assistants)
        thread_id = data.get("thread_id")
        if thread_id is not None and str(thread_id) != "":
            return str(thread_id)

        # 5. Last resort: random UUID
        generated = str(uuid.uuid4())
        logger.warning(
            "No session ID found in request headers or metadata. "
            "Generated fallback UUID: %s. "
            "Set 'x-sentinel-session-id' header or 'metadata.session_id' "
            "in the request body for reliable session tracking.",
            generated,
        )
        return generated
