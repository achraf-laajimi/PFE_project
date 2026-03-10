"""MCP Client wrapper using HTTP transport with short-lived JWT bearer auth."""

import asyncio
import os
import secrets
import time
from typing import Any, Dict, Optional

from authlib.jose import JsonWebToken
from fastmcp import Client
from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams

from parent_agent.utils.logger import get_logger

logger = get_logger(__name__)

_JWT = JsonWebToken(["HS256"])
_RECOVERABLE_AUTH_ERRORS = (
    "status=401",
    "unauthorized",
    "invalid_token",
    "missing_or_invalid_bearer_token",
    "client is not connected",
    "not connected",
)


# ── Sampling handler factory ──────────────────────────────────────────────────

def _make_sampling_handler(llm):
    """
    Return an async sampling handler that forwards MCP sampling requests to
    the supplied LLMService instance.

    The handler is called by FastMCP whenever an MCP server tool calls
    ctx.sample(). It converts the SamplingMessage list into an OpenAI-style
    conversation and returns the generated text.
    """

    async def _handler(
        messages: list[SamplingMessage],
        params: SamplingParams,
        context: RequestContext,
    ) -> str:
        # Extract conversation turns
        conversation: list[str] = []
        for msg in messages:
            content = (
                msg.content.text
                if hasattr(msg.content, "text")
                else str(msg.content)
            )
            conversation.append(f"{msg.role}: {content}")

        user_text = "\n".join(conversation)
        system_text = params.systemPrompt or "You are a helpful assistant."

        logger.debug(
            f"[sampling] Forwarding request to LLM "
            f"({len(messages)} message(s), system={bool(params.systemPrompt)})"
        )

        # LLMService.generate() is synchronous — run it off the event loop
        result = await asyncio.to_thread(
            llm.generate,
            user_text,
            system=system_text,
            temperature=params.temperature if params.temperature is not None else 0.3,
            max_tokens=params.maxTokens if params.maxTokens is not None else 1000,
        )

        logger.debug(f"[sampling] LLM produced {len(result)} chars")
        return result

    return _handler


# ── MCPClient ─────────────────────────────────────────────────────────────────

class MCPClient:
    """
    Thin wrapper around fastmcp.Client that provides:
      - Persistent session (connect once, call many times)
      - Auto-connect on first call if connect() was not called
      - HTTP transport only (no in-process bypass)
            - Sampling handler wired to the agent's LLMService
            - JWT bearer auth with expiration (claims-based identity)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        llm=None,
    ):
        # Build the sampling handler if an LLM service was provided.
        sampling_handler = _make_sampling_handler(llm) if llm is not None else None

        mcp_url = url or os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
        self._jwt_secret = (jwt_secret or os.getenv("MCP_JWT_SECRET") or "").strip()
        if not self._jwt_secret:
            raise ValueError("Missing MCP_JWT_SECRET.")

        self._issuer = os.getenv("MCP_JWT_ISSUER", "").strip()
        self._audience = os.getenv("MCP_JWT_AUDIENCE", "").strip()
        self._agent_id = os.getenv("MCP_AGENT_ID", "").strip()
        self._parent_id = os.getenv("MCP_PARENT_ID", "").strip() or None
        self._ttl_seconds = int(os.getenv("MCP_JWT_TTL_SECONDS", "300"))
        self._scope = os.getenv("MCP_JWT_SCOPE", "").strip()

        self._client = Client(
            mcp_url,
            sampling_handler=sampling_handler,
        )
        self._mode = "remote"

        # Force Bearer auth header on every MCP HTTP call.
        transport_headers = getattr(self._client.transport, "headers", None)
        if transport_headers is None:
            raise RuntimeError("MCP transport does not support custom headers")
        self._transport_headers = transport_headers

        self._current_token: Optional[str] = None
        self._token_expiry: float = 0.0

        self._connected = False
        self._conn_lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────

    def _mint_jwt(self) -> tuple[str, float]:
        """Create a short-lived HS256 JWT used by MCP gateway auth."""
        now = int(time.time())
        exp = now + max(60, self._ttl_seconds)
        payload = {
            "sub": self._parent_id,
            "client_id": self._agent_id,
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "exp": exp,
            "jti": secrets.token_hex(8),
            "scope": self._scope,
        }
        logger.debug(
            "mint_jwt | client_id=%s sub=%s iat=%s exp=%s jti=%s scope=%s",
            payload.get("client_id"),
            payload.get("sub"),
            payload.get("iat"),
            payload.get("exp"),
            payload.get("jti"),
            payload.get("scope"),
        )

        token_bytes = _JWT.encode({"alg": "HS256"}, payload, self._jwt_secret)
        return token_bytes.decode("utf-8"), float(exp)

    def _ensure_bearer_header(self) -> None:
        """Refresh Authorization header if token is missing or near expiry."""
        refresh_margin_seconds = 30
        now = time.time()
        if (not self._current_token) or (now >= (self._token_expiry - refresh_margin_seconds)):
            self._current_token, self._token_expiry = self._mint_jwt()
            self._transport_headers["authorization"] = f"Bearer {self._current_token}"

    async def connect(self):
        """Open persistent MCP session."""
        async with self._conn_lock:
            if not self._connected:
                self._ensure_bearer_header()
                await self._client.__aenter__()
                self._connected = True
                logger.info(f"MCPClient connected ({self._mode})")

    async def disconnect(self):
        """Close the MCP session."""
        async with self._conn_lock:
            if self._connected:
                # Refresh token/header right before lifecycle DELETE to reduce 401s at shutdown.
                self._ensure_bearer_header()
                await self._client.__aexit__(None, None, None)
                self._connected = False
                logger.info("MCPClient disconnected")

    async def _ensure_connected(self):
        if not self._connected:
            await self.connect()

    async def _force_reconnect(self) -> None:
        """Reset transport session and auth header after auth/disconnect failures."""
        async with self._conn_lock:
            if self._connected:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:
                    pass
                self._connected = False

            self._current_token = None
            self._token_expiry = 0.0
            self._ensure_bearer_header()
            await self._client.__aenter__()
            self._connected = True
            logger.info("MCPClient reconnected after auth/session failure")

    @staticmethod
    def _is_recoverable_auth_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(token in msg for token in _RECOVERABLE_AUTH_ERRORS)

    async def _call_with_reauth(self, op_name: str, fn, *args, **kwargs):
        await self._ensure_connected()
        self._ensure_bearer_header()
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if not self._is_recoverable_auth_error(e):
                raise
            logger.warning("%s failed (%s); refreshing token + reconnect", op_name, e)
            await self._force_reconnect()
            self._ensure_bearer_header()
            return await fn(*args, **kwargs)

    # ── MCP operations ───────────────────────────────────────

    async def list_tools(self):
        """Return the list of Tool objects from the server."""
        return await self._call_with_reauth("list_tools", self._client.list_tools)

    async def call_tool(self, name: str, params: Dict[str, Any] = None):
        """
        Call a tool by name.

        Returns a CallToolResult with .content list
        (compatible with what DAGExecutor expects).
        """
        return await self._call_with_reauth(
            f"call_tool:{name}",
            self._client.call_tool,
            name,
            params or {},
        )


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    async def main():
        client = MCPClient(url="http://localhost:8000/mcp")
        await client.connect()
        try:
            tools = await client.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")
        finally:
            await client.disconnect()

    asyncio.run(main())