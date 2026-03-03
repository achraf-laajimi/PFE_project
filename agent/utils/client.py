"""
MCP Client wrapper — supports both in-process (FastMCP instance)
and remote (HTTP URL) transports.

Usage in agent/server.py:
    from mcp_server.server import mcp          # FastMCP instance
    client = MCPClient(server_instance=mcp)    # in-process, no network
    await client.connect()
    tools = await client.list_tools()

Sampling support:
    When an MCP server calls ctx.sample(), the request travels back to this
    client via the sampling handler.  The handler forwards it to the agent's
    existing LLMService so the server stays model-agnostic.
"""

import asyncio
from typing import Any, Dict, Optional

from fastmcp import Client
from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams

from agent.utils.logger import get_logger

logger = get_logger(__name__)


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
      - Support for local FastMCP instance (in-process) or remote URL
      - Sampling handler wired to the agent's LLMService
    """

    def __init__(
        self,
        server_instance=None,
        url: Optional[str] = None,
        llm=None,
    ):
        # Build the sampling handler if an LLM service was provided
        sampling_handler = _make_sampling_handler(llm) if llm is not None else None

        if server_instance is not None:
            self._client = Client(
                server_instance,
                sampling_handler=sampling_handler,
            )
            self._mode = "in-process"
        else:
            self._client = Client(
                url or "http://localhost:8000/mcp",
                sampling_handler=sampling_handler,
            )
            self._mode = "remote"

        self._connected = False

    # ── Lifecycle ────────────────────────────────────────────

    async def connect(self):
        """Open persistent MCP session."""
        if not self._connected:
            await self._client.__aenter__()
            self._connected = True
            logger.info(f"MCPClient connected ({self._mode})")

    async def disconnect(self):
        """Close the MCP session."""
        if self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            logger.info("MCPClient disconnected")

    async def _ensure_connected(self):
        if not self._connected:
            await self.connect()

    # ── MCP operations ───────────────────────────────────────

    async def list_tools(self):
        """Return the list of Tool objects from the server."""
        await self._ensure_connected()
        return await self._client.list_tools()

    async def call_tool(self, name: str, params: Dict[str, Any] = None):
        """
        Call a tool by name.

        Returns a CallToolResult with .content list
        (compatible with what DAGExecutor expects).
        """
        await self._ensure_connected()
        return await self._client.call_tool(name, params or {})


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