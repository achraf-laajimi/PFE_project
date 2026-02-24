"""
MCP Client wrapper — supports both in-process (FastMCP instance)
and remote (HTTP URL) transports.

Usage in agent/server.py:
    from mcp_server.server import mcp          # FastMCP instance
    client = MCPClient(server_instance=mcp)    # in-process, no network
    await client.connect()
    tools = await client.list_tools()
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastmcp import Client

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Thin wrapper around fastmcp.Client that provides:
      - Persistent session (connect once, call many times)
      - Auto-connect on first call if connect() was not called
      - Support for local FastMCP instance (in-process) or remote URL
    """

    def __init__(
        self,
        server_instance=None,
        url: Optional[str] = None,
    ):
        if server_instance is not None:
            self._client = Client(server_instance)
            self._mode = "in-process"
        else:
            self._client = Client(url or "http://localhost:8000/mcp")
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