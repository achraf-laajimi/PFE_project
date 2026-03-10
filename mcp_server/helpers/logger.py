"""
Centralized logging configuration for the mcp_server package.

Usage in any mcp_server module:
    from mcp_server.helpers.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

_configured = False


def _configure_once() -> None:
    """Apply basicConfig exactly once across all imports.

    The log level can be set to DEBUG by setting one of the following
    environment variables: `MCP_SERVER_DEBUG`, `MCP_DEBUG`, or
    `FASTMCP_DEBUG` to `1`, `true` or `yes`.
    """
    global _configured
    if _configured:
        return
    _configured = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Keep focused debug visibility for security/auth flow only.
    logging.getLogger("mcp_server.gateway").setLevel(logging.DEBUG)
    logging.getLogger("mcp_server.gateway.auth").setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring global config is applied."""
    _configure_once()
    return logging.getLogger(name)