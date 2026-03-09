"""
Centralized logging configuration for the agent package.

Usage in any agent module:
    from agent.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

_configured = False


def _configure_once() -> None:
    """Apply basicConfig exactly once across all imports."""
    global _configured
    if _configured:
        return
    _configured = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring global config is applied."""
    _configure_once()
    return logging.getLogger(name)
