"""Authentication helpers for MCP gateway API-key auth."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Mapping, Optional, Tuple

from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_api_key_map() -> dict[str, str]:
	"""Load API key -> agent mapping from environment only."""
	raw_json = os.getenv("MCP_API_KEYS_JSON", "").strip()
	if raw_json:
		try:
			parsed = json.loads(raw_json)
			if isinstance(parsed, dict):
				cleaned = {
					str(key).strip(): str(value).strip()
					for key, value in parsed.items()
					if str(key).strip() and str(value).strip()
				}
				if cleaned:
					return cleaned
		except Exception as exc:  # noqa: BLE001
			logger.warning("Invalid MCP_API_KEYS_JSON: %s", exc)

	parent_key = os.getenv("PARENT_AGENT_API_KEY", "").strip()
	exam_key = os.getenv("EXAM_AGENT_API_KEY", "").strip()
	from_env = {}
	if parent_key:
		from_env[parent_key] = "parent_agent"
	if exam_key:
		from_env[exam_key] = "exam_agent"
	if from_env:
		return from_env

	logger.error("No API keys configured. Set MCP_API_KEYS_JSON or PARENT_AGENT_API_KEY/EXAM_AGENT_API_KEY.")
	return {}


def _extract_api_key(headers: Mapping[str, str]) -> Optional[str]:
	"""Extract API key from X-API-KEY"""
	key = (headers.get("x-api-key") or "").strip()
	if key:
		return key
	return None


def authenticate_headers(headers: Mapping[str, str]) -> Tuple[Optional[str], str]:
	"""Authenticate request headers and return (agent_name, reason)."""
	key = _extract_api_key(headers)
	if not key:
		return None, "missing_api_key"

	agent = load_api_key_map().get(key)
	if not agent:
		return None, "invalid_api_key"

	return agent, "ok"

