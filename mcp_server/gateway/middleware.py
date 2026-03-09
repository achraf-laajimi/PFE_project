"""Gateway checks executed before tool business logic."""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional
from fastmcp.server.dependencies import get_http_headers
from pydantic import BaseModel, Field, ValidationError
from mcp_server.gateway.auth import authenticate_headers
from mcp_server.gateway.policy import is_tool_allowed
from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)

_VALID_SUBJECT_IDS = {3, 4, 10, 11, 15}


class StudentScopedRequest(BaseModel):
	student_id: str


class SubjectScopedRequest(BaseModel):
	student_id: str
	subject_id: int


class RecentActivityRequest(BaseModel):
	student_id: str
	limit: int = Field(default=10, ge=1, le=50)


def _audit(event: str, *, agent: Optional[str], tool: str, status: str, detail: str = "") -> None:
	payload = {
		"ts": datetime.now(timezone.utc).isoformat(),
		"event": event,
		"agent": agent,
		"tool": tool,
		"status": status,
		"detail": detail,
	}
	logger.info("AUDIT %s", json.dumps(payload, ensure_ascii=True))


def _validate_tool_input(tool_name: str, params: dict[str, Any]) -> Optional[str]:
	try:
		if tool_name in {
			"get_student_identity",
			"compare_subjects",
			"get_study_habits",
		}:
			StudentScopedRequest(**params)
			return None

		if tool_name in {"get_subject_performance", "get_subject_roadmap"}:
			validated = SubjectScopedRequest(**params)
			if validated.subject_id not in _VALID_SUBJECT_IDS:
				return "invalid subject_id"
			return None

		if tool_name == "get_recent_activity":
			RecentActivityRequest(**params)
			return None

		return None
	except ValidationError:
		return "invalid input parameters"


def enforce_tool_access(tool_name: str, params: dict[str, Any]) -> Optional[dict]:
	"""Run auth + policy + validation checks before tool logic executes."""
	headers = get_http_headers(include_all=False)

	agent, reason = authenticate_headers(headers)
	if not agent:
		_audit("tool_auth", agent=None, tool=tool_name, status="denied", detail=reason)
		return {"error": "Unauthorized: missing or invalid API key."}

	if not is_tool_allowed(agent, tool_name, action="execute"):
		_audit(
			"tool_authorization",
			agent=agent,
			tool=tool_name,
			status="denied",
			detail="forbidden",
		)
		return {"error": "Forbidden: agent is not allowed to execute this tool."}

	validation_error = _validate_tool_input(tool_name, params)
	if validation_error:
		_audit(
			"tool_validation",
			agent=agent,
			tool=tool_name,
			status="denied",
			detail=validation_error,
		)
		return {"error": f"Validation error: {validation_error}."}

	_audit("tool_access", agent=agent, tool=tool_name, status="allowed")
	return None

