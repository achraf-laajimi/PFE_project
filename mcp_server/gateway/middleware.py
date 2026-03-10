"""Gateway checks executed before tool business logic."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field, ValidationError
from mcp_server.gateway.auth import authenticate_request
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


def _identity_audit_kwargs(identity) -> dict[str, Optional[str]]:
	"""Shared audit fields derived from authenticated identity."""
	return {
		"parent_id": identity.parent_id,
		"client_id": identity.client_id,
		"token_id": identity.token_id,
	}


def _audit(
	event: str,
	*,
	agent: Optional[str],
	tool: str,
	status: str,
	detail: str = "",
	parent_id: Optional[str] = None,
	client_id: Optional[str] = None,
	token_id: Optional[str] = None,
) -> None:
	payload = {
		"ts": datetime.now(timezone.utc).isoformat(),
		"event": event,
		"agent": agent,
		"parent_id": parent_id,
		"client_id": client_id,
		"token_id": token_id,
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
	identity, reason = authenticate_request()
	if identity is None:
		_audit("tool_auth", agent=None, tool=tool_name, status="denied", detail=reason)
		return {"error": f"Unauthorized: {reason}"}

	requested_id = str(params.get("student_id") or "").strip()
	parent_id = identity.parent_id or ""

	if not is_tool_allowed(
		identity.agent,
		tool_name,
		action="execute",
		parent_id=parent_id,
		requested_id=requested_id,
	):
		detail = "forbidden"
		if requested_id in {"CURRENT_STUDENT", "UNKNOWN", ""}:
			detail = "invalid student identity resolution"
		_audit(
			"tool_authorization",
			agent=identity.agent,
			tool=tool_name,
			status="denied",
			detail=detail,
			**_identity_audit_kwargs(identity),
		)
		if requested_id in {"CURRENT_STUDENT", "UNKNOWN", ""}:
			return {"error": "Invalid input: Please provide a specific numeric student_id."}
		return {"error": "Forbidden: Access denied for this student identity."}

	validation_error = _validate_tool_input(tool_name, params)
	if validation_error:
		_audit(
			"tool_validation",
			agent=identity.agent,
			tool=tool_name,
			status="denied",
			detail=validation_error,
			**_identity_audit_kwargs(identity),
		)
		return {"error": f"Validation error: {validation_error}."}

	_audit(
		"tool_access",
		agent=identity.agent,
		tool=tool_name,
		status="allowed",
		**_identity_audit_kwargs(identity),
	)
	return None

