"""JWT authentication helpers for MCP gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.dependencies import get_access_token
import redis
from mcp_server.helpers.logger import get_logger
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_jwt_verifier() -> JWTVerifier:
	"""Create and cache JWT verifier using FastMCP-compatible env settings."""
	secret = os.getenv("MCP_JWT_SECRET", "").strip()
	if not secret:
		raise ValueError("Missing JWT secret key.")

	algorithm = os.getenv("MCP_JWT_ALGORITHM", "HS256").strip()
	issuer = os.getenv("MCP_JWT_ISSUER", "").strip()
	audience = os.getenv("MCP_JWT_AUDIENCE", "").strip()
	required_scopes = _required_scopes()

	return JWTVerifier(
		public_key=secret,
		algorithm=algorithm,
		issuer=issuer,
		audience=audience,
		required_scopes=required_scopes,
	)


def _required_scopes() -> list[str] | None:
	"""Read required scopes from FASTMCP/MCP env vars."""
	raw = os.getenv("MCP_JWT_REQUIRED_SCOPES", "").strip()
	scopes = [scope for scope in raw.replace(",", " ").split() if scope]
	return scopes or None


def _scopes_to_set(scopes: Iterable[str] | None) -> set[str]:
	return {str(scope).strip() for scope in (scopes or []) if str(scope).strip()}


@dataclass(frozen=True)
class RequestIdentity:
	"""Identity extracted from a verified JWT access token."""
	agent: str
	parent_id: str | None
	client_id: str | None
	token_id: str | None
	scopes: set[str]


def authenticate_request() -> tuple[RequestIdentity | None, str]:
	"""Read verified JWT token from FastMCP auth context and extract identity."""
	access_token = get_access_token()
	if access_token is None:
		return None, "missing_or_invalid_bearer_token"

	claims = access_token.claims or {}
	agent = str(access_token.client_id or "").strip()
	if not agent:
		return None, "missing_agent_claim"

	parent_id = (
		str(claims.get("sub") or "").strip()
		or str(claims.get("sub_user") or "").strip()
		or None
	)
	token_id = str(claims.get("jti") or "").strip() or None
	scopes = _scopes_to_set(getattr(access_token, "scopes", None))

	logger.debug(
		"authenticate_request | client_id=%s sub=%s jti=%s exp=%s scopes=%s",
		agent,
		parent_id,
		token_id,
		claims.get("exp"),
		sorted(scopes),
	)

	identity = RequestIdentity(
		agent=agent,
		parent_id=parent_id,
		client_id=str(access_token.client_id or "").strip() or None,
		token_id=token_id,
		scopes=scopes,
	)
	return identity, "ok"

