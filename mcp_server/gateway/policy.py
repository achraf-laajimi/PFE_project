"""Authorization helpers using Casbin."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import casbin
from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)

def _default_model_path() -> Path:
	return Path(__file__).parent / "casbin" / "model.conf"

def _default_policy_path() -> Path:
	return Path(__file__).parent / "casbin" / "policy.csv"

@lru_cache(maxsize=1)
def get_enforcer() -> casbin.Enforcer:
	"""Create and cache a Casbin enforcer instance."""
	model_path = Path(str(_default_model_path()))
	policy_path = Path(str(_default_policy_path()))
	enforcer = casbin.Enforcer(str(model_path), str(policy_path))
	logger.info(
		"Casbin loaded | model=%s policy=%s",
		model_path.as_posix(),
		policy_path.as_posix(),
	)
	return enforcer


def is_tool_allowed(agent: str, tool_name: str, action: str = "execute") -> bool:
	"""Return True if the agent is allowed to execute the given tool."""
	try:
		return bool(get_enforcer().enforce(agent, tool_name, action))
	except Exception as exc:  # noqa: BLE001
		logger.error("Casbin enforce failure | agent=%s tool=%s err=%s", agent, tool_name, exc)
		return False

