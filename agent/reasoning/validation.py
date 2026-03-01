"""Tool parameter validation"""

from typing import Dict, Any, Tuple

from agent.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_type(param_info: dict) -> str | None:
    """
    Extract the effective type string from a JSON-Schema property.

    Handles:
      - {"type": "integer"}                       → "integer"
      - {"anyOf": [{"type":"integer"},{"type":"null"}]}  → "integer"
      - {"oneOf": [...]}                          → first non-null type
    """
    direct = param_info.get("type")
    if direct:
        return direct

    for key in ("anyOf", "oneOf"):
        variants = param_info.get(key, [])
        for v in variants:
            t = v.get("type")
            if t and t != "null":
                return t
    return None


def _coerce(value: Any, expected_type: str | None) -> Any:
    """
    Try to coerce *value* to match *expected_type*.

    LLMs frequently produce "4" instead of 4 — this auto-converts
    string-encoded numbers so the call doesn't fail on type check.
    """
    if expected_type in ("integer", "number") and isinstance(value, str):
        try:
            return int(value) if expected_type == "integer" else float(value)
        except ValueError:
            pass  # can't coerce → let the type check report the error
    return value


def validate_tool_params(
    tool_name: str,
    params: Dict[str, Any],
    tools_schema: Dict,
) -> Tuple[bool, str]:
    """
    Validate (and lightly coerce) tool parameters against schema.

    Returns:
        Tuple of (is_valid, error_message).
        On success, *params* may be mutated (string→int coercion).
    """

    # Check if tool exists
    if tool_name not in tools_schema:
        error = f"Tool '{tool_name}' not found in schema"
        logger.error(error)
        return False, error

    tool_def = tools_schema[tool_name]
    all_params = tool_def.get("params", {})
    required_list = tool_def.get("required", [])

    # Check only truly required parameters
    for param_name in required_list:
        if param_name not in params:
            error = f"Missing required parameter: {param_name}"
            logger.error(f"Validation failed for {tool_name}: {error}")
            return False, error

    # Type-check (and coerce) provided parameters
    for param_name, value in list(params.items()):
        if value is None:
            continue
        if param_name not in all_params:
            continue

        param_info = all_params[param_name]
        param_type = _resolve_type(param_info)

        if param_type is None:
            continue  # truly unknown schema — skip

        # Auto-coerce string→number if appropriate
        value = _coerce(value, param_type)
        params[param_name] = value  # write back the coerced value

        if param_type == "string" and not isinstance(value, str):
            error = f"Parameter '{param_name}' must be a string, got {type(value).__name__}"
            logger.error(f"Validation failed for {tool_name}: {error}")
            return False, error

        elif param_type in ("number", "integer") and not isinstance(value, (int, float)):
            error = f"Parameter '{param_name}' must be a number, got {type(value).__name__}"
            logger.error(f"Validation failed for {tool_name}: {error}")
            return False, error

    logger.info(f"Validation passed for {tool_name}")
    return True, ""