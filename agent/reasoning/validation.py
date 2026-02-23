"""Tool parameter validation"""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_tool_params(
    tool_name: str,
    params: Dict[str, Any],
    tools_schema: Dict
) -> Tuple[bool, str]:
    """
    Validate tool parameters against schema
    
    Args:
        tool_name: Name of the tool
        params: Parameters to validate
        tools_schema: Schema of available tools
    
    Returns:
        Tuple of (is_valid, error_message)
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
    
    # Type-check all provided parameters
    for param_name, value in params.items():
        if param_name in all_params:
            param_info = all_params[param_name]
            param_type = param_info.get("type", "string")
            
            if param_type == "string" and not isinstance(value, str):
                error = f"Parameter '{param_name}' must be a string, got {type(value).__name__}"
                logger.error(f"Validation failed for {tool_name}: {error}")
                return False, error
            
            elif param_type == "number" and not isinstance(value, (int, float)):
                error = f"Parameter '{param_name}' must be a number, got {type(value).__name__}"
                logger.error(f"Validation failed for {tool_name}: {error}")
                return False, error
    
    logger.info(f"Validation passed for {tool_name}")
    return True, ""