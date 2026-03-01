"""Response synthesis"""

from agent.utils.llm_service import LLMService
from agent.utils.logger import get_logger
from agent.utils.schemas import ToolResult
from functools import lru_cache
from pathlib import Path
from typing import List
import json

logger = get_logger(__name__)

@lru_cache(maxsize=1)
def _load_synthesis_prompt() -> str:
    """Load synthesis prompt template (cached)"""
    path = Path(__file__).parent.parent / "prompts" / "synthesis.txt"
    return path.read_text(encoding='utf-8')

@lru_cache(maxsize=1)
def _load_chitchat_prompt() -> str:
    """Load chitchat prompt template (cached)"""
    path = Path(__file__).parent.parent / "prompts" / "chitchat.txt"
    return path.read_text(encoding='utf-8')

def synthesize_response(
    query: str,
    tool_results: List[ToolResult],
    llm: LLMService
) -> str:
    """
    Generate final response from tool results using LLM
    
    Args:
        query: Original user query
        tool_results: Results from tool execution
        llm: LLM service instance
    
    Returns:
        Natural language response
    """
    
    logger.info(f"Synthesizing response from {len(tool_results)} tool results")
    
    prompt_template = _load_synthesis_prompt()
    
    # Format tool results
    results_text = _format_tool_results(tool_results)
    
    prompt = prompt_template.format(
        query=query,
        tool_results=results_text
    )
    
    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.7,  # Higher temp for natural, varied responses
            max_tokens=500
        )
        
        logger.info(f"Response synthesized: {len(response)} chars")
        return response.strip()
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        # Fallback to basic response
        return "I processed your request, but had trouble generating a response. Please try again."

def generate_chitchat_response(query: str, llm: LLMService) -> str:
    """
    Generate chitchat response using LLM
    
    Args:
        query: User query
        llm: LLM service instance
    
    Returns:
        Natural language chitchat response
    """
    
    logger.info(f"Generating chitchat response for: {query}")
    
    prompt_template = _load_chitchat_prompt()
    prompt = prompt_template.format(query=query)
    
    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.8,  # Higher temp for natural conversation
            max_tokens=200
        )
        
        logger.info(f"Chitchat response generated: {len(response)} chars")
        return response.strip()
    
    except Exception as e:
        logger.error(f"Chitchat generation failed: {e}")
        # Fallback
        return "مرحباً! أنا مساعدك التعليمي. كيف يمكنني مساعدتك اليوم؟"

def _format_tool_results(tool_results: List[ToolResult]) -> str:
    """Format tool results for LLM prompt"""
    
    formatted = []
    for result in tool_results:
        if result.success:
            formatted.append(
                f"Tool: {result.tool_name}\n"
                f"Status: Success\n"
                f"Result: {json.dumps(result.data, indent=2, ensure_ascii=False)}\n"
            )
        else:
            formatted.append(
                f"Tool: {result.tool_name}\n"
                f"Status: Failed\n"
                f"Error: {result.error}\n"
            )
    
    return "\n---\n".join(formatted) if formatted else "No tool results"