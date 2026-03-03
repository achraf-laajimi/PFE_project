"""Response synthesis"""

from agent.utils.llm_service import LLMService
from agent.utils.logger import get_logger
from agent.utils.schemas import ToolResult
from functools import lru_cache
from pathlib import Path
from typing import List
import json
import re

logger = get_logger(__name__)
_MAX_TOOL_RESULT_CHARS = 3500
_MAX_TOOL_ERROR_CHARS = 500
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_ARABIZI_DIGIT_RE = re.compile(r"\b[a-zA-Z]*[3579][a-zA-Z]*\b")
_ARABIZI_KEYWORD_RE = re.compile(
    r"\b(mta3|5ater|besh|ey|bahi|mezien|chayma|weldi|weldha|nheb|kifeh|chnoua|chnowa|fi|maa?ref|3andh|darss)\b",
    re.IGNORECASE,
)


def _is_arabizi_query(text: str) -> bool:
    """Detect Latin-script Tunisian Darija (Arabizi) reliably enough for output locking."""
    if not text:
        return False
    # If Arabic script is already present, treat as Arabic-script query, not Arabizi.
    if _ARABIC_CHAR_RE.search(text):
        return False
    if not _LATIN_CHAR_RE.search(text):
        return False
    return bool(_ARABIZI_DIGIT_RE.search(text) or _ARABIZI_KEYWORD_RE.search(text))

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
    llm: LLMService,
    history: str = "",
) -> str:
    """
    Generate final response from tool results using LLM
    
    Args:
        query: Original user query
        tool_results: Results from tool execution
        llm: LLM service instance
        history: Formatted conversation history for anaphora resolution
    
    Returns:
        Natural language response
    """
    
    logger.info(f"Synthesizing response from {len(tool_results)} tool results")
    
    prompt_template = _load_synthesis_prompt()
    
    # Format tool results
    results_text = _format_tool_results(tool_results)
    
    prompt = prompt_template.format(
        query=query,
        tool_results=results_text,
        history=history or "(no previous conversation)",
    )
    logger.debug(f"Synthesis prompt length: {len(prompt)} chars")
    
    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.45,
            max_tokens=500,
            timeout=35,
        )

        # Guardrail A: Arabic-script query with Latin leakage → force Arabic rewrite.
        if _ARABIC_CHAR_RE.search(query or "") and _LATIN_CHAR_RE.search(response or ""):
            logger.warning("Synthesis language leak: Arabic query with Latin output; forcing Arabic rewrite")
            rewrite_prompt = (
                "أعد صياغة النص التالي بنفس الحقائق والأرقام، لكن بالدّارجة/العربية "
                "وبالحروف العربية فقط. ممنوع أي كلمات بحروف لاتينية. "
                "حافظ على المعنى والترتيب العام، وأنهِ الرد بنصيحة قصيرة تبدأ بـ 💡.\n\n"
                f"النص الحالي:\n{response}"
            )
            response = llm.generate(
                prompt=rewrite_prompt,
                temperature=0.2,
                max_tokens=500,
                timeout=20,
            )

        # Guardrail B: Arabizi query must return Arabic-script Darija (per prompt policy).
        if _is_arabizi_query(query or "") and _LATIN_CHAR_RE.search(response or ""):
            logger.warning("Synthesis language lock: Arabizi query produced Latin output; forcing Arabic-script Darija")
            rewrite_prompt = (
                "حوّل الرد التالي إلى الدارجة التونسية بالحروف العربية فقط، "
                "مع الحفاظ على نفس الحقائق والأرقام والترتيب العام. "
                "ممنوع أي كلمات بحروف لاتينية. اختم بنصيحة قصيرة تبدأ بـ 💡.\n\n"
                f"الرد الحالي:\n{response}"
            )
            response = llm.generate(
                prompt=rewrite_prompt,
                temperature=0.2,
                max_tokens=500,
                timeout=20,
            )
        
        logger.info(f"Response synthesized: {len(response)} chars")
        return response.strip()
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        # Fallback to basic response
        return "I processed your request, but had trouble generating a response. Please try again."

def generate_chitchat_response(query: str, llm: LLMService, history: str = "") -> str:
    """
    Generate chitchat response using LLM
    
    Args:
        query: User query
        llm: LLM service instance
        history: Formatted conversation history so short replies resolve correctly
    
    Returns:
        Natural language chitchat response
    """
    
    logger.info(f"Generating chitchat response for: {query}")
    
    prompt_template = _load_chitchat_prompt()
    prompt = prompt_template.format(query=query, history=history or "(no previous conversation)")
    
    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.5,
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
            payload = json.dumps(result.data, indent=2, ensure_ascii=False)
            if len(payload) > _MAX_TOOL_RESULT_CHARS:
                payload = (
                    payload[:_MAX_TOOL_RESULT_CHARS]
                    + "\n...[truncated for synthesis prompt]"
                )
            formatted.append(
                f"Tool: {result.tool_name}\n"
                f"Status: Success\n"
                f"Result: {payload}\n"
            )
        else:
            error_text = (result.error or "")[:_MAX_TOOL_ERROR_CHARS]
            formatted.append(
                f"Tool: {result.tool_name}\n"
                f"Status: Failed\n"
                f"Error: {error_text}\n"
            )
    
    return "\n---\n".join(formatted) if formatted else "No tool results"