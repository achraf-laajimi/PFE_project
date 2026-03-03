"""Intent classification"""

from agent.utils.llm_service import LLMService
from agent.utils.logger import get_logger
from agent.utils.schemas import IntentClassification
from functools import lru_cache
from pathlib import Path

logger = get_logger(__name__)

@lru_cache(maxsize=1)
def _load_prompt() -> str:
    """Load intent prompt template (cached)"""
    path = Path(__file__).parent.parent / "prompts" / "intent.txt"
    return path.read_text(encoding='utf-8')

def classify_intent(query: str, llm: LLMService, conversation_context: str = "") -> IntentClassification:
    """
    Classify user intent using LLM
    
    Args:
        query: User query string
        llm: LLM service instance
        conversation_context: Recent conversation history for resolving
                              follow-up queries and anaphora
    
    Returns:
        IntentClassification object
    """
    
    logger.info(f"Classifying intent for: {query}")
    
    prompt_template = _load_prompt()
    # Use replace() instead of .format() so JSON examples in the template
    # (which contain literal { }) don't trigger a KeyError.
    prompt = (
        prompt_template
        .replace("{conversation_context}", conversation_context or "(no previous conversation)")
        .replace("{query}", query)
    )
    
    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.1  # Low temp for consistent classification
        )
        
        result_dict = llm.extract_json(response)
        intent = IntentClassification(**result_dict)
        
        logger.info(f"Intent classified as: {intent.intent} (confidence: {intent.confidence})")
        return intent
    
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to tool_required")
        # Safe fallback
        return IntentClassification(
            intent="tool_required",
            reasoning=f"Failed to classify: {e}",
            entities=[],
            confidence=0.5
        )