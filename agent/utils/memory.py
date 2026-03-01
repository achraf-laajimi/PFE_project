"""
Session-based conversation memory with Mem0 long-term semantic layer.

Two memory tiers:
  1. Short-term (_store)  — last N turns for anaphora / context continuity.
  2. Long-term  (Mem0)    — behavioral facts & user preferences stored as
                            vector embeddings for semantic retrieval.

Mem0 is configured to use a local Qdrant instance by default.
Set QDRANT_URL / QDRANT_API_KEY env vars for a remote Qdrant cluster.
"""

import os
import re
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from mem0 import Memory
from openai import AsyncOpenAI

from agent.utils.logger import get_logger

# Load .env (same file agent/ uses for OPENAI_API_KEY)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

logger = get_logger(__name__)

# Max turns kept per session (oldest dropped first)
_MAX_HISTORY = 10

# ── Mem0 configuration ───────────────────────────────────────
# Uses OpenAI embeddings (text-embedding-3-small) + Qdrant for storage.
# Falls back to in-memory Qdrant if no QDRANT_URL is set.

def _build_mem0_config() -> dict:
    """Build Mem0 config dict from environment variables."""
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1500,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "classquiz_memories",
                "embedding_model_dims": 1536,
            },
        },
        "version": "v1.1",
    }

    # Remote Qdrant cluster — set these env vars in production
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        config["vector_store"]["config"]["url"] = qdrant_url
        if qdrant_api_key:
            config["vector_store"]["config"]["api_key"] = qdrant_api_key
        logger.info(f"Mem0: using remote Qdrant at {qdrant_url}")
    else:
        # Local Qdrant — store on disk next to the agent package
        local_path = str(Path(__file__).resolve().parent.parent / "qdrant_data")
        config["vector_store"]["config"]["path"] = local_path
        logger.info(f"Mem0: using local Qdrant at {local_path}")

    return config


# ── Sanitization Patterns: Replace noisy data with placeholders ──────────
# Instead of rejecting text with numbers, we sanitize it by replacing
# raw statistics with semantic placeholders like [score] and [data].

# Pattern 1: Raw JSON field names from tool outputs (to be stripped)
_RAW_FIELD_PATTERN = re.compile(
    r"\b("
    r"stars_earned|stars_total|completion_percentage|exercises_total|"
    r"exercises_done|focus_score|excellence_rate|average_stars|"
    r"completion_rate|average_mistake|nb_mistakes|time_seconds|"
    r"activity_feed|progress_list|kpis|usage_summary|score|percentage|"
    r"total_exercises|done_exercises|remaining_exercises|"
    r"avg_time|max_score|min_score"
    r")\s*[:=]\s*\d+",
    re.IGNORECASE,
)

# Pattern 2: Percentage symbols (e.g., "22%" → "[score]")
_PERCENTAGE_PATTERN = re.compile(r"\d+\s*%")

# Pattern 3: Numeric comparisons (e.g., "4/10", "85 out of 100" → "[data]")
_NUMERIC_COMPARISON_PATTERN = re.compile(
    r"\d+\s*(?:out of|of|/|sur|من)\s*\d+",
    re.IGNORECASE
)

# Pattern 4: Standalone large numbers (e.g., "284 exercises" → "[count] exercises")
_STANDALONE_NUMBER_PATTERN = re.compile(r"\b\d{3,}\b")


def _sanitize_text(text: str) -> str:
    """
    Sanitize text by replacing raw statistics with semantic placeholders.
    
    Transformations:
    - "22%" → "[score]"
    - "4/10 exercises" → "[data] exercises"
    - "stars_earned: 450" → "" (stripped)
    - "284 exercises" → "[count] exercises"
    
    Examples:
    Input:  "Chayma got 20% in Math and is struggling with logic"
    Output: "Chayma got [score] in Math and is struggling with logic"
    
    Input:  "Ahmed completed 4/10 exercises, focus_score: 85"
    Output: "Ahmed completed [data] exercises"
    """
    if not text:
        return text
    
    sanitized = text
    
    # Step 1: Strip raw JSON field patterns (e.g., "stars_earned: 450")
    sanitized = _RAW_FIELD_PATTERN.sub("", sanitized)
    
    # Step 2: Replace percentages with [score]
    sanitized = _PERCENTAGE_PATTERN.sub("[score]", sanitized)
    
    # Step 3: Replace numeric comparisons with [data]
    sanitized = _NUMERIC_COMPARISON_PATTERN.sub("[data]", sanitized)
    
    # Step 4: Replace large standalone numbers with [count]
    # (only numbers >= 100, to preserve ages, dates, small counts)
    sanitized = _STANDALONE_NUMBER_PATTERN.sub("[count]", sanitized)
    
    # Step 5: Clean up extra whitespace and commas
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(r",\s*,", ",", sanitized)
    sanitized = re.sub(r"\(\s*\)", "", sanitized)  # Remove empty parens
    
    return sanitized.strip()


# ── Semantic Memory Router ───────────────────────────────────
# Vector-based behavioral detection that works for Arabizi, French, Arabic, English

class SemanticMemoryRouter:
    """
    Semantic router that uses vector embeddings to detect behavioral content.
    
    This replaces brittle keyword matching with semantic similarity scoring,
    making it work seamlessly across languages and dialects (including Tunisian Arabizi).
    
    Key benefits:
    - Works for "yew7al", "يوحل", "struggle" equally well (semantic understanding)
    - No keyword list maintenance required
    - Fast (~100ms per check with caching)
    - Cost-effective (embeddings are 100x cheaper than LLM completions)
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        
        # These represent the "pure concepts" of what we want to save
        # They are language-agnostic at the vector level
        self.reference_concepts = [
            "Student struggles with a specific academic topic or concept",
            "Parent expresses a preference for language, style, or communication format",
            "Behavioral observation about student's focus, motivation, or study habits",
            "Academic strength or weakness mentioned about the student",
            "Parental feedback or concern about educational progress",
            "Request for specific teaching approach or intervention",
            "Student's preferred learning time or environment",
        ]
        
        # Concept vectors loaded lazily on first call — cannot await in __init__
        self.concept_vectors = None
    
    async def _ensure_initialized(self) -> None:
        """Lazily initialize concept vectors on first use (async-safe)."""
        if self.concept_vectors is not None:
            return
        try:
            self.concept_vectors = await self._get_embeddings(self.reference_concepts)
            logger.info(f"[SemanticRouter] Initialized with {len(self.concept_vectors)} reference concepts")
        except Exception as e:
            logger.error(f"[SemanticRouter] Failed to initialize concept vectors: {e}")
            self.concept_vectors = None
    
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts (async, non-blocking)."""
        response = await self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [record.embedding for record in response.data]
    
    async def get_behavioral_score(self, text: str) -> float:
        """
        Calculate how 'behavioral' a text is using semantic similarity.
        
        Returns a score between 0.0 and 1.0, where:
        - 0.0-0.5: Not behavioral (pure data, greetings, etc.)
        - 0.5-0.7: Potentially behavioral (borderline)
        - 0.7+: Definitely behavioral (should be stored)
        
        This works for ANY language because vectors capture semantic meaning:
        - "yew7al" (Arabizi) → high similarity to "struggle"
        - "عندو مشكل" (Arabic) → high similarity to "has a problem"
        - "il a des difficultés" (French) → high similarity to "has difficulties"
        
        Args:
            text: The text to score
            
        Returns:
            float: Behavioral score (0.0-1.0)
        """
        if not text or len(text.strip()) < 10:
            return 0.0

        await self._ensure_initialized()

        if self.concept_vectors is None:
            logger.warning("[SemanticRouter] Concept vectors not initialized, falling back to heuristic")
            return 0.5  # Neutral score if router failed to initialize
        
        try:
            # Get embedding for the input text
            text_vector = (await self._get_embeddings([text]))[0]
            
            # Calculate cosine similarity against all reference concepts
            similarities = []
            for concept_vector in self.concept_vectors:
                similarity = np.dot(text_vector, concept_vector) / (
                    np.linalg.norm(text_vector) * np.linalg.norm(concept_vector)
                )
                similarities.append(similarity)
            
            # Return the highest similarity score
            max_score = max(similarities)
            logger.debug(f"[SemanticRouter] Behavioral score: {max_score:.3f} for text: '{text[:50]}...'")
            return max_score
            
        except Exception as e:
            logger.error(f"[SemanticRouter] Error calculating behavioral score: {e}")
            return 0.5  # Neutral score on error


class MemoryManager:
    """
    Two-tier memory manager keyed by session_id.

    Tier 1 — Short-term (_store):
        Last N turns (query + response + entities) for anaphora resolution
        and conversational continuity.

    Tier 2 — Long-term (Mem0):
        Semantic vector memory storing behavioral facts & user preferences.
        Uses semantic routing for robust behavioral detection across languages.
        Raw statistics are SANITIZED (replaced with placeholders) before storage.
        
    Key features:
    - Semantic routing (works for Arabizi, Arabic, French, English)
    - Asynchronous memory storage (zero user latency)
    - Sanitization instead of hard rejection
    - Direct student_id as user_id for cross-session persistence
    """

    def __init__(self, max_history: int = _MAX_HISTORY):
        self._max = max_history
        # session_id → list of Turn dicts
        self._store: Dict[str, List[Dict]] = defaultdict(list)
        # session_id → arbitrary context (student_id, prefs, …)
        self._context: Dict[str, Dict] = defaultdict(dict)

        # ── Mem0 long-term memory ────────────────────────────
        try:
            mem0_config = _build_mem0_config()
            self.mem0 = Memory.from_config(mem0_config)
            logger.info("Mem0 long-term memory initialized successfully")
        except Exception as e:
            logger.error(f"Mem0 initialization failed (falling back to short-term only): {e}")
            self.mem0 = None
        
        # ── Background task set (strong references prevent GC) ─────
        self._background_tasks: set = set()

        # ── Semantic Router ──────────────────────────────────
        # Initialize semantic router for behavioral detection
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.semantic_router = SemanticMemoryRouter(api_key)
                logger.info("SemanticMemoryRouter initialized successfully")
            except Exception as e:
                logger.error(f"SemanticMemoryRouter initialization failed: {e}")
                self.semantic_router = None
        else:
            logger.warning("OPENAI_API_KEY not found, semantic routing disabled")
            self.semantic_router = None

    # ── Public API ─────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        entities: List[str] = None,
        intent: str = None,
        tools_used: List[str] = None,
    ) -> None:
        """
        Record a completed turn in short-term memory AND push
        behavioral observations to Mem0 long-term memory asynchronously.
        
        The memory storage happens in the background, so the user
        gets their response immediately without waiting for Mem0.
        """
        turn = {
            "query": query,
            "response": response,
            "entities": entities or [],
            "intent": intent,
            "tools_used": tools_used or [],
        }
        history = self._store[session_id]
        history.append(turn)
        # Evict oldest when over limit
        if len(history) > self._max:
            self._store[session_id] = history[-self._max :]
        logger.debug(f"[Memory] {session_id}: {len(self._store[session_id])} turns stored")

        # ── Mem0: store behavioral context (asynchronously) ──
        # Wrapped in error-resilient coroutine; strong reference prevents GC.
        async def _safe_mem0_task() -> None:
            try:
                await self._store_in_mem0_async(
                    session_id, query, response, intent, tools_used or []
                )
            except Exception as exc:
                logger.error(
                    f"[Mem0] Background task failed unexpectedly: {exc}",
                    exc_info=True,
                )

        task = asyncio.create_task(_safe_mem0_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def get_history(self, session_id: str) -> List[Dict]:
        """Return the full turn history for a session."""
        return list(self._store.get(session_id, []))

    def get_recent_entities(self, session_id: str, n: int = 3) -> List[str]:
        """
        Collect unique entities from the last *n* turns (most-recent-first).

        Useful for resolving pronouns: if the user says "give me his exercises"
        and the last turn mentioned "Ahmed", the planner can use "Ahmed".
        """
        history = self._store.get(session_id, [])
        seen = []
        for turn in reversed(history[-n:]):
            for entity in turn.get("entities", []):
                if entity not in seen:
                    seen.append(entity)
        return seen

    def get_last_student_id(self, session_id: str) -> Optional[str]:
        """
        Return the student_id for this session.

        Checks context first (set explicitly via update_context),
        then falls back to scanning turn entities.
        """
        # Fast path: explicit context
        ctx_id = self._context.get(session_id, {}).get("student_id")
        if ctx_id:
            return str(ctx_id)

        # Fallback: walk backwards through history
        for turn in reversed(self._store.get(session_id, [])):
            for entity in turn.get("entities", []):
                if entity.isdigit():
                    return entity
        return None

    # ── Session context ───────────────────────────────────────

    def update_context(self, session_id: str, data: Dict) -> None:
        """
        Merge *data* into the session's context dict.
        
        Critical: Call this BEFORE add_turn() so that student_id
        is available for Mem0 user_id resolution.
        """
        self._context[session_id].update(data)
        logger.debug(f"[Memory] Context updated for {session_id}: {list(data.keys())}")

    def get_context(self, session_id: str) -> Dict:
        """Return the full context dict for a session."""
        return dict(self._context.get(session_id, {}))

    def get_student_id(self, session_id: str) -> Optional[str]:
        """Shortcut: return student_id from context, or None."""
        return self._context.get(session_id, {}).get("student_id")

    # ── Mem0 long-term memory ─────────────────────────────────

    async def _store_in_mem0_async(
        self,
        session_id: str,
        query: str,
        response: str,
        intent: Optional[str],
        tools_used: List[str],
    ) -> None:
        """
        Asynchronously push observations to Mem0 with semantic routing.
        
        This runs in the background after the user gets their response,
        ensuring zero latency impact on the conversation.
        
        Process:
        1. Use semantic router to score query and response (fast, ~100ms)
        2. If score > 0.72, sanitize and store in Mem0 (expensive, ~2s)
        3. Only spend expensive LLM tokens when we're confident it's behavioral
        
        Threshold explanation:
        - 0.72+ = High confidence behavioral content (store it)
        - 0.5-0.72 = Borderline (skip to avoid noise)
        - <0.5 = Not behavioral (skip)
        """
        if self.mem0 is None or self.semantic_router is None:
            return

        user_id = self._resolve_user_id(session_id)
        student_id = self.get_student_id(session_id)
        
        # Always include student_id in metadata for context
        metadata = {"session_id": session_id}
        if student_id:
            metadata["student_id"] = student_id

        # ── Store user query (if behavioral) ─────────────────
        if query:
            try:
                query_score = await self.semantic_router.get_behavioral_score(query)
                
                if query_score > 0.72:  # High confidence threshold
                    # Sanitize before storage
                    sanitized_query = _sanitize_text(query)
                    
                    if sanitized_query and len(sanitized_query) > 10:
                        self.mem0.add(
                            sanitized_query,
                            user_id=user_id,
                            metadata=metadata,
                        )
                        logger.debug(
                            f"[Mem0] Stored query (score: {query_score:.3f}): '{sanitized_query[:50]}...'"
                        )
                    else:
                        logger.debug(f"[Mem0] Query too short after sanitization, skipping")
                else:
                    logger.debug(
                        f"[Mem0] Query score too low ({query_score:.3f}), skipping"
                    )
            except Exception as e:
                logger.warning(f"[Mem0] Failed to store query: {e}")

        # ── Store assistant response (if behavioral) ─────────
        if response:
            try:
                response_score = await self.semantic_router.get_behavioral_score(response)
                
                if response_score > 0.72:  # High confidence threshold
                    # Sanitize before storage
                    sanitized_response = _sanitize_text(response)
                    
                    if sanitized_response and len(sanitized_response) > 20:
                        self.mem0.add(
                            sanitized_response,
                            user_id=user_id,
                            metadata=metadata,
                        )
                        logger.debug(
                            f"[Mem0] Stored response (score: {response_score:.3f}): '{sanitized_response[:50]}...'"
                        )
                        
                        if sanitized_response != response:
                            logger.debug(
                                f"[Mem0] Sanitized response: '{response[:80]}...' → '{sanitized_response[:80]}...'"
                            )
                    else:
                        logger.debug(f"[Mem0] Response too short after sanitization, skipping")
                else:
                    logger.debug(
                        f"[Mem0] Response score too low ({response_score:.3f}), skipping"
                    )
            except Exception as e:
                logger.warning(f"[Mem0] Failed to store response: {e}")

    def get_semantic_memories(self, session_id: str, query: str = "") -> str:
        """
        Retrieve long-term behavioral context from Mem0.

        Returns a formatted string of relevant memories, e.g.:
          "- Parent prefers Darija responses
           - Student usually studies at night
           - Parent asked for concise reports"

        If Mem0 is unavailable returns empty string.
        """
        if self.mem0 is None:
            return ""

        user_id = self._resolve_user_id(session_id)

        try:
            # Search memories relevant to the current query
            if query:
                results = self.mem0.search(query, user_id=user_id, limit=5)
            else:
                results = self.mem0.get_all(user_id=user_id)

            if not results:
                return ""

            # results can be a dict with "results" key or a list directly
            memories = results.get("results", results) if isinstance(results, dict) else results

            if not memories:
                return ""

            lines = []
            for mem in memories:
                text = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
                if text.strip():
                    lines.append(f"- {text.strip()}")

            return "\n".join(lines) if lines else ""

        except Exception as e:
            logger.warning(f"[Mem0] Failed to retrieve memories: {e}")
            return ""

    # ── Prompt formatting ─────────────────────────────────────

    def format_for_prompt(self, session_id: str, max_turns: int = 3, query: str = "") -> str:
        """
        Format recent history + long-term context for LLM prompts.

        Returns:
            [Long-Term Behavioral Context]
            - Parent prefers short summaries
            - Student studies at night

            [Short-Term Conversation History]
            User: كيفاش أحمد في الرياضيات؟
            Assistant: أحمد عنده 85% في الهندسة...
        """
        sections = []

        # ── Long-term: Mem0 semantic memories ────────────────
        semantic = self.get_semantic_memories(session_id, query=query)
        if semantic:
            sections.append(f"[Long-Term Behavioral Context]\n{semantic}")

        # ── Short-term: recent turns ─────────────────────────
        history = self._store.get(session_id, [])
        if history:
            lines = []
            for turn in history[-max_turns:]:
                lines.append(f"User: {turn['query']}")
                lines.append(f"Assistant: {turn['response']}")
            sections.append(f"[Short-Term Conversation History]\n" + "\n".join(lines))
        else:
            sections.append("[Short-Term Conversation History]\n(no previous conversation)")

        return "\n\n".join(sections)

    # ── Lifecycle ─────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Gracefully wait for all in-flight background memory tasks.

        Call this during server shutdown (FastAPI lifespan cleanup) so that
        no pending Mem0 writes are lost on restart.
        """
        if self._background_tasks:
            logger.info(
                f"[Memory] Waiting for {len(self._background_tasks)} "
                "background memory task(s) before shutdown..."
            )
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            logger.info("[Memory] All background memory tasks completed.")

    # ── Cleanup ───────────────────────────────────────────────

    def clear(self, session_id: str) -> None:
        """Wipe short-term history for a session (Mem0 persists)."""
        self._store.pop(session_id, None)
        self._context.pop(session_id, None)
        logger.info(f"[Memory] Cleared session {session_id}")

    # ── Helpers ───────────────────────────────────────────────

    def _resolve_user_id(self, session_id: str) -> str:
        """
        Derive a stable user_id for Mem0 indexing.
        
        Priority:
        1. student_id from context → student_id directly (no prefix)
           (ensures memory persists across sessions for same student)
        2. Fallback → "session_{session_id}"
        
        Critical: This must be called AFTER update_context() has been
        called with student_id, otherwise it will fall back to session_id
        and memories won't persist across sessions.
        """
        student_id = self.get_student_id(session_id)
        if student_id:
            # Return student_id directly (no "parent_" prefix)
            logger.debug(f"[Mem0] Resolved user_id: {student_id} (from student_id)")
            return student_id
        
        # Fallback to session-based ID
        user_id = f"session_{session_id}"
        logger.debug(f"[Mem0] Resolved user_id: {user_id} (fallback to session)")
        return user_id