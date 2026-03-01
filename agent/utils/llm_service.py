"""LLM service for OpenAI GPT integration"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from agent.utils.logger import get_logger

# Load .env from agent/ directory (where the key lives)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

logger = get_logger(__name__)


class LLMService:
    """Service for interacting with OpenAI GPT models"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before starting the agent."
            )
        self.model = model
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized LLM service with model: {model}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate completion from OpenAI GPT

        Args:
            prompt: The user prompt
            system: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Sending request to OpenAI ({self.model})")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            generated_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM response length: {len(generated_text)} chars")
            return generated_text

        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise Exception(f"LLM generation failed: {e}")

    def extract_json(self, text: str) -> Dict:
        """
        Extract JSON from LLM response

        Handles:
        - Markdown code blocks (```json...```)
        - Plain JSON
        - JSON embedded in text

        Args:
            text: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        # Clean up text
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text using balanced brace matching
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        raise ValueError(
            f"Could not extract valid JSON from LLM response: {text[:100]}..."
        )

    def health_check(self) -> bool:
        """Check if OpenAI API is reachable"""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False