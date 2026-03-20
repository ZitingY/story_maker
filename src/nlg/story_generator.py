"""Story generation via OpenAI chat completions.

Provides ``generate_opening()`` and ``continue_story()`` — both thin wrappers
around the shared ``llm_client`` singleton.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from src.nlg.prompt_templates import (
    OPENING_PROMPT,
    STORY_CONTINUE_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class StoryGenerator:
    """LLM-powered narrator for the text adventure."""

    def generate_opening(self, genre: str = "fantasy") -> str:
        """Generate the opening scene of a new game."""
        from src.utils.api_client import llm_client

        user_msg = OPENING_PROMPT.format(genre=genre)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return llm_client.chat(messages)

    def continue_story(
        self,
        player_input: str,
        intent: str,
        kg_summary: str,
        history: str,
        emotion: str = "neutral",
    ) -> str:
        """Continue the story based on the player's action and emotion."""
        from src.utils.api_client import llm_client

        user_msg = STORY_CONTINUE_PROMPT.format(
            kg_summary=kg_summary,
            history=history,
            intent=intent,
            player_input=player_input,
            emotion=emotion,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return llm_client.chat(messages)
