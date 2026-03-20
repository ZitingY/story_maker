"""Tests for NLG modules: prompt_templates, story_generator, option_generator."""
import pytest
from unittest.mock import patch

from src.nlg.prompt_templates import (
    SYSTEM_PROMPT,
    OPENING_PROMPT,
    STORY_CONTINUE_PROMPT,
    OPTION_GENERATION_PROMPT,
)
from src.nlg.story_generator import StoryGenerator
from src.nlg.option_generator import OptionGenerator, StoryOption


# ── Prompt templates ────────────────────────────────────────────────

class TestPromptTemplates:
    def test_system_prompt_nonempty(self):
        assert len(SYSTEM_PROMPT) > 50

    def test_opening_prompt_has_genre_slot(self):
        rendered = OPENING_PROMPT.format(genre="sci-fi")
        assert "sci-fi" in rendered

    def test_continue_prompt_slots(self):
        rendered = STORY_CONTINUE_PROMPT.format(
            kg_summary="hero: person",
            history="Turn 1 ...",
            intent="explore",
            player_input="go north",
            emotion="neutral",
        )
        assert "hero: person" in rendered
        assert "go north" in rendered

    def test_option_prompt_slots(self):
        rendered = OPTION_GENERATION_PROMPT.format(
            num_options=3, story_text="...", kg_summary="...",
        )
        assert "3" in rendered


# ── StoryGenerator (mocked LLM) ────────────────────────────────────

class TestStoryGenerator:
    @pytest.fixture
    def gen(self):
        return StoryGenerator()

    @patch("src.utils.api_client.llm_client")
    def test_generate_opening(self, mock_client, gen):
        mock_client.chat.return_value = "Once upon a time…"
        text = gen.generate_opening("fantasy")
        assert text == "Once upon a time…"
        mock_client.chat.assert_called_once()

    @patch("src.utils.api_client.llm_client")
    def test_continue_story(self, mock_client, gen):
        mock_client.chat.return_value = "The hero pressed on."
        text = gen.continue_story(
            player_input="go forward",
            intent="action",
            kg_summary="hero: person",
            history="Turn 1 ...",
        )
        assert "hero" in text.lower()


# ── OptionGenerator (mocked LLM + fallback) ────────────────────────

class TestOptionGenerator:
    @pytest.fixture
    def gen(self):
        return OptionGenerator()

    @patch("src.utils.api_client.llm_client")
    def test_generate_returns_story_options(self, mock_client, gen):
        mock_client.chat_json.return_value = {
            "options": [
                {"text": "Go north", "intent_hint": "explore", "risk_level": "low"},
                {"text": "Attack", "intent_hint": "action", "risk_level": "high"},
                {"text": "Rest", "intent_hint": "rest", "risk_level": "low"},
            ]
        }
        opts = gen.generate("story…", "kg…")
        assert len(opts) == 3
        assert all(isinstance(o, StoryOption) for o in opts)
        assert opts[0].text == "Go north"

    @patch("src.utils.api_client.llm_client")
    def test_fallback_on_error(self, mock_client, gen):
        mock_client.chat_json.side_effect = RuntimeError("fail")
        opts = gen.generate("story", "kg")
        assert len(opts) >= 1  # at least the fallback set
        assert all(isinstance(o, StoryOption) for o in opts)

    def test_story_option_defaults(self):
        opt = StoryOption(text="Do something")
        assert opt.intent_hint == "other"
        assert opt.risk_level == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
