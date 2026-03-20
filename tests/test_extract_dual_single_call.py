"""Tests for KG-2: merged LLM calls in extract_dual."""
import pytest
from unittest.mock import patch

from src.knowledge_graph.relation_extractor import RelationExtractor


class TestExtractDualSingleCall:
    """Test that extract_dual uses a single LLM call."""

    @patch("src.utils.api_client.llm_client")
    def test_single_llm_call(self, mock_client):
        """extract_dual should make only one LLM call when story_text is provided."""
        mock_client.chat_json.return_value = {
            "entities": [
                {"name": "Hero", "type": "person"},
                {"name": "Cave", "type": "location"},
            ],
            "relations": [
                {"source": "Hero", "target": "Cave", "relation": "located_at"},
            ],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract_dual(
            player_input="I enter the cave.",
            story_text="The hero steps into the dark cave.",
        )

        # Should have made exactly 1 call
        assert mock_client.chat_json.call_count == 1
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1

    @patch("src.utils.api_client.llm_client")
    def test_combined_prompt_contains_both_texts(self, mock_client):
        """The prompt should contain both player input and story text."""
        mock_client.chat_json.return_value = {"entities": [], "relations": []}
        extractor = RelationExtractor(enhanced=True)
        extractor.extract_dual(
            player_input="I attack the dragon.",
            story_text="The dragon roars in fury.",
        )

        call_args = mock_client.chat_json.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "I attack the dragon." in user_msg
        assert "The dragon roars in fury." in user_msg
        assert "PLAYER INPUT" in user_msg
        assert "STORY TEXT" in user_msg

    @patch("src.utils.api_client.llm_client")
    def test_empty_story_falls_back_to_player_only(self, mock_client):
        """When story_text is empty, should only extract from player input."""
        mock_client.chat_json.return_value = {
            "entities": [{"name": "Sword", "type": "item"}],
            "relations": [],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract_dual(
            player_input="I pick up the sword.",
            story_text="",
        )

        # Should have called _extract_player_input, not the dual prompt
        assert len(result["entities"]) == 1

    @patch("src.utils.api_client.llm_client")
    def test_existing_entities_hint_included(self, mock_client):
        """Existing entities should be included as a hint in the prompt."""
        mock_client.chat_json.return_value = {"entities": [], "relations": []}
        extractor = RelationExtractor(enhanced=True)
        extractor.extract_dual(
            player_input="I talk to Gandalf.",
            story_text="Gandalf greets you warmly.",
            existing_entities=["Gandalf", "Frodo", "Shire"],
        )

        call_args = mock_client.chat_json.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Gandalf" in user_msg
        assert "Frodo" in user_msg
        assert "Existing entities" in user_msg

    @patch("src.utils.api_client.llm_client")
    def test_single_call_failure_falls_back(self, mock_client):
        """If single-call fails, should fall back to split extraction."""
        # First call (single-call) fails, second/third calls (fallback) succeed
        mock_client.chat_json.side_effect = [
            RuntimeError("API error"),  # single call fails
            {"entities": [{"name": "Hero", "type": "person"}], "relations": []},  # story extract
            {"entities": [{"name": "Cave", "type": "location"}], "relations": []},  # player extract
        ]
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract_dual(
            player_input="I enter.",
            story_text="Hero enters the cave.",
        )

        # Should have made 3 calls: 1 failed single + 2 fallback
        assert mock_client.chat_json.call_count == 3
        # Both entities from fallback should be present
        names = {e["name"].lower() for e in result["entities"]}
        assert "hero" in names
        assert "cave" in names


class TestExtractDualNormalization:
    """Test that types are normalized in dual extraction."""

    @patch("src.utils.api_client.llm_client")
    def test_types_normalized(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [
                {"name": "Aragorn", "type": "character"},
                {"name": "Rivendell", "type": "place"},
            ],
            "relations": [],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract_dual("Visit Aragorn.", "They go to Rivendell.")
        types = {e["name"]: e["type"] for e in result["entities"]}
        assert types["Aragorn"] == "person"
        assert types["Rivendell"] == "location"
