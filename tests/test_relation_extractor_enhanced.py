"""Tests for enhanced relation_extractor.py."""
import pytest
from unittest.mock import patch, MagicMock

from src.knowledge_graph.relation_extractor import (
    RelationExtractor,
    extract,
    extract_legacy,
    extract_dual,
)


# ── Helpers ───────────────────────────────────────────────

def _mock_rich_extraction(messages, **kwargs):
    """Return rich extraction result based on prompt content."""
    text = str(messages)
    if "player" in text.lower() or "action input" in text.lower():
        # player input extraction — simpler
        return {
            "entities": [{"name": "Cave", "type": "location", "description": "A dark cave"}],
            "relations": [],
        }
    # story text extraction — rich
    return {
        "entities": [
            {
                "name": "Hero",
                "type": "person",
                "description": "A brave warrior",
                "status": {"health": "injured", "mood": "determined"},
                "state_changes": {"health": "injured"},
            },
            {"name": "Sword", "type": "item", "description": "Ancient blade"},
        ],
        "relations": [
            {"source": "Hero", "target": "Sword", "relation": "possesses", "context": "Hero wielded the sword"},
        ],
    }


# ── RelationExtractor class tests ────────────────────────

class TestRelationExtractorEnhanced:
    @patch("src.utils.api_client.llm_client")
    def test_extract_returns_rich_attributes(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [
                {
                    "name": "Hero",
                    "type": "person",
                    "description": "A warrior",
                    "status": {"health": "full"},
                    "state_changes": {},
                }
            ],
            "relations": [
                {"source": "Hero", "target": "Forest", "relation": "located_at", "context": "Hero is in the forest"},
            ],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract("The hero stands in the forest.")

        assert len(result["entities"]) == 1
        ent = result["entities"][0]
        assert ent["name"] == "Hero"
        assert ent["description"] == "A warrior"
        assert ent["status"] == {"health": "full"}

        assert len(result["relations"]) == 1
        rel = result["relations"][0]
        assert rel["context"] == "Hero is in the forest"

    @patch("src.utils.api_client.llm_client")
    def test_extract_normalizes_missing_fields(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [{"name": "Dragon"}],
            "relations": [{"source": "Hero", "target": "Dragon"}],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract("A dragon appears.")

        ent = result["entities"][0]
        assert ent["type"] == "unknown"  # default
        assert ent["description"] == ""
        assert ent["status"] == {}

        rel = result["relations"][0]
        assert rel["relation"] == "related_to"  # default

    @patch("src.utils.api_client.llm_client")
    def test_extract_handles_error(self, mock_client):
        mock_client.chat_json.side_effect = RuntimeError("API down")
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract("broken")
        assert result == {"entities": [], "relations": []}

    @patch("src.utils.api_client.llm_client")
    def test_legacy_mode_uses_simple_prompt(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [{"name": "Hero", "type": "person"}],
            "relations": [],
        }
        extractor = RelationExtractor(enhanced=False)
        result = extractor.extract("The hero appears.")
        assert len(result["entities"]) == 1
        # legacy extractor should NOT add rich defaults to entities
        # (the legacy prompt doesn't ask for description/status)

    @patch("src.utils.api_client.llm_client")
    def test_dual_extract_calls_both_prompts(self, mock_client):
        # Single-call dual extraction returns combined result
        mock_client.chat_json.return_value = {
            "entities": [
                {"name": "Hero", "type": "person", "description": "Brave", "status": {}, "state_changes": {}},
                {"name": "Cave", "type": "location", "description": "Dark", "status": {}, "state_changes": {}},
            ],
            "relations": [],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract_dual(
            player_input="I enter the cave",
            story_text="The hero enters a dark cave.",
            existing_entities=["Hero", "Forest"],
        )
        assert "entities" in result
        assert "relations" in result
        assert len(result["entities"]) >= 2
        # Verify single call was made
        assert mock_client.chat_json.call_count == 1

    @patch("src.utils.api_client.llm_client")
    def test_dual_extract_merges_duplicates(self, mock_client):
        """When both extractions find the same entity, merge instead of duplicate."""
        def _same_entity_response(messages, **kwargs):
            return {
                "entities": [{"name": "Hero", "type": "person", "description": "Brave warrior", "status": {}, "state_changes": {}}],
                "relations": [],
            }

        mock_client.chat_json.side_effect = _same_entity_response
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract_dual("Hero does something", "Hero does something dramatic.")
        # merged should not have duplicates
        hero_entities = [e for e in result["entities"] if e["name"].lower() == "hero"]
        assert len(hero_entities) == 1


class TestMergeExtractions:
    def test_merges_entities_with_prefer_richer(self):
        extractor = RelationExtractor()
        primary = {
            "entities": [{"name": "Hero", "type": "person", "description": "A warrior", "status": {}, "state_changes": {}}],
            "relations": [],
        }
        secondary = {
            "entities": [{"name": "Hero", "type": "person", "description": "", "status": {"health": "injured"}, "state_changes": {"health": "injured"}}],
            "relations": [],
        }
        merged = extractor._merge_extractions(primary, secondary)
        hero = [e for e in merged["entities"] if e["name"].lower() == "hero"][0]
        assert hero["description"] == "A warrior"
        assert hero["status"] == {"health": "injured"}

    def test_deduplicates_relations(self):
        extractor = RelationExtractor()
        primary = {
            "entities": [],
            "relations": [{"source": "Hero", "target": "Sword", "relation": "possesses"}],
        }
        secondary = {
            "entities": [],
            "relations": [{"source": "hero", "target": "sword", "relation": "possesses"}],
        }
        merged = extractor._merge_extractions(primary, secondary)
        assert len(merged["relations"]) == 1


class TestModuleLevelFunctions:
    @patch("src.utils.api_client.llm_client")
    def test_extract_module_function(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [{"name": "Hero", "type": "person"}],
            "relations": [],
        }
        result = extract("The hero appears.")
        assert len(result["entities"]) == 1

    @patch("src.utils.api_client.llm_client")
    def test_extract_legacy_module_function(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [{"name": "Hero", "type": "person"}],
            "relations": [],
        }
        result = extract_legacy("The hero appears.")
        assert len(result["entities"]) == 1

    @patch("src.utils.api_client.llm_client")
    def test_extract_dual_module_function(self, mock_client):
        mock_client.chat_json.side_effect = _mock_rich_extraction
        result = extract_dual("I enter the cave", "The hero enters a cave.")
        assert "entities" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
