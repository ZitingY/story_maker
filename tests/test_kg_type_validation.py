"""Tests for KG-3: entity type normalization and validation."""
import pytest
from unittest.mock import patch

from src.knowledge_graph.relation_extractor import _normalize_type, RelationExtractor
from src.knowledge_graph.graph import KnowledgeGraph


class TestNormalizeType:
    """Test _normalize_type function."""

    def test_exact_match(self):
        assert _normalize_type("person") == "person"
        assert _normalize_type("location") == "location"
        assert _normalize_type("item") == "item"
        assert _normalize_type("creature") == "creature"
        assert _normalize_type("event") == "event"

    def test_case_insensitive(self):
        assert _normalize_type("PERSON") == "person"
        assert _normalize_type("Location") == "location"
        assert _normalize_type("ITEM") == "item"

    def test_synonym_mapping(self):
        assert _normalize_type("character") == "person"
        assert _normalize_type("npc") == "person"
        assert _normalize_type("weapon") == "item"
        assert _normalize_type("place") == "location"
        assert _normalize_type("animal") == "creature"
        assert _normalize_type("quest") == "event"
        assert _normalize_type("monster") == "creature"

    def test_unknown_returns_unknown(self):
        assert _normalize_type("foobar") == "unknown"
        assert _normalize_type("xyz") == "unknown"

    def test_empty_returns_unknown(self):
        assert _normalize_type("") == "unknown"
        assert _normalize_type("   ") == "unknown"

    def test_thing_maps_to_item(self):
        assert _normalize_type("thing") == "item"
        assert _normalize_type("object") == "item"

    def test_whitespace_stripped(self):
        assert _normalize_type("  person  ") == "person"


class TestKGEntityTypeValidation:
    """Test KnowledgeGraph.add_entity rejects invalid types."""

    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_valid_type_accepted(self, kg):
        kg.add_entity("Hero", "person")
        assert kg.get_entity("hero")["entity_type"] == "person"

    def test_unknown_type_accepted(self, kg):
        kg.add_entity("Mystery", "unknown")
        assert kg.get_entity("mystery")["entity_type"] == "unknown"

    def test_invalid_type_falls_back_to_unknown(self, kg):
        kg.add_entity("WeirdThing", "foobar")
        assert kg.get_entity("weirdthing")["entity_type"] == "unknown"

    def test_all_valid_types_accepted(self, kg):
        for t in ["person", "location", "item", "creature", "event"]:
            kg.add_entity(f"entity_{t}", t)
            assert kg.get_entity(f"entity_{t}")["entity_type"] == t


class TestExtractorTypeNormalization:
    """Test that RelationExtractor normalizes types in extracted entities."""

    @patch("src.utils.api_client.llm_client")
    def test_extract_normalizes_types(self, mock_client):
        mock_client.chat_json.return_value = {
            "entities": [
                {"name": "Gandalf", "type": "character"},
                {"name": "Excalibur", "type": "weapon"},
                {"name": "Mordor", "type": "place"},
            ],
            "relations": [],
        }
        extractor = RelationExtractor(enhanced=True)
        result = extractor.extract("Gandalf carried Excalibur to Mordor.")
        types = {e["name"]: e["type"] for e in result["entities"]}
        assert types["Gandalf"] == "person"
        assert types["Excalibur"] == "item"
        assert types["Mordor"] == "location"


class TestEntityExtractorLabelMapValidation:
    """Test that EntityExtractor LABEL_MAP values are all valid KG_ENTITY_TYPES."""

    def test_label_map_values_valid(self):
        from src.nlu.entity_extractor import LABEL_MAP
        from config import settings

        for spacy_label, game_type in LABEL_MAP.items():
            assert game_type in settings.KG_ENTITY_TYPES or game_type == "unknown", (
                f"LABEL_MAP['{spacy_label}'] = '{game_type}' is not in KG_ENTITY_TYPES"
            )
