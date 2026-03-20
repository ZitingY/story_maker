"""Tests for KG-1: JSON persistence (save/load)."""
import json
import pytest
import tempfile
from pathlib import Path

from src.knowledge_graph.graph import KnowledgeGraph


class TestKnowledgeGraphToDict:
    """Test to_dict serialization."""

    def test_empty_graph(self):
        kg = KnowledgeGraph()
        data = kg.to_dict()
        assert data["version"] == 1
        assert data["turn"] == 0
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_with_entities(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", description="A brave warrior", turn_id=1)
        kg.add_entity("Sword", "item", turn_id=2)
        data = kg.to_dict()
        assert len(data["nodes"]) == 2
        node_keys = {n["key"] for n in data["nodes"]}
        assert "hero" in node_keys
        assert "sword" in node_keys

    def test_with_relations(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        kg.add_entity("Forest", "location", turn_id=1)
        kg.add_relation("Hero", "Forest", "located_at", turn_id=1, confidence=0.9)
        data = kg.to_dict()
        assert len(data["edges"]) == 1
        assert data["edges"][0]["source"] == "hero"
        assert data["edges"][0]["target"] == "forest"
        assert data["edges"][0]["relation"] == "located_at"
        assert data["edges"][0]["confidence"] == 0.9

    def test_turn_preserved(self):
        kg = KnowledgeGraph()
        kg.set_turn(42)
        data = kg.to_dict()
        assert data["turn"] == 42

    def test_json_serializable(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", description="A warrior", status={"health": "full"}, turn_id=1)
        kg.add_relation("Hero", "Cave", "located_at", context="Hero enters", turn_id=1)
        data = kg.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


class TestKnowledgeGraphFromDict:
    """Test from_dict deserialization."""

    def test_empty_data(self):
        kg = KnowledgeGraph.from_dict({"version": 1, "turn": 0, "nodes": [], "edges": []})
        assert kg.num_nodes == 0
        assert kg.num_edges == 0

    def test_round_trip(self):
        # Build original
        kg1 = KnowledgeGraph()
        kg1.set_turn(5)
        kg1.add_entity("Hero", "person", description="A brave warrior",
                       status={"health": "injured"}, turn_id=1, is_player_mentioned=True)
        kg1.add_entity("Sword", "item", turn_id=2)
        kg1.add_relation("Hero", "Sword", "possesses", context="Found in cave", turn_id=3, confidence=0.85)

        # Serialize and deserialize
        data = kg1.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)

        # Verify
        assert kg2._current_turn == 5
        assert kg2.num_nodes == 2
        assert kg2.num_edges == 1

        hero = kg2.get_entity("hero")
        assert hero is not None
        assert hero["entity_type"] == "person"
        assert hero["description"] == "A brave warrior"
        assert hero["status"]["health"] == "injured"
        assert hero["mention_count"] == 1

        rels = kg2.get_relations("hero")
        assert len(rels) == 1
        assert rels[0]["relation"] == "possesses"
        assert rels[0]["confidence"] == 0.85

    def test_preserves_node_attributes(self):
        kg1 = KnowledgeGraph()
        kg1.add_entity("Dragon", "creature", description="Fearsome beast",
                       status={"mood": "angry"}, turn_id=3, is_player_mentioned=True)
        data = kg1.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)

        dragon = kg2.get_entity("dragon")
        assert dragon["entity_type"] == "creature"
        assert dragon["description"] == "Fearsome beast"
        assert dragon["status"]["mood"] == "angry"
        assert dragon["created_turn"] == 3


class TestKnowledgeGraphSaveLoad:
    """Test save() and load() file operations."""

    def test_save_creates_file(self, tmp_path):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        filepath = str(tmp_path / "test_kg.json")
        kg.save(filepath)
        assert Path(filepath).exists()

    def test_save_content_valid_json(self, tmp_path):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        filepath = str(tmp_path / "test_kg.json")
        kg.save(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["nodes"]) == 1

    def test_load_restores_graph(self, tmp_path):
        # Save
        kg1 = KnowledgeGraph()
        kg1.set_turn(10)
        kg1.add_entity("Hero", "person", description="Warrior", turn_id=1)
        kg1.add_entity("Dragon", "creature", turn_id=5)
        kg1.add_relation("Hero", "Dragon", "enemy_of", turn_id=7, confidence=0.9)
        filepath = str(tmp_path / "test_kg.json")
        kg1.save(filepath)

        # Load
        kg2 = KnowledgeGraph.load(filepath)
        assert kg2._current_turn == 10
        assert kg2.num_nodes == 2
        assert kg2.num_edges == 1
        assert kg2.get_entity("hero")["description"] == "Warrior"
        rels = kg2.get_relations("hero")
        assert any(r["relation"] == "enemy_of" for r in rels)

    def test_load_nonexistent_returns_empty(self):
        kg = KnowledgeGraph.load("/nonexistent/path/test.json")
        assert kg.num_nodes == 0
        assert kg.num_edges == 0

    def test_round_trip_file(self, tmp_path):
        """Full round-trip: build graph -> save -> load -> verify."""
        kg1 = KnowledgeGraph()
        kg1.set_turn(15)
        kg1.add_entity("Hero", "person", status={"health": "full"}, turn_id=1, is_player_mentioned=True)
        kg1.add_entity("Cave", "location", description="Dark and damp", turn_id=2)
        kg1.add_entity("Torch", "item", turn_id=3)
        kg1.add_relation("Hero", "Cave", "located_at", context="Hero entered cave", turn_id=4)
        kg1.add_relation("Hero", "Torch", "possesses", turn_id=5, confidence=0.8)

        filepath = str(tmp_path / "roundtrip.json")
        kg1.save(filepath)
        kg2 = KnowledgeGraph.load(filepath)

        # Nodes
        assert kg2.num_nodes == 3
        assert kg2.get_entity("hero")["status"]["health"] == "full"
        assert kg2.get_entity("cave")["description"] == "Dark and damp"
        assert kg2.get_entity("torch")["entity_type"] == "item"

        # Edges
        assert kg2.num_edges == 2
        hero_rels = kg2.get_relations("hero")
        rel_types = {r["relation"] for r in hero_rels}
        assert "located_at" in rel_types
        assert "possesses" in rel_types

        # Turn
        assert kg2._current_turn == 15

    def test_save_creates_parent_dirs(self, tmp_path):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        filepath = str(tmp_path / "sub" / "dir" / "test_kg.json")
        kg.save(filepath)
        assert Path(filepath).exists()


class TestAutoSave:
    """Test auto-save configuration."""

    def test_auto_save_config_exists(self):
        from config import settings
        assert hasattr(settings, "KG_AUTO_SAVE")
        assert hasattr(settings, "KG_SAVE_DIR")
        assert hasattr(settings, "KG_SNAPSHOT_INTERVAL")
        assert isinstance(settings.KG_AUTO_SAVE, bool)
        assert isinstance(settings.KG_SNAPSHOT_INTERVAL, int)
