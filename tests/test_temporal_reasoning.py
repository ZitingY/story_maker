"""Tests for KG-4: temporal and causal reasoning."""
import pytest

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.conflict_detector import ConflictDetector


class TestStatusHistory:
    """Test entity status history tracking."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", status={"health": "full"}, turn_id=1)
        return kg

    def test_initial_status_no_history(self, kg):
        hero = kg.get_entity("hero")
        assert hero.get("status_history", []) == []

    def test_status_change_creates_history(self, kg):
        kg.add_entity("Hero", "person", status={"health": "injured"}, turn_id=3)
        hero = kg.get_entity("hero")
        history = hero.get("status_history", [])
        assert len(history) >= 1
        assert history[-1]["turn"] == 3
        assert "health" in history[-1]["changes"]

    def test_multiple_status_changes(self, kg):
        kg.add_entity("Hero", "person", status={"health": "injured"}, turn_id=3)
        kg.add_entity("Hero", "person", status={"health": "critical"}, turn_id=5)
        history = kg.get_entity("hero").get("status_history", [])
        assert len(history) >= 2

    def test_status_history_capped_at_10(self, kg):
        for i in range(15):
            kg.add_entity("Hero", "person", status={"turn": str(i)}, turn_id=i + 2)
        history = kg.get_entity("hero").get("status_history", [])
        assert len(history) <= 10

    def test_new_field_not_in_history(self, kg):
        # Adding a new field shouldn't create a history entry
        kg.add_entity("Hero", "person", status={"mood": "happy"}, turn_id=3)
        hero = kg.get_entity("hero")
        history = hero.get("status_history", [])
        # "mood" is new, should have "(new)" in changes
        if history:
            changes = history[-1].get("changes", {})
            assert "mood" in changes


class TestGetEntityHistory:
    """Test get_entity_history method."""

    def test_returns_list(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        assert kg.get_entity_history("hero") == []

    def test_returns_history_entries(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", status={"health": "full"}, turn_id=1)
        kg.add_entity("Hero", "person", status={"health": "injured"}, turn_id=3)
        history = kg.get_entity_history("hero")
        assert len(history) >= 1
        assert history[0]["turn"] == 3

    def test_nonexistent_entity_returns_empty(self):
        kg = KnowledgeGraph()
        assert kg.get_entity_history("nonexistent") == []


class TestGetEntityStatusAtTurn:
    """Test get_entity_status_at_turn method."""

    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", status={"health": "full", "mood": "calm"}, turn_id=1)
        kg.add_entity("Hero", "person", status={"health": "injured"}, turn_id=5)
        kg.add_entity("Hero", "person", status={"health": "critical", "mood": "desperate"}, turn_id=8)
        return kg

    def test_status_at_creation(self, kg):
        status = kg.get_entity_status_at_turn("hero", 1)
        assert status.get("health") == "full"

    def test_status_after_first_change(self, kg):
        status = kg.get_entity_status_at_turn("hero", 5)
        assert status.get("health") == "injured"

    def test_status_at_final(self, kg):
        status = kg.get_entity_status_at_turn("hero", 10)
        assert status.get("health") == "critical"

    def test_before_creation_returns_empty(self, kg):
        status = kg.get_entity_status_at_turn("hero", 0)
        assert status == {}

    def test_nonexistent_entity_returns_empty(self):
        kg = KnowledgeGraph()
        assert kg.get_entity_status_at_turn("ghost", 5) == {}


class TestCausalRelationTypes:
    """Test that causal relation types are in config."""

    def test_causes_in_config(self):
        from config import settings
        assert "causes" in settings.KG_RELATION_TYPES

    def test_prevents_in_config(self):
        from config import settings
        assert "prevents" in settings.KG_RELATION_TYPES

    def test_enables_in_config(self):
        from config import settings
        assert "enables" in settings.KG_RELATION_TYPES

    def test_follows_in_config(self):
        from config import settings
        assert "follows" in settings.KG_RELATION_TYPES


class TestTemporalConflictDetection:
    """Test temporal conflict detection."""

    def test_dead_entity_action_detected(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", status={"status": "dead"}, turn_id=3)
        kg.add_entity("Sword", "item", turn_id=1)
        # Add relation after death
        kg.add_relation("Hero", "Sword", "possesses", turn_id=5)
        det = ConflictDetector(kg)
        conflicts = det._temporal_check()
        dead_conflicts = [c for c in conflicts if c.get("subtype") == "dead_entity_action"]
        assert len(dead_conflicts) >= 1

    def test_causal_inversion_detected(self):
        kg = KnowledgeGraph()
        # B created after A, but A causes B — this is fine
        kg.add_entity("Trap", "item", turn_id=3)
        kg.add_entity("Injury", "event", turn_id=5)
        kg.add_relation("Trap", "Injury", "causes", turn_id=5)
        det = ConflictDetector(kg)
        conflicts = det._temporal_check()
        assert len([c for c in conflicts if c.get("subtype") == "causal_inversion"]) == 0

    def test_causal_inversion_real(self):
        kg = KnowledgeGraph()
        # B created BEFORE A, but A causes B — this is a conflict
        kg.add_entity("Injury", "event", turn_id=2)
        kg.add_entity("Trap", "item", turn_id=5)
        kg.add_relation("Trap", "Injury", "causes", turn_id=5)
        det = ConflictDetector(kg)
        conflicts = det._temporal_check()
        inversions = [c for c in conflicts if c.get("subtype") == "causal_inversion"]
        assert len(inversions) >= 1

    def test_no_temporal_conflicts_when_clean(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1)
        kg.add_entity("Forest", "location", turn_id=1)
        kg.add_relation("Hero", "Forest", "located_at", turn_id=1)
        det = ConflictDetector(kg)
        conflicts = det._temporal_check()
        assert conflicts == []


class TestSummaryStatusHistory:
    """Test that KG summary includes status history."""

    def test_summary_shows_history(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", status={"health": "full"}, turn_id=1, is_player_mentioned=True)
        kg.add_entity("Hero", "person", status={"health": "injured"}, turn_id=5)
        kg.add_entity("Hero", "person", status={"health": "critical"}, turn_id=8)
        summary = kg.to_summary()
        assert "History:" in summary
        assert "turn 5" in summary or "turn 8" in summary

    def test_summary_shows_emotion(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1, emotion="joy", is_player_mentioned=True)
        summary = kg.to_summary()
        assert "Emotion: joy" in summary

    def test_empty_history_not_shown(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person", turn_id=1, is_player_mentioned=True)
        summary = kg.to_summary()
        assert "History:" not in summary
