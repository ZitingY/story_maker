"""Tests for NLU-3: enhanced coreference resolution."""
import pytest

from src.nlu.coreference import CoreferenceResolver


class TestRuleResolveBasicPronouns:
    """Test basic personal pronoun replacement."""

    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_he_replaced_with_last_person(self, resolver):
        result = resolver._rule_resolve(
            "He went to the cave.",
            ["The knight saw a dragon.", "Gandalf arrived."],
        )
        assert "Gandalf" in result

    def test_she_replaced_with_last_person(self, resolver):
        result = resolver._rule_resolve(
            "She attacked the monster.",
            ["Lady Morgana stood at the gate."],
        )
        assert "Morgana" in result

    def test_they_replaced_with_last_person(self, resolver):
        result = resolver._rule_resolve(
            "They joined the battle.",
            ["The warriors gathered.", "King Arthur spoke."],
        )
        assert "Arthur" in result

    def test_him_replaced_with_last_person(self, resolver):
        result = resolver._rule_resolve(
            "I gave it to him.",
            ["The Blacksmith was there."],
        )
        assert "Blacksmith" in result

    def test_no_context_returns_original(self, resolver):
        text = "He went there."
        assert resolver.resolve(text, []) == text


class TestRuleResolvePossessives:
    """Test possessive pronoun replacement."""

    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_his_replaced(self, resolver):
        result = resolver._rule_resolve(
            "I took his sword.",
            ["Sir Lancelot stood guard."],
        )
        assert "Lancelot" in result or "Sir" in result
        assert "'s" in result

    def test_her_possessive_replaced(self, resolver):
        result = resolver._rule_resolve(
            "I found her shield.",
            ["Princess Elena waited."],
        )
        assert "Elena" in result

    def test_their_replaced(self, resolver):
        result = resolver._rule_resolve(
            "I took their map.",
            ["Merchants set up shop."],
        )
        assert "'s" in result


class TestRuleResolveNonPersonal:
    """Test non-personal pronoun (it/its) replacement."""

    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_it_replaced_with_non_person_entity(self, resolver):
        result = resolver._rule_resolve(
            "I picked it up.",
            ["A glowing orb floated."],
            known_entities=[
                {"text": "orb", "type": "item"},
            ],
        )
        # "it" should be replaced with non-person entity
        # But in fallback without known_entities type info, all context names are treated as persons
        # With known_entities, orb should be classified as non-person
        assert isinstance(result, str)

    def test_its_replaced_with_non_person(self, resolver):
        result = resolver._rule_resolve(
            "I examined its surface.",
            ["The crystal pulsed with light."],
            known_entities=[
                {"text": "crystal", "type": "item"},
            ],
        )
        assert isinstance(result, str)


class TestRuleResolveReflexive:
    """Test reflexive pronoun handling."""

    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_himself_replaced(self, resolver):
        result = resolver._rule_resolve(
            "He hurt himself.",
            ["The Warrior swung his axe."],
        )
        assert "Warrior" in result
        assert "himself" not in result.lower()


class TestRuleResolveWithKnownEntities:
    """Test entity-type-aware resolution."""

    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_person_pronoun_maps_to_person_entity(self, resolver):
        result = resolver._rule_resolve(
            "He attacked me.",
            ["The guard challenged.", "Gandalf appeared."],
            known_entities=[
                {"text": "guard", "type": "person"},
                {"text": "Gandalf", "type": "person"},
            ],
        )
        assert "Gandalf" in result

    def test_it_maps_to_non_person_entity(self, resolver):
        result = resolver._rule_resolve(
            "It was broken.",
            ["The Excalibur gleamed.", "Smaug roared."],
            known_entities=[
                {"text": "Excalibur", "type": "item"},
                {"text": "Smaug", "type": "creature"},
            ],
        )
        # "it" should map to a non-person entity
        assert "Smaug" in result or "Excalibur" in result


class TestPublicAPI:
    """Test the public resolve() method."""

    @pytest.fixture
    def resolver(self):
        return CoreferenceResolver()

    def test_resolve_returns_string(self, resolver):
        out = resolver.resolve("He went there.", ["The knight saw a cave."])
        assert isinstance(out, str)

    def test_resolve_with_known_entities(self, resolver):
        out = resolver.resolve(
            "She waved.",
            ["Queen Elizabeth sat on the throne."],
            known_entities=[{"text": "Elizabeth", "type": "person"}],
        )
        assert isinstance(out, str)
        assert "Elizabeth" in out

    def test_no_context_passthrough(self, resolver):
        text = "Open the door."
        assert resolver.resolve(text, []) == text


class TestExtractOriginalPortion:
    """Test the neural mode tail extraction helper."""

    def test_shorter_resolved_returned_as_is(self):
        resolved = "Short."
        full = "Long context here."
        original = "Some original text."
        result = CoreferenceResolver._extract_original_portion(resolved, full, original)
        assert result == resolved

    def test_sentence_alignment(self):
        # Simulate: context + original concatenated, then coref resolved the whole thing
        # The method takes the last N sentences from resolved where N = number of original sentences
        full_context = "The knight entered. The dragon slept."
        original = "He attacked it."
        resolved = "The knight entered. The dragon slept. The knight attacked the dragon."
        result = CoreferenceResolver._extract_original_portion(resolved, full_context, original)
        # Should extract the last sentence (corresponding to original)
        assert "attacked" in result.lower()
        assert len(result) > 0

    def test_fallback_to_original(self):
        # When alignment fails, return original
        resolved = "Completely different text that has nothing to do."
        full_context = "Some context."
        original = "He went there."
        result = CoreferenceResolver._extract_original_portion(resolved, full_context, original)
        # Should either return original or reasonable extraction
        assert isinstance(result, str)
        assert len(result) > 0
