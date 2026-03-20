"""Entity extraction: spaCy NER + noun-phrase heuristic + KG context.

Enhanced with:
- Expanded word lists (60+ creatures, 50+ locations, 50+ items, magic words)
- KG context-assisted extraction (fuzzy matching against known entities)
- Multi-word entity preservation
- Possessive handling ("dragon's lair" → dragon)

Returns list of dicts: {"text", "type", "start", "end", "source"}.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

from config import settings

logger = logging.getLogger(__name__)

# Map spaCy NER labels to our game types (validated against KG_ENTITY_TYPES)
_LABEL_MAP_RAW: Dict[str, str] = {
    "PERSON": "person",
    "ORG": "person",       # factions / groups → person
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "EVENT": "event",
    "PRODUCT": "item",
    "WORK_OF_ART": "item",
}

LABEL_MAP: Dict[str, str] = {}
for _spacy_label, _game_type in _LABEL_MAP_RAW.items():
    if _game_type in settings.KG_ENTITY_TYPES:
        LABEL_MAP[_spacy_label] = _game_type
    else:
        logger.warning(
            "LABEL_MAP value '%s' not in KG_ENTITY_TYPES, mapping '%s' -> 'unknown'",
            _game_type, _spacy_label,
        )
        LABEL_MAP[_spacy_label] = "unknown"

# ── Expanded heuristic word sets ──────────────────────────

_CREATURE_WORDS: Set[str] = {
    # Original (27)
    "dragon", "wolf", "goblin", "troll", "orc", "spider", "serpent",
    "bear", "beast", "demon", "wraith", "ghost", "undead", "skeleton",
    "rat", "bat", "griffin", "phoenix", "basilisk", "hydra", "wyvern",
    # Expanded fantasy creatures (35+)
    "zombie", "vampire", "werewolf", "lich", "imp", "djinn", "elemental",
    "golem", "harpy", "manticore", "chimera", "minotaur", "cyclops",
    "centaur", "fairy", "elf", "dwarf", "hobbit", "pixie", "sprite",
    "nymph", "dryad", "ogre", "gnoll", "kobold", "bugbear", "hobgoblin",
    "wyrm", "drake", "cockatrice", "beholder", "displacer", "owlbear",
    "horse", "eagle", "hawk", "snake", "lion", "tiger", "fox", "deer",
    "warrior", "knight", "mage", "archer", "rogue", "paladin", "cleric",
}

_LOCATION_WORDS: Set[str] = {
    # Original (27)
    "forest", "cave", "castle", "village", "town", "city", "river",
    "mountain", "temple", "dungeon", "tower", "bridge", "lake", "desert",
    "island", "palace", "fortress", "ruins", "shrine", "tavern", "market",
    "harbor", "garden", "cemetery", "mine", "swamp", "valley", "cliff",
    # Expanded (25+)
    "lair", "den", "pit", "abyss", "chasm", "cavern", "grotto", "hollow",
    "plains", "meadow", "field", "ocean", "sea", "bay", "port", "docks",
    "camp", "campsite", "outpost", "fort", "barracks", "academy", "library",
    "church", "sanctuary", "cathedral", "crypt", "tomb", "catacombs",
    "sewer", "passage", "corridor", "hall", "chamber", "throne",
    "kingdom", "realm", "domain", "territory", "land", "world",
}

_ITEM_WORDS: Set[str] = {
    # Original (30)
    "sword", "shield", "potion", "scroll", "amulet", "ring", "staff",
    "bow", "arrow", "gem", "key", "map", "book", "dagger", "axe",
    "armor", "helmet", "cloak", "orb", "crystal", "relic", "artifact",
    "tome", "wand", "lantern", "rope", "torch", "coin", "chest",
    # Expanded (25+)
    "blade", "spear", "mace", "flail", "halberd", "crossbow", "bolt",
    "quiver", "gauntlet", "greaves", "boots", "belt", "crown", "tiara",
    "necklace", "bracelet", "brooch", "pendant", "chalice", "goblet",
    "vial", "flask", "bottle", "bag", "sack", "pack", "crate", "box",
    "mirror", "compass", "hourglass", "instrument", "lute", "harp",
    "letter", "note", "diary", "journal", "contract", "deed",
    "food", "bread", "meat", "water", "wine", "ale", "herb", "flower",
    "seed", "stone", "rock", "boulder", "log", "plank", "nail", "chain",
    "rope", "hook", "lock", "trap", "bomb", "mine", "signal", "flag",
}

_MAGIC_WORDS: Set[str] = {
    "fireball", "lightning", "teleport", "heal", "curse", "charm",
    "shield", "barrier", "ward", "illusion", "invisibility", "levitate",
    "summon", "banish", "enchant", "disenchant", "dispel", "silence",
    "sleep", "fear", "haste", "slow", "polymorph", "transform",
    "spell", "incantation", "ritistic", "hex", "jinx", "charm",
    "blessing", "aura", "sigil", "glyph", "rune", "seal",
}

# All combined for quick lookup
_ALL_TYPED_WORDS: Dict[str, str] = {}
for _w in _CREATURE_WORDS:
    _ALL_TYPED_WORDS[_w] = "creature"
for _w in _LOCATION_WORDS:
    _ALL_TYPED_WORDS[_w] = "location"
for _w in _ITEM_WORDS:
    _ALL_TYPED_WORDS[_w] = "item"
for _w in _MAGIC_WORDS:
    _ALL_TYPED_WORDS[_w] = "item"  # Magic items → item type


class EntityExtractor:
    """spaCy NER + noun-phrase extraction with type inference and KG context."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model_name = spacy_model
        self.nlp = None

    def load(self) -> None:
        try:
            import spacy
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info("spaCy model loaded: %s", self.spacy_model_name)
        except Exception as exc:
            logger.warning("spaCy load failed (%s) – using noun-phrase only.", exc)
            self.nlp = None

    # ── public API ────────────────────────────────────────
    def extract(
        self,
        text: str,
        known_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        """Return a deduplicated list of entity dicts.

        Args:
            text: The text to extract entities from.
            known_entities: Optional list of known entity names from the KG,
                used for fuzzy-matching and type inference.
        """
        entities: List[Dict[str, object]] = []
        if self.nlp:
            entities.extend(self._spacy_extract(text))
        entities.extend(self._noun_phrase_extract(text))

        # KG context-assisted enrichment
        if known_entities:
            entities = self._enrich_with_kg_context(entities, known_entities, text)

        return self._deduplicate(entities)

    # ── spaCy NER ─────────────────────────────────────────
    def _spacy_extract(self, text: str) -> List[Dict[str, object]]:
        doc = self.nlp(text)  # type: ignore[union-attr]
        results: List[Dict[str, object]] = []
        for ent in doc.ents:
            game_type = LABEL_MAP.get(ent.label_)
            if game_type:
                results.append({
                    "text": ent.text,
                    "type": game_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy",
                })
        return results

    # ── noun phrase + heuristic type ──────────────────────
    def _noun_phrase_extract(self, text: str) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        if self.nlp:
            doc = self.nlp(text)  # type: ignore[union-attr]
            for chunk in doc.noun_chunks:
                # skip very short / pronoun chunks
                head = chunk.root.text.lower()
                if head in ("i", "you", "he", "she", "it", "we", "they", "me", "my"):
                    continue

                # Handle possessives: "dragon's lair" → extract "dragon"
                chunk_text = chunk.text
                possessive_match = re.match(r"^(\w+)'s\s+", chunk_text)
                if possessive_match:
                    possessor = possessive_match.group(1)
                    possessor_type = self._infer_type(possessor)
                    if possessor_type:
                        results.append({
                            "text": possessor,
                            "type": possessor_type,
                            "start": chunk.start_char,
                            "end": chunk.start_char + len(possessor),
                            "source": "possessive",
                        })

                # Infer type from head word or entire phrase
                inferred = self._infer_type(head)
                if not inferred:
                    # Try inferring from the whole chunk text
                    inferred = self._infer_chunk_type(chunk.text)

                if inferred:
                    results.append({
                        "text": chunk.text,
                        "type": inferred,
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "source": "noun_phrase",
                    })
        else:
            # Regex fallback for common patterns
            for word in re.findall(r"\b[A-Za-z]{3,}\b", text):
                inferred = self._infer_type(word.lower())
                if inferred:
                    start = text.lower().find(word.lower())
                    results.append({
                        "text": word,
                        "type": inferred,
                        "start": start,
                        "end": start + len(word),
                        "source": "regex",
                    })
        return results

    def _infer_type(self, word: str) -> Optional[str]:
        """Infer entity type from a single word."""
        return _ALL_TYPED_WORDS.get(word.lower())

    def _infer_chunk_type(self, chunk_text: str) -> Optional[str]:
        """Infer entity type from a noun phrase chunk by checking all words."""
        words = chunk_text.lower().split()
        # Check if any word in the chunk matches a known type
        # Prefer more specific types: creature > location > item
        for word in words:
            w = word.strip("'s")
            if w in _CREATURE_WORDS:
                return "creature"
        for word in words:
            w = word.strip("'s")
            if w in _LOCATION_WORDS:
                return "location"
        for word in words:
            w = word.strip("'s")
            if w in _ITEM_WORDS:
                return "item"
        return None

    # ── KG context enrichment ─────────────────────────────
    def _enrich_with_kg_context(
        self,
        entities: List[Dict[str, object]],
        known_entities: List[str],
        original_text: str = "",
    ) -> List[Dict[str, object]]:
        """Enrich extracted entities by fuzzy-matching against KG entities.

        - If an extracted entity fuzzy-matches a known entity, use the known name.
        - If the original text mentions a known entity that wasn't extracted, add it.
        """
        if not known_entities:
            return entities

        # Build set of already extracted names (lowered)
        extracted_names: Set[str] = {str(e["text"]).lower() for e in entities}
        known_lower_map: Dict[str, str] = {k.lower(): k for k in known_entities}

        # Enrich existing entities via fuzzy matching
        for ent in entities:
            ent_name = str(ent["text"]).lower()
            best_match = self._fuzzy_match(ent_name, list(known_lower_map.keys()), threshold=0.8)
            if best_match and best_match != ent_name:
                ent["text"] = known_lower_map[best_match]
                ent["source"] = str(ent.get("source", "")) + "+kg"

        # Scan original text for known entities that weren't extracted
        text_lower = original_text.lower()
        for known_name in known_entities:
            known_lower = known_name.lower()
            if known_lower in text_lower and known_lower not in extracted_names:
                pos = text_lower.find(known_lower)
                entities.append({
                    "text": known_name,
                    "type": "unknown",
                    "start": pos,
                    "end": pos + len(known_name),
                    "source": "kg_context",
                })

        return entities

    @staticmethod
    def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.8) -> Optional[str]:
        """Find the best fuzzy match for query in candidates."""
        if not query or not candidates:
            return None
        best_score = 0.0
        best_match = None
        for candidate in candidates:
            score = SequenceMatcher(None, query, candidate).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        return best_match

    # ── dedup ─────────────────────────────────────────────
    @staticmethod
    def _deduplicate(entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen: Dict[str, Dict[str, object]] = {}
        for ent in entities:
            key = str(ent["text"]).lower()
            # prefer spacy source
            existing_source = seen.get(key, {}).get("source", "")
            new_source = str(ent.get("source", ""))
            if key not in seen or "spacy" in new_source:
                seen[key] = ent
            elif "kg_context" in new_source and "spacy" not in existing_source:
                # Merge kg_context info into existing entity
                existing = seen[key]
                if existing.get("type") == "unknown" and ent.get("type") != "unknown":
                    existing["type"] = ent["type"]
        return list(seen.values())
