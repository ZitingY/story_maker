"""LLM-based entity + relation extraction from story text.

Sends text to the OpenAI API (via ``llm_client``) in JSON mode and returns
structured entities and relations for KG ingestion.

Enhanced with:
- Rich entity attributes (description, status, state_changes)
- Rich relation attributes (context)
- Dual extraction mode (player input + story text)
- State change detection for existing entities
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# Synonym mapping: non-standard type names → canonical KG_ENTITY_TYPES
_TYPE_SYNONYMS: Dict[str, str] = {
    "character": "person",
    "npc": "person",
    "villain": "person",
    "hero": "person",
    "king": "person",
    "queen": "person",
    "guard": "person",
    "merchant": "person",
    "wizard": "person",
    "warrior": "person",
    "place": "location",
    "room": "location",
    "area": "location",
    "region": "location",
    "building": "location",
    "kingdom": "location",
    "weapon": "item",
    "armor": "item",
    "tool": "item",
    "object": "item",
    "artifact": "item",
    "relic": "item",
    "treasure": "item",
    "thing": "item",
    "animal": "creature",
    "monster": "creature",
    "beast": "creature",
    "enemy": "creature",
    "boss": "creature",
    "quest": "event",
    "mission": "event",
    "battle": "event",
    "encounter": "event",
    "situation": "event",
}


def _normalize_type(raw_type: str) -> str:
    """Normalize an entity type string to a canonical KG_ENTITY_TYPES value.

    Returns 'unknown' if the type cannot be mapped.
    """
    normalized = raw_type.strip().lower()
    if not normalized:
        return "unknown"
    # Direct match
    if normalized in settings.KG_ENTITY_TYPES:
        return normalized
    # Synonym match
    mapped = _TYPE_SYNONYMS.get(normalized)
    if mapped and mapped in settings.KG_ENTITY_TYPES:
        return mapped
    return "unknown"

_EXTRACTION_SYSTEM = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "Given a passage of story text, extract entities and relations with rich attributes.\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event", '
    '"description": str, "status": {"key": "value"}, "state_changes": {"key": "value"}}], '
    '"relations": [{"source": str, "target": str, "relation": str, "context": str}]}\n'
    "- description: a brief narrative description of the entity\n"
    "- status: current dynamic state (e.g. health, mood, location)\n"
    "- state_changes: state fields that changed in this passage (e.g. {\"health\": \"injured\"})\n"
    "- context: brief description of how this relation was established or confirmed\n"
    "- Keep relation types simple and lowercase: located_at, possesses, ally_of, enemy_of, knows, part_of, caused_by, has_attribute\n"
    "- Also extract causal relations when present:\n"
    '  "causes": A directly causes B\n'
    '  "prevents": A prevents B from happening\n'
    '  "enables": A enables B to happen\n'
    '  "follows": B temporally follows A (for event chains)\n'
    "- If an entity already exists, focus on state_changes rather than duplicating description\n"
    "- Be thorough but avoid trivial or redundant extractions"
)

_EXTRACTION_SYSTEM_LEGACY = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "Given a passage of story text, extract entities and relations.\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event"}], '
    '"relations": [{"source": str, "target": str, "relation": str}]}\n'
    "Keep relation types simple and lowercase, e.g. located_at, possesses, ally_of, enemy_of, knows, etc."
)

_EXTRACTION_SYSTEM_DUAL = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "You will receive two text passages:\n"
    "1. PLAYER INPUT: the player's action or dialogue\n"
    "2. STORY TEXT: the narrator's response describing events\n\n"
    "Extract entities and relations from BOTH passages.\n"
    "Player input typically contains the player's intended actions and newly mentioned entities.\n"
    "Story text contains narrative events, NPC reactions, and world state changes.\n\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event", '
    '"description": str, "status": {"key": "value"}, "state_changes": {"key": "value"}}], '
    '"relations": [{"source": str, "target": str, "relation": str, "context": str}]}\n'
    "- description: a brief narrative description of the entity\n"
    "- status: current dynamic state (e.g. health, mood, location)\n"
    "- state_changes: state fields that changed in this passage (e.g. {\"health\": \"injured\"})\n"
    "- context: brief description of how this relation was established or confirmed\n"
    "- Keep relation types simple and lowercase: located_at, possesses, ally_of, enemy_of, knows, part_of, caused_by, has_attribute\n"
    "- Also extract causal relations when present:\n"
    '  "causes": A directly causes B\n'
    '  "prevents": A prevents B from happening\n'
    '  "enables": A enables B to happen\n'
    '  "follows": B temporally follows A (for event chains)\n'
    "- If an entity already exists, focus on state_changes rather than duplicating description\n"
    "- Merge entities mentioned in both passages into a single entry\n"
    "- Be thorough but avoid trivial or redundant extractions"
)


class RelationExtractor:
    """Extract entities and relations from story text using the LLM."""

    def __init__(self, enhanced: bool = True) -> None:
        self.enhanced = enhanced

    def extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return ``{"entities": [...], "relations": [...]}`` parsed from LLM JSON.

        On any failure an empty result is returned so the game never crashes.
        """
        try:
            from src.utils.api_client import llm_client

            system_prompt = _EXTRACTION_SYSTEM if self.enhanced else _EXTRACTION_SYSTEM_LEGACY
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            data = llm_client.chat_json(messages, temperature=0.2, max_tokens=512)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # normalize entity fields
            for ent in entities:
                ent.setdefault("name", "")
                ent["type"] = _normalize_type(ent.get("type", "unknown"))
                ent.setdefault("description", "")
                ent.setdefault("status", {})
                ent.setdefault("state_changes", {})

            # normalize relation fields
            for rel in relations:
                rel.setdefault("source", "")
                rel.setdefault("target", "")
                rel.setdefault("relation", "related_to")
                rel.setdefault("context", "")

            logger.info(
                "[Extractor][extract] Extracted %d entities, %d relations from text (len=%d)",
                len(entities), len(relations), len(text),
            )
            return {"entities": entities, "relations": relations}
        except Exception as exc:
            logger.warning("[Extractor][extract] Failed: %s", exc)
            return {"entities": [], "relations": []}

    def extract_dual(
        self,
        player_input: str,
        story_text: str,
        existing_entities: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract from both player input and story text in a single LLM call.

        Returns a single combined result with deduplication.
        Falls back to player_input-only extraction if story_text is empty.
        """
        logger.debug(
            "[Extractor][dual_extract] Player input (len=%d), story text (len=%d), existing_entities=%d",
            len(player_input), len(story_text), len(existing_entities or []),
        )

        # If no story text, just extract from player input
        if not story_text.strip():
            return self._extract_player_input(player_input)

        # Build context hint about existing entities
        existing_hint = ""
        if existing_entities:
            existing_hint = (
                f"\n\nExisting entities in the world: {', '.join(existing_entities[:20])}\n"
                "For existing entities, focus on state_changes rather than duplicating descriptions."
            )

        # Single LLM call with both passages
        try:
            from src.utils.api_client import llm_client

            combined_text = (
                f"### PLAYER INPUT:\n{player_input}\n\n"
                f"### STORY TEXT:\n{story_text}"
                f"{existing_hint}"
            )

            messages = [
                {"role": "system", "content": _EXTRACTION_SYSTEM_DUAL},
                {"role": "user", "content": combined_text},
            ]
            data = llm_client.chat_json(messages, temperature=0.2, max_tokens=512)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # Normalize entity fields
            for ent in entities:
                ent.setdefault("name", "")
                ent["type"] = _normalize_type(ent.get("type", "unknown"))
                ent.setdefault("description", "")
                ent.setdefault("status", {})
                ent.setdefault("state_changes", {})

            # Normalize relation fields
            for rel in relations:
                rel.setdefault("source", "")
                rel.setdefault("target", "")
                rel.setdefault("relation", "related_to")
                rel.setdefault("context", "")

            logger.info(
                "[Extractor][dual_extract] Single-call extracted: %d entities, %d relations",
                len(entities), len(relations),
            )
            return {"entities": entities, "relations": relations}

        except Exception as exc:
            logger.warning("[Extractor][dual_extract] Single-call failed (%s), falling back to split", exc)
            # Fallback: extract separately and merge
            story_data = self.extract(story_text + existing_hint)
            player_data = self._extract_player_input(player_input + existing_hint)
            return self._merge_extractions(story_data, player_data)

    def _extract_player_input(self, player_input: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities/relations from player input (shorter text, simpler prompt)."""
        try:
            from src.utils.api_client import llm_client

            system = (
                "You are a knowledge-graph extraction engine. "
                "Given a player's action input, extract any NEW entities or relations mentioned.\n"
                "Return ONLY valid JSON:\n"
                '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event", '
                '"description": str}], '
                '"relations": [{"source": str, "target": str, "relation": str, "context": str}]}\n'
                "Only extract what is explicitly mentioned. If nothing new, return empty arrays."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": player_input},
            ]
            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            for ent in entities:
                ent.setdefault("name", "")
                ent["type"] = _normalize_type(ent.get("type", "unknown"))
                ent.setdefault("description", "")
                ent.setdefault("status", {})
                ent.setdefault("state_changes", {})
            for rel in relations:
                rel.setdefault("source", "")
                rel.setdefault("target", "")
                rel.setdefault("relation", "related_to")
                rel.setdefault("context", "")

            return {"entities": entities, "relations": relations}
        except Exception as exc:
            logger.warning("[Extractor][_extract_player_input] Failed: %s", exc)
            return {"entities": [], "relations": []}

    def _merge_extractions(
        self,
        primary: Dict[str, List[Dict[str, Any]]],
        secondary: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Merge two extraction results, deduplicating by name (case-insensitive)."""
        # Merge entities — keep richer version on conflict
        entity_map: Dict[str, Dict[str, Any]] = {}
        for ent in primary.get("entities", []) + secondary.get("entities", []):
            name = (ent.get("name") or "").strip().lower()
            if not name:
                continue
            if name in entity_map:
                # merge: prefer non-empty fields from whichever has more info
                existing = entity_map[name]
                for key in ("description", "status", "state_changes"):
                    if not existing.get(key) and ent.get(key):
                        existing[key] = ent[key]
                    elif isinstance(existing.get(key), dict) and isinstance(ent.get(key), dict):
                        existing[key] = {**existing[key], **ent[key]}
                # prefer non-unknown type
                if ent.get("type", "unknown") != "unknown":
                    existing["type"] = ent["type"]
            else:
                entity_map[name] = dict(ent)

        # Merge relations — deduplicate by (source, target, relation)
        rel_set: set = set()
        merged_rels: List[Dict[str, Any]] = []
        for rel in primary.get("relations", []) + secondary.get("relations", []):
            src = (rel.get("source") or "").strip().lower()
            tgt = (rel.get("target") or "").strip().lower()
            label = rel.get("relation", "related_to")
            key = (src, tgt, label)
            if key not in rel_set and src and tgt:
                rel_set.add(key)
                merged_rels.append(rel)

        return {"entities": list(entity_map.values()), "relations": merged_rels}


# Module-level convenience singletons
_extractor_enhanced = RelationExtractor(enhanced=True)
_extractor_legacy = RelationExtractor(enhanced=False)


def extract(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Module-level shortcut using enhanced extraction."""
    return _extractor_enhanced.extract(text)


def extract_legacy(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Module-level shortcut using legacy (simple) extraction for backward compat."""
    return _extractor_legacy.extract(text)


def extract_dual(
    player_input: str,
    story_text: str,
    existing_entities: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Module-level shortcut for dual extraction."""
    return _extractor_enhanced.extract_dual(player_input, story_text, existing_entities)
