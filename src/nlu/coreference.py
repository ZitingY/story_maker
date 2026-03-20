"""Coreference resolution using fastcoref FCoref.

Resolves pronouns in player input given recent story context.

Enhanced with:
- Possessive pronoun support (his, her, their, its)
- Multi-pronoun batch replacement
- Reflexive pronoun handling (himself, herself, themselves)
- Entity-type-aware pronoun disambiguation
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Pronoun groups with their target entity types
_PERSONAL_PRONOUNS = {
    "subject": ["he", "she", "they"],
    "object": ["him", "her", "them"],
    "reflexive": ["himself", "herself", "themselves"],
}

_NON_PERSONAL_PRONOUNS = {
    "subject": ["it"],
    "object": ["it"],
    "reflexive": ["itself"],
    "possessive": ["its"],
}

_POSSESSIVE_PRONOUNS = {
    "person": ["his", "her", "their"],
    "non_person": ["its"],
}

# All pronouns we can handle
_ALL_PERSONAL = (
    _PERSONAL_PRONOUNS["subject"]
    + _PERSONAL_PRONOUNS["object"]
    + _PERSONAL_PRONOUNS["reflexive"]
    + _POSSESSIVE_PRONOUNS["person"]
)
_ALL_NON_PERSONAL = (
    _NON_PERSONAL_PRONOUNS["subject"]
    + _NON_PERSONAL_PRONOUNS["object"]
    + _NON_PERSONAL_PRONOUNS["reflexive"]
    + _NON_PERSONAL_PRONOUNS["possessive"]
)

# Pronoun → possessive form mapping for replacement
_POSSESSIVE_FORM = {
    "he": "his", "she": "her", "they": "their",
    "him": "his", "her": "her", "them": "their",
    "his": "his", "her": "her", "their": "their",
    "it": "its", "its": "its",
}


class CoreferenceResolver:
    """Resolve pronouns → antecedents using fastcoref (or rule fallback)."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        try:
            # ── Compatibility patch for transformers 5.2.0 x fastcoref 2.x ──────────────────
            from transformers.modeling_utils import PreTrainedModel

            class _TiedWeightsCompat(dict):
                def __init__(self):
                    super().__init__()

            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = _TiedWeightsCompat()

            from fastcoref import FCoref  # type: ignore[import-untyped]
            self.model = FCoref(device="cpu")
            logger.info("Coreference resolver loaded (fastcoref)")
        except Exception as exc:
            logger.warning("fastcoref unavailable (%s) – rule-based fallback.", exc)
            self.model = None

    # ── public API ────────────────────────────────────────
    def resolve(
        self,
        text: str,
        context: Optional[List[str]] = None,
        known_entities: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Return *text* with pronouns replaced by antecedents when possible.

        Args:
            text: The player input text to resolve.
            context: Recent story history entries for context.
            known_entities: Optional list of dicts with 'text' and 'type' keys
                from the KG, used for entity-type-aware resolution.
        """
        if not context:
            return text

        full = " ".join(context[-3:]) + " " + text

        if self.model is not None:
            return self._neural_resolve(full, text)
        return self._rule_resolve(text, context, known_entities)

    # ── neural ────────────────────────────────────────────
    def _neural_resolve(self, full_context: str, original: str) -> str:
        try:
            preds = self.model.predict(texts=[full_context])
            if preds and hasattr(preds[0], "get_resolved_text"):
                resolved = preds[0].get_resolved_text()
                # Find the boundary: use last sentence of original as anchor
                return self._extract_original_portion(resolved, full_context, original)
        except Exception as exc:
            logger.warning("Neural coref failed: %s", exc)
        return original

    @staticmethod
    def _extract_original_portion(resolved: str, full_context: str, original: str) -> str:
        """Extract the portion of resolved text corresponding to original input.

        Strategy:
        1. If resolved is same length as original, return as-is.
        2. Try to match context sentences at the start of resolved to find the boundary.
        3. Fall back to length-based extraction.
        """
        if len(resolved) <= len(original):
            return resolved

        # Split context into sentences to find where context ends
        context_part = full_context[: -len(original)].strip() if full_context.endswith(original) else ""
        if not context_part:
            # full_context was created by joining context[-3:] + " " + original
            # Try to find where context ends by splitting the full context
            # We know the last N chars correspond to original
            # But the original may have been coref-resolved, so we use sentence count
            original_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', original.strip()) if s.strip()]
            resolved_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', resolved.strip()) if s.strip()]

            if original_sentences and len(resolved_sentences) >= len(original_sentences):
                # Take the last N sentences from resolved where N = len(original_sentences)
                extracted_sentences = resolved_sentences[-len(original_sentences):]
                return " ".join(extracted_sentences)

        # If we can identify the context part, remove it from resolved
        if context_part:
            context_lower = context_part.lower()
            resolved_lower = resolved.lower()
            # Find where context ends in resolved
            # Try matching the last few words of context
            context_words = context_part.split()
            if len(context_words) >= 3:
                anchor = " ".join(context_words[-3:]).lower()
                pos = resolved_lower.find(anchor)
                if pos >= 0:
                    end_pos = pos + len(anchor)
                    remainder = resolved[end_pos:].strip()
                    # Remove leading punctuation/space
                    remainder = re.sub(r'^[\s.!?]+', '', remainder)
                    if remainder:
                        return remainder

        # Fallback: length-based extraction
        diff = len(resolved) - len(full_context)
        if 0 < diff < len(original) * 2:
            return resolved[diff:]

        return original

    # ── rule fallback ─────────────────────────────────────
    @staticmethod
    def _rule_resolve(
        text: str,
        context: List[str],
        known_entities: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Rule-based pronoun resolution with entity-type awareness.

        Supports:
        - Personal pronouns (he/she/they → person names)
        - Non-personal pronouns (it → item/creature/location)
        - Possessive pronouns (his/her/their/its)
        - Reflexive pronouns (himself/herself/themselves/itself)
        - Multi-pronoun batch replacement
        """
        recent = " ".join(context[-3:]) if context else ""

        # Build entity lists from context and known_entities
        person_names: List[str] = []
        non_person_names: List[str] = []

        # Extract names from context (capitalized words, excluding common sentence-start words)
        _STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall", "not",
            "no", "so", "if", "then", "than", "that", "this", "these", "those",
            "it", "its", "he", "she", "they", "we", "i", "you", "my", "your",
            "his", "her", "their", "our", "when", "where", "why", "how", "what",
            "which", "who", "whom", "all", "each", "every", "both", "few",
            "more", "most", "other", "some", "such", "only", "own", "same",
        }
        all_names_raw = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", recent)
        all_names = [n for n in all_names_raw if n.lower() not in _STOP_WORDS]

        # If known_entities provided, use type info for classification
        if known_entities:
            entity_type_map: Dict[str, str] = {}
            for ent in known_entities:
                ent_text = ent.get("text", "").strip()
                ent_type = ent.get("type", "unknown")
                if ent_text:
                    entity_type_map[ent_text.lower()] = ent_type

            for name in all_names:
                name_lower = name.lower()
                # Check exact match first, then substring match for multi-word names
                ent_type = entity_type_map.get(name_lower)
                if ent_type is None:
                    # For multi-word names like "Queen Elizabeth", check if any word
                    # or the last word matches a known entity
                    name_parts = name_lower.split()
                    for part in reversed(name_parts):
                        if part in entity_type_map:
                            ent_type = entity_type_map[part]
                            break
                    # Also check if any known entity is a substring of the name
                    if ent_type is None:
                        for known_name, ktype in entity_type_map.items():
                            if known_name in name_lower or name_lower in known_name:
                                ent_type = ktype
                                break
                if ent_type is None:
                    ent_type = "unknown"

                if ent_type in ("person",):
                    person_names.append(name)
                else:
                    non_person_names.append(name)
        else:
            # Without type info, treat all context names as persons
            person_names = all_names

        # Deduplicate while preserving order
        seen = set()
        person_names = [n for n in person_names if n.lower() not in seen and not seen.add(n.lower())]
        seen.clear()
        non_person_names = [n for n in non_person_names if n.lower() not in seen and not seen.add(n.lower())]

        result = text

        # Replace personal pronouns → person names (most recent first)
        if person_names:
            last_person = person_names[-1]
            for pronoun in _PERSONAL_PRONOUNS["subject"] + _PERSONAL_PRONOUNS["object"]:
                pat = rf"\b{pronoun}\b"
                if re.search(pat, result, re.IGNORECASE):
                    result = re.sub(pat, last_person, result, count=1, flags=re.IGNORECASE)
                    break

            # Possessive pronouns → name's
            for pronoun in _POSSESSIVE_PRONOUNS["person"]:
                pat = rf"\b{pronoun}\b"
                if re.search(pat, result, re.IGNORECASE):
                    possessive = f"{last_person}'s"
                    result = re.sub(pat, possessive, result, count=1, flags=re.IGNORECASE)
                    break

            # Reflexive pronouns
            for pronoun in _PERSONAL_PRONOUNS["reflexive"]:
                pat = rf"\b{pronoun}\b"
                if re.search(pat, result, re.IGNORECASE):
                    result = re.sub(pat, f"{last_person} themselves", result, count=1, flags=re.IGNORECASE)
                    break

        # Replace non-personal pronouns → non-person entity names
        if non_person_names:
            last_non_person = non_person_names[-1]
            for pronoun in _NON_PERSONAL_PRONOUNS["subject"] + _NON_PERSONAL_PRONOUNS["object"]:
                pat = rf"\b{pronoun}\b"
                if re.search(pat, result, re.IGNORECASE):
                    result = re.sub(pat, last_non_person, result, count=1, flags=re.IGNORECASE)
                    break

            for pronoun in _NON_PERSONAL_PRONOUNS["possessive"]:
                pat = rf"\b{pronoun}\b"
                if re.search(pat, result, re.IGNORECASE):
                    result = re.sub(pat, f"{last_non_person}'s", result, count=1, flags=re.IGNORECASE)
                    break

        return result
