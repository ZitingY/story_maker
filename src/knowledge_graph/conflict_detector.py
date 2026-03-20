"""Rule-based + LLM conflict detection and multi-strategy resolution for the knowledge graph.

Detection layers:
- Layer 1 — deterministic rules (exclusive relation pairs, attribute contradictions)
- Layer 2 — LLM reasoning against the KG summary

Resolution strategies:
- ``keep_latest`` — retain newer information, remove older contradictory data
- ``llm_arbitrate`` — let the LLM decide which information to keep
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)

# Pairs of relations that cannot coexist between the same source→target
EXCLUSIVE_PAIRS: List[tuple[str, str]] = [
    ("ally_of", "enemy_of"),
    ("alive", "dead"),
]


# ══════════════════════════════════════════════════════════
#  Detection
# ══════════════════════════════════════════════════════════

class ConflictDetector:
    """Detect contradictions in a :class:`KnowledgeGraph`."""

    def __init__(self, kg: Any) -> None:  # avoid circular import typing
        self.kg = kg

    # ── public entry point ────────────────────────────────
    def check_all(self, new_text: str = "") -> List[Dict[str, str]]:
        """Return a list of conflict dicts (may be empty)."""
        conflicts: List[Dict[str, str]] = []
        conflicts.extend(self._rule_based_check())
        conflicts.extend(self._temporal_check())
        if new_text:
            conflicts.extend(self._llm_check(new_text))
        logger.info(
            "[ConflictDetector][check_all] Found %d conflicts (rule=%d, temporal=%d, llm=%d)",
            len(conflicts),
            len([c for c in conflicts if c.get("type") not in ("llm", "temporal")]),
            len([c for c in conflicts if c.get("type") == "temporal"]),
            len([c for c in conflicts if c.get("type") == "llm"]),
        )
        return conflicts

    # ── layer 1: rule-based ───────────────────────────────
    def _rule_based_check(self) -> List[Dict[str, str]]:
        conflicts: List[Dict[str, str]] = []
        g = self.kg.graph

        # 1. exclusive relation pairs
        for src, tgt, data in g.edges(data=True):
            rel = data.get("relation", "")
            for a, b in EXCLUSIVE_PAIRS:
                opposite = b if rel == a else (a if rel == b else None)
                if opposite is None:
                    continue
                if g.has_edge(src, tgt):
                    for _k, edata in g[src][tgt].items():
                        if edata.get("relation") == opposite:
                            conflicts.append({
                                "type": "exclusive_relation",
                                "source": src,
                                "target": tgt,
                                "relation_a": rel,
                                "relation_b": opposite,
                                "description": (
                                    f"{src} has both '{rel}' and '{opposite}' towards {tgt}."
                                ),
                            })

        # 2. dead entity with positive-health relation
        for node, ndata in g.nodes(data=True):
            status = ndata.get("status", {})
            if isinstance(status, dict) and status.get("status") == "dead":
                for _, tgt, edata in g.out_edges(node, data=True):
                    rel = edata.get("relation", "")
                    if rel in ("possesses", "located_at", "ally_of"):
                        conflicts.append({
                            "type": "dead_active",
                            "source": node,
                            "target": tgt,
                            "relation": rel,
                            "description": f"{node} is dead but has relation '{rel}' → {tgt}.",
                        })
        return conflicts

    # ── layer 1b: temporal ────────────────────────────────
    def _temporal_check(self) -> List[Dict[str, str]]:
        """Check for temporal/causal inconsistencies.

        Detects:
        1. Dead entity actions after death: relations created after entity is marked dead
        2. Causal inversion: B.created_turn < A.created_turn but A --[causes]--> B
        """
        conflicts: List[Dict[str, str]] = []
        g = self.kg.graph

        # 1. Check for dead entity actions after death
        for node, ndata in g.nodes(data=True):
            status = ndata.get("status", {})
            if not isinstance(status, dict) or status.get("status") != "dead":
                continue

            # Find when entity died (look in status_history)
            death_turn = None
            for entry in reversed(ndata.get("status_history", [])):
                changes = entry.get("changes", {})
                if "status" in changes and "dead" in changes["status"]:
                    death_turn = entry["turn"]
                    break

            if death_turn is None:
                # Death was set directly in status, use last_mentioned_turn as proxy
                death_turn = ndata.get("last_mentioned_turn", 999)

            # Check for relations created after death
            for _, tgt, edata in g.out_edges(node, data=True):
                rel_turn = edata.get("created_turn", 0)
                if rel_turn > death_turn:
                    rel = edata.get("relation", "?")
                    conflicts.append({
                        "type": "temporal",
                        "subtype": "dead_entity_action",
                        "source": node,
                        "target": tgt,
                        "relation": rel,
                        "death_turn": death_turn,
                        "relation_turn": rel_turn,
                        "description": (
                            f"{node} (died turn {death_turn}) has relation "
                            f"'{rel}' → {tgt} created at turn {rel_turn}."
                        ),
                    })

        # 2. Check for causal inversion
        for src, tgt, data in g.edges(data=True):
            relation = data.get("relation", "")
            if relation not in ("causes", "enables"):
                continue
            if not g.has_node(src) or not g.has_node(tgt):
                continue

            src_created = g.nodes[src].get("created_turn", 0)
            tgt_created = g.nodes[tgt].get("created_turn", 0)

            if tgt_created < src_created:
                conflicts.append({
                    "type": "temporal",
                    "subtype": "causal_inversion",
                    "source": src,
                    "target": tgt,
                    "relation": relation,
                    "description": (
                        f"Causal inversion: {src} (created turn {src_created}) "
                        f"'{relation}' {tgt} (created turn {tgt_created}), "
                        f"but {tgt} was created first."
                    ),
                })

        return conflicts

    # ── layer 2: LLM ─────────────────────────────────────
    def _llm_check(self, new_text: str) -> List[Dict[str, str]]:
        try:
            from src.utils.api_client import llm_client

            kg_summary = self.kg.to_summary()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a consistency checker for a text adventure knowledge graph.\n"
                        "Given the current world state and a new story passage, identify any "
                        "logical contradictions.  Return JSON: "
                        '{"conflicts": [{"description": str}]}  '
                        "Return an empty list if there are none."
                    ),
                },
                {
                    "role": "user",
                    "content": f"World state:\n{kg_summary}\n\nNew text:\n{new_text}",
                },
            ]
            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            raw = data.get("conflicts", [])
            return [{"type": "llm", "description": c.get("description", str(c))} for c in raw]
        except Exception as exc:
            logger.warning("[ConflictDetector][_llm_check] Failed: %s", exc)
            return []


# ══════════════════════════════════════════════════════════
#  Resolution Strategies
# ══════════════════════════════════════════════════════════

class ConflictResolutionStrategy(ABC):
    """Abstract base for conflict resolution strategies."""

    @abstractmethod
    def resolve(self, conflicts: List[Dict[str, str]], kg: Any) -> List[Dict[str, str]]:
        """Resolve conflicts in the KG. Return list of unresolved conflicts."""
        ...


class KeepLatestResolver(ConflictResolutionStrategy):
    """Resolve conflicts by keeping the newer information.

    For exclusive relation pairs, removes the edge with the lower
    ``last_confirmed_turn`` value.
    """

    def resolve(self, conflicts: List[Dict[str, str]], kg: Any) -> List[Dict[str, str]]:
        unresolved: List[Dict[str, str]] = []
        for conflict in conflicts:
            ctype = conflict.get("type", "")

            if ctype == "exclusive_relation":
                resolved = self._resolve_exclusive(conflict, kg)
                if not resolved:
                    unresolved.append(conflict)
            elif ctype == "dead_active":
                # for dead entity with active relation, remove the relation
                src = conflict.get("source", "")
                rel = conflict.get("relation", "")
                tgt = conflict.get("target", "")
                if src and tgt and rel:
                    self._remove_relation(kg, src, tgt, rel)
                    logger.info(
                        "[KeepLatestResolver] Removed dead-active relation %s --[%s]--> %s",
                        src, rel, tgt,
                    )
            elif ctype == "llm":
                # LLM-detected conflicts: keep as unresolved (no deterministic way to resolve)
                unresolved.append(conflict)
            else:
                unresolved.append(conflict)

        return unresolved

    def _resolve_exclusive(self, conflict: Dict[str, str], kg: Any) -> bool:
        """Remove the older of two exclusive relations. Returns True if resolved."""
        src = conflict.get("source", "")
        tgt = conflict.get("target", "")
        rel_a = conflict.get("relation_a", "")
        rel_b = conflict.get("relation_b", "")

        if not (src and tgt and rel_a and rel_b):
            return False

        g = kg.graph
        if not g.has_edge(src, tgt):
            return False

        turn_a, turn_b = -1, -1
        key_a, key_b = None, None

        for k, data in g[src][tgt].items():
            if data.get("relation") == rel_a:
                turn_a = data.get("last_confirmed_turn", 0)
                key_a = k
            elif data.get("relation") == rel_b:
                turn_b = data.get("last_confirmed_turn", 0)
                key_b = k

        if turn_a >= turn_b and key_b is not None:
            g.remove_edge(src, tgt, key=key_b)
            logger.info(
                "[KeepLatestResolver] Removed older relation %s --[%s]--> %s (turn %d < %d)",
                src, rel_b, tgt, turn_b, turn_a,
            )
            return True
        elif turn_b > turn_a and key_a is not None:
            g.remove_edge(src, tgt, key=key_a)
            logger.info(
                "[KeepLatestResolver] Removed older relation %s --[%s]--> %s (turn %d < %d)",
                src, rel_a, tgt, turn_a, turn_b,
            )
            return True
        return False

    @staticmethod
    def _remove_relation(kg: Any, src: str, tgt: str, relation: str) -> None:
        g = kg.graph
        if not g.has_edge(src, tgt):
            return
        keys_to_remove = []
        for k, data in g[src][tgt].items():
            if data.get("relation") == relation:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            g.remove_edge(src, tgt, key=k)


class LLMArbitrateResolver(ConflictResolutionStrategy):
    """Resolve conflicts by asking the LLM which information to keep."""

    def resolve(self, conflicts: List[Dict[str, str]], kg: Any) -> List[Dict[str, str]]:
        unresolved: List[Dict[str, str]] = []

        for conflict in conflicts:
            ctype = conflict.get("type", "")

            if ctype in ("exclusive_relation", "dead_active", "llm"):
                resolved = self._arbitrate_single(conflict, kg)
                if not resolved:
                    unresolved.append(conflict)
            else:
                unresolved.append(conflict)

        return unresolved

    def _arbitrate_single(self, conflict: Dict[str, str], kg: Any) -> bool:
        """Use LLM to decide how to resolve one conflict. Returns True if resolved."""
        try:
            from src.utils.api_client import llm_client

            kg_summary = kg.to_summary()
            conflict_desc = conflict.get("description", str(conflict))

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conflict resolver for a text adventure knowledge graph.\n"
                        "Given a conflict and the current world state, decide how to resolve it.\n"
                        "Return ONLY valid JSON:\n"
                        '{"resolution": "keep_new"|"keep_old"|"remove_relation"|"update_entity"|"no_action", '
                        '"target_entity": str, "target_relation": str, "reason": str}\n'
                        "- keep_new: remove the older/conflicting relation, keep the newer one\n"
                        "- keep_old: remove the newer relation\n"
                        "- remove_relation: remove a specific relation\n"
                        "- update_entity: update an entity's status\n"
                        "- no_action: conflict is not a real problem, ignore it"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conflict: {conflict_desc}\n"
                        f"Conflict type: {conflict.get('type', 'unknown')}\n"
                        f"World state:\n{kg_summary}"
                    ),
                },
            ]

            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            resolution = data.get("resolution", "no_action")
            reason = data.get("reason", "")

            logger.info(
                "[LLMArbitrateResolver] Conflict: '%s' | resolution=%s | reason=%s",
                conflict_desc[:80], resolution, reason,
            )

            if resolution == "no_action":
                return True  # LLM says it's fine, consider it resolved

            if resolution in ("keep_new", "remove_relation"):
                return self._apply_remove(conflict, kg)

            if resolution == "keep_old":
                return self._apply_remove(conflict, kg, remove_newer=True)

            if resolution == "update_entity":
                return self._apply_entity_update(conflict, kg, data)

            return False

        except Exception as exc:
            logger.warning("[LLMArbitrateResolver] Arbitration failed: %s", exc)
            return False

    def _apply_remove(self, conflict: Dict[str, str], kg: Any, remove_newer: bool = False) -> bool:
        """Remove a conflicting relation edge from the graph."""
        ctype = conflict.get("type", "")
        g = kg.graph

        if ctype == "exclusive_relation":
            src = conflict.get("source", "")
            tgt = conflict.get("target", "")
            rel_a = conflict.get("relation_a", "")
            rel_b = conflict.get("relation_b", "")
            if not (src and tgt and rel_a and rel_b and g.has_edge(src, tgt)):
                return False

            # find turns
            turns: Dict[str, int] = {}
            keys: Dict[str, Any] = {}
            for k, data in g[src][tgt].items():
                rel = data.get("relation", "")
                if rel in (rel_a, rel_b):
                    turns[rel] = data.get("last_confirmed_turn", 0)
                    keys[rel] = k

            if remove_newer:
                target_rel = max(turns, key=turns.get) if turns else None
            else:
                target_rel = min(turns, key=turns.get) if turns else None

            if target_rel and target_rel in keys:
                g.remove_edge(src, tgt, key=keys[target_rel])
                logger.info("[LLMArbitrateResolver] Removed '%s' edge %s→%s", target_rel, src, tgt)
                return True

        elif ctype == "dead_active":
            src = conflict.get("source", "")
            tgt = conflict.get("target", "")
            rel = conflict.get("relation", "")
            if src and tgt and rel:
                KeepLatestResolver._remove_relation(kg, src, tgt, rel)
                logger.info("[LLMArbitrateResolver] Removed dead-active %s→%s [%s]", src, tgt, rel)
                return True

        return False

    def _apply_entity_update(self, conflict: Dict[str, str], kg: Any, llm_data: Dict) -> bool:
        """Update an entity's status based on LLM arbitration."""
        target = llm_data.get("target_entity", "")
        if not target:
            return False
        # mark as resolved — the status update will be handled by the engine
        logger.info("[LLMArbitrateResolver] Entity update suggested for '%s'", target)
        return True


# ══════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════

def get_resolver(mode: str = "") -> ConflictResolutionStrategy:
    """Factory: return the appropriate resolution strategy.

    Falls back to ``settings.KG_CONFLICT_RESOLUTION`` if *mode* is empty.
    """
    mode = mode or settings.KG_CONFLICT_RESOLUTION
    if mode == "keep_latest":
        logger.debug("[ConflictResolver] Using KeepLatestResolver")
        return KeepLatestResolver()
    elif mode == "llm_arbitrate":
        logger.debug("[ConflictResolver] Using LLMArbitrateResolver")
        return LLMArbitrateResolver()
    else:
        logger.warning("[ConflictResolver] Unknown mode '%s', falling back to llm_arbitrate", mode)
        return LLMArbitrateResolver()
