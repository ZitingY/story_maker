"""KnowledgeGraph: MultiDiGraph-based world state tracker.

Node keys are always **lowercased** names.  Supports multiple edges between the
same pair (NetworkX ``MultiDiGraph``).

Enhanced with:
- Rich node attributes (description, status, temporal tracking, importance)
- Rich edge attributes (context, confidence, temporal tracking)
- Per-turn decay and importance recalculation
- Layered summary generation for LLM prompts
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Real-time world-state graph backed by ``nx.MultiDiGraph``."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._current_turn: int = 0
        logger.debug("[KG][init] New KnowledgeGraph created")

    # ── node helpers ──────────────────────────────────────
    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def set_turn(self, turn_id: int) -> None:
        """Update the current turn counter for temporal tracking."""
        self._current_turn = turn_id
        logger.debug("[KG][set_turn] Turn updated to %d", turn_id)

    # ── entity management ─────────────────────────────────
    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        status: Optional[Dict[str, Any]] = None,
        turn_id: Optional[int] = None,
        is_player_mentioned: bool = False,
        emotion: Optional[str] = None,
    ) -> str:
        """Upsert a node with rich attributes.  Returns the lowered key.

        Attributes managed automatically:
        - ``created_turn``: first turn this entity appeared
        - ``last_mentioned_turn``: most recent turn referencing this entity
        - ``mention_count``: total times referenced
        - ``player_mention_count``: times directly mentioned by the player
        - ``importance_score``: composite importance (0-1)
        """
        key = self._key(name)
        turn = turn_id if turn_id is not None else self._current_turn

        # Validate entity_type against KG_ENTITY_TYPES
        if entity_type and entity_type != "unknown" and entity_type not in settings.KG_ENTITY_TYPES:
            logger.warning(
                "[KG][add_entity] Invalid entity_type '%s' for '%s', falling back to 'unknown'",
                entity_type, key,
            )
            entity_type = "unknown"

        if self.graph.has_node(key):
            node = self.graph.nodes[key]
            # update description: append new info if different
            if description and description not in node.get("description", ""):
                old_desc = node.get("description", "")
                node["description"] = f"{old_desc}. {description}".strip(". ")

            # merge status dict (with history tracking)
            if status:
                # Snapshot old status BEFORE merge
                old_status = dict(node.get("status", {}))

                # Merge new status
                existing_status = node.get("status", {})
                existing_status.update(status)
                node["status"] = existing_status

                # Track changes
                changes = {}
                for k, v in status.items():
                    old_val = old_status.get(k)
                    if old_val is not None and old_val != v:
                        changes[k] = f"{old_val}→{v}"
                    elif old_val is None:
                        changes[k] = f"(new) {v}"
                if changes:
                    history = node.get("status_history", [])
                    history.append({"turn": turn, "changes": changes})
                    # Keep only last 10 entries
                    node["status_history"] = history[-10:]

            # update entity_type if provided
            if entity_type and entity_type != "unknown":
                node["entity_type"] = entity_type

            # temporal tracking
            node["last_mentioned_turn"] = turn
            node["mention_count"] = node.get("mention_count", 0) + 1
            if is_player_mentioned:
                node["player_mention_count"] = node.get("player_mention_count", 0) + 1

            # boost importance
            boost = settings.KG_IMPORTANCE_MENTION_BOOST
            if is_player_mentioned:
                boost += settings.KG_IMPORTANCE_PLAYER_BOOST
            node["importance_score"] = min(1.0, node.get("importance_score", 0.5) + boost)

            # update emotion
            if emotion:
                node["last_emotion"] = emotion

            logger.debug(
                "[KG][add_entity] Updated '%s' type=%s turn=%d mentions=%d importance=%.2f | total_nodes=%d",
                key, entity_type, turn, node["mention_count"],
                node["importance_score"], self.num_nodes,
            )
        else:
            importance = 0.5
            if is_player_mentioned:
                importance += settings.KG_IMPORTANCE_PLAYER_BOOST
            self.graph.add_node(
                key,
                name=name,
                entity_type=entity_type,
                description=description or "",
                status=status or {},
                created_turn=turn,
                last_mentioned_turn=turn,
                mention_count=1,
                player_mention_count=1 if is_player_mentioned else 0,
                importance_score=min(1.0, importance),
                last_emotion=emotion or "",
                status_history=[],
            )
            self._enforce_limit()
            logger.info(
                "[KG][add_entity] Created '%s' type=%s turn=%d importance=%.2f | total_nodes=%d",
                key, entity_type, turn, min(1.0, importance), self.num_nodes,
            )
        return key

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        key = self._key(name)
        if key in self.graph:
            return dict(self.graph.nodes[key])
        return None

    def remove_entity(self, name: str) -> None:
        key = self._key(name)
        if key in self.graph:
            self.graph.remove_node(key)
            logger.debug("[KG][remove_entity] Removed '%s' | total_nodes=%d", key, self.num_nodes)

    # ── temporal queries ──────────────────────────────────
    def get_entity_history(self, name: str) -> List[Dict[str, Any]]:
        """Return the status change history for an entity."""
        key = self._key(name)
        if key in self.graph:
            return list(self.graph.nodes[key].get("status_history", []))
        return []

    def get_entity_status_at_turn(self, name: str, turn: int) -> Dict[str, Any]:
        """Reconstruct an entity's status as of a specific turn."""
        key = self._key(name)
        if key not in self.graph:
            return {}

        node = self.graph.nodes[key]
        created = node.get("created_turn", 0)
        if turn < created:
            return {}

        # Start with current status and undo changes that happened after target turn
        current_status = dict(node.get("status", {}))
        history = node.get("status_history", [])

        for entry in reversed(history):
            if entry["turn"] <= turn:
                continue  # This change was before/at target turn, keep it
            # Undo this change (it happened after target turn)
            for field, change_str in entry.get("changes", {}).items():
                if "→" in change_str:
                    old_val = change_str.split("→")[0]
                    current_status[field] = old_val
                elif "(new)" in change_str:
                    current_status.pop(field, None)

        return current_status

    # ── relation management ───────────────────────────────
    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        context: str = "",
        turn_id: Optional[int] = None,
        confidence: float = 1.0,
    ) -> None:
        """Add an edge with rich attributes.  Auto-creates missing nodes.

        Duplicate (source, target, relation) triples update ``last_confirmed_turn``
        instead of creating a new edge.
        """
        src = self._key(source)
        tgt = self._key(target)
        turn = turn_id if turn_id is not None else self._current_turn

        if not self.graph.has_node(src):
            self.add_entity(source, "unknown", turn_id=turn)
        if not self.graph.has_node(tgt):
            self.add_entity(target, "unknown", turn_id=turn)

        # skip if exact same relation already exists — update confirmation time
        if self.graph.has_edge(src, tgt):
            for _k, data in self.graph[src][tgt].items():
                if data.get("relation") == relation:
                    data["last_confirmed_turn"] = turn
                    data["confidence"] = max(data.get("confidence", 0.5), confidence)
                    logger.debug(
                        "[KG][add_relation] Confirmed existing %s --[%s]--> %s turn=%d",
                        src, relation, tgt, turn,
                    )
                    return

        self.graph.add_edge(
            src, tgt,
            relation=relation,
            context=context or "",
            created_turn=turn,
            last_confirmed_turn=turn,
            confidence=confidence,
        )
        logger.info(
            "[KG][add_relation] %s --[%s]--> %s turn=%d confidence=%.2f | edges=%d",
            src, relation, tgt, turn, confidence, self.num_edges,
        )

    def get_relations(self, name: str) -> List[Dict[str, Any]]:
        key = self._key(name)
        results: List[Dict[str, Any]] = []
        if key not in self.graph:
            return results
        for _, tgt, data in self.graph.out_edges(key, data=True):
            results.append({"direction": "out", "target": tgt, **data})
        for src, _, data in self.graph.in_edges(key, data=True):
            results.append({"direction": "in", "source": src, **data})
        return results

    # ── entity state updates ──────────────────────────────
    def update_entity_state(
        self, name: str, state_updates: Dict[str, Any], turn_id: Optional[int] = None
    ) -> bool:
        """Update specific status fields of an existing entity.

        Returns True if the entity was found and updated.
        """
        key = self._key(name)
        if not self.graph.has_node(key):
            logger.warning("[KG][update_entity_state] Entity '%s' not found", key)
            return False

        node = self.graph.nodes[key]
        turn = turn_id if turn_id is not None else self._current_turn
        existing_status = node.get("status", {})
        existing_status.update(state_updates)
        node["status"] = existing_status
        node["last_mentioned_turn"] = turn

        logger.debug(
            "[KG][update_entity_state] '%s' updated %s turn=%d",
            key, state_updates, turn,
        )
        return True

    # ── batch mention refresh ─────────────────────────────
    def refresh_mentions(
        self,
        mentioned_names: List[str],
        turn_id: Optional[int] = None,
        player_mentioned_names: Optional[List[str]] = None,
    ) -> None:
        """Batch-update mention tracking and importance for referenced entities.

        Entities NOT mentioned get a small importance decay.
        """
        turn = turn_id if turn_id is not None else self._current_turn
        player_set: Set[str] = set()
        if player_mentioned_names:
            player_set = {self._key(n) for n in player_mentioned_names}

        mentioned_keys: Set[str] = set()
        for name in mentioned_names:
            key = self._key(name)
            mentioned_keys.add(key)
            if not self.graph.has_node(key):
                continue

            node = self.graph.nodes[key]
            node["last_mentioned_turn"] = turn
            node["mention_count"] = node.get("mention_count", 0) + 1
            if key in player_set:
                node["player_mention_count"] = node.get("player_mention_count", 0) + 1
                boost = settings.KG_IMPORTANCE_MENTION_BOOST + settings.KG_IMPORTANCE_PLAYER_BOOST
            else:
                boost = settings.KG_IMPORTANCE_MENTION_BOOST
            node["importance_score"] = min(1.0, node.get("importance_score", 0.5) + boost)

        # decay unmentioned entities
        for key in self.graph.nodes():
            if key not in mentioned_keys:
                node = self.graph.nodes[key]
                node["importance_score"] = max(
                    0.0,
                    node.get("importance_score", 0.5) * settings.KG_IMPORTANCE_DECAY_FACTOR,
                )

        logger.debug(
            "[KG][refresh_mentions] Mentioned=%d player=%d | total_nodes=%d",
            len(mentioned_keys), len(player_set), self.num_nodes,
        )

    # ── temporal decay ────────────────────────────────────
    def apply_decay(self, turn_id: Optional[int] = None) -> None:
        """Reduce confidence of relations not confirmed recently. Prune weak ones."""
        turn = turn_id if turn_id is not None else self._current_turn
        edges_to_remove: List[tuple] = []

        for src, tgt, key, data in list(self.graph.edges(data=True, keys=True)):
            turns_since = turn - data.get("last_confirmed_turn", turn)
            if turns_since > 0:
                data["confidence"] = max(
                    0.0,
                    data.get("confidence", 1.0) * (settings.KG_RELATION_DECAY_FACTOR ** turns_since),
                )
                if data["confidence"] < settings.KG_RELATION_MIN_CONFIDENCE:
                    edges_to_remove.append((src, tgt, key))

        for src, tgt, key in edges_to_remove:
            rel = self.graph.edges[src, tgt, key].get("relation", "?")
            self.graph.remove_edge(src, tgt, key=key)
            logger.info(
                "[KG][apply_decay] Removed weak relation %s --[%s]--> %s (confidence below threshold)",
                src, rel, tgt,
            )

        if edges_to_remove:
            logger.info("[KG][apply_decay] Pruned %d low-confidence edges", len(edges_to_remove))

    # ── importance recalculation ──────────────────────────
    def recalculate_importance(self) -> None:
        """Recalculate importance scores using composite formula.

        importance = 0.3*norm(degree) + 0.3*recency + 0.2*norm(mention_count) + 0.2*norm(player_mentions)
        """
        if self.graph.number_of_nodes() == 0:
            return

        max_degree = max((self.graph.degree(n) for n in self.graph.nodes()), default=1) or 1
        max_mentions = max(
            (self.graph.nodes[n].get("mention_count", 0) for n in self.graph.nodes()), default=1
        ) or 1
        max_player = max(
            (self.graph.nodes[n].get("player_mention_count", 0) for n in self.graph.nodes()), default=1
        ) or 1

        for key in self.graph.nodes():
            node = self.graph.nodes[key]
            degree_score = self.graph.degree(key) / max_degree
            mentions_score = node.get("mention_count", 0) / max_mentions
            player_score = node.get("player_mention_count", 0) / max_player

            # recency: exponential decay based on turns since last mention
            turns_since = self._current_turn - node.get("last_mentioned_turn", self._current_turn)
            recency_score = 0.95 ** turns_since

            if settings.KG_IMPORTANCE_MODE == "composite":
                node["importance_score"] = (
                    0.3 * degree_score
                    + 0.3 * recency_score
                    + 0.2 * mentions_score
                    + 0.2 * player_score
                )
            else:  # degree_only (backward compat)
                node["importance_score"] = degree_score

        logger.debug(
            "[KG][recalculate_importance] Mode=%s | nodes recalculated=%d",
            settings.KG_IMPORTANCE_MODE, self.num_nodes,
        )

    # ── timeline ──────────────────────────────────────────
    def get_timeline(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the most recent events/relations as a timeline.

        Sorts edges by last_confirmed_turn descending.
        """
        max_entries = n or settings.KG_MAX_TIMELINE_ENTRIES
        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "relation": data.get("relation", "related_to"),
                "context": data.get("context", ""),
                "last_confirmed_turn": data.get("last_confirmed_turn", 0),
            })
        edges.sort(key=lambda e: e["last_confirmed_turn"], reverse=True)
        return edges[:max_entries]

    # ── summary for LLM context ──────────────────────────
    def to_summary(self, max_entities: int = 30) -> str:
        """Produce a textual world-state block consumable by the LLM.

        Supports two modes via ``settings.KG_SUMMARY_MODE``:
        - ``"flat"``: original format (backward compatible)
        - ``"layered"``: importance-ranked with descriptions, status, and timeline
        """
        if settings.KG_SUMMARY_MODE == "layered":
            return self._to_summary_layered(max_entities)
        return self._to_summary_flat(max_entities)

    def _to_summary_flat(self, max_entities: int) -> str:
        """Original flat summary format (backward compatible)."""
        nodes = list(self.graph.nodes(data=True))[:max_entities]
        if not nodes:
            return "=== World State ===\n(empty)"

        lines = ["=== World State ==="]
        for key, data in nodes:
            etype = data.get("entity_type", "unknown")
            name = data.get("name", key)
            extra = {k: v for k, v in data.items() if k not in ("name", "entity_type")}
            extra_str = f" {extra}" if extra else ""
            lines.append(f"- {name} [{etype}]{extra_str}")

        lines.append("\n=== Relations ===")
        for src, tgt, data in list(self.graph.edges(data=True)):
            rel = data.get("relation", "related_to")
            lines.append(f"- {src} --[{rel}]--> {tgt}")

        return "\n".join(lines)

    def _to_summary_layered(self, max_entities: int) -> str:
        """Layered summary: Core / Secondary / Background + Timeline."""
        if self.graph.number_of_nodes() == 0:
            return "=== World State ===\n(empty)"

        # sort by importance
        scored = []
        for key, data in self.graph.nodes(data=True):
            scored.append((key, data, data.get("importance_score", 0.0)))
        scored.sort(key=lambda x: x[2], reverse=True)

        core: List[tuple] = []
        secondary: List[tuple] = []
        background: List[tuple] = []

        for key, data, score in scored[:max_entities]:
            if score >= 0.6:
                core.append((key, data, score))
            elif score >= 0.3:
                secondary.append((key, data, score))
            else:
                background.append((key, data, score))

        lines: List[str] = []

        # Core entities
        if core:
            lines.append("=== Core Entities (High Importance) ===")
            for key, data, score in core:
                lines.extend(self._format_entity_block(key, data, score))

        # Secondary entities
        if secondary:
            lines.append("\n=== Secondary Entities ===")
            for key, data, score in secondary:
                lines.extend(self._format_entity_block(key, data, score))

        # Background entities
        if background:
            lines.append("\n=== Background ===")
            for key, data, score in background:
                name = data.get("name", key)
                etype = data.get("entity_type", "unknown")
                last_seen = data.get("last_mentioned_turn", "?")
                lines.append(
                    f"- {name} [{etype}] (importance: {score:.2f}, last seen turn {last_seen})"
                )

        # Timeline
        timeline = self.get_timeline()
        if timeline:
            lines.append("\n=== Recent Timeline ===")
            for entry in timeline:
                ctx = entry["context"] or f"{entry['source']} {entry['relation']} {entry['target']}"
                lines.append(f"- Turn {entry['last_confirmed_turn']}: {ctx}")

        return "\n".join(lines)

    def _format_entity_block(self, key: str, data: Dict[str, Any], score: float) -> List[str]:
        """Format a single entity block with description, status, and relations."""
        name = data.get("name", key)
        etype = data.get("entity_type", "unknown")
        last_turn = data.get("last_mentioned_turn", "?")
        lines = [f"\n- {name} [{etype}] (importance: {score:.2f}, turn {last_turn})"]

        desc = data.get("description", "")
        if desc:
            lines.append(f"  Description: {desc}")

        status = data.get("status", {})
        if status:
            status_str = ", ".join(f"{k}: {v}" for k, v in status.items())
            lines.append(f"  Status: {{{status_str}}}")

        emotion = data.get("last_emotion", "")
        if emotion:
            lines.append(f"  Emotion: {emotion}")

        # Status history (last 3 entries)
        history = data.get("status_history", [])
        if history:
            recent_history = history[-3:]
            history_str = "; ".join(
                f"turn {h['turn']}: {', '.join(f'{k}={v}' for k, v in h.get('changes', {}).items())}"
                for h in recent_history
            )
            lines.append(f"  History: {history_str}")

        # outgoing relations
        out_rels = []
        if key in self.graph:
            for _, tgt, edata in self.graph.out_edges(key, data=True):
                rel = edata.get("relation", "related_to")
                conf = edata.get("confidence", 1.0)
                out_rels.append(f"{rel}→{tgt} ({conf:.1f})")
        if out_rels:
            lines.append(f"  Relations: {', '.join(out_rels)}")

        return lines

    # ── size enforcement ──────────────────────────────────
    def _enforce_limit(self) -> None:
        """Remove lowest-importance node when exceeding ``KG_MAX_NODES``."""
        while self.graph.number_of_nodes() > settings.KG_MAX_NODES:
            if settings.KG_IMPORTANCE_MODE == "composite":
                least = min(
                    self.graph.nodes,
                    key=lambda n: self.graph.nodes[n].get("importance_score", 0.0),
                )
            else:
                least = min(self.graph.nodes, key=lambda n: self.graph.degree(n))
            name = self.graph.nodes[least].get("name", least)
            self.graph.remove_node(least)
            logger.info("[KG][_enforce_limit] Removed '%s' (low importance) | nodes=%d", name, self.num_nodes)

    # ── stats ─────────────────────────────────────────────
    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    # ── persistence ───────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the knowledge graph to a plain dict.

        Returns a dict with keys: version, turn, nodes, edges.
        All non-basic types are converted to JSON-safe values.
        """
        nodes = []
        for key, data in self.graph.nodes(data=True):
            node = {"key": key}
            for k, v in data.items():
                # Ensure all values are JSON-serializable
                if isinstance(v, (str, int, float, bool, type(None))):
                    node[k] = v
                elif isinstance(v, dict):
                    node[k] = v
                elif isinstance(v, list):
                    node[k] = v
                else:
                    node[k] = str(v)
            nodes.append(node)

        edges = []
        for src, tgt, key, data in self.graph.edges(data=True, keys=True):
            edge = {"source": src, "target": tgt, "key": key}
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    edge[k] = v
                elif isinstance(v, dict):
                    edge[k] = v
                elif isinstance(v, list):
                    edge[k] = v
                else:
                    edge[k] = str(v)
            edges.append(edge)

        return {
            "version": 1,
            "turn": self._current_turn,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Deserialize a knowledge graph from a plain dict.

        Accepts the format produced by ``to_dict()``.
        """
        kg = cls()
        kg._current_turn = data.get("turn", 0)

        # Rebuild nodes
        for node_data in data.get("nodes", []):
            key = node_data.pop("key", "")
            if key:
                kg.graph.add_node(key, **node_data)

        # Rebuild edges
        for edge_data in data.get("edges", []):
            src = edge_data.pop("source", "")
            tgt = edge_data.pop("target", "")
            edge_key = edge_data.pop("key", 0)
            if src and tgt:
                kg.graph.add_edge(src, tgt, key=edge_key, **edge_data)

        return kg

    def save(self, filepath: str) -> None:
        """Save the knowledge graph to a JSON file."""
        import json
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            "[KG][save] Saved to %s | %d nodes, %d edges, turn=%d",
            path, len(data["nodes"]), len(data["edges"]), data["turn"],
        )

    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """Load a knowledge graph from a JSON file."""
        import json
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            logger.warning("[KG][load] File not found: %s, returning empty graph", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 0)
        if version != 1:
            logger.warning("[KG][load] Unknown version %d, attempting to load anyway", version)

        kg = cls.from_dict(data)
        logger.info(
            "[KG][load] Loaded from %s | %d nodes, %d edges, turn=%d",
            path, kg.num_nodes, kg.num_edges, kg._current_turn,
        )
        return kg
