"""Main game engine orchestrator for StoryWeaver.

Pipeline per turn:
1. Coreference resolution
2. Intent classification
3. Entity extraction
4. Story generation (LLM)
5. KG update (dual extraction + state update + decay + importance refresh)
6. Conflict detection + resolution (configurable strategy)
7. Option generation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from src.engine.state import GameState
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.nlu.coreference import CoreferenceResolver
from src.nlu.sentiment_analyzer import SentimentAnalyzer
from src.nlg.story_generator import StoryGenerator
from src.nlg.option_generator import OptionGenerator, StoryOption
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.relation_extractor import (
    extract as kg_extract,
    extract_dual as kg_extract_dual,
)
from src.knowledge_graph.conflict_detector import ConflictDetector, get_resolver
from src.knowledge_graph.visualizer import render_kg_html

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Container returned after every game turn."""
    story_text: str
    options: List[StoryOption]
    nlu_debug: Dict = field(default_factory=dict)
    kg_html: str = ""
    conflicts: List[str] = field(default_factory=list)


class GameEngine:
    """Coordinates the full NLU → NLG → KG pipeline."""

    def __init__(
        self,
        genre: str = "fantasy",
        intent_model_path: Optional[str] = None,
        auto_load_nlu: bool = True,
        conflict_resolution: Optional[str] = None,
        extraction_mode: Optional[str] = None,
        importance_mode: Optional[str] = None,
        summary_mode: Optional[str] = None,
    ):
        self.genre = genre
        self.state = GameState()
        self.kg = KnowledgeGraph()

        # Strategy configuration (fallback to settings defaults)
        self.conflict_resolution = conflict_resolution or settings.KG_CONFLICT_RESOLUTION
        self.extraction_mode = extraction_mode or settings.KG_EXTRACTION_MODE
        self.importance_mode = importance_mode or settings.KG_IMPORTANCE_MODE
        self.summary_mode = summary_mode or settings.KG_SUMMARY_MODE

        logger.info(
            "[Engine][init] genre=%s | strategies: conflict=%s extraction=%s importance=%s summary=%s",
            genre, self.conflict_resolution, self.extraction_mode,
            self.importance_mode, self.summary_mode,
        )

        # Evaluation tracking
        self.turn_conflict_counts: List[int] = []

        # NLU
        self.coref = CoreferenceResolver()
        self.intent_clf = IntentClassifier(model_path=intent_model_path)
        self.entity_ext = EntityExtractor()
        self.sentiment = SentimentAnalyzer()
        self.nlu_status: Dict[str, object] = {
            "coref_loaded": False,
            "intent_model_loaded": False,
            "intent_backend": "rule_fallback",
            "entity_model_loaded": False,
            "sentiment_loaded": False,
        }

        if auto_load_nlu:
            self._load_nlu_components()
        else:
            logger.info("NLU auto-load disabled. Engine will run with lazy/fallback behavior.")

        # NLG
        self.story_gen = StoryGenerator()
        self.option_gen = OptionGenerator()

        # KG helpers
        self.conflict_det = ConflictDetector(self.kg)
        self.conflict_resolver = get_resolver(self.conflict_resolution)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> TurnResult:
        """Begin a new adventure and return the opening turn."""
        logger.info("[Engine][start_game] Starting new game | genre=%s", self.genre)
        self.state = GameState()
        self.kg = KnowledgeGraph()
        self.conflict_det = ConflictDetector(self.kg)
        self.conflict_resolver = get_resolver(self.conflict_resolution)
        self.turn_conflict_counts = []

        story_text = self.story_gen.generate_opening(self.genre)
        self.state.add_narration(story_text)

        # Seed the KG from the opening
        self.kg.set_turn(self.state.turn_id)
        self._apply_kg_update(story_text, turn_id=self.state.turn_id)

        kg_summary = self.kg.to_summary()
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg.graph)

        logger.info("[Engine][start_game] Opening generated | KG: %d nodes, %d edges", self.kg.num_nodes, self.kg.num_edges)

        return TurnResult(
            story_text=story_text,
            options=options,
            kg_html=kg_html,
        )

    def process_turn(self, player_input: str) -> TurnResult:
        """Run the full pipeline for one player turn."""
        logger.info(
            "[Engine][process_turn] === Turn %d START === | input='%s'",
            self.state.turn_id + 1, player_input[:60],
        )

        # 1. Coreference resolution
        recent_entries = self.state.story_history[-4:]
        recent_texts = [t["text"] for t in recent_entries]
        known_entities = [
            {"text": data.get("name", key), "type": data.get("entity_type", "unknown")}
            for key, data in self.kg.graph.nodes(data=True)
        ]
        resolved = self.coref.resolve(player_input, recent_texts, known_entities=known_entities)
        logger.debug("[Engine][coref] Resolved: '%s' → '%s'", player_input[:40], resolved[:40])

        # 2. Intent classification
        intent_result = self.intent_clf.predict(resolved)
        intent = intent_result["intent"]
        logger.debug("[Engine][intent] intent=%s confidence=%.2f", intent, intent_result["confidence"])

        # 2b. Sentiment analysis
        emotion_result = self.sentiment.analyze(resolved)
        logger.debug("[Engine][sentiment] emotion=%s confidence=%.2f", emotion_result["emotion"], emotion_result["confidence"])

        # 3. Entity extraction (NLU layer)
        kg_entity_names = list(self.kg.graph.nodes())
        entities = self.entity_ext.extract(resolved, known_entities=kg_entity_names)
        entity_names = [e["text"] for e in entities]
        logger.debug("[Engine][nlu_entities] Extracted %d entities: %s", len(entities), entity_names)

        # Record player input
        self.state.add_player_input(player_input)

        # 4. Story generation
        kg_summary = self.kg.to_summary()
        history = self.state.recent_history(6)
        story_text = self.story_gen.continue_story(
            player_input=resolved,
            intent=intent,
            kg_summary=kg_summary,
            history=history,
            emotion=emotion_result["emotion"],
        )
        self.state.add_narration(story_text)
        current_turn = self.state.turn_id

        # 5. KG update
        self.kg.set_turn(current_turn)
        self._apply_kg_update(
            story_text=story_text,
            player_input=resolved,
            nlu_entities=entities,
            turn_id=current_turn,
            emotion=emotion_result["emotion"],
        )

        # 6. Conflict detection + resolution
        raw_conflicts = self.conflict_det.check_all(story_text)
        unresolved = self.conflict_resolver.resolve(raw_conflicts, self.kg)
        conflict_descriptions = [c.get("description", str(c)) for c in unresolved]
        self.turn_conflict_counts.append(len(unresolved))

        logger.info(
            "[Engine][conflicts] Detected=%d resolved=%d remaining=%d via %s",
            len(raw_conflicts),
            len(raw_conflicts) - len(unresolved),
            len(unresolved),
            self.conflict_resolution,
        )

        # 7. Option generation
        kg_summary = self.kg.to_summary()  # refresh after KG update
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg.graph)

        nlu_debug = {
            "resolved_input": resolved,
            "intent": intent,
            "confidence": intent_result["confidence"],
            "entities": entities,
            "emotion": emotion_result["emotion"],
            "emotion_confidence": emotion_result["confidence"],
            "emotion_scores": emotion_result.get("scores", {}),
            "intent_backend": self.nlu_status["intent_backend"],
            "intent_model_loaded": self.nlu_status["intent_model_loaded"],
            "coref_loaded": self.nlu_status["coref_loaded"],
            "entity_model_loaded": self.nlu_status["entity_model_loaded"],
            "sentiment_loaded": self.nlu_status["sentiment_loaded"],
        }

        logger.info(
            "[Engine][process_turn] === Turn %d END === | KG: %d nodes, %d edges, %d conflicts",
            current_turn, self.kg.num_nodes, self.kg.num_edges, len(unresolved),
        )

        # Auto-save
        self._auto_save()

        return TurnResult(
            story_text=story_text,
            options=options,
            nlu_debug=nlu_debug,
            kg_html=kg_html,
            conflicts=conflict_descriptions,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @property
    def all_story_texts(self) -> List[str]:
        """Return all narrator texts from the session (for evaluation)."""
        return [
            e["text"] for e in self.state.story_history if e["role"] == "narrator"
        ]

    @property
    def kg_entity_names(self) -> List[str]:
        """Return the display names of all KG entities."""
        return [
            data.get("name", key)
            for key, data in self.kg.graph.nodes(data=True)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_nlu_components(self) -> None:
        """Load all NLU submodules with graceful degradation on failure."""
        try:
            self.coref.load()
            self.nlu_status["coref_loaded"] = self.coref.model is not None
        except Exception as exc:
            logger.warning("Coreference resolver init failed: %s", exc)

        try:
            self.intent_clf.load()
            self.nlu_status["intent_model_loaded"] = (
                self.intent_clf.model is not None and self.intent_clf.tokenizer is not None
            )
            self.nlu_status["intent_backend"] = self.intent_clf.backend
        except Exception as exc:
            logger.warning("Intent classifier init failed: %s", exc)
            self.nlu_status["intent_backend"] = "rule_fallback"

        try:
            self.entity_ext.load()
            self.nlu_status["entity_model_loaded"] = self.entity_ext.nlp is not None
        except Exception as exc:
            logger.warning("Entity extractor init failed: %s", exc)

        try:
            self.sentiment.load()
            self.nlu_status["sentiment_loaded"] = self.sentiment.model is not None
        except Exception as exc:
            logger.warning("Sentiment analyzer init failed: %s", exc)

        logger.info(
            "NLU load status: coref_loaded=%s, intent_model_loaded=%s, intent_backend=%s, entity_model_loaded=%s",
            self.nlu_status["coref_loaded"],
            self.nlu_status["intent_model_loaded"],
            self.nlu_status["intent_backend"],
            self.nlu_status["entity_model_loaded"],
        )

    def _apply_kg_update(
        self,
        story_text: str,
        player_input: str = "",
        nlu_entities: Optional[List[Dict]] = None,
        turn_id: int = 0,
        emotion: Optional[str] = None,
    ) -> None:
        """Apply KG updates based on configured extraction mode.

        Steps:
        1. Extract entities/relations from story text (+ optionally player input)
        2. Add/update entities with rich attributes
        3. Add/update relations with context and confidence
        4. Update entity states from state_changes
        5. Refresh mention tracking
        6. Apply temporal decay
        7. Recalculate importance scores
        """
        all_entities: List[Dict] = []
        all_relations: List[Dict] = []

        # --- Step 1: Extraction ---
        try:
            if self.extraction_mode == "dual_extract" and player_input:
                existing_names = list(self.kg.graph.nodes())
                data = kg_extract_dual(player_input, story_text, existing_names)
                logger.debug(
                    "[Engine][kg_update] Dual extract: %d entities, %d relations",
                    len(data.get("entities", [])), len(data.get("relations", [])),
                )
            else:
                data = kg_extract(story_text)
                logger.debug(
                    "[Engine][kg_update] Story-only extract: %d entities, %d relations",
                    len(data.get("entities", [])), len(data.get("relations", [])),
                )
            all_entities.extend(data.get("entities", []))
            all_relations.extend(data.get("relations", []))
        except Exception as exc:
            logger.warning("[Engine][kg_update] Extraction failed: %s", exc)

        # --- Step 2: Add NLU entities ---
        nlu_names: List[str] = []
        if nlu_entities:
            for ent in nlu_entities:
                name = ent.get("text", "")
                etype = ent.get("type", "thing")
                if name:
                    self.kg.add_entity(name, etype, turn_id=turn_id, is_player_mentioned=True, emotion=emotion)
                    nlu_names.append(name)

        # --- Step 3: Add extracted entities with rich attributes ---
        extracted_names: List[str] = []
        for ent in all_entities:
            name = ent.get("name") or ent.get("text", "")
            etype = ent.get("type", "unknown")
            desc = ent.get("description", "")
            status = ent.get("status", {})
            state_changes = ent.get("state_changes", {})
            if not name:
                continue

            # add/update the entity
            self.kg.add_entity(
                name, etype,
                description=desc,
                status=status,
                turn_id=turn_id,
            )
            extracted_names.append(name)

            # apply state_changes to existing entities
            if state_changes:
                updated = self.kg.update_entity_state(name, state_changes, turn_id=turn_id)
                if updated:
                    logger.debug("[Engine][kg_update] State change for '%s': %s", name, state_changes)

        # --- Step 4: Add extracted relations ---
        for rel in all_relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            label = rel.get("relation", "related_to")
            context = rel.get("context", "")
            if src and tgt:
                self.kg.add_relation(src, tgt, label, context=context, turn_id=turn_id)

        # --- Step 5: Refresh mentions ---
        all_mentioned = list(set(extracted_names + nlu_names))
        if all_mentioned:
            self.kg.refresh_mentions(all_mentioned, turn_id=turn_id, player_mentioned_names=nlu_names)

        # --- Step 6: Apply temporal decay ---
        self.kg.apply_decay(turn_id=turn_id)

        # --- Step 7: Recalculate importance ---
        self.kg.recalculate_importance()

        logger.info(
            "[Engine][kg_update] Turn %d complete | added: %d entities, %d relations | KG: %d nodes, %d edges",
            turn_id, len(extracted_names) + len(nlu_names), len(all_relations),
            self.kg.num_nodes, self.kg.num_edges,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_game(self, filepath: Optional[str] = None) -> str:
        """Save the current game state (KG + story history) to a JSON file.

        Returns the filepath used.
        """
        import json
        from pathlib import Path

        if filepath is None:
            save_dir = Path(settings.KG_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(save_dir / f"{self.genre}_latest.json")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        game_data = {
            "version": 1,
            "genre": self.genre,
            "state": {
                "turn_id": self.state.turn_id,
                "genre": self.state.genre,
                "story_history": self.state.story_history,
            },
            "kg": self.kg.to_dict(),
            "conflict_resolution": self.conflict_resolution,
            "extraction_mode": self.extraction_mode,
            "importance_mode": self.importance_mode,
            "summary_mode": self.summary_mode,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)

        logger.info("[Engine][save_game] Saved to %s | turn=%d", path, self.state.turn_id)
        return str(path)

    def load_game(self, filepath: str) -> None:
        """Load a saved game state from a JSON file."""
        import json
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            logger.warning("[Engine][load_game] File not found: %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            game_data = json.load(f)

        # Restore state
        state_data = game_data.get("state", {})
        self.state.turn_id = state_data.get("turn_id", 0)
        self.state.genre = state_data.get("genre", self.genre)
        self.state.story_history = state_data.get("story_history", [])

        # Restore KG
        kg_data = game_data.get("kg", {})
        self.kg = KnowledgeGraph.from_dict(kg_data)
        self.conflict_det = ConflictDetector(self.kg)
        self.conflict_resolver = get_resolver(self.conflict_resolution)

        # Restore strategies
        self.conflict_resolution = game_data.get("conflict_resolution", self.conflict_resolution)
        self.extraction_mode = game_data.get("extraction_mode", self.extraction_mode)

        logger.info(
            "[Engine][load_game] Loaded from %s | turn=%d | KG: %d nodes, %d edges",
            path, self.state.turn_id, self.kg.num_nodes, self.kg.num_edges,
        )

    def _auto_save(self) -> None:
        """Auto-save game state if enabled."""
        if not settings.KG_AUTO_SAVE:
            return
        try:
            save_dir = Path(settings.KG_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save latest
            self.save_game(str(save_dir / f"{self.genre}_latest.json"))

            # Periodic snapshot
            if self.state.turn_id % settings.KG_SNAPSHOT_INTERVAL == 0:
                self.save_game(str(save_dir / f"{self.genre}_turn_{self.state.turn_id}.json"))
        except Exception as exc:
            logger.warning("[Engine][auto_save] Failed: %s", exc)
