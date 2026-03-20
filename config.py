"""Global configuration for StoryWeaver — hybrid NLU + API-based NLG."""
from pathlib import Path
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:  # graceful fallback
    from pydantic import BaseSettings  # type: ignore[no-redef]

from pydantic import Field


class Settings(BaseSettings):
    """Centralised settings read from .env file automatically."""

    # ── Paths ──────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = Path(__file__).parent / "data"

    # ── OpenAI / LLM API ──────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_BASE_URL: str = Field(default="", description="OpenAI-compatible API base URL (e.g. https://your-server.com/v1)")
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 1024
    OPENAI_TEMPERATURE: float = 0.85
    OPENAI_TOP_P: float = 0.95

    # ── NLU Config ────────────────────────────────────────
    INTENT_MODEL_NAME: str = "distilbert-base-uncased"
    INTENT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "intent_classifier"
    INTENT_MAX_LENGTH: int = 128
    INTENT_CPU_BATCH_SIZE: int = 8
    INTENT_LABELS: List[str] = [
        "action", "dialogue", "explore", "use_item",
        "ask_info", "rest", "trade", "other",
    ]
    SPACY_MODEL: str = "en_core_web_sm"

    # ── NLG Config ────────────────────────────────────────
    NUM_OPTIONS: int = 3

    # ── Knowledge Graph Config ────────────────────────────
    KG_MAX_NODES: int = 200
    KG_ENTITY_TYPES: List[str] = ["person", "location", "item", "creature", "event"]
    KG_RELATION_TYPES: List[str] = [
        "located_at", "possesses", "ally_of", "enemy_of",
        "knows", "part_of", "caused_by", "has_attribute",
        "causes", "prevents", "enables", "follows",
    ]

    # ── KG Strategy Config ────────────────────────────────
    KG_CONFLICT_RESOLUTION: str = "llm_arbitrate"
    # Options: "keep_latest" | "llm_arbitrate"

    KG_EXTRACTION_MODE: str = "dual_extract"
    # Options: "story_only" | "dual_extract"

    KG_IMPORTANCE_MODE: str = "composite"
    # Options: "degree_only" | "composite"

    KG_SUMMARY_MODE: str = "layered"
    # Options: "flat" | "layered"

    # ── KG Tuning Params ──────────────────────────────────
    KG_IMPORTANCE_DECAY_FACTOR: float = 0.95
    KG_RELATION_DECAY_FACTOR: float = 0.90
    KG_RELATION_MIN_CONFIDENCE: float = 0.2
    KG_IMPORTANCE_MENTION_BOOST: float = 0.15
    KG_IMPORTANCE_PLAYER_BOOST: float = 0.3
    KG_MAX_TIMELINE_ENTRIES: int = 5

    # ── Game Config ───────────────────────────────────────
    NARRATIVE_HISTORY_WINDOW: int = 6
    MAX_CONTEXT_TOKENS: int = 512

    # ── KG Persistence ───────────────────────────────────
    KG_SAVE_DIR: Path = PROJECT_ROOT / "saves"
    KG_AUTO_SAVE: bool = True
    KG_SNAPSHOT_INTERVAL: int = 5  # Save snapshot every N turns

    # ── Streamlit ─────────────────────────────────────────
    STREAMLIT_PORT: int = 7860

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton settings instance used by every module
settings = Settings()
