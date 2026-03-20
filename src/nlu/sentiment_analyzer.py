"""Sentiment / emotion analysis for player input.

6-class Ekman model: anger, disgust, fear, joy, sadness, surprise + neutral.

Primary: HuggingFace distilroberta model.
Fallback: keyword-based rule matching.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# Keyword fallback rules
_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "anger": [
        "angry", "furious", "rage", "kill", "damn", "stupid", "hate",
        "destroy", "revenge", "wrath", "annoyed", "irritated", "frustrated",
        "attack", "fight", "smash", "crush", "vengeance",
    ],
    "disgust": [
        "disgusting", "gross", "vile", "repulsive", "revolting", "nasty",
        "foul", "sickening", "horrible", "terrible", "awful", "rotten",
        "filthy", "putrid", "loathsome",
    ],
    "fear": [
        "afraid", "scared", "terrified", "danger", "flee", "escape",
        "hide", "panic", "dread", "horror", "frightened", "worried",
        "nervous", "anxious", "creepy", "spooky", "threatening", "menacing",
    ],
    "joy": [
        "happy", "wonderful", "great", "excited", "love", "amazing",
        "fantastic", "excellent", "delighted", "cheerful", "glad", "pleased",
        "celebrate", "triumph", "victory", "win", "success", "beautiful",
        "brilliant", "magnificent",
    ],
    "sadness": [
        "sad", "sorry", "lost", "miss", "cry", "mourn", "grief",
        "depressed", "unhappy", "miserable", "heartbroken", "lonely",
        "hopeless", "tragic", "weep", "tears", "despair", "gloomy",
    ],
    "surprise": [
        "wow", "amazing", "unexpected", "sudden", "shocked", "astonished",
        "startled", "incredible", "unbelievable", "surprised", "stunned",
        "bewildered", "astounded",
    ],
}


class SentimentAnalyzer:
    """Analyze emotion in player input text."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        self.backend = "rule_fallback"

    def load(self) -> None:
        """Try to load the emotion classification model."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "j-hartmann/emotion-english-distilroberta-base"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.backend = "distilroberta"
            logger.info("Sentiment analyzer loaded: %s", model_name)
        except Exception as exc:
            logger.warning("Sentiment model unavailable (%s) – using rule_fallback.", exc)
            self.model = None
            self.tokenizer = None
            self.backend = "rule_fallback"

    def analyze(self, text: str) -> Dict[str, object]:
        """Analyze the emotion in text.

        Returns:
            {
                "emotion": str,          # dominant emotion label
                "confidence": float,     # confidence score (0-1)
                "scores": Dict[str, float]  # all emotion scores
            }
        """
        if self.model is not None and self.tokenizer is not None:
            return self._model_analyze(text)
        return self._rule_fallback(text)

    def _model_analyze(self, text: str) -> Dict[str, object]:
        """Analyze using the neural model."""
        import torch

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        scores = {}
        for i, label in enumerate(EMOTION_LABELS):
            if i < len(probs):
                scores[label] = round(probs[i].item(), 4)

        # Find dominant emotion
        max_idx = torch.argmax(probs).item()
        dominant = EMOTION_LABELS[max_idx] if max_idx < len(EMOTION_LABELS) else "neutral"
        confidence = probs[max_idx].item()

        return {
            "emotion": dominant,
            "confidence": round(confidence, 4),
            "scores": scores,
        }

    def _rule_fallback(self, text: str) -> Dict[str, object]:
        """Keyword-based emotion detection."""
        text_lower = text.lower()
        scores: Dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}

        for emotion, keywords in _EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                scores[emotion] = round(min(matches / 3.0, 1.0), 4)

        # Find dominant emotion
        max_score = 0.0
        dominant = "neutral"
        for emotion, score in scores.items():
            if score > max_score:
                max_score = score
                dominant = emotion

        if max_score < 0.1:
            dominant = "neutral"
            max_score = 0.5

        return {
            "emotion": dominant,
            "confidence": round(max_score, 4),
            "scores": scores,
        }
