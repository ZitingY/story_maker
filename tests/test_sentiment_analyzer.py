"""Tests for NLU-4: sentiment / emotion analysis."""
import pytest

from src.nlu.sentiment_analyzer import SentimentAnalyzer, EMOTION_LABELS


class TestSentimentAnalyzerRuleFallback:
    """Test the rule-based fallback mode."""

    @pytest.fixture
    def analyzer(self):
        """Analyzer with rule fallback (no model loaded)."""
        a = SentimentAnalyzer()
        assert a.backend == "rule_fallback"
        return a

    def test_returns_dict(self, analyzer):
        result = analyzer.analyze("Hello world")
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert "scores" in result

    def test_anger_detection(self, analyzer):
        result = analyzer.analyze("I'm furious and want to destroy everything!")
        assert result["emotion"] == "anger"
        assert result["confidence"] > 0

    def test_fear_detection(self, analyzer):
        result = analyzer.analyze("I'm terrified, we need to flee from this danger!")
        assert result["emotion"] == "fear"
        assert result["confidence"] > 0

    def test_joy_detection(self, analyzer):
        result = analyzer.analyze("This is wonderful and amazing! I'm so happy!")
        assert result["emotion"] == "joy"
        assert result["confidence"] > 0

    def test_sadness_detection(self, analyzer):
        result = analyzer.analyze("I feel so sad and lost. I miss them terribly.")
        assert result["emotion"] == "sadness"
        assert result["confidence"] > 0

    def test_surprise_detection(self, analyzer):
        result = analyzer.analyze("Wow, that's completely unexpected! I'm shocked!")
        assert result["emotion"] == "surprise"
        assert result["confidence"] > 0

    def test_disgust_detection(self, analyzer):
        result = analyzer.analyze("That's disgusting and vile! So repulsive!")
        assert result["emotion"] == "disgust"
        assert result["confidence"] > 0

    def test_neutral_default(self, analyzer):
        result = analyzer.analyze("I walk to the door.")
        assert result["emotion"] == "neutral"

    def test_confidence_range(self, analyzer):
        result = analyzer.analyze("Attack the dragon!")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_contains_all_labels(self, analyzer):
        result = analyzer.analyze("I'm happy!")
        for label in EMOTION_LABELS:
            assert label in result["scores"]

    def test_empty_input(self, analyzer):
        result = analyzer.analyze("")
        assert result["emotion"] == "neutral"


class TestSentimentAnalyzerAPI:
    """Test the public API."""

    def test_load_method_exists(self):
        a = SentimentAnalyzer()
        # Should not raise even if model isn't available
        a.load()
        assert a.backend in ("rule_fallback", "distilroberta")

    def test_backend_tracking(self):
        a = SentimentAnalyzer()
        assert a.backend == "rule_fallback"
