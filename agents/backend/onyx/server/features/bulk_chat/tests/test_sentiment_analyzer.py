"""
Tests for Sentiment Analyzer
=============================
"""

import pytest
from ..core.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def sentiment_analyzer():
    """Create sentiment analyzer for testing."""
    return SentimentAnalyzer()


def test_analyze_sentiment_positive(sentiment_analyzer):
    """Test analyzing positive sentiment."""
    result = sentiment_analyzer.analyze("I love this product! It's amazing!")
    
    assert result is not None
    assert "sentiment" in result or "polarity" in result
    # Should be positive
    polarity = result.get("polarity", result.get("sentiment_score", 0))
    assert polarity > 0 or result.get("sentiment", "").lower() in ["positive", "pos"]


def test_analyze_sentiment_negative(sentiment_analyzer):
    """Test analyzing negative sentiment."""
    result = sentiment_analyzer.analyze("I hate this! It's terrible!")
    
    assert result is not None
    # Should be negative
    polarity = result.get("polarity", result.get("sentiment_score", 0))
    assert polarity < 0 or result.get("sentiment", "").lower() in ["negative", "neg"]


def test_analyze_sentiment_neutral(sentiment_analyzer):
    """Test analyzing neutral sentiment."""
    result = sentiment_analyzer.analyze("This is a table.")
    
    assert result is not None
    polarity = result.get("polarity", result.get("sentiment_score", 0))
    # Neutral should be around 0
    assert abs(polarity) < 0.3 or result.get("sentiment", "").lower() in ["neutral", "neu"]


def test_extract_emotions(sentiment_analyzer):
    """Test extracting emotions."""
    result = sentiment_analyzer.extract_emotions("I'm so happy and excited about this!")
    
    assert result is not None
    assert isinstance(result, dict) or isinstance(result, list)
    # Should contain emotion information
    assert len(result) > 0


def test_extract_keywords(sentiment_analyzer):
    """Test extracting keywords."""
    result = sentiment_analyzer.extract_keywords(
        "Python programming is amazing and powerful for data science"
    )
    
    assert result is not None
    assert isinstance(result, list)
    # Should contain keywords
    assert len(result) > 0


def test_get_sentiment_analyzer_summary(sentiment_analyzer):
    """Test getting sentiment analyzer summary."""
    sentiment_analyzer.analyze("Positive text")
    sentiment_analyzer.analyze("Negative text")
    
    summary = sentiment_analyzer.get_sentiment_analyzer_summary()
    
    assert summary is not None
    assert "total_analyses" in summary or "analyses_count" in summary


