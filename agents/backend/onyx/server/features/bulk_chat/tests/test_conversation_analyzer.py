"""
Tests for Conversation Analyzer
================================
"""

import pytest
from ..core.conversation_analyzer import ConversationAnalyzer
from ..core.chat_session import ChatSession, ChatMessage


@pytest.fixture
def analyzer():
    """Create conversation analyzer for testing."""
    return ConversationAnalyzer()


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    session = ChatSession(
        session_id="test_session",
        user_id="test_user"
    )
    session.messages = [
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing well, thank you!"),
        ChatMessage(role="user", content="What's the weather like?"),
    ]
    return session


def test_analyze_sentiment(analyzer, sample_session):
    """Test sentiment analysis."""
    analysis = analyzer.analyze_sentiment(sample_session)
    
    assert analysis is not None
    assert "sentiment" in analysis or "polarity" in analysis


def test_extract_topics(analyzer, sample_session):
    """Test topic extraction."""
    topics = analyzer.extract_topics(sample_session)
    
    assert topics is not None
    assert isinstance(topics, list) or isinstance(topics, dict)


def test_get_statistics(analyzer, sample_session):
    """Test getting conversation statistics."""
    stats = analyzer.get_statistics(sample_session)
    
    assert stats is not None
    assert "total_messages" in stats or "message_count" in stats
    assert stats.get("total_messages", stats.get("message_count", 0)) >= 3


def test_analyze_conversation(analyzer, sample_session):
    """Test full conversation analysis."""
    analysis = analyzer.analyze_conversation(sample_session)
    
    assert analysis is not None
    assert isinstance(analysis, dict)
    # Should contain various analysis results
    assert len(analysis) > 0


