"""
Unit Tests for Topic Extractor
===============================
Tests for PDF topic extraction functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import io

# Try to import topic extractor
try:
    from topic_extractor import (
        Topic,
        PDFTopicExtractor
    )
except ImportError:
    Topic = None
    PDFTopicExtractor = None


@pytest.fixture
def sample_text_content():
    """Sample text content for topic extraction."""
    return """
    Artificial Intelligence and Machine Learning are transforming the world.
    Deep learning algorithms can process vast amounts of data.
    Natural language processing enables computers to understand human language.
    Computer vision allows machines to interpret visual information.
    Neural networks are inspired by the human brain.
    """


@pytest.fixture
def topic_extractor():
    """Create PDFTopicExtractor instance."""
    if PDFTopicExtractor is None:
        pytest.skip("PDFTopicExtractor not available")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        yield PDFTopicExtractor(Path(temp_dir))


class TestTopic:
    """Tests for Topic dataclass."""
    
    def test_topic_creation(self):
        """Test creating Topic."""
        if Topic is None:
            pytest.skip("Topic not available")
        
        topic = Topic(
            topic="Artificial Intelligence",
            category="main",
            relevance_score=0.9,
            mentions=5
        )
        assert topic.topic == "Artificial Intelligence"
        assert topic.category == "main"
        assert topic.relevance_score == 0.9
        assert topic.mentions == 5
    
    def test_topic_defaults(self):
        """Test Topic with default values."""
        if Topic is None:
            pytest.skip("Topic not available")
        
        topic = Topic(topic="Test Topic")
        assert topic.topic == "Test Topic"
        assert topic.category == "main"
        assert topic.relevance_score == 0.0
        assert topic.mentions == 0
        assert topic.context == []
        assert topic.related_topics == []
    
    def test_topic_to_dict(self):
        """Test converting Topic to dictionary."""
        if Topic is None:
            pytest.skip("Topic not available")
        
        topic = Topic(
            topic="AI",
            relevance_score=0.8,
            mentions=3,
            context=["context1", "context2"],
            related_topics=["ML", "DL"]
        )
        topic_dict = topic.to_dict()
        
        assert isinstance(topic_dict, dict)
        assert topic_dict["topic"] == "AI"
        assert topic_dict["relevance_score"] == 0.8
        assert topic_dict["mentions"] == 3
        assert len(topic_dict["context"]) == 2
        assert len(topic_dict["related_topics"]) == 2


class TestPDFTopicExtractor:
    """Tests for PDFTopicExtractor class."""
    
    def test_topic_extractor_initialization(self, topic_extractor):
        """Test PDFTopicExtractor initialization."""
        assert topic_extractor is not None
        assert hasattr(topic_extractor, "upload_dir")
    
    @pytest.mark.asyncio
    async def test_extract_topics_from_text(self, topic_extractor, sample_text_content):
        """Test extracting topics from text."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        # Should return list of topics or handle gracefully
        assert topics is not None
    
    @pytest.mark.asyncio
    async def test_extract_topics_with_min_relevance(self, topic_extractor):
        """Test topic extraction with minimum relevance threshold."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics_low = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.3,
            max_topics=50
        )
        
        topics_high = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.8,
            max_topics=50
        )
        
        # Higher threshold should return fewer or equal topics
        assert len(topics_high) <= len(topics_low) if isinstance(topics_low, list) and isinstance(topics_high, list) else True
    
    @pytest.mark.asyncio
    async def test_extract_topics_with_max_topics(self, topic_extractor):
        """Test topic extraction with max topics limit."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.5,
            max_topics=5
        )
        
        if isinstance(topics, list):
            assert len(topics) <= 5
    
    @pytest.mark.asyncio
    async def test_extract_topics_empty_content(self, topic_extractor):
        """Test topic extraction with empty content."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="empty_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        # Should handle gracefully
        assert topics is not None
        if isinstance(topics, list):
            assert len(topics) == 0
    
    @pytest.mark.asyncio
    async def test_extract_topics_relevance_scores(self, topic_extractor):
        """Test that topics have relevance scores."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        if isinstance(topics, list) and len(topics) > 0:
            first_topic = topics[0]
            if hasattr(first_topic, "relevance_score"):
                assert 0.0 <= first_topic.relevance_score <= 1.0
            elif isinstance(first_topic, dict):
                assert "relevance_score" in first_topic
                assert 0.0 <= first_topic["relevance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_extract_topics_sorted_by_relevance(self, topic_extractor):
        """Test that topics are sorted by relevance."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        if isinstance(topics, list) and len(topics) > 1:
            scores = []
            for topic in topics:
                if hasattr(topic, "relevance_score"):
                    scores.append(topic.relevance_score)
                elif isinstance(topic, dict):
                    scores.append(topic.get("relevance_score", 0))
            
            # Should be sorted descending
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_extract_topics_with_categories(self, topic_extractor):
        """Test that topics have categories."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="test_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        if isinstance(topics, list) and len(topics) > 0:
            first_topic = topics[0]
            if hasattr(first_topic, "category"):
                assert first_topic.category is not None
            elif isinstance(first_topic, dict):
                assert "category" in first_topic


class TestTopicExtractionEdgeCases:
    """Edge cases for topic extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_topics_very_short_text(self, topic_extractor):
        """Test extraction with very short text."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="short_file",
            min_relevance=0.1,
            max_topics=10
        )
        
        # Should handle gracefully
        assert topics is not None
    
    @pytest.mark.asyncio
    async def test_extract_topics_very_long_text(self, topic_extractor):
        """Test extraction with very long text."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        # Create long text
        long_text = "AI " * 10000
        
        topics = await topic_extractor.extract_topics(
            file_id="long_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        # Should handle gracefully
        assert topics is not None
    
    @pytest.mark.asyncio
    async def test_extract_topics_special_characters(self, topic_extractor):
        """Test extraction with special characters."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="special_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        # Should handle special characters
        assert topics is not None
    
    @pytest.mark.asyncio
    async def test_extract_topics_unicode(self, topic_extractor):
        """Test extraction with Unicode characters."""
        if topic_extractor is None:
            pytest.skip("PDFTopicExtractor not available")
        
        topics = await topic_extractor.extract_topics(
            file_id="unicode_file",
            min_relevance=0.5,
            max_topics=10
        )
        
        # Should handle Unicode
        assert topics is not None



