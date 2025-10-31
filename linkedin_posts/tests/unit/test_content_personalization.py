import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Mock data structures
class MockUserProfile:
    def __init__(self):
        self.interests = []
        self.industry = ""
        self.experience_level = ""
        self.preferred_content_types = []
        self.engagement_history = []

class MockPersonalizationEngine:
    def __init__(self):
        self.user_preferences = {}
        self.content_recommendations = []
        self.adaptation_rules = []

class MockPersonalizedContent:
    def __init__(self, original_content: str, personalized_content: str, score: float):
        self.original_content = original_content
        self.personalized_content = personalized_content
        self.score = score
        self.adaptations = []
        self.recommendations = []

class TestContentPersonalization:
    """Test content personalization and user preference learning"""
    
    @pytest.fixture
    def mock_personalization_service(self):
        """Mock personalization service"""
        service = AsyncMock()
        
        # Mock user preference learning
        service.learn_user_preferences.return_value = {
            "interests": ["technology", "innovation", "leadership"],
            "content_preferences": ["articles", "videos", "infographics"],
            "engagement_patterns": {"morning": 0.8, "afternoon": 0.6, "evening": 0.4},
            "confidence_score": 0.85
        }
        
        # Mock content personalization
        service.personalize_content.return_value = MockPersonalizedContent(
            original_content="Original post content",
            personalized_content="Personalized post content for tech professionals",
            score=0.92
        )
        
        # Mock recommendation generation
        service.generate_recommendations.return_value = [
            {"type": "article", "title": "Tech Leadership Tips", "relevance": 0.9},
            {"type": "video", "title": "Innovation Strategies", "relevance": 0.85},
            {"type": "infographic", "title": "Industry Trends", "relevance": 0.78}
        ]
        
        return service
    
    @pytest.fixture
    def mock_user_repository(self):
        """Mock user repository for personalization tests"""
        repo = AsyncMock()
        
        # Mock user profile
        repo.get_user_profile.return_value = {
            "id": "user_123",
            "interests": ["technology", "leadership", "innovation"],
            "industry": "technology",
            "experience_level": "senior",
            "preferred_content_types": ["articles", "videos"],
            "engagement_history": [
                {"post_id": "post_1", "engagement_type": "like", "timestamp": datetime.now() - timedelta(days=1)},
                {"post_id": "post_2", "engagement_type": "share", "timestamp": datetime.now() - timedelta(days=2)}
            ]
        }
        
        # Mock user preferences
        repo.get_user_preferences.return_value = {
            "content_tone": "professional",
            "posting_frequency": "daily",
            "preferred_topics": ["leadership", "technology", "innovation"],
            "avoided_topics": ["politics", "controversial"]
        }
        
        return repo
    
    @pytest.fixture
    def post_service(self, mock_user_repository, mock_personalization_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_user_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            personalization_service=mock_personalization_service
        )
        return service
    
    async def test_user_preference_learning(self, post_service, mock_personalization_service):
        """Test learning user preferences from engagement history"""
        # Arrange
        user_id = "user_123"
        engagement_data = [
            {"post_id": "post_1", "action": "like", "content_type": "article"},
            {"post_id": "post_2", "action": "share", "content_type": "video"},
            {"post_id": "post_3", "action": "comment", "content_type": "infographic"}
        ]
        
        # Act
        preferences = await post_service.learn_user_preferences(user_id, engagement_data)
        
        # Assert
        assert preferences is not None
        assert "interests" in preferences
        assert "content_preferences" in preferences
        assert "engagement_patterns" in preferences
        assert preferences["confidence_score"] > 0.8
        mock_personalization_service.learn_user_preferences.assert_called_once()
    
    async def test_content_personalization_workflow(self, post_service, mock_personalization_service):
        """Test complete content personalization workflow"""
        # Arrange
        user_id = "user_123"
        content = "General post about business trends"
        user_profile = {
            "interests": ["technology", "innovation"],
            "industry": "technology",
            "experience_level": "senior"
        }
        
        # Act
        personalized_content = await post_service.personalize_content(user_id, content, user_profile)
        
        # Assert
        assert personalized_content is not None
        assert personalized_content.original_content == content
        assert personalized_content.personalized_content != content
        assert personalized_content.score > 0.8
        mock_personalization_service.personalize_content.assert_called_once()
    
    async def test_dynamic_content_adaptation(self, post_service, mock_personalization_service):
        """Test dynamic content adaptation based on user behavior"""
        # Arrange
        user_id = "user_123"
        base_content = "Standard business post"
        user_behavior = {
            "recent_engagements": ["leadership", "technology"],
            "time_of_day": "morning",
            "device_type": "mobile"
        }
        
        # Act
        adapted_content = await post_service.adapt_content_dynamically(user_id, base_content, user_behavior)
        
        # Assert
        assert adapted_content is not None
        assert "adapted_content" in adapted_content
        assert "adaptation_reason" in adapted_content
        assert adapted_content["relevance_score"] > 0.7
        mock_personalization_service.personalize_content.assert_called()
    
    async def test_personalized_recommendations(self, post_service, mock_personalization_service):
        """Test generating personalized content recommendations"""
        # Arrange
        user_id = "user_123"
        user_profile = {
            "interests": ["technology", "leadership"],
            "recent_activity": ["article_read", "video_watched"]
        }
        
        # Act
        recommendations = await post_service.get_personalized_recommendations(user_id, user_profile)
        
        # Assert
        assert recommendations is not None
        assert len(recommendations) > 0
        assert all("type" in rec for rec in recommendations)
        assert all("relevance" in rec for rec in recommendations)
        assert all(rec["relevance"] > 0.7 for rec in recommendations)
        mock_personalization_service.generate_recommendations.assert_called_once()
    
    async def test_user_behavior_analysis(self, post_service, mock_personalization_service):
        """Test analyzing user behavior patterns"""
        # Arrange
        user_id = "user_123"
        behavior_data = [
            {"action": "like", "content_type": "article", "timestamp": datetime.now() - timedelta(hours=1)},
            {"action": "share", "content_type": "video", "timestamp": datetime.now() - timedelta(hours=2)},
            {"action": "comment", "content_type": "infographic", "timestamp": datetime.now() - timedelta(hours=3)}
        ]
        
        # Act
        analysis = await post_service.analyze_user_behavior(user_id, behavior_data)
        
        # Assert
        assert analysis is not None
        assert "preferred_content_types" in analysis
        assert "engagement_patterns" in analysis
        assert "peak_activity_times" in analysis
        assert "content_preferences" in analysis
        mock_personalization_service.learn_user_preferences.assert_called()
    
    async def test_content_tone_personalization(self, post_service, mock_personalization_service):
        """Test personalizing content tone based on user preferences"""
        # Arrange
        user_id = "user_123"
        content = "Business update post"
        user_preferences = {
            "preferred_tone": "professional",
            "communication_style": "direct",
            "formality_level": "high"
        }
        
        # Act
        personalized_tone = await post_service.personalize_content_tone(user_id, content, user_preferences)
        
        # Assert
        assert personalized_tone is not None
        assert "tone_adjusted_content" in personalized_tone
        assert "tone_score" in personalized_tone
        assert personalized_tone["tone_score"] > 0.8
        mock_personalization_service.personalize_content.assert_called()
    
    async def test_timing_optimization(self, post_service, mock_personalization_service):
        """Test optimizing post timing based on user behavior"""
        # Arrange
        user_id = "user_123"
        user_behavior = {
            "activity_patterns": {"morning": 0.9, "afternoon": 0.6, "evening": 0.3},
            "timezone": "EST",
            "work_schedule": "9-5"
        }
        
        # Act
        optimal_timing = await post_service.get_optimal_posting_time(user_id, user_behavior)
        
        # Assert
        assert optimal_timing is not None
        assert "optimal_time" in optimal_timing
        assert "confidence_score" in optimal_timing
        assert "reasoning" in optimal_timing
        assert optimal_timing["confidence_score"] > 0.7
        mock_personalization_service.learn_user_preferences.assert_called()
    
    async def test_content_frequency_personalization(self, post_service, mock_personalization_service):
        """Test personalizing content frequency based on user engagement"""
        # Arrange
        user_id = "user_123"
        engagement_history = [
            {"date": datetime.now() - timedelta(days=1), "engagement_rate": 0.8},
            {"date": datetime.now() - timedelta(days=2), "engagement_rate": 0.6},
            {"date": datetime.now() - timedelta(days=3), "engagement_rate": 0.9}
        ]
        
        # Act
        frequency_recommendation = await post_service.get_personalized_frequency(user_id, engagement_history)
        
        # Assert
        assert frequency_recommendation is not None
        assert "recommended_frequency" in frequency_recommendation
        assert "reasoning" in frequency_recommendation
        assert "optimal_times" in frequency_recommendation
        mock_personalization_service.learn_user_preferences.assert_called()
    
    async def test_cross_platform_personalization(self, post_service, mock_personalization_service):
        """Test personalizing content for different platforms"""
        # Arrange
        user_id = "user_123"
        content = "General business content"
        platform_preferences = {
            "linkedin": {"tone": "professional", "length": "medium"},
            "twitter": {"tone": "casual", "length": "short"},
            "facebook": {"tone": "friendly", "length": "long"}
        }
        
        # Act
        platform_content = await post_service.personalize_for_platforms(user_id, content, platform_preferences)
        
        # Assert
        assert platform_content is not None
        assert "linkedin" in platform_content
        assert "twitter" in platform_content
        assert "facebook" in platform_content
        assert all("personalized_content" in platform_content[platform] for platform in platform_content)
        mock_personalization_service.personalize_content.assert_called()
    
    async def test_personalization_performance_metrics(self, post_service, mock_user_repository):
        """Test personalization performance metrics"""
        # Arrange
        user_id = "user_123"
        
        # Act
        metrics = await post_service.get_personalization_metrics(user_id)
        
        # Assert
        assert metrics is not None
        assert "engagement_improvement" in metrics
        assert "personalization_accuracy" in metrics
        assert "user_satisfaction" in metrics
        assert "content_relevance" in metrics
        mock_user_repository.get_user_profile.assert_called()
    
    async def test_personalization_error_handling(self, post_service, mock_personalization_service):
        """Test error handling in personalization"""
        # Arrange
        mock_personalization_service.personalize_content.side_effect = Exception("Personalization service error")
        
        # Act & Assert
        with pytest.raises(Exception):
            await post_service.personalize_content("user_123", "content", {})
    
    async def test_personalization_caching(self, post_service, mock_personalization_service, mock_cache_service):
        """Test personalization result caching"""
        # Arrange
        user_id = "user_123"
        cache_key = f"personalization_{user_id}"
        
        # Mock cache hit
        mock_cache_service.get.return_value = {
            "personalized_content": "Cached personalized content",
            "score": 0.92
        }
        
        # Act
        result = await post_service.personalize_content(user_id, "content", {})
        
        # Assert
        assert result is not None
        mock_cache_service.get.assert_called_with(cache_key)
        # Should not call personalization service if cached
        mock_personalization_service.personalize_content.assert_not_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
