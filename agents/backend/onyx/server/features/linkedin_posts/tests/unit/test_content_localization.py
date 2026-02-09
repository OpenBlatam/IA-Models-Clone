"""
Content Localization Tests for LinkedIn Posts

This module contains comprehensive tests for content localization functionality,
including translation, cultural adaptation, regional compliance, and multi-language support.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock
from typing import List, Dict, Any
from enum import Enum


# Mock data structures
class Language(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"


class Region(Enum):
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "ap"
    LATIN_AMERICA = "la"
    MIDDLE_EAST = "me"


class MockLocalizedContent:
    def __init__(self, original_content: str, language: Language, region: Region):
        self.original_content = original_content
        self.language = language
        self.region = region
        self.translated_content = f"Translated: {original_content}"
        self.cultural_adaptations = []
        self.compliance_checks = []
        self.id = f"localized_{hash(original_content)}"


class MockTranslationService:
    def __init__(self):
        self.supported_languages = [lang.value for lang in Language]
        self.translation_quality = 0.95


class MockCulturalAdapter:
    def __init__(self):
        self.regional_preferences = {
            Region.NORTH_AMERICA: ["professional", "direct"],
            Region.EUROPE: ["formal", "detailed"],
            Region.ASIA_PACIFIC: ["respectful", "hierarchical"]
        }


class MockComplianceChecker:
    def __init__(self):
        self.regional_regulations = {
            Region.EUROPE: ["GDPR", "privacy_laws"],
            Region.ASIA_PACIFIC: ["data_protection", "content_restrictions"],
            Region.NORTH_AMERICA: ["FTC", "advertising_standards"]
        }


class TestContentLocalization:
    """Test content localization and cultural adaptation"""
    
    @pytest.fixture
    def mock_localization_service(self):
        """Mock localization service"""
        service = AsyncMock()
        
        # Mock translation
        service.translate_content.return_value = {
            "translated_text": "Contenido traducido",
            "confidence_score": 0.95,
            "source_language": "en",
            "target_language": "es"
        }
        
        # Mock cultural adaptation
        service.adapt_culturally.return_value = {
            "adapted_content": "Culturally adapted content",
            "adaptations_made": ["tone_adjustment", "reference_localization"],
            "cultural_score": 0.92
        }
        
        # Mock regional compliance
        service.check_regional_compliance.return_value = {
            "compliant": True,
            "violations": [],
            "recommendations": ["add_disclaimer", "adjust_tone"]
        }
        
        return service
    
    @pytest.fixture
    def mock_translation_service(self):
        """Mock translation service"""
        return MockTranslationService()
    
    @pytest.fixture
    def mock_cultural_adapter(self):
        """Mock cultural adapter"""
        return MockCulturalAdapter()
    
    @pytest.fixture
    def mock_compliance_checker(self):
        """Mock compliance checker"""
        return MockComplianceChecker()
    
    @pytest.fixture
    def mock_localization_repository(self):
        """Mock localization repository"""
        repo = AsyncMock()
        
        # Mock localized content
        repo.get_localized_content.return_value = [
            MockLocalizedContent("Original content", Language.ENGLISH, Region.NORTH_AMERICA),
            MockLocalizedContent("Contenido original", Language.SPANISH, Region.LATIN_AMERICA)
        ]
        
        # Mock localization analytics
        repo.get_localization_analytics.return_value = {
            "translation_accuracy": 0.95,
            "cultural_adaptation_success": 0.92,
            "compliance_rate": 0.98,
            "popular_languages": ["en", "es", "fr"]
        }
        
        return repo
    
    @pytest.fixture
    def post_service(self, mock_localization_repository, mock_localization_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_localization_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            localization_service=mock_localization_service
        )
        return service
    
    async def test_content_translation_workflow(self, post_service, mock_localization_service):
        """Test content translation workflow"""
        # Arrange
        original_content = "Original English content"
        target_language = Language.SPANISH
        source_language = Language.ENGLISH
        
        # Act
        translation_result = await post_service.translate_content(
            original_content, source_language, target_language
        )
        
        # Assert
        assert translation_result is not None
        assert "translated_text" in translation_result
        assert "confidence_score" in translation_result
        assert translation_result["confidence_score"] > 0.8
        mock_localization_service.translate_content.assert_called_once()
    
    async def test_cultural_adaptation_process(self, post_service, mock_localization_service):
        """Test cultural adaptation process"""
        # Arrange
        content = "Content for cultural adaptation"
        target_region = Region.EUROPE
        target_language = Language.FRENCH
        
        # Act
        adaptation_result = await post_service.adapt_content_culturally(
            content, target_region, target_language
        )
        
        # Assert
        assert adaptation_result is not None
        assert "adapted_content" in adaptation_result
        assert "adaptations_made" in adaptation_result
        assert "cultural_score" in adaptation_result
        mock_localization_service.adapt_culturally.assert_called_once()
    
    async def test_regional_compliance_checking(self, post_service, mock_localization_service):
        """Test regional compliance checking"""
        # Arrange
        content = "Content for compliance check"
        target_region = Region.EUROPE
        content_type = "marketing"
        
        # Act
        compliance_result = await post_service.check_regional_compliance(
            content, target_region, content_type
        )
        
        # Assert
        assert compliance_result is not None
        assert "compliant" in compliance_result
        assert "violations" in compliance_result
        assert "recommendations" in compliance_result
        mock_localization_service.check_regional_compliance.assert_called_once()
    
    async def test_multi_language_content_creation(self, post_service, mock_localization_service):
        """Test creating content in multiple languages"""
        # Arrange
        original_content = "Multi-language content"
        target_languages = [Language.SPANISH, Language.FRENCH, Language.GERMAN]
        
        # Act
        localized_versions = await post_service.create_multi_language_content(
            original_content, target_languages
        )
        
        # Assert
        assert localized_versions is not None
        assert len(localized_versions) == len(target_languages)
        assert all(lang in [version["language"] for version in localized_versions] 
                  for lang in target_languages)
        mock_localization_service.translate_content.assert_called()
    
    async def test_localization_quality_assessment(self, post_service, mock_localization_service):
        """Test assessing localization quality"""
        # Arrange
        original_content = "Quality assessment content"
        translated_content = "Contenido de evaluación de calidad"
        target_language = Language.SPANISH
        
        # Act
        quality_score = await post_service.assess_localization_quality(
            original_content, translated_content, target_language
        )
        
        # Assert
        assert quality_score is not None
        assert 0 <= quality_score <= 1
        mock_localization_service.assess_quality.assert_called_once()
    
    async def test_regional_preference_learning(self, post_service, mock_localization_service):
        """Test learning regional preferences"""
        # Arrange
        region = Region.ASIA_PACIFIC
        user_feedback = {
            "preferred_tone": "respectful",
            "content_style": "detailed",
            "engagement_rate": 0.08
        }
        
        # Act
        updated_preferences = await post_service.learn_regional_preferences(
            region, user_feedback
        )
        
        # Assert
        assert updated_preferences is not None
        assert "preferred_tone" in updated_preferences
        mock_localization_service.update_regional_preferences.assert_called_once()
    
    async def test_localization_error_handling(self, post_service, mock_localization_service):
        """Test error handling in localization processes"""
        # Arrange
        mock_localization_service.translate_content.side_effect = Exception("Translation failed")
        
        # Act & Assert
        with pytest.raises(Exception):
            await post_service.translate_content("content", Language.ENGLISH, Language.SPANISH)
    
    async def test_localization_validation(self, post_service, mock_localization_service):
        """Test validation of localization parameters"""
        # Arrange
        unsupported_language = "invalid_lang"
        
        # Act & Assert
        with pytest.raises(ValueError):
            await post_service.validate_localization_language(unsupported_language)
    
    async def test_localization_analytics_tracking(self, post_service, mock_localization_repository):
        """Test tracking localization analytics"""
        # Arrange
        time_period = "last_30_days"
        
        # Act
        analytics = await post_service.get_localization_analytics(time_period)
        
        # Assert
        assert analytics is not None
        assert "translation_accuracy" in analytics
        assert "cultural_adaptation_success" in analytics
        assert "compliance_rate" in analytics
        mock_localization_repository.get_localization_analytics.assert_called_once()
    
    async def test_localization_batch_processing(self, post_service, mock_localization_service):
        """Test batch processing of localization tasks"""
        # Arrange
        content_batch = [
            {"content": "Content 1", "target_language": Language.SPANISH},
            {"content": "Content 2", "target_language": Language.FRENCH},
            {"content": "Content 3", "target_language": Language.GERMAN}
        ]
        
        # Act
        processed_count = await post_service.batch_localize_content(content_batch)
        
        # Assert
        assert processed_count == len(content_batch)
        mock_localization_service.translate_content.assert_called()
    
    async def test_localization_performance_monitoring(self, post_service, mock_localization_service):
        """Test monitoring localization performance metrics"""
        # Arrange
        monitoring_period = "last_24_hours"
        
        # Act
        performance_metrics = await post_service.monitor_localization_performance(monitoring_period)
        
        # Assert
        assert performance_metrics is not None
        assert "translation_throughput" in performance_metrics
        assert "average_quality_score" in performance_metrics
        assert "localization_accuracy" in performance_metrics
        mock_localization_service.get_performance_metrics.assert_called_once()
    
    async def test_localization_cache_management(self, post_service, mock_localization_service):
        """Test caching of localization results"""
        # Arrange
        content = "Cacheable content"
        target_language = Language.SPANISH
        
        # Act
        cached_result = await post_service.get_cached_localization(content, target_language)
        
        # Assert
        assert cached_result is not None
        mock_localization_service.get_cached_translation.assert_called_once()
    
    async def test_localization_workflow_optimization(self, post_service, mock_localization_service):
        """Test optimization of localization workflows"""
        # Arrange
        workflow_config = {
            "parallel_processing": True,
            "quality_threshold": 0.9,
            "cache_enabled": True
        }
        
        # Act
        optimized_workflow = await post_service.optimize_localization_workflow(workflow_config)
        
        # Assert
        assert optimized_workflow is not None
        assert "optimization_score" in optimized_workflow
        mock_localization_service.optimize_workflow.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
