"""
Content Moderation and Filtering Tests
=====================================

Comprehensive tests for content moderation and filtering features including:
- Content screening and validation
- Spam detection and prevention
- Inappropriate content filtering
- Moderation workflows and approval
- Content quality checks and scoring
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from typing import List, Dict, Any, Optional

# Test data
SAMPLE_MODERATION_CONFIG = {
    "screening_rules": {
        "profanity_check": True,
        "spam_detection": True,
        "inappropriate_content": True,
        "copyright_check": True,
        "brand_safety": True
    },
    "filtering_rules": {
        "auto_filter": True,
        "manual_review": True,
        "appeal_process": True
    },
    "quality_thresholds": {
        "min_quality_score": 0.7,
        "max_spam_score": 0.3,
        "min_readability": 0.6
    }
}

SAMPLE_CONTENT_FOR_MODERATION = {
    "post_id": str(uuid4()),
    "content": "This is a legitimate business post about AI technology",
    "author_id": str(uuid4()),
    "created_at": datetime.now(),
    "language": "en",
    "content_type": "text",
    "has_media": False,
    "mentions": ["@techcompany"],
    "hashtags": ["#AI", "#Technology"],
    "links": ["https://example.com/article"]
}

SAMPLE_MODERATION_RESULT = {
    "post_id": str(uuid4()),
    "screening_result": {
        "passed": True,
        "spam_score": 0.1,
        "inappropriate_score": 0.05,
        "quality_score": 0.85,
        "readability_score": 0.78
    },
    "filtering_result": {
        "filtered": False,
        "filter_reason": None,
        "requires_review": False
    },
    "moderation_status": "approved",
    "moderated_at": datetime.now(),
    "moderator_id": str(uuid4()),
    "review_notes": "Content meets quality standards"
}

SAMPLE_SPAM_DETECTION = {
    "spam_score": 0.85,
    "spam_indicators": [
        "excessive_hashtags",
        "repetitive_content",
        "suspicious_links"
    ],
    "detection_method": "ml_model",
    "confidence": 0.92
}

SAMPLE_QUALITY_ASSESSMENT = {
    "overall_score": 0.78,
    "readability": 0.75,
    "originality": 0.80,
    "relevance": 0.85,
    "engagement_potential": 0.70,
    "improvement_suggestions": [
        "Add more specific examples",
        "Include industry statistics",
        "Use more engaging headlines"
    ]
}

class TestContentModerationFiltering:
    """Test content moderation and filtering features"""
    
    @pytest.fixture
    def mock_moderation_service(self):
        """Mock moderation service"""
        service = AsyncMock()
        service.screen_content.return_value = SAMPLE_MODERATION_RESULT
        service.detect_spam.return_value = SAMPLE_SPAM_DETECTION
        service.assess_quality.return_value = SAMPLE_QUALITY_ASSESSMENT
        service.filter_content.return_value = {
            "filtered": False,
            "reason": None,
            "requires_review": False
        }
        service.approve_content.return_value = True
        service.reject_content.return_value = True
        service.flag_for_review.return_value = True
        return service
    
    @pytest.fixture
    def mock_moderation_repository(self):
        """Mock moderation repository"""
        repo = AsyncMock()
        repo.save_moderation_result.return_value = SAMPLE_MODERATION_RESULT
        repo.get_moderation_result.return_value = SAMPLE_MODERATION_RESULT
        repo.update_moderation_status.return_value = SAMPLE_MODERATION_RESULT
        repo.get_moderation_history.return_value = [SAMPLE_MODERATION_RESULT]
        repo.get_pending_reviews.return_value = [SAMPLE_CONTENT_FOR_MODERATION]
        repo.get_moderation_analytics.return_value = {
            "total_posts": 15000,
            "approved_count": 13500,
            "rejected_count": 1000,
            "pending_review": 500,
            "avg_moderation_time": 2.5
        }
        return repo
    
    @pytest.fixture
    def mock_filtering_service(self):
        """Mock filtering service"""
        service = AsyncMock()
        service.apply_filters.return_value = {
            "passed_filters": True,
            "filter_results": {
                "profanity": False,
                "spam": False,
                "inappropriate": False
            }
        }
        service.check_profanity.return_value = {"contains_profanity": False, "score": 0.1}
        service.check_spam.return_value = {"is_spam": False, "score": 0.15}
        service.check_inappropriate.return_value = {"is_inappropriate": False, "score": 0.05}
        return service
    
    @pytest.fixture
    def post_service(self, mock_moderation_repository, mock_moderation_service, mock_filtering_service):
        from services.post_service import PostService
        service = PostService(
            repository=mock_moderation_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            moderation_service=mock_moderation_service,
            filtering_service=mock_filtering_service
        )
        return service
    
    async def test_screen_content(self, post_service, mock_moderation_service):
        """Test content screening functionality"""
        content = SAMPLE_CONTENT_FOR_MODERATION
        
        result = await post_service.screen_content(content)
        
        mock_moderation_service.screen_content.assert_called_once_with(content)
        assert result == SAMPLE_MODERATION_RESULT
        assert "screening_result" in result
        assert "filtering_result" in result
    
    async def test_detect_spam(self, post_service, mock_moderation_service):
        """Test spam detection"""
        content = "Buy now! Limited time offer! Click here!"
        
        result = await post_service.detect_spam(content)
        
        mock_moderation_service.detect_spam.assert_called_once_with(content)
        assert result == SAMPLE_SPAM_DETECTION
        assert "spam_score" in result
        assert "spam_indicators" in result
    
    async def test_assess_content_quality(self, post_service, mock_moderation_service):
        """Test content quality assessment"""
        content = SAMPLE_CONTENT_FOR_MODERATION
        
        result = await post_service.assess_content_quality(content)
        
        mock_moderation_service.assess_quality.assert_called_once_with(content)
        assert result == SAMPLE_QUALITY_ASSESSMENT
        assert "overall_score" in result
        assert "improvement_suggestions" in result
    
    async def test_filter_content(self, post_service, mock_moderation_service):
        """Test content filtering"""
        content = SAMPLE_CONTENT_FOR_MODERATION
        
        result = await post_service.filter_content(content)
        
        mock_moderation_service.filter_content.assert_called_once_with(content)
        assert "filtered" in result
        assert "reason" in result
        assert "requires_review" in result
    
    async def test_approve_content(self, post_service, mock_moderation_service):
        """Test content approval"""
        post_id = str(uuid4())
        moderator_id = str(uuid4())
        notes = "Content meets guidelines"
        
        result = await post_service.approve_content(post_id, moderator_id, notes)
        
        mock_moderation_service.approve_content.assert_called_once_with(post_id, moderator_id, notes)
        assert result is True
    
    async def test_reject_content(self, post_service, mock_moderation_service):
        """Test content rejection"""
        post_id = str(uuid4())
        moderator_id = str(uuid4())
        reason = "Violates community guidelines"
        
        result = await post_service.reject_content(post_id, moderator_id, reason)
        
        mock_moderation_service.reject_content.assert_called_once_with(post_id, moderator_id, reason)
        assert result is True
    
    async def test_flag_for_review(self, post_service, mock_moderation_service):
        """Test flagging content for review"""
        post_id = str(uuid4())
        reason = "Needs manual review"
        
        result = await post_service.flag_for_review(post_id, reason)
        
        mock_moderation_service.flag_for_review.assert_called_once_with(post_id, reason)
        assert result is True
    
    async def test_save_moderation_result(self, post_service, mock_moderation_repository):
        """Test saving moderation result"""
        result = SAMPLE_MODERATION_RESULT
        
        saved_result = await post_service.save_moderation_result(result)
        
        mock_moderation_repository.save_moderation_result.assert_called_once_with(result)
        assert saved_result == SAMPLE_MODERATION_RESULT
    
    async def test_get_moderation_result(self, post_service, mock_moderation_repository):
        """Test retrieving moderation result"""
        post_id = str(uuid4())
        
        result = await post_service.get_moderation_result(post_id)
        
        mock_moderation_repository.get_moderation_result.assert_called_once_with(post_id)
        assert result == SAMPLE_MODERATION_RESULT
    
    async def test_update_moderation_status(self, post_service, mock_moderation_repository):
        """Test updating moderation status"""
        post_id = str(uuid4())
        status = "approved"
        moderator_id = str(uuid4())
        
        result = await post_service.update_moderation_status(post_id, status, moderator_id)
        
        mock_moderation_repository.update_moderation_status.assert_called_once_with(post_id, status, moderator_id)
        assert result == SAMPLE_MODERATION_RESULT
    
    async def test_get_moderation_history(self, post_service, mock_moderation_repository):
        """Test retrieving moderation history"""
        post_id = str(uuid4())
        
        result = await post_service.get_moderation_history(post_id)
        
        mock_moderation_repository.get_moderation_history.assert_called_once_with(post_id)
        assert isinstance(result, list)
        assert len(result) > 0
    
    async def test_get_pending_reviews(self, post_service, mock_moderation_repository):
        """Test retrieving pending reviews"""
        result = await post_service.get_pending_reviews()
        
        mock_moderation_repository.get_pending_reviews.assert_called_once()
        assert isinstance(result, list)
        assert len(result) > 0
    
    async def test_get_moderation_analytics(self, post_service, mock_moderation_repository):
        """Test moderation analytics retrieval"""
        result = await post_service.get_moderation_analytics()
        
        mock_moderation_repository.get_moderation_analytics.assert_called_once()
        assert "total_posts" in result
        assert "approved_count" in result
        assert "rejected_count" in result
        assert "pending_review" in result
    
    async def test_apply_content_filters(self, post_service, mock_filtering_service):
        """Test applying content filters"""
        content = SAMPLE_CONTENT_FOR_MODERATION
        
        result = await post_service.apply_content_filters(content)
        
        mock_filtering_service.apply_filters.assert_called_once_with(content)
        assert "passed_filters" in result
        assert "filter_results" in result
    
    async def test_check_profanity(self, post_service, mock_filtering_service):
        """Test profanity checking"""
        content = "This is a clean business post"
        
        result = await post_service.check_profanity(content)
        
        mock_filtering_service.check_profanity.assert_called_once_with(content)
        assert "contains_profanity" in result
        assert "score" in result
    
    async def test_check_spam_content(self, post_service, mock_filtering_service):
        """Test spam content checking"""
        content = "Legitimate business content"
        
        result = await post_service.check_spam_content(content)
        
        mock_filtering_service.check_spam.assert_called_once_with(content)
        assert "is_spam" in result
        assert "score" in result
    
    async def test_check_inappropriate_content(self, post_service, mock_filtering_service):
        """Test inappropriate content checking"""
        content = "Professional business content"
        
        result = await post_service.check_inappropriate_content(content)
        
        mock_filtering_service.check_inappropriate.assert_called_once_with(content)
        assert "is_inappropriate" in result
        assert "score" in result
    
    async def test_bulk_moderation(self, post_service, mock_moderation_service):
        """Test bulk moderation operations"""
        content_list = [SAMPLE_CONTENT_FOR_MODERATION] * 10
        
        # Mock bulk operations
        mock_moderation_service.bulk_screen_content.return_value = [SAMPLE_MODERATION_RESULT] * 10
        mock_moderation_service.bulk_approve_content.return_value = True
        mock_moderation_service.bulk_reject_content.return_value = True
        
        # Test bulk screening
        result_screen = await post_service.bulk_screen_content(content_list)
        mock_moderation_service.bulk_screen_content.assert_called_once_with(content_list)
        assert len(result_screen) == 10
        
        # Test bulk approval
        post_ids = [str(uuid4()) for _ in range(10)]
        result_approve = await post_service.bulk_approve_content(post_ids)
        mock_moderation_service.bulk_approve_content.assert_called_once_with(post_ids)
        assert result_approve is True
        
        # Test bulk rejection
        result_reject = await post_service.bulk_reject_content(post_ids, "Bulk rejection")
        mock_moderation_service.bulk_reject_content.assert_called_once_with(post_ids, "Bulk rejection")
        assert result_reject is True
    
    async def test_moderation_workflow(self, post_service, mock_moderation_service):
        """Test complete moderation workflow"""
        content = SAMPLE_CONTENT_FOR_MODERATION
        
        # Mock workflow steps
        mock_moderation_service.screen_content.return_value = SAMPLE_MODERATION_RESULT
        mock_moderation_service.assess_quality.return_value = SAMPLE_QUALITY_ASSESSMENT
        mock_moderation_service.approve_content.return_value = True
        
        # Execute workflow
        screening_result = await post_service.screen_content(content)
        quality_result = await post_service.assess_content_quality(content)
        approval_result = await post_service.approve_content(
            content["post_id"], 
            str(uuid4()), 
            "Approved after review"
        )
        
        assert screening_result == SAMPLE_MODERATION_RESULT
        assert quality_result == SAMPLE_QUALITY_ASSESSMENT
        assert approval_result is True
    
    async def test_moderation_escalation(self, post_service, mock_moderation_service):
        """Test moderation escalation process"""
        post_id = str(uuid4())
        escalation_reason = "Complex content requiring senior review"
        
        mock_moderation_service.escalate_moderation.return_value = {
            "escalated": True,
            "escalation_level": "senior_moderator",
            "estimated_review_time": "24 hours"
        }
        
        result = await post_service.escalate_moderation(post_id, escalation_reason)
        
        mock_moderation_service.escalate_moderation.assert_called_once_with(post_id, escalation_reason)
        assert "escalated" in result
        assert "escalation_level" in result
        assert "estimated_review_time" in result
    
    async def test_moderation_appeal_process(self, post_service, mock_moderation_service):
        """Test moderation appeal process"""
        post_id = str(uuid4())
        appeal_reason = "Content was incorrectly flagged"
        user_id = str(uuid4())
        
        mock_moderation_service.submit_appeal.return_value = {
            "appeal_id": str(uuid4()),
            "status": "pending",
            "submitted_at": datetime.now()
        }
        
        result = await post_service.submit_moderation_appeal(post_id, appeal_reason, user_id)
        
        mock_moderation_service.submit_appeal.assert_called_once_with(post_id, appeal_reason, user_id)
        assert "appeal_id" in result
        assert "status" in result
        assert "submitted_at" in result
    
    async def test_moderation_automation_rules(self, post_service, mock_moderation_service):
        """Test moderation automation rules"""
        content = SAMPLE_CONTENT_FOR_MODERATION
        
        mock_moderation_service.apply_automation_rules.return_value = {
            "auto_approved": True,
            "auto_rejected": False,
            "requires_review": False,
            "applied_rules": ["quality_threshold", "spam_check"]
        }
        
        result = await post_service.apply_moderation_automation_rules(content)
        
        mock_moderation_service.apply_automation_rules.assert_called_once_with(content)
        assert "auto_approved" in result
        assert "auto_rejected" in result
        assert "requires_review" in result
        assert "applied_rules" in result
    
    async def test_moderation_performance_metrics(self, post_service, mock_moderation_repository):
        """Test moderation performance metrics"""
        date_range = "last_30_days"
        
        mock_moderation_repository.get_moderation_performance.return_value = {
            "avg_response_time": 2.5,
            "accuracy_rate": 0.95,
            "false_positive_rate": 0.03,
            "false_negative_rate": 0.02,
            "moderator_efficiency": 0.88
        }
        
        result = await post_service.get_moderation_performance(date_range)
        
        mock_moderation_repository.get_moderation_performance.assert_called_once_with(date_range)
        assert "avg_response_time" in result
        assert "accuracy_rate" in result
        assert "false_positive_rate" in result
        assert "false_negative_rate" in result
        assert "moderator_efficiency" in result
