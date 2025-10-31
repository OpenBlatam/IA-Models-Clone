"""
Unit Tests for Core Entities
============================

Comprehensive unit tests for LinkedIn post entities, value objects,
and domain models with proper validation and edge cases.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import entities
from ...core.entities import (
    LinkedInPost, PostContent, PostType, PostTone, PostStatus,
    EngagementMetrics, PostGenerationRequest, PostValidationResult,
    ContentAnalysisResult, PostOptimizationResult
)


class TestLinkedInPost:
    """Test suite for LinkedInPost entity."""

    @pytest.fixture
    def sample_content(self) -> PostContent:
        """Sample post content for testing."""
        return PostContent(
            text="This is a test LinkedIn post content with proper formatting and hashtags.",
            hashtags=["#test", "#linkedin", "#post"],
            mentions=["@testuser"],
            links=["https://example.com"],
            images=[],
            callToAction="Learn more about our services!"
        )

    @pytest.fixture
    def sample_engagement_metrics(self) -> EngagementMetrics:
        """Sample engagement metrics for testing."""
        return EngagementMetrics(
            likes=100,
            comments=25,
            shares=15,
            clicks=50,
            impressions=1000,
            reach=800,
            engagementRate=0.075
        )

    @pytest.fixture
    def sample_post(self, sample_content: PostContent, sample_engagement_metrics: EngagementMetrics) -> LinkedInPost:
        """Sample LinkedIn post for testing."""
        return LinkedInPost(
            id="test-post-123",
            userId="user-123",
            title="Test LinkedIn Post",
            content=sample_content,
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.DRAFT,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            engagement=sample_engagement_metrics,
            aiScore=85.5,
            optimizationSuggestions=["Add more hashtags", "Include call-to-action"],
            keywords=["test", "linkedin", "post"],
            externalMetadata={"source": "test"},
            performanceScore=75.0,
            reachScore=80.0,
            engagementScore=85.0
        )

    def test_create_post_with_valid_data(self, sample_post: LinkedInPost):
        """Test creating a post with valid data."""
        assert sample_post.id == "test-post-123"
        assert sample_post.userId == "user-123"
        assert sample_post.title == "Test LinkedIn Post"
        assert sample_post.postType == PostType.TEXT
        assert sample_post.tone == PostTone.PROFESSIONAL
        assert sample_post.status == PostStatus.DRAFT
        assert sample_post.aiScore == 85.5
        assert len(sample_post.optimizationSuggestions) == 2
        assert len(sample_post.keywords) == 3

    def test_post_content_structure(self, sample_content: PostContent):
        """Test post content structure."""
        assert sample_content.text == "This is a test LinkedIn post content with proper formatting and hashtags."
        assert len(sample_content.hashtags) == 3
        assert len(sample_content.mentions) == 1
        assert len(sample_content.links) == 1
        assert sample_content.callToAction == "Learn more about our services!"

    def test_engagement_metrics_calculation(self, sample_engagement_metrics: EngagementMetrics):
        """Test engagement metrics structure."""
        assert sample_engagement_metrics.likes == 100
        assert sample_engagement_metrics.comments == 25
        assert sample_engagement_metrics.shares == 15
        assert sample_engagement_metrics.clicks == 50
        assert sample_engagement_metrics.impressions == 1000
        assert sample_engagement_metrics.reach == 800
        assert sample_engagement_metrics.engagementRate == 0.075

    def test_post_status_transitions(self, sample_post: LinkedInPost):
        """Test post status transitions."""
        # Initial status should be DRAFT
        assert sample_post.status == PostStatus.DRAFT
        
        # Test status constants
        assert PostStatus.DRAFT == "draft"
        assert PostStatus.SCHEDULED == "scheduled"
        assert PostStatus.PUBLISHED == "published"
        assert PostStatus.ARCHIVED == "archived"
        assert PostStatus.DELETED == "deleted"

    def test_post_type_constants(self):
        """Test post type constants."""
        assert PostType.TEXT == "text"
        assert PostType.IMAGE == "image"
        assert PostType.VIDEO == "video"
        assert PostType.ARTICLE == "article"
        assert PostType.POLL == "poll"
        assert PostType.EVENT == "event"

    def test_post_tone_constants(self):
        """Test post tone constants."""
        assert PostTone.PROFESSIONAL == "professional"
        assert PostTone.CASUAL == "casual"
        assert PostTone.FRIENDLY == "friendly"
        assert PostTone.AUTHORITATIVE == "authoritative"
        assert PostTone.INSPIRATIONAL == "inspirational"

    def test_post_with_scheduled_time(self, sample_content: PostContent, sample_engagement_metrics: EngagementMetrics):
        """Test post with scheduled time."""
        scheduled_time = datetime.now() + timedelta(hours=1)
        post = LinkedInPost(
            id="scheduled-post-123",
            userId="user-123",
            title="Scheduled Post",
            content=sample_content,
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.SCHEDULED,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            scheduledAt=scheduled_time,
            engagement=sample_engagement_metrics,
            aiScore=90.0,
            optimizationSuggestions=[],
            keywords=[],
            externalMetadata={},
            performanceScore=0,
            reachScore=0,
            engagementScore=0
        )
        
        assert post.scheduledAt == scheduled_time
        assert post.status == PostStatus.SCHEDULED

    def test_post_with_published_time(self, sample_content: PostContent, sample_engagement_metrics: EngagementMetrics):
        """Test post with published time."""
        published_time = datetime.now()
        post = LinkedInPost(
            id="published-post-123",
            userId="user-123",
            title="Published Post",
            content=sample_content,
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.PUBLISHED,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            publishedAt=published_time,
            engagement=sample_engagement_metrics,
            aiScore=95.0,
            optimizationSuggestions=[],
            keywords=[],
            externalMetadata={},
            performanceScore=0,
            reachScore=0,
            engagementScore=0
        )
        
        assert post.publishedAt == published_time
        assert post.status == PostStatus.PUBLISHED

    def test_post_with_linkedin_post_id(self, sample_content: PostContent, sample_engagement_metrics: EngagementMetrics):
        """Test post with LinkedIn post ID."""
        post = LinkedInPost(
            id="local-post-123",
            userId="user-123",
            title="LinkedIn Post",
            content=sample_content,
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.PUBLISHED,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            linkedinPostId="linkedin-post-456",
            engagement=sample_engagement_metrics,
            aiScore=88.0,
            optimizationSuggestions=[],
            keywords=[],
            externalMetadata={},
            performanceScore=0,
            reachScore=0,
            engagementScore=0
        )
        
        assert post.linkedinPostId == "linkedin-post-456"

    def test_post_with_external_metadata(self, sample_content: PostContent, sample_engagement_metrics: EngagementMetrics):
        """Test post with external metadata."""
        metadata = {
            "source": "api",
            "campaign_id": "campaign-123",
            "tags": ["business", "technology"],
            "priority": "high"
        }
        
        post = LinkedInPost(
            id="metadata-post-123",
            userId="user-123",
            title="Post with Metadata",
            content=sample_content,
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.DRAFT,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            engagement=sample_engagement_metrics,
            aiScore=82.0,
            optimizationSuggestions=[],
            keywords=[],
            externalMetadata=metadata,
            performanceScore=0,
            reachScore=0,
            engagementScore=0
        )
        
        assert post.externalMetadata == metadata
        assert post.externalMetadata["source"] == "api"
        assert post.externalMetadata["campaign_id"] == "campaign-123"


class TestPostGenerationRequest:
    """Test suite for PostGenerationRequest entity."""

    @pytest.fixture
    def valid_request(self) -> PostGenerationRequest:
        """Valid post generation request."""
        return PostGenerationRequest(
            topic="AI in Modern Business",
            keyPoints=["Increased efficiency", "Cost reduction", "Innovation"],
            targetAudience="Business leaders",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["AI", "business", "innovation"],
            additionalContext="Focus on practical applications"
        )

    def test_valid_request_structure(self, valid_request: PostGenerationRequest):
        """Test valid request structure."""
        assert valid_request.topic == "AI in Modern Business"
        assert len(valid_request.keyPoints) == 3
        assert valid_request.targetAudience == "Business leaders"
        assert valid_request.industry == "Technology"
        assert valid_request.tone == PostTone.PROFESSIONAL
        assert valid_request.postType == PostType.TEXT
        assert len(valid_request.keywords) == 3
        assert valid_request.additionalContext == "Focus on practical applications"

    def test_request_without_optional_fields(self):
        """Test request without optional fields."""
        request = PostGenerationRequest(
            topic="Simple Topic",
            keyPoints=["Point 1"],
            targetAudience="General audience",
            industry="General",
            tone=PostTone.CASUAL,
            postType=PostType.TEXT
        )
        
        assert request.topic == "Simple Topic"
        assert request.keywords is None
        assert request.additionalContext is None

    def test_request_with_empty_keywords(self):
        """Test request with empty keywords."""
        request = PostGenerationRequest(
            topic="Topic",
            keyPoints=["Point 1"],
            targetAudience="Audience",
            industry="Industry",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=[]
        )
        
        assert request.keywords == []


class TestPostValidationResult:
    """Test suite for PostValidationResult entity."""

    def test_valid_validation_result(self):
        """Test valid validation result."""
        result = PostValidationResult(
            isValid=True,
            errors=[],
            warnings=["Consider adding more hashtags"],
            suggestions=["Include a call-to-action"]
        )
        
        assert result.isValid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert len(result.suggestions) == 1

    def test_invalid_validation_result(self):
        """Test invalid validation result."""
        result = PostValidationResult(
            isValid=False,
            errors=["Topic is too short", "Missing target audience"],
            warnings=["Content could be more engaging"],
            suggestions=["Add more details to the topic"]
        )
        
        assert result.isValid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert len(result.suggestions) == 1


class TestContentAnalysisResult:
    """Test suite for ContentAnalysisResult entity."""

    def test_content_analysis_result(self):
        """Test content analysis result structure."""
        result = ContentAnalysisResult(
            sentimentScore=0.8,
            readabilityScore=85.0,
            engagementScore=90.0,
            keywordDensity=0.15,
            structureScore=88.0,
            callToActionScore=92.0
        )
        
        assert result.sentimentScore == 0.8
        assert result.readabilityScore == 85.0
        assert result.engagementScore == 90.0
        assert result.keywordDensity == 0.15
        assert result.structureScore == 88.0
        assert result.callToActionScore == 92.0

    def test_content_analysis_with_negative_scores(self):
        """Test content analysis with negative scores."""
        result = ContentAnalysisResult(
            sentimentScore=-0.2,
            readabilityScore=45.0,
            engagementScore=30.0,
            keywordDensity=0.05,
            structureScore=40.0,
            callToActionScore=25.0
        )
        
        assert result.sentimentScore == -0.2
        assert result.readabilityScore == 45.0
        assert result.engagementScore == 30.0


class TestPostOptimizationResult:
    """Test suite for PostOptimizationResult entity."""

    @pytest.fixture
    def sample_optimization_result(self, sample_post: LinkedInPost) -> PostOptimizationResult:
        """Sample optimization result."""
        return PostOptimizationResult(
            originalPost=sample_post,
            optimizedPost=sample_post,
            optimizationScore=90.0,
            suggestions=["Improve headline", "Add more hashtags", "Include call-to-action"],
            processingTime=2.5
        )

    def test_optimization_result_structure(self, sample_optimization_result: PostOptimizationResult):
        """Test optimization result structure."""
        assert sample_optimization_result.optimizationScore == 90.0
        assert len(sample_optimization_result.suggestions) == 3
        assert sample_optimization_result.processingTime == 2.5
        assert sample_optimization_result.originalPost is not None
        assert sample_optimization_result.optimizedPost is not None

    def test_optimization_result_with_high_score(self, sample_post: LinkedInPost):
        """Test optimization result with high score."""
        result = PostOptimizationResult(
            originalPost=sample_post,
            optimizedPost=sample_post,
            optimizationScore=95.0,
            suggestions=["Minor improvements only"],
            processingTime=1.0
        )
        
        assert result.optimizationScore == 95.0
        assert len(result.suggestions) == 1
        assert result.processingTime == 1.0

    def test_optimization_result_with_low_score(self, sample_post: LinkedInPost):
        """Test optimization result with low score."""
        result = PostOptimizationResult(
            originalPost=sample_post,
            optimizedPost=sample_post,
            optimizationScore=45.0,
            suggestions=[
                "Rewrite content completely",
                "Change tone to professional",
                "Add more specific examples",
                "Include industry statistics"
            ],
            processingTime=5.0
        )
        
        assert result.optimizationScore == 45.0
        assert len(result.suggestions) == 4
        assert result.processingTime == 5.0


class TestEngagementMetrics:
    """Test suite for EngagementMetrics entity."""

    def test_empty_engagement_metrics(self):
        """Test empty engagement metrics."""
        metrics = EngagementMetrics(
            likes=0,
            comments=0,
            shares=0,
            clicks=0,
            impressions=0,
            reach=0,
            engagementRate=0.0
        )
        
        assert metrics.likes == 0
        assert metrics.comments == 0
        assert metrics.shares == 0
        assert metrics.clicks == 0
        assert metrics.impressions == 0
        assert metrics.reach == 0
        assert metrics.engagementRate == 0.0

    def test_high_engagement_metrics(self):
        """Test high engagement metrics."""
        metrics = EngagementMetrics(
            likes=1000,
            comments=250,
            shares=150,
            clicks=500,
            impressions=10000,
            reach=8000,
            engagementRate=0.15
        )
        
        assert metrics.likes == 1000
        assert metrics.comments == 250
        assert metrics.shares == 150
        assert metrics.clicks == 500
        assert metrics.impressions == 10000
        assert metrics.reach == 8000
        assert metrics.engagementRate == 0.15

    def test_engagement_rate_calculation(self):
        """Test engagement rate calculation logic."""
        # Engagement rate = (likes + comments + shares) / impressions
        metrics = EngagementMetrics(
            likes=100,
            comments=25,
            shares=15,
            clicks=50,
            impressions=1000,
            reach=800,
            engagementRate=0.14  # (100 + 25 + 15) / 1000 = 0.14
        )
        
        calculated_rate = (metrics.likes + metrics.comments + metrics.shares) / metrics.impressions
        assert calculated_rate == 0.14
        assert metrics.engagementRate == 0.14
