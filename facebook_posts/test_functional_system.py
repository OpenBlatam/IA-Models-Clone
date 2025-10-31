#!/usr/bin/env python3
"""
Functional Facebook Posts System Tests
======================================

Comprehensive test suite following functional programming principles
"""

import asyncio
import pytest
import json
from typing import Dict, Any, List
from datetime import datetime

# Test imports
from api.schemas import (
    PostRequest, PostResponse, FacebookPost, PostMetrics,
    ContentType, AudienceType, OptimizationLevel
)
from core.config import get_settings, validate_environment
from services.async_ai_service import (
    build_ai_request_data, parse_ai_response, validate_ai_content,
    extract_hashtags, extract_emojis, analyze_content_sentiment,
    calculate_engagement_score
)
from core.async_engine import (
    create_post_id, calculate_processing_time, determine_post_status,
    build_optimization_metadata, calculate_quality_score,
    build_post_metrics, create_facebook_post, create_post_response
)


# Pure function tests

def test_build_ai_request_data():
    """Test AI request data building - pure function"""
    data = build_ai_request_data(
        topic="AI and Machine Learning",
        content_type="text",
        audience_type="professionals",
        tone="professional",
        language="en",
        max_length=280,
        custom_instructions="Include technical details"
    )
    
    assert "prompt" in data
    assert "AI and Machine Learning" in data["prompt"]
    assert data["max_tokens"] == 280
    assert data["temperature"] == 0.7
    assert data["content_type"] == "text"


def test_parse_ai_response():
    """Test AI response parsing - pure function"""
    # Valid response
    valid_response = {
        "choices": [{"text": "Generated content here"}],
        "usage": {"total_tokens": 100},
        "model": "gpt-3.5-turbo",
        "created": 1234567890
    }
    
    result = parse_ai_response(valid_response)
    assert result["content"] == "Generated content here"
    assert result["usage"]["total_tokens"] == 100
    assert result["model"] == "gpt-3.5-turbo"
    
    # Invalid response
    invalid_response = {}
    result = parse_ai_response(invalid_response)
    assert result["content"] == ""
    assert "error" in result


def test_validate_ai_content():
    """Test AI content validation - pure function"""
    # Valid content
    assert validate_ai_content("Valid content", 280) == True
    assert validate_ai_content("Short", 280) == True
    
    # Invalid content
    assert validate_ai_content("", 280) == False
    assert validate_ai_content("   ", 280) == False
    assert validate_ai_content("Very long content " * 100, 280) == False


def test_extract_hashtags():
    """Test hashtag extraction - pure function"""
    content = "This is a #test post with #hashtags #ai #ml"
    hashtags = extract_hashtags(content)
    assert "#test" in hashtags
    assert "#hashtags" in hashtags
    assert "#ai" in hashtags
    assert "#ml" in hashtags
    assert len(hashtags) == 4


def test_extract_emojis():
    """Test emoji extraction - pure function"""
    content = "This is a test post with emojis 😀 🚀 💡"
    emojis = extract_emojis(content)
    assert "😀" in emojis
    assert "🚀" in emojis
    assert "💡" in emojis
    assert len(emojis) == 3


def test_analyze_content_sentiment():
    """Test content sentiment analysis - pure function"""
    # Positive content
    positive_content = "This is great! Amazing work! Wonderful results!"
    sentiment = analyze_content_sentiment(positive_content)
    assert sentiment > 0
    
    # Negative content
    negative_content = "This is bad. Terrible work. Awful results."
    sentiment = analyze_content_sentiment(negative_content)
    assert sentiment < 0
    
    # Neutral content
    neutral_content = "This is a normal post without strong sentiment."
    sentiment = analyze_content_sentiment(neutral_content)
    assert sentiment == 0.0


def test_calculate_engagement_score():
    """Test engagement score calculation - pure function"""
    # High engagement content
    content = "Great post with #hashtags and emojis 😀 🚀"
    hashtags = ["#hashtags"]
    emojis = ["😀", "🚀"]
    
    score = calculate_engagement_score(content, hashtags, emojis)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be high engagement
    
    # Low engagement content
    content = "Short post"
    hashtags = []
    emojis = []
    
    score = calculate_engagement_score(content, hashtags, emojis)
    assert 0.0 <= score <= 1.0
    assert score < 0.5  # Should be lower engagement


def test_create_post_id():
    """Test post ID creation - pure function"""
    post_id = create_post_id()
    assert isinstance(post_id, str)
    assert len(post_id) > 0
    
    # Should be unique
    post_id2 = create_post_id()
    assert post_id != post_id2


def test_calculate_processing_time():
    """Test processing time calculation - pure function"""
    start_time = 1000.0
    end_time = 1005.5
    processing_time = calculate_processing_time(start_time, end_time)
    assert processing_time == 5.5


def test_determine_post_status():
    """Test post status determination - pure function"""
    # Draft status
    assert determine_post_status("") == "draft"
    assert determine_post_status("Short") == "draft"
    
    # Generated status
    assert determine_post_status("This is a valid post content") == "generated"
    
    # Optimized status
    assert determine_post_status("This is a valid post content", True) == "optimized"


def test_build_optimization_metadata():
    """Test optimization metadata building - pure function"""
    optimizations = ["ai_generation", "sentiment_analysis", "hashtag_optimization"]
    metadata = build_optimization_metadata(optimizations)
    
    assert metadata["optimizations_applied"] == optimizations
    assert metadata["optimization_count"] == 3
    assert "last_optimized" in metadata


def test_calculate_quality_score():
    """Test quality score calculation - pure function"""
    analysis = {
        "engagement_score": 0.8,
        "readability_score": 0.9,
        "sentiment_score": 0.5
    }
    
    score = calculate_quality_score(analysis)
    assert 0.0 <= score <= 1.0
    assert score > 0.7  # Should be high quality


def test_build_post_metrics():
    """Test post metrics building - pure function"""
    analysis = {
        "engagement_score": 0.8,
        "readability_score": 0.9,
        "sentiment_score": 0.5,
        "viral_potential": 0.7,
        "hashtags": ["#test"],
        "emojis": ["😀"]
    }
    
    metrics = build_post_metrics(analysis, "Test content")
    
    assert isinstance(metrics, PostMetrics)
    assert metrics.engagement_score == 0.8
    assert metrics.readability_score == 0.9
    assert metrics.sentiment_score == 0.5
    assert metrics.viral_potential == 0.7
    assert metrics.estimated_reach == 1000


def test_create_facebook_post():
    """Test Facebook post creation - pure function"""
    request = PostRequest(
        topic="Test topic",
        content_type=ContentType.TEXT,
        audience_type=AudienceType.GENERAL,
        tone="professional",
        language="en",
        max_length=280
    )
    
    content = "Test content"
    analysis = {
        "engagement_score": 0.8,
        "readability_score": 0.9,
        "sentiment_score": 0.5,
        "viral_potential": 0.7,
        "hashtags": ["#test"],
        "emojis": ["😀"]
    }
    
    post = create_facebook_post(content, request, analysis)
    
    assert isinstance(post, FacebookPost)
    assert post.content == content
    assert post.topic == "Test topic"
    assert post.content_type == ContentType.TEXT
    assert post.audience_type == AudienceType.GENERAL
    assert post.hashtags == ["#test"]
    assert post.emojis == ["😀"]
    assert post.status == "generated"


def test_create_post_response():
    """Test post response creation - pure function"""
    # Success response
    post = FacebookPost(
        content="Test content",
        content_type=ContentType.TEXT,
        audience_type=AudienceType.GENERAL,
        topic="Test topic",
        tone="professional",
        language="en"
    )
    
    response = create_post_response(
        success=True,
        post=post,
        processing_time=1.5,
        optimizations_applied=["ai_generation"]
    )
    
    assert isinstance(response, PostResponse)
    assert response.success == True
    assert response.post == post
    assert response.processing_time == 1.5
    assert "ai_generation" in response.optimizations_applied
    
    # Error response
    error_response = create_post_response(
        success=False,
        error="Test error",
        processing_time=0.5
    )
    
    assert error_response.success == False
    assert error_response.error == "Test error"
    assert error_response.post is None


# Schema validation tests

def test_post_request_validation():
    """Test PostRequest validation"""
    # Valid request
    request = PostRequest(
        topic="Test topic",
        content_type=ContentType.TEXT,
        audience_type=AudienceType.GENERAL
    )
    
    assert request.topic == "Test topic"
    assert request.content_type == ContentType.TEXT
    assert request.audience_type == AudienceType.GENERAL
    assert request.tone == "professional"  # Default value
    assert request.language == "en"  # Default value
    assert request.max_length == 280  # Default value
    
    # Invalid request
    with pytest.raises(ValueError):
        PostRequest(
            topic="",  # Empty topic should fail
            content_type=ContentType.TEXT,
            audience_type=AudienceType.GENERAL
        )


def test_post_metrics_validation():
    """Test PostMetrics validation"""
    metrics = PostMetrics(
        engagement_score=0.8,
        readability_score=0.9,
        sentiment_score=0.5,
        viral_potential=0.7,
        quality_score=0.85,
        estimated_reach=1000,
        estimated_impressions=5000,
        estimated_clicks=100,
        estimated_likes=50,
        estimated_shares=10,
        estimated_comments=5
    )
    
    assert metrics.engagement_score == 0.8
    assert metrics.readability_score == 0.9
    assert metrics.sentiment_score == 0.5
    assert metrics.viral_potential == 0.7
    assert metrics.quality_score == 0.85
    assert metrics.estimated_reach == 1000


# Configuration tests

def test_get_settings():
    """Test settings retrieval"""
    settings = get_settings()
    assert settings.api_title == "Ultimate Facebook Posts API"
    assert settings.api_version == "4.0.0"
    assert settings.debug == False  # Default value


def test_validate_environment():
    """Test environment validation"""
    # This test depends on the actual environment
    # In a real test environment, you might mock this
    result = validate_environment()
    assert isinstance(result, bool)


# Integration tests

@pytest.mark.asyncio
async def test_ai_service_integration():
    """Test AI service integration"""
    from services.async_ai_service import AsyncAIService
    
    # Mock AI service for testing
    ai_service = AsyncAIService("test_key", "gpt-3.5-turbo")
    
    # Test health check
    health = await ai_service.health_check()
    assert "status" in health
    
    # Test content analysis
    analysis = await ai_service.analyze_content("Test content with #hashtags 😀")
    assert "content_length" in analysis
    assert "hashtags" in analysis
    assert "emojis" in analysis
    assert "sentiment_score" in analysis
    assert "engagement_score" in analysis


@pytest.mark.asyncio
async def test_engine_integration():
    """Test engine integration"""
    from core.async_engine import AsyncFacebookPostsEngine
    from services.async_ai_service import AsyncAIService
    
    # Create mock services
    ai_service = AsyncAIService("test_key", "gpt-3.5-turbo")
    engine = AsyncFacebookPostsEngine(ai_service)
    
    # Test health check
    health = await engine.health_check()
    assert "status" in health
    assert "ai_service" in health
    assert "statistics" in health
    
    # Test statistics
    stats = await engine.get_statistics()
    assert "total_requests" in stats
    assert "successful_requests" in stats
    assert "failed_requests" in stats


# Performance tests

def test_performance_pure_functions():
    """Test performance of pure functions"""
    import time
    
    # Test hashtag extraction performance
    content = "This is a test post with #hashtags #ai #ml #python #fastapi " * 100
    
    start_time = time.time()
    for _ in range(1000):
        extract_hashtags(content)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < 1.0  # Should be fast
    
    # Test emoji extraction performance
    content = "This is a test post with emojis 😀 🚀 💡 🎉 🔥 " * 100
    
    start_time = time.time()
    for _ in range(1000):
        extract_emojis(content)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < 1.0  # Should be fast


def test_memory_usage():
    """Test memory usage of pure functions"""
    import sys
    
    # Test that functions don't create memory leaks
    initial_objects = len(gc.get_objects())
    
    # Run many iterations
    for _ in range(1000):
        create_post_id()
        calculate_processing_time(1000.0, 1005.0)
        determine_post_status("Test content")
    
    final_objects = len(gc.get_objects())
    
    # Should not create excessive objects
    assert final_objects - initial_objects < 1000


# Error handling tests

def test_error_handling_pure_functions():
    """Test error handling in pure functions"""
    # Test with invalid inputs
    assert validate_ai_content("", 280) == False
    assert validate_ai_content(None, 280) == False
    assert validate_ai_content("Valid", -1) == False
    
    # Test with edge cases
    assert extract_hashtags("") == []
    assert extract_emojis("") == []
    assert analyze_content_sentiment("") == 0.0


def test_edge_cases():
    """Test edge cases"""
    # Very long content
    long_content = "A" * 10000
    assert validate_ai_content(long_content, 10000) == True
    assert validate_ai_content(long_content, 1000) == False
    
    # Special characters
    special_content = "Test with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
    hashtags = extract_hashtags(special_content)
    emojis = extract_emojis(special_content)
    assert len(hashtags) == 0  # No hashtags in this content
    assert len(emojis) == 0  # No emojis in this content


# Main test runner

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Facebook Posts System Tests")
    print("=" * 50)
    
    # Run pure function tests
    test_functions = [
        test_build_ai_request_data,
        test_parse_ai_response,
        test_validate_ai_content,
        test_extract_hashtags,
        test_extract_emojis,
        test_analyze_content_sentiment,
        test_calculate_engagement_score,
        test_create_post_id,
        test_calculate_processing_time,
        test_determine_post_status,
        test_build_optimization_metadata,
        test_calculate_quality_score,
        test_build_post_metrics,
        test_create_facebook_post,
        test_create_post_response,
        test_post_request_validation,
        test_post_metrics_validation,
        test_get_settings,
        test_validate_environment,
        test_performance_pure_functions,
        test_memory_usage,
        test_error_handling_pure_functions,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    import gc
    run_all_tests()

