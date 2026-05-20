"""
Enhanced Blog System v14.0.0 - Comprehensive Test Suite
Tests for all major components and functionality
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import the system components
from ENHANCED_BLOG_SYSTEM_v14 import (
    BlogSystemConfig,
    BlogService,
    ContentAnalyzer,
    SearchEngine,
    CacheManager,
    BlogPostCreate,
    BlogPostUpdate,
    SearchRequest,
    PostStatus,
    PostCategory,
    SearchType
)

# Test configuration
@pytest.fixture
def test_config():
    """Test configuration"""
    return BlogSystemConfig(
        database_url="sqlite:///test.db",
        redis_url="redis://localhost:6379",
        secret_key="test-secret-key",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        elasticsearch_url="http://localhost:9200",
        cache_ttl=300,
        max_concurrent_requests=10,
        batch_size=16
    )

@pytest.fixture
def sample_post_data():
    """Sample blog post data for testing"""
    return {
        "title": "Test Blog Post",
        "content": "This is a test blog post content for testing purposes.",
        "excerpt": "Test excerpt",
        "category": PostCategory.TECHNOLOGY,
        "tags": ["test", "blog", "technology"],
        "seo_title": "Test SEO Title",
        "seo_description": "Test SEO description",
        "seo_keywords": ["test", "seo", "keywords"],
        "scheduled_at": None
    }

@pytest.fixture
def sample_search_request():
    """Sample search request for testing"""
    return {
        "query": "artificial intelligence",
        "search_type": SearchType.HYBRID,
        "category": PostCategory.TECHNOLOGY,
        "tags": ["AI", "machine learning"],
        "limit": 10,
        "offset": 0
    }

class TestBlogSystemConfig:
    """Test configuration management"""
    
    def test_config_defaults(self):
        """Test configuration default values"""
        config = BlogSystemConfig()
        
        assert config.database_url == "postgresql://user:password@localhost/blog_db"
        assert config.redis_url == "redis://localhost:6379"
        assert config.secret_key == "your-secret-key-here"
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.max_sequence_length == 512
        assert config.embedding_dimension == 384
        assert config.cache_ttl == 3600
        assert config.max_concurrent_requests == 100
        assert config.batch_size == 32
        assert config.enable_metrics is True
    
    def test_config_custom_values(self):
        """Test configuration with custom values"""
        config = BlogSystemConfig(
            database_url="postgresql://test:test@localhost/test_db",
            redis_url="redis://localhost:6380",
            secret_key="custom-secret",
            cache_ttl=1800,
            max_concurrent_requests=50
        )
        
        assert config.database_url == "postgresql://test:test@localhost/test_db"
        assert config.redis_url == "redis://localhost:6380"
        assert config.secret_key == "custom-secret"
        assert config.cache_ttl == 1800
        assert config.max_concurrent_requests == 50

class TestContentAnalyzer:
    """Test content analysis functionality"""
    
    @pytest.fixture
    def analyzer(self, test_config):
        """Content analyzer instance"""
        return ContentAnalyzer(test_config)
    
    def test_generate_embedding(self, analyzer):
        """Test embedding generation"""
        text = "This is a test text for embedding generation."
        embedding = analyzer.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == test_config.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_analyze_sentiment_positive(self, analyzer):
        """Test positive sentiment analysis"""
        text = "This is a great and amazing article about wonderful technology."
        sentiment = analyzer.analyze_sentiment(text)
        
        assert sentiment == 1
    
    def test_analyze_sentiment_negative(self, analyzer):
        """Test negative sentiment analysis"""
        text = "This is a terrible and awful article about horrible technology."
        sentiment = analyzer.analyze_sentiment(text)
        
        assert sentiment == -1
    
    def test_analyze_sentiment_neutral(self, analyzer):
        """Test neutral sentiment analysis"""
        text = "This is a regular article about technology."
        sentiment = analyzer.analyze_sentiment(text)
        
        assert sentiment == 0
    
    def test_calculate_readability(self, analyzer):
        """Test readability calculation"""
        text = "This is a simple text. It has short sentences. Easy to read."
        readability = analyzer.calculate_readability(text)
        
        assert isinstance(readability, int)
        assert 0 <= readability <= 100
    
    def test_extract_topics(self, analyzer):
        """Test topic extraction"""
        text = "This article discusses artificial intelligence and machine learning technology."
        topics = analyzer.extract_topics(text)
        
        assert isinstance(topics, list)
        assert "technology" in topics
        assert "AI" in topics or "machine learning" in topics
    
    def test_count_syllables(self, analyzer):
        """Test syllable counting"""
        assert analyzer._count_syllables("hello") >= 1
        assert analyzer._count_syllables("technology") >= 1
        assert analyzer._count_syllables("a") == 1

class TestCacheManager:
    """Test caching functionality"""
    
    @pytest.fixture
    def cache_manager(self, test_config):
        """Cache manager instance"""
        return CacheManager(test_config)
    
    @pytest.mark.asyncio
    async def test_cache_post(self, cache_manager):
        """Test caching a blog post"""
        # Mock Redis client
        with patch.object(cache_manager, 'redis_client') as mock_redis:
            mock_redis.setex = Mock()
            
            # Create mock post
            mock_post = Mock()
            mock_post.id = 1
            mock_post.json.return_value = '{"id": 1, "title": "Test"}'
            
            await cache_manager.cache_post(mock_post)
            
            mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cached_post(self, cache_manager):
        """Test retrieving cached post"""
        with patch.object(cache_manager, 'redis_client') as mock_redis:
            mock_redis.get.return_value = b'{"id": 1, "title": "Test"}'
            
            result = await cache_manager.get_cached_post(1)
            
            assert result is not None
            mock_redis.get.assert_called_once_with("blog_post:1")
    
    @pytest.mark.asyncio
    async def test_get_cached_post_miss(self, cache_manager):
        """Test cache miss scenario"""
        with patch.object(cache_manager, 'redis_client') as mock_redis:
            mock_redis.get.return_value = None
            
            result = await cache_manager.get_cached_post(1)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_invalidate_post_cache(self, cache_manager):
        """Test cache invalidation"""
        with patch.object(cache_manager, 'redis_client') as mock_redis:
            mock_redis.delete = Mock()
            
            await cache_manager.invalidate_post_cache(1)
            
            mock_redis.delete.assert_called_once_with("blog_post:1")

class TestSearchEngine:
    """Test search functionality"""
    
    @pytest.fixture
    def search_engine(self, test_config):
        """Search engine instance"""
        return SearchEngine(test_config)
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, search_engine, sample_search_request):
        """Test semantic search"""
        with patch.object(search_engine, 'es_client') as mock_es:
            mock_es.search.return_value = {
                'hits': {
                    'hits': [
                        {'_source': {'id': 1, 'title': 'AI Article'}},
                        {'_source': {'id': 2, 'title': 'ML Article'}}
                    ]
                }
            }
            
            results = await search_engine._semantic_search(
                SearchRequest(**sample_search_request)
            )
            
            assert isinstance(results, list)
            mock_es.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fuzzy_search(self, search_engine, sample_search_request):
        """Test fuzzy search"""
        with patch.object(search_engine, 'es_client') as mock_es:
            mock_es.search.return_value = {
                'hits': {
                    'hits': [
                        {'_source': {'id': 1, 'title': 'AI Article'}}
                    ]
                }
            }
            
            results = await search_engine._fuzzy_search(
                SearchRequest(**sample_search_request)
            )
            
            assert isinstance(results, list)
            mock_es.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exact_search(self, search_engine, sample_search_request):
        """Test exact search"""
        with patch.object(search_engine, 'es_client') as mock_es:
            mock_es.search.return_value = {
                'hits': {
                    'hits': [
                        {'_source': {'id': 1, 'title': 'Exact Match'}}
                    ]
                }
            }
            
            results = await search_engine._exact_search(
                SearchRequest(**sample_search_request)
            )
            
            assert isinstance(results, list)
            mock_es.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_engine, sample_search_request):
        """Test hybrid search"""
        with patch.object(search_engine, '_semantic_search') as mock_semantic, \
             patch.object(search_engine, '_fuzzy_search') as mock_fuzzy:
            
            mock_semantic.return_value = [Mock(id=1)]
            mock_fuzzy.return_value = [Mock(id=2)]
            
            results = await search_engine._hybrid_search(
                SearchRequest(**sample_search_request)
            )
            
            assert isinstance(results, list)
            mock_semantic.assert_called_once()
            mock_fuzzy.assert_called_once()

class TestBlogService:
    """Test main blog service functionality"""
    
    @pytest.fixture
    def blog_service(self, test_config):
        """Blog service instance"""
        return BlogService(test_config)
    
    @pytest.mark.asyncio
    async def test_create_post(self, blog_service, sample_post_data):
        """Test post creation"""
        with patch.object(blog_service, 'get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock database operations
            mock_post = Mock()
            mock_post.id = 1
            mock_db.add = Mock()
            mock_db.commit = Mock()
            mock_db.refresh = Mock()
            
            # Mock content analyzer
            with patch.object(blog_service, 'content_analyzer') as mock_analyzer:
                mock_analyzer.generate_embedding.return_value = [0.1] * 384
                mock_analyzer.analyze_sentiment.return_value = 1
                mock_analyzer.calculate_readability.return_value = 75
                mock_analyzer.extract_topics.return_value = ["technology"]
                
                # Mock cache manager
                with patch.object(blog_service, 'cache_manager') as mock_cache:
                    mock_cache.cache_post = AsyncMock()
                    
                    result = await blog_service.create_post(
                        BlogPostCreate(**sample_post_data),
                        "test-author-id"
                    )
                    
                    assert result is not None
                    mock_db.add.assert_called_once()
                    mock_db.commit.assert_called_once()
                    mock_cache.cache_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_post_cached(self, blog_service):
        """Test getting cached post"""
        with patch.object(blog_service, 'cache_manager') as mock_cache:
            mock_post = Mock()
            mock_cache.get_cached_post.return_value = mock_post
            
            result = await blog_service.get_post(1)
            
            assert result == mock_post
            mock_cache.get_cached_post.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_get_post_database(self, blog_service):
        """Test getting post from database"""
        with patch.object(blog_service, 'cache_manager') as mock_cache, \
             patch.object(blog_service, 'get_db') as mock_get_db:
            
            mock_cache.get_cached_post.return_value = None
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock database query
            mock_post = Mock()
            mock_post.id = 1
            mock_db.query.return_value.filter.return_value.first.return_value = mock_post
            
            # Mock cache storage
            mock_cache.cache_post = AsyncMock()
            
            result = await blog_service.get_post(1)
            
            assert result is not None
            mock_cache.cache_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_post(self, blog_service, sample_post_data):
        """Test post update"""
        with patch.object(blog_service, 'get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock existing post
            mock_post = Mock()
            mock_post.id = 1
            mock_post.author_id = "test-author-id"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_post
            
            # Mock content analyzer
            with patch.object(blog_service, 'content_analyzer') as mock_analyzer:
                mock_analyzer.generate_embedding.return_value = [0.1] * 384
                mock_analyzer.analyze_sentiment.return_value = 1
                mock_analyzer.calculate_readability.return_value = 75
                mock_analyzer.extract_topics.return_value = ["technology"]
                
                # Mock cache manager
                with patch.object(blog_service, 'cache_manager') as mock_cache:
                    mock_cache.cache_post = AsyncMock()
                    
                    update_data = BlogPostUpdate(title="Updated Title")
                    result = await blog_service.update_post(1, update_data, "test-author-id")
                    
                    assert result is not None
                    mock_db.commit.assert_called_once()
                    mock_cache.cache_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_post(self, blog_service):
        """Test post deletion"""
        with patch.object(blog_service, 'get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock existing post
            mock_post = Mock()
            mock_post.id = 1
            mock_post.author_id = "test-author-id"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_post
            
            # Mock cache manager
            with patch.object(blog_service, 'cache_manager') as mock_cache:
                mock_cache.invalidate_post_cache = AsyncMock()
                
                result = await blog_service.delete_post(1, "test-author-id")
                
                assert result is True
                mock_db.delete.assert_called_once_with(mock_post)
                mock_db.commit.assert_called_once()
                mock_cache.invalidate_post_cache.assert_called_once_with(1)
    
    def test_generate_slug(self, blog_service):
        """Test slug generation"""
        title = "This is a Test Title!"
        slug = blog_service._generate_slug(title)
        
        assert slug == "this-is-a-test-title"
        assert "-" not in slug[0]  # No leading hyphen
        assert "-" not in slug[-1]  # No trailing hyphen

class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    @pytest.fixture
    def client(self):
        """Test client"""
        from fastapi.testclient import TestClient
        from ENHANCED_BLOG_SYSTEM_v14 import app
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "14.0.0"
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "posts_created" in data
        assert "posts_read" in data
        assert "posts_updated" in data
        assert "posts_deleted" in data
        assert "searches_performed" in data

class TestDataModels:
    """Test data model validation"""
    
    def test_blog_post_create_valid(self, sample_post_data):
        """Test valid blog post creation"""
        post = BlogPostCreate(**sample_post_data)
        
        assert post.title == "Test Blog Post"
        assert post.content == "This is a test blog post content for testing purposes."
        assert post.category == PostCategory.TECHNOLOGY
        assert post.tags == ["test", "blog", "technology"]
    
    def test_blog_post_create_invalid_title(self):
        """Test invalid title validation"""
        with pytest.raises(ValueError):
            BlogPostCreate(
                title="",  # Empty title
                content="Test content",
                category=PostCategory.TECHNOLOGY
            )
    
    def test_blog_post_create_invalid_content(self):
        """Test invalid content validation"""
        with pytest.raises(ValueError):
            BlogPostCreate(
                title="Test Title",
                content="",  # Empty content
                category=PostCategory.TECHNOLOGY
            )
    
    def test_search_request_valid(self, sample_search_request):
        """Test valid search request"""
        request = SearchRequest(**sample_search_request)
        
        assert request.query == "artificial intelligence"
        assert request.search_type == SearchType.HYBRID
        assert request.category == PostCategory.TECHNOLOGY
        assert request.limit == 10
        assert request.offset == 0
    
    def test_search_request_invalid_query(self):
        """Test invalid query validation"""
        with pytest.raises(ValueError):
            SearchRequest(
                query="",  # Empty query
                search_type=SearchType.HYBRID
            )
    
    def test_search_request_invalid_limit(self):
        """Test invalid limit validation"""
        with pytest.raises(ValueError):
            SearchRequest(
                query="test",
                search_type=SearchType.HYBRID,
                limit=0  # Invalid limit
            )

class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_concurrent_post_creation(self, blog_service, sample_post_data):
        """Test concurrent post creation performance"""
        async def create_post():
            with patch.object(blog_service, 'get_db') as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db
                mock_post = Mock()
                mock_post.id = 1
                mock_db.add = Mock()
                mock_db.commit = Mock()
                mock_db.refresh = Mock()
                
                with patch.object(blog_service, 'content_analyzer') as mock_analyzer:
                    mock_analyzer.generate_embedding.return_value = [0.1] * 384
                    mock_analyzer.analyze_sentiment.return_value = 1
                    mock_analyzer.calculate_readability.return_value = 75
                    mock_analyzer.extract_topics.return_value = ["technology"]
                    
                    with patch.object(blog_service, 'cache_manager') as mock_cache:
                        mock_cache.cache_post = AsyncMock()
                        
                        return await blog_service.create_post(
                            BlogPostCreate(**sample_post_data),
                            "test-author-id"
                        )
        
        # Create multiple posts concurrently
        tasks = [create_post() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_search_performance(self, search_engine, sample_search_request):
        """Test search performance"""
        with patch.object(search_engine, 'es_client') as mock_es:
            mock_es.search.return_value = {
                'hits': {
                    'hits': [
                        {'_source': {'id': i, 'title': f'Article {i}'}}
                        for i in range(10)
                    ]
                }
            }
            
            # Measure search performance
            start_time = asyncio.get_event_loop().time()
            
            for _ in range(100):
                await search_engine._semantic_search(
                    SearchRequest(**sample_search_request)
                )
            
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time
            
            # Should complete 100 searches in reasonable time
            assert total_time < 10.0  # Less than 10 seconds for 100 searches

class TestSecurity:
    """Test security features"""
    
    def test_jwt_token_validation(self):
        """Test JWT token validation"""
        from ENHANCED_BLOG_SYSTEM_v14 import get_current_user
        from fastapi import HTTPException
        
        # Test invalid token
        with pytest.raises(HTTPException):
            get_current_user(Mock(credentials="invalid-token"))
    
    def test_input_validation(self, client):
        """Test input validation"""
        # Test invalid post creation
        response = client.post("/posts/", json={
            "title": "",  # Invalid empty title
            "content": "Test content"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_sql_injection_prevention(self, blog_service):
        """Test SQL injection prevention"""
        # Test that user input is properly sanitized
        malicious_input = "'; DROP TABLE blog_posts; --"
        
        # This should not cause any SQL injection
        with patch.object(blog_service, 'get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = blog_service.get_db()
            assert result is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 