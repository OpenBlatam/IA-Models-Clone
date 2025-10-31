"""
Content Metadata Management Tests
================================

Comprehensive tests for content metadata management features including:
- Metadata extraction and parsing
- Content tagging and categorization
- Search indexing and optimization
- Metadata analytics and insights
- Metadata validation and quality
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from typing import List, Dict, Any, Optional

# Test data
SAMPLE_METADATA_CONFIG = {
    "extraction_rules": {
        "keywords": True,
        "entities": True,
        "sentiment": True,
        "topics": True,
        "language": True
    },
    "tagging_rules": {
        "auto_tagging": True,
        "suggested_tags": True,
        "tag_validation": True
    },
    "indexing_config": {
        "search_engine": "elasticsearch",
        "index_name": "linkedin_posts",
        "refresh_interval": "1s"
    }
}

SAMPLE_CONTENT_METADATA = {
    "post_id": str(uuid4()),
    "title": "AI in Modern Business",
    "content": "Artificial Intelligence is transforming how businesses operate...",
    "keywords": ["AI", "business", "transformation", "technology"],
    "entities": ["Artificial Intelligence", "Business", "Technology"],
    "sentiment": "positive",
    "topics": ["technology", "business", "innovation"],
    "language": "en",
    "readability_score": 75.5,
    "word_count": 150,
    "hashtags": ["#AI", "#Business", "#Innovation"],
    "mentions": ["@techcompany", "@ai_expert"],
    "links": ["https://example.com/article"],
    "media_type": "text",
    "created_at": datetime.now(),
    "updated_at": datetime.now()
}

SAMPLE_TAG_DATA = {
    "tag_id": str(uuid4()),
    "name": "Artificial Intelligence",
    "category": "technology",
    "description": "Posts related to AI and machine learning",
    "usage_count": 1250,
    "trending_score": 8.5,
    "created_at": datetime.now(),
    "updated_at": datetime.now()
}

SAMPLE_SEARCH_INDEX = {
    "index_name": "linkedin_posts",
    "document_count": 15000,
    "size_bytes": 2048576,
    "last_updated": datetime.now(),
    "search_analytics": {
        "total_searches": 50000,
        "avg_response_time": 0.15,
        "top_queries": ["AI", "business", "technology"]
    }
}

class TestContentMetadataManagement:
    """Test content metadata management features"""
    
    @pytest.fixture
    def mock_metadata_service(self):
        """Mock metadata service"""
        service = AsyncMock()
        service.extract_metadata.return_value = SAMPLE_CONTENT_METADATA
        service.parse_content.return_value = {
            "keywords": ["AI", "business"],
            "entities": ["Artificial Intelligence"],
            "sentiment": "positive"
        }
        service.generate_tags.return_value = ["AI", "Business", "Technology"]
        service.validate_metadata.return_value = True
        service.analyze_metadata_quality.return_value = {
            "completeness": 0.95,
            "accuracy": 0.92,
            "relevance": 0.88
        }
        return service
    
    @pytest.fixture
    def mock_metadata_repository(self):
        """Mock metadata repository"""
        repo = AsyncMock()
        repo.save_metadata.return_value = SAMPLE_CONTENT_METADATA
        repo.get_metadata.return_value = SAMPLE_CONTENT_METADATA
        repo.update_metadata.return_value = SAMPLE_CONTENT_METADATA
        repo.delete_metadata.return_value = True
        repo.search_metadata.return_value = [SAMPLE_CONTENT_METADATA]
        repo.get_metadata_analytics.return_value = {
            "total_posts": 15000,
            "avg_metadata_completeness": 0.87,
            "top_keywords": ["AI", "business", "technology"]
        }
        return repo
    
    @pytest.fixture
    def mock_search_service(self):
        """Mock search service"""
        service = AsyncMock()
        service.index_content.return_value = True
        service.search_content.return_value = [SAMPLE_CONTENT_METADATA]
        service.update_index.return_value = True
        service.delete_from_index.return_value = True
        service.get_search_analytics.return_value = {
            "total_searches": 50000,
            "avg_response_time": 0.15,
            "top_queries": ["AI", "business"]
        }
        return service
    
    @pytest.fixture
    def post_service(self, mock_metadata_repository, mock_metadata_service, mock_search_service):
        from services.post_service import PostService
        service = PostService(
            repository=mock_metadata_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            metadata_service=mock_metadata_service,
            search_service=mock_search_service
        )
        return service
    
    async def test_extract_metadata(self, post_service, mock_metadata_service):
        """Test metadata extraction from content"""
        content = "Artificial Intelligence is transforming business operations"
        
        result = await post_service.extract_metadata(content)
        
        mock_metadata_service.extract_metadata.assert_called_once_with(content)
        assert result == SAMPLE_CONTENT_METADATA
        assert "keywords" in result
        assert "entities" in result
        assert "sentiment" in result
    
    async def test_parse_content(self, post_service, mock_metadata_service):
        """Test content parsing for metadata"""
        content = "AI and machine learning are revolutionizing industries"
        
        result = await post_service.parse_content(content)
        
        mock_metadata_service.parse_content.assert_called_once_with(content)
        assert "keywords" in result
        assert "entities" in result
        assert "sentiment" in result
    
    async def test_generate_tags(self, post_service, mock_metadata_service):
        """Test automatic tag generation"""
        content = "Digital transformation in healthcare"
        
        result = await post_service.generate_tags(content)
        
        mock_metadata_service.generate_tags.assert_called_once_with(content)
        assert isinstance(result, list)
        assert len(result) > 0
    
    async def test_validate_metadata(self, post_service, mock_metadata_service):
        """Test metadata validation"""
        metadata = SAMPLE_CONTENT_METADATA
        
        result = await post_service.validate_metadata(metadata)
        
        mock_metadata_service.validate_metadata.assert_called_once_with(metadata)
        assert result is True
    
    async def test_analyze_metadata_quality(self, post_service, mock_metadata_service):
        """Test metadata quality analysis"""
        metadata = SAMPLE_CONTENT_METADATA
        
        result = await post_service.analyze_metadata_quality(metadata)
        
        mock_metadata_service.analyze_metadata_quality.assert_called_once_with(metadata)
        assert "completeness" in result
        assert "accuracy" in result
        assert "relevance" in result
    
    async def test_save_metadata(self, post_service, mock_metadata_repository):
        """Test saving metadata to repository"""
        metadata = SAMPLE_CONTENT_METADATA
        
        result = await post_service.save_metadata(metadata)
        
        mock_metadata_repository.save_metadata.assert_called_once_with(metadata)
        assert result == SAMPLE_CONTENT_METADATA
    
    async def test_get_metadata(self, post_service, mock_metadata_repository):
        """Test retrieving metadata from repository"""
        post_id = str(uuid4())
        
        result = await post_service.get_metadata(post_id)
        
        mock_metadata_repository.get_metadata.assert_called_once_with(post_id)
        assert result == SAMPLE_CONTENT_METADATA
    
    async def test_update_metadata(self, post_service, mock_metadata_repository):
        """Test updating metadata"""
        post_id = str(uuid4())
        updates = {"keywords": ["AI", "ML"], "sentiment": "positive"}
        
        result = await post_service.update_metadata(post_id, updates)
        
        mock_metadata_repository.update_metadata.assert_called_once_with(post_id, updates)
        assert result == SAMPLE_CONTENT_METADATA
    
    async def test_delete_metadata(self, post_service, mock_metadata_repository):
        """Test deleting metadata"""
        post_id = str(uuid4())
        
        result = await post_service.delete_metadata(post_id)
        
        mock_metadata_repository.delete_metadata.assert_called_once_with(post_id)
        assert result is True
    
    async def test_search_metadata(self, post_service, mock_metadata_repository):
        """Test searching metadata"""
        query = "AI business"
        filters = {"sentiment": "positive", "language": "en"}
        
        result = await post_service.search_metadata(query, filters)
        
        mock_metadata_repository.search_metadata.assert_called_once_with(query, filters)
        assert isinstance(result, list)
        assert len(result) > 0
    
    async def test_get_metadata_analytics(self, post_service, mock_metadata_repository):
        """Test metadata analytics retrieval"""
        result = await post_service.get_metadata_analytics()
        
        mock_metadata_repository.get_metadata_analytics.assert_called_once()
        assert "total_posts" in result
        assert "avg_metadata_completeness" in result
        assert "top_keywords" in result
    
    async def test_index_content(self, post_service, mock_search_service):
        """Test content indexing for search"""
        content = SAMPLE_CONTENT_METADATA
        
        result = await post_service.index_content(content)
        
        mock_search_service.index_content.assert_called_once_with(content)
        assert result is True
    
    async def test_search_content(self, post_service, mock_search_service):
        """Test content search functionality"""
        query = "artificial intelligence"
        filters = {"language": "en", "date_range": "last_30_days"}
        
        result = await post_service.search_content(query, filters)
        
        mock_search_service.search_content.assert_called_once_with(query, filters)
        assert isinstance(result, list)
        assert len(result) > 0
    
    async def test_update_search_index(self, post_service, mock_search_service):
        """Test updating search index"""
        content = SAMPLE_CONTENT_METADATA
        
        result = await post_service.update_search_index(content)
        
        mock_search_service.update_index.assert_called_once_with(content)
        assert result is True
    
    async def test_delete_from_search_index(self, post_service, mock_search_service):
        """Test deleting content from search index"""
        post_id = str(uuid4())
        
        result = await post_service.delete_from_search_index(post_id)
        
        mock_search_service.delete_from_index.assert_called_once_with(post_id)
        assert result is True
    
    async def test_get_search_analytics(self, post_service, mock_search_service):
        """Test search analytics retrieval"""
        result = await post_service.get_search_analytics()
        
        mock_search_service.get_search_analytics.assert_called_once()
        assert "total_searches" in result
        assert "avg_response_time" in result
        assert "top_queries" in result
    
    async def test_bulk_metadata_operations(self, post_service, mock_metadata_repository):
        """Test bulk metadata operations"""
        metadata_list = [SAMPLE_CONTENT_METADATA] * 10
        
        # Mock bulk operations
        mock_metadata_repository.bulk_save_metadata.return_value = metadata_list
        mock_metadata_repository.bulk_update_metadata.return_value = metadata_list
        mock_metadata_repository.bulk_delete_metadata.return_value = True
        
        # Test bulk save
        result_save = await post_service.bulk_save_metadata(metadata_list)
        mock_metadata_repository.bulk_save_metadata.assert_called_once_with(metadata_list)
        assert result_save == metadata_list
        
        # Test bulk update
        updates = [{"keywords": ["AI"]}] * 10
        result_update = await post_service.bulk_update_metadata(updates)
        mock_metadata_repository.bulk_update_metadata.assert_called_once_with(updates)
        assert result_update == metadata_list
        
        # Test bulk delete
        post_ids = [str(uuid4()) for _ in range(10)]
        result_delete = await post_service.bulk_delete_metadata(post_ids)
        mock_metadata_repository.bulk_delete_metadata.assert_called_once_with(post_ids)
        assert result_delete is True
    
    async def test_metadata_validation_rules(self, post_service, mock_metadata_service):
        """Test metadata validation with custom rules"""
        metadata = SAMPLE_CONTENT_METADATA
        validation_rules = {
            "required_fields": ["keywords", "sentiment"],
            "max_keywords": 10,
            "min_word_count": 50
        }
        
        mock_metadata_service.validate_metadata_with_rules.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        result = await post_service.validate_metadata_with_rules(metadata, validation_rules)
        
        mock_metadata_service.validate_metadata_with_rules.assert_called_once_with(metadata, validation_rules)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
    
    async def test_metadata_quality_improvement(self, post_service, mock_metadata_service):
        """Test metadata quality improvement suggestions"""
        metadata = SAMPLE_CONTENT_METADATA
        
        mock_metadata_service.suggest_metadata_improvements.return_value = {
            "suggestions": [
                "Add more specific keywords",
                "Include industry tags",
                "Add location metadata"
            ],
            "confidence_score": 0.85
        }
        
        result = await post_service.suggest_metadata_improvements(metadata)
        
        mock_metadata_service.suggest_metadata_improvements.assert_called_once_with(metadata)
        assert "suggestions" in result
        assert "confidence_score" in result
    
    async def test_metadata_trend_analysis(self, post_service, mock_metadata_repository):
        """Test metadata trend analysis"""
        date_range = "last_30_days"
        
        mock_metadata_repository.analyze_metadata_trends.return_value = {
            "trending_keywords": ["AI", "ML", "automation"],
            "sentiment_trends": {"positive": 0.65, "negative": 0.15, "neutral": 0.20},
            "topic_distribution": {"technology": 0.40, "business": 0.35, "innovation": 0.25}
        }
        
        result = await post_service.analyze_metadata_trends(date_range)
        
        mock_metadata_repository.analyze_metadata_trends.assert_called_once_with(date_range)
        assert "trending_keywords" in result
        assert "sentiment_trends" in result
        assert "topic_distribution" in result
    
    async def test_metadata_export_import(self, post_service, mock_metadata_repository):
        """Test metadata export and import functionality"""
        export_format = "json"
        
        mock_metadata_repository.export_metadata.return_value = {
            "data": [SAMPLE_CONTENT_METADATA],
            "format": export_format,
            "exported_at": datetime.now()
        }
        
        mock_metadata_repository.import_metadata.return_value = {
            "imported_count": 100,
            "errors": [],
            "imported_at": datetime.now()
        }
        
        # Test export
        result_export = await post_service.export_metadata(export_format)
        mock_metadata_repository.export_metadata.assert_called_once_with(export_format)
        assert "data" in result_export
        assert "format" in result_export
        
        # Test import
        import_data = [SAMPLE_CONTENT_METADATA]
        result_import = await post_service.import_metadata(import_data)
        mock_metadata_repository.import_metadata.assert_called_once_with(import_data)
        assert "imported_count" in result_import
        assert "errors" in result_import
