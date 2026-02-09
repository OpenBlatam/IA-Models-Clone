"""
Content Versioning Tests

Tests for content versioning, history management, and version control features.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional


class TestContentVersioning:
    """Test content versioning and history management"""
    
    @pytest.fixture
    def mock_versioning_service(self):
        """Mock versioning service"""
        service = AsyncMock()
        service.create_version.return_value = {
            "version_id": "v1.2.3",
            "version_number": 3,
            "created_at": datetime.now(),
            "author": "user123",
            "changes": ["Updated content", "Added hashtags"]
        }
        service.get_version_history.return_value = [
            {"version": "v1.2.3", "timestamp": datetime.now(), "author": "user123"},
            {"version": "v1.2.2", "timestamp": datetime.now(), "author": "user456"},
            {"version": "v1.2.1", "timestamp": datetime.now(), "author": "user789"}
        ]
        service.rollback_to_version.return_value = {
            "success": True,
            "rolled_back_to": "v1.2.1",
            "current_version": "v1.2.4",
            "rollback_reason": "Content quality issues"
        }
        service.compare_versions.return_value = {
            "differences": [
                {"type": "content_change", "old": "Old content", "new": "New content"},
                {"type": "formatting_change", "old": "Old format", "new": "New format"}
            ],
            "similarity_score": 0.75
        }
        return service
    
    @pytest.fixture
    def mock_versioning_repository(self):
        """Mock versioning repository"""
        repository = AsyncMock()
        repository.get_version_data.return_value = {
            "content": "Version content",
            "metadata": {"author": "user123", "timestamp": datetime.now()},
            "version_info": {"major": 1, "minor": 2, "patch": 3}
        }
        repository.get_all_versions.return_value = [
            {"version": "v1.2.3", "status": "current", "created_at": datetime.now()},
            {"version": "v1.2.2", "status": "archived", "created_at": datetime.now()},
            {"version": "v1.2.1", "status": "archived", "created_at": datetime.now()}
        ]
        repository.save_version.return_value = True
        return repository
    
    @pytest.fixture
    def mock_history_service(self):
        """Mock history service"""
        service = AsyncMock()
        service.track_change.return_value = {
            "change_id": "change_123",
            "timestamp": datetime.now(),
            "change_type": "content_update"
        }
        service.get_change_log.return_value = [
            {"action": "created", "user": "user123", "timestamp": datetime.now()},
            {"action": "updated", "user": "user456", "timestamp": datetime.now()},
            {"action": "published", "user": "user789", "timestamp": datetime.now()}
        ]
        service.export_history.return_value = {
            "export_id": "export_456",
            "file_path": "/exports/history_2024-01-15.json",
            "total_records": 25
        }
        return service
    
    @pytest.fixture
    def post_service(self, mock_versioning_repository, mock_versioning_service, mock_history_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_versioning_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            versioning_service=mock_versioning_service,
            history_service=mock_history_service
        )
        return service
    
    async def test_version_creation(self, post_service, mock_versioning_service):
        """Test version creation"""
        post_id = "post123"
        content_data = {
            "content": "Updated post content",
            "author": "user123",
            "change_reason": "Content improvement"
        }
        
        result = await post_service.create_version(post_id, content_data)
        
        assert result["version_id"] == "v1.2.3"
        assert result["version_number"] == 3
        assert result["author"] == "user123"
        assert len(result["changes"]) == 2
        mock_versioning_service.create_version.assert_called_once_with(post_id, content_data)
    
    async def test_version_history_retrieval(self, post_service, mock_versioning_service):
        """Test version history retrieval"""
        post_id = "post123"
        
        result = await post_service.get_version_history(post_id)
        
        assert len(result) == 3
        assert result[0]["version"] == "v1.2.3"
        assert result[1]["version"] == "v1.2.2"
        assert result[2]["version"] == "v1.2.1"
        mock_versioning_service.get_version_history.assert_called_once_with(post_id)
    
    async def test_version_rollback(self, post_service, mock_versioning_service):
        """Test version rollback"""
        post_id = "post123"
        target_version = "v1.2.1"
        rollback_reason = "Content quality issues"
        
        result = await post_service.rollback_to_version(post_id, target_version, rollback_reason)
        
        assert result["success"] is True
        assert result["rolled_back_to"] == "v1.2.1"
        assert result["current_version"] == "v1.2.4"
        assert result["rollback_reason"] == "Content quality issues"
        mock_versioning_service.rollback_to_version.assert_called_once_with(post_id, target_version, rollback_reason)
    
    async def test_version_comparison(self, post_service, mock_versioning_service):
        """Test version comparison"""
        post_id = "post123"
        version1 = "v1.2.2"
        version2 = "v1.2.3"
        
        result = await post_service.compare_versions(post_id, version1, version2)
        
        assert len(result["differences"]) == 2
        assert result["similarity_score"] == 0.75
        mock_versioning_service.compare_versions.assert_called_once_with(post_id, version1, version2)
    
    async def test_version_data_retrieval(self, post_service, mock_versioning_repository):
        """Test version data retrieval"""
        post_id = "post123"
        version = "v1.2.3"
        
        result = await post_service.get_version_data(post_id, version)
        
        assert result["content"] == "Version content"
        assert result["metadata"]["author"] == "user123"
        assert result["version_info"]["major"] == 1
        assert result["version_info"]["minor"] == 2
        assert result["version_info"]["patch"] == 3
        mock_versioning_repository.get_version_data.assert_called_once_with(post_id, version)
    
    async def test_all_versions_retrieval(self, post_service, mock_versioning_repository):
        """Test retrieval of all versions"""
        post_id = "post123"
        
        result = await post_service.get_all_versions(post_id)
        
        assert len(result) == 3
        assert result[0]["status"] == "current"
        assert result[1]["status"] == "archived"
        assert result[2]["status"] == "archived"
        mock_versioning_repository.get_all_versions.assert_called_once_with(post_id)
    
    async def test_change_tracking(self, post_service, mock_history_service):
        """Test change tracking"""
        change_data = {
            "post_id": "post123",
            "user_id": "user456",
            "action": "content_update",
            "details": "Updated post content"
        }
        
        result = await post_service.track_change(change_data)
        
        assert result["change_id"] == "change_123"
        assert result["change_type"] == "content_update"
        mock_history_service.track_change.assert_called_once_with(change_data)
    
    async def test_change_log_retrieval(self, post_service, mock_history_service):
        """Test change log retrieval"""
        post_id = "post123"
        
        result = await post_service.get_change_log(post_id)
        
        assert len(result) == 3
        assert result[0]["action"] == "created"
        assert result[1]["action"] == "updated"
        assert result[2]["action"] == "published"
        mock_history_service.get_change_log.assert_called_once_with(post_id)
    
    async def test_history_export(self, post_service, mock_history_service):
        """Test history export"""
        post_id = "post123"
        export_format = "json"
        
        result = await post_service.export_history(post_id, export_format)
        
        assert result["export_id"] == "export_456"
        assert result["file_path"] == "/exports/history_2024-01-15.json"
        assert result["total_records"] == 25
        mock_history_service.export_history.assert_called_once_with(post_id, export_format)
    
    async def test_version_data_persistence(self, post_service, mock_versioning_repository):
        """Test version data persistence"""
        version_data = {
            "post_id": "post123",
            "version": "v1.2.3",
            "content": "Version content",
            "metadata": {"author": "user123", "timestamp": datetime.now()}
        }
        
        result = await post_service.save_version(version_data)
        
        assert result is True
        mock_versioning_repository.save_version.assert_called_once_with(version_data)
    
    async def test_version_branching(self, post_service, mock_versioning_service):
        """Test version branching"""
        post_id = "post123"
        base_version = "v1.2.2"
        branch_name = "feature-branch"
        
        mock_versioning_service.create_branch.return_value = {
            "branch_id": "branch_789",
            "base_version": "v1.2.2",
            "branch_name": "feature-branch",
            "created_at": datetime.now()
        }
        
        result = await post_service.create_version_branch(post_id, base_version, branch_name)
        
        assert result["branch_id"] == "branch_789"
        assert result["base_version"] == "v1.2.2"
        assert result["branch_name"] == "feature-branch"
        mock_versioning_service.create_branch.assert_called_once_with(post_id, base_version, branch_name)
    
    async def test_version_merging(self, post_service, mock_versioning_service):
        """Test version merging"""
        post_id = "post123"
        source_version = "v1.2.3"
        target_version = "v1.2.2"
        
        mock_versioning_service.merge_versions.return_value = {
            "merge_id": "merge_123",
            "merged_version": "v1.2.4",
            "conflicts_resolved": 2,
            "merge_status": "completed"
        }
        
        result = await post_service.merge_versions(post_id, source_version, target_version)
        
        assert result["merge_id"] == "merge_123"
        assert result["merged_version"] == "v1.2.4"
        assert result["conflicts_resolved"] == 2
        assert result["merge_status"] == "completed"
        mock_versioning_service.merge_versions.assert_called_once_with(post_id, source_version, target_version)
    
    async def test_version_tagging(self, post_service, mock_versioning_service):
        """Test version tagging"""
        post_id = "post123"
        version = "v1.2.3"
        tag = "stable"
        
        mock_versioning_service.add_tag.return_value = {
            "tag_id": "tag_456",
            "version": "v1.2.3",
            "tag": "stable",
            "added_at": datetime.now()
        }
        
        result = await post_service.add_version_tag(post_id, version, tag)
        
        assert result["tag_id"] == "tag_456"
        assert result["version"] == "v1.2.3"
        assert result["tag"] == "stable"
        mock_versioning_service.add_tag.assert_called_once_with(post_id, version, tag)
    
    async def test_version_cleanup(self, post_service, mock_versioning_service):
        """Test version cleanup"""
        post_id = "post123"
        retention_policy = {
            "max_versions": 10,
            "keep_major_versions": True,
            "archive_after_days": 30
        }
        
        mock_versioning_service.cleanup_versions.return_value = {
            "versions_deleted": 5,
            "versions_archived": 3,
            "space_freed": "2.5MB",
            "cleanup_status": "completed"
        }
        
        result = await post_service.cleanup_versions(post_id, retention_policy)
        
        assert result["versions_deleted"] == 5
        assert result["versions_archived"] == 3
        assert result["space_freed"] == "2.5MB"
        assert result["cleanup_status"] == "completed"
        mock_versioning_service.cleanup_versions.assert_called_once_with(post_id, retention_policy)
    
    async def test_version_analytics(self, post_service, mock_versioning_service):
        """Test version analytics"""
        post_id = "post123"
        time_range = "last_30_days"
        
        mock_versioning_service.get_version_analytics.return_value = {
            "total_versions": 15,
            "avg_versions_per_day": 0.5,
            "most_active_author": "user123",
            "version_frequency": {
                "daily": 2,
                "weekly": 8,
                "monthly": 15
            },
            "rollback_frequency": 3
        }
        
        result = await post_service.get_version_analytics(post_id, time_range)
        
        assert result["total_versions"] == 15
        assert result["avg_versions_per_day"] == 0.5
        assert result["most_active_author"] == "user123"
        assert result["rollback_frequency"] == 3
        mock_versioning_service.get_version_analytics.assert_called_once_with(post_id, time_range)
