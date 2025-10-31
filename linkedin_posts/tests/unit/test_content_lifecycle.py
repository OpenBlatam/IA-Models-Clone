"""
Content Lifecycle Tests
=======================

Tests for content lifecycle management, state transitions, archiving,
retention policies, and lifecycle workflows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Test data
SAMPLE_POST_DATA = {
    "id": "test-post-123",
    "content": "This is a LinkedIn post in the lifecycle management system.",
    "author_id": "user-123",
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
    "status": "draft",
    "lifecycle_state": "creation"
}

SAMPLE_LIFECYCLE_STATE = {
    "post_id": "test-post-123",
    "current_state": "draft",
    "previous_state": "creation",
    "state_transition_time": datetime.now(),
    "state_duration": timedelta(hours=2),
    "state_metadata": {
        "reviewer_id": "reviewer-001",
        "approval_notes": "Content looks good",
        "next_review_date": datetime.now() + timedelta(days=1)
    },
    "lifecycle_phase": "content_development"
}

SAMPLE_STATE_TRANSITION = {
    "transition_id": "trans-001",
    "post_id": "test-post-123",
    "from_state": "draft",
    "to_state": "review",
    "transition_time": datetime.now(),
    "triggered_by": "user-123",
    "transition_reason": "manual_submission",
    "transition_metadata": {
        "approval_required": True,
        "auto_publish": False,
        "notify_reviewers": True
    }
}

SAMPLE_RETENTION_POLICY = {
    "policy_id": "retention-001",
    "name": "Standard Content Retention",
    "description": "Standard retention policy for LinkedIn posts",
    "retention_period": timedelta(days=365),
    "archive_after": timedelta(days=30),
    "delete_after": timedelta(days=365),
    "exceptions": [
        {
            "condition": "high_engagement",
            "retention_period": timedelta(days=730)
        },
        {
            "condition": "regulatory_requirement",
            "retention_period": timedelta(days=2555)
        }
    ],
    "auto_archive": True,
    "auto_delete": True
}

SAMPLE_ARCHIVE_RECORD = {
    "archive_id": "archive-001",
    "post_id": "test-post-123",
    "archive_date": datetime.now(),
    "archive_reason": "retention_policy",
    "original_location": "active_posts",
    "archive_location": "archived_posts",
    "archive_metadata": {
        "engagement_at_archive": 150,
        "days_since_creation": 45,
        "archive_performed_by": "system"
    },
    "restore_available": True
}

SAMPLE_LIFECYCLE_WORKFLOW = {
    "workflow_id": "lifecycle-001",
    "post_id": "test-post-123",
    "workflow_type": "content_lifecycle",
    "current_phase": "content_development",
    "phases": [
        {
            "phase_id": "phase-001",
            "name": "Content Creation",
            "status": "completed",
            "start_time": datetime.now() - timedelta(hours=3),
            "end_time": datetime.now() - timedelta(hours=2),
            "duration": timedelta(hours=1)
        },
        {
            "phase_id": "phase-002",
            "name": "Content Review",
            "status": "in_progress",
            "start_time": datetime.now() - timedelta(hours=2),
            "estimated_end_time": datetime.now() + timedelta(hours=1)
        },
        {
            "phase_id": "phase-003",
            "name": "Content Publication",
            "status": "pending",
            "estimated_start_time": datetime.now() + timedelta(hours=1)
        }
    ],
    "total_estimated_duration": timedelta(hours=4),
    "actual_duration": timedelta(hours=2)
}

SAMPLE_CONTENT_VERSION_HISTORY = {
    "post_id": "test-post-123",
    "versions": [
        {
            "version_id": "v1.0",
            "version_number": 1,
            "created_at": datetime.now() - timedelta(hours=3),
            "created_by": "user-123",
            "content": "Initial version of the post",
            "state": "draft",
            "changes": "Initial creation"
        },
        {
            "version_id": "v1.1",
            "version_number": 2,
            "created_at": datetime.now() - timedelta(hours=2),
            "created_by": "user-123",
            "content": "Updated version with improvements",
            "state": "review",
            "changes": "Added more details and improved formatting"
        }
    ],
    "current_version": "v1.1",
    "total_versions": 2
}

SAMPLE_LIFECYCLE_METRICS = {
    "post_id": "test-post-123",
    "metrics": {
        "time_in_draft": timedelta(hours=2),
        "time_in_review": timedelta(hours=1),
        "time_in_published": timedelta(days=5),
        "total_lifecycle_time": timedelta(days=5, hours=3),
        "state_transitions": 3,
        "review_cycles": 1,
        "approval_time": timedelta(hours=1)
    },
    "performance_indicators": {
        "efficiency_score": 85.5,
        "bottleneck_phases": ["review"],
        "optimization_opportunities": ["automate_review"]
    }
}


class TestContentLifecycle:
    """Test content lifecycle management"""
    
    @pytest.fixture
    def mock_lifecycle_service(self):
        """Mock lifecycle service"""
        service = AsyncMock()
        
        # Mock state management
        service.get_current_state.return_value = SAMPLE_LIFECYCLE_STATE
        service.transition_state.return_value = SAMPLE_STATE_TRANSITION
        service.get_state_history.return_value = [SAMPLE_LIFECYCLE_STATE]
        
        # Mock retention policies
        service.get_retention_policy.return_value = SAMPLE_RETENTION_POLICY
        service.apply_retention_policy.return_value = {
            "policy_applied": "retention-001",
            "next_review_date": datetime.now() + timedelta(days=30),
            "archive_date": datetime.now() + timedelta(days=30)
        }
        
        # Mock archiving
        service.archive_content.return_value = SAMPLE_ARCHIVE_RECORD
        service.restore_content.return_value = {"restored": True}
        
        # Mock workflow management
        service.get_lifecycle_workflow.return_value = SAMPLE_LIFECYCLE_WORKFLOW
        service.update_workflow_phase.return_value = SAMPLE_LIFECYCLE_WORKFLOW
        
        # Mock version history
        service.get_version_history.return_value = SAMPLE_CONTENT_VERSION_HISTORY
        
        # Mock metrics
        service.get_lifecycle_metrics.return_value = SAMPLE_LIFECYCLE_METRICS
        
        return service
    
    @pytest.fixture
    def mock_lifecycle_repository(self):
        """Mock lifecycle repository"""
        repository = AsyncMock()
        
        # Mock state storage
        repository.save_state.return_value = SAMPLE_LIFECYCLE_STATE
        repository.get_state.return_value = SAMPLE_LIFECYCLE_STATE
        repository.save_transition.return_value = SAMPLE_STATE_TRANSITION
        
        # Mock retention storage
        repository.save_retention_policy.return_value = SAMPLE_RETENTION_POLICY
        repository.get_retention_policies.return_value = [SAMPLE_RETENTION_POLICY]
        
        # Mock archive storage
        repository.save_archive_record.return_value = SAMPLE_ARCHIVE_RECORD
        repository.get_archive_records.return_value = [SAMPLE_ARCHIVE_RECORD]
        
        # Mock workflow storage
        repository.save_workflow.return_value = SAMPLE_LIFECYCLE_WORKFLOW
        repository.get_workflow.return_value = SAMPLE_LIFECYCLE_WORKFLOW
        
        return repository
    
    @pytest.fixture
    def mock_state_service(self):
        """Mock state management service"""
        service = AsyncMock()
        
        service.validate_transition.return_value = {
            "valid": True,
            "allowed_transitions": ["review", "published"],
            "requirements": ["content_validation"]
        }
        
        service.get_available_transitions.return_value = ["review", "published", "archived"]
        
        return service
    
    @pytest.fixture
    def post_service(self, mock_lifecycle_repository, mock_lifecycle_service, mock_state_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_lifecycle_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            lifecycle_service=mock_lifecycle_service,
            state_service=mock_state_service
        )
        return service
    
    async def test_state_transition(self, post_service, mock_lifecycle_service):
        """Test transitioning content states"""
        # Arrange
        post_id = "test-post-123"
        new_state = "review"
        transition_reason = "manual_submission"
        
        # Act
        transition = await post_service.transition_content_state(
            post_id, new_state, transition_reason
        )
        
        # Assert
        assert transition["from_state"] == "draft"
        assert transition["to_state"] == new_state
        assert transition["transition_reason"] == transition_reason
        mock_lifecycle_service.transition_state.assert_called_once()
    
    async def test_current_state_retrieval(self, post_service, mock_lifecycle_service):
        """Test retrieving current content state"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        state = await post_service.get_current_lifecycle_state(post_id)
        
        # Assert
        assert state["current_state"] == "draft"
        assert state["post_id"] == post_id
        mock_lifecycle_service.get_current_state.assert_called_once_with(post_id)
    
    async def test_state_history_tracking(self, post_service, mock_lifecycle_service):
        """Test tracking state history"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        history = await post_service.get_state_history(post_id)
        
        # Assert
        assert len(history) == 1
        assert history[0]["post_id"] == post_id
        mock_lifecycle_service.get_state_history.assert_called_once_with(post_id)
    
    async def test_retention_policy_application(self, post_service, mock_lifecycle_service):
        """Test applying retention policies"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        policy_id = "retention-001"
        
        # Act
        result = await post_service.apply_retention_policy(post_data, policy_id)
        
        # Assert
        assert result["policy_applied"] == policy_id
        assert "next_review_date" in result
        assert "archive_date" in result
        mock_lifecycle_service.apply_retention_policy.assert_called_once()
    
    async def test_content_archiving(self, post_service, mock_lifecycle_service):
        """Test archiving content"""
        # Arrange
        post_id = "test-post-123"
        archive_reason = "retention_policy"
        
        # Act
        archive_record = await post_service.archive_content(post_id, archive_reason)
        
        # Assert
        assert archive_record["archive_id"] == "archive-001"
        assert archive_record["post_id"] == post_id
        assert archive_record["archive_reason"] == archive_reason
        mock_lifecycle_service.archive_content.assert_called_once()
    
    async def test_content_restoration(self, post_service, mock_lifecycle_service):
        """Test restoring archived content"""
        # Arrange
        archive_id = "archive-001"
        
        # Act
        result = await post_service.restore_content(archive_id)
        
        # Assert
        assert result["restored"] is True
        mock_lifecycle_service.restore_content.assert_called_once_with(archive_id)
    
    async def test_lifecycle_workflow_creation(self, post_service, mock_lifecycle_service):
        """Test creating lifecycle workflows"""
        # Arrange
        post_id = "test-post-123"
        workflow_type = "content_lifecycle"
        
        # Act
        workflow = await post_service.create_lifecycle_workflow(post_id, workflow_type)
        
        # Assert
        assert workflow["workflow_id"] == "lifecycle-001"
        assert workflow["post_id"] == post_id
        assert workflow["workflow_type"] == workflow_type
        mock_lifecycle_service.get_lifecycle_workflow.assert_called_once()
    
    async def test_workflow_phase_update(self, post_service, mock_lifecycle_service):
        """Test updating workflow phases"""
        # Arrange
        workflow_id = "lifecycle-001"
        phase_id = "phase-002"
        status = "completed"
        
        # Act
        updated_workflow = await post_service.update_workflow_phase(
            workflow_id, phase_id, status
        )
        
        # Assert
        assert updated_workflow["workflow_id"] == workflow_id
        mock_lifecycle_service.update_workflow_phase.assert_called_once()
    
    async def test_version_history_tracking(self, post_service, mock_lifecycle_service):
        """Test tracking version history"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        version_history = await post_service.get_version_history(post_id)
        
        # Assert
        assert version_history["post_id"] == post_id
        assert len(version_history["versions"]) == 2
        assert version_history["current_version"] == "v1.1"
        mock_lifecycle_service.get_version_history.assert_called_once_with(post_id)
    
    async def test_lifecycle_metrics_calculation(self, post_service, mock_lifecycle_service):
        """Test calculating lifecycle metrics"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        metrics = await post_service.get_lifecycle_metrics(post_id)
        
        # Assert
        assert metrics["post_id"] == post_id
        assert "metrics" in metrics
        assert "performance_indicators" in metrics
        mock_lifecycle_service.get_lifecycle_metrics.assert_called_once_with(post_id)
    
    async def test_state_transition_validation(self, post_service, mock_state_service):
        """Test validating state transitions"""
        # Arrange
        current_state = "draft"
        target_state = "review"
        
        # Act
        validation = await post_service.validate_state_transition(current_state, target_state)
        
        # Assert
        assert validation["valid"] is True
        assert "allowed_transitions" in validation
        mock_state_service.validate_transition.assert_called_once()
    
    async def test_available_transitions_retrieval(self, post_service, mock_state_service):
        """Test retrieving available state transitions"""
        # Arrange
        current_state = "draft"
        
        # Act
        transitions = await post_service.get_available_transitions(current_state)
        
        # Assert
        assert len(transitions) == 3
        assert "review" in transitions
        assert "published" in transitions
        mock_state_service.get_available_transitions.assert_called_once_with(current_state)
    
    async def test_retention_policy_creation(self, post_service, mock_lifecycle_repository):
        """Test creating retention policies"""
        # Arrange
        policy_data = SAMPLE_RETENTION_POLICY.copy()
        
        # Act
        created_policy = await post_service.create_retention_policy(policy_data)
        
        # Assert
        assert created_policy["policy_id"] == "retention-001"
        assert created_policy["name"] == "Standard Content Retention"
        mock_lifecycle_repository.save_retention_policy.assert_called_once()
    
    async def test_archive_records_retrieval(self, post_service, mock_lifecycle_repository):
        """Test retrieving archive records"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        archive_records = await post_service.get_archive_records(post_id)
        
        # Assert
        assert len(archive_records) == 1
        assert archive_records[0]["post_id"] == post_id
        mock_lifecycle_repository.get_archive_records.assert_called_once_with(post_id)
    
    async def test_lifecycle_automation_rules(self, post_service, mock_lifecycle_service):
        """Test lifecycle automation rules"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        automation_result = await post_service.apply_lifecycle_automation(post_data)
        
        # Assert
        assert automation_result is not None
        assert "automated_actions" in automation_result
        mock_lifecycle_service.apply_automation.assert_called_once()
    
    async def test_content_expiration_handling(self, post_service, mock_lifecycle_service):
        """Test handling content expiration"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        expiration_result = await post_service.handle_content_expiration(post_id)
        
        # Assert
        assert expiration_result is not None
        assert "expiration_action" in expiration_result
        mock_lifecycle_service.handle_expiration.assert_called_once()
    
    async def test_lifecycle_performance_analysis(self, post_service, mock_lifecycle_service):
        """Test analyzing lifecycle performance"""
        # Arrange
        time_period = "last_30_days"
        
        # Act
        analysis = await post_service.analyze_lifecycle_performance(time_period)
        
        # Assert
        assert analysis is not None
        assert "average_lifecycle_time" in analysis
        assert "bottleneck_phases" in analysis
        mock_lifecycle_service.analyze_performance.assert_called_once()
    
    async def test_content_migration_workflow(self, post_service, mock_lifecycle_service):
        """Test content migration workflows"""
        # Arrange
        post_id = "test-post-123"
        target_environment = "production"
        
        # Act
        migration = await post_service.migrate_content_lifecycle(post_id, target_environment)
        
        # Assert
        assert migration is not None
        assert "migration_status" in migration
        mock_lifecycle_service.migrate_lifecycle.assert_called_once()
    
    async def test_lifecycle_backup_restoration(self, post_service, mock_lifecycle_service):
        """Test lifecycle backup and restoration"""
        # Arrange
        post_id = "test-post-123"
        backup_id = "backup-001"
        
        # Act
        restoration = await post_service.restore_lifecycle_backup(post_id, backup_id)
        
        # Assert
        assert restoration is not None
        assert "restoration_status" in restoration
        mock_lifecycle_service.restore_backup.assert_called_once()
    
    async def test_content_lifecycle_audit(self, post_service, mock_lifecycle_service):
        """Test auditing content lifecycle"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        audit = await post_service.audit_content_lifecycle(post_id)
        
        # Assert
        assert audit is not None
        assert "audit_findings" in audit
        assert "compliance_status" in audit
        mock_lifecycle_service.audit_lifecycle.assert_called_once()
    
    async def test_lifecycle_optimization_suggestions(self, post_service, mock_lifecycle_service):
        """Test generating lifecycle optimization suggestions"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        suggestions = await post_service.get_lifecycle_optimization_suggestions(post_id)
        
        # Assert
        assert suggestions is not None
        assert "optimization_opportunities" in suggestions
        mock_lifecycle_service.get_optimization_suggestions.assert_called_once()
    
    async def test_content_lifecycle_cleanup(self, post_service, mock_lifecycle_service):
        """Test cleaning up lifecycle data"""
        # Arrange
        post_id = "test-post-123"
        cleanup_type = "archived_content"
        
        # Act
        cleanup_result = await post_service.cleanup_lifecycle_data(post_id, cleanup_type)
        
        # Assert
        assert cleanup_result is not None
        assert "cleanup_status" in cleanup_result
        mock_lifecycle_service.cleanup_data.assert_called_once()
