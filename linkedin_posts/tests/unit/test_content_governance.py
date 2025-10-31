"""
Content Governance Tests
========================

Tests for content governance, policy management, regulatory compliance,
audit trails, and governance workflows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Test data
SAMPLE_POST_DATA = {
    "id": "test-post-123",
    "content": "This is a LinkedIn post that needs governance review and compliance checks.",
    "author_id": "user-123",
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
    "status": "pending_governance_review"
}

SAMPLE_GOVERNANCE_POLICY = {
    "policy_id": "gov-policy-001",
    "name": "Content Governance Policy",
    "version": "1.0",
    "effective_date": datetime.now(),
    "rules": [
        {
            "rule_id": "rule-001",
            "name": "No confidential information",
            "description": "Posts must not contain confidential or proprietary information",
            "severity": "high",
            "enforcement": "block"
        },
        {
            "rule_id": "rule-002",
            "name": "Professional tone",
            "description": "Content must maintain professional tone",
            "severity": "medium",
            "enforcement": "warn"
        }
    ],
    "review_required": True,
    "approval_workflow": "manager_approval"
}

SAMPLE_AUDIT_TRAIL = {
    "audit_id": "audit-001",
    "post_id": "test-post-123",
    "action": "governance_review",
    "reviewer_id": "gov-reviewer-001",
    "timestamp": datetime.now(),
    "details": {
        "policy_applied": "Content Governance Policy v1.0",
        "rules_checked": ["rule-001", "rule-002"],
        "compliance_status": "compliant",
        "notes": "Post passed all governance checks"
    },
    "previous_status": "pending_governance_review",
    "new_status": "approved"
}

SAMPLE_REGULATORY_COMPLIANCE = {
    "compliance_id": "comp-001",
    "post_id": "test-post-123",
    "regulations": [
        {
            "regulation_id": "GDPR-001",
            "name": "General Data Protection Regulation",
            "compliance_status": "compliant",
            "checks_performed": [
                "data_minimization",
                "consent_validation",
                "right_to_erasure"
            ]
        },
        {
            "regulation_id": "SOX-001",
            "name": "Sarbanes-Oxley Act",
            "compliance_status": "compliant",
            "checks_performed": [
                "financial_disclosure",
                "internal_controls",
                "audit_trail"
            ]
        }
    ],
    "overall_compliance": "compliant",
    "compliance_score": 95.0,
    "review_date": datetime.now()
}

SAMPLE_GOVERNANCE_WORKFLOW = {
    "workflow_id": "workflow-001",
    "post_id": "test-post-123",
    "workflow_type": "content_governance",
    "current_step": "policy_review",
    "steps": [
        {
            "step_id": "step-001",
            "name": "Initial Review",
            "status": "completed",
            "assignee": "gov-reviewer-001",
            "completed_at": datetime.now(),
            "notes": "Initial governance review completed"
        },
        {
            "step_id": "step-002",
            "name": "Policy Review",
            "status": "in_progress",
            "assignee": "policy-expert-001",
            "due_date": datetime.now() + timedelta(hours=2)
        },
        {
            "step_id": "step-003",
            "name": "Final Approval",
            "status": "pending",
            "assignee": "governance-manager-001"
        }
    ],
    "created_at": datetime.now(),
    "estimated_completion": datetime.now() + timedelta(hours=4)
}


class TestContentGovernance:
    """Test content governance and policy management"""
    
    @pytest.fixture
    def mock_governance_service(self):
        """Mock governance service"""
        service = AsyncMock()
        
        # Mock policy management
        service.get_policies.return_value = [SAMPLE_GOVERNANCE_POLICY]
        service.apply_policy.return_value = {
            "policy_id": "gov-policy-001",
            "compliance_status": "compliant",
            "violations": [],
            "recommendations": []
        }
        
        # Mock audit trail
        service.create_audit_entry.return_value = SAMPLE_AUDIT_TRAIL
        service.get_audit_trail.return_value = [SAMPLE_AUDIT_TRAIL]
        
        # Mock regulatory compliance
        service.check_regulatory_compliance.return_value = SAMPLE_REGULATORY_COMPLIANCE
        
        # Mock workflow management
        service.create_workflow.return_value = SAMPLE_GOVERNANCE_WORKFLOW
        service.update_workflow_step.return_value = SAMPLE_GOVERNANCE_WORKFLOW
        
        return service
    
    @pytest.fixture
    def mock_governance_repository(self):
        """Mock governance repository"""
        repository = AsyncMock()
        
        # Mock policy storage
        repository.save_policy.return_value = SAMPLE_GOVERNANCE_POLICY
        repository.get_policy.return_value = SAMPLE_GOVERNANCE_POLICY
        repository.list_policies.return_value = [SAMPLE_GOVERNANCE_POLICY]
        
        # Mock audit trail storage
        repository.save_audit_entry.return_value = SAMPLE_AUDIT_TRAIL
        repository.get_audit_entries.return_value = [SAMPLE_AUDIT_TRAIL]
        
        # Mock compliance storage
        repository.save_compliance_check.return_value = SAMPLE_REGULATORY_COMPLIANCE
        repository.get_compliance_history.return_value = [SAMPLE_REGULATORY_COMPLIANCE]
        
        # Mock workflow storage
        repository.save_workflow.return_value = SAMPLE_GOVERNANCE_WORKFLOW
        repository.get_workflow.return_value = SAMPLE_GOVERNANCE_WORKFLOW
        
        return repository
    
    @pytest.fixture
    def mock_policy_service(self):
        """Mock policy service"""
        service = AsyncMock()
        
        service.validate_policy.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        service.evaluate_policy.return_value = {
            "compliant": True,
            "violations": [],
            "score": 95.0
        }
        
        return service
    
    @pytest.fixture
    def post_service(self, mock_governance_repository, mock_governance_service, mock_policy_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_governance_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            governance_service=mock_governance_service,
            policy_service=mock_policy_service
        )
        return service
    
    async def test_policy_application(self, post_service, mock_governance_service):
        """Test applying governance policies to posts"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        policy_id = "gov-policy-001"
        
        # Act
        result = await post_service.apply_governance_policy(post_data, policy_id)
        
        # Assert
        assert result["policy_id"] == policy_id
        assert result["compliance_status"] == "compliant"
        assert len(result["violations"]) == 0
        mock_governance_service.apply_policy.assert_called_once_with(post_data, policy_id)
    
    async def test_audit_trail_creation(self, post_service, mock_governance_service):
        """Test creating audit trail entries"""
        # Arrange
        post_id = "test-post-123"
        action = "governance_review"
        reviewer_id = "gov-reviewer-001"
        details = {"policy_applied": "Content Governance Policy"}
        
        # Act
        audit_entry = await post_service.create_governance_audit_entry(
            post_id, action, reviewer_id, details
        )
        
        # Assert
        assert audit_entry["audit_id"] == "audit-001"
        assert audit_entry["post_id"] == post_id
        assert audit_entry["action"] == action
        assert audit_entry["reviewer_id"] == reviewer_id
        mock_governance_service.create_audit_entry.assert_called_once()
    
    async def test_regulatory_compliance_check(self, post_service, mock_governance_service):
        """Test checking regulatory compliance"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        regulations = ["GDPR", "SOX"]
        
        # Act
        compliance_result = await post_service.check_regulatory_compliance(post_data, regulations)
        
        # Assert
        assert compliance_result["overall_compliance"] == "compliant"
        assert compliance_result["compliance_score"] == 95.0
        assert len(compliance_result["regulations"]) == 2
        mock_governance_service.check_regulatory_compliance.assert_called_once()
    
    async def test_governance_workflow_creation(self, post_service, mock_governance_service):
        """Test creating governance workflows"""
        # Arrange
        post_id = "test-post-123"
        workflow_type = "content_governance"
        
        # Act
        workflow = await post_service.create_governance_workflow(post_id, workflow_type)
        
        # Assert
        assert workflow["workflow_id"] == "workflow-001"
        assert workflow["post_id"] == post_id
        assert workflow["workflow_type"] == workflow_type
        assert workflow["current_step"] == "policy_review"
        mock_governance_service.create_workflow.assert_called_once()
    
    async def test_policy_validation(self, post_service, mock_policy_service):
        """Test validating governance policies"""
        # Arrange
        policy_data = SAMPLE_GOVERNANCE_POLICY.copy()
        
        # Act
        validation_result = await post_service.validate_governance_policy(policy_data)
        
        # Assert
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        mock_policy_service.validate_policy.assert_called_once_with(policy_data)
    
    async def test_policy_evaluation(self, post_service, mock_policy_service):
        """Test evaluating policies against content"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        policy_id = "gov-policy-001"
        
        # Act
        evaluation_result = await post_service.evaluate_policy_against_content(post_data, policy_id)
        
        # Assert
        assert evaluation_result["compliant"] is True
        assert evaluation_result["score"] == 95.0
        assert len(evaluation_result["violations"]) == 0
        mock_policy_service.evaluate_policy.assert_called_once()
    
    async def test_audit_trail_retrieval(self, post_service, mock_governance_service):
        """Test retrieving audit trail for a post"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        audit_trail = await post_service.get_governance_audit_trail(post_id)
        
        # Assert
        assert len(audit_trail) == 1
        assert audit_trail[0]["post_id"] == post_id
        mock_governance_service.get_audit_trail.assert_called_once_with(post_id)
    
    async def test_compliance_history_tracking(self, post_service, mock_governance_repository):
        """Test tracking compliance history"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        compliance_history = await post_service.get_compliance_history(post_id)
        
        # Assert
        assert len(compliance_history) == 1
        assert compliance_history[0]["post_id"] == post_id
        mock_governance_repository.get_compliance_history.assert_called_once_with(post_id)
    
    async def test_workflow_step_update(self, post_service, mock_governance_service):
        """Test updating workflow steps"""
        # Arrange
        workflow_id = "workflow-001"
        step_id = "step-002"
        status = "completed"
        notes = "Policy review completed"
        
        # Act
        updated_workflow = await post_service.update_workflow_step(
            workflow_id, step_id, status, notes
        )
        
        # Assert
        assert updated_workflow["workflow_id"] == workflow_id
        mock_governance_service.update_workflow_step.assert_called_once()
    
    async def test_policy_rule_enforcement(self, post_service, mock_governance_service):
        """Test enforcing policy rules"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        rule_id = "rule-001"
        
        # Act
        enforcement_result = await post_service.enforce_policy_rule(post_data, rule_id)
        
        # Assert
        assert enforcement_result is not None
        mock_governance_service.enforce_rule.assert_called_once()
    
    async def test_governance_report_generation(self, post_service, mock_governance_service):
        """Test generating governance reports"""
        # Arrange
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Act
        report = await post_service.generate_governance_report(start_date, end_date)
        
        # Assert
        assert report is not None
        assert "summary" in report
        assert "violations" in report
        assert "compliance_rate" in report
        mock_governance_service.generate_report.assert_called_once()
    
    async def test_policy_version_management(self, post_service, mock_governance_repository):
        """Test managing policy versions"""
        # Arrange
        policy_id = "gov-policy-001"
        new_version = "2.0"
        
        # Act
        versioned_policy = await post_service.create_policy_version(policy_id, new_version)
        
        # Assert
        assert versioned_policy["version"] == new_version
        mock_governance_repository.create_policy_version.assert_called_once()
    
    async def test_governance_metrics_tracking(self, post_service, mock_governance_service):
        """Test tracking governance metrics"""
        # Arrange
        time_period = "last_30_days"
        
        # Act
        metrics = await post_service.get_governance_metrics(time_period)
        
        # Assert
        assert metrics is not None
        assert "compliance_rate" in metrics
        assert "average_review_time" in metrics
        assert "violation_count" in metrics
        mock_governance_service.get_metrics.assert_called_once()
    
    async def test_regulatory_requirement_mapping(self, post_service, mock_governance_service):
        """Test mapping content to regulatory requirements"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        requirements = await post_service.map_regulatory_requirements(post_data)
        
        # Assert
        assert requirements is not None
        assert len(requirements) > 0
        mock_governance_service.map_requirements.assert_called_once()
    
    async def test_governance_alert_generation(self, post_service, mock_governance_service):
        """Test generating governance alerts"""
        # Arrange
        post_id = "test-post-123"
        alert_type = "policy_violation"
        severity = "high"
        
        # Act
        alert = await post_service.create_governance_alert(post_id, alert_type, severity)
        
        # Assert
        assert alert is not None
        assert alert["alert_type"] == alert_type
        assert alert["severity"] == severity
        mock_governance_service.create_alert.assert_called_once()
    
    async def test_policy_effectiveness_analysis(self, post_service, mock_governance_service):
        """Test analyzing policy effectiveness"""
        # Arrange
        policy_id = "gov-policy-001"
        analysis_period = "last_quarter"
        
        # Act
        analysis = await post_service.analyze_policy_effectiveness(policy_id, analysis_period)
        
        # Assert
        assert analysis is not None
        assert "effectiveness_score" in analysis
        assert "improvement_suggestions" in analysis
        mock_governance_service.analyze_effectiveness.assert_called_once()
    
    async def test_governance_automation_rules(self, post_service, mock_governance_service):
        """Test governance automation rules"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        automation_result = await post_service.apply_governance_automation(post_data)
        
        # Assert
        assert automation_result is not None
        assert "automated_actions" in automation_result
        mock_governance_service.apply_automation.assert_called_once()
    
    async def test_governance_escalation_workflow(self, post_service, mock_governance_service):
        """Test governance escalation workflows"""
        # Arrange
        post_id = "test-post-123"
        escalation_reason = "high_risk_content"
        
        # Act
        escalation = await post_service.escalate_governance_review(post_id, escalation_reason)
        
        # Assert
        assert escalation is not None
        assert escalation["escalation_reason"] == escalation_reason
        mock_governance_service.escalate_review.assert_called_once()
    
    async def test_governance_compliance_certification(self, post_service, mock_governance_service):
        """Test governance compliance certification"""
        # Arrange
        post_id = "test-post-123"
        certification_type = "regulatory_compliance"
        
        # Act
        certification = await post_service.certify_governance_compliance(post_id, certification_type)
        
        # Assert
        assert certification is not None
        assert certification["certification_type"] == certification_type
        assert certification["certified"] is True
        mock_governance_service.certify_compliance.assert_called_once()
    
    async def test_governance_policy_rollback(self, post_service, mock_governance_service):
        """Test rolling back governance policies"""
        # Arrange
        policy_id = "gov-policy-001"
        rollback_reason = "policy_issues"
        
        # Act
        rollback_result = await post_service.rollback_governance_policy(policy_id, rollback_reason)
        
        # Assert
        assert rollback_result is not None
        assert rollback_result["rolled_back"] is True
        mock_governance_service.rollback_policy.assert_called_once()
