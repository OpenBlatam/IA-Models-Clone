"""
Content Compliance Tests
=======================

Tests for content compliance, governance, regulatory requirements, and audit trails.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Test data
SAMPLE_POST_DATA = {
    "id": "test-post-123",
    "content": "This is a compliant LinkedIn post that follows all guidelines and regulations.",
    "author_id": "user-123",
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
    "status": "draft"
}

SAMPLE_COMPLIANCE_CHECK = {
    "post_id": "test-post-123",
    "compliance_status": "compliant",
    "compliance_score": 95.2,
    "violations": [],
    "warnings": [],
    "recommendations": [
        "Consider adding disclaimer for industry-specific content"
    ],
    "checked_at": datetime.now(),
    "checker_id": "compliance-bot-001"
}

SAMPLE_AUDIT_TRAIL = {
    "audit_id": "audit-123",
    "post_id": "test-post-123",
    "action": "compliance_check",
    "user_id": "user-123",
    "timestamp": datetime.now(),
    "details": {
        "compliance_score": 95.2,
        "violations_found": 0,
        "warnings_issued": 1
    },
    "ip_address": "192.168.1.100",
    "user_agent": "LinkedIn-Post-Manager/1.0"
}

SAMPLE_GOVERNANCE_POLICY = {
    "policy_id": "policy-001",
    "name": "Content Standards Policy",
    "version": "1.2",
    "effective_date": datetime.now(),
    "rules": [
        {
            "rule_id": "rule-001",
            "name": "No Profanity",
            "description": "Content must not contain profanity or offensive language",
            "severity": "high",
            "enabled": True
        },
        {
            "rule_id": "rule-002",
            "name": "Industry Compliance",
            "description": "Content must comply with industry-specific regulations",
            "severity": "medium",
            "enabled": True
        }
    ]
}


class TestContentCompliance:
    """Test content compliance and governance"""
    
    @pytest.fixture
    def mock_compliance_service(self):
        """Mock compliance service"""
        service = AsyncMock()
        
        # Mock compliance checking
        service.check_compliance.return_value = SAMPLE_COMPLIANCE_CHECK
        service.validate_regulatory_requirements.return_value = True
        service.screen_content.return_value = {
            "screening_status": "passed",
            "risk_level": "low",
            "flagged_issues": []
        }
        
        # Mock governance
        service.apply_governance_policies.return_value = {
            "policies_applied": 2,
            "violations": 0,
            "warnings": 1
        }
        service.get_governance_policies.return_value = [SAMPLE_GOVERNANCE_POLICY]
        
        # Mock audit
        service.create_audit_trail.return_value = SAMPLE_AUDIT_TRAIL
        service.get_audit_history.return_value = [SAMPLE_AUDIT_TRAIL]
        
        return service
    
    @pytest.fixture
    def mock_compliance_repository(self):
        """Mock compliance repository"""
        repository = AsyncMock()
        
        # Mock compliance data persistence
        repository.save_compliance_check.return_value = "check-123"
        repository.get_compliance_check.return_value = SAMPLE_COMPLIANCE_CHECK
        repository.save_audit_trail.return_value = "audit-123"
        repository.get_audit_history.return_value = [SAMPLE_AUDIT_TRAIL]
        
        return repository
    
    @pytest.fixture
    def mock_governance_service(self):
        """Mock governance service"""
        service = AsyncMock()
        
        # Mock policy management
        service.get_active_policies.return_value = [SAMPLE_GOVERNANCE_POLICY]
        service.validate_policy_compliance.return_value = {
            "compliant": True,
            "violations": [],
            "warnings": ["Consider adding disclaimer"]
        }
        service.update_policy.return_value = True
        
        return service
    
    @pytest.fixture
    def post_service(self, mock_compliance_repository, mock_compliance_service, mock_governance_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_compliance_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            compliance_service=mock_compliance_service,
            governance_service=mock_governance_service
        )
        return service
    
    async def test_compliance_check(self, post_service, mock_compliance_service):
        """Test compliance check"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.check_compliance(post_data)
        
        # Assert
        assert result == SAMPLE_COMPLIANCE_CHECK
        assert result["compliance_status"] == "compliant"
        assert result["compliance_score"] == 95.2
        assert len(result["violations"]) == 0
        mock_compliance_service.check_compliance.assert_called_once_with(post_data)
    
    async def test_regulatory_requirements_validation(self, post_service, mock_compliance_service):
        """Test regulatory requirements validation"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.validate_regulatory_requirements(post_data)
        
        # Assert
        assert result is True
        mock_compliance_service.validate_regulatory_requirements.assert_called_once_with(post_data)
    
    async def test_content_screening(self, post_service, mock_compliance_service):
        """Test content screening"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.screen_content(post_data)
        
        # Assert
        assert result["screening_status"] == "passed"
        assert result["risk_level"] == "low"
        assert len(result["flagged_issues"]) == 0
        mock_compliance_service.screen_content.assert_called_once_with(post_data)
    
    async def test_governance_policies_application(self, post_service, mock_compliance_service):
        """Test governance policies application"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.apply_governance_policies(post_data)
        
        # Assert
        assert result["policies_applied"] == 2
        assert result["violations"] == 0
        assert result["warnings"] == 1
        mock_compliance_service.apply_governance_policies.assert_called_once_with(post_data)
    
    async def test_governance_policies_retrieval(self, post_service, mock_compliance_service):
        """Test governance policies retrieval"""
        # Arrange
        
        # Act
        result = await post_service.get_governance_policies()
        
        # Assert
        assert len(result) == 1
        assert result[0]["policy_id"] == "policy-001"
        assert result[0]["name"] == "Content Standards Policy"
        mock_compliance_service.get_governance_policies.assert_called_once()
    
    async def test_audit_trail_creation(self, post_service, mock_compliance_service):
        """Test audit trail creation"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        action = "compliance_check"
        user_id = "user-123"
        
        # Act
        result = await post_service.create_audit_trail(post_data, action, user_id)
        
        # Assert
        assert result == SAMPLE_AUDIT_TRAIL
        assert result["audit_id"] == "audit-123"
        assert result["action"] == action
        assert result["user_id"] == user_id
        mock_compliance_service.create_audit_trail.assert_called_once_with(post_data, action, user_id)
    
    async def test_audit_history_retrieval(self, post_service, mock_compliance_service):
        """Test audit history retrieval"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        result = await post_service.get_audit_history(post_id)
        
        # Assert
        assert len(result) == 1
        assert result[0]["post_id"] == post_id
        mock_compliance_service.get_audit_history.assert_called_once_with(post_id)
    
    async def test_compliance_check_persistence(self, post_service, mock_compliance_repository):
        """Test compliance check persistence"""
        # Arrange
        post_id = "test-post-123"
        compliance_check = SAMPLE_COMPLIANCE_CHECK.copy()
        
        # Act
        result = await post_service.save_compliance_check(post_id, compliance_check)
        
        # Assert
        assert result == "check-123"
        mock_compliance_repository.save_compliance_check.assert_called_once_with(post_id, compliance_check)
    
    async def test_compliance_check_retrieval(self, post_service, mock_compliance_repository):
        """Test compliance check retrieval"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        result = await post_service.get_compliance_check(post_id)
        
        # Assert
        assert result == SAMPLE_COMPLIANCE_CHECK
        mock_compliance_repository.get_compliance_check.assert_called_once_with(post_id)
    
    async def test_audit_trail_persistence(self, post_service, mock_compliance_repository):
        """Test audit trail persistence"""
        # Arrange
        audit_trail = SAMPLE_AUDIT_TRAIL.copy()
        
        # Act
        result = await post_service.save_audit_trail(audit_trail)
        
        # Assert
        assert result == "audit-123"
        mock_compliance_repository.save_audit_trail.assert_called_once_with(audit_trail)
    
    async def test_audit_history_persistence(self, post_service, mock_compliance_repository):
        """Test audit history persistence"""
        # Arrange
        post_id = "test-post-123"
        audit_history = [SAMPLE_AUDIT_TRAIL.copy()]
        
        # Act
        result = await post_service.save_audit_history(post_id, audit_history)
        
        # Assert
        assert result is True
        mock_compliance_repository.save_audit_history.assert_called_once_with(post_id, audit_history)
    
    async def test_policy_compliance_validation(self, post_service, mock_governance_service):
        """Test policy compliance validation"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.validate_policy_compliance(post_data)
        
        # Assert
        assert result["compliant"] is True
        assert len(result["violations"]) == 0
        assert len(result["warnings"]) == 1
        mock_governance_service.validate_policy_compliance.assert_called_once_with(post_data)
    
    async def test_active_policies_retrieval(self, post_service, mock_governance_service):
        """Test active policies retrieval"""
        # Arrange
        
        # Act
        result = await post_service.get_active_policies()
        
        # Assert
        assert len(result) == 1
        assert result[0]["policy_id"] == "policy-001"
        mock_governance_service.get_active_policies.assert_called_once()
    
    async def test_policy_update(self, post_service, mock_governance_service):
        """Test policy update"""
        # Arrange
        policy_id = "policy-001"
        policy_update = {
            "name": "Updated Content Standards Policy",
            "version": "1.3"
        }
        
        # Act
        result = await post_service.update_policy(policy_id, policy_update)
        
        # Assert
        assert result is True
        mock_governance_service.update_policy.assert_called_once_with(policy_id, policy_update)
    
    async def test_compliance_score_calculation(self, post_service, mock_compliance_service):
        """Test compliance score calculation"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.calculate_compliance_score(post_data)
        
        # Assert
        assert result == 95.2
        mock_compliance_service.calculate_compliance_score.assert_called_once_with(post_data)
    
    async def test_compliance_violations_detection(self, post_service, mock_compliance_service):
        """Test compliance violations detection"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.detect_compliance_violations(post_data)
        
        # Assert
        assert len(result) == 0  # No violations in compliant post
        mock_compliance_service.detect_compliance_violations.assert_called_once_with(post_data)
    
    async def test_compliance_warnings_generation(self, post_service, mock_compliance_service):
        """Test compliance warnings generation"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.generate_compliance_warnings(post_data)
        
        # Assert
        assert len(result) == 1
        assert "disclaimer" in result[0].lower()
        mock_compliance_service.generate_compliance_warnings.assert_called_once_with(post_data)
    
    async def test_compliance_report_generation(self, post_service, mock_compliance_service):
        """Test compliance report generation"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.generate_compliance_report(post_data)
        
        # Assert
        assert "compliance_status" in result
        assert "compliance_score" in result
        assert "violations" in result
        assert "warnings" in result
        mock_compliance_service.generate_compliance_report.assert_called_once_with(post_data)
    
    async def test_compliance_history_tracking(self, post_service, mock_compliance_service):
        """Test compliance history tracking"""
        # Arrange
        post_id = "test-post-123"
        
        # Act
        result = await post_service.track_compliance_history(post_id)
        
        # Assert
        assert "tracking_id" in result
        assert "status" in result
        mock_compliance_service.track_compliance_history.assert_called_once_with(post_id)
    
    async def test_compliance_metrics_analysis(self, post_service, mock_compliance_service):
        """Test compliance metrics analysis"""
        # Arrange
        post_data = SAMPLE_POST_DATA.copy()
        
        # Act
        result = await post_service.analyze_compliance_metrics(post_data)
        
        # Assert
        assert "compliance_rate" in result
        assert "violation_rate" in result
        assert "average_score" in result
        mock_compliance_service.analyze_compliance_metrics.assert_called_once_with(post_data)
