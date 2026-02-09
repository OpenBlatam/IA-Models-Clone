"""
Tests for data transformers
Tests for request/response transformation functions
"""

import pytest
from datetime import datetime
from typing import Dict, Any


class TestAssessmentTransformers:
    """Tests for assessment transformers"""
    
    def test_transform_assessment_request_to_dict(self):
        """Test transforming assessment request to dictionary"""
        try:
            from api.routes.assessment.transformers import transform_assessment_request_to_dict
            from schemas.assessment import AssessmentRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        request = AssessmentRequest(
            addiction_type="smoking",
            severity="moderate",
            frequency="daily",
            user_id="user_123"
        )
        
        result = transform_assessment_request_to_dict(request)
        
        assert isinstance(result, dict)
        assert result["addiction_type"] == "smoking"
        assert result["severity"] == "moderate"
        assert result["frequency"] == "daily"
    
    def test_transform_analysis_to_response(self):
        """Test transforming analysis to response"""
        try:
            from api.routes.assessment.transformers import transform_analysis_to_response
            from schemas.assessment import AssessmentRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        analysis = {
            "severity": "moderate",
            "risk_score": 0.65,
            "recommendations": ["Seek help"]
        }
        
        request = AssessmentRequest(
            addiction_type="smoking",
            severity="moderate",
            frequency="daily"
        )
        
        response = transform_analysis_to_response(analysis, request)
        
        assert hasattr(response, "severity") or isinstance(response, dict)
        if isinstance(response, dict):
            assert "severity" in response
            assert "risk_score" in response


class TestProgressTransformers:
    """Tests for progress transformers"""
    
    def test_transform_log_entry_request_to_dict(self):
        """Test transforming log entry request to dictionary"""
        try:
            from api.routes.progress.transformers import transform_log_entry_request_to_dict
            from schemas.progress import LogEntryRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        request = LogEntryRequest(
            user_id="user_123",
            date=datetime.now().isoformat(),
            mood="good",
            cravings_level=5
        )
        
        result = transform_log_entry_request_to_dict(request)
        
        assert isinstance(result, dict)
        assert result["user_id"] == "user_123"
        assert result["mood"] == "good"
        assert result["cravings_level"] == 5
    
    def test_transform_entry_to_response(self):
        """Test transforming entry to response"""
        try:
            from api.routes.progress.transformers import transform_entry_to_response
        except ImportError:
            pytest.skip("Transformers not available")
        
        entry = {
            "entry_id": "entry_123",
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 5
        }
        
        response = transform_entry_to_response(entry)
        
        assert hasattr(response, "entry_id") or isinstance(response, dict)
        if isinstance(response, dict):
            assert "entry_id" in response
            assert response["entry_id"] == "entry_123"


class TestRelapseTransformers:
    """Tests for relapse transformers"""
    
    def test_transform_relapse_request_to_dict(self):
        """Test transforming relapse request to dictionary"""
        try:
            from api.routes.relapse.transformers import transform_relapse_request_to_dict
            from schemas.relapse import RelapseRiskRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        request = RelapseRiskRequest(
            user_id="user_123",
            current_mood="anxious",
            stress_level=7,
            triggers=["work"]
        )
        
        result = transform_relapse_request_to_dict(request)
        
        assert isinstance(result, dict)
        assert result["user_id"] == "user_123"
        assert result["stress_level"] == 7
        assert "triggers" in result


class TestSupportTransformers:
    """Tests for support transformers"""
    
    def test_transform_coaching_request_to_dict(self):
        """Test transforming coaching request to dictionary"""
        try:
            from api.routes.support.transformers import transform_coaching_request_to_dict
            from schemas.support import CoachingRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        request = CoachingRequest(
            user_id="user_123",
            context="Feeling stressed",
            current_situation="High stress at work"
        )
        
        result = transform_coaching_request_to_dict(request)
        
        assert isinstance(result, dict)
        assert result["user_id"] == "user_123"
        assert "context" in result


class TestTransformerEdgeCases:
    """Tests for transformer edge cases"""
    
    def test_transform_with_none_values(self):
        """Test transformers with None values"""
        try:
            from api.routes.assessment.transformers import transform_assessment_request_to_dict
            from schemas.assessment import AssessmentRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        # Create request with optional None values
        request = AssessmentRequest(
            addiction_type="smoking",
            severity="moderate",
            frequency="daily"
        )
        
        result = transform_assessment_request_to_dict(request)
        
        # Should handle None values gracefully
        assert isinstance(result, dict)
    
    def test_transform_with_empty_lists(self):
        """Test transformers with empty lists"""
        try:
            from api.routes.relapse.transformers import transform_relapse_request_to_dict
            from schemas.relapse import RelapseRiskRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        request = RelapseRiskRequest(
            user_id="user_123",
            current_mood="good",
            stress_level=3,
            triggers=[]  # Empty list
        )
        
        result = transform_relapse_request_to_dict(request)
        
        assert isinstance(result, dict)
        assert result["triggers"] == []
    
    def test_transform_preserves_data_types(self):
        """Test that transformers preserve data types"""
        try:
            from api.routes.progress.transformers import transform_log_entry_request_to_dict
            from schemas.progress import LogEntryRequest
        except ImportError:
            pytest.skip("Transformers not available")
        
        request = LogEntryRequest(
            user_id="user_123",
            date=datetime.now().isoformat(),
            mood="good",
            cravings_level=5,
            consumed=False
        )
        
        result = transform_log_entry_request_to_dict(request)
        
        assert isinstance(result["cravings_level"], int)
        assert isinstance(result["consumed"], bool)
        assert isinstance(result["user_id"], str)



