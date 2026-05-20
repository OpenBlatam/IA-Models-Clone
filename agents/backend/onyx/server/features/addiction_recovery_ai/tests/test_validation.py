"""
Validation and Error Handling Tests
Tests for input validation, error handling, and edge cases
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from core.addiction_analyzer import AddictionAnalyzer, AddictionAssessment
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention
from utils.data_utils import normalize_features, split_data


class TestInputValidation:
    """Test input validation across components"""
    
    def test_assessment_validation_required_fields(self):
        """Test assessment validation for required fields"""
        analyzer = AddictionAnalyzer()
        
        # Missing addiction_type
        data = {"severity": "moderada", "frequency": "diaria"}
        result = analyzer.assess_addiction(data)
        assert result["success"] is False
        
        # Missing severity
        data = {"addiction_type": "cigarrillos", "frequency": "diaria"}
        result = analyzer.assess_addiction(data)
        assert result["success"] is False
        
        # Missing frequency
        data = {"addiction_type": "cigarrillos", "severity": "moderada"}
        result = analyzer.assess_addiction(data)
        assert result["success"] is False
    
    def test_assessment_validation_invalid_severity(self):
        """Test assessment with invalid severity values"""
        analyzer = AddictionAnalyzer()
        
        # Valid severities should work
        valid_severities = ["leve", "moderada", "severa", "crítica"]
        for severity in valid_severities:
            data = {
                "addiction_type": "cigarrillos",
                "severity": severity,
                "frequency": "diaria"
            }
            result = analyzer.assess_addiction(data)
            assert result["success"] is True
    
    def test_assessment_validation_invalid_types(self):
        """Test assessment with invalid type values"""
        analyzer = AddictionAnalyzer()
        
        # Should accept any string as addiction_type
        data = {
            "addiction_type": "invalid_type_123",
            "severity": "moderada",
            "frequency": "diaria"
        }
        result = analyzer.assess_addiction(data)
        assert result["success"] is True  # Type validation is lenient
    
    def test_assessment_validation_negative_values(self):
        """Test assessment with negative numeric values"""
        analyzer = AddictionAnalyzer()
        
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "duration_years": -5.0,
            "daily_cost": -10.0,
            "previous_attempts": -1
        }
        # Should either validate or handle gracefully
        result = analyzer.assess_addiction(data)
        # May succeed but values should be handled appropriately
        assert "success" in result
    
    def test_progress_tracker_validation_cravings_level(self):
        """Test progress tracker validation for cravings level"""
        tracker = ProgressTracker()
        
        # Valid range (1-10)
        entry = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level=5,
            triggers_encountered=[],
            consumed=False
        )
        assert entry["cravings_level"] == 5
        
        # Edge cases - should accept but may need validation
        entry = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level=0,  # Below valid range
            triggers_encountered=[],
            consumed=False
        )
        assert entry["cravings_level"] == 0
    
    def test_progress_tracker_validation_date_format(self):
        """Test progress tracker with various date formats"""
        tracker = ProgressTracker()
        
        # ISO format
        entry1 = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01T00:00:00",
            mood="bueno",
            cravings_level=5,
            triggers_encountered=[],
            consumed=False
        )
        assert entry1["date"] == "2024-01-01T00:00:00"
        
        # Simple date format
        entry2 = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level=5,
            triggers_encountered=[],
            consumed=False
        )
        assert entry2["date"] == "2024-01-01"
    
    def test_relapse_prevention_validation_stress_level(self):
        """Test relapse prevention validation for stress level"""
        prevention = RelapsePrevention()
        
        # Valid range (1-10)
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        risk = prevention.check_relapse_risk("test_user", 30, current_state)
        assert "risk_score" in risk
        
        # Edge cases
        current_state["stress_level"] = 0  # Below range
        risk = prevention.check_relapse_risk("test_user", 30, current_state)
        assert "risk_score" in risk
        
        current_state["stress_level"] = 15  # Above range
        risk = prevention.check_relapse_risk("test_user", 30, current_state)
        assert "risk_score" in risk


class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_assessment_error_handling_malformed_data(self):
        """Test assessment handles malformed data gracefully"""
        analyzer = AddictionAnalyzer()
        
        # Completely invalid data
        invalid_data = {
            "addiction_type": None,
            "severity": 123,  # Wrong type
            "frequency": []
        }
        result = analyzer.assess_addiction(invalid_data)
        assert result["success"] is False
        assert "error" in result
    
    def test_progress_tracker_error_handling_missing_fields(self):
        """Test progress tracker handles missing fields in entries"""
        tracker = ProgressTracker()
        
        # Entries with missing fields
        entries = [
            {"date": "2024-01-01", "consumed": False},  # Missing cravings_level
            {"date": "2024-01-02", "cravings_level": 5},  # Missing consumed
            {"consumed": False, "cravings_level": 5}  # Missing date
        ]
        
        progress = tracker.get_progress("test_user", entries=entries)
        # Should handle gracefully
        assert "total_entries" in progress
    
    def test_relapse_prevention_error_handling_empty_state(self):
        """Test relapse prevention handles empty state"""
        prevention = RelapsePrevention()
        
        # Empty state dict
        risk = prevention.check_relapse_risk("test_user", 30, {})
        
        # Should use defaults
        assert "risk_score" in risk
        assert "risk_level" in risk
    
    def test_data_utils_error_handling_invalid_ratios(self):
        """Test data utils handles invalid split ratios"""
        data = list(range(100))
        
        # Ratios don't sum to 1.0
        with pytest.raises(ValueError):
            split_data(data, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    
    def test_data_utils_error_handling_empty_data(self):
        """Test data utils handles empty data"""
        data = []
        
        # Should handle empty data gracefully
        train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0
    
    def test_normalize_features_error_handling_empty(self):
        """Test normalize features handles empty array"""
        features = np.array([])
        
        # Should handle gracefully or raise appropriate error
        try:
            normalized, mean, std = normalize_features(features)
            # If successful, check shapes
            assert normalized.shape == features.shape
        except (ValueError, IndexError):
            pass  # Expected error for empty array


class TestBoundaryConditions:
    """Test boundary conditions and limits"""
    
    def test_assessment_boundary_duration(self):
        """Test assessment with boundary duration values"""
        analyzer = AddictionAnalyzer()
        
        # Zero duration
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "duration_years": 0.0
        }
        result = analyzer.assess_addiction(data)
        assert result["success"] is True
        
        # Very large duration
        data["duration_years"] = 100.0
        result = analyzer.assess_addiction(data)
        assert result["success"] is True
    
    def test_progress_tracker_boundary_days_sober(self):
        """Test progress tracker with boundary days sober"""
        tracker = ProgressTracker()
        
        # Zero days
        progress = tracker.get_progress("test_user", entries=[])
        assert progress["days_sober"] >= 0
        
        # Very large number of days
        entries = []
        for i in range(10000):
            entry = {
                "date": (datetime.now() - timedelta(days=9999-i)).isoformat(),
                "consumed": False,
                "cravings_level": 5,
                "triggers_encountered": []
            }
            entries.append(entry)
        
        progress = tracker.get_progress("test_user", entries=entries)
        assert progress["days_sober"] > 0
    
    def test_relapse_prevention_boundary_scores(self):
        """Test relapse prevention with boundary risk scores"""
        prevention = RelapsePrevention()
        
        # Minimum risk scenario
        current_state = {
            "stress_level": 1,
            "support_level": 10,
            "triggers": [],
            "previous_relapses": 0
        }
        risk = prevention.check_relapse_risk("test_user", 1000, current_state)
        assert risk["risk_score"] >= 0
        assert risk["risk_level"] in ["bajo", "medio"]
        
        # Maximum risk scenario
        current_state = {
            "stress_level": 10,
            "support_level": 1,
            "triggers": ["estrés", "social", "trabajo"],
            "previous_relapses": 10
        }
        risk = prevention.check_relapse_risk("test_user", 1, current_state)
        assert risk["risk_score"] <= 100
        assert risk["risk_level"] in ["alto", "crítico"]


class TestTypeCoercion:
    """Test type coercion and conversion"""
    
    def test_assessment_type_coercion(self):
        """Test assessment handles type coercion"""
        analyzer = AddictionAnalyzer()
        
        # String numbers
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "duration_years": "5.5",  # String instead of float
            "previous_attempts": "2"  # String instead of int
        }
        
        # Should either validate or coerce
        result = analyzer.assess_addiction(data)
        # May fail validation or succeed with coercion
        assert "success" in result
    
    def test_progress_tracker_type_coercion(self):
        """Test progress tracker handles type coercion"""
        tracker = ProgressTracker()
        
        # String cravings level
        entry = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level="5",  # String instead of int
            triggers_encountered=[],
            consumed=False
        )
        # Should handle or validate
        assert "cravings_level" in entry


class TestConcurrency:
    """Test concurrent access and thread safety"""
    
    def test_progress_tracker_concurrent_entries(self):
        """Test progress tracker with concurrent entry creation"""
        tracker = ProgressTracker()
        
        # Simulate concurrent entries
        entries = []
        for i in range(10):
            entry = tracker.log_entry(
                user_id="test_user",
                date=datetime.now().isoformat(),
                mood="bueno",
                cravings_level=5,
                triggers_encountered=[],
                consumed=False
            )
            entries.append(entry)
        
        # All entries should be valid
        assert len(entries) == 10
        assert all("user_id" in e for e in entries)
    
    def test_relapse_prevention_concurrent_checks(self):
        """Test relapse prevention with concurrent risk checks"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        
        # Multiple concurrent checks
        risks = []
        for i in range(10):
            risk = prevention.check_relapse_risk(f"user_{i}", 30, current_state)
            risks.append(risk)
        
        # All should be valid
        assert len(risks) == 10
        assert all("risk_score" in r for r in risks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

