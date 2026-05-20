"""
Integration Tests and Edge Cases
Tests for complete workflows and edge cases
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch

from core.addiction_analyzer import AddictionAnalyzer
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention


class TestCompleteWorkflow:
    """Test complete recovery workflow from assessment to tracking"""
    
    def test_complete_recovery_workflow(self):
        """Test complete workflow: assessment -> plan -> tracking -> prevention"""
        # Step 1: Assessment
        analyzer = AddictionAnalyzer()
        assessment_data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "duration_years": 5.0,
            "daily_cost": 10.0,
            "triggers": ["estrés", "social"],
            "motivations": ["salud", "economía"],
            "previous_attempts": 1,
            "support_system": True
        }
        assessment = analyzer.assess_addiction(assessment_data)
        assert assessment["success"] is True
        
        # Step 2: Create recovery plan
        planner = RecoveryPlanner()
        plan = planner.create_plan(
            user_id="test_user",
            addiction_type=assessment["addiction_type"],
            assessment_data=assessment_data
        )
        assert plan["user_id"] == "test_user"
        assert len(plan["goals"]) > 0
        
        # Step 3: Track progress
        tracker = ProgressTracker()
        entries = []
        for i in range(7):
            entry = tracker.log_entry(
                user_id="test_user",
                date=(datetime.now() - timedelta(days=6-i)).isoformat(),
                mood="bueno" if i % 2 == 0 else "regular",
                cravings_level=5 - i,
                triggers_encountered=[] if i < 3 else ["estrés"],
                consumed=False
            )
            entries.append(entry)
        
        progress = tracker.get_progress("test_user", entries=entries)
        assert progress["days_without_consumption"] == 7
        assert progress["success_rate"] == 100.0
        
        # Step 4: Check relapse risk
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 3,
            "support_level": 8,
            "triggers": [],
            "previous_relapses": 0
        }
        risk = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=progress["days_sober"],
            current_state=current_state
        )
        assert risk["risk_level"] in ["bajo", "medio", "alto", "crítico"]
    
    def test_workflow_with_relapse(self):
        """Test workflow including a relapse"""
        tracker = ProgressTracker()
        entries = []
        
        # 5 days sober
        for i in range(5):
            entry = tracker.log_entry(
                user_id="test_user",
                date=(datetime.now() - timedelta(days=4-i)).isoformat(),
                mood="bueno",
                cravings_level=3,
                triggers_encountered=[],
                consumed=False
            )
            entries.append(entry)
        
        # Relapse on day 6
        entry = tracker.log_entry(
            user_id="test_user",
            date=(datetime.now() - timedelta(days=0)).isoformat(),
            mood="malo",
            cravings_level=9,
            triggers_encountered=["estrés", "social"],
            consumed=True,
            notes="Relapse occurred"
        )
        entries.append(entry)
        
        progress = tracker.get_progress("test_user", entries=entries)
        assert progress["days_without_consumption"] == 5
        assert progress["success_rate"] < 100.0
        
        # Check relapse risk after relapse
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 7,
            "support_level": 5,
            "triggers": ["estrés"],
            "previous_relapses": 1
        }
        risk = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=0,  # Just relapsed
            current_state=current_state
        )
        assert risk["risk_level"] in ["medio", "alto", "crítico"]


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_assessment_extreme_values(self):
        """Test assessment with extreme values"""
        analyzer = AddictionAnalyzer()
        
        # Very long duration
        data = {
            "addiction_type": "cigarrillos",
            "severity": "crítica",
            "frequency": "diaria",
            "duration_years": 50.0,
            "daily_cost": 1000.0
        }
        result = analyzer.assess_addiction(data)
        assert result["success"] is True
        assert result["risk_level"] in ["medio", "alto"]
    
    def test_progress_tracker_empty_history(self):
        """Test progress tracker with no history"""
        tracker = ProgressTracker()
        progress = tracker.get_progress("test_user", entries=[])
        
        assert progress["days_sober"] >= 0
        assert progress["total_entries"] == 0
        assert progress["success_rate"] == 0
    
    def test_progress_tracker_single_entry(self):
        """Test progress tracker with single entry"""
        tracker = ProgressTracker()
        entry = tracker.log_entry(
            user_id="test_user",
            date=datetime.now().isoformat(),
            mood="bueno",
            cravings_level=5,
            triggers_encountered=[],
            consumed=False
        )
        progress = tracker.get_progress("test_user", entries=[entry])
        
        assert progress["total_entries"] == 1
        assert progress["days_without_consumption"] == 1
    
    def test_relapse_prevention_very_early_days(self):
        """Test relapse prevention for very early days"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        risk = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=1,
            current_state=current_state
        )
        
        assert risk["risk_score"] > 0
        assert risk["risk_level"] in ["bajo", "medio", "alto", "crítico"]
    
    def test_relapse_prevention_long_sobriety(self):
        """Test relapse prevention after long sobriety"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 2,
            "support_level": 9,
            "triggers": [],
            "previous_relapses": 0
        }
        risk = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=1000,
            current_state=current_state
        )
        
        assert risk["risk_score"] < 50  # Should be lower risk
        assert risk["risk_level"] in ["bajo", "medio"]
    
    def test_recovery_plan_all_addiction_types(self):
        """Test recovery plan for all addiction types"""
        planner = RecoveryPlanner()
        assessment = {"severity": "moderada"}
        
        addiction_types = ["cigarrillos", "tabaco", "alcohol", "drogas"]
        for addiction_type in addiction_types:
            plan = planner.create_plan(
                user_id="test_user",
                addiction_type=addiction_type,
                assessment_data=assessment
            )
            assert plan["addiction_type"] == addiction_type
            assert len(plan["strategies"]) > 0
    
    def test_progress_tracker_many_entries(self):
        """Test progress tracker with many entries"""
        tracker = ProgressTracker()
        entries = []
        
        for i in range(365):
            entry = tracker.log_entry(
                user_id="test_user",
                date=(datetime.now() - timedelta(days=364-i)).isoformat(),
                mood="bueno" if i % 7 < 5 else "regular",
                cravings_level=max(1, 5 - (i // 30)),
                triggers_encountered=[] if i % 10 != 0 else ["estrés"],
                consumed=False if i < 360 else (i == 360)  # One relapse
            )
            entries.append(entry)
        
        progress = tracker.get_progress("test_user", entries=entries)
        stats = tracker.get_stats("test_user", entries=entries)
        
        assert progress["total_entries"] == 365
        assert stats["total_days_tracked"] == 365
        assert len(stats["day_of_week_analysis"]) > 0


class TestDataConsistency:
    """Test data consistency across components"""
    
    def test_assessment_plan_consistency(self):
        """Test consistency between assessment and plan"""
        analyzer = AddictionAnalyzer()
        planner = RecoveryPlanner()
        
        assessment_data = {
            "addiction_type": "alcohol",
            "severity": "severa",
            "frequency": "diaria"
        }
        
        assessment = analyzer.assess_addiction(assessment_data)
        plan = planner.create_plan(
            user_id="test_user",
            addiction_type=assessment_data["addiction_type"],
            assessment_data=assessment_data
        )
        
        assert assessment["addiction_type"] == plan["addiction_type"]
        assert plan["approach"] == "abstinencia_total"  # Severe cases
    
    def test_progress_tracking_consistency(self):
        """Test consistency in progress tracking"""
        tracker = ProgressTracker()
        
        # Create entries with known pattern
        entries = []
        for i in range(10):
            entry = tracker.log_entry(
                user_id="test_user",
                date=(datetime.now() - timedelta(days=9-i)).isoformat(),
                mood="bueno",
                cravings_level=5,
                triggers_encountered=["estrés"] if i % 3 == 0 else [],
                consumed=False
            )
            entries.append(entry)
        
        progress = tracker.get_progress("test_user", entries=entries)
        stats = tracker.get_stats("test_user", entries=entries)
        timeline = tracker.get_timeline("test_user", entries=entries)
        
        # Check consistency
        assert progress["total_entries"] == stats["total_days_tracked"]
        assert len(timeline) >= len(entries)  # Timeline may have milestones
    
    def test_relapse_risk_consistency(self):
        """Test consistency in relapse risk assessment"""
        prevention = RelapsePrevention()
        
        # Same state should give similar risk
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        
        risk1 = prevention.check_relapse_risk("user1", 30, current_state)
        risk2 = prevention.check_relapse_risk("user2", 30, current_state)
        
        # Risk scores should be similar (within reasonable range)
        assert abs(risk1["risk_score"] - risk2["risk_score"]) < 5


class TestErrorRecovery:
    """Test error recovery and graceful degradation"""
    
    def test_assessment_invalid_data_recovery(self):
        """Test assessment recovers from invalid data"""
        analyzer = AddictionAnalyzer()
        
        # Missing required fields
        invalid_data = {"addiction_type": "cigarrillos"}
        result = analyzer.assess_addiction(invalid_data)
        
        assert result["success"] is False
        assert "error" in result
    
    def test_progress_tracker_malformed_entries(self):
        """Test progress tracker handles malformed entries"""
        tracker = ProgressTracker()
        
        # Mix of valid and malformed entries
        entries = [
            {"date": "2024-01-01", "consumed": False, "cravings_level": 3},
            {"date": "invalid", "consumed": False},  # Missing fields
            {"date": "2024-01-03", "consumed": False, "cravings_level": 5}
        ]
        
        progress = tracker.get_progress("test_user", entries=entries)
        
        # Should handle gracefully
        assert progress["total_entries"] == 3
    
    def test_relapse_prevention_missing_fields(self):
        """Test relapse prevention with missing fields"""
        prevention = RelapsePrevention()
        
        # Missing some fields
        current_state = {
            "stress_level": 5
            # Missing support_level, triggers, etc.
        }
        
        risk = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=30,
            current_state=current_state
        )
        
        # Should use defaults
        assert "risk_score" in risk
        assert "risk_level" in risk


class TestPerformance:
    """Test performance with large datasets"""
    
    def test_progress_tracker_large_dataset(self):
        """Test progress tracker with large dataset"""
        tracker = ProgressTracker()
        entries = []
        
        # Create 1000 entries
        for i in range(1000):
            entry = tracker.log_entry(
                user_id="test_user",
                date=(datetime.now() - timedelta(days=999-i)).isoformat(),
                mood="bueno",
                cravings_level=5,
                triggers_encountered=[],
                consumed=False
            )
            entries.append(entry)
        
        # Should complete in reasonable time
        progress = tracker.get_progress("test_user", entries=entries)
        assert progress["total_entries"] == 1000
    
    def test_relapse_prevention_many_checks(self):
        """Test multiple relapse risk checks"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        
        # Perform many checks
        for i in range(100):
            risk = prevention.check_relapse_risk(
                user_id=f"user_{i}",
                days_sober=30 + i,
                current_state=current_state
            )
            assert "risk_score" in risk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


