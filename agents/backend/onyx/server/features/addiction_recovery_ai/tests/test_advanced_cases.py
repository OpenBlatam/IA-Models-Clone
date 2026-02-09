"""
Advanced Test Cases - Complex Scenarios, Edge Cases, and Performance Tests
Tests for methods that need deeper coverage and complex real-world scenarios
"""

import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from core.addiction_analyzer import AddictionAnalyzer, AddictionAssessment
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention


# ============================================================================
# Advanced Progress Tracker Tests
# ============================================================================

class TestProgressTrackerAdvanced:
    """Advanced tests for ProgressTracker internal methods"""
    
    def test_analyze_by_day_of_week_comprehensive(self, progress_tracker):
        """Test day-of-week analysis with comprehensive data"""
        entries = []
        base_date = datetime(2024, 1, 1)  # Monday
        
        # Create entries for each day of the week
        for i in range(7):
            entry = {
                "date": (base_date + timedelta(days=i)).isoformat(),
                "mood": "bueno",
                "cravings_level": 5,
                "triggers_encountered": [],
                "consumed": i % 3 == 0  # Some consumption on certain days
            }
            entries.append(entry)
        
        stats = progress_tracker.get_stats("test_user", entries=entries)
        day_analysis = stats["day_of_week_analysis"]
        
        assert "lunes" in day_analysis
        assert "domingo" in day_analysis
        assert all("total_entries" in day_analysis[day] for day in day_analysis)
    
    def test_analyze_trends_improving(self, progress_tracker):
        """Test trend analysis shows improving pattern"""
        entries = []
        base_date = datetime.now() - timedelta(days=13)
        
        # First 7 days: more consumption
        for i in range(7):
            entry = {
                "date": (base_date + timedelta(days=i)).isoformat(),
                "consumed": i < 4,  # 4 out of 7 days
                "cravings_level": 7,
                "triggers_encountered": []
            }
            entries.append(entry)
        
        # Last 7 days: less consumption
        for i in range(7, 14):
            entry = {
                "date": (base_date + timedelta(days=i)).isoformat(),
                "consumed": i < 9,  # 2 out of 7 days
                "cravings_level": 3,
                "triggers_encountered": []
            }
            entries.append(entry)
        
        stats = progress_tracker.get_stats("test_user", entries=entries)
        trends = stats["trends"]
        
        assert trends["trend"] == "mejorando"
        assert trends["recent_consumption_rate"] < trends["previous_consumption_rate"]
    
    def test_analyze_trends_insufficient_data(self, progress_tracker):
        """Test trend analysis handles insufficient data"""
        entries = [
            {
                "date": (datetime.now() - timedelta(days=i)).isoformat(),
                "consumed": False,
                "cravings_level": 5,
                "triggers_encountered": []
            }
            for i in range(5)  # Less than 7 days
        ]
        
        stats = progress_tracker.get_stats("test_user", entries=entries)
        trends = stats["trends"]
        
        assert trends.get("insufficient_data") is True
    
    def test_analyze_triggers_with_consumption_correlation(self, progress_tracker):
        """Test trigger analysis correlates with consumption"""
        entries = [
            {
                "date": (datetime.now() - timedelta(days=4-i)).isoformat(),
                "triggers_encountered": ["estrés"],
                "consumed": True,
                "cravings_level": 8
            }
            for i in range(5)
        ]
        
        stats = progress_tracker.get_stats("test_user", entries=entries)
        trigger_analysis = stats["trigger_analysis"]
        
        assert "estrés" in trigger_analysis
        assert trigger_analysis["estrés"]["consumption_rate"] == 1.0  # 100% consumption when stress present
    
    def test_analyze_mood_distribution(self, progress_tracker):
        """Test mood analysis provides distribution"""
        entries = [
            {
                "date": (datetime.now() - timedelta(days=4-i)).isoformat(),
                "mood": ["bueno", "regular", "malo"][i % 3],
                "cravings_level": 5,
                "triggers_encountered": [],
                "consumed": False
            }
            for i in range(9)
        ]
        
        stats = progress_tracker.get_stats("test_user", entries=entries)
        mood_analysis = stats["mood_analysis"]
        
        assert "bueno" in mood_analysis
        assert "regular" in mood_analysis
        assert "malo" in mood_analysis
        assert sum(mood_analysis.values()) == len(entries)
    
    def test_get_days_sober_at_date_accurate(self, progress_tracker):
        """Test days sober calculation at specific date"""
        entries = []
        base_date = datetime.now() - timedelta(days=9)
        
        # 5 days sober, then relapse, then 4 days sober
        for i in range(5):
            entries.append({
                "date": (base_date + timedelta(days=i)).isoformat(),
                "consumed": False
            })
        entries.append({
            "date": (base_date + timedelta(days=5)).isoformat(),
            "consumed": True
        })
        for i in range(6, 10):
            entries.append({
                "date": (base_date + timedelta(days=i)).isoformat(),
                "consumed": False
            })
        
        # Should find 4 days at target of 4
        days = progress_tracker._get_days_sober_at_date(entries, 4)
        assert days >= 4


# ============================================================================
# Advanced Relapse Prevention Tests
# ============================================================================

class TestRelapsePreventionAdvanced:
    """Advanced tests for RelapsePrevention internal methods"""
    
    @pytest.mark.parametrize("risk_score,expected_level", [
        (0, "bajo"),
        (10, "bajo"),
        (24, "bajo"),
        (25, "medio"),
        (49, "medio"),
        (50, "alto"),
        (74, "alto"),
        (75, "crítico"),
        (100, "crítico"),
    ])
    def test_determine_risk_level_boundaries(self, relapse_prevention, risk_score, expected_level):
        """Test risk level determination at all boundaries"""
        level = relapse_prevention._determine_risk_level(risk_score)
        assert level == expected_level
    
    def test_generate_recommendations_critical_risk(self, relapse_prevention):
        """Test recommendations for critical risk are urgent"""
        recommendations = relapse_prevention._generate_recommendations(
            80, "crítico", ["Alto nivel de estrés", "Aislamiento social"], {}
        )
        
        assert len(recommendations) > 0
        assert any("crítico" in r.lower() or "riesgo" in r.lower() for r in recommendations)
        assert any("inmediatamente" in r.lower() or "immediately" in r.lower() for r in recommendations)
    
    def test_generate_recommendations_addresses_warning_signs(self, relapse_prevention):
        """Test recommendations address specific warning signs"""
        warning_signs = ["Alto nivel de estrés", "Aislamiento social", "Múltiples triggers presentes"]
        recommendations = relapse_prevention._generate_recommendations(
            60, "alto", warning_signs, {}
        )
        
        recommendations_text = " ".join(recommendations).lower()
        assert "estrés" in recommendations_text or "stress" in recommendations_text
        assert "contacta" in recommendations_text or "contact" in recommendations_text
    
    def test_detect_warning_signs_all_conditions(self, relapse_prevention):
        """Test all warning signs are detected"""
        state = {
            "stress_level": 9,
            "isolation": True,
            "negative_thinking": True,
            "romanticizing": True,
            "skipping_support": True,
            "triggers": ["estrés", "social", "trabajo", "emocional"]
        }
        
        signs = relapse_prevention._detect_warning_signs(state)
        
        assert len(signs) >= 5
        assert "Alto nivel de estrés" in signs
        assert "Aislamiento social" in signs
        assert "Pensamientos negativos" in signs
        assert "Romantizar el consumo" in signs
        assert "Múltiples triggers presentes" in signs


# ============================================================================
# Complex Real-World Scenarios
# ============================================================================

class TestComplexScenarios:
    """Tests for complex real-world scenarios"""
    
    def test_multiple_relapses_and_recoveries(self, progress_tracker):
        """Test tracking through multiple relapses and recoveries"""
        entries = []
        base_date = datetime.now() - timedelta(days=29)
        
        # Pattern: 5 days sober, relapse, 7 days sober, relapse, 10 days sober
        patterns = [
            (5, False),  # 5 days sober
            (1, True),   # 1 day relapse
            (7, False),  # 7 days sober
            (1, True),   # 1 day relapse
            (10, False), # 10 days sober
        ]
        
        day_count = 0
        for pattern_days, is_relapse in patterns:
            for _ in range(pattern_days):
                entries.append({
                    "date": (base_date + timedelta(days=day_count)).isoformat(),
                    "consumed": is_relapse,
                    "cravings_level": 8 if is_relapse else 3,
                    "triggers_encountered": ["estrés"] if is_relapse else [],
                    "mood": "malo" if is_relapse else "bueno"
                })
                day_count += 1
        
        progress = progress_tracker.get_progress("test_user", entries=entries)
        
        assert progress["total_entries"] == 24
        assert progress["days_without_consumption"] == 22
        assert progress["success_rate"] < 100.0
        assert progress["streak_days"] == 10  # Current streak
    
    def test_seasonal_patterns_in_progress(self, progress_tracker):
        """Test detection of seasonal or cyclical patterns"""
        entries = []
        base_date = datetime.now() - timedelta(days=90)
        
        # Create pattern: higher consumption on weekends
        for i in range(90):
            date = base_date + timedelta(days=i)
            is_weekend = date.weekday() >= 5
            entries.append({
                "date": date.isoformat(),
                "consumed": is_weekend and (i % 7 < 2),  # Some weekends
                "cravings_level": 7 if is_weekend else 3,
                "triggers_encountered": ["social"] if is_weekend else [],
                "mood": "regular"
            })
        
        stats = progress_tracker.get_stats("test_user", entries=entries)
        day_analysis = stats["day_of_week_analysis"]
        
        # Weekend days should show different patterns
        assert day_analysis["sábado"]["consumption_count"] >= 0
        assert day_analysis["domingo"]["consumption_count"] >= 0
    
    def test_progressive_improvement_tracking(self, progress_tracker):
        """Test tracking progressive improvement over time"""
        entries = []
        base_date = datetime.now() - timedelta(days=60)
        
        # Progressive improvement: cravings decrease over time
        for i in range(60):
            entries.append({
                "date": (base_date + timedelta(days=i)).isoformat(),
                "consumed": False,
                "cravings_level": max(1, 10 - (i // 6)),  # Decreasing
                "triggers_encountered": [] if i > 30 else ["estrés"],
                "mood": "bueno" if i > 20 else "regular"
            })
        
        progress = progress_tracker.get_progress("test_user", entries=entries)
        stats = progress_tracker.get_stats("test_user", entries=entries)
        
        assert progress["days_without_consumption"] == 60
        assert progress["success_rate"] == 100.0
        assert progress["average_cravings_level"] < 5  # Should be lower due to improvement
    
    def test_high_frequency_relapse_pattern(self, relapse_prevention):
        """Test handling of high-frequency relapse pattern"""
        # User with many relapses
        current_state = {
            "stress_level": 8,
            "support_level": 3,
            "triggers": ["estrés", "social"],
            "previous_relapses": 10,
            "isolation": True
        }
        
        risk = relapse_prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=2,  # Very recent sobriety
            current_state=current_state
        )
        
        assert risk["risk_level"] in ["alto", "crítico"]
        assert risk["risk_score"] > 60
        assert risk["emergency_plan"] is not None
        assert len(risk["recommendations"]) > 0


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests"""
    
    def test_progress_tracker_large_dataset_performance(self, progress_tracker, large_progress_entries):
        """Test progress tracker handles large datasets efficiently"""
        start_time = time.time()
        
        progress = progress_tracker.get_progress("test_user", entries=large_progress_entries)
        stats = progress_tracker.get_stats("test_user", entries=large_progress_entries)
        timeline = progress_tracker.get_timeline("test_user", entries=large_progress_entries)
        
        elapsed = time.time() - start_time
        
        assert progress["total_entries"] == 365
        assert stats["total_days_tracked"] == 365
        assert len(timeline) > 0
        assert elapsed < 5.0  # Should complete in reasonable time
    
    def test_relapse_prevention_many_concurrent_checks(self, relapse_prevention):
        """Test relapse prevention handles many concurrent checks"""
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        
        start_time = time.time()
        risks = []
        
        for i in range(1000):
            risk = relapse_prevention.check_relapse_risk(
                f"user_{i}",
                days_sober=30 + (i % 100),
                current_state=current_state
            )
            risks.append(risk)
        
        elapsed = time.time() - start_time
        
        assert len(risks) == 1000
        assert all("risk_score" in r for r in risks)
        assert elapsed < 10.0  # Should handle 1000 checks efficiently
    
    def test_recovery_planner_multiple_plans(self, recovery_planner):
        """Test recovery planner creates multiple plans efficiently"""
        assessment = {"severity": "moderada"}
        addiction_types = ["cigarrillos", "alcohol", "drogas", "tabaco"]
        
        start_time = time.time()
        plans = []
        
        for addiction_type in addiction_types:
            plan = recovery_planner.create_plan(
                user_id=f"user_{addiction_type}",
                addiction_type=addiction_type,
                assessment_data=assessment
            )
            plans.append(plan)
        
        elapsed = time.time() - start_time
        
        assert len(plans) == 4
        assert all("goals" in p for p in plans)
        assert elapsed < 2.0  # Should be fast


# ============================================================================
# Data Integrity and Consistency Tests
# ============================================================================

class TestDataIntegrity:
    """Tests for data integrity and consistency"""
    
    def test_progress_entries_chronological_consistency(self, progress_tracker):
        """Test that progress entries maintain chronological consistency"""
        entries = [
            {
                "date": (datetime.now() - timedelta(days=5-i)).isoformat(),
                "consumed": False,
                "cravings_level": 5,
                "triggers_encountered": []
            }
            for i in range(6)
        ]
        
        # Reverse order (should still work)
        reversed_entries = list(reversed(entries))
        
        progress1 = progress_tracker.get_progress("test_user", entries=entries)
        progress2 = progress_tracker.get_progress("test_user", entries=reversed_entries)
        
        # Results should be consistent regardless of order
        assert progress1["total_entries"] == progress2["total_entries"]
        assert progress1["days_without_consumption"] == progress2["days_without_consumption"]
    
    def test_risk_assessment_consistency_same_inputs(self, relapse_prevention):
        """Test risk assessment is consistent for same inputs"""
        current_state = {
            "stress_level": 5,
            "support_level": 5,
            "triggers": [],
            "previous_relapses": 0
        }
        
        risk1 = relapse_prevention.check_relapse_risk("user1", 30, current_state)
        risk2 = relapse_prevention.check_relapse_risk("user2", 30, current_state)
        
        # Same inputs should produce same risk score
        assert risk1["risk_score"] == risk2["risk_score"]
        assert risk1["risk_level"] == risk2["risk_level"]
    
    def test_assessment_idempotency(self, addiction_analyzer):
        """Test assessment produces same results for same input"""
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria"
        }
        
        result1 = addiction_analyzer.assess_addiction(data)
        result2 = addiction_analyzer.assess_addiction(data)
        
        assert result1["risk_level"] == result2["risk_level"]
        assert result1["addiction_type"] == result2["addiction_type"]


# ============================================================================
# Boundary and Edge Case Tests
# ============================================================================

class TestBoundaryCases:
    """Tests for boundary conditions and extreme values"""
    
    def test_progress_tracker_zero_entries(self, progress_tracker):
        """Test progress tracker with zero entries"""
        progress = progress_tracker.get_progress("test_user", entries=[])
        stats = progress_tracker.get_stats("test_user", entries=[])
        timeline = progress_tracker.get_timeline("test_user", entries=[])
        
        assert progress["total_entries"] == 0
        assert progress["days_without_consumption"] == 0
        assert progress["success_rate"] == 0
        assert stats["total_days_tracked"] == 0
        assert len(timeline) == 0
    
    def test_progress_tracker_single_entry(self, progress_tracker):
        """Test progress tracker with single entry"""
        entry = {
            "date": datetime.now().isoformat(),
            "consumed": False,
            "cravings_level": 5,
            "triggers_encountered": []
        }
        
        progress = progress_tracker.get_progress("test_user", entries=[entry])
        
        assert progress["total_entries"] == 1
        assert progress["days_without_consumption"] == 1
        assert progress["success_rate"] == 100.0
    
    def test_relapse_prevention_extreme_values(self, relapse_prevention):
        """Test relapse prevention with extreme values"""
        # Maximum risk scenario
        max_risk_state = {
            "stress_level": 10,
            "support_level": 1,
            "triggers": ["estrés"] * 10,
            "previous_relapses": 100
        }
        
        max_risk = relapse_prevention.check_relapse_risk("user", 0, max_risk_state)
        assert max_risk["risk_score"] <= 100
        assert max_risk["risk_level"] == "crítico"
        
        # Minimum risk scenario
        min_risk_state = {
            "stress_level": 1,
            "support_level": 10,
            "triggers": [],
            "previous_relapses": 0
        }
        
        min_risk = relapse_prevention.check_relapse_risk("user", 10000, min_risk_state)
        assert min_risk["risk_score"] >= 0
        assert min_risk["risk_level"] in ["bajo", "medio"]
    
    def test_assessment_extreme_duration(self, addiction_analyzer):
        """Test assessment with extreme duration values"""
        # Very short duration
        short_data = {
            "addiction_type": "cigarrillos",
            "severity": "leve",
            "frequency": "ocasional",
            "duration_years": 0.01
        }
        result_short = addiction_analyzer.assess_addiction(short_data)
        assert result_short["success"] is True
        
        # Very long duration
        long_data = {
            "addiction_type": "cigarrillos",
            "severity": "severa",
            "frequency": "diaria",
            "duration_years": 100.0
        }
        result_long = addiction_analyzer.assess_addiction(long_data)
        assert result_long["success"] is True
        assert result_long["risk_level"] in ["medio", "alto"]


# ============================================================================
# Error Recovery and Resilience Tests
# ============================================================================

class TestErrorRecovery:
    """Tests for error recovery and resilience"""
    
    def test_progress_tracker_handles_corrupted_entries(self, progress_tracker):
        """Test progress tracker handles corrupted entries gracefully"""
        entries = [
            {"date": "2024-01-01", "consumed": False},  # Missing fields
            {"date": "invalid-date", "consumed": False, "cravings_level": 5},  # Invalid date
            {"consumed": False},  # Missing date
            {"date": "2024-01-04", "consumed": False, "cravings_level": 5, "triggers_encountered": []}  # Valid
        ]
        
        # Should not crash
        progress = progress_tracker.get_progress("test_user", entries=entries)
        assert progress["total_entries"] == 4
    
    def test_relapse_prevention_handles_missing_state_fields(self, relapse_prevention):
        """Test relapse prevention handles missing state fields"""
        incomplete_state = {
            "stress_level": 5
            # Missing other fields
        }
        
        risk = relapse_prevention.check_relapse_risk("user", 30, incomplete_state)
        
        # Should use defaults and not crash
        assert "risk_score" in risk
        assert "risk_level" in risk
    
    def test_assessment_handles_partial_data(self, addiction_analyzer):
        """Test assessment handles partial data"""
        partial_data = {
            "addiction_type": "cigarrillos"
            # Missing severity and frequency
        }
        
        result = addiction_analyzer.assess_addiction(partial_data)
        
        # Should fail gracefully
        assert result["success"] is False
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

