"""
Enhanced Comprehensive Unit Tests for Core Classes
Improved with fixtures, parametrization, and more intuitive test cases
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch

from core.addiction_analyzer import AddictionAnalyzer, AddictionAssessment
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention


# ============================================================================
# AddictionAnalyzer Enhanced Tests
# ============================================================================

class TestAddictionAnalyzerEnhanced:
    """Enhanced test suite for AddictionAnalyzer with fixtures and parametrization"""
    
    def test_initialization_without_dependencies(self, addiction_analyzer):
        """Verify analyzer initializes correctly without external dependencies"""
        assert addiction_analyzer is not None
        assert addiction_analyzer.openai_client is None
    
    def test_initialization_with_ai_client(self, addiction_analyzer_with_ai):
        """Verify analyzer initializes correctly with AI client"""
        assert addiction_analyzer_with_ai.openai_client is not None
    
    @pytest.mark.parametrize("addiction_type,severity,frequency,expected_risk", [
        ("cigarrillos", "leve", "ocasional", "bajo"),
        ("cigarrillos", "moderada", "diaria", "medio"),
        ("alcohol", "severa", "diaria", "alto"),
        ("drogas", "crítica", "diaria", "alto"),
    ])
    def test_assessment_risk_levels_by_severity(
        self, addiction_analyzer, addiction_type, severity, frequency, expected_risk
    ):
        """Test that risk levels are calculated correctly based on severity"""
        data = {
            "addiction_type": addiction_type,
            "severity": severity,
            "frequency": frequency
        }
        result = addiction_analyzer.assess_addiction(data)
        
        assert result["success"] is True
        assert result["risk_level"] == expected_risk
    
    def test_assessment_with_minimal_required_fields(self, addiction_analyzer, minimal_assessment_data):
        """Test assessment succeeds with only required fields"""
        result = addiction_analyzer.assess_addiction(minimal_assessment_data)
        
        assert result["success"] is True
        assert result["addiction_type"] == minimal_assessment_data["addiction_type"]
        assert "risk_level" in result
        assert "recommended_approach" in result
        assert isinstance(result["key_insights"], list)
        assert len(result["key_insights"]) > 0
    
    def test_assessment_with_complete_information(self, addiction_analyzer, complete_assessment_data):
        """Test assessment with all available information provides comprehensive analysis"""
        result = addiction_analyzer.assess_addiction(complete_assessment_data)
        
        assert result["success"] is True
        assert result["addiction_type"] == complete_assessment_data["addiction_type"]
        assert len(result["key_insights"]) > 0
        assert len(result["immediate_actions"]) > 0
        assert len(result["long_term_goals"]) > 0
        assert any("ahorrar" in goal.lower() or "$" in goal for goal in result["long_term_goals"])
    
    def test_assessment_handles_missing_required_fields(self, addiction_analyzer, empty_assessment_data):
        """Test assessment gracefully handles missing required fields"""
        result = addiction_analyzer.assess_addiction(empty_assessment_data)
        
        assert result["success"] is False
        assert "error" in result
        assert "validando" in result["error"].lower() or "validation" in result["error"].lower()
    
    def test_assessment_handles_invalid_data_types(self, addiction_analyzer, invalid_assessment_data):
        """Test assessment handles invalid data types gracefully"""
        result = addiction_analyzer.assess_addiction(invalid_assessment_data)
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.parametrize("severity,expected_approach_contains", [
        ("crítica", "supervisión médica"),
        ("severa", "supervisión médica"),
        ("moderada", "abstinencia"),
        ("leve", "abstinencia"),
    ])
    def test_recommended_approach_by_severity(
        self, addiction_analyzer, severity, expected_approach_contains
    ):
        """Test that recommended approach is appropriate for severity level"""
        data = {
            "addiction_type": "cigarrillos",
            "severity": severity,
            "frequency": "diaria"
        }
        result = addiction_analyzer.assess_addiction(data)
        
        assert expected_approach_contains.lower() in result["recommended_approach"].lower()
    
    def test_insights_generated_for_triggers(self, addiction_analyzer):
        """Test that insights are generated when triggers are present"""
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "triggers": ["estrés", "social", "trabajo"]
        }
        result = addiction_analyzer.assess_addiction(data)
        
        insights_text = " ".join(result["key_insights"]).lower()
        assert "trigger" in insights_text or "estrés" in insights_text or "social" in insights_text
    
    def test_insights_generated_for_previous_attempts(self, addiction_analyzer):
        """Test that insights acknowledge previous attempts positively"""
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "previous_attempts": 3
        }
        result = addiction_analyzer.assess_addiction(data)
        
        insights_text = " ".join(result["key_insights"]).lower()
        assert "intentado" in insights_text or "vez" in insights_text or "attempt" in insights_text
    
    def test_insights_for_missing_support_system(self, addiction_analyzer):
        """Test that insights recommend support when none is present"""
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "support_system": False
        }
        result = addiction_analyzer.assess_addiction(data)
        
        insights_text = " ".join(result["key_insights"]).lower()
        assert "apoyo" in insights_text or "support" in insights_text
    
    def test_immediate_actions_for_severe_cases(self, addiction_analyzer):
        """Test that severe cases get professional medical recommendations"""
        data = {
            "addiction_type": "drogas",
            "severity": "severa",
            "frequency": "diaria"
        }
        result = addiction_analyzer.assess_addiction(data)
        
        actions_text = " ".join(result["immediate_actions"]).lower()
        assert "profesional" in actions_text or "médica" in actions_text or "médico" in actions_text
    
    def test_long_term_goals_include_financial_savings(self, addiction_analyzer):
        """Test that long-term goals include financial savings when cost is provided"""
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "daily_cost": 15.0
        }
        result = addiction_analyzer.assess_addiction(data)
        
        goals_text = " ".join(result["long_term_goals"]).lower()
        assert "ahorrar" in goals_text or "$" in goals_text or "savings" in goals_text
    
    def test_ai_enhancement_when_client_available(self, addiction_analyzer_with_ai, complete_assessment_data):
        """Test that AI enhancement is applied when OpenAI client is available"""
        result = addiction_analyzer_with_ai.assess_addiction(complete_assessment_data)
        
        assert result["success"] is True
        # AI enhancement should add additional fields
        ai_fields = [key for key in result.keys() if "ai_" in key.lower()]
        assert len(ai_fields) > 0
    
    def test_ai_enhancement_handles_api_errors(self):
        """Test that AI enhancement handles API errors gracefully"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        analyzer = AddictionAnalyzer(openai_client=mock_client)
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria"
        }
        
        result = analyzer.assess_addiction(data)
        # Should still succeed with base analysis
        assert result["success"] is True


# ============================================================================
# RecoveryPlanner Enhanced Tests
# ============================================================================

class TestRecoveryPlannerEnhanced:
    """Enhanced test suite for RecoveryPlanner with fixtures and parametrization"""
    
    def test_planner_initialization_loads_strategies(self, recovery_planner):
        """Verify planner loads strategies on initialization"""
        assert recovery_planner.strategies_by_type is not None
        assert isinstance(recovery_planner.strategies_by_type, dict)
        assert len(recovery_planner.strategies_by_type) > 0
    
    @pytest.mark.parametrize("addiction_type", [
        "cigarrillos", "tabaco", "alcohol", "drogas"
    ])
    def test_plan_creation_for_all_addiction_types(
        self, recovery_planner, addiction_type, minimal_assessment_data
    ):
        """Test that plans can be created for all supported addiction types"""
        minimal_assessment_data["addiction_type"] = addiction_type
        plan = recovery_planner.create_plan(
            user_id="test_user",
            addiction_type=addiction_type,
            assessment_data=minimal_assessment_data
        )
        
        assert plan["addiction_type"] == addiction_type
        assert len(plan["strategies"]) > 0
        assert len(plan["goals"]) > 0
        assert len(plan["daily_tasks"]) > 0
    
    def test_plan_includes_all_required_sections(self, recovery_planner, complete_assessment_data):
        """Test that created plan includes all required sections"""
        plan = recovery_planner.create_plan(
            user_id="test_user",
            addiction_type=complete_assessment_data["addiction_type"],
            assessment_data=complete_assessment_data
        )
        
        required_sections = [
            "user_id", "addiction_type", "start_date", "approach",
            "goals", "milestones", "strategies", "daily_tasks",
            "weekly_tasks", "support_resources", "created_at", "updated_at"
        ]
        
        for section in required_sections:
            assert section in plan, f"Missing required section: {section}"
    
    def test_plan_approach_determined_by_severity(self, recovery_planner):
        """Test that plan approach is determined by severity"""
        severe_data = {"severity": "severa"}
        plan_severe = recovery_planner.create_plan(
            user_id="test", addiction_type="alcohol", assessment_data=severe_data
        )
        assert plan_severe["approach"] == "abstinencia_total"
        
        moderate_data = {"severity": "moderada"}
        plan_moderate = recovery_planner.create_plan(
            user_id="test", addiction_type="cigarrillos", assessment_data=moderate_data
        )
        assert plan_moderate["approach"] == "abstinencia_total"
    
    def test_plan_goals_include_financial_when_cost_provided(self, recovery_planner, complete_assessment_data):
        """Test that financial goals are included when daily cost is provided"""
        plan = recovery_planner.create_plan(
            user_id="test_user",
            addiction_type=complete_assessment_data["addiction_type"],
            assessment_data=complete_assessment_data
        )
        
        goal_ids = [goal.get("id", "") for goal in plan["goals"]]
        assert any("money" in goal_id or "financiero" in goal_id.lower() for goal_id in goal_ids)
    
    def test_plan_strategies_personalized_by_triggers(self, recovery_planner):
        """Test that strategies are personalized based on triggers"""
        data_with_stress = {
            "severity": "moderada",
            "triggers": ["estrés", "stress"]
        }
        plan = recovery_planner.create_plan(
            user_id="test", addiction_type="cigarrillos", assessment_data=data_with_stress
        )
        
        strategy_names = [s.get("name", "").lower() for s in plan["strategies"]]
        assert any("estrés" in name or "stress" in name for name in strategy_names)
    
    def test_plan_milestones_are_progressive(self, recovery_planner, minimal_assessment_data):
        """Test that milestones are progressive and meaningful"""
        plan = recovery_planner.create_plan(
            user_id="test", addiction_type="cigarrillos", assessment_data=minimal_assessment_data
        )
        
        milestones = plan["milestones"]
        assert len(milestones) > 0
        
        # Check that milestones are in ascending order
        days = [m.get("days", 0) for m in milestones]
        assert days == sorted(days)
        
        # Check that each milestone has required fields
        for milestone in milestones:
            assert "days" in milestone
            assert "title" in milestone
            assert "reward" in milestone
    
    def test_plan_daily_tasks_are_structured(self, recovery_planner, minimal_assessment_data):
        """Test that daily tasks are well-structured"""
        plan = recovery_planner.create_plan(
            user_id="test", addiction_type="cigarrillos", assessment_data=minimal_assessment_data
        )
        
        for task in plan["daily_tasks"]:
            assert "task" in task
            assert "time" in task
            assert "description" in task
            assert task["time"] in ["mañana", "tarde", "noche", "morning", "afternoon", "evening"]
    
    def test_plan_support_resources_are_comprehensive(self, recovery_planner, minimal_assessment_data):
        """Test that support resources cover multiple types"""
        plan = recovery_planner.create_plan(
            user_id="test", addiction_type="cigarrillos", assessment_data=minimal_assessment_data
        )
        
        resource_types = [r.get("type", "") for r in plan["support_resources"]]
        assert len(set(resource_types)) > 1  # Multiple types of resources


# ============================================================================
# ProgressTracker Enhanced Tests
# ============================================================================

class TestProgressTrackerEnhanced:
    """Enhanced test suite for ProgressTracker with fixtures and parametrization"""
    
    def test_log_entry_creates_complete_record(self, progress_tracker):
        """Test that log entry creates a complete, timestamped record"""
        entry = progress_tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level=5,
            triggers_encountered=["estrés"],
            consumed=False,
            notes="Feeling positive today"
        )
        
        assert entry["user_id"] == "test_user"
        assert entry["date"] == "2024-01-01"
        assert entry["mood"] == "bueno"
        assert entry["cravings_level"] == 5
        assert entry["triggers_encountered"] == ["estrés"]
        assert entry["consumed"] is False
        assert entry["notes"] == "Feeling positive today"
        assert "logged_at" in entry
    
    def test_progress_calculation_with_sober_entries(self, progress_tracker, sample_progress_entries):
        """Test progress calculation with all sober entries"""
        progress = progress_tracker.get_progress("test_user", entries=sample_progress_entries)
        
        assert progress["total_entries"] == len(sample_progress_entries)
        assert progress["days_without_consumption"] == len(sample_progress_entries)
        assert progress["success_rate"] == 100.0
        assert progress["days_sober"] > 0
    
    def test_progress_calculation_with_relapse(self, progress_tracker, progress_entries_with_relapse):
        """Test progress calculation accurately reflects relapse"""
        progress = progress_tracker.get_progress("test_user", entries=progress_entries_with_relapse)
        
        assert progress["total_entries"] == len(progress_entries_with_relapse)
        assert progress["days_without_consumption"] < len(progress_entries_with_relapse)
        assert progress["success_rate"] < 100.0
        assert progress["last_consumption_date"] is not None
    
    def test_progress_includes_milestone_information(self, progress_tracker, sample_progress_entries):
        """Test that progress includes current and next milestone information"""
        progress = progress_tracker.get_progress("test_user", entries=sample_progress_entries)
        
        assert "current_milestone" in progress or progress["current_milestone"] is None
        assert "next_milestone" in progress
        assert "milestone_message" in progress
    
    def test_stats_provide_comprehensive_analysis(self, progress_tracker, sample_progress_entries):
        """Test that stats provide comprehensive analysis"""
        stats = progress_tracker.get_stats("test_user", entries=sample_progress_entries)
        
        assert stats["total_days_tracked"] == len(sample_progress_entries)
        assert "day_of_week_analysis" in stats
        assert "trends" in stats
        assert "trigger_analysis" in stats
        assert "mood_analysis" in stats
    
    def test_timeline_creates_chronological_sequence(self, progress_tracker, progress_entries_with_relapse):
        """Test that timeline creates chronological sequence of events"""
        timeline = progress_tracker.get_timeline("test_user", entries=progress_entries_with_relapse)
        
        assert len(timeline) > 0
        assert any(e["type"] == "sober_day" for e in timeline)
        assert any(e["type"] == "consumption" for e in timeline)
        
        # Check chronological order
        dates = [e.get("date", "") for e in timeline if e.get("date")]
        if len(dates) > 1:
            assert dates == sorted(dates)
    
    def test_streak_calculation_accurate(self, progress_tracker, progress_entries_with_relapse):
        """Test that streak calculation is accurate"""
        progress = progress_tracker.get_progress("test_user", entries=progress_entries_with_relapse)
        
        assert "streak_days" in progress
        assert progress["streak_days"] >= 0
        # After relapse, streak should reset
        assert progress["streak_days"] <= 4  # 4 days after relapse
    
    @pytest.mark.parametrize("days_sober,expected_milestone", [
        (1, 1),
        (7, 7),
        (30, 30),
        (90, 90),
        (365, 365),
    ])
    def test_milestone_detection(self, progress_tracker, days_sober, expected_milestone):
        """Test that milestones are detected correctly"""
        # Create entries to reach milestone
        entries = []
        base_date = datetime.now() - timedelta(days=days_sober-1)
        for i in range(days_sober):
            entry = {
                "date": (base_date + timedelta(days=i)).isoformat(),
                "consumed": False,
                "cravings_level": 3,
                "triggers_encountered": []
            }
            entries.append(entry)
        
        progress = progress_tracker.get_progress("test_user", entries=entries)
        
        if days_sober in [1, 7, 30, 90, 365]:
            assert progress["current_milestone"] is not None
            assert progress["current_milestone"]["days"] <= days_sober


# ============================================================================
# RelapsePrevention Enhanced Tests
# ============================================================================

class TestRelapsePreventionEnhanced:
    """Enhanced test suite for RelapsePrevention with fixtures and parametrization"""
    
    def test_risk_assessment_low_risk_scenario(
        self, relapse_prevention, current_state_low_risk
    ):
        """Test risk assessment for low-risk scenario"""
        risk = relapse_prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=100,
            current_state=current_state_low_risk
        )
        
        assert risk["risk_level"] in ["bajo", "medio"]
        assert risk["risk_score"] < 50
        assert len(risk["warning_signs"]) == 0 or len(risk["warning_signs"]) < 2
        assert risk["emergency_plan"] is None
    
    def test_risk_assessment_high_risk_scenario(
        self, relapse_prevention, current_state_high_risk
    ):
        """Test risk assessment for high-risk scenario"""
        risk = relapse_prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=5,
            current_state=current_state_high_risk
        )
        
        assert risk["risk_level"] in ["alto", "crítico"]
        assert risk["risk_score"] > 50
        assert len(risk["warning_signs"]) > 0
        assert risk["emergency_plan"] is not None
    
    def test_risk_assessment_critical_risk_scenario(
        self, relapse_prevention, current_state_critical_risk
    ):
        """Test risk assessment for critical-risk scenario"""
        risk = relapse_prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=1,
            current_state=current_state_critical_risk
        )
        
        assert risk["risk_level"] == "crítico"
        assert risk["risk_score"] >= 75
        assert len(risk["warning_signs"]) >= 3
        assert risk["emergency_plan"] is not None
        assert len(risk["recommendations"]) > 0
    
    @pytest.mark.parametrize("situation,expected_strategies", [
        ("craving", "cravings"),
        ("ansiedad", "cravings"),
        ("estrés", "stress"),
        ("social", "social"),
    ])
    def test_coping_strategies_by_situation(
        self, relapse_prevention, situation, expected_strategies
    ):
        """Test that appropriate coping strategies are returned for situations"""
        strategies = relapse_prevention.get_coping_strategies(situation)
        
        assert len(strategies) > 0
        assert all("name" in s and "description" in s for s in strategies)
    
    def test_emergency_plan_structure(self, relapse_prevention, current_state_high_risk):
        """Test that emergency plan has proper structure"""
        plan = relapse_prevention.generate_emergency_plan("test_user", current_state_high_risk)
        
        assert plan["user_id"] == "test_user"
        assert "steps" in plan
        assert len(plan["steps"]) > 0
        assert all("step" in s and "action" in s and "description" in s for s in plan["steps"])
        assert "emergency_contacts" in plan
        assert "safe_places" in plan
        assert "distraction_activities" in plan
    
    def test_warning_signs_detection(self, relapse_prevention):
        """Test that warning signs are detected correctly"""
        state_with_warnings = {
            "stress_level": 9,
            "isolation": True,
            "negative_thinking": True,
            "triggers": ["estrés", "social", "trabajo"]
        }
        
        risk = relapse_prevention.check_relapse_risk("test_user", 30, state_with_warnings)
        
        assert len(risk["warning_signs"]) >= 3
        assert any("estrés" in sign.lower() or "stress" in sign.lower() for sign in risk["warning_signs"])
        assert any("aislamiento" in sign.lower() or "isolation" in sign.lower() for sign in risk["warning_signs"])


# ============================================================================
# Integration Tests
# ============================================================================

class TestCompleteWorkflowEnhanced:
    """Enhanced integration tests for complete workflows"""
    
    def test_complete_recovery_journey(self, complete_user_journey, addiction_analyzer, 
                                       recovery_planner, progress_tracker, relapse_prevention):
        """Test complete recovery journey from assessment to prevention"""
        # Step 1: Assessment
        assessment = addiction_analyzer.assess_addiction(complete_user_journey["assessment"])
        assert assessment["success"] is True
        
        # Step 2: Create plan
        plan = recovery_planner.create_plan(
            user_id=complete_user_journey["user_id"],
            addiction_type=assessment["addiction_type"],
            assessment_data=complete_user_journey["assessment"]
        )
        assert plan["user_id"] == complete_user_journey["user_id"]
        
        # Step 3: Track progress
        progress = progress_tracker.get_progress(
            complete_user_journey["user_id"],
            entries=complete_user_journey["entries"]
        )
        assert progress["days_without_consumption"] > 0
        
        # Step 4: Check relapse risk
        risk = relapse_prevention.check_relapse_risk(
            complete_user_journey["user_id"],
            days_sober=progress["days_sober"],
            current_state=complete_user_journey["current_state"]
        )
        assert risk["risk_level"] in ["bajo", "medio", "alto", "crítico"]


