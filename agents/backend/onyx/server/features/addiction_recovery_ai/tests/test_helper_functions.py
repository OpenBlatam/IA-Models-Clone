"""
Tests for Helper Functions and Utility Methods
Tests for internal helper functions and utility methods
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from core.addiction_analyzer import AddictionAnalyzer
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention


# ============================================================================
# Helper Function Tests for AddictionAnalyzer
# ============================================================================

class TestAddictionAnalyzerHelpers:
    """Tests for internal helper methods of AddictionAnalyzer"""
    
    @pytest.mark.parametrize("severity,frequency,duration,support,expected_risk", [
        ("crítica", "diaria", 15.0, False, "alto"),
        ("severa", "diaria", 10.0, False, "alto"),
        ("severa", "semanal", 5.0, True, "medio"),
        ("moderada", "diaria", 3.0, True, "medio"),
        ("moderada", "semanal", 2.0, True, "bajo"),
        ("leve", "ocasional", 1.0, True, "bajo"),
    ])
    def test_calculate_risk_level_comprehensive(
        self, addiction_analyzer, severity, frequency, duration, support, expected_risk
    ):
        """Test risk level calculation with various combinations"""
        from core.addiction_analyzer import AddictionAssessment
        
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity=severity,
            frequency=frequency,
            duration_years=duration,
            support_system=support
        )
        
        risk = addiction_analyzer._calculate_risk_level(assessment)
        assert risk == expected_risk
    
    def test_recommend_approach_all_severities(self, addiction_analyzer):
        """Test approach recommendation for all severity levels"""
        from core.addiction_analyzer import AddictionAssessment
        
        severities = ["leve", "moderada", "severa", "crítica"]
        for severity in severities:
            assessment = AddictionAssessment(
                addiction_type="cigarrillos",
                severity=severity,
                frequency="diaria"
            )
            approach = addiction_analyzer._recommend_approach(assessment)
            assert len(approach) > 0
            assert "abstinencia" in approach.lower() or "reducción" in approach.lower()
    
    def test_generate_insights_comprehensive(self, addiction_analyzer):
        """Test insight generation with all possible data"""
        from core.addiction_analyzer import AddictionAssessment
        
        assessment = AddictionAssessment(
            addiction_type="alcohol",
            severity="moderada",
            frequency="diaria",
            triggers=["estrés", "social", "trabajo"],
            motivations=["salud", "familia"],
            previous_attempts=3,
            support_system=False,
            medical_conditions=["ansiedad"]
        )
        
        insights = addiction_analyzer._generate_insights(assessment)
        
        assert len(insights) > 0
        # Should mention triggers
        insights_text = " ".join(insights).lower()
        assert "trigger" in insights_text or "estrés" in insights_text or "social" in insights_text
        # Should mention previous attempts
        assert "intentado" in insights_text or "vez" in insights_text or "attempt" in insights_text
        # Should mention support
        assert "apoyo" in insights_text or "support" in insights_text
    
    def test_get_immediate_actions_all_severities(self, addiction_analyzer):
        """Test immediate actions for all severity levels"""
        from core.addiction_analyzer import AddictionAssessment
        
        for severity in ["leve", "moderada", "severa", "crítica"]:
            assessment = AddictionAssessment(
                addiction_type="drogas",
                severity=severity,
                frequency="diaria"
            )
            actions = addiction_analyzer._get_immediate_actions(assessment)
            
            assert len(actions) > 0
            assert all(isinstance(action, str) for action in actions)
            
            if severity in ["severa", "crítica"]:
                actions_text = " ".join(actions).lower()
                assert "profesional" in actions_text or "médica" in actions_text or "médico" in actions_text


# ============================================================================
# Helper Function Tests for RecoveryPlanner
# ============================================================================

class TestRecoveryPlannerHelpers:
    """Tests for internal helper methods of RecoveryPlanner"""
    
    def test_determine_approach_logic(self, recovery_planner):
        """Test approach determination logic"""
        # Severe cases should get total abstinence
        severe_data = {"severity": "severa"}
        approach = recovery_planner._determine_approach(severe_data)
        assert approach == "abstinencia_total"
        
        # Critical cases should get total abstinence
        critical_data = {"severity": "crítica"}
        approach = recovery_planner._determine_approach(critical_data)
        assert approach == "abstinencia_total"
        
        # Moderate cases should also get total abstinence (default)
        moderate_data = {"severity": "moderada"}
        approach = recovery_planner._determine_approach(moderate_data)
        assert approach == "abstinencia_total"
    
    def test_create_goals_structure_and_content(self, recovery_planner):
        """Test goal creation structure and content"""
        goals = recovery_planner._create_goals("cigarrillos", {})
        
        assert len(goals) >= 3
        for goal in goals:
            assert "id" in goal
            assert "title" in goal
            assert "description" in goal
            assert "target_date" in goal
            assert "status" in goal
            assert goal["status"] == "pending"
        
        # Check first goal is for first week
        assert "semana" in goals[0]["title"].lower() or "week" in goals[0]["title"].lower()
    
    def test_create_goals_with_financial_goal(self, recovery_planner):
        """Test goal creation includes financial goal when cost provided"""
        assessment = {"daily_cost": 20.0}
        goals = recovery_planner._create_goals("cigarrillos", assessment)
        
        goal_ids = [g.get("id", "") for g in goals]
        assert any("money" in gid or "financiero" in gid.lower() for gid in goal_ids)
    
    def test_create_milestones_progressive(self, recovery_planner):
        """Test milestone creation is progressive"""
        milestones = recovery_planner._create_milestones()
        
        assert len(milestones) > 0
        days = [m["days"] for m in milestones]
        assert days == sorted(days)  # Should be in ascending order
        
        # Check milestone structure
        for milestone in milestones:
            assert "days" in milestone
            assert "title" in milestone
            assert "reward" in milestone
            assert milestone["days"] > 0
    
    def test_get_strategies_by_addiction_type(self, recovery_planner):
        """Test strategies are returned for different addiction types"""
        addiction_types = ["cigarrillos", "tabaco", "alcohol", "drogas"]
        
        for addiction_type in addiction_types:
            strategies = recovery_planner._get_strategies(addiction_type, {})
            assert len(strategies) > 0
            assert all("name" in s and "description" in s for s in strategies)
    
    def test_get_strategies_personalized_by_triggers(self, recovery_planner):
        """Test strategies are personalized based on triggers"""
        # With stress trigger
        data_with_stress = {"triggers": ["estrés", "stress"]}
        strategies = recovery_planner._get_strategies("cigarrillos", data_with_stress)
        
        strategy_names = " ".join([s.get("name", "").lower() for s in strategies])
        assert "estrés" in strategy_names or "stress" in strategy_names
        
        # With social trigger
        data_with_social = {"triggers": ["social", "amigos"]}
        strategies = recovery_planner._get_strategies("cigarrillos", data_with_social)
        
        strategy_names = " ".join([s.get("name", "").lower() for s in strategies])
        assert "social" in strategy_names or "amigos" in strategy_names


# ============================================================================
# Helper Function Tests for ProgressTracker
# ============================================================================

class TestProgressTrackerHelpers:
    """Tests for internal helper methods of ProgressTracker"""
    
    def test_get_last_consumption_date_with_consumption(self, progress_tracker):
        """Test getting last consumption date when consumption exists"""
        entries = [
            {"date": "2024-01-01", "consumed": False},
            {"date": "2024-01-02", "consumed": True},
            {"date": "2024-01-03", "consumed": False},
            {"date": "2024-01-04", "consumed": True},
        ]
        
        last_date = progress_tracker._get_last_consumption_date(entries)
        assert last_date is not None
        assert "2024-01-04" in last_date.isoformat()
    
    def test_get_last_consumption_date_no_consumption(self, progress_tracker):
        """Test getting last consumption date when no consumption"""
        entries = [
            {"date": "2024-01-01", "consumed": False},
            {"date": "2024-01-02", "consumed": False},
        ]
        
        last_date = progress_tracker._get_last_consumption_date(entries)
        assert last_date is None
    
    def test_calculate_average_cravings_various_scenarios(self, progress_tracker):
        """Test average cravings calculation in various scenarios"""
        # All same cravings
        entries1 = [{"cravings_level": 5} for _ in range(10)]
        avg1 = progress_tracker._calculate_average_cravings(entries1)
        assert avg1 == 5.0
        
        # Mixed cravings
        entries2 = [
            {"cravings_level": 3},
            {"cravings_level": 5},
            {"cravings_level": 7}
        ]
        avg2 = progress_tracker._calculate_average_cravings(entries2)
        assert avg2 == 5.0
        
        # Empty entries
        avg3 = progress_tracker._calculate_average_cravings([])
        assert avg3 == 0.0
    
    def test_get_most_common_triggers_ranking(self, progress_tracker):
        """Test most common triggers are ranked correctly"""
        entries = [
            {"triggers_encountered": ["estrés"]},
            {"triggers_encountered": ["estrés", "social"]},
            {"triggers_encountered": ["estrés"]},
            {"triggers_encountered": ["social"]},
            {"triggers_encountered": ["trabajo"]},
        ]
        
        triggers = progress_tracker._get_most_common_triggers(entries)
        
        assert len(triggers) > 0
        assert triggers[0]["trigger"] == "estrés"  # Most common
        assert triggers[0]["count"] == 3
        assert len(triggers) <= 5  # Top 5
    
    def test_get_current_milestone_all_milestones(self, progress_tracker):
        """Test current milestone detection for all milestone days"""
        milestones = [1, 7, 30, 90, 180, 365]
        
        for milestone in milestones:
            current = progress_tracker._get_current_milestone(milestone)
            assert current is not None
            assert current["days"] == milestone
            assert current["achieved"] is True
    
    def test_get_next_milestone_progression(self, progress_tracker):
        """Test next milestone progression"""
        # Before first milestone
        next_milestone = progress_tracker._get_next_milestone(0)
        assert next_milestone is not None
        assert next_milestone["days"] == 1
        assert next_milestone["achieved"] is False
        assert next_milestone["days_remaining"] == 1
        
        # Between milestones
        next_milestone = progress_tracker._get_next_milestone(15)
        assert next_milestone is not None
        assert next_milestone["days"] == 30
        assert next_milestone["days_remaining"] == 15
        
        # Past all milestones
        next_milestone = progress_tracker._get_next_milestone(1000)
        assert next_milestone is None
    
    def test_calculate_streak_various_patterns(self, progress_tracker):
        """Test streak calculation with various patterns"""
        # Perfect streak
        entries1 = [
            {"date": f"2024-01-{i:02d}", "consumed": False}
            for i in range(1, 8)
        ]
        streak1 = progress_tracker._calculate_streak(entries1)
        assert streak1 == 7
        
        # Streak broken
        entries2 = [
            {"date": "2024-01-01", "consumed": False},
            {"date": "2024-01-02", "consumed": False},
            {"date": "2024-01-03", "consumed": True},
            {"date": "2024-01-04", "consumed": False},
        ]
        streak2 = progress_tracker._calculate_streak(entries2)
        assert streak2 == 1  # Only last day


# ============================================================================
# Helper Function Tests for RelapsePrevention
# ============================================================================

class TestRelapsePreventionHelpers:
    """Tests for internal helper methods of RelapsePrevention"""
    
    def test_load_warning_signs_structure(self, relapse_prevention):
        """Test warning signs are loaded with proper structure"""
        warning_signs = relapse_prevention._load_warning_signs()
        
        assert "emotional" in warning_signs
        assert "behavioral" in warning_signs
        assert "environmental" in warning_signs
        
        # Check each category has signs
        for category in warning_signs.values():
            assert isinstance(category, list)
            assert len(category) > 0
    
    def test_load_coping_strategies_comprehensive(self, relapse_prevention):
        """Test coping strategies are loaded comprehensively"""
        strategies = relapse_prevention._load_coping_strategies()
        
        assert "general" in strategies
        assert "cravings" in strategies
        assert "stress" in strategies
        assert "social" in strategies
        
        # Check each category has strategies
        for category in strategies.values():
            assert isinstance(category, list)
            assert len(category) > 0
            # Check strategy structure
            for strategy in category:
                assert "name" in strategy
                assert "description" in strategy
                assert "duration" in strategy
    
    def test_detect_warning_signs_all_types(self, relapse_prevention):
        """Test all types of warning signs are detected"""
        state = {
            "stress_level": 9,  # High stress
            "isolation": True,  # Behavioral
            "negative_thinking": True,  # Emotional
            "romanticizing": True,  # Behavioral
            "skipping_support": True,  # Behavioral
            "triggers": ["estrés", "social", "trabajo"]  # Environmental
        }
        
        signs = relapse_prevention._detect_warning_signs(state)
        
        assert len(signs) >= 5
        assert "Alto nivel de estrés" in signs
        assert "Aislamiento social" in signs
        assert "Pensamientos negativos" in signs
        assert "Romantizar el consumo" in signs
        assert "Múltiples triggers presentes" in signs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

