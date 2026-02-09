"""
Comprehensive Unit Tests for Core Classes
Tests for AddictionAnalyzer, RecoveryPlanner, ProgressTracker, RelapsePrevention
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

# Import core classes
from core.addiction_analyzer import AddictionAnalyzer, AddictionAssessment
from core.recovery_planner import RecoveryPlanner, RecoveryPlan
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention


class TestAddictionAnalyzer:
    """Test suite for AddictionAnalyzer class"""
    
    def test_init_without_openai_client(self):
        """Test initializing analyzer without OpenAI client"""
        analyzer = AddictionAnalyzer()
        assert analyzer.openai_client is None
    
    def test_init_with_openai_client(self):
        """Test initializing analyzer with OpenAI client"""
        mock_client = Mock()
        analyzer = AddictionAnalyzer(openai_client=mock_client)
        assert analyzer.openai_client == mock_client
    
    def test_assess_addiction_minimal_data(self):
        """Test assessment with minimal required data"""
        analyzer = AddictionAnalyzer()
        data = {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria"
        }
        result = analyzer.assess_addiction(data)
        
        assert result["success"] is True
        assert result["addiction_type"] == "cigarrillos"
        assert result["severity"] == "moderada"
        assert "risk_level" in result
        assert "recommended_approach" in result
        assert "key_insights" in result
        assert isinstance(result["key_insights"], list)
    
    def test_assess_addiction_complete_data(self):
        """Test assessment with complete data"""
        analyzer = AddictionAnalyzer()
        data = {
            "addiction_type": "alcohol",
            "severity": "severa",
            "frequency": "diaria",
            "duration_years": 5.5,
            "daily_cost": 25.50,
            "triggers": ["estrés", "social", "trabajo"],
            "motivations": ["salud", "familia", "economía"],
            "previous_attempts": 2,
            "support_system": True,
            "medical_conditions": ["ansiedad"],
            "additional_info": "Test info"
        }
        result = analyzer.assess_addiction(data)
        
        assert result["success"] is True
        assert result["addiction_type"] == "alcohol"
        assert result["severity"] == "severa"
        assert len(result["key_insights"]) > 0
        assert len(result["immediate_actions"]) > 0
        assert len(result["long_term_goals"]) > 0
    
    def test_assess_addiction_invalid_data(self):
        """Test assessment with invalid data"""
        analyzer = AddictionAnalyzer()
        data = {
            "addiction_type": "cigarrillos"
            # Missing required fields
        }
        result = analyzer.assess_addiction(data)
        
        assert result["success"] is False
        assert "error" in result
    
    def test_calculate_risk_level_critical(self):
        """Test risk level calculation for critical severity"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="drogas",
            severity="crítica",
            frequency="diaria",
            duration_years=15.0,
            support_system=False
        )
        risk = analyzer._calculate_risk_level(assessment)
        assert risk == "alto"
    
    def test_calculate_risk_level_low(self):
        """Test risk level calculation for low severity"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="leve",
            frequency="ocasional",
            duration_years=1.0,
            support_system=True
        )
        risk = analyzer._calculate_risk_level(assessment)
        assert risk == "bajo"
    
    def test_calculate_risk_level_medium(self):
        """Test risk level calculation for medium severity"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="alcohol",
            severity="moderada",
            frequency="semanal",
            duration_years=3.0,
            support_system=True
        )
        risk = analyzer._calculate_risk_level(assessment)
        assert risk in ["bajo", "medio"]
    
    def test_recommend_approach_severe(self):
        """Test approach recommendation for severe cases"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="drogas",
            severity="severa",
            frequency="diaria"
        )
        approach = analyzer._recommend_approach(assessment)
        assert "supervisión médica" in approach.lower()
    
    def test_recommend_approach_moderate(self):
        """Test approach recommendation for moderate cases"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria"
        )
        approach = analyzer._recommend_approach(assessment)
        assert len(approach) > 0
    
    def test_generate_insights_with_triggers(self):
        """Test insight generation with triggers"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="alcohol",
            severity="moderada",
            frequency="diaria",
            triggers=["estrés", "social", "trabajo"]
        )
        insights = analyzer._generate_insights(assessment)
        assert len(insights) > 0
        assert any("trigger" in insight.lower() for insight in insights)
    
    def test_generate_insights_with_previous_attempts(self):
        """Test insight generation with previous attempts"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria",
            previous_attempts=3
        )
        insights = analyzer._generate_insights(assessment)
        assert any("intentado" in insight.lower() or "vez" in insight.lower() for insight in insights)
    
    def test_generate_insights_without_support(self):
        """Test insight generation without support system"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="alcohol",
            severity="moderada",
            frequency="diaria",
            support_system=False
        )
        insights = analyzer._generate_insights(assessment)
        assert any("apoyo" in insight.lower() for insight in insights)
    
    def test_get_immediate_actions_severe(self):
        """Test immediate actions for severe cases"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="drogas",
            severity="severa",
            frequency="diaria"
        )
        actions = analyzer._get_immediate_actions(assessment)
        assert len(actions) > 0
        assert any("profesional" in action.lower() or "médica" in action.lower() for action in actions)
    
    def test_get_long_term_goals_with_cost(self):
        """Test long-term goals with daily cost"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria",
            daily_cost=10.0
        )
        goals = analyzer._get_long_term_goals(assessment)
        assert len(goals) > 0
        assert any("ahorrar" in goal.lower() or "$" in goal for goal in goals)
    
    def test_enhance_with_ai_success(self):
        """Test AI enhancement with successful response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"psychological_analysis": "test", "personalized_strategies": [], "motivational_message": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        analyzer = AddictionAnalyzer(openai_client=mock_client)
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria"
        )
        base_analysis = {}
        
        result = analyzer._enhance_with_ai(assessment, base_analysis)
        assert "ai_psychological_analysis" in result or "ai_enhanced_insights" in result
    
    def test_enhance_with_ai_no_client(self):
        """Test AI enhancement without client"""
        analyzer = AddictionAnalyzer()
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria"
        )
        result = analyzer._enhance_with_ai(assessment, {})
        assert result == {}
    
    def test_enhance_with_ai_json_error(self):
        """Test AI enhancement with JSON decode error"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not valid JSON"
        mock_client.chat.completions.create.return_value = mock_response
        
        analyzer = AddictionAnalyzer(openai_client=mock_client)
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria"
        )
        
        result = analyzer._enhance_with_ai(assessment, {})
        assert "ai_enhanced_insights" in result or result == {}
    
    def test_enhance_with_ai_exception(self):
        """Test AI enhancement with exception"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        analyzer = AddictionAnalyzer(openai_client=mock_client)
        assessment = AddictionAssessment(
            addiction_type="cigarrillos",
            severity="moderada",
            frequency="diaria"
        )
        
        result = analyzer._enhance_with_ai(assessment, {})
        assert result == {}


class TestRecoveryPlanner:
    """Test suite for RecoveryPlanner class"""
    
    def test_init(self):
        """Test planner initialization"""
        planner = RecoveryPlanner()
        assert planner.strategies_by_type is not None
        assert isinstance(planner.strategies_by_type, dict)
    
    def test_create_plan_minimal(self):
        """Test creating plan with minimal data"""
        planner = RecoveryPlanner()
        assessment = {
            "severity": "moderada"
        }
        plan = planner.create_plan(
            user_id="test_user",
            addiction_type="cigarrillos",
            assessment_data=assessment
        )
        
        assert plan["user_id"] == "test_user"
        assert plan["addiction_type"] == "cigarrillos"
        assert plan["approach"] == "abstinencia_total"
        assert "goals" in plan
        assert "milestones" in plan
        assert "strategies" in plan
        assert "daily_tasks" in plan
        assert "weekly_tasks" in plan
        assert "support_resources" in plan
        assert "created_at" in plan
        assert "updated_at" in plan
    
    def test_create_plan_with_approach(self):
        """Test creating plan with specified approach"""
        planner = RecoveryPlanner()
        plan = planner.create_plan(
            user_id="test_user",
            addiction_type="alcohol",
            assessment_data={},
            approach="reduccion_gradual"
        )
        assert plan["approach"] == "reduccion_gradual"
    
    def test_create_plan_with_triggers(self):
        """Test creating plan with triggers"""
        planner = RecoveryPlanner()
        assessment = {
            "severity": "moderada",
            "triggers": ["estrés", "social"]
        }
        plan = planner.create_plan(
            user_id="test_user",
            addiction_type="cigarrillos",
            assessment_data=assessment
        )
        assert len(plan["strategies"]) > 0
    
    def test_create_plan_with_daily_cost(self):
        """Test creating plan with daily cost"""
        planner = RecoveryPlanner()
        assessment = {
            "severity": "moderada",
            "daily_cost": 15.0
        }
        plan = planner.create_plan(
            user_id="test_user",
            addiction_type="cigarrillos",
            assessment_data=assessment
        )
        assert any("goal_money" in goal.get("id", "") for goal in plan["goals"])
    
    def test_determine_approach_severe(self):
        """Test approach determination for severe cases"""
        planner = RecoveryPlanner()
        assessment = {"severity": "severa"}
        approach = planner._determine_approach(assessment)
        assert approach == "abstinencia_total"
    
    def test_determine_approach_critical(self):
        """Test approach determination for critical cases"""
        planner = RecoveryPlanner()
        assessment = {"severity": "crítica"}
        approach = planner._determine_approach(assessment)
        assert approach == "abstinencia_total"
    
    def test_determine_approach_moderate(self):
        """Test approach determination for moderate cases"""
        planner = RecoveryPlanner()
        assessment = {"severity": "moderada"}
        approach = planner._determine_approach(assessment)
        assert approach == "abstinencia_total"
    
    def test_create_goals_structure(self):
        """Test goals structure"""
        planner = RecoveryPlanner()
        goals = planner._create_goals("cigarrillos", {})
        assert len(goals) >= 3
        for goal in goals:
            assert "id" in goal
            assert "title" in goal
            assert "description" in goal
            assert "target_date" in goal
            assert "status" in goal
    
    def test_create_milestones(self):
        """Test milestones creation"""
        planner = RecoveryPlanner()
        milestones = planner._create_milestones()
        assert len(milestones) > 0
        assert all("days" in m and "title" in m and "reward" in m for m in milestones)
    
    def test_get_strategies_for_cigarettes(self):
        """Test strategies for cigarette addiction"""
        planner = RecoveryPlanner()
        strategies = planner._get_strategies("cigarrillos", {})
        assert len(strategies) > 0
    
    def test_get_strategies_for_alcohol(self):
        """Test strategies for alcohol addiction"""
        planner = RecoveryPlanner()
        strategies = planner._get_strategies("alcohol", {})
        assert len(strategies) > 0
    
    def test_get_strategies_with_stress_trigger(self):
        """Test strategies with stress trigger"""
        planner = RecoveryPlanner()
        assessment = {"triggers": ["estrés"]}
        strategies = planner._get_strategies("cigarrillos", assessment)
        assert any("estrés" in s.get("name", "").lower() or "stress" in s.get("name", "").lower() for s in strategies)
    
    def test_create_daily_tasks(self):
        """Test daily tasks creation"""
        planner = RecoveryPlanner()
        tasks = planner._create_daily_tasks("cigarrillos", "abstinencia_total")
        assert len(tasks) > 0
        assert all("task" in t and "time" in t and "description" in t for t in tasks)
    
    def test_create_daily_tasks_for_cigarettes(self):
        """Test daily tasks for cigarettes include exercise"""
        planner = RecoveryPlanner()
        tasks = planner._create_daily_tasks("cigarrillos", "abstinencia_total")
        assert any("ejercicio" in t.get("task", "").lower() or "exercise" in t.get("task", "").lower() for t in tasks)
    
    def test_create_weekly_tasks(self):
        """Test weekly tasks creation"""
        planner = RecoveryPlanner()
        tasks = planner._create_weekly_tasks("cigarrillos")
        assert len(tasks) > 0
        assert all("task" in t and "day" in t and "description" in t for t in tasks)
    
    def test_get_support_resources(self):
        """Test support resources"""
        planner = RecoveryPlanner()
        resources = planner._get_support_resources("cigarrillos")
        assert len(resources) > 0
        assert all("type" in r and "name" in r and "description" in r for r in resources)
    
    def test_load_strategies_structure(self):
        """Test strategies loading structure"""
        planner = RecoveryPlanner()
        strategies = planner._load_strategies()
        assert isinstance(strategies, dict)
        assert "cigarrillos" in strategies or "tabaco" in strategies


class TestProgressTracker:
    """Test suite for ProgressTracker class"""
    
    def test_init(self):
        """Test tracker initialization"""
        tracker = ProgressTracker()
        assert tracker is not None
    
    def test_log_entry_minimal(self):
        """Test logging entry with minimal data"""
        tracker = ProgressTracker()
        entry = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level=5,
            triggers_encountered=[],
            consumed=False
        )
        
        assert entry["user_id"] == "test_user"
        assert entry["date"] == "2024-01-01"
        assert entry["mood"] == "bueno"
        assert entry["cravings_level"] == 5
        assert entry["consumed"] is False
        assert "logged_at" in entry
    
    def test_log_entry_with_notes(self):
        """Test logging entry with notes"""
        tracker = ProgressTracker()
        entry = tracker.log_entry(
            user_id="test_user",
            date="2024-01-01",
            mood="bueno",
            cravings_level=3,
            triggers_encountered=["estrés"],
            consumed=False,
            notes="Feeling good today"
        )
        assert entry["notes"] == "Feeling good today"
        assert entry["triggers_encountered"] == ["estrés"]
    
    def test_get_progress_empty_entries(self):
        """Test getting progress with empty entries"""
        tracker = ProgressTracker()
        progress = tracker.get_progress("test_user", entries=[])
        
        assert progress["user_id"] == "test_user"
        assert progress["days_sober"] >= 0
        assert progress["total_entries"] == 0
        assert progress["days_without_consumption"] == 0
        assert progress["success_rate"] == 0
    
    def test_get_progress_with_entries(self):
        """Test getting progress with entries"""
        tracker = ProgressTracker()
        entries = [
            {
                "date": "2024-01-01",
                "consumed": False,
                "cravings_level": 3,
                "triggers_encountered": []
            },
            {
                "date": "2024-01-02",
                "consumed": False,
                "cravings_level": 2,
                "triggers_encountered": []
            }
        ]
        progress = tracker.get_progress("test_user", entries=entries)
        
        assert progress["total_entries"] == 2
        assert progress["days_without_consumption"] == 2
        assert progress["success_rate"] == 100.0
    
    def test_get_progress_with_consumption(self):
        """Test getting progress with consumption entries"""
        tracker = ProgressTracker()
        entries = [
            {"date": "2024-01-01", "consumed": False, "cravings_level": 3, "triggers_encountered": []},
            {"date": "2024-01-02", "consumed": True, "cravings_level": 8, "triggers_encountered": ["estrés"]},
            {"date": "2024-01-03", "consumed": False, "cravings_level": 2, "triggers_encountered": []}
        ]
        progress = tracker.get_progress("test_user", entries=entries)
        
        assert progress["total_entries"] == 3
        assert progress["days_without_consumption"] == 2
        assert progress["success_rate"] < 100.0
    
    def test_get_stats_empty(self):
        """Test getting stats with empty entries"""
        tracker = ProgressTracker()
        stats = tracker.get_stats("test_user", entries=[])
        assert stats["user_id"] == "test_user"
        assert stats["total_days_tracked"] == 0
    
    def test_get_stats_with_entries(self):
        """Test getting stats with entries"""
        tracker = ProgressTracker()
        entries = [
            {"date": "2024-01-01", "mood": "bueno", "consumed": False, "cravings_level": 3, "triggers_encountered": ["estrés"]},
            {"date": "2024-01-02", "mood": "regular", "consumed": False, "cravings_level": 2, "triggers_encountered": []}
        ]
        stats = tracker.get_stats("test_user", entries=entries)
        
        assert stats["total_days_tracked"] == 2
        assert "day_of_week_analysis" in stats
        assert "trends" in stats
        assert "trigger_analysis" in stats
        assert "mood_analysis" in stats
    
    def test_get_timeline_empty(self):
        """Test getting timeline with empty entries"""
        tracker = ProgressTracker()
        timeline = tracker.get_timeline("test_user", entries=[])
        assert isinstance(timeline, list)
        assert len(timeline) == 0
    
    def test_get_timeline_with_entries(self):
        """Test getting timeline with entries"""
        tracker = ProgressTracker()
        entries = [
            {"date": "2024-01-01", "mood": "bueno", "consumed": False, "cravings_level": 3, "notes": "Good day"},
            {"date": "2024-01-02", "mood": "regular", "consumed": True, "cravings_level": 8, "notes": "Relapse"}
        ]
        timeline = tracker.get_timeline("test_user", entries=entries)
        
        assert len(timeline) > 0
        assert any(e["type"] == "sober_day" for e in timeline)
        assert any(e["type"] == "consumption" for e in timeline)
    
    def test_calculate_average_cravings(self):
        """Test average cravings calculation"""
        tracker = ProgressTracker()
        entries = [
            {"cravings_level": 5},
            {"cravings_level": 3},
            {"cravings_level": 7}
        ]
        avg = tracker._calculate_average_cravings(entries)
        assert avg == 5.0
    
    def test_calculate_average_cravings_empty(self):
        """Test average cravings with empty entries"""
        tracker = ProgressTracker()
        avg = tracker._calculate_average_cravings([])
        assert avg == 0.0
    
    def test_get_most_common_triggers(self):
        """Test getting most common triggers"""
        tracker = ProgressTracker()
        entries = [
            {"triggers_encountered": ["estrés", "trabajo"]},
            {"triggers_encountered": ["estrés"]},
            {"triggers_encountered": ["social"]}
        ]
        triggers = tracker._get_most_common_triggers(entries)
        assert len(triggers) > 0
        assert triggers[0]["trigger"] == "estrés"
        assert triggers[0]["count"] == 2
    
    def test_calculate_streak_no_consumption(self):
        """Test streak calculation with no consumption"""
        tracker = ProgressTracker()
        entries = [
            {"date": "2024-01-01", "consumed": False},
            {"date": "2024-01-02", "consumed": False},
            {"date": "2024-01-03", "consumed": False}
        ]
        streak = tracker._calculate_streak(entries)
        assert streak == 3
    
    def test_calculate_streak_with_consumption(self):
        """Test streak calculation with consumption"""
        tracker = ProgressTracker()
        entries = [
            {"date": "2024-01-01", "consumed": False},
            {"date": "2024-01-02", "consumed": True},
            {"date": "2024-01-03", "consumed": False}
        ]
        streak = tracker._calculate_streak(entries)
        assert streak == 1  # Only the most recent entry


class TestRelapsePrevention:
    """Test suite for RelapsePrevention class"""
    
    def test_init(self):
        """Test prevention system initialization"""
        prevention = RelapsePrevention()
        assert prevention.warning_signs is not None
        assert prevention.coping_strategies is not None
    
    def test_check_relapse_risk_low(self):
        """Test relapse risk check with low risk"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 2,
            "support_level": 9,
            "triggers": [],
            "previous_relapses": 0
        }
        result = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=100,
            current_state=current_state
        )
        
        assert result["user_id"] == "test_user"
        assert "risk_score" in result
        assert "risk_level" in result
        assert result["risk_level"] in ["bajo", "medio", "alto", "crítico"]
        assert "warning_signs" in result
        assert "recommendations" in result
    
    def test_check_relapse_risk_high(self):
        """Test relapse risk check with high risk"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 9,
            "support_level": 2,
            "triggers": ["estrés", "social", "trabajo"],
            "previous_relapses": 3,
            "isolation": True,
            "negative_thinking": True
        }
        result = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=5,
            current_state=current_state
        )
        
        assert result["risk_level"] in ["alto", "crítico"]
        assert len(result["warning_signs"]) > 0
        assert result["emergency_plan"] is not None
    
    def test_check_relapse_risk_critical(self):
        """Test relapse risk check with critical risk"""
        prevention = RelapsePrevention()
        current_state = {
            "stress_level": 10,
            "support_level": 1,
            "triggers": ["estrés", "social", "trabajo", "emocional"],
            "previous_relapses": 5
        }
        result = prevention.check_relapse_risk(
            user_id="test_user",
            days_sober=1,
            current_state=current_state
        )
        
        assert result["risk_level"] in ["alto", "crítico"]
        assert result["emergency_plan"] is not None
    
    def test_get_coping_strategies_general(self):
        """Test getting general coping strategies"""
        prevention = RelapsePrevention()
        strategies = prevention.get_coping_strategies("general")
        assert len(strategies) > 0
        assert all("name" in s and "description" in s for s in strategies)
    
    def test_get_coping_strategies_cravings(self):
        """Test getting coping strategies for cravings"""
        prevention = RelapsePrevention()
        strategies = prevention.get_coping_strategies("craving")
        assert len(strategies) > 0
    
    def test_get_coping_strategies_stress(self):
        """Test getting coping strategies for stress"""
        prevention = RelapsePrevention()
        strategies = prevention.get_coping_strategies("estrés")
        assert len(strategies) > 0
    
    def test_get_coping_strategies_social(self):
        """Test getting coping strategies for social situations"""
        prevention = RelapsePrevention()
        strategies = prevention.get_coping_strategies("social")
        assert len(strategies) > 0
    
    def test_get_coping_strategies_with_trigger_type(self):
        """Test getting coping strategies with trigger type"""
        prevention = RelapsePrevention()
        strategies = prevention.get_coping_strategies("craving", trigger_type="estrés")
        assert len(strategies) > 0
    
    def test_generate_emergency_plan(self):
        """Test emergency plan generation"""
        prevention = RelapsePrevention()
        situation = {
            "emergency_contacts": ["123-456-7890"],
            "safe_places": ["home"],
            "distraction_activities": ["exercise"]
        }
        plan = prevention.generate_emergency_plan("test_user", situation)
        
        assert plan["user_id"] == "test_user"
        assert "steps" in plan
        assert len(plan["steps"]) > 0
        assert "emergency_contacts" in plan
        assert "safe_places" in plan
        assert "distraction_activities" in plan
    
    def test_detect_warning_signs_high_stress(self):
        """Test detecting warning signs with high stress"""
        prevention = RelapsePrevention()
        state = {"stress_level": 9}
        signs = prevention._detect_warning_signs(state)
        assert "Alto nivel de estrés" in signs
    
    def test_detect_warning_signs_isolation(self):
        """Test detecting warning signs with isolation"""
        prevention = RelapsePrevention()
        state = {"isolation": True}
        signs = prevention._detect_warning_signs(state)
        assert "Aislamiento social" in signs
    
    def test_detect_warning_signs_multiple_triggers(self):
        """Test detecting warning signs with multiple triggers"""
        prevention = RelapsePrevention()
        state = {"triggers": ["estrés", "social", "trabajo", "emocional"]}
        signs = prevention._detect_warning_signs(state)
        assert "Múltiples triggers presentes" in signs
    
    def test_determine_risk_level_critical(self):
        """Test risk level determination for critical score"""
        prevention = RelapsePrevention()
        level = prevention._determine_risk_level(80)
        assert level == "crítico"
    
    def test_determine_risk_level_high(self):
        """Test risk level determination for high score"""
        prevention = RelapsePrevention()
        level = prevention._determine_risk_level(60)
        assert level == "alto"
    
    def test_determine_risk_level_medium(self):
        """Test risk level determination for medium score"""
        prevention = RelapsePrevention()
        level = prevention._determine_risk_level(30)
        assert level == "medio"
    
    def test_determine_risk_level_low(self):
        """Test risk level determination for low score"""
        prevention = RelapsePrevention()
        level = prevention._determine_risk_level(10)
        assert level == "bajo"
    
    def test_generate_recommendations_critical(self):
        """Test recommendations for critical risk"""
        prevention = RelapsePrevention()
        recommendations = prevention._generate_recommendations(
            80, "crítico", ["Alto nivel de estrés"], {}
        )
        assert len(recommendations) > 0
        assert any("crítico" in r.lower() or "riesgo" in r.lower() for r in recommendations)
    
    def test_generate_recommendations_high(self):
        """Test recommendations for high risk"""
        prevention = RelapsePrevention()
        recommendations = prevention._generate_recommendations(
            60, "alto", [], {}
        )
        assert len(recommendations) > 0
    
    def test_load_warning_signs(self):
        """Test warning signs loading"""
        prevention = RelapsePrevention()
        signs = prevention._load_warning_signs()
        assert "emotional" in signs
        assert "behavioral" in signs
        assert "environmental" in signs
    
    def test_load_coping_strategies(self):
        """Test coping strategies loading"""
        prevention = RelapsePrevention()
        strategies = prevention._load_coping_strategies()
        assert "general" in strategies
        assert "cravings" in strategies
        assert "stress" in strategies
        assert "social" in strategies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


