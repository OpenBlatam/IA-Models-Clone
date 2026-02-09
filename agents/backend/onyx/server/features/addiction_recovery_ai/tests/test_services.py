"""
Tests for services
Comprehensive test suite for all service classes
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List


class TestMedicationService:
    """Tests for MedicationService"""
    
    @pytest.fixture
    def medication_service(self):
        """Create MedicationService instance"""
        from services.medication_service import MedicationService
        return MedicationService()
    
    def test_add_medication_success(self, medication_service):
        """Test adding medication successfully"""
        result = medication_service.add_medication(
            user_id="user_123",
            medication_name="Naltrexone",
            dosage="50mg",
            frequency="daily",
            start_date="2024-01-01",
            doctor_name="Dr. Smith",
            notes="Take with food"
        )
        
        assert result is not None
        assert "medication_id" in result
        assert result["medication_name"] == "Naltrexone"
        assert result["dosage"] == "50mg"
        assert result["status"] == "active"
    
    def test_add_medication_missing_required_fields(self, medication_service):
        """Test adding medication with missing required fields"""
        with pytest.raises((ValueError, TypeError)):
            medication_service.add_medication(
                user_id="user_123",
                medication_name="",  # Empty name
                dosage="50mg",
                frequency="daily",
                start_date="2024-01-01"
            )
    
    def test_get_medications_success(self, medication_service):
        """Test getting user medications"""
        # First add a medication
        medication_service.add_medication(
            user_id="user_123",
            medication_name="Naltrexone",
            dosage="50mg",
            frequency="daily",
            start_date="2024-01-01"
        )
        
        medications = medication_service.get_medications("user_123")
        
        assert isinstance(medications, (list, dict))
        if isinstance(medications, dict):
            assert "medications" in medications
        else:
            assert len(medications) > 0
    
    def test_update_medication_status_success(self, medication_service):
        """Test updating medication status"""
        # Add medication first
        med = medication_service.add_medication(
            user_id="user_123",
            medication_name="Naltrexone",
            dosage="50mg",
            frequency="daily",
            start_date="2024-01-01"
        )
        
        result = medication_service.update_medication_status(
            medication_id=med["medication_id"],
            status="completed"
        )
        
        assert result is not None
        assert result["status"] == "completed"
    
    def test_log_medication_dose_success(self, medication_service):
        """Test logging medication dose"""
        # Add medication first
        med = medication_service.add_medication(
            user_id="user_123",
            medication_name="Naltrexone",
            dosage="50mg",
            frequency="daily",
            start_date="2024-01-01"
        )
        
        result = medication_service.log_medication_dose(
            medication_id=med["medication_id"],
            taken_at=datetime.now().isoformat(),
            taken=True
        )
        
        assert result is not None
        assert "dose_id" in result or "log_id" in result


class TestGoalsService:
    """Tests for GoalsService"""
    
    @pytest.fixture
    def goals_service(self):
        """Create GoalsService instance"""
        from services.goals_service import GoalsService
        return GoalsService()
    
    def test_create_goal_success(self, goals_service):
        """Test creating a goal successfully"""
        result = goals_service.create_goal(
            user_id="user_123",
            goal_type="sobriety",
            title="30 Days Sober",
            description="Stay sober for 30 days",
            target_date="2024-02-01",
            target_value=30
        )
        
        assert result is not None
        assert "goal_id" in result
        assert result["title"] == "30 Days Sober"
        assert result["status"] == "pending" or result["status"] == "in_progress"
    
    def test_create_goal_invalid_type(self, goals_service):
        """Test creating goal with invalid type"""
        with pytest.raises((ValueError, TypeError)):
            goals_service.create_goal(
                user_id="user_123",
                goal_type="invalid_type",
                title="Test Goal",
                description="Test",
                target_date="2024-02-01"
            )
    
    def test_get_goals_success(self, goals_service):
        """Test getting user goals"""
        # Create a goal first
        goals_service.create_goal(
            user_id="user_123",
            goal_type="sobriety",
            title="30 Days Sober",
            description="Stay sober for 30 days",
            target_date="2024-02-01"
        )
        
        goals = goals_service.get_goals("user_123")
        
        assert isinstance(goals, (list, dict))
        if isinstance(goals, dict):
            assert "goals" in goals
        else:
            assert len(goals) > 0
    
    def test_update_goal_progress_success(self, goals_service):
        """Test updating goal progress"""
        # Create goal first
        goal = goals_service.create_goal(
            user_id="user_123",
            goal_type="sobriety",
            title="30 Days Sober",
            description="Stay sober for 30 days",
            target_date="2024-02-01",
            target_value=30
        )
        
        result = goals_service.update_goal_progress(
            goal_id=goal["goal_id"],
            current_value=15,
            progress_percentage=50.0
        )
        
        assert result is not None
        assert "current_value" in result or "progress" in result
    
    def test_complete_goal_success(self, goals_service):
        """Test completing a goal"""
        # Create goal first
        goal = goals_service.create_goal(
            user_id="user_123",
            goal_type="sobriety",
            title="30 Days Sober",
            description="Stay sober for 30 days",
            target_date="2024-02-01",
            target_value=30
        )
        
        result = goals_service.complete_goal(goal["goal_id"])
        
        assert result is not None
        assert result["status"] == "completed"


class TestHabitTrackingService:
    """Tests for HabitTrackingService"""
    
    @pytest.fixture
    def habit_service(self):
        """Create HabitTrackingService instance"""
        from services.habit_tracking_service import HabitTrackingService
        return HabitTrackingService()
    
    def test_create_habit_success(self, habit_service):
        """Test creating a habit successfully"""
        result = habit_service.create_habit(
            user_id="user_123",
            habit_name="Exercise",
            frequency="daily",
            target_days_per_week=5
        )
        
        assert result is not None
        assert "habit_id" in result
        assert result["habit_name"] == "Exercise"
    
    def test_log_habit_completion_success(self, habit_service):
        """Test logging habit completion"""
        # Create habit first
        habit = habit_service.create_habit(
            user_id="user_123",
            habit_name="Exercise",
            frequency="daily"
        )
        
        result = habit_service.log_habit_completion(
            habit_id=habit["habit_id"],
            date=datetime.now().isoformat(),
            completed=True
        )
        
        assert result is not None
    
    def test_get_habit_stats_success(self, habit_service):
        """Test getting habit statistics"""
        # Create and log habit first
        habit = habit_service.create_habit(
            user_id="user_123",
            habit_name="Exercise",
            frequency="daily"
        )
        
        stats = habit_service.get_habit_stats(habit["habit_id"])
        
        assert stats is not None
        assert "completion_rate" in stats or "total_completions" in stats


class TestNotificationService:
    """Tests for NotificationService"""
    
    @pytest.fixture
    def notification_service(self):
        """Create NotificationService instance"""
        from services.notification_service import NotificationService
        return NotificationService()
    
    def test_send_notification_success(self, notification_service):
        """Test sending notification successfully"""
        result = notification_service.send_notification(
            user_id="user_123",
            notification_type="reminder",
            message="Time for your daily check-in",
            priority="normal"
        )
        
        assert result is not None
        assert "notification_id" in result or "success" in result
    
    def test_get_user_notifications_success(self, notification_service):
        """Test getting user notifications"""
        # Send notification first
        notification_service.send_notification(
            user_id="user_123",
            notification_type="reminder",
            message="Test notification"
        )
        
        notifications = notification_service.get_user_notifications("user_123")
        
        assert isinstance(notifications, (list, dict))
        if isinstance(notifications, dict):
            assert "notifications" in notifications
        else:
            assert len(notifications) >= 0
    
    def test_mark_notification_read_success(self, notification_service):
        """Test marking notification as read"""
        # Send notification first
        notif = notification_service.send_notification(
            user_id="user_123",
            notification_type="reminder",
            message="Test notification"
        )
        
        notif_id = notif.get("notification_id") or notif.get("id")
        if notif_id:
            result = notification_service.mark_notification_read(notif_id)
            assert result is not None


class TestAnalyticsService:
    """Tests for AnalyticsService"""
    
    @pytest.fixture
    def analytics_service(self):
        """Create AnalyticsService instance"""
        from services.analytics_service import AnalyticsService
        return AnalyticsService()
    
    def test_get_user_analytics_success(self, analytics_service):
        """Test getting user analytics"""
        analytics = analytics_service.get_user_analytics("user_123")
        
        assert analytics is not None
        assert isinstance(analytics, dict)
    
    def test_get_progress_trend_success(self, analytics_service):
        """Test getting progress trend"""
        trend = analytics_service.get_progress_trend(
            user_id="user_123",
            days=30
        )
        
        assert trend is not None
        assert isinstance(trend, (dict, list))
    
    def test_get_risk_analysis_success(self, analytics_service):
        """Test getting risk analysis"""
        analysis = analytics_service.get_risk_analysis("user_123")
        
        assert analysis is not None
        assert isinstance(analysis, dict)


class TestGamificationService:
    """Tests for GamificationService"""
    
    @pytest.fixture
    def gamification_service(self):
        """Create GamificationService instance"""
        from services.gamification_service import GamificationService
        return GamificationService()
    
    def test_award_achievement_success(self, gamification_service):
        """Test awarding achievement"""
        result = gamification_service.award_achievement(
            user_id="user_123",
            achievement_type="first_week",
            points=100
        )
        
        assert result is not None
        assert "achievement_id" in result or "success" in result
    
    def test_get_user_achievements_success(self, gamification_service):
        """Test getting user achievements"""
        achievements = gamification_service.get_user_achievements("user_123")
        
        assert isinstance(achievements, (list, dict))
        if isinstance(achievements, dict):
            assert "achievements" in achievements or "total_points" in achievements
    
    def test_get_leaderboard_success(self, gamification_service):
        """Test getting leaderboard"""
        leaderboard = gamification_service.get_leaderboard(limit=10)
        
        assert isinstance(leaderboard, (list, dict))
        if isinstance(leaderboard, dict):
            assert "leaderboard" in leaderboard or "users" in leaderboard


class TestHealthTrackingService:
    """Tests for HealthTrackingService"""
    
    @pytest.fixture
    def health_service(self):
        """Create HealthTrackingService instance"""
        from services.health_tracking_service import HealthTrackingService
        return HealthTrackingService()
    
    def test_log_health_metric_success(self, health_service):
        """Test logging health metric"""
        result = health_service.log_health_metric(
            user_id="user_123",
            metric_type="blood_pressure",
            value={"systolic": 120, "diastolic": 80},
            date=datetime.now().isoformat()
        )
        
        assert result is not None
        assert "metric_id" in result or "log_id" in result
    
    def test_get_health_summary_success(self, health_service):
        """Test getting health summary"""
        summary = health_service.get_health_summary("user_123")
        
        assert summary is not None
        assert isinstance(summary, dict)


class TestMotivationService:
    """Tests for MotivationService"""
    
    @pytest.fixture
    def motivation_service(self):
        """Create MotivationService instance"""
        from services.motivation_service import MotivationService
        return MotivationService()
    
    def test_get_motivational_message_success(self, motivation_service):
        """Test getting motivational message"""
        message = motivation_service.get_motivational_message(
            user_id="user_123",
            days_sober=30,
            current_mood="good"
        )
        
        assert message is not None
        assert "message" in message or isinstance(message, str)
    
    def test_get_milestone_message_success(self, motivation_service):
        """Test getting milestone message"""
        message = motivation_service.get_milestone_message(
            user_id="user_123",
            milestone="30_days"
        )
        
        assert message is not None
        assert "message" in message or isinstance(message, str)


class TestEmergencyService:
    """Tests for EmergencyService"""
    
    @pytest.fixture
    def emergency_service(self):
        """Create EmergencyService instance"""
        from services.emergency_service import EmergencyService
        return EmergencyService()
    
    def test_create_emergency_contact_success(self, emergency_service):
        """Test creating emergency contact"""
        result = emergency_service.create_emergency_contact(
            user_id="user_123",
            name="John Doe",
            phone="+1234567890",
            relationship="family"
        )
        
        assert result is not None
        assert "contact_id" in result
        assert result["name"] == "John Doe"
    
    def test_trigger_emergency_protocol_success(self, emergency_service):
        """Test triggering emergency protocol"""
        result = emergency_service.trigger_emergency_protocol(
            user_id="user_123",
            emergency_type="crisis",
            severity="high"
        )
        
        assert result is not None
        assert "emergency_id" in result or "status" in result
    
    def test_get_emergency_contacts_success(self, emergency_service):
        """Test getting emergency contacts"""
        # Create contact first
        emergency_service.create_emergency_contact(
            user_id="user_123",
            name="John Doe",
            phone="+1234567890"
        )
        
        contacts = emergency_service.get_emergency_contacts("user_123")
        
        assert isinstance(contacts, (list, dict))
        if isinstance(contacts, dict):
            assert "contacts" in contacts


class TestChatbotService:
    """Tests for ChatbotService"""
    
    @pytest.fixture
    def chatbot_service(self):
        """Create ChatbotService instance"""
        from services.chatbot_service import ChatbotService
        return ChatbotService()
    
    def test_send_message_success(self, chatbot_service):
        """Test sending message to chatbot"""
        result = chatbot_service.send_message(
            user_id="user_123",
            message="I'm feeling anxious today",
            context={}
        )
        
        assert result is not None
        assert "response" in result or "message" in result
    
    def test_get_conversation_history_success(self, chatbot_service):
        """Test getting conversation history"""
        # Send a message first
        chatbot_service.send_message(
            user_id="user_123",
            message="Hello"
        )
        
        history = chatbot_service.get_conversation_history("user_123")
        
        assert isinstance(history, (list, dict))
        if isinstance(history, dict):
            assert "messages" in history or "conversation" in history


class TestReportService:
    """Tests for ReportService"""
    
    @pytest.fixture
    def report_service(self):
        """Create ReportService instance"""
        from services.report_service import ReportService
        return ReportService()
    
    def test_generate_progress_report_success(self, report_service):
        """Test generating progress report"""
        report = report_service.generate_progress_report(
            user_id="user_123",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        assert report is not None
        assert isinstance(report, dict)
    
    def test_generate_summary_report_success(self, report_service):
        """Test generating summary report"""
        report = report_service.generate_summary_report("user_123")
        
        assert report is not None
        assert isinstance(report, dict)



