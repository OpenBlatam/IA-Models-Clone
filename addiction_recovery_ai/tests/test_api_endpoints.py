"""
Tests for API endpoints
Comprehensive test suite for all API routes
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any
import json


@pytest.fixture
def mock_analyzer():
    """Mock AddictionAnalyzer dependency"""
    analyzer = Mock()
    analyzer.analyze = AsyncMock(return_value={
        "severity": "moderate",
        "risk_score": 0.65,
        "recommendations": ["Seek professional help", "Join support group"],
        "addiction_type": "smoking"
    })
    return analyzer


@pytest.fixture
def mock_tracker():
    """Mock ProgressTracker dependency"""
    tracker = Mock()
    tracker.log_entry = AsyncMock(return_value={
        "entry_id": "entry_123",
        "date": "2024-01-15",
        "mood": "good",
        "cravings_level": 3
    })
    tracker.get_progress = AsyncMock(return_value={
        "user_id": "user_123",
        "days_sober": 30,
        "total_entries": 30,
        "progress_percentage": 75.0
    })
    tracker.get_stats = AsyncMock(return_value={
        "days_sober": 30,
        "money_saved": 450.0,
        "health_improvements": ["Better sleep", "Improved breathing"]
    })
    tracker.get_timeline = AsyncMock(return_value={
        "timeline": [
            {"date": "2024-01-01", "event": "Started recovery"},
            {"date": "2024-01-15", "event": "30 days milestone"}
        ]
    })
    return tracker


@pytest.fixture
def mock_planner():
    """Mock RecoveryPlanner dependency"""
    planner = Mock()
    planner.create_plan = AsyncMock(return_value={
        "plan_id": "plan_123",
        "duration_days": 90,
        "milestones": ["Week 1", "Week 4", "Week 12"]
    })
    return planner


@pytest.fixture
def mock_relapse_prevention():
    """Mock RelapsePrevention dependency"""
    prevention = Mock()
    prevention.assess_risk = AsyncMock(return_value={
        "risk_score": 0.3,
        "risk_level": "low",
        "recommendations": ["Continue current plan"]
    })
    prevention.log_relapse = AsyncMock(return_value={
        "relapse_id": "relapse_123",
        "date": "2024-01-15",
        "severity": "minor"
    })
    return prevention


@pytest.fixture
def mock_emergency_service():
    """Mock EmergencyService dependency"""
    service = Mock()
    service.create_contact = AsyncMock(return_value={
        "contact_id": "contact_123",
        "name": "John Doe",
        "phone": "+1234567890"
    })
    service.get_contacts = AsyncMock(return_value={
        "contacts": [
            {"contact_id": "contact_123", "name": "John Doe", "phone": "+1234567890"}
        ]
    })
    service.trigger_emergency = AsyncMock(return_value={
        "emergency_id": "emergency_123",
        "status": "activated",
        "contacts_notified": 2
    })
    return service


@pytest.fixture
def client(mock_analyzer, mock_tracker, mock_planner, mock_relapse_prevention, mock_emergency_service):
    """Create test client with mocked dependencies"""
    # Use canonical API router instead of deprecated recovery_api
    from api.recovery_api_refactored import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    # Override dependencies
    app.dependency_overrides = {}
    
    with patch('api.dependencies.AddictionAnalyzerDep', return_value=mock_analyzer):
        with patch('api.dependencies.ProgressTrackerDep', return_value=mock_tracker):
            with patch('api.dependencies.RecoveryPlannerDep', return_value=mock_planner):
                with patch('api.dependencies.RelapsePreventionDep', return_value=mock_relapse_prevention):
                    with patch('api.dependencies.EmergencyServiceDep', return_value=mock_emergency_service):
                        yield TestClient(app)


class TestAssessmentEndpoints:
    """Tests for assessment endpoints"""
    
    def test_assess_addiction_success(self, client, mock_analyzer):
        """Test successful addiction assessment"""
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "duration_years": 5,
            "user_id": "user_123"
        }
        
        response = client.post("/assessment/assess", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "severity" in data
        assert "risk_score" in data
        assert "recommendations" in data
        mock_analyzer.analyze.assert_called_once()
    
    def test_assess_addiction_invalid_data(self, client):
        """Test assessment with invalid data"""
        request_data = {
            "addiction_type": "invalid_type",
            "severity": "invalid_severity"
        }
        
        response = client.post("/assessment/assess", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.parametrize("addiction_type,severity,frequency", [
        ("smoking", "low", "occasional"),
        ("alcohol", "moderate", "weekly"),
        ("drugs", "high", "daily"),
        ("gambling", "severe", "daily"),
    ])
    def test_assess_addiction_various_types(self, client, mock_analyzer, addiction_type, severity, frequency):
        """Test assessment with various addiction types and severities"""
        request_data = {
            "addiction_type": addiction_type,
            "severity": severity,
            "frequency": frequency,
            "user_id": "user_123"
        }
        
        response = client.post("/assessment/assess", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "severity" in data or "risk_score" in data
    
    def test_assess_addiction_missing_required_fields(self, client):
        """Test assessment with missing required fields"""
        request_data = {
            "addiction_type": "smoking"
            # Missing severity and frequency
        }
        
        response = client.post("/assessment/assess", json=request_data)
        
        assert response.status_code == 422
    
    def test_assess_addiction_service_error(self, client, mock_analyzer):
        """Test assessment when service fails"""
        mock_analyzer.analyze.side_effect = Exception("Service error")
        
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": "user_123"
        }
        
        with patch('api.routes.assessment.handlers.process_assessment') as mock_process:
            mock_process.side_effect = HTTPException(status_code=500, detail="Internal server error")
            
            response = client.post("/assessment/assess", json=request_data)
            
            # Should handle error gracefully
            assert response.status_code in [500, 503]
    
    def test_get_profile_success(self, client):
        """Test getting user profile"""
        user_id = "user_123"
        
        with patch('api.routes.assessment.handlers.get_user_profile') as mock_get:
            mock_get.return_value = {
                "user_id": user_id,
                "addiction_type": "smoking",
                "severity": "moderate",
                "created_at": "2024-01-01T00:00:00"
            }
            
            response = client.get(f"/assessment/profile/{user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == user_id
    
    def test_get_profile_not_found(self, client):
        """Test getting non-existent profile"""
        user_id = "nonexistent_user"
        
        with patch('api.routes.assessment.handlers.get_user_profile') as mock_get:
            from fastapi import HTTPException
            mock_get.side_effect = HTTPException(status_code=404, detail="User not found")
            
            response = client.get(f"/assessment/profile/{user_id}")
            
            assert response.status_code == 404
    
    def test_update_profile_success(self, client):
        """Test updating user profile"""
        request_data = {
            "user_id": "user_123",
            "name": "Updated Name",
            "email": "updated@example.com"
        }
        
        with patch('api.routes.assessment.handlers.update_user_profile') as mock_update:
            mock_update.return_value = {
                "success": True,
                "message": "Profile updated successfully"
            }
            
            response = client.post("/assessment/update-profile", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestProgressEndpoints:
    """Tests for progress tracking endpoints"""
    
    def test_log_entry_success(self, client, mock_tracker):
        """Test logging a daily entry"""
        request_data = {
            "user_id": "user_123",
            "date": "2024-01-15",
            "mood": "good",
            "cravings_level": 3,
            "notes": "Feeling better today"
        }
        
        response = client.post("/progress/log-entry", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
        assert data["date"] == "2024-01-15"
        mock_tracker.log_entry.assert_called_once()
    
    def test_log_entry_invalid_data(self, client):
        """Test logging entry with invalid data"""
        request_data = {
            "user_id": "user_123",
            "date": "invalid-date",
            "mood": "invalid_mood",
            "cravings_level": 15  # Invalid (should be 0-10)
        }
        
        response = client.post("/progress/log-entry", json=request_data)
        
        assert response.status_code == 422
    
    @pytest.mark.parametrize("mood,cravings_level", [
        ("excellent", 0),
        ("good", 2),
        ("neutral", 5),
        ("poor", 8),
        ("terrible", 10),
    ])
    def test_log_entry_various_moods(self, client, mock_tracker, mood, cravings_level):
        """Test logging entries with various moods and craving levels"""
        request_data = {
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": mood,
            "cravings_level": cravings_level
        }
        
        response = client.post("/progress/log-entry", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["mood"] == mood
        assert data["cravings_level"] == cravings_level
    
    def test_log_entry_boundary_values(self, client, mock_tracker):
        """Test logging entry with boundary values"""
        # Test minimum values
        request_data = {
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": "excellent",
            "cravings_level": 0
        }
        
        response = client.post("/progress/log-entry", json=request_data)
        assert response.status_code == 201
        
        # Test maximum values
        request_data["cravings_level"] = 10
        request_data["mood"] = "terrible"
        response = client.post("/progress/log-entry", json=request_data)
        assert response.status_code == 201
    
    def test_log_entry_missing_optional_fields(self, client, mock_tracker):
        """Test logging entry with only required fields"""
        request_data = {
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3
            # No notes, triggers, etc.
        }
        
        response = client.post("/progress/log-entry", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == "user_123"
    
    def test_get_progress_success(self, client, mock_tracker):
        """Test getting user progress"""
        user_id = "user_123"
        
        response = client.get(f"/progress/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "days_sober" in data
        mock_tracker.get_progress.assert_called_once()
    
    def test_get_progress_with_date_filter(self, client, mock_tracker):
        """Test getting progress with date filter"""
        user_id = "user_123"
        start_date = "2024-01-01"
        
        response = client.get(f"/progress/{user_id}?start_date={start_date}")
        
        assert response.status_code == 200
        mock_tracker.get_progress.assert_called_once()
    
    def test_get_stats_success(self, client, mock_tracker):
        """Test getting user statistics"""
        user_id = "user_123"
        
        response = client.get(f"/progress/{user_id}/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "days_sober" in data
        assert "money_saved" in data
        mock_tracker.get_stats.assert_called_once()
    
    def test_get_timeline_success(self, client, mock_tracker):
        """Test getting user timeline"""
        user_id = "user_123"
        
        response = client.get(f"/progress/{user_id}/timeline")
        
        assert response.status_code == 200
        data = response.json()
        assert "timeline" in data
        assert isinstance(data["timeline"], list)
        mock_tracker.get_timeline.assert_called_once()


class TestRelapseEndpoints:
    """Tests for relapse prevention endpoints"""
    
    def test_assess_relapse_risk_success(self, client, mock_relapse_prevention):
        """Test assessing relapse risk"""
        request_data = {
            "user_id": "user_123",
            "current_mood": "anxious",
            "stress_level": 7,
            "triggers": ["work stress", "social event"]
        }
        
        with patch('api.routes.relapse.handlers.assess_relapse_risk') as mock_assess:
            mock_assess.return_value = {
                "risk_score": 0.3,
                "risk_level": "low",
                "recommendations": ["Continue current plan"]
            }
            
            response = client.post("/relapse/assess-risk", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "risk_score" in data
            assert "risk_level" in data
    
    def test_log_relapse_success(self, client, mock_relapse_prevention):
        """Test logging a relapse"""
        request_data = {
            "user_id": "user_123",
            "date": "2024-01-15",
            "severity": "minor",
            "circumstances": "Stressful work day",
            "triggers": ["work stress"]
        }
        
        with patch('api.routes.relapse.handlers.log_relapse') as mock_log:
            mock_log.return_value = {
                "relapse_id": "relapse_123",
                "date": "2024-01-15",
                "severity": "minor"
            }
            
            response = client.post("/relapse/log", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "relapse_id" in data
    
    def test_get_relapse_history_success(self, client):
        """Test getting relapse history"""
        user_id = "user_123"
        
        with patch('api.routes.relapse.handlers.get_relapse_history') as mock_get:
            mock_get.return_value = {
                "relapses": [
                    {
                        "relapse_id": "relapse_123",
                        "date": "2024-01-15",
                        "severity": "minor"
                    }
                ],
                "total_count": 1
            }
            
            response = client.get(f"/relapse/{user_id}/history")
            
            assert response.status_code == 200
            data = response.json()
            assert "relapses" in data


class TestSupportEndpoints:
    """Tests for support and coaching endpoints"""
    
    def test_get_coaching_success(self, client):
        """Test getting coaching advice"""
        request_data = {
            "user_id": "user_123",
            "context": "Feeling stressed about work",
            "current_situation": "High stress at work"
        }
        
        with patch('api.routes.support.handlers.get_coaching') as mock_coaching:
            mock_coaching.return_value = {
                "advice": "Take deep breaths and focus on your recovery goals",
                "suggestions": ["Practice mindfulness", "Reach out to support group"]
            }
            
            response = client.post("/support/coaching", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "advice" in data
    
    def test_get_motivation_success(self, client):
        """Test getting motivational message"""
        request_data = {
            "user_id": "user_123",
            "days_sober": 30,
            "current_mood": "good"
        }
        
        with patch('api.routes.support.handlers.get_motivation') as mock_motivation:
            mock_motivation.return_value = {
                "message": "Congratulations on 30 days! Keep going!",
                "milestone": "30 days sober"
            }
            
            response = client.post("/support/motivation", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data


class TestEmergencyEndpoints:
    """Tests for emergency services endpoints"""
    
    def test_create_emergency_contact_success(self, client, mock_emergency_service):
        """Test creating emergency contact"""
        request_data = {
            "user_id": "user_123",
            "name": "John Doe",
            "phone": "+1234567890",
            "relationship": "family",
            "is_primary": True
        }
        
        response = client.post("/emergency/contact", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert "contact_id" in data
        assert data["name"] == "John Doe"
        mock_emergency_service.create_contact.assert_called_once()
    
    def test_get_emergency_contacts_success(self, client, mock_emergency_service):
        """Test getting emergency contacts"""
        user_id = "user_123"
        
        response = client.get(f"/emergency/{user_id}/contacts")
        
        assert response.status_code == 200
        data = response.json()
        assert "contacts" in data
        assert isinstance(data["contacts"], list)
        mock_emergency_service.get_contacts.assert_called_once()
    
    def test_trigger_emergency_success(self, client, mock_emergency_service):
        """Test triggering emergency protocol"""
        request_data = {
            "user_id": "user_123",
            "emergency_type": "crisis",
            "severity": "high",
            "location": "Home"
        }
        
        response = client.post("/emergency/trigger", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "emergency_id" in data
        assert "status" in data
        mock_emergency_service.trigger_emergency.assert_called_once()
    
    def test_get_crisis_resources_success(self, client):
        """Test getting crisis resources"""
        with patch('api.routes.emergency.get_crisis_resources') as mock_resources:
            mock_resources.return_value = {
                "resources": [
                    {
                        "name": "Crisis Hotline",
                        "phone": "1-800-XXX-XXXX",
                        "available_24_7": True
                    }
                ]
            }
            
            response = client.get("/emergency/resources")
            
            assert response.status_code == 200
            data = response.json()
            assert "resources" in data


class TestGamificationEndpoints:
    """Tests for gamification endpoints"""
    
    def test_get_achievements_success(self, client):
        """Test getting user achievements"""
        user_id = "user_123"
        
        with patch('api.routes.gamification.get_user_achievements') as mock_achievements:
            mock_achievements.return_value = {
                "achievements": [
                    {
                        "achievement_id": "ach_123",
                        "name": "First Week",
                        "description": "Complete first week sober",
                        "unlocked_at": "2024-01-07T00:00:00"
                    }
                ],
                "total_points": 100
            }
            
            response = client.get(f"/gamification/{user_id}/achievements")
            
            assert response.status_code == 200
            data = response.json()
            assert "achievements" in data
    
    def test_get_leaderboard_success(self, client):
        """Test getting leaderboard"""
        with patch('api.routes.gamification.get_leaderboard') as mock_leaderboard:
            mock_leaderboard.return_value = {
                "leaderboard": [
                    {"user_id": "user_1", "rank": 1, "points": 1000, "days_sober": 100},
                    {"user_id": "user_2", "rank": 2, "points": 900, "days_sober": 90}
                ]
            }
            
            response = client.get("/gamification/leaderboard")
            
            assert response.status_code == 200
            data = response.json()
            assert "leaderboard" in data
    
    def test_get_rewards_success(self, client):
        """Test getting user rewards"""
        user_id = "user_123"
        
        with patch('api.routes.gamification.get_user_rewards') as mock_rewards:
            mock_rewards.return_value = {
                "rewards": [
                    {
                        "reward_id": "reward_123",
                        "name": "Badge: 30 Days",
                        "points_cost": 50,
                        "unlocked": True
                    }
                ]
            }
            
            response = client.get(f"/gamification/{user_id}/rewards")
            
            assert response.status_code == 200
            data = response.json()
            assert "rewards" in data


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints"""
    
    def test_get_user_analytics_success(self, client):
        """Test getting user analytics"""
        user_id = "user_123"
        
        with patch('api.routes.analytics.get_user_analytics') as mock_analytics:
            mock_analytics.return_value = {
                "user_id": user_id,
                "progress_trend": "improving",
                "risk_trend": "decreasing",
                "engagement_score": 0.85
            }
            
            response = client.get(f"/analytics/{user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "user_id" in data
    
    def test_get_insights_success(self, client):
        """Test getting insights"""
        request_data = {
            "user_id": "user_123",
            "analysis_type": "behavioral_patterns"
        }
        
        with patch('api.routes.analytics.get_insights') as mock_insights:
            mock_insights.return_value = {
                "insights": [
                    {
                        "type": "pattern",
                        "description": "Higher cravings on weekends",
                        "confidence": 0.85
                    }
                ]
            }
            
            response = client.post("/analytics/insights", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "insights" in data


class TestUsersEndpoints:
    """Tests for user management endpoints"""
    
    def test_create_user_success(self, client):
        """Test creating a new user"""
        request_data = {
            "email": "test@example.com",
            "name": "Test User",
            "addiction_type": "smoking"
        }
        
        with patch('api.routes.users.create_user') as mock_create:
            mock_create.return_value = {
                "user_id": "user_123",
                "email": "test@example.com",
                "created_at": "2024-01-01T00:00:00"
            }
            
            response = client.post("/users", json=request_data)
            
            assert response.status_code == 201
            data = response.json()
            assert "user_id" in data
    
    def test_get_user_success(self, client):
        """Test getting user information"""
        user_id = "user_123"
        
        with patch('api.routes.users.get_user') as mock_get:
            mock_get.return_value = {
                "user_id": user_id,
                "email": "test@example.com",
                "name": "Test User",
                "addiction_type": "smoking"
            }
            
            response = client.get(f"/users/{user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == user_id
    
    def test_update_user_success(self, client):
        """Test updating user information"""
        user_id = "user_123"
        request_data = {
            "name": "Updated Name",
            "email": "updated@example.com"
        }
        
        with patch('api.routes.users.update_user') as mock_update:
            mock_update.return_value = {
                "user_id": user_id,
                "name": "Updated Name",
                "email": "updated@example.com"
            }
            
            response = client.post(f"/users/{user_id}", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Name"


class TestNotificationsEndpoints:
    """Tests for notifications endpoints"""
    
    def test_get_notifications_success(self, client):
        """Test getting user notifications"""
        user_id = "user_123"
        
        with patch('api.routes.notifications.get_user_notifications') as mock_notifications:
            mock_notifications.return_value = {
                "notifications": [
                    {
                        "notification_id": "notif_123",
                        "type": "reminder",
                        "message": "Time for your daily check-in",
                        "read": False
                    }
                ]
            }
            
            response = client.get(f"/notifications/{user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "notifications" in data
    
    def test_mark_notification_read_success(self, client):
        """Test marking notification as read"""
        notification_id = "notif_123"
        
        with patch('api.routes.notifications.mark_notification_read') as mock_mark:
            mock_mark.return_value = {
                "success": True,
                "message": "Notification marked as read"
            }
            
            response = client.post(f"/notifications/{notification_id}/read")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

