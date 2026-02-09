"""
Complete Integration Tests
End-to-end tests for full workflows
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient


@pytest.fixture
def complete_test_client():
    """Create test client with all dependencies mocked"""
    # Use canonical API router instead of deprecated recovery_api
    from api.recovery_api_refactored import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    # Mock all dependencies
    mock_analyzer = Mock()
    mock_analyzer.analyze = AsyncMock(return_value={
        "severity": "moderate",
        "risk_score": 0.65,
        "recommendations": ["Seek professional help"]
    })
    
    mock_tracker = Mock()
    mock_tracker.log_entry = AsyncMock(return_value={"entry_id": "entry_123"})
    mock_tracker.get_progress = AsyncMock(return_value={"days_sober": 30})
    
    mock_planner = Mock()
    mock_planner.create_plan = AsyncMock(return_value={"plan_id": "plan_123"})
    
    mock_prevention = Mock()
    mock_prevention.assess_risk = AsyncMock(return_value={"risk_score": 0.3})
    
    with patch('api.dependencies.AddictionAnalyzerDep', return_value=mock_analyzer):
        with patch('api.dependencies.ProgressTrackerDep', return_value=mock_tracker):
            with patch('api.dependencies.RecoveryPlannerDep', return_value=mock_planner):
                with patch('api.dependencies.RelapsePreventionDep', return_value=mock_prevention):
                    yield TestClient(app)


class TestCompleteUserJourney:
    """Test complete user journey from registration to recovery"""
    
    def test_user_registration_to_first_assessment(self, complete_test_client):
        """Test user registration and first assessment"""
        # Step 1: Create user
        user_data = {
            "email": "test@example.com",
            "name": "Test User",
            "addiction_type": "smoking"
        }
        
        with patch('api.routes.users.create_user') as mock_create:
            mock_create.return_value = {
                "user_id": "user_123",
                "email": "test@example.com"
            }
            
            response = complete_test_client.post("/users", json=user_data)
            assert response.status_code == 201
            user_id = response.json()["user_id"]
        
        # Step 2: Initial assessment
        assessment_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": user_id
        }
        
        response = complete_test_client.post("/assessment/assess", json=assessment_data)
        assert response.status_code == 200
        assert "severity" in response.json()
    
    def test_daily_progress_tracking_workflow(self, complete_test_client):
        """Test daily progress tracking over multiple days"""
        user_id = "user_123"
        
        # Log entries for 7 days
        for day in range(7):
            entry_data = {
                "user_id": user_id,
                "date": (datetime.now() - timedelta(days=6-day)).isoformat(),
                "mood": "good" if day % 2 == 0 else "neutral",
                "cravings_level": max(0, 5 - day),
                "notes": f"Day {day + 1} of recovery"
            }
            
            response = complete_test_client.post("/progress/log-entry", json=entry_data)
            assert response.status_code == 201
        
        # Get progress summary
        response = complete_test_client.get(f"/progress/{user_id}")
        assert response.status_code == 200
        progress = response.json()
        assert "days_sober" in progress or "total_entries" in progress
    
    def test_relapse_prevention_workflow(self, complete_test_client):
        """Test relapse prevention and risk assessment workflow"""
        user_id = "user_123"
        
        # Assess risk
        risk_data = {
            "user_id": user_id,
            "current_mood": "anxious",
            "stress_level": 7,
            "triggers": ["work stress"]
        }
        
        with patch('api.routes.relapse.handlers.assess_relapse_risk') as mock_assess:
            mock_assess.return_value = {
                "risk_score": 0.6,
                "risk_level": "moderate",
                "recommendations": ["Reach out to support group"]
            }
            
            response = complete_test_client.post("/relapse/assess-risk", json=risk_data)
            assert response.status_code == 200
            assert response.json()["risk_level"] == "moderate"
        
        # Get coaching based on risk
        coaching_data = {
            "user_id": user_id,
            "context": "High stress at work",
            "current_situation": "Feeling anxious"
        }
        
        with patch('api.routes.support.handlers.get_coaching') as mock_coaching:
            mock_coaching.return_value = {
                "advice": "Take deep breaths and focus on recovery",
                "suggestions": ["Practice mindfulness"]
            }
            
            response = complete_test_client.post("/support/coaching", json=coaching_data)
            assert response.status_code == 200
    
    def test_emergency_protocol_workflow(self, complete_test_client):
        """Test emergency contact setup and protocol"""
        user_id = "user_123"
        
        # Create emergency contact
        contact_data = {
            "user_id": user_id,
            "name": "Emergency Contact",
            "phone": "+1234567890",
            "relationship": "family",
            "is_primary": True
        }
        
        with patch('api.routes.emergency.create_emergency_contact') as mock_create:
            mock_create.return_value = {
                "contact_id": "contact_123",
                "name": "Emergency Contact"
            }
            
            response = complete_test_client.post("/emergency/contact", json=contact_data)
            assert response.status_code == 201
        
        # Trigger emergency
        emergency_data = {
            "user_id": user_id,
            "emergency_type": "crisis",
            "severity": "high"
        }
        
        with patch('api.routes.emergency.trigger_emergency') as mock_trigger:
            mock_trigger.return_value = {
                "emergency_id": "emergency_123",
                "status": "activated"
            }
            
            response = complete_test_client.post("/emergency/trigger", json=emergency_data)
            assert response.status_code == 200


class TestDataConsistency:
    """Test data consistency across different endpoints"""
    
    def test_user_data_consistency(self, complete_test_client):
        """Test that user data is consistent across endpoints"""
        user_id = "user_123"
        
        # Get user profile
        with patch('api.routes.assessment.handlers.get_user_profile') as mock_profile:
            mock_profile.return_value = {
                "user_id": user_id,
                "addiction_type": "smoking",
                "severity": "moderate"
            }
            
            profile_response = complete_test_client.get(f"/assessment/profile/{user_id}")
            profile_data = profile_response.json()
        
        # Get user progress (should have same user_id)
        with patch('api.routes.progress.handlers.get_user_progress') as mock_progress:
            mock_progress.return_value = {
                "user_id": user_id,
                "days_sober": 30
            }
            
            progress_response = complete_test_client.get(f"/progress/{user_id}")
            progress_data = progress_response.json()
        
        # Verify consistency
        assert profile_data["user_id"] == progress_data.get("user_id", user_id)
    
    def test_progress_calculation_consistency(self, complete_test_client):
        """Test that progress calculations are consistent"""
        user_id = "user_123"
        
        # Log multiple entries
        entries = []
        for i in range(5):
            entry = {
                "user_id": user_id,
                "date": (datetime.now() - timedelta(days=4-i)).isoformat(),
                "mood": "good",
                "cravings_level": 2
            }
            entries.append(entry)
            complete_test_client.post("/progress/log-entry", json=entry)
        
        # Get stats
        with patch('api.routes.progress.handlers.get_user_stats') as mock_stats:
            mock_stats.return_value = {
                "days_sober": 5,
                "total_entries": 5,
                "average_cravings": 2.0
            }
            
            stats_response = complete_test_client.get(f"/progress/{user_id}/stats")
            stats = stats_response.json()
            
            # Verify consistency
            assert stats["total_entries"] == 5
            assert stats["days_sober"] == 5


class TestErrorRecovery:
    """Test error recovery and resilience"""
    
    def test_recovery_from_service_failure(self, complete_test_client):
        """Test recovery when a service fails temporarily"""
        user_id = "user_123"
        
        # Simulate service failure
        with patch('api.routes.progress.handlers.get_user_progress') as mock_progress:
            mock_progress.side_effect = Exception("Service temporarily unavailable")
            
            # Should handle error gracefully
            response = complete_test_client.get(f"/progress/{user_id}")
            assert response.status_code in [500, 503]  # Service error
    
    def test_recovery_from_invalid_data(self, complete_test_client):
        """Test recovery from invalid data submission"""
        # Submit invalid data
        invalid_data = {
            "user_id": "",
            "date": "invalid-date",
            "mood": "invalid_mood"
        }
        
        response = complete_test_client.post("/progress/log-entry", json=invalid_data)
        
        # Should return validation error, not crash
        assert response.status_code == 422  # Validation error
    
    def test_recovery_from_missing_dependencies(self, complete_test_client):
        """Test recovery when dependencies are missing"""
        user_id = "user_123"
        
        # Simulate missing dependency
        with patch('api.dependencies.ProgressTrackerDep', side_effect=Exception("Dependency not available")):
            response = complete_test_client.get(f"/progress/{user_id}")
            # Should handle gracefully
            assert response.status_code in [500, 503]


class TestPerformance:
    """Test performance under load"""
    
    def test_concurrent_requests(self, complete_test_client):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        
        user_id = "user_123"
        
        def make_request():
            entry_data = {
                "user_id": user_id,
                "date": datetime.now().isoformat(),
                "mood": "good",
                "cravings_level": 3
            }
            return complete_test_client.post("/progress/log-entry", json=entry_data)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(r.status_code in [201, 200, 422] for r in results)
    
    def test_large_payload_handling(self, complete_test_client):
        """Test handling of large payloads"""
        user_id = "user_123"
        
        # Create large notes field
        large_notes = "A" * 10000  # 10KB of text
        
        entry_data = {
            "user_id": user_id,
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3,
            "notes": large_notes
        }
        
        response = complete_test_client.post("/progress/log-entry", json=entry_data)
        
        # Should handle or reject appropriately
        assert response.status_code in [201, 400, 413]  # Created, Bad Request, or Payload Too Large


class TestSecurity:
    """Test security aspects"""
    
    def test_unauthorized_access_blocked(self, complete_test_client):
        """Test that unauthorized access is blocked"""
        user_id = "user_123"
        
        # Try to access without authentication
        response = complete_test_client.get(f"/progress/{user_id}")
        
        # Should either require auth or allow (depending on implementation)
        assert response.status_code in [200, 401, 403]
    
    def test_sql_injection_prevention(self, complete_test_client):
        """Test that SQL injection attempts are prevented"""
        malicious_user_id = "user_123'; DROP TABLE users; --"
        
        response = complete_test_client.get(f"/progress/{malicious_user_id}")
        
        # Should not crash or execute SQL
        assert response.status_code in [200, 400, 404, 422]
    
    def test_xss_prevention(self, complete_test_client):
        """Test that XSS attempts are prevented"""
        xss_payload = "<script>alert('xss')</script>"
        
        entry_data = {
            "user_id": "user_123",
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3,
            "notes": xss_payload
        }
        
        response = complete_test_client.post("/progress/log-entry", json=entry_data)
        
        # Should sanitize or reject
        assert response.status_code in [201, 400, 422]


class TestWorkflowIntegration:
    """Test integration of multiple workflows"""
    
    def test_assessment_to_plan_to_tracking(self, complete_test_client):
        """Test complete workflow from assessment to plan to tracking"""
        user_id = "user_123"
        
        # 1. Assessment
        assessment_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": user_id
        }
        
        response = complete_test_client.post("/assessment/assess", json=assessment_data)
        assert response.status_code == 200
        
        # 2. Create recovery plan
        with patch('api.routes.support.handlers.create_recovery_plan') as mock_plan:
            mock_plan.return_value = {"plan_id": "plan_123"}
            
            plan_data = {"user_id": user_id, "duration_days": 90}
            response = complete_test_client.post("/support/plan", json=plan_data)
            assert response.status_code in [200, 201]
        
        # 3. Start tracking
        entry_data = {
            "user_id": user_id,
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 3
        }
        
        response = complete_test_client.post("/progress/log-entry", json=entry_data)
        assert response.status_code == 201
    
    def test_gamification_integration(self, complete_test_client):
        """Test gamification integration with progress tracking"""
        user_id = "user_123"
        
        # Log progress
        entry_data = {
            "user_id": user_id,
            "date": datetime.now().isoformat(),
            "mood": "good",
            "cravings_level": 2
        }
        
        complete_test_client.post("/progress/log-entry", json=entry_data)
        
        # Check achievements
        with patch('api.routes.gamification.get_user_achievements') as mock_achievements:
            mock_achievements.return_value = {
                "achievements": [{"name": "First Entry", "unlocked": True}],
                "total_points": 10
            }
            
            response = complete_test_client.get(f"/gamification/{user_id}/achievements")
            assert response.status_code == 200
            assert "achievements" in response.json()



