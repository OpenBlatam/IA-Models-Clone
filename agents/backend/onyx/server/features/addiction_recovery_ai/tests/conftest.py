"""
Pytest Configuration and Fixtures
Shared fixtures for all tests
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, MagicMock

from core.addiction_analyzer import AddictionAnalyzer, AddictionAssessment
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker
from core.relapse_prevention import RelapsePrevention


# ============================================================================
# Core Class Fixtures
# ============================================================================

@pytest.fixture
def addiction_analyzer():
    """Fixture for AddictionAnalyzer without OpenAI client"""
    return AddictionAnalyzer()


@pytest.fixture
def addiction_analyzer_with_ai():
    """Fixture for AddictionAnalyzer with mocked OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"psychological_analysis": "test", "personalized_strategies": [], "motivational_message": "test"}'
    mock_client.chat.completions.create.return_value = mock_response
    return AddictionAnalyzer(openai_client=mock_client)


@pytest.fixture
def recovery_planner():
    """Fixture for RecoveryPlanner"""
    return RecoveryPlanner()


@pytest.fixture
def progress_tracker():
    """Fixture for ProgressTracker"""
    return ProgressTracker()


@pytest.fixture
def relapse_prevention():
    """Fixture for RelapsePrevention"""
    return RelapsePrevention()


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def minimal_assessment_data():
    """Minimal valid assessment data"""
    return {
        "addiction_type": "cigarrillos",
        "severity": "moderada",
        "frequency": "diaria"
    }


@pytest.fixture
def complete_assessment_data():
    """Complete assessment data with all fields"""
    return {
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
        "additional_info": "User has been struggling for several years"
    }


@pytest.fixture
def assessment_data_variations():
    """Multiple assessment data variations for parametrized tests"""
    return [
        {
            "name": "cigarettes_moderate",
            "data": {
                "addiction_type": "cigarrillos",
                "severity": "moderada",
                "frequency": "diaria",
                "duration_years": 3.0,
                "daily_cost": 10.0
            }
        },
        {
            "name": "alcohol_severe",
            "data": {
                "addiction_type": "alcohol",
                "severity": "severa",
                "frequency": "diaria",
                "duration_years": 10.0,
                "daily_cost": 30.0
            }
        },
        {
            "name": "drugs_critical",
            "data": {
                "addiction_type": "drogas",
                "severity": "crítica",
                "frequency": "diaria",
                "duration_years": 15.0,
                "daily_cost": 50.0
            }
        },
        {
            "name": "tobacco_light",
            "data": {
                "addiction_type": "tabaco",
                "severity": "leve",
                "frequency": "ocasional",
                "duration_years": 1.0,
                "daily_cost": 5.0
            }
        }
    ]


@pytest.fixture
def sample_progress_entries():
    """Sample progress entries for testing"""
    entries = []
    base_date = datetime.now() - timedelta(days=6)
    
    for i in range(7):
        entry = {
            "user_id": "test_user",
            "date": (base_date + timedelta(days=i)).isoformat(),
            "mood": "bueno" if i % 2 == 0 else "regular",
            "cravings_level": max(1, 5 - i),
            "triggers_encountered": [] if i < 3 else ["estrés"],
            "consumed": False,
            "notes": f"Day {i+1} entry"
        }
        entries.append(entry)
    
    return entries


@pytest.fixture
def progress_entries_with_relapse():
    """Progress entries including a relapse"""
    entries = []
    base_date = datetime.now() - timedelta(days=9)
    
    # 5 days sober
    for i in range(5):
        entry = {
            "user_id": "test_user",
            "date": (base_date + timedelta(days=i)).isoformat(),
            "mood": "bueno",
            "cravings_level": 3,
            "triggers_encountered": [],
            "consumed": False
        }
        entries.append(entry)
    
    # Relapse on day 6
    entry = {
        "user_id": "test_user",
        "date": (base_date + timedelta(days=5)).isoformat(),
        "mood": "malo",
        "cravings_level": 9,
        "triggers_encountered": ["estrés", "social"],
        "consumed": True,
        "notes": "Relapse occurred"
    }
    entries.append(entry)
    
    # Recovery after relapse
    for i in range(6, 10):
        entry = {
            "user_id": "test_user",
            "date": (base_date + timedelta(days=i)).isoformat(),
            "mood": "regular",
            "cravings_level": 4,
            "triggers_encountered": [],
            "consumed": False
        }
        entries.append(entry)
    
    return entries


@pytest.fixture
def current_state_low_risk():
    """Current state with low relapse risk"""
    return {
        "stress_level": 2,
        "support_level": 9,
        "triggers": [],
        "previous_relapses": 0,
        "isolation": False,
        "negative_thinking": False
    }


@pytest.fixture
def current_state_high_risk():
    """Current state with high relapse risk"""
    return {
        "stress_level": 9,
        "support_level": 2,
        "triggers": ["estrés", "social", "trabajo"],
        "previous_relapses": 3,
        "isolation": True,
        "negative_thinking": True,
        "romanticizing": True,
        "skipping_support": True
    }


@pytest.fixture
def current_state_critical_risk():
    """Current state with critical relapse risk"""
    return {
        "stress_level": 10,
        "support_level": 1,
        "triggers": ["estrés", "social", "trabajo", "emocional", "físico"],
        "previous_relapses": 5,
        "isolation": True,
        "negative_thinking": True,
        "romanticizing": True
    }


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def sample_tensor():
    """Sample tensor for model testing"""
    return torch.tensor([[0.3, 0.4, 0.5, 0.7]], dtype=torch.float32)


@pytest.fixture
def sample_batch_tensor():
    """Sample batch tensor for model testing"""
    return torch.tensor([
        [0.3, 0.4, 0.5, 0.7],
        [0.2, 0.3, 0.4, 0.6],
        [0.5, 0.6, 0.7, 0.8]
    ], dtype=torch.float32)


@pytest.fixture
def sample_features_array():
    """Sample numpy features array"""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_data_list():
    """Sample data list for utility testing"""
    return list(range(100))


@pytest.fixture
def sample_sequence_data():
    """Sample sequence data for testing"""
    return [
        {"feature1": float(i), "feature2": float(i*2), "target": float(i*0.5)}
        for i in range(20)
    ]


# ============================================================================
# Edge Case Fixtures
# ============================================================================

@pytest.fixture
def empty_assessment_data():
    """Empty assessment data for error testing"""
    return {}


@pytest.fixture
def invalid_assessment_data():
    """Invalid assessment data for error testing"""
    return {
        "addiction_type": None,
        "severity": 123,  # Wrong type
        "frequency": []  # Wrong type
    }


@pytest.fixture
def empty_progress_entries():
    """Empty progress entries"""
    return []


@pytest.fixture
def malformed_progress_entries():
    """Malformed progress entries for error testing"""
    return [
        {"date": "2024-01-01", "consumed": False},  # Missing fields
        {"date": "invalid", "cravings_level": 5},  # Invalid date
        {"consumed": False}  # Missing date
    ]


@pytest.fixture
def empty_current_state():
    """Empty current state for error testing"""
    return {}


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def large_progress_entries():
    """Large dataset of progress entries for performance testing"""
    entries = []
    base_date = datetime.now() - timedelta(days=364)
    
    for i in range(365):
        entry = {
            "user_id": "test_user",
            "date": (base_date + timedelta(days=i)).isoformat(),
            "mood": "bueno" if i % 7 < 5 else "regular",
            "cravings_level": max(1, 5 - (i // 30)),
            "triggers_encountered": [] if i % 10 != 0 else ["estrés"],
            "consumed": False if i < 360 else (i == 360),
            "notes": f"Day {i+1} entry"
        }
        entries.append(entry)
    
    return entries


@pytest.fixture
def large_data_list():
    """Large data list for performance testing"""
    return list(range(10000))


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def complete_user_journey():
    """Complete user journey data for integration testing"""
    return {
        "user_id": "integration_test_user",
        "assessment": {
            "addiction_type": "cigarrillos",
            "severity": "moderada",
            "frequency": "diaria",
            "duration_years": 5.0,
            "daily_cost": 10.0,
            "triggers": ["estrés", "social"],
            "motivations": ["salud", "economía"],
            "previous_attempts": 1,
            "support_system": True
        },
        "entries": [
            {
                "date": (datetime.now() - timedelta(days=6-i)).isoformat(),
                "mood": "bueno" if i % 2 == 0 else "regular",
                "cravings_level": max(1, 5 - i),
                "triggers_encountered": [] if i < 3 else ["estrés"],
                "consumed": False
            }
            for i in range(7)
        ],
        "current_state": {
            "stress_level": 3,
            "support_level": 8,
            "triggers": [],
            "previous_relapses": 0
        }
    }


# ============================================================================
# API Test Fixtures
# ============================================================================

@pytest.fixture
def api_test_user():
    """Test user data for API tests"""
    return {
        "user_id": "test_user_123",
        "email": "test@example.com",
        "name": "Test User",
        "addiction_type": "smoking"
    }


@pytest.fixture
def api_assessment_request():
    """Assessment request data for API tests"""
    return {
        "addiction_type": "smoking",
        "severity": "moderate",
        "frequency": "daily",
        "duration_years": 5,
        "user_id": "test_user_123"
    }


@pytest.fixture
def api_progress_entry():
    """Progress entry data for API tests"""
    return {
        "user_id": "test_user_123",
        "date": datetime.now().isoformat(),
        "mood": "good",
        "cravings_level": 3,
        "notes": "Feeling good today"
    }


@pytest.fixture
def api_emergency_contact():
    """Emergency contact data for API tests"""
    return {
        "user_id": "test_user_123",
        "name": "Emergency Contact",
        "phone": "+1234567890",
        "relationship": "family",
        "is_primary": True
    }


@pytest.fixture
def api_relapse_risk_assessment():
    """Relapse risk assessment data for API tests"""
    return {
        "user_id": "test_user_123",
        "current_mood": "anxious",
        "stress_level": 7,
        "triggers": ["work stress", "social event"]
    }


@pytest.fixture
def mock_fastapi_dependencies():
    """Mock FastAPI dependencies"""
    from unittest.mock import Mock, AsyncMock
    
    mocks = {
        "analyzer": Mock(),
        "tracker": Mock(),
        "planner": Mock(),
        "prevention": Mock(),
        "emergency": Mock()
    }
    
    # Setup analyzer
    mocks["analyzer"].analyze = AsyncMock(return_value={
        "severity": "moderate",
        "risk_score": 0.65,
        "recommendations": ["Seek professional help"]
    })
    
    # Setup tracker
    mocks["tracker"].log_entry = AsyncMock(return_value={"entry_id": "entry_123"})
    mocks["tracker"].get_progress = AsyncMock(return_value={"days_sober": 30})
    mocks["tracker"].get_stats = AsyncMock(return_value={"days_sober": 30, "money_saved": 450.0})
    
    # Setup planner
    mocks["planner"].create_plan = AsyncMock(return_value={"plan_id": "plan_123"})
    
    # Setup prevention
    mocks["prevention"].assess_risk = AsyncMock(return_value={"risk_score": 0.3, "risk_level": "low"})
    
    # Setup emergency
    mocks["emergency"].create_contact = AsyncMock(return_value={"contact_id": "contact_123"})
    mocks["emergency"].trigger_emergency = AsyncMock(return_value={"emergency_id": "emergency_123"})
    
    return mocks


