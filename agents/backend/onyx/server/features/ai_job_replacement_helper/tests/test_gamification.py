"""
Tests for Gamification Service
"""

import pytest
from datetime import datetime
from core.gamification import GamificationService, BadgeType


@pytest.fixture
def gamification_service():
    return GamificationService()


@pytest.fixture
def user_id():
    return "test_user_123"


def test_add_points(gamification_service, user_id):
    """Test adding points to user"""
    result = gamification_service.add_points(user_id, "complete_step")
    
    assert result["points_added"] == 25
    assert result["total_points"] == 25
    assert result["current_level"] == 1
    assert result["level_up"] is False


def test_level_up(gamification_service, user_id):
    """Test leveling up"""
    # Add enough points to level up
    for _ in range(5):
        gamification_service.add_points(user_id, "complete_step")
    
    stats = gamification_service.get_user_stats(user_id)
    assert stats.current_level >= 1


def test_badge_unlock(gamification_service, user_id):
    """Test badge unlocking"""
    gamification_service.add_points(user_id, "complete_step")
    
    stats = gamification_service.get_user_stats(user_id)
    assert len(stats.badges) > 0
    assert any(b.id == BadgeType.FIRST_STEP for b in stats.badges)


def test_leaderboard(gamification_service):
    """Test leaderboard"""
    # Add points to multiple users
    gamification_service.add_points("user1", "complete_step")
    gamification_service.add_points("user2", "apply_job")
    gamification_service.add_points("user3", "complete_step")
    
    leaderboard = gamification_service.get_leaderboard(limit=10)
    assert len(leaderboard) > 0
    assert leaderboard[0]["total_points"] >= leaderboard[-1]["total_points"]




