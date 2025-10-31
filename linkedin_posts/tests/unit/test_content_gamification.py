"""
Content Gamification Tests
==========================

Comprehensive tests for content gamification features including:
- Engagement rewards and points systems
- Achievement and badge systems
- Leaderboards and rankings
- Challenges and competitions
- Interactive content features
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from typing import List, Dict, Any, Optional

# Test data
SAMPLE_GAMIFICATION_CONFIG = {
    "reward_system": {
        "points_per_engagement": 10,
        "points_per_share": 25,
        "points_per_comment": 15,
        "bonus_multipliers": {
            "viral_content": 2.0,
            "high_quality": 1.5,
            "consistent_posting": 1.2
        }
    },
    "achievement_system": {
        "badges": {
            "first_post": {"name": "First Post", "points": 50},
            "viral_content": {"name": "Viral Creator", "points": 200},
            "consistent_poster": {"name": "Consistent", "points": 100}
        },
        "levels": {
            "beginner": {"min_points": 0, "max_points": 500},
            "intermediate": {"min_points": 501, "max_points": 2000},
            "expert": {"min_points": 2001, "max_points": 5000},
            "master": {"min_points": 5001, "max_points": None}
        }
    },
    "leaderboard_config": {
        "update_frequency": "daily",
        "ranking_criteria": ["total_points", "engagement_rate", "content_quality"],
        "leaderboard_size": 100
    }
}

SAMPLE_USER_GAMIFICATION_DATA = {
    "user_id": "user123",
    "total_points": 1250,
    "current_level": "intermediate",
    "badges_earned": [
        {"badge_id": "first_post", "earned_date": datetime.now() - timedelta(days=30)},
        {"badge_id": "consistent_poster", "earned_date": datetime.now() - timedelta(days=15)}
    ],
    "achievements": [
        {"achievement_id": "viral_content", "progress": 0.8, "completed": False},
        {"achievement_id": "engagement_master", "progress": 1.0, "completed": True}
    ],
    "leaderboard_rank": 15,
    "weekly_points": 180
}

SAMPLE_CHALLENGE_DATA = {
    "challenge_id": str(uuid4()),
    "challenge_name": "Tech Innovation Week",
    "challenge_description": "Post 5 tech innovation articles this week",
    "challenge_type": "content_creation",
    "challenge_goals": {
        "posts_required": 5,
        "engagement_target": 0.1,
        "time_limit": "7_days"
    },
    "challenge_rewards": {
        "points": 500,
        "badge": "tech_innovator",
        "leaderboard_boost": 1.5
    },
    "participants": ["user123", "user456", "user789"],
    "start_date": datetime.now(),
    "end_date": datetime.now() + timedelta(days=7)
}

class TestContentGamification:
    """Test content gamification features"""
    
    @pytest.fixture
    def mock_gamification_service(self):
        """Mock gamification service."""
        service = AsyncMock()
        service.award_points.return_value = {
            "points_awarded": 25,
            "total_points": 1275,
            "level_up": False,
            "new_badges": []
        }
        service.check_achievements.return_value = {
            "achievements_unlocked": ["viral_content"],
            "progress_updates": [
                {"achievement_id": "engagement_master", "progress": 1.0}
            ],
            "badges_earned": ["viral_creator"]
        }
        service.update_leaderboard.return_value = {
            "leaderboard_updated": True,
            "new_rank": 12,
            "rank_change": -3
        }
        return service
    
    @pytest.fixture
    def mock_challenge_service(self):
        """Mock challenge service."""
        service = AsyncMock()
        service.create_challenge.return_value = SAMPLE_CHALLENGE_DATA
        service.join_challenge.return_value = {
            "joined": True,
            "challenge_id": str(uuid4()),
            "start_date": datetime.now()
        }
        service.update_challenge_progress.return_value = {
            "progress_updated": True,
            "current_progress": 0.6,
            "goals_remaining": ["posts_required"]
        }
        service.complete_challenge.return_value = {
            "challenge_completed": True,
            "rewards_awarded": {
                "points": 500,
                "badge": "tech_innovator"
            },
            "completion_time": datetime.now()
        }
        return service
    
    @pytest.fixture
    def mock_reward_service(self):
        """Mock reward service."""
        service = AsyncMock()
        service.calculate_rewards.return_value = {
            "base_points": 25,
            "bonus_points": 10,
            "multiplier": 1.5,
            "total_points": 52
        }
        service.distribute_rewards.return_value = {
            "rewards_distributed": True,
            "recipients": ["user123", "user456"],
            "total_rewards": 104
        }
        service.validate_reward_eligibility.return_value = {
            "eligible": True,
            "eligibility_reason": "valid_engagement",
            "restrictions": []
        }
        return service
    
    @pytest.fixture
    def mock_gamification_repository(self):
        """Mock gamification repository."""
        repository = AsyncMock()
        repository.save_gamification_data.return_value = {
            "gamification_id": str(uuid4()),
            "saved": True,
            "timestamp": datetime.now()
        }
        repository.get_user_gamification_data.return_value = SAMPLE_USER_GAMIFICATION_DATA
        repository.save_challenge_data.return_value = {
            "challenge_id": str(uuid4()),
            "saved": True
        }
        return repository
    
    @pytest.fixture
    def post_service(self, mock_gamification_repository, mock_gamification_service, mock_challenge_service, mock_reward_service):
        from services.post_service import PostService
        service = PostService(
            repository=mock_gamification_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            gamification_service=mock_gamification_service,
            challenge_service=mock_challenge_service,
            reward_service=mock_reward_service
        )
        return service
    
    @pytest.mark.asyncio
    async def test_points_awarding(self, post_service, mock_gamification_service):
        """Test awarding points for content engagement."""
        engagement_data = {
            "user_id": "user123",
            "engagement_type": "share",
            "content_id": str(uuid4()),
            "engagement_value": 1
        }
        
        result = await post_service.award_engagement_points(engagement_data)
        
        assert "points_awarded" in result
        assert "total_points" in result
        assert "level_up" in result
        mock_gamification_service.award_points.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_achievement_checking(self, post_service, mock_gamification_service):
        """Test checking and unlocking achievements."""
        user_id = "user123"
        content_data = {
            "content_id": str(uuid4()),
            "engagement_metrics": {"likes": 200, "shares": 50},
            "content_quality": "high"
        }
        
        achievements = await post_service.check_user_achievements(user_id, content_data)
        
        assert "achievements_unlocked" in achievements
        assert "progress_updates" in achievements
        assert "badges_earned" in achievements
        mock_gamification_service.check_achievements.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_leaderboard_update(self, post_service, mock_gamification_service):
        """Test updating user leaderboard position."""
        user_id = "user123"
        performance_data = {
            "total_points": 1275,
            "engagement_rate": 0.085,
            "content_quality": 0.92
        }
        
        leaderboard = await post_service.update_leaderboard_position(user_id, performance_data)
        
        assert "leaderboard_updated" in leaderboard
        assert "new_rank" in leaderboard
        assert "rank_change" in leaderboard
        mock_gamification_service.update_leaderboard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_challenge_creation(self, post_service, mock_challenge_service):
        """Test creating content challenges."""
        challenge_config = {
            "challenge_name": "Tech Innovation Week",
            "challenge_description": "Post 5 tech innovation articles this week",
            "challenge_goals": {"posts_required": 5, "engagement_target": 0.1},
            "challenge_rewards": {"points": 500, "badge": "tech_innovator"}
        }
        
        challenge = await post_service.create_content_challenge(challenge_config)
        
        assert "challenge_id" in challenge
        assert "challenge_name" in challenge
        assert "challenge_goals" in challenge
        mock_challenge_service.create_challenge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_challenge_joining(self, post_service, mock_challenge_service):
        """Test joining content challenges."""
        challenge_id = str(uuid4())
        user_id = "user123"
        
        result = await post_service.join_challenge(challenge_id, user_id)
        
        assert result["joined"] is True
        assert "challenge_id" in result
        assert "start_date" in result
        mock_challenge_service.join_challenge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_challenge_progress_update(self, post_service, mock_challenge_service):
        """Test updating challenge progress."""
        challenge_id = str(uuid4())
        user_id = "user123"
        progress_data = {
            "posts_created": 3,
            "engagement_achieved": 0.12,
            "time_remaining": "3_days"
        }
        
        progress = await post_service.update_challenge_progress(challenge_id, user_id, progress_data)
        
        assert "progress_updated" in progress
        assert "current_progress" in progress
        assert "goals_remaining" in progress
        mock_challenge_service.update_challenge_progress.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_challenge_completion(self, post_service, mock_challenge_service):
        """Test completing content challenges."""
        challenge_id = str(uuid4())
        user_id = "user123"
        completion_data = {
            "posts_created": 5,
            "engagement_achieved": 0.15,
            "completion_time": datetime.now()
        }
        
        completion = await post_service.complete_challenge(challenge_id, user_id, completion_data)
        
        assert "challenge_completed" in completion
        assert "rewards_awarded" in completion
        assert "completion_time" in completion
        mock_challenge_service.complete_challenge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reward_calculation(self, post_service, mock_reward_service):
        """Test calculating engagement rewards."""
        engagement_data = {
            "engagement_type": "share",
            "engagement_value": 1,
            "content_quality": "high",
            "user_level": "intermediate"
        }
        
        rewards = await post_service.calculate_engagement_rewards(engagement_data)
        
        assert "base_points" in rewards
        assert "bonus_points" in rewards
        assert "total_points" in rewards
        mock_reward_service.calculate_rewards.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reward_distribution(self, post_service, mock_reward_service):
        """Test distributing rewards to users."""
        reward_data = {
            "reward_type": "engagement_bonus",
            "points_per_user": 25,
            "recipients": ["user123", "user456"],
            "content_id": str(uuid4())
        }
        
        distribution = await post_service.distribute_rewards(reward_data)
        
        assert "rewards_distributed" in distribution
        assert "recipients" in distribution
        assert "total_rewards" in distribution
        mock_reward_service.distribute_rewards.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reward_eligibility_validation(self, post_service, mock_reward_service):
        """Test validating reward eligibility."""
        user_id = "user123"
        engagement_data = {
            "engagement_type": "share",
            "content_id": str(uuid4()),
            "timestamp": datetime.now()
        }
        
        eligibility = await post_service.validate_reward_eligibility(user_id, engagement_data)
        
        assert "eligible" in eligibility
        assert "eligibility_reason" in eligibility
        assert "restrictions" in eligibility
        mock_reward_service.validate_reward_eligibility.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_data_persistence(self, post_service, mock_gamification_repository):
        """Test persisting gamification data."""
        gamification_data = SAMPLE_USER_GAMIFICATION_DATA.copy()
        
        result = await post_service.save_gamification_data(gamification_data)
        
        assert "gamification_id" in result
        assert result["saved"] is True
        assert "timestamp" in result
        mock_gamification_repository.save_gamification_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_user_gamification_data_retrieval(self, post_service, mock_gamification_repository):
        """Test retrieving user gamification data."""
        user_id = "user123"
        
        data = await post_service.get_user_gamification_data(user_id)
        
        assert "total_points" in data
        assert "current_level" in data
        assert "badges_earned" in data
        assert "leaderboard_rank" in data
        mock_gamification_repository.get_user_gamification_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_challenge_data_persistence(self, post_service, mock_gamification_repository):
        """Test persisting challenge data."""
        challenge_data = SAMPLE_CHALLENGE_DATA.copy()
        
        result = await post_service.save_challenge_data(challenge_data)
        
        assert "challenge_id" in result
        assert result["saved"] is True
        mock_gamification_repository.save_challenge_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_level_progression(self, post_service, mock_gamification_service):
        """Test user level progression."""
        user_id = "user123"
        current_points = 1250
        
        progression = await post_service.check_level_progression(user_id, current_points)
        
        assert "level_up" in progression
        assert "new_level" in progression
        assert "level_rewards" in progression
        mock_gamification_service.check_level_up.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_badge_awarding(self, post_service, mock_gamification_service):
        """Test awarding badges to users."""
        user_id = "user123"
        badge_criteria = {
            "badge_type": "viral_content",
            "achievement_threshold": 1000,
            "current_value": 1200
        }
        
        badge = await post_service.award_badge(user_id, badge_criteria)
        
        assert "badge_awarded" in badge
        assert "badge_id" in badge
        assert "award_date" in badge
        mock_gamification_service.award_badge.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_leaderboard_ranking(self, post_service, mock_gamification_service):
        """Test calculating leaderboard rankings."""
        ranking_criteria = ["total_points", "engagement_rate", "content_quality"]
        
        rankings = await post_service.calculate_leaderboard_rankings(ranking_criteria)
        
        assert "rankings" in rankings
        assert "total_participants" in rankings
        assert "ranking_criteria" in rankings
        mock_gamification_service.calculate_rankings.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_analytics(self, post_service, mock_gamification_service):
        """Test gamification analytics and insights."""
        time_range = {
            "start": datetime.now() - timedelta(days=30),
            "end": datetime.now()
        }
        
        analytics = await post_service.get_gamification_analytics(time_range)
        
        assert "engagement_metrics" in analytics
        assert "achievement_statistics" in analytics
        assert "leaderboard_insights" in analytics
        mock_gamification_service.get_analytics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_challenge_leaderboard(self, post_service, mock_challenge_service):
        """Test challenge-specific leaderboards."""
        challenge_id = str(uuid4())
        
        leaderboard = await post_service.get_challenge_leaderboard(challenge_id)
        
        assert "participants" in leaderboard
        assert "rankings" in leaderboard
        assert "challenge_stats" in leaderboard
        mock_challenge_service.get_leaderboard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_notifications(self, post_service, mock_gamification_service):
        """Test gamification notification system."""
        notification_data = {
            "user_id": "user123",
            "notification_type": "level_up",
            "notification_data": {"new_level": "expert", "rewards": ["badge", "points"]}
        }
        
        notification = await post_service.send_gamification_notification(notification_data)
        
        assert "notification_sent" in notification
        assert "recipient" in notification
        assert "notification_id" in notification
        mock_gamification_service.send_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_error_handling(self, post_service, mock_gamification_service):
        """Test gamification error handling."""
        mock_gamification_service.award_points.side_effect = Exception("Gamification service unavailable")
        
        engagement_data = {"user_id": "user123", "engagement_type": "like"}
        
        with pytest.raises(Exception):
            await post_service.award_engagement_points(engagement_data)
    
    @pytest.mark.asyncio
    async def test_gamification_validation(self, post_service, mock_gamification_service):
        """Test gamification data validation."""
        gamification_data = SAMPLE_USER_GAMIFICATION_DATA.copy()
        
        validation = await post_service.validate_gamification_data(gamification_data)
        
        assert "validation_passed" in validation
        assert "validation_checks" in validation
        assert "data_integrity" in validation
        mock_gamification_service.validate_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_performance_monitoring(self, post_service, mock_gamification_service):
        """Test monitoring gamification performance."""
        monitoring_config = {
            "engagement_thresholds": {"min_engagement": 0.05, "target_engagement": 0.1},
            "reward_frequency": "daily",
            "performance_metrics": ["points_awarded", "badges_earned", "level_ups"]
        }
        
        monitoring = await post_service.monitor_gamification_performance(monitoring_config)
        
        assert "monitoring_active" in monitoring
        assert "performance_metrics" in monitoring
        assert "performance_trends" in monitoring
        mock_gamification_service.monitor_performance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_automation(self, post_service, mock_gamification_service):
        """Test gamification automation features."""
        automation_config = {
            "auto_award_points": True,
            "auto_check_achievements": True,
            "auto_update_leaderboard": True,
            "auto_send_notifications": True
        }
        
        automation = await post_service.setup_gamification_automation(automation_config)
        
        assert "automation_active" in automation
        assert "automation_rules" in automation
        assert "automation_status" in automation
        mock_gamification_service.setup_automation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gamification_reporting(self, post_service, mock_gamification_service):
        """Test gamification reporting and analytics."""
        report_config = {
            "report_type": "engagement_summary",
            "time_period": "monthly",
            "metrics": ["points_awarded", "achievements_unlocked", "leaderboard_changes"]
        }
        
        report = await post_service.generate_gamification_report(report_config)
        
        assert "report_data" in report
        assert "report_metrics" in report
        assert "report_insights" in report
        mock_gamification_service.generate_report.assert_called_once()
