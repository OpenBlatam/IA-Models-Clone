"""
Advanced Gamification API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_gamification_service import AdvancedGamificationService, AchievementType, BadgeType, QuestType, ChallengeType, RewardType
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class AwardPointsRequest(BaseModel):
    """Request model for awarding points."""
    points: int = Field(..., ge=1, description="Points to award")
    reason: str = Field(..., description="Reason for awarding points")
    activity_type: str = Field(default="general", description="Activity type")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class AwardAchievementRequest(BaseModel):
    """Request model for awarding achievement."""
    achievement_key: str = Field(..., description="Achievement key")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class AwardBadgeRequest(BaseModel):
    """Request model for awarding badge."""
    badge_key: str = Field(..., description="Badge key")
    reason: str = Field(..., description="Reason for awarding badge")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class CreateQuestRequest(BaseModel):
    """Request model for creating quest."""
    quest_key: str = Field(..., description="Quest template key")
    custom_condition: Optional[Dict[str, Any]] = Field(default=None, description="Custom quest condition")


class UpdateQuestProgressRequest(BaseModel):
    """Request model for updating quest progress."""
    quest_key: str = Field(..., description="Quest key")
    progress_data: Dict[str, Any] = Field(..., description="Progress data")


class CreateChallengeRequest(BaseModel):
    """Request model for creating challenge."""
    challenge_key: str = Field(..., description="Challenge template key")
    participants: Optional[List[str]] = Field(default=None, description="Initial participants")
    custom_condition: Optional[Dict[str, Any]] = Field(default=None, description="Custom challenge condition")


class JoinChallengeRequest(BaseModel):
    """Request model for joining challenge."""
    challenge_id: str = Field(..., description="Challenge ID")


class UpdateStreakRequest(BaseModel):
    """Request model for updating streak."""
    activity_type: str = Field(default="posting", description="Activity type")


class LeaderboardRequest(BaseModel):
    """Request model for leaderboard."""
    leaderboard_type: str = Field(default="points", description="Leaderboard type")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of entries")
    time_period: str = Field(default="all_time", description="Time period")


async def get_gamification_service(session: DatabaseSessionDep) -> AdvancedGamificationService:
    """Get gamification service instance."""
    return AdvancedGamificationService(session)


@router.post("/points/award", response_model=Dict[str, Any])
async def award_points(
    request: AwardPointsRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Award points to a user."""
    try:
        result = await gamification_service.award_points(
            user_id=str(current_user.id),
            points=request.points,
            reason=request.reason,
            activity_type=request.activity_type,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Points awarded successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to award points"
        )


@router.post("/achievements/award", response_model=Dict[str, Any])
async def award_achievement(
    request: AwardAchievementRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Award an achievement to a user."""
    try:
        result = await gamification_service.award_achievement(
            user_id=str(current_user.id),
            achievement_key=request.achievement_key,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Achievement awarded successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to award achievement"
        )


@router.post("/badges/award", response_model=Dict[str, Any])
async def award_badge(
    request: AwardBadgeRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Award a badge to a user."""
    try:
        result = await gamification_service.award_badge(
            user_id=str(current_user.id),
            badge_key=request.badge_key,
            reason=request.reason,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Badge awarded successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to award badge"
        )


@router.post("/quests/create", response_model=Dict[str, Any])
async def create_quest(
    request: CreateQuestRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a quest for a user."""
    try:
        result = await gamification_service.create_quest(
            user_id=str(current_user.id),
            quest_key=request.quest_key,
            custom_condition=request.custom_condition
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quest created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create quest"
        )


@router.put("/quests/progress", response_model=Dict[str, Any])
async def update_quest_progress(
    request: UpdateQuestProgressRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Update quest progress for a user."""
    try:
        result = await gamification_service.update_quest_progress(
            user_id=str(current_user.id),
            quest_key=request.quest_key,
            progress_data=request.progress_data
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quest progress updated successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update quest progress"
        )


@router.post("/challenges/create", response_model=Dict[str, Any])
async def create_challenge(
    request: CreateChallengeRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a challenge."""
    try:
        result = await gamification_service.create_challenge(
            challenge_key=request.challenge_key,
            creator_id=str(current_user.id),
            participants=request.participants,
            custom_condition=request.custom_condition
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Challenge created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create challenge"
        )


@router.post("/challenges/join", response_model=Dict[str, Any])
async def join_challenge(
    request: JoinChallengeRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Join a challenge."""
    try:
        result = await gamification_service.join_challenge(
            user_id=str(current_user.id),
            challenge_id=request.challenge_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Successfully joined challenge"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to join challenge"
        )


@router.put("/streak/update", response_model=Dict[str, Any])
async def update_streak(
    request: UpdateStreakRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Update user streak."""
    try:
        result = await gamification_service.update_streak(
            user_id=str(current_user.id),
            activity_type=request.activity_type
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Streak updated successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update streak"
        )


@router.get("/profile", response_model=Dict[str, Any])
async def get_user_gamification_profile(
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get user's gamification profile."""
    try:
        result = await gamification_service.get_user_gamification_profile(
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Gamification profile retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gamification profile"
        )


@router.get("/profile/{user_id}", response_model=Dict[str, Any])
async def get_user_gamification_profile_by_id(
    user_id: str,
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get user's gamification profile by ID."""
    try:
        result = await gamification_service.get_user_gamification_profile(user_id=user_id)
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Gamification profile retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gamification profile"
        )


@router.post("/leaderboard", response_model=Dict[str, Any])
async def get_leaderboard(
    request: LeaderboardRequest = Depends(),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get leaderboard."""
    try:
        result = await gamification_service.get_leaderboard(
            leaderboard_type=request.leaderboard_type,
            limit=request.limit,
            time_period=request.time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Leaderboard retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get leaderboard"
        )


@router.get("/leaderboard", response_model=Dict[str, Any])
async def get_leaderboard_get(
    leaderboard_type: str = Query(default="points", description="Leaderboard type"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of entries"),
    time_period: str = Query(default="all_time", description="Time period"),
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get leaderboard via GET request."""
    try:
        result = await gamification_service.get_leaderboard(
            leaderboard_type=leaderboard_type,
            limit=limit,
            time_period=time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Leaderboard retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get leaderboard"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_gamification_stats(
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get gamification system statistics."""
    try:
        result = await gamification_service.get_gamification_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Gamification statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gamification statistics"
        )


@router.get("/achievement-types", response_model=Dict[str, Any])
async def get_achievement_types():
    """Get available achievement types."""
    achievement_types = {
        "content_creation": {
            "name": "Content Creation",
            "description": "Achievements for creating content",
            "icon": "📝",
            "color": "#4CAF50"
        },
        "social_interaction": {
            "name": "Social Interaction",
            "description": "Achievements for social activities",
            "icon": "👥",
            "color": "#2196F3"
        },
        "engagement": {
            "name": "Engagement",
            "description": "Achievements for user engagement",
            "icon": "❤️",
            "color": "#E91E63"
        },
        "consistency": {
            "name": "Consistency",
            "description": "Achievements for consistent activity",
            "icon": "🔥",
            "color": "#FF9800"
        },
        "community": {
            "name": "Community",
            "description": "Achievements for community building",
            "icon": "🏘️",
            "color": "#9C27B0"
        },
        "expertise": {
            "name": "Expertise",
            "description": "Achievements for expertise and knowledge",
            "icon": "🎓",
            "color": "#607D8B"
        },
        "collaboration": {
            "name": "Collaboration",
            "description": "Achievements for collaboration",
            "icon": "🤝",
            "color": "#795548"
        },
        "innovation": {
            "name": "Innovation",
            "description": "Achievements for innovation and creativity",
            "icon": "💡",
            "color": "#FFC107"
        }
    }
    
    return {
        "success": True,
        "data": {
            "achievement_types": achievement_types,
            "total_types": len(achievement_types)
        },
        "message": "Achievement types retrieved successfully"
    }


@router.get("/badge-types", response_model=Dict[str, Any])
async def get_badge_types():
    """Get available badge types."""
    badge_types = {
        "bronze": {
            "name": "Bronze",
            "description": "Basic level badge",
            "icon": "🥉",
            "rarity": "common",
            "color": "#CD7F32"
        },
        "silver": {
            "name": "Silver",
            "description": "Intermediate level badge",
            "icon": "🥈",
            "rarity": "uncommon",
            "color": "#C0C0C0"
        },
        "gold": {
            "name": "Gold",
            "description": "Advanced level badge",
            "icon": "🥇",
            "rarity": "rare",
            "color": "#FFD700"
        },
        "platinum": {
            "name": "Platinum",
            "description": "Expert level badge",
            "icon": "💎",
            "rarity": "epic",
            "color": "#E5E4E2"
        },
        "diamond": {
            "name": "Diamond",
            "description": "Master level badge",
            "icon": "💠",
            "rarity": "legendary",
            "color": "#B9F2FF"
        },
        "legendary": {
            "name": "Legendary",
            "description": "Legendary level badge",
            "icon": "👑",
            "rarity": "mythic",
            "color": "#FF6B35"
        },
        "special": {
            "name": "Special",
            "description": "Special event badge",
            "icon": "⭐",
            "rarity": "special",
            "color": "#FF1493"
        },
        "seasonal": {
            "name": "Seasonal",
            "description": "Seasonal event badge",
            "icon": "🎉",
            "rarity": "seasonal",
            "color": "#32CD32"
        }
    }
    
    return {
        "success": True,
        "data": {
            "badge_types": badge_types,
            "total_types": len(badge_types)
        },
        "message": "Badge types retrieved successfully"
    }


@router.get("/quest-types", response_model=Dict[str, Any])
async def get_quest_types():
    """Get available quest types."""
    quest_types = {
        "daily": {
            "name": "Daily Quest",
            "description": "Quests that reset daily",
            "icon": "📅",
            "duration": "24 hours",
            "frequency": "daily"
        },
        "weekly": {
            "name": "Weekly Quest",
            "description": "Quests that reset weekly",
            "icon": "📆",
            "duration": "7 days",
            "frequency": "weekly"
        },
        "monthly": {
            "name": "Monthly Quest",
            "description": "Quests that reset monthly",
            "icon": "🗓️",
            "duration": "30 days",
            "frequency": "monthly"
        },
        "seasonal": {
            "name": "Seasonal Quest",
            "description": "Quests for special seasons",
            "icon": "🌸",
            "duration": "3 months",
            "frequency": "seasonal"
        },
        "special": {
            "name": "Special Quest",
            "description": "Special event quests",
            "icon": "🎊",
            "duration": "variable",
            "frequency": "special"
        },
        "chain": {
            "name": "Chain Quest",
            "description": "Sequential quest chains",
            "icon": "🔗",
            "duration": "variable",
            "frequency": "chain"
        },
        "community": {
            "name": "Community Quest",
            "description": "Community-wide quests",
            "icon": "🌍",
            "duration": "variable",
            "frequency": "community"
        },
        "personal": {
            "name": "Personal Quest",
            "description": "Personalized quests",
            "icon": "👤",
            "duration": "variable",
            "frequency": "personal"
        }
    }
    
    return {
        "success": True,
        "data": {
            "quest_types": quest_types,
            "total_types": len(quest_types)
        },
        "message": "Quest types retrieved successfully"
    }


@router.get("/challenge-types", response_model=Dict[str, Any])
async def get_challenge_types():
    """Get available challenge types."""
    challenge_types = {
        "solo": {
            "name": "Solo Challenge",
            "description": "Individual challenges",
            "icon": "👤",
            "participants": "1",
            "competition": "self"
        },
        "team": {
            "name": "Team Challenge",
            "description": "Team-based challenges",
            "icon": "👥",
            "participants": "2-10",
            "competition": "team"
        },
        "community": {
            "name": "Community Challenge",
            "description": "Community-wide challenges",
            "icon": "🌍",
            "participants": "unlimited",
            "competition": "community"
        },
        "global": {
            "name": "Global Challenge",
            "description": "Global platform challenges",
            "icon": "🌐",
            "participants": "unlimited",
            "competition": "global"
        },
        "competitive": {
            "name": "Competitive Challenge",
            "description": "Competitive challenges with rankings",
            "icon": "🏆",
            "participants": "unlimited",
            "competition": "competitive"
        },
        "collaborative": {
            "name": "Collaborative Challenge",
            "description": "Collaborative challenges",
            "icon": "🤝",
            "participants": "unlimited",
            "competition": "collaborative"
        },
        "time_limited": {
            "name": "Time Limited Challenge",
            "description": "Challenges with time constraints",
            "icon": "⏰",
            "participants": "unlimited",
            "competition": "time-based"
        },
        "skill_based": {
            "name": "Skill Based Challenge",
            "description": "Challenges based on skill levels",
            "icon": "🎯",
            "participants": "unlimited",
            "competition": "skill-based"
        }
    }
    
    return {
        "success": True,
        "data": {
            "challenge_types": challenge_types,
            "total_types": len(challenge_types)
        },
        "message": "Challenge types retrieved successfully"
    }


@router.get("/reward-types", response_model=Dict[str, Any])
async def get_reward_types():
    """Get available reward types."""
    reward_types = {
        "points": {
            "name": "Points",
            "description": "Experience points",
            "icon": "⭐",
            "value": "numeric"
        },
        "badge": {
            "name": "Badge",
            "description": "Achievement badges",
            "icon": "🏅",
            "value": "badge"
        },
        "achievement": {
            "name": "Achievement",
            "description": "Achievement unlocks",
            "icon": "🎖️",
            "value": "achievement"
        },
        "level_up": {
            "name": "Level Up",
            "description": "Level progression",
            "icon": "⬆️",
            "value": "level"
        },
        "currency": {
            "name": "Currency",
            "description": "Virtual currency",
            "icon": "💰",
            "value": "currency"
        },
        "item": {
            "name": "Item",
            "description": "Virtual items",
            "icon": "🎁",
            "value": "item"
        },
        "privilege": {
            "name": "Privilege",
            "description": "Special privileges",
            "icon": "👑",
            "value": "privilege"
        },
        "recognition": {
            "name": "Recognition",
            "description": "Public recognition",
            "icon": "🌟",
            "value": "recognition"
        }
    }
    
    return {
        "success": True,
        "data": {
            "reward_types": reward_types,
            "total_types": len(reward_types)
        },
        "message": "Reward types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_gamification_health(
    gamification_service: AdvancedGamificationService = Depends(get_gamification_service),
    current_user: CurrentUserDep = Depends()
):
    """Get gamification system health status."""
    try:
        # Get gamification stats
        stats = await gamification_service.get_gamification_stats()
        
        # Calculate health metrics
        total_users = stats["data"].get("total_users", 0)
        active_users = stats["data"].get("active_users", 0)
        total_points_awarded = stats["data"].get("total_points_awarded", 0)
        total_achievements_unlocked = stats["data"].get("total_achievements_unlocked", 0)
        total_badges_earned = stats["data"].get("total_badges_earned", 0)
        average_user_level = stats["data"].get("average_user_level", 1)
        engagement_rate = stats["data"].get("engagement_rate", 0)
        retention_rate = stats["data"].get("retention_rate", 0)
        
        # Calculate health score
        health_score = 100
        
        # Check user engagement
        if total_users > 0:
            active_ratio = active_users / total_users
            if active_ratio < 0.3:
                health_score -= 30
            elif active_ratio < 0.5:
                health_score -= 20
            elif active_ratio > 0.8:
                health_score -= 10
        
        # Check achievement system
        if total_users > 0:
            achievement_ratio = total_achievements_unlocked / total_users
            if achievement_ratio < 0.5:
                health_score -= 20
            elif achievement_ratio > 3.0:
                health_score -= 10
        
        # Check badge system
        if total_users > 0:
            badge_ratio = total_badges_earned / total_users
            if badge_ratio < 0.3:
                health_score -= 15
        
        # Check level progression
        if average_user_level < 2:
            health_score -= 15
        elif average_user_level > 10:
            health_score -= 5
        
        # Check engagement rate
        if engagement_rate < 50:
            health_score -= 20
        elif engagement_rate > 90:
            health_score -= 5
        
        # Check retention rate
        if retention_rate < 70:
            health_score -= 15
        elif retention_rate > 95:
            health_score -= 5
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_users": total_users,
                "active_users": active_users,
                "total_points_awarded": total_points_awarded,
                "total_achievements_unlocked": total_achievements_unlocked,
                "total_badges_earned": total_badges_earned,
                "average_user_level": average_user_level,
                "engagement_rate": engagement_rate,
                "retention_rate": retention_rate,
                "active_user_ratio": active_ratio if total_users > 0 else 0,
                "achievement_ratio": achievement_ratio if total_users > 0 else 0,
                "badge_ratio": badge_ratio if total_users > 0 else 0,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Gamification health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gamification health status"
        )
























