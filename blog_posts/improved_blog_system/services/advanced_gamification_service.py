"""
Advanced Gamification Service for comprehensive gamification features
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from dataclasses import dataclass
from enum import Enum
import uuid
from decimal import Decimal
import random
import hashlib

from ..models.database import (
    User, UserProfile, UserAchievement, UserBadge, UserLevel, UserPoints,
    UserStreak, UserQuest, UserChallenge, UserLeaderboard, UserReward,
    GamificationEvent, GamificationRule, GamificationAction, GamificationTrigger,
    UserActivity, UserProgress, UserMilestone, UserRanking, UserCompetition
)
from ..core.exceptions import DatabaseError, ValidationError


class AchievementType(Enum):
    """Achievement type enumeration."""
    CONTENT_CREATION = "content_creation"
    SOCIAL_INTERACTION = "social_interaction"
    ENGAGEMENT = "engagement"
    CONSISTENCY = "consistency"
    COMMUNITY = "community"
    EXPERTISE = "expertise"
    COLLABORATION = "collaboration"
    INNOVATION = "innovation"


class BadgeType(Enum):
    """Badge type enumeration."""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"
    LEGENDARY = "legendary"
    SPECIAL = "special"
    SEASONAL = "seasonal"


class QuestType(Enum):
    """Quest type enumeration."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    SPECIAL = "special"
    CHAIN = "chain"
    COMMUNITY = "community"
    PERSONAL = "personal"


class ChallengeType(Enum):
    """Challenge type enumeration."""
    SOLO = "solo"
    TEAM = "team"
    COMMUNITY = "community"
    GLOBAL = "global"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"
    TIME_LIMITED = "time_limited"
    SKILL_BASED = "skill_based"


class RewardType(Enum):
    """Reward type enumeration."""
    POINTS = "points"
    BADGE = "badge"
    ACHIEVEMENT = "achievement"
    LEVEL_UP = "level_up"
    CURRENCY = "currency"
    ITEM = "item"
    PRIVILEGE = "privilege"
    RECOGNITION = "recognition"


@dataclass
class UserGamificationProfile:
    """User gamification profile structure."""
    user_id: str
    total_points: int
    current_level: int
    current_xp: int
    xp_to_next_level: int
    badges_count: int
    achievements_count: int
    current_streak: int
    longest_streak: int
    rank: int
    percentile: float
    recent_activities: List[Dict[str, Any]]


@dataclass
class GamificationMetrics:
    """Gamification metrics structure."""
    total_users: int
    active_users: int
    total_points_awarded: int
    total_achievements_unlocked: int
    total_badges_earned: int
    average_user_level: float
    engagement_rate: float
    retention_rate: float


class AdvancedGamificationService:
    """Service for advanced gamification operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.gamification_cache = {}
        self.achievement_rules = {}
        self.badge_rules = {}
        self.quest_templates = {}
        self.challenge_templates = {}
        self._initialize_gamification_system()
    
    def _initialize_gamification_system(self):
        """Initialize gamification system with rules and templates."""
        try:
            # Initialize achievement rules
            self.achievement_rules = {
                "first_post": {
                    "type": AchievementType.CONTENT_CREATION,
                    "name": "First Steps",
                    "description": "Create your first blog post",
                    "points": 100,
                    "badge": BadgeType.BRONZE,
                    "condition": {"posts_created": 1}
                },
                "post_master": {
                    "type": AchievementType.CONTENT_CREATION,
                    "name": "Post Master",
                    "description": "Create 50 blog posts",
                    "points": 500,
                    "badge": BadgeType.GOLD,
                    "condition": {"posts_created": 50}
                },
                "social_butterfly": {
                    "type": AchievementType.SOCIAL_INTERACTION,
                    "name": "Social Butterfly",
                    "description": "Follow 100 users",
                    "points": 300,
                    "badge": BadgeType.SILVER,
                    "condition": {"users_followed": 100}
                },
                "engagement_expert": {
                    "type": AchievementType.ENGAGEMENT,
                    "name": "Engagement Expert",
                    "description": "Receive 1000 likes on your posts",
                    "points": 1000,
                    "badge": BadgeType.PLATINUM,
                    "condition": {"likes_received": 1000}
                },
                "consistency_champion": {
                    "type": AchievementType.CONSISTENCY,
                    "name": "Consistency Champion",
                    "description": "Maintain a 30-day posting streak",
                    "points": 750,
                    "badge": BadgeType.GOLD,
                    "condition": {"posting_streak": 30}
                },
                "community_leader": {
                    "type": AchievementType.COMMUNITY,
                    "name": "Community Leader",
                    "description": "Create and manage a successful group",
                    "points": 800,
                    "badge": BadgeType.PLATINUM,
                    "condition": {"groups_created": 1, "group_members": 100}
                },
                "expert_author": {
                    "type": AchievementType.EXPERTISE,
                    "name": "Expert Author",
                    "description": "Write 10 high-quality articles",
                    "points": 600,
                    "badge": BadgeType.GOLD,
                    "condition": {"high_quality_posts": 10}
                },
                "collaboration_master": {
                    "type": AchievementType.COLLABORATION,
                    "name": "Collaboration Master",
                    "description": "Collaborate on 5 group projects",
                    "points": 400,
                    "badge": BadgeType.SILVER,
                    "condition": {"collaborations": 5}
                },
                "innovation_pioneer": {
                    "type": AchievementType.INNOVATION,
                    "name": "Innovation Pioneer",
                    "description": "Create unique and innovative content",
                    "points": 900,
                    "badge": BadgeType.DIAMOND,
                    "condition": {"innovative_posts": 5}
                }
            }
            
            # Initialize badge rules
            self.badge_rules = {
                "newcomer": {
                    "type": BadgeType.BRONZE,
                    "name": "Newcomer",
                    "description": "Welcome to the community!",
                    "icon": "🌟",
                    "rarity": "common"
                },
                "rising_star": {
                    "type": BadgeType.SILVER,
                    "name": "Rising Star",
                    "description": "Making your mark in the community",
                    "icon": "⭐",
                    "rarity": "uncommon"
                },
                "community_champion": {
                    "type": BadgeType.GOLD,
                    "name": "Community Champion",
                    "description": "A true champion of the community",
                    "icon": "🏆",
                    "rarity": "rare"
                },
                "legendary_creator": {
                    "type": BadgeType.LEGENDARY,
                    "name": "Legendary Creator",
                    "description": "A legendary content creator",
                    "icon": "👑",
                    "rarity": "legendary"
                },
                "seasonal_celebrant": {
                    "type": BadgeType.SEASONAL,
                    "name": "Seasonal Celebrant",
                    "description": "Celebrating the seasons with content",
                    "icon": "🎉",
                    "rarity": "seasonal"
                }
            }
            
            # Initialize quest templates
            self.quest_templates = {
                "daily_poster": {
                    "type": QuestType.DAILY,
                    "name": "Daily Poster",
                    "description": "Create a new post today",
                    "points": 50,
                    "duration_hours": 24,
                    "condition": {"posts_created": 1}
                },
                "social_connector": {
                    "type": QuestType.DAILY,
                    "name": "Social Connector",
                    "description": "Interact with 5 posts today",
                    "points": 30,
                    "duration_hours": 24,
                    "condition": {"interactions": 5}
                },
                "weekly_creator": {
                    "type": QuestType.WEEKLY,
                    "name": "Weekly Creator",
                    "description": "Create 3 posts this week",
                    "points": 200,
                    "duration_hours": 168,
                    "condition": {"posts_created": 3}
                },
                "community_builder": {
                    "type": QuestType.WEEKLY,
                    "name": "Community Builder",
                    "description": "Join 2 new groups this week",
                    "points": 150,
                    "duration_hours": 168,
                    "condition": {"groups_joined": 2}
                },
                "monthly_explorer": {
                    "type": QuestType.MONTHLY,
                    "name": "Monthly Explorer",
                    "description": "Explore 20 different topics this month",
                    "points": 500,
                    "duration_hours": 720,
                    "condition": {"topics_explored": 20}
                }
            }
            
            # Initialize challenge templates
            self.challenge_templates = {
                "writing_marathon": {
                    "type": ChallengeType.SOLO,
                    "name": "Writing Marathon",
                    "description": "Write 10 posts in 7 days",
                    "points": 1000,
                    "duration_hours": 168,
                    "condition": {"posts_created": 10},
                    "difficulty": "hard"
                },
                "social_butterfly_challenge": {
                    "type": ChallengeType.COMMUNITY,
                    "name": "Social Butterfly Challenge",
                    "description": "Community challenge to increase engagement",
                    "points": 500,
                    "duration_hours": 336,
                    "condition": {"community_engagement": 1000},
                    "difficulty": "medium"
                },
                "innovation_contest": {
                    "type": ChallengeType.COMPETITIVE,
                    "name": "Innovation Contest",
                    "description": "Create the most innovative content",
                    "points": 2000,
                    "duration_hours": 720,
                    "condition": {"innovative_content": 1},
                    "difficulty": "expert"
                }
            }
            
        except Exception as e:
            print(f"Warning: Could not initialize gamification system: {e}")
    
    async def award_points(
        self,
        user_id: str,
        points: int,
        reason: str,
        activity_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Award points to a user."""
        try:
            # Get or create user points record
            points_query = select(UserPoints).where(UserPoints.user_id == user_id)
            points_result = await self.session.execute(points_query)
            user_points = points_result.scalar_one_or_none()
            
            if not user_points:
                user_points = UserPoints(
                    user_id=user_id,
                    total_points=0,
                    current_level=1,
                    current_xp=0,
                    created_at=datetime.utcnow()
                )
                self.session.add(user_points)
            
            # Update points
            user_points.total_points += points
            user_points.current_xp += points
            
            # Check for level up
            new_level = self._calculate_level(user_points.current_xp)
            level_up = new_level > user_points.current_level
            
            if level_up:
                user_points.current_level = new_level
                # Award level up achievement
                await self._award_achievement(user_id, "level_up", {"level": new_level})
            
            user_points.updated_at = datetime.utcnow()
            
            # Create gamification event
            event = GamificationEvent(
                user_id=user_id,
                event_type="points_awarded",
                points=points,
                reason=reason,
                activity_type=activity_type,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            self.session.add(event)
            
            await self.session.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "points_awarded": points,
                "total_points": user_points.total_points,
                "current_level": user_points.current_level,
                "level_up": level_up,
                "new_level": new_level if level_up else None,
                "message": f"Points awarded successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to award points: {str(e)}")
    
    async def award_achievement(
        self,
        user_id: str,
        achievement_key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Award an achievement to a user."""
        try:
            # Check if achievement exists
            if achievement_key not in self.achievement_rules:
                raise ValidationError(f"Achievement '{achievement_key}' not found")
            
            achievement_rule = self.achievement_rules[achievement_key]
            
            # Check if user already has this achievement
            existing_achievement = await self._get_user_achievement(user_id, achievement_key)
            if existing_achievement:
                raise ValidationError("User already has this achievement")
            
            # Create achievement record
            achievement = UserAchievement(
                user_id=user_id,
                achievement_key=achievement_key,
                achievement_type=achievement_rule["type"].value,
                name=achievement_rule["name"],
                description=achievement_rule["description"],
                points=achievement_rule["points"],
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            self.session.add(achievement)
            
            # Award points for achievement
            await self.award_points(
                user_id=user_id,
                points=achievement_rule["points"],
                reason=f"Achievement: {achievement_rule['name']}",
                activity_type="achievement",
                metadata={"achievement_key": achievement_key}
            )
            
            # Award badge if specified
            if "badge" in achievement_rule:
                await self.award_badge(
                    user_id=user_id,
                    badge_key=achievement_key,
                    reason=f"Achievement: {achievement_rule['name']}"
                )
            
            await self.session.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "achievement_key": achievement_key,
                "achievement_name": achievement_rule["name"],
                "points_awarded": achievement_rule["points"],
                "message": "Achievement awarded successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to award achievement: {str(e)}")
    
    async def award_badge(
        self,
        user_id: str,
        badge_key: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Award a badge to a user."""
        try:
            # Check if badge exists
            if badge_key not in self.badge_rules:
                raise ValidationError(f"Badge '{badge_key}' not found")
            
            badge_rule = self.badge_rules[badge_key]
            
            # Check if user already has this badge
            existing_badge = await self._get_user_badge(user_id, badge_key)
            if existing_badge:
                raise ValidationError("User already has this badge")
            
            # Create badge record
            badge = UserBadge(
                user_id=user_id,
                badge_key=badge_key,
                badge_type=badge_rule["type"].value,
                name=badge_rule["name"],
                description=badge_rule["description"],
                icon=badge_rule["icon"],
                rarity=badge_rule["rarity"],
                reason=reason,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            self.session.add(badge)
            
            await self.session.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "badge_key": badge_key,
                "badge_name": badge_rule["name"],
                "badge_icon": badge_rule["icon"],
                "message": "Badge awarded successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to award badge: {str(e)}")
    
    async def create_quest(
        self,
        user_id: str,
        quest_key: str,
        custom_condition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a quest for a user."""
        try:
            # Check if quest template exists
            if quest_key not in self.quest_templates:
                raise ValidationError(f"Quest template '{quest_key}' not found")
            
            quest_template = self.quest_templates[quest_key]
            
            # Check if user already has this quest active
            existing_quest = await self._get_active_user_quest(user_id, quest_key)
            if existing_quest:
                raise ValidationError("User already has this quest active")
            
            # Generate quest ID
            quest_id = str(uuid.uuid4())
            
            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=quest_template["duration_hours"])
            
            # Create quest record
            quest = UserQuest(
                quest_id=quest_id,
                user_id=user_id,
                quest_key=quest_key,
                quest_type=quest_template["type"].value,
                name=quest_template["name"],
                description=quest_template["description"],
                points_reward=quest_template["points"],
                condition=quest_template["condition"],
                custom_condition=custom_condition,
                status="active",
                progress=0,
                expires_at=expires_at,
                created_at=datetime.utcnow()
            )
            self.session.add(quest)
            
            await self.session.commit()
            
            return {
                "success": True,
                "quest_id": quest_id,
                "user_id": user_id,
                "quest_name": quest_template["name"],
                "points_reward": quest_template["points"],
                "expires_at": expires_at.isoformat(),
                "message": "Quest created successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create quest: {str(e)}")
    
    async def update_quest_progress(
        self,
        user_id: str,
        quest_key: str,
        progress_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update quest progress for a user."""
        try:
            # Get active quest
            quest = await self._get_active_user_quest(user_id, quest_key)
            if not quest:
                raise ValidationError("No active quest found")
            
            # Check if quest has expired
            if quest.expires_at < datetime.utcnow():
                quest.status = "expired"
                await self.session.commit()
                raise ValidationError("Quest has expired")
            
            # Update progress
            quest.progress += 1
            quest.updated_at = datetime.utcnow()
            
            # Check if quest is completed
            quest_completed = False
            if self._check_quest_completion(quest, progress_data):
                quest.status = "completed"
                quest.completed_at = datetime.utcnow()
                quest_completed = True
                
                # Award quest rewards
                await self.award_points(
                    user_id=user_id,
                    points=quest.points_reward,
                    reason=f"Quest completed: {quest.name}",
                    activity_type="quest_completion",
                    metadata={"quest_id": quest.quest_id}
                )
            
            await self.session.commit()
            
            return {
                "success": True,
                "quest_id": quest.quest_id,
                "progress": quest.progress,
                "completed": quest_completed,
                "points_awarded": quest.points_reward if quest_completed else 0,
                "message": "Quest progress updated successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update quest progress: {str(e)}")
    
    async def create_challenge(
        self,
        challenge_key: str,
        creator_id: str,
        participants: Optional[List[str]] = None,
        custom_condition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a challenge."""
        try:
            # Check if challenge template exists
            if challenge_key not in self.challenge_templates:
                raise ValidationError(f"Challenge template '{challenge_key}' not found")
            
            challenge_template = self.challenge_templates[challenge_key]
            
            # Generate challenge ID
            challenge_id = str(uuid.uuid4())
            
            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=challenge_template["duration_hours"])
            
            # Create challenge record
            challenge = UserChallenge(
                challenge_id=challenge_id,
                challenge_key=challenge_key,
                challenge_type=challenge_template["type"].value,
                name=challenge_template["name"],
                description=challenge_template["description"],
                points_reward=challenge_template["points"],
                condition=challenge_template["condition"],
                custom_condition=custom_condition,
                creator_id=creator_id,
                participants=participants or [],
                status="active",
                difficulty=challenge_template["difficulty"],
                expires_at=expires_at,
                created_at=datetime.utcnow()
            )
            self.session.add(challenge)
            
            await self.session.commit()
            
            return {
                "success": True,
                "challenge_id": challenge_id,
                "challenge_name": challenge_template["name"],
                "points_reward": challenge_template["points"],
                "participants": participants or [],
                "expires_at": expires_at.isoformat(),
                "message": "Challenge created successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create challenge: {str(e)}")
    
    async def join_challenge(
        self,
        user_id: str,
        challenge_id: str
    ) -> Dict[str, Any]:
        """Join a challenge."""
        try:
            # Get challenge
            challenge_query = select(UserChallenge).where(UserChallenge.challenge_id == challenge_id)
            challenge_result = await self.session.execute(challenge_query)
            challenge = challenge_result.scalar_one_or_none()
            
            if not challenge:
                raise ValidationError("Challenge not found")
            
            # Check if challenge is still active
            if challenge.status != "active":
                raise ValidationError("Challenge is not active")
            
            # Check if challenge has expired
            if challenge.expires_at < datetime.utcnow():
                challenge.status = "expired"
                await self.session.commit()
                raise ValidationError("Challenge has expired")
            
            # Check if user is already a participant
            if user_id in challenge.participants:
                raise ValidationError("User is already a participant")
            
            # Add user to participants
            challenge.participants.append(user_id)
            challenge.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            return {
                "success": True,
                "challenge_id": challenge_id,
                "user_id": user_id,
                "participants_count": len(challenge.participants),
                "message": "Successfully joined challenge"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to join challenge: {str(e)}")
    
    async def update_streak(
        self,
        user_id: str,
        activity_type: str = "posting"
    ) -> Dict[str, Any]:
        """Update user streak."""
        try:
            # Get or create streak record
            streak_query = select(UserStreak).where(
                and_(UserStreak.user_id == user_id, UserStreak.activity_type == activity_type)
            )
            streak_result = await self.session.execute(streak_query)
            streak = streak_result.scalar_one_or_none()
            
            if not streak:
                streak = UserStreak(
                    user_id=user_id,
                    activity_type=activity_type,
                    current_streak=0,
                    longest_streak=0,
                    last_activity_date=None,
                    created_at=datetime.utcnow()
                )
                self.session.add(streak)
            
            today = datetime.utcnow().date()
            
            # Check if user already has activity today
            if streak.last_activity_date == today:
                return {
                    "success": True,
                    "user_id": user_id,
                    "current_streak": streak.current_streak,
                    "longest_streak": streak.longest_streak,
                    "message": "Streak already updated today"
                }
            
            # Check if streak should continue or reset
            if streak.last_activity_date:
                days_diff = (today - streak.last_activity_date).days
                if days_diff == 1:
                    # Continue streak
                    streak.current_streak += 1
                elif days_diff > 1:
                    # Reset streak
                    streak.current_streak = 1
                else:
                    # Same day, no change
                    pass
            else:
                # First activity
                streak.current_streak = 1
            
            # Update longest streak
            if streak.current_streak > streak.longest_streak:
                streak.longest_streak = streak.current_streak
            
            streak.last_activity_date = today
            streak.updated_at = datetime.utcnow()
            
            # Award streak bonuses
            streak_bonus = self._calculate_streak_bonus(streak.current_streak)
            if streak_bonus > 0:
                await self.award_points(
                    user_id=user_id,
                    points=streak_bonus,
                    reason=f"Streak bonus: {streak.current_streak} days",
                    activity_type="streak_bonus",
                    metadata={"streak_days": streak.current_streak, "activity_type": activity_type}
                )
            
            await self.session.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "current_streak": streak.current_streak,
                "longest_streak": streak.longest_streak,
                "streak_bonus": streak_bonus,
                "message": "Streak updated successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update streak: {str(e)}")
    
    async def get_user_gamification_profile(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get user's gamification profile."""
        try:
            # Get user points
            points_query = select(UserPoints).where(UserPoints.user_id == user_id)
            points_result = await self.session.execute(points_query)
            user_points = points_result.scalar_one_or_none()
            
            if not user_points:
                user_points = UserPoints(
                    user_id=user_id,
                    total_points=0,
                    current_level=1,
                    current_xp=0,
                    created_at=datetime.utcnow()
                )
                self.session.add(user_points)
                await self.session.commit()
            
            # Get badges count
            badges_query = select(func.count(UserBadge.id)).where(UserBadge.user_id == user_id)
            badges_result = await self.session.execute(badges_query)
            badges_count = badges_result.scalar()
            
            # Get achievements count
            achievements_query = select(func.count(UserAchievement.id)).where(UserAchievement.user_id == user_id)
            achievements_result = await self.session.execute(achievements_query)
            achievements_count = achievements_result.scalar()
            
            # Get current streak
            streak_query = select(UserStreak).where(
                and_(UserStreak.user_id == user_id, UserStreak.activity_type == "posting")
            )
            streak_result = await self.session.execute(streak_query)
            streak = streak_result.scalar_one_or_none()
            
            current_streak = streak.current_streak if streak else 0
            longest_streak = streak.longest_streak if streak else 0
            
            # Calculate rank and percentile
            rank_data = await self._calculate_user_rank(user_id)
            
            # Get recent activities
            recent_activities = await self._get_recent_activities(user_id, limit=10)
            
            profile = UserGamificationProfile(
                user_id=user_id,
                total_points=user_points.total_points,
                current_level=user_points.current_level,
                current_xp=user_points.current_xp,
                xp_to_next_level=self._calculate_xp_to_next_level(user_points.current_xp),
                badges_count=badges_count,
                achievements_count=achievements_count,
                current_streak=current_streak,
                longest_streak=longest_streak,
                rank=rank_data["rank"],
                percentile=rank_data["percentile"],
                recent_activities=recent_activities
            )
            
            return {
                "success": True,
                "data": profile.__dict__
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get gamification profile: {str(e)}")
    
    async def get_leaderboard(
        self,
        leaderboard_type: str = "points",
        limit: int = 100,
        time_period: str = "all_time"
    ) -> Dict[str, Any]:
        """Get leaderboard."""
        try:
            # Build query based on type
            if leaderboard_type == "points":
                query = select(UserPoints).order_by(desc(UserPoints.total_points))
            elif leaderboard_type == "level":
                query = select(UserPoints).order_by(desc(UserPoints.current_level), desc(UserPoints.current_xp))
            elif leaderboard_type == "streak":
                query = select(UserStreak).where(UserStreak.activity_type == "posting").order_by(desc(UserStreak.current_streak))
            else:
                raise ValidationError(f"Invalid leaderboard type: {leaderboard_type}")
            
            # Add time period filter if needed
            if time_period != "all_time":
                # This would implement time-based filtering
                pass
            
            # Add limit
            query = query.limit(limit)
            
            # Execute query
            result = await self.session.execute(query)
            leaderboard_data = result.scalars().all()
            
            # Format leaderboard
            formatted_leaderboard = []
            for i, item in enumerate(leaderboard_data, 1):
                if leaderboard_type == "points":
                    formatted_leaderboard.append({
                        "rank": i,
                        "user_id": item.user_id,
                        "total_points": item.total_points,
                        "current_level": item.current_level
                    })
                elif leaderboard_type == "level":
                    formatted_leaderboard.append({
                        "rank": i,
                        "user_id": item.user_id,
                        "current_level": item.current_level,
                        "current_xp": item.current_xp
                    })
                elif leaderboard_type == "streak":
                    formatted_leaderboard.append({
                        "rank": i,
                        "user_id": item.user_id,
                        "current_streak": item.current_streak,
                        "longest_streak": item.longest_streak
                    })
            
            return {
                "success": True,
                "data": {
                    "leaderboard": formatted_leaderboard,
                    "type": leaderboard_type,
                    "time_period": time_period,
                    "total_entries": len(formatted_leaderboard)
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get leaderboard: {str(e)}")
    
    async def get_gamification_stats(self) -> Dict[str, Any]:
        """Get gamification system statistics."""
        try:
            # Get total users
            users_query = select(func.count(User.id))
            users_result = await self.session.execute(users_query)
            total_users = users_result.scalar()
            
            # Get active users (users with points)
            active_users_query = select(func.count(UserPoints.id))
            active_users_result = await self.session.execute(active_users_query)
            active_users = active_users_result.scalar()
            
            # Get total points awarded
            total_points_query = select(func.sum(UserPoints.total_points))
            total_points_result = await self.session.execute(total_points_query)
            total_points_awarded = total_points_result.scalar() or 0
            
            # Get total achievements unlocked
            achievements_query = select(func.count(UserAchievement.id))
            achievements_result = await self.session.execute(achievements_query)
            total_achievements_unlocked = achievements_result.scalar()
            
            # Get total badges earned
            badges_query = select(func.count(UserBadge.id))
            badges_result = await self.session.execute(badges_query)
            total_badges_earned = badges_result.scalar()
            
            # Get average user level
            avg_level_query = select(func.avg(UserPoints.current_level))
            avg_level_result = await self.session.execute(avg_level_query)
            average_user_level = float(avg_level_result.scalar() or 1)
            
            # Calculate engagement rate
            engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
            
            # Calculate retention rate (placeholder)
            retention_rate = 85.0  # This would be calculated based on actual data
            
            metrics = GamificationMetrics(
                total_users=total_users,
                active_users=active_users,
                total_points_awarded=total_points_awarded,
                total_achievements_unlocked=total_achievements_unlocked,
                total_badges_earned=total_badges_earned,
                average_user_level=average_user_level,
                engagement_rate=engagement_rate,
                retention_rate=retention_rate
            )
            
            return {
                "success": True,
                "data": metrics.__dict__
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get gamification stats: {str(e)}")
    
    def _calculate_level(self, xp: int) -> int:
        """Calculate level based on XP."""
        # Simple level calculation: 100 XP per level
        return (xp // 100) + 1
    
    def _calculate_xp_to_next_level(self, current_xp: int) -> int:
        """Calculate XP needed for next level."""
        current_level = self._calculate_level(current_xp)
        next_level_xp = current_level * 100
        return next_level_xp - current_xp
    
    def _calculate_streak_bonus(self, streak_days: int) -> int:
        """Calculate streak bonus points."""
        if streak_days >= 30:
            return 100
        elif streak_days >= 14:
            return 50
        elif streak_days >= 7:
            return 25
        elif streak_days >= 3:
            return 10
        else:
            return 0
    
    def _check_quest_completion(self, quest: UserQuest, progress_data: Dict[str, Any]) -> bool:
        """Check if quest is completed."""
        # This would implement quest completion logic
        # For now, simple progress-based completion
        return quest.progress >= 1
    
    async def _get_user_achievement(self, user_id: str, achievement_key: str) -> Optional[UserAchievement]:
        """Get user achievement."""
        try:
            query = select(UserAchievement).where(
                and_(UserAchievement.user_id == user_id, UserAchievement.achievement_key == achievement_key)
            )
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _get_user_badge(self, user_id: str, badge_key: str) -> Optional[UserBadge]:
        """Get user badge."""
        try:
            query = select(UserBadge).where(
                and_(UserBadge.user_id == user_id, UserBadge.badge_key == badge_key)
            )
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _get_active_user_quest(self, user_id: str, quest_key: str) -> Optional[UserQuest]:
        """Get active user quest."""
        try:
            query = select(UserQuest).where(
                and_(
                    UserQuest.user_id == user_id,
                    UserQuest.quest_key == quest_key,
                    UserQuest.status == "active"
                )
            )
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _calculate_user_rank(self, user_id: str) -> Dict[str, Any]:
        """Calculate user rank and percentile."""
        try:
            # Get user's total points
            user_points_query = select(UserPoints.total_points).where(UserPoints.user_id == user_id)
            user_points_result = await self.session.execute(user_points_query)
            user_points = user_points_result.scalar() or 0
            
            # Count users with more points
            higher_users_query = select(func.count(UserPoints.id)).where(UserPoints.total_points > user_points)
            higher_users_result = await self.session.execute(higher_users_query)
            higher_users = higher_users_result.scalar()
            
            # Get total users
            total_users_query = select(func.count(UserPoints.id))
            total_users_result = await self.session.execute(total_users_query)
            total_users = total_users_result.scalar()
            
            rank = higher_users + 1
            percentile = ((total_users - rank + 1) / total_users * 100) if total_users > 0 else 0
            
            return {
                "rank": rank,
                "percentile": percentile
            }
        except Exception:
            return {"rank": 1, "percentile": 100.0}
    
    async def _get_recent_activities(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent user activities."""
        try:
            query = select(GamificationEvent).where(
                GamificationEvent.user_id == user_id
            ).order_by(desc(GamificationEvent.created_at)).limit(limit)
            
            result = await self.session.execute(query)
            activities = result.scalars().all()
            
            return [
                {
                    "event_type": activity.event_type,
                    "points": activity.points,
                    "reason": activity.reason,
                    "activity_type": activity.activity_type,
                    "created_at": activity.created_at.isoformat()
                }
                for activity in activities
            ]
        except Exception:
            return []
























