"""
Gamification system for student engagement and motivation.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BadgeType(str, Enum):
    """Types of badges students can earn."""
    FIRST_QUESTION = "first_question"
    DAILY_STREAK = "daily_streak"
    WEEKLY_STREAK = "weekly_streak"
    SUBJECT_MASTER = "subject_master"
    QUIZ_CHAMPION = "quiz_champion"
    PERFECT_SCORE = "perfect_score"
    HELPER = "helper"
    EXPLORER = "explorer"
    DEDICATED = "dedicated"


@dataclass
class Badge:
    """Represents a badge achievement."""
    badge_type: BadgeType
    name: str
    description: str
    icon: str
    earned_at: datetime
    points: int = 10


@dataclass
class Achievement:
    """Represents an achievement milestone."""
    achievement_id: str
    name: str
    description: str
    requirement: str
    reward_points: int
    unlocked: bool = False
    unlocked_at: Optional[datetime] = None


class GamificationSystem:
    """
    Gamification system to increase student engagement.
    """
    
    def __init__(self):
        self.student_badges: Dict[str, List[Badge]] = {}
        self.student_points: Dict[str, int] = {}
        self.student_levels: Dict[str, int] = {}
        self.student_streaks: Dict[str, Dict[str, int]] = {}
        self.achievements: Dict[str, List[Achievement]] = {}
        self._initialize_achievements()
    
    def _initialize_achievements(self):
        """Initialize default achievements."""
        default_achievements = [
            Achievement(
                achievement_id="first_steps",
                name="Primeros Pasos",
                description="Haz tu primera pregunta",
                requirement="ask_question:1",
                reward_points=50
            ),
            Achievement(
                achievement_id="curious_mind",
                name="Mente Curiosa",
                description="Haz 10 preguntas",
                requirement="ask_question:10",
                reward_points=100
            ),
            Achievement(
                achievement_id="knowledge_seeker",
                name="Buscador de Conocimiento",
                description="Haz 50 preguntas",
                requirement="ask_question:50",
                reward_points=500
            ),
            Achievement(
                achievement_id="quiz_master",
                name="Maestro de Quizzes",
                description="Completa 10 quizzes",
                requirement="complete_quiz:10",
                reward_points=300
            ),
            Achievement(
                achievement_id="subject_expert",
                name="Experto en Materia",
                description="Domina 5 materias diferentes",
                requirement="master_subject:5",
                reward_points=400
            ),
            Achievement(
                achievement_id="streak_warrior",
                name="Guerrero de Racha",
                description="Mantén una racha de 7 días",
                requirement="daily_streak:7",
                reward_points=200
            )
        ]
        
        # Store achievements template
        self.achievements_template = default_achievements
    
    def record_action(
        self,
        student_id: str,
        action_type: str,
        metadata: Optional[Dict] = None
    ):
        """
        Record a student action and check for badges/achievements.
        
        Args:
            student_id: Student identifier
            action_type: Type of action (ask_question, complete_quiz, etc.)
            metadata: Additional metadata about the action
        """
        if student_id not in self.student_points:
            self.student_points[student_id] = 0
            self.student_levels[student_id] = 1
            self.student_badges[student_id] = []
            self.student_streaks[student_id] = {"daily": 0, "weekly": 0}
            self.achievements[student_id] = [
                Achievement(**a.__dict__) for a in self.achievements_template
            ]
        
        # Update streak
        self._update_streak(student_id)
        
        # Check for badges
        self._check_badges(student_id, action_type, metadata)
        
        # Check for achievements
        self._check_achievements(student_id, action_type)
        
        # Update level based on points
        self._update_level(student_id)
    
    def _update_streak(self, student_id: str):
        """Update daily and weekly streaks."""
        today = datetime.now().date()
        last_activity = self.student_streaks[student_id].get("last_activity_date")
        
        if last_activity is None:
            self.student_streaks[student_id]["daily"] = 1
            self.student_streaks[student_id]["last_activity_date"] = today
        elif last_activity == today:
            # Already counted today
            pass
        elif (today - last_activity).days == 1:
            # Consecutive day
            self.student_streaks[student_id]["daily"] += 1
            self.student_streaks[student_id]["last_activity_date"] = today
        else:
            # Streak broken
            self.student_streaks[student_id]["daily"] = 1
            self.student_streaks[student_id]["last_activity_date"] = today
        
        # Weekly streak (simplified)
        if self.student_streaks[student_id]["daily"] >= 7:
            self.student_streaks[student_id]["weekly"] += 1
    
    def _check_badges(self, student_id: str, action_type: str, metadata: Optional[Dict]):
        """Check and award badges."""
        badges = self.student_badges[student_id]
        existing_badge_types = {b.badge_type for b in badges}
        
        # First question badge
        if action_type == "ask_question" and BadgeType.FIRST_QUESTION not in existing_badge_types:
            if self.student_points[student_id] == 0:
                badge = Badge(
                    badge_type=BadgeType.FIRST_QUESTION,
                    name="Primera Pregunta",
                    description="Hiciste tu primera pregunta",
                    icon="🎯",
                    earned_at=datetime.now(),
                    points=10
                )
                badges.append(badge)
                self.student_points[student_id] += badge.points
        
        # Daily streak badges
        daily_streak = self.student_streaks[student_id]["daily"]
        if daily_streak >= 7 and BadgeType.DAILY_STREAK not in existing_badge_types:
            badge = Badge(
                badge_type=BadgeType.DAILY_STREAK,
                name="Racha Diaria",
                description=f"7 días consecutivos activo",
                icon="🔥",
                earned_at=datetime.now(),
                points=50
            )
            badges.append(badge)
            self.student_points[student_id] += badge.points
    
    def _check_achievements(self, student_id: str, action_type: str):
        """Check and unlock achievements."""
        achievements = self.achievements[student_id]
        
        for achievement in achievements:
            if achievement.unlocked:
                continue
            
            req_type, req_value = achievement.requirement.split(":")
            req_value = int(req_value)
            
            if req_type == action_type:
                # Count actions (simplified - would need proper tracking)
                # This is a placeholder - would need actual action counting
                pass
    
    def _update_level(self, student_id: str):
        """Update student level based on points."""
        points = self.student_points[student_id]
        # Level formula: level = sqrt(points / 100)
        new_level = int((points / 100) ** 0.5) + 1
        
        if new_level > self.student_levels[student_id]:
            self.student_levels[student_id] = new_level
            logger.info(f"Student {student_id} leveled up to {new_level}")
    
    def get_student_profile(self, student_id: str) -> Dict[str, Any]:
        """Get complete gamification profile for student."""
        if student_id not in self.student_points:
            return {
                "points": 0,
                "level": 1,
                "badges": [],
                "streaks": {"daily": 0, "weekly": 0},
                "achievements": []
            }
        
        return {
            "points": self.student_points[student_id],
            "level": self.student_levels[student_id],
            "badges": [
                {
                    "type": b.badge_type.value,
                    "name": b.name,
                    "description": b.description,
                    "icon": b.icon,
                    "earned_at": b.earned_at.isoformat()
                }
                for b in self.student_badges[student_id]
            ],
            "streaks": self.student_streaks[student_id],
            "achievements": [
                {
                    "id": a.achievement_id,
                    "name": a.name,
                    "description": a.description,
                    "unlocked": a.unlocked,
                    "unlocked_at": a.unlocked_at.isoformat() if a.unlocked_at else None
                }
                for a in self.achievements[student_id]
            ]
        }
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top students leaderboard."""
        sorted_students = sorted(
            self.student_points.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "student_id": student_id,
                "points": points,
                "level": self.student_levels.get(student_id, 1),
                "rank": idx + 1
            }
            for idx, (student_id, points) in enumerate(sorted_students)
        ]

