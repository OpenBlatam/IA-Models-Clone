"""
Adaptive learning system for personalized education.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LearningStyle(Enum):
    """Learning style preferences."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"


class DifficultyLevel(Enum):
    """Difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AdaptiveLearningEngine:
    """
    Adaptive learning engine that personalizes content based on student performance.
    """
    
    def __init__(self):
        self.student_profiles: Dict[str, Dict[str, Any]] = {}
        self.learning_paths: Dict[str, List[str]] = {}
    
    def create_student_profile(
        self,
        student_id: str,
        initial_level: DifficultyLevel = DifficultyLevel.BEGINNER,
        learning_style: Optional[LearningStyle] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create or update student learning profile.
        
        Args:
            student_id: Student identifier
            initial_level: Starting difficulty level
            learning_style: Preferred learning style
            preferences: Additional preferences
        
        Returns:
            Student profile
        """
        profile = {
            "student_id": student_id,
            "current_level": initial_level.value,
            "learning_style": learning_style.value if learning_style else None,
            "preferences": preferences or {},
            "performance_history": [],
            "mastered_topics": [],
            "struggling_topics": [],
            "recommended_path": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.student_profiles[student_id] = profile
        logger.info(f"Created profile for student {student_id}")
        
        return profile
    
    def update_performance(
        self,
        student_id: str,
        topic: str,
        score: float,
        time_taken: Optional[float] = None,
        attempts: int = 1
    ) -> Dict[str, Any]:
        """
        Update student performance and adapt learning path.
        
        Args:
            student_id: Student identifier
            topic: Topic assessed
            score: Performance score (0-1)
            time_taken: Time taken in seconds
            attempts: Number of attempts
        
        Returns:
            Updated profile with recommendations
        """
        if student_id not in self.student_profiles:
            self.create_student_profile(student_id)
        
        profile = self.student_profiles[student_id]
        
        # Record performance
        performance_record = {
            "topic": topic,
            "score": score,
            "time_taken": time_taken,
            "attempts": attempts,
            "timestamp": datetime.now().isoformat()
        }
        
        profile["performance_history"].append(performance_record)
        
        # Update mastered/struggling topics
        if score >= 0.8:
            if topic not in profile["mastered_topics"]:
                profile["mastered_topics"].append(topic)
            if topic in profile["struggling_topics"]:
                profile["struggling_topics"].remove(topic)
        elif score < 0.6:
            if topic not in profile["struggling_topics"]:
                profile["struggling_topics"].append(topic)
        
        # Adapt difficulty level
        self._adapt_difficulty(student_id)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(student_id)
        profile["recommended_path"] = recommendations
        
        profile["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Updated performance for {student_id} on {topic}: {score}")
        
        return {
            "profile": profile,
            "recommendations": recommendations,
            "next_steps": self._get_next_steps(student_id)
        }
    
    def _adapt_difficulty(self, student_id: str) -> None:
        """Adapt difficulty level based on performance."""
        profile = self.student_profiles[student_id]
        
        if not profile["performance_history"]:
            return
        
        # Analyze recent performance
        recent_performances = profile["performance_history"][-10:]
        avg_score = sum(p["score"] for p in recent_performances) / len(recent_performances)
        
        current_level = DifficultyLevel(profile["current_level"])
        
        # Adjust level
        if avg_score >= 0.85 and current_level != DifficultyLevel.EXPERT:
            # Move up
            level_map = {
                DifficultyLevel.BEGINNER: DifficultyLevel.INTERMEDIATE,
                DifficultyLevel.INTERMEDIATE: DifficultyLevel.ADVANCED,
                DifficultyLevel.ADVANCED: DifficultyLevel.EXPERT
            }
            profile["current_level"] = level_map.get(current_level, current_level).value
            logger.info(f"Upgraded {student_id} to {profile['current_level']}")
        
        elif avg_score < 0.6 and current_level != DifficultyLevel.BEGINNER:
            # Move down
            level_map = {
                DifficultyLevel.EXPERT: DifficultyLevel.ADVANCED,
                DifficultyLevel.ADVANCED: DifficultyLevel.INTERMEDIATE,
                DifficultyLevel.INTERMEDIATE: DifficultyLevel.BEGINNER
            }
            profile["current_level"] = level_map.get(current_level, current_level).value
            logger.info(f"Adjusted {student_id} to {profile['current_level']}")
    
    def _generate_recommendations(self, student_id: str) -> List[str]:
        """Generate personalized learning recommendations."""
        profile = self.student_profiles[student_id]
        recommendations = []
        
        # Recommend review for struggling topics
        if profile["struggling_topics"]:
            recommendations.append(f"Review: {', '.join(profile['struggling_topics'][:3])}")
        
        # Recommend next topics based on mastered
        if profile["mastered_topics"]:
            recommendations.append("Ready for next level topics")
        
        # Recommend practice based on learning style
        if profile["learning_style"]:
            style_recommendations = {
                LearningStyle.VISUAL.value: "Try visual exercises and diagrams",
                LearningStyle.AUDITORY.value: "Listen to explanations and discussions",
                LearningStyle.KINESTHETIC.value: "Engage in hands-on practice",
                LearningStyle.READING.value: "Read detailed explanations and examples"
            }
            recommendations.append(style_recommendations.get(profile["learning_style"]))
        
        return recommendations
    
    def _get_next_steps(self, student_id: str) -> List[Dict[str, Any]]:
        """Get next learning steps for student."""
        profile = self.student_profiles[student_id]
        
        next_steps = []
        
        # If struggling, recommend review
        if profile["struggling_topics"]:
            next_steps.append({
                "action": "review",
                "topics": profile["struggling_topics"][:2],
                "priority": "high"
            })
        
        # Recommend progression topics
        if len(profile["mastered_topics"]) >= 3:
            next_steps.append({
                "action": "progress",
                "level": profile["current_level"],
                "priority": "medium"
            })
        
        return next_steps
    
    def get_personalized_content(
        self,
        student_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Get personalized content for student.
        
        Args:
            student_id: Student identifier
            topic: Topic to learn
        
        Returns:
            Personalized content configuration
        """
        if student_id not in self.student_profiles:
            self.create_student_profile(student_id)
        
        profile = self.student_profiles[student_id]
        
        return {
            "topic": topic,
            "difficulty": profile["current_level"],
            "learning_style": profile["learning_style"],
            "format": self._get_preferred_format(profile),
            "pacing": self._get_pacing(profile),
            "support_level": self._get_support_level(profile, topic)
        }
    
    def _get_preferred_format(self, profile: Dict[str, Any]) -> str:
        """Get preferred content format based on learning style."""
        style_format_map = {
            LearningStyle.VISUAL.value: "visual",
            LearningStyle.AUDITORY.value: "audio",
            LearningStyle.KINESTHETIC.value: "interactive",
            LearningStyle.READING.value: "text"
        }
        return style_format_map.get(profile.get("learning_style"), "mixed")
    
    def _get_pacing(self, profile: Dict[str, Any]) -> str:
        """Determine learning pacing based on performance."""
        if not profile["performance_history"]:
            return "normal"
        
        recent = profile["performance_history"][-5:]
        avg_score = sum(p["score"] for p in recent) / len(recent)
        
        if avg_score >= 0.85:
            return "fast"
        elif avg_score < 0.6:
            return "slow"
        else:
            return "normal"
    
    def _get_support_level(
        self,
        profile: Dict[str, Any],
        topic: str
    ) -> str:
        """Determine level of support needed."""
        if topic in profile["struggling_topics"]:
            return "high"
        elif topic in profile["mastered_topics"]:
            return "low"
        else:
            return "medium"
    
    def get_learning_path(self, student_id: str) -> List[Dict[str, Any]]:
        """Get complete learning path for student."""
        if student_id not in self.student_profiles:
            return []
        
        profile = self.student_profiles[student_id]
        
        return {
            "student_id": student_id,
            "current_level": profile["current_level"],
            "mastered_topics": profile["mastered_topics"],
            "in_progress": [],
            "upcoming": [],
            "recommendations": profile["recommended_path"]
        }




