"""
Coaching DTOs

Data Transfer Objects for coaching data.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class LearningStepDTO:
    """DTO for a learning step"""
    step: int
    title: str
    description: str
    duration: str
    exercises: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step": self.step,
            "title": self.title,
            "description": self.description,
            "duration": self.duration,
            "exercises": self.exercises or []
        }


@dataclass
class ExerciseDTO:
    """DTO for a practice exercise"""
    name: str
    description: str
    difficulty: str
    duration: str
    focus_areas: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "duration": self.duration,
            "focus_areas": self.focus_areas or []
        }


@dataclass
class CoachingDTO:
    """DTO for complete coaching analysis"""
    overview: Dict[str, Any]
    learning_path: List[LearningStepDTO]
    practice_exercises: List[ExerciseDTO]
    performance_tips: List[str]
    difficulty_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overview": self.overview,
            "learning_path": [step.to_dict() for step in self.learning_path],
            "practice_exercises": [exercise.to_dict() for exercise in self.practice_exercises],
            "performance_tips": self.performance_tips,
            "difficulty_level": self.difficulty_level
        }




