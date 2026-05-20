"""
Data Transfer Objects (DTOs)

DTOs for transferring data between layers.
"""

from .analysis import AnalysisResultDTO, TrackAnalysisDTO
from .recommendations import RecommendationDTO, PlaylistDTO
from .coaching import CoachingDTO, LearningStepDTO, ExerciseDTO

__all__ = [
    "AnalysisResultDTO",
    "TrackAnalysisDTO",
    "RecommendationDTO",
    "PlaylistDTO",
    "CoachingDTO",
    "LearningStepDTO",
    "ExerciseDTO",
]




