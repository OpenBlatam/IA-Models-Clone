"""
Python SDK for AI Tutor Educacional API.
"""

from .client import TutorClient
from .models import QuestionRequest, ConceptRequest, ExerciseRequest, QuizRequest

__version__ = "1.0.0"
__all__ = [
    "TutorClient",
    "QuestionRequest",
    "ConceptRequest",
    "ExerciseRequest",
    "QuizRequest",
]






