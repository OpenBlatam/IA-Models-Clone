"""
Data models for the SDK.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="The student's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    subject: Optional[str] = Field(None, description="Subject area")
    difficulty: Optional[str] = Field(None, description="Difficulty level")


class ConceptRequest(BaseModel):
    """Request model for explaining concepts."""
    concept: str = Field(..., description="Concept to explain")
    subject: str = Field(..., description="Subject area")
    difficulty: str = Field("intermedio", description="Difficulty level")


class ExerciseRequest(BaseModel):
    """Request model for generating exercises."""
    topic: str = Field(..., description="Topic for exercises")
    subject: str = Field(..., description="Subject area")
    difficulty: str = Field("intermedio", description="Difficulty level")
    num_exercises: int = Field(3, description="Number of exercises")


class QuizRequest(BaseModel):
    """Request model for generating quizzes."""
    topic: str = Field(..., description="Topic for the quiz")
    subject: str = Field(..., description="Subject area")
    difficulty: str = Field("intermedio", description="Difficulty level")
    num_questions: int = Field(10, description="Number of questions")
    question_types: Optional[List[str]] = Field(None, description="Types of questions")






