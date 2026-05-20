"""
Python SDK client for AI Tutor Educacional API.
"""

import httpx
from typing import Dict, List, Optional, Any
from .models import QuestionRequest, ConceptRequest, ExerciseRequest, QuizRequest


class TutorClient:
    """
    Python client for AI Tutor Educacional API.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 60
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {})
            }
        )
    
    def ask_question(
        self,
        question: str,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ask a question to the tutor."""
        response = self.client.post(
            "/api/tutor/ask",
            json={
                "question": question,
                "subject": subject,
                "difficulty": difficulty,
                "conversation_id": conversation_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def explain_concept(
        self,
        concept: str,
        subject: str,
        difficulty: str = "intermedio"
    ) -> Dict[str, Any]:
        """Get explanation of a concept."""
        response = self.client.post(
            "/api/tutor/explain",
            json={
                "concept": concept,
                "subject": subject,
                "difficulty": difficulty
            }
        )
        response.raise_for_status()
        return response.json()
    
    def generate_exercises(
        self,
        topic: str,
        subject: str,
        difficulty: str = "intermedio",
        num_exercises: int = 3
    ) -> Dict[str, Any]:
        """Generate practice exercises."""
        response = self.client.post(
            "/api/tutor/exercises",
            json={
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
                "num_exercises": num_exercises
            }
        )
        response.raise_for_status()
        return response.json()
    
    def generate_quiz(
        self,
        topic: str,
        subject: str,
        difficulty: str = "intermedio",
        num_questions: int = 10,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a quiz."""
        response = self.client.post(
            "/api/tutor/quiz",
            json={
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
                "num_questions": num_questions,
                "question_types": question_types
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        response = self.client.get("/api/tutor/metrics")
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        response = self.client.get("/api/tutor/health")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()






