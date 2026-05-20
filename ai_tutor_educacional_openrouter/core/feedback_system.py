"""
Feedback system for continuous improvement.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""
    POSITIVE = "positive"
    CONSTRUCTIVE = "constructive"
    CORRECTIVE = "corrective"
    ENCOURAGING = "encouraging"


class FeedbackSystem:
    """
    System for providing and managing feedback.
    """
    
    def __init__(self):
        self.feedback_history: Dict[str, List[Dict[str, Any]]] = {}
        self.feedback_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """Initialize feedback templates."""
        self.feedback_templates = {
            "correct": {
                "type": FeedbackType.POSITIVE.value,
                "messages": [
                    "¡Excelente! Has respondido correctamente.",
                    "Muy bien, respuesta correcta.",
                    "Perfecto, continúa así."
                ]
            },
            "incorrect": {
                "type": FeedbackType.CORRECTIVE.value,
                "messages": [
                    "No es correcto. Revisa el concepto.",
                    "Inténtalo de nuevo. Puedes hacerlo mejor.",
                    "Incorrecto. Te sugiero repasar este tema."
                ]
            },
            "partial": {
                "type": FeedbackType.CONSTRUCTIVE.value,
                "messages": [
                    "Bien, pero puedes mejorar.",
                    "Estás en el camino correcto, pero falta algo.",
                    "Casi correcto, revisa los detalles."
                ]
            },
            "encouragement": {
                "type": FeedbackType.ENCOURAGING.value,
                "messages": [
                    "¡Sigue así! Estás mejorando.",
                    "Tu esfuerzo se nota, continúa.",
                    "Vas por buen camino, no te rindas."
                ]
            }
        }
    
    def provide_feedback(
        self,
        student_id: str,
        context: str,
        feedback_type: FeedbackType,
        message: str,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Provide feedback to a student.
        
        Args:
            student_id: Student identifier
            context: Context of the feedback
            feedback_type: Type of feedback
            message: Feedback message
            suggestions: Optional suggestions for improvement
            metadata: Additional metadata
        
        Returns:
            Feedback record
        """
        feedback = {
            "feedback_id": f"feedback_{datetime.now().timestamp()}",
            "student_id": student_id,
            "context": context,
            "feedback_type": feedback_type.value,
            "message": message,
            "suggestions": suggestions or [],
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        if student_id not in self.feedback_history:
            self.feedback_history[student_id] = []
        
        self.feedback_history[student_id].append(feedback)
        logger.info(f"Provided {feedback_type.value} feedback to {student_id}")
        
        return feedback
    
    def generate_automated_feedback(
        self,
        student_id: str,
        context: str,
        is_correct: bool,
        score: Optional[float] = None,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate automated feedback based on performance.
        
        Args:
            student_id: Student identifier
            context: Context (e.g., "quiz", "exercise")
            is_correct: Whether answer was correct
            score: Optional score (0-1)
            topic: Optional topic
        
        Returns:
            Generated feedback
        """
        if is_correct:
            template = self.feedback_templates["correct"]
            message = self._select_message(template["messages"])
            feedback_type = FeedbackType.POSITIVE
        elif score and score >= 0.5:
            template = self.feedback_templates["partial"]
            message = self._select_message(template["messages"])
            feedback_type = FeedbackType.CONSTRUCTIVE
        else:
            template = self.feedback_templates["incorrect"]
            message = self._select_message(template["messages"])
            feedback_type = FeedbackType.CORRECTIVE
        
        # Add topic-specific suggestions
        suggestions = []
        if topic:
            suggestions.append(f"Revisa más sobre {topic}")
        if not is_correct:
            suggestions.append("Intenta practicar ejercicios similares")
        
        return self.provide_feedback(
            student_id=student_id,
            context=context,
            feedback_type=feedback_type,
            message=message,
            suggestions=suggestions,
            metadata={"score": score, "topic": topic}
        )
    
    def _select_message(self, messages: List[str]) -> str:
        """Select a random message from list."""
        import random
        return random.choice(messages) if messages else ""
    
    def get_feedback_history(
        self,
        student_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get feedback history for a student.
        
        Args:
            student_id: Student identifier
            limit: Maximum number of feedbacks to return
        
        Returns:
            List of feedback records
        """
        if student_id not in self.feedback_history:
            return []
        
        return self.feedback_history[student_id][-limit:]
    
    def get_feedback_summary(
        self,
        student_id: str
    ) -> Dict[str, Any]:
        """
        Get feedback summary for a student.
        
        Args:
            student_id: Student identifier
        
        Returns:
            Feedback summary
        """
        if student_id not in self.feedback_history:
            return {"error": "No feedback found"}
        
        feedbacks = self.feedback_history[student_id]
        
        # Count by type
        type_counts = {}
        for feedback in feedbacks:
            ftype = feedback["feedback_type"]
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        # Recent feedbacks
        recent = feedbacks[-5:] if len(feedbacks) >= 5 else feedbacks
        
        return {
            "student_id": student_id,
            "total_feedbacks": len(feedbacks),
            "by_type": type_counts,
            "recent_feedbacks": recent
        }
    
    def add_feedback_template(
        self,
        template_name: str,
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a custom feedback template.
        
        Args:
            template_name: Name of the template
            template: Template definition
        
        Returns:
            Confirmation
        """
        self.feedback_templates[template_name] = template
        logger.info(f"Added feedback template: {template_name}")
        
        return {"status": "success", "template_name": template_name}




