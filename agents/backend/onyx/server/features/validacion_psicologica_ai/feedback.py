"""
Sistema de Feedback para Validación Psicológica AI
====================================================
Recopilación y procesamiento de feedback de usuarios
"""

from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import structlog

from .models import PsychologicalValidation

logger = structlog.get_logger()


class FeedbackType(str, Enum):
    """Tipos de feedback"""
    ACCURACY = "accuracy"
    USEFULNESS = "usefulness"
    RECOMMENDATIONS = "recommendations"
    INTERFACE = "interface"
    GENERAL = "general"


class FeedbackRating(str, Enum):
    """Calificación de feedback"""
    VERY_POOR = "very_poor"
    POOR = "poor"
    NEUTRAL = "neutral"
    GOOD = "good"
    EXCELLENT = "excellent"


class Feedback:
    """Representa un feedback"""
    
    def __init__(
        self,
        validation_id: UUID,
        user_id: UUID,
        feedback_type: FeedbackType,
        rating: FeedbackRating,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid4()
        self.validation_id = validation_id
        self.user_id = user_id
        self.feedback_type = feedback_type
        self.rating = rating
        self.comment = comment
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "validation_id": str(self.validation_id),
            "user_id": str(self.user_id),
            "feedback_type": self.feedback_type.value,
            "rating": self.rating.value,
            "comment": self.comment,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class FeedbackManager:
    """Gestor de feedback"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._feedback: List[Feedback] = []
        logger.info("FeedbackManager initialized")
    
    def submit_feedback(
        self,
        validation_id: UUID,
        user_id: UUID,
        feedback_type: FeedbackType,
        rating: FeedbackRating,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """
        Enviar feedback
        
        Args:
            validation_id: ID de la validación
            user_id: ID del usuario
            feedback_type: Tipo de feedback
            rating: Calificación
            comment: Comentario (opcional)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            Feedback creado
        """
        feedback = Feedback(
            validation_id=validation_id,
            user_id=user_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            metadata=metadata
        )
        
        self._feedback.append(feedback)
        
        logger.info(
            "Feedback submitted",
            feedback_id=str(feedback.id),
            validation_id=str(validation_id),
            rating=rating.value
        )
        
        return feedback
    
    def get_feedback(
        self,
        validation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        feedback_type: Optional[FeedbackType] = None,
        limit: int = 100
    ) -> List[Feedback]:
        """
        Obtener feedback
        
        Args:
            validation_id: Filtrar por validación
            user_id: Filtrar por usuario
            feedback_type: Filtrar por tipo
            limit: Límite de resultados
            
        Returns:
            Lista de feedback
        """
        feedback = self._feedback
        
        if validation_id:
            feedback = [f for f in feedback if f.validation_id == validation_id]
        
        if user_id:
            feedback = [f for f in feedback if f.user_id == user_id]
        
        if feedback_type:
            feedback = [f for f in feedback if f.feedback_type == feedback_type]
        
        feedback.sort(key=lambda x: x.created_at, reverse=True)
        return feedback[:limit]
    
    def get_feedback_stats(
        self,
        validation_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de feedback
        
        Args:
            validation_id: Filtrar por validación (opcional)
            
        Returns:
            Estadísticas
        """
        feedback = self.get_feedback(validation_id=validation_id)
        
        if not feedback:
            return {
                "total": 0,
                "average_rating": None,
                "rating_distribution": {}
            }
        
        # Calcular distribución de ratings
        rating_map = {
            FeedbackRating.VERY_POOR: 1,
            FeedbackRating.POOR: 2,
            FeedbackRating.NEUTRAL: 3,
            FeedbackRating.GOOD: 4,
            FeedbackRating.EXCELLENT: 5
        }
        
        ratings = [rating_map.get(f.rating, 3) for f in feedback]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        rating_dist = {}
        for rating in FeedbackRating:
            count = len([f for f in feedback if f.rating == rating])
            rating_dist[rating.value] = count
        
        # Distribución por tipo
        type_dist = {}
        for ftype in FeedbackType:
            count = len([f for f in feedback if f.feedback_type == ftype])
            type_dist[ftype.value] = count
        
        return {
            "total": len(feedback),
            "average_rating": avg_rating,
            "rating_distribution": rating_dist,
            "type_distribution": type_dist,
            "with_comments": len([f for f in feedback if f.comment])
        }
    
    def get_improvement_suggestions(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener sugerencias de mejora basadas en feedback
        
        Args:
            limit: Límite de sugerencias
            
        Returns:
            Lista de sugerencias
        """
        # Obtener feedback negativo
        negative_feedback = [
            f for f in self._feedback
            if f.rating in [FeedbackRating.VERY_POOR, FeedbackRating.POOR]
        ]
        
        suggestions = []
        
        # Agrupar por tipo
        by_type = {}
        for feedback in negative_feedback:
            ftype = feedback.feedback_type.value
            if ftype not in by_type:
                by_type[ftype] = []
            by_type[ftype].append(feedback)
        
        # Generar sugerencias
        for ftype, feedbacks in by_type.items():
            if len(feedbacks) >= 3:  # Si hay al menos 3 feedbacks negativos
                suggestions.append({
                    "area": ftype,
                    "priority": "high" if len(feedbacks) >= 10 else "medium",
                    "feedback_count": len(feedbacks),
                    "common_issues": self._extract_common_issues(feedbacks)
                })
        
        suggestions.sort(key=lambda x: (
            1 if x["priority"] == "high" else 2,
            -x["feedback_count"]
        ))
        
        return suggestions[:limit]
    
    def _extract_common_issues(
        self,
        feedbacks: List[Feedback]
    ) -> List[str]:
        """Extraer problemas comunes de feedbacks"""
        # Simplificado - en producción usar NLP para extraer temas
        issues = []
        
        comments = [f.comment for f in feedbacks if f.comment]
        if comments:
            # Palabras clave comunes
            keywords = ["slow", "inaccurate", "confusing", "error", "bug"]
            for keyword in keywords:
                if any(keyword.lower() in comment.lower() for comment in comments):
                    issues.append(f"Multiple mentions of '{keyword}'")
        
        return issues[:5]  # Top 5 issues


# Instancia global del gestor de feedback
feedback_manager = FeedbackManager()

