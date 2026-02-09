"""
Feedback System - Sistema de feedback y aprendizaje
===================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackSystem:
    """
    Sistema de feedback para mejorar el modelo basándose en retroalimentación.
    """
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        """
        Inicializar sistema de feedback.
        
        Args:
            feedback_dir: Directorio para almacenar feedback
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
    
    def submit_feedback(
        self,
        improvement_id: str,
        rating: int,
        comments: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Envía feedback sobre una mejora.
        
        Args:
            improvement_id: ID de la mejora
            rating: Calificación (1-5)
            comments: Comentarios (opcional)
            user_id: ID del usuario (opcional)
            
        Returns:
            Información del feedback
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating debe estar entre 1 y 5")
        
        feedback = {
            "feedback_id": f"{improvement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "improvement_id": improvement_id,
            "rating": rating,
            "comments": comments,
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }
        
        # Guardar feedback
        feedback_file = self.feedback_dir / f"{feedback['feedback_id']}.json"
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Feedback recibido: {feedback['feedback_id']} (rating: {rating})")
        
        return feedback
    
    def get_feedback_stats(self, improvement_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de feedback.
        
        Args:
            improvement_id: ID de mejora específica (opcional)
            
        Returns:
            Estadísticas de feedback
        """
        feedbacks = []
        
        for feedback_file in self.feedback_dir.glob("*.json"):
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedback = json.load(f)
                
                if not improvement_id or feedback.get("improvement_id") == improvement_id:
                    feedbacks.append(feedback)
            except Exception as e:
                logger.warning(f"Error procesando {feedback_file}: {e}")
                continue
        
        if not feedbacks:
            return {
                "total": 0,
                "average_rating": 0,
                "ratings_distribution": {}
            }
        
        ratings = [f["rating"] for f in feedbacks]
        
        return {
            "total": len(feedbacks),
            "average_rating": sum(ratings) / len(ratings),
            "ratings_distribution": {
                i: ratings.count(i) for i in range(1, 6)
            },
            "with_comments": sum(1 for f in feedbacks if f.get("comments"))
        }
    
    def get_improvement_suggestions(self, improvement_id: str) -> List[str]:
        """
        Obtiene sugerencias de mejora basadas en feedback.
        
        Args:
            improvement_id: ID de la mejora
            
        Returns:
            Lista de sugerencias
        """
        stats = self.get_feedback_stats(improvement_id)
        
        suggestions = []
        
        if stats["total"] > 0:
            avg_rating = stats["average_rating"]
            
            if avg_rating < 3:
                suggestions.append("Considerar revisar la estrategia de mejora")
                suggestions.append("Analizar papers más relevantes")
            
            if stats["ratings_distribution"].get(1, 0) > stats["total"] * 0.3:
                suggestions.append("Muchos usuarios reportaron problemas - revisar urgente")
        
        return suggestions




