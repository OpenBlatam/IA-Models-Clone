"""
Sistema de feedback y ratings
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


class FeedbackType(str, Enum):
    """Tipos de feedback"""
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    PRODUCT = "product"
    SERVICE = "service"
    GENERAL = "general"


@dataclass
class Feedback:
    """Feedback del usuario"""
    id: str
    user_id: str
    type: FeedbackType
    target_id: str  # ID del recurso (análisis, recomendación, etc.)
    rating: float  # 0-5
    comment: Optional[str] = None
    helpful: Optional[bool] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value,
            "target_id": self.target_id,
            "rating": self.rating,
            "comment": self.comment,
            "helpful": self.helpful,
            "created_at": self.created_at
        }


class FeedbackSystem:
    """Sistema de feedback"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.feedbacks: Dict[str, Feedback] = {}
        self.feedback_by_target: Dict[str, List[str]] = defaultdict(list)  # target_id -> [feedback_ids]
    
    def submit_feedback(self, user_id: str, feedback_type: FeedbackType,
                       target_id: str, rating: float,
                       comment: Optional[str] = None,
                       helpful: Optional[bool] = None) -> str:
        """
        Envía feedback
        
        Args:
            user_id: ID del usuario
            feedback_type: Tipo de feedback
            target_id: ID del recurso
            rating: Rating (0-5)
            comment: Comentario opcional
            helpful: Si fue útil (opcional)
            
        Returns:
            ID del feedback
        """
        feedback_id = str(uuid.uuid4())
        
        feedback = Feedback(
            id=feedback_id,
            user_id=user_id,
            type=feedback_type,
            target_id=target_id,
            rating=rating,
            comment=comment,
            helpful=helpful
        )
        
        self.feedbacks[feedback_id] = feedback
        self.feedback_by_target[target_id].append(feedback_id)
        
        return feedback_id
    
    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Obtiene un feedback"""
        return self.feedbacks.get(feedback_id)
    
    def get_target_feedback(self, target_id: str) -> List[Feedback]:
        """Obtiene feedback de un recurso"""
        feedback_ids = self.feedback_by_target.get(target_id, [])
        return [self.feedbacks[fid] for fid in feedback_ids if fid in self.feedbacks]
    
    def get_user_feedback(self, user_id: str) -> List[Feedback]:
        """Obtiene feedback de un usuario"""
        return [f for f in self.feedbacks.values() if f.user_id == user_id]
    
    def get_average_rating(self, target_id: str) -> Optional[float]:
        """Obtiene rating promedio de un recurso"""
        feedbacks = self.get_target_feedback(target_id)
        
        if not feedbacks:
            return None
        
        ratings = [f.rating for f in feedbacks]
        return sum(ratings) / len(ratings)
    
    def get_feedback_stats(self, target_id: Optional[str] = None) -> Dict:
        """Obtiene estadísticas de feedback"""
        if target_id:
            feedbacks = self.get_target_feedback(target_id)
        else:
            feedbacks = list(self.feedbacks.values())
        
        if not feedbacks:
            return {
                "total": 0,
                "average_rating": 0,
                "with_comments": 0
            }
        
        ratings = [f.rating for f in feedbacks]
        with_comments = sum(1 for f in feedbacks if f.comment)
        helpful_count = sum(1 for f in feedbacks if f.helpful is True)
        
        return {
            "total": len(feedbacks),
            "average_rating": sum(ratings) / len(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            "with_comments": with_comments,
            "helpful_count": helpful_count,
            "helpful_percentage": (helpful_count / len(feedbacks) * 100) if feedbacks else 0
        }






