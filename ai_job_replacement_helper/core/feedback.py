"""
Feedback Service - Sistema de feedback
======================================

Sistema para recopilar y gestionar feedback de usuarios.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Tipos de feedback"""
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    GENERAL_FEEDBACK = "general_feedback"
    RATING = "rating"
    TESTIMONIAL = "testimonial"


class FeedbackStatus(str, Enum):
    """Estado del feedback"""
    PENDING = "pending"
    REVIEWED = "reviewed"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    REJECTED = "rejected"


@dataclass
class Feedback:
    """Feedback"""
    id: str
    user_id: str
    feedback_type: FeedbackType
    title: str
    description: str
    rating: Optional[int] = None  # 1-5
    status: FeedbackStatus = FeedbackStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackService:
    """Servicio de feedback"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.feedback_items: List[Feedback] = []
        logger.info("FeedbackService initialized")
    
    def submit_feedback(
        self,
        user_id: str,
        feedback_type: FeedbackType,
        title: str,
        description: str,
        rating: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Enviar feedback"""
        feedback = Feedback(
            id=f"feedback_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            feedback_type=feedback_type,
            title=title,
            description=description,
            rating=rating,
            metadata=metadata or {},
        )
        
        self.feedback_items.append(feedback)
        
        logger.info(f"Feedback submitted by user {user_id}: {title}")
        return feedback
    
    def get_feedback(
        self,
        feedback_type: Optional[FeedbackType] = None,
        status: Optional[FeedbackStatus] = None,
        limit: int = 50
    ) -> List[Feedback]:
        """Obtener feedback"""
        feedback = self.feedback_items.copy()
        
        if feedback_type:
            feedback = [f for f in feedback if f.feedback_type == feedback_type]
        
        if status:
            feedback = [f for f in feedback if f.status == status]
        
        # Ordenar por fecha
        feedback.sort(key=lambda x: x.created_at, reverse=True)
        
        return feedback[:limit]
    
    def get_user_feedback(self, user_id: str) -> List[Feedback]:
        """Obtener feedback del usuario"""
        return [f for f in self.feedback_items if f.user_id == user_id]
    
    def update_feedback_status(
        self,
        feedback_id: str,
        status: FeedbackStatus,
        response: Optional[str] = None
    ) -> Optional[Feedback]:
        """Actualizar estado del feedback"""
        feedback = next((f for f in self.feedback_items if f.id == feedback_id), None)
        if not feedback:
            return None
        
        feedback.status = status
        feedback.reviewed_at = datetime.now()
        if response:
            feedback.response = response
        
        return feedback
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de feedback"""
        total = len(self.feedback_items)
        
        by_type = {}
        for feedback_type in FeedbackType:
            count = sum(1 for f in self.feedback_items if f.feedback_type == feedback_type)
            if count > 0:
                by_type[feedback_type.value] = count
        
        by_status = {}
        for status in FeedbackStatus:
            count = sum(1 for f in self.feedback_items if f.status == status)
            if count > 0:
                by_status[status.value] = count
        
        # Rating promedio
        ratings = [f.rating for f in self.feedback_items if f.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total": total,
            "by_type": by_type,
            "by_status": by_status,
            "average_rating": round(avg_rating, 2),
            "total_ratings": len(ratings),
        }




