"""
Feedback Service
User feedback and comments
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Feedback:
    """Represents user feedback"""
    
    def __init__(
        self,
        feedback_id: str,
        video_id: UUID,
        user_id: str,
        rating: int,  # 1-5
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_at: Optional[datetime] = None
    ):
        self.feedback_id = feedback_id
        self.video_id = video_id
        self.user_id = user_id
        self.rating = rating
        self.comment = comment
        self.tags = tags or []
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "feedback_id": self.feedback_id,
            "video_id": str(self.video_id),
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }


class FeedbackService:
    """Manages user feedback"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.feedbacks: Dict[str, Feedback] = {}
        self.video_feedbacks: Dict[UUID, List[str]] = {}  # video_id -> feedback_ids
    
    def submit_feedback(
        self,
        video_id: UUID,
        user_id: str,
        rating: int,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Feedback:
        """
        Submit feedback for video
        
        Args:
            video_id: Video ID
            user_id: User ID
            rating: Rating (1-5)
            comment: Optional comment
            tags: Optional tags
            
        Returns:
            Created feedback
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback_id = f"feedback_{len(self.feedbacks) + 1}"
        
        feedback = Feedback(
            feedback_id=feedback_id,
            video_id=video_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            tags=tags
        )
        
        self.feedbacks[feedback_id] = feedback
        
        if video_id not in self.video_feedbacks:
            self.video_feedbacks[video_id] = []
        self.video_feedbacks[video_id].append(feedback_id)
        
        logger.info(f"Feedback submitted: {feedback_id} for video {video_id}")
        return feedback
    
    def get_video_feedback(self, video_id: UUID) -> List[Feedback]:
        """Get all feedback for video"""
        feedback_ids = self.video_feedbacks.get(video_id, [])
        return [self.feedbacks[fid] for fid in feedback_ids if fid in self.feedbacks]
    
    def get_feedback_stats(self, video_id: UUID) -> Dict[str, Any]:
        """Get feedback statistics for video"""
        feedbacks = self.get_video_feedback(video_id)
        
        if not feedbacks:
            return {
                "total": 0,
                "average_rating": 0.0,
                "rating_distribution": {i: 0 for i in range(1, 6)},
            }
        
        ratings = [f.rating for f in feedbacks]
        average = sum(ratings) / len(ratings)
        
        distribution = {i: ratings.count(i) for i in range(1, 6)}
        
        return {
            "total": len(feedbacks),
            "average_rating": round(average, 2),
            "rating_distribution": distribution,
        }
    
    def get_all_feedback(self, limit: int = 100) -> List[Feedback]:
        """Get all feedback (recent first)"""
        feedbacks = sorted(
            self.feedbacks.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return feedbacks[:limit]


_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get feedback service instance (singleton)"""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service

