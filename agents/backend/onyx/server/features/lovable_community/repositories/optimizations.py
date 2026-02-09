"""
Query Optimizations

Advanced query optimizations for better performance.
Following database optimization best practices.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload, selectinload, contains_eager
from sqlalchemy import func, desc, asc, or_, and_, case

from ..models import PublishedChat, ChatVote, ChatView


class QueryOptimizer:
    """
    Query optimizer for common operations.
    
    Provides optimized queries with eager loading and batch operations.
    """
    
    @staticmethod
    def get_chats_with_relations(
        db: Session,
        chat_ids: List[str],
        include_votes: bool = False,
        include_views: bool = False
    ) -> List[PublishedChat]:
        """
        Get multiple chats with eager loading of relations.
        
        Args:
            db: Database session
            chat_ids: List of chat IDs
            include_votes: Whether to eager load votes
            include_views: Whether to eager load views
            
        Returns:
            List of chats with relations loaded
        """
        query = db.query(PublishedChat).filter(
            PublishedChat.id.in_(chat_ids)
        )
        
        # Eager load relations to avoid N+1 queries
        if include_votes:
            query = query.options(selectinload(PublishedChat.votes))
        
        if include_views:
            query = query.options(selectinload(PublishedChat.views))
        
        return query.all()
    
    @staticmethod
    def get_chats_batch_optimized(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "score",
        order: str = "desc",
        user_id: Optional[str] = None
    ) -> tuple[List[PublishedChat], int]:
        """
        Optimized batch query with count in single query.
        
        Args:
            db: Database session
            skip: Offset
            limit: Limit
            sort_by: Sort field
            order: Sort order
            user_id: Optional user filter
            
        Returns:
            Tuple of (chats, total_count)
        """
        # Base query
        base_query = db.query(PublishedChat).filter(
            PublishedChat.is_public == True
        )
        
        if user_id:
            base_query = base_query.filter(PublishedChat.user_id == user_id)
        
        # Get total count efficiently
        total = base_query.count()
        
        # Apply sorting
        sort_field = getattr(PublishedChat, sort_by, PublishedChat.score)
        if order == "desc":
            query = base_query.order_by(desc(sort_field))
        else:
            query = base_query.order_by(asc(sort_field))
        
        # Apply pagination
        chats = query.offset(skip).limit(limit).all()
        
        return chats, total
    
    @staticmethod
    def bulk_update_scores(
        db: Session,
        score_updates: dict[str, float]
    ) -> int:
        """
        Bulk update scores for multiple chats.
        
        Args:
            db: Database session
            score_updates: Dictionary mapping chat_id to new score
            
        Returns:
            Number of updated rows
        """
        if not score_updates:
            return 0
        
        from sqlalchemy import update
        
        # Use bulk update for better performance
        stmt = (
            update(PublishedChat)
            .where(PublishedChat.id.in_(score_updates.keys()))
            .values(score=func.case(
                {chat_id: score for chat_id, score in score_updates.items()},
                else_=PublishedChat.score
            ))
        )
        
        result = db.execute(stmt)
        db.commit()
        
        return result.rowcount
    
    @staticmethod
    def get_user_votes_batch_optimized(
        db: Session,
        chat_ids: List[str],
        user_id: str
    ) -> dict[str, ChatVote]:
        """
        Get user votes for multiple chats in single query.
        
        Args:
            db: Database session
            chat_ids: List of chat IDs
            user_id: User ID
            
        Returns:
            Dictionary mapping chat_id to vote
        """
        if not chat_ids:
            return {}
        
        votes = db.query(ChatVote).filter(
            ChatVote.chat_id.in_(chat_ids),
            ChatVote.user_id == user_id
        ).all()
        
        return {vote.chat_id: vote for vote in votes}
    
    @staticmethod
    def get_chat_stats_batch(
        db: Session,
        chat_ids: List[str]
    ) -> dict[str, dict]:
        """
        Get statistics for multiple chats in optimized queries.
        
        Args:
            db: Database session
            chat_ids: List of chat IDs
            
        Returns:
            Dictionary mapping chat_id to stats
        """
        if not chat_ids:
            return {}
        
        # Get vote counts
        vote_stats = db.query(
            ChatVote.chat_id,
            func.sum(case((ChatVote.vote_type == "upvote", 1), else_=0)).label('upvotes'),
            func.sum(case((ChatVote.vote_type == "downvote", 1), else_=0)).label('downvotes')
        ).filter(
            ChatVote.chat_id.in_(chat_ids)
        ).group_by(ChatVote.chat_id).all()
        
        # Get view counts
        view_counts = db.query(
            ChatView.chat_id,
            func.count(ChatView.id).label('view_count')
        ).filter(
            ChatView.chat_id.in_(chat_ids)
        ).group_by(ChatView.chat_id).all()
        
        # Combine results
        stats = {}
        for chat_id in chat_ids:
            stats[chat_id] = {
                "upvotes": 0,
                "downvotes": 0,
                "view_count": 0
            }
        
        for stat in vote_stats:
            stats[stat.chat_id]["upvotes"] = stat.upvotes or 0
            stats[stat.chat_id]["downvotes"] = stat.downvotes or 0
        
        for view in view_counts:
            stats[view.chat_id]["view_count"] = view.view_count or 0
        
        return stats

