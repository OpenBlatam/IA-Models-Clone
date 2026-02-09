"""
Recommendation Service using embeddings and collaborative filtering

Provides AI-powered recommendations based on:
- Semantic similarity (embeddings)
- User behavior patterns
- Content similarity
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import numpy as np

from ...models import PublishedChat, ChatEmbedding, ChatVote, ChatView, ChatAIMetadata
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for AI-powered content recommendations"""
    
    def __init__(self, db: Session, embedding_service: EmbeddingService):
        self.db = db
        self.embedding_service = embedding_service
    
    def recommend_similar_chats(
        self,
        chat_id: str,
        limit: int = 10,
        exclude_original: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recommend similar chats based on embeddings
        
        Args:
            chat_id: ID of the reference chat
            limit: Maximum number of recommendations
            exclude_original: Whether to exclude the original chat
            
        Returns:
            List of recommended chats with similarity scores
        """
        try:
            # Get embedding for the reference chat
            embedding = self.embedding_service.get_embedding(chat_id)
            
            if not embedding:
                # Generate embedding if it doesn't exist
                chat = self.db.query(PublishedChat).filter(
                    PublishedChat.id == chat_id
                ).first()
                if chat:
                    embedding = self.embedding_service.create_or_update_embedding(chat_id, chat=chat)
                else:
                    return []
            
            # Find similar chats
            query_text = f"{embedding.chat.title} {embedding.chat.description or ''}"
            similar_chats = self.embedding_service.find_similar_chats(
                query_text,
                limit=limit + 1 if exclude_original else limit,
                min_similarity=0.3
            )
            
            # Filter out original chat if needed
            if exclude_original:
                similar_chats = [
                    item for item in similar_chats
                    if item["chat_id"] != chat_id
                ][:limit]
            
            return similar_chats
            
        except Exception as e:
            logger.error(f"Error recommending similar chats: {e}", exc_info=True)
            return []
    
    def recommend_for_user(
        self,
        user_id: str,
        limit: int = 10,
        use_sentiment: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recommend chats for a specific user based on their preferences
        
        Args:
            user_id: ID of the user
            limit: Maximum number of recommendations
            use_sentiment: Whether to consider sentiment preferences
            
        Returns:
            List of recommended chats
        """
        try:
            # Get user's voting history
            user_votes = self.db.query(ChatVote).filter(
                ChatVote.user_id == user_id
            ).all()
            
            if not user_votes:
                # If user has no history, return trending chats
                return self._get_trending_recommendations(limit)
            
            # Get liked chats (upvoted)
            liked_chat_ids = [
                vote.chat_id for vote in user_votes
                if vote.vote_type == "upvote"
            ]
            
            if not liked_chat_ids:
                return self._get_trending_recommendations(limit)
            
            # Get embeddings of liked chats
            liked_embeddings = self.db.query(ChatEmbedding).filter(
                ChatEmbedding.chat_id.in_(liked_chat_ids)
            ).all()
            
            if not liked_embeddings:
                return self._get_trending_recommendations(limit)
            
            # Calculate average embedding of liked content
            embedding_vectors = [np.array(emb.embedding) for emb in liked_embeddings]
            avg_embedding = np.mean(embedding_vectors, axis=0)
            
            # Find similar chats
            all_embeddings = self.db.query(ChatEmbedding).join(
                PublishedChat
            ).filter(
                PublishedChat.is_public == True,
                ~ChatEmbedding.chat_id.in_(liked_chat_ids)  # Exclude already liked
            ).all()
            
            similarities = []
            for emb in all_embeddings:
                embedding_vector = np.array(emb.embedding)
                similarity = np.dot(avg_embedding, embedding_vector) / (
                    np.linalg.norm(avg_embedding) * np.linalg.norm(embedding_vector)
                )
                
                similarities.append({
                    "chat_id": emb.chat_id,
                    "similarity": float(similarity),
                    "chat": emb.chat
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Apply sentiment filter if enabled
            if use_sentiment:
                # Get user's sentiment preference
                user_sentiment = self._get_user_sentiment_preference(user_id)
                if user_sentiment:
                    similarities = self._filter_by_sentiment(
                        similarities,
                        user_sentiment
                    )
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error recommending for user: {e}", exc_info=True)
            return self._get_trending_recommendations(limit)
    
    def recommend_by_tags(
        self,
        tags: List[str],
        limit: int = 10
    ) -> List[PublishedChat]:
        """
        Recommend chats based on tags
        
        Args:
            tags: List of tags
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended chats
        """
        try:
            # Build query directly to avoid circular dependency
            from sqlalchemy import or_, func
            
            q = self.db.query(PublishedChat).filter(PublishedChat.is_public == True)
            
            # Filter by tags
            if tags:
                tag_filters = [
                    func.lower(PublishedChat.tags).like(f"%{tag.lower()}%")
                    for tag in tags
                ]
                if tag_filters:
                    q = q.filter(or_(*tag_filters))
            
            # Order by score
            chats = q.order_by(
                desc(PublishedChat.score),
                desc(PublishedChat.created_at)
            ).limit(limit).all()
            
            return chats
            
        except Exception as e:
            logger.error(f"Error recommending by tags: {e}", exc_info=True)
            return []
    
    def _get_trending_recommendations(self, limit: int) -> List[Dict[str, Any]]:
        """Get trending chats as fallback recommendations"""
        try:
            chats = self.db.query(PublishedChat).filter(
                PublishedChat.is_public == True
            ).order_by(
                desc(PublishedChat.score),
                desc(PublishedChat.created_at)
            ).limit(limit).all()
            
            return [
                {"chat_id": chat.id, "similarity": chat.score, "chat": chat}
                for chat in chats
            ]
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}", exc_info=True)
            return []
    
    def _get_user_sentiment_preference(self, user_id: str) -> Optional[str]:
        """Get user's preferred sentiment based on their liked content"""
        try:
            # Get sentiment of user's upvoted chats
            upvoted_chats = self.db.query(ChatVote.chat_id).filter(
                ChatVote.user_id == user_id,
                ChatVote.vote_type == "upvote"
            ).all()
            
            if not upvoted_chats:
                return None
            
            chat_ids = [vote.chat_id for vote in upvoted_chats]
            
            # Get sentiment metadata
            sentiments = self.db.query(ChatAIMetadata.sentiment_label).filter(
                ChatAIMetadata.chat_id.in_(chat_ids),
                ChatAIMetadata.sentiment_label.isnot(None)
            ).all()
            
            if not sentiments:
                return None
            
            # Count sentiment labels
            from collections import Counter
            sentiment_counts = Counter([s.sentiment_label for s in sentiments])
            
            # Return most common sentiment
            return sentiment_counts.most_common(1)[0][0] if sentiment_counts else None
            
        except Exception as e:
            logger.error(f"Error getting user sentiment preference: {e}", exc_info=True)
            return None
    
    def _filter_by_sentiment(
        self,
        recommendations: List[Dict[str, Any]],
        preferred_sentiment: str
    ) -> List[Dict[str, Any]]:
        """Filter recommendations by sentiment preference"""
        try:
            chat_ids = [rec["chat_id"] for rec in recommendations]
            
            # Get sentiment metadata
            sentiment_map = {}
            sentiments = self.db.query(ChatAIMetadata).filter(
                ChatAIMetadata.chat_id.in_(chat_ids),
                ChatAIMetadata.sentiment_label == preferred_sentiment
            ).all()
            
            for sent in sentiments:
                sentiment_map[sent.chat_id] = sent.sentiment_label
            
            # Prioritize chats with matching sentiment
            filtered = []
            for rec in recommendations:
                if rec["chat_id"] in sentiment_map:
                    rec["sentiment_match"] = True
                    filtered.insert(0, rec)  # Add to front
                else:
                    rec["sentiment_match"] = False
                    filtered.append(rec)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering by sentiment: {e}", exc_info=True)
            return recommendations

