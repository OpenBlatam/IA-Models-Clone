import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import desc, asc, func, or_, update

from ..models import PublishedChat, ChatRemix, ChatVote, ChatView
from ..exceptions import (
    ChatNotFoundError,
    InvalidChatError,
    DuplicateVoteError,
    RemixError,
    DatabaseError
)
from .ranking import RankingService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.ranking_service = RankingService()
        self._ai_services_initialized = False
        self._embedding_service = None
        self._sentiment_service = None
        self._moderation_service = None
    
    def _init_ai_services(self):
        """Lazy initialization of AI services"""
        if self._ai_services_initialized:
            return
        
        try:
            from ...config import settings
            if settings.ai_enabled:
                from .ai import EmbeddingService, SentimentService, ModerationService
                self._embedding_service = EmbeddingService(self.db)
                self._sentiment_service = SentimentService(self.db)
                self._moderation_service = ModerationService(self.db)
            self._ai_services_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize AI services: {e}")
            self._ai_services_initialized = True  # Mark as initialized to avoid repeated attempts

    def publish_chat(
        self,
        user_id: str,
        title: str,
        chat_content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = True,
        original_chat_id: Optional[str] = None
    ) -> PublishedChat:
        if not user_id or not user_id.strip():
            raise InvalidChatError("User ID cannot be empty")
        
        if not title or not title.strip():
            raise InvalidChatError("Title cannot be empty")
        
        if not chat_content or not chat_content.strip():
            raise InvalidChatError("Chat content cannot be empty")
        
        if original_chat_id:
            original_chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == original_chat_id
            ).first()
            if not original_chat:
                raise ChatNotFoundError(original_chat_id)
        
        try:
            chat_id = str(uuid.uuid4())
            
            tags_str = None
            if tags:
                valid_tags = {tag.strip().lower() for tag in tags if tag and tag.strip()}
                if valid_tags:
                    tags_str = ",".join(list(valid_tags)[:10])
            
            chat = PublishedChat(
                id=chat_id,
                user_id=user_id.strip(),
                title=title.strip(),
                description=description.strip() if description else None,
                chat_content=chat_content.strip(),
                tags=tags_str,
                is_public=is_public,
                original_chat_id=original_chat_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(chat)
            self.db.commit()
            self.db.refresh(chat)
            
            logger.info(f"Chat published: {chat_id} by user: {user_id}")
            
            # Process with AI services (async/background - don't block)
            try:
                self._process_chat_with_ai(chat)
            except Exception as ai_error:
                # Log but don't fail the publish operation
                logger.warning(f"AI processing failed for chat {chat_id}: {ai_error}")
            
            return chat
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error publishing chat: {e}", exc_info=True)
            if isinstance(e, (InvalidChatError, ChatNotFoundError)):
                raise
            raise DatabaseError(f"Failed to publish chat: {str(e)}")
    
    def get_chat(self, chat_id: str, user_id: Optional[str] = None) -> Optional[PublishedChat]:
        if not chat_id or not chat_id.strip():
            return None
        
        try:
            chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == chat_id.strip()
            ).first()
            
            if chat and user_id:
                self.record_view(chat_id, user_id, chat)
            
            return chat
        except Exception as e:
            logger.error(f"Error getting chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get chat: {str(e)}")
    
    def record_view(self, chat_id: str, user_id: Optional[str] = None, chat: Optional[PublishedChat] = None) -> None:
        if not chat_id or not chat_id.strip():
            raise InvalidChatError("Chat ID cannot be empty")
        
        try:
            if not chat:
                chat = self.db.query(PublishedChat).filter(
                    PublishedChat.id == chat_id.strip()
                ).first()
            
            if not chat:
                raise ChatNotFoundError(chat_id)
            
            view_id = str(uuid.uuid4())
            view = ChatView(
                id=view_id,
                chat_id=chat_id,
                user_id=user_id.strip() if user_id else None,
                created_at=datetime.utcnow()
            )
            
            self.db.add(view)
            
            new_view_count = chat.view_count + 1
            new_score = self.ranking_service.calculate_score(
                chat.vote_count,
                chat.remix_count,
                new_view_count,
                chat.created_at
            )
            
            self.db.execute(
                update(PublishedChat)
                .where(PublishedChat.id == chat_id)
                .values(view_count=new_view_count, score=new_score)
            )
            
            self.db.commit()
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error recording view for chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to record view: {str(e)}")
    
    def vote_chat(self, chat_id: str, user_id: str, vote_type: str) -> ChatVote:
        if not chat_id or not chat_id.strip():
            raise InvalidChatError("Chat ID cannot be empty")
        
        if not user_id or not user_id.strip():
            raise InvalidChatError("User ID cannot be empty")
        
        if vote_type not in ("upvote", "downvote"):
            raise InvalidChatError(f"Invalid vote type: {vote_type}. Must be 'upvote' or 'downvote'")
        
        try:
            chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == chat_id.strip()
            ).first()
            
            if not chat:
                raise ChatNotFoundError(chat_id)
            
            existing_vote = self.db.query(ChatVote).filter(
                ChatVote.chat_id == chat_id,
                ChatVote.user_id == user_id
            ).first()
            
            if existing_vote:
                if existing_vote.vote_type == vote_type:
                    return existing_vote
                
                if existing_vote.vote_type == "upvote" and vote_type == "downvote":
                    new_vote_count = chat.vote_count - 2
                elif existing_vote.vote_type == "downvote" and vote_type == "upvote":
                    new_vote_count = chat.vote_count + 2
                else:
                    new_vote_count = chat.vote_count
                
                existing_vote.vote_type = vote_type
                vote = existing_vote
            else:
                vote_id = str(uuid.uuid4())
                vote = ChatVote(
                    id=vote_id,
                    chat_id=chat_id,
                    user_id=user_id,
                    vote_type=vote_type,
                    created_at=datetime.utcnow()
                )
                self.db.add(vote)
                
                if vote_type == "upvote":
                    new_vote_count = chat.vote_count + 1
                else:
                    new_vote_count = chat.vote_count - 1
            
            new_score = self.ranking_service.calculate_score(
                new_vote_count,
                chat.remix_count,
                chat.view_count,
                chat.created_at
            )
            
            self.db.execute(
                update(PublishedChat)
                .where(PublishedChat.id == chat_id)
                .values(vote_count=new_vote_count, score=new_score)
            )
            
            chat.vote_count = new_vote_count
            chat.score = new_score
            
            self.db.commit()
            self.db.refresh(vote)
            
            logger.info(f"Vote recorded: {vote_type} on chat {chat_id} by user {user_id}")
            
            return vote
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error voting chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to vote chat: {str(e)}")
    
    def remix_chat(
        self,
        original_chat_id: str,
        user_id: str,
        title: str,
        chat_content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[PublishedChat, ChatRemix]:
        if not original_chat_id or not original_chat_id.strip():
            raise InvalidChatError("Original chat ID cannot be empty")
        
        if not user_id or not user_id.strip():
            raise InvalidChatError("User ID cannot be empty")
        
        if not title or not title.strip():
            raise InvalidChatError("Title cannot be empty")
        
        if not chat_content or not chat_content.strip():
            raise InvalidChatError("Chat content cannot be empty")
        
        try:
            original_chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == original_chat_id.strip()
            ).first()
            
            if not original_chat:
                raise ChatNotFoundError(original_chat_id)
            
            remix_chat = self.publish_chat(
                user_id=user_id,
                title=title,
                chat_content=chat_content,
                description=description,
                tags=tags,
                original_chat_id=original_chat_id
            )
            
            remix_id = str(uuid.uuid4())
            remix = ChatRemix(
                id=remix_id,
                original_chat_id=original_chat_id,
                remix_chat_id=remix_chat.id,
                user_id=user_id,
                created_at=datetime.utcnow()
            )
            
            self.db.add(remix)
            
            new_remix_count = original_chat.remix_count + 1
            new_score = self.ranking_service.calculate_score(
                original_chat.vote_count,
                new_remix_count,
                original_chat.view_count,
                original_chat.created_at
            )
            
            self.db.execute(
                update(PublishedChat)
                .where(PublishedChat.id == original_chat_id)
                .values(remix_count=new_remix_count, score=new_score)
            )
            
            original_chat.remix_count = new_remix_count
            original_chat.score = new_score
            
            self.db.commit()
            self.db.refresh(remix)
            
            logger.info(f"Chat remixed: {remix_chat.id} from {original_chat_id} by user: {user_id}")
            
            return remix_chat, remix
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error remixing chat {original_chat_id}: {e}", exc_info=True)
            raise RemixError(str(e), original_chat_id)
    
    def search_chats(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        sort_by: str = "score",
        order: str = "desc",
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[PublishedChat], int]:
        try:
            from ..utils import validate_page_params
            from ..config import settings
            
            page, page_size = validate_page_params(
                page,
                page_size,
                max_page=settings.max_page,
                max_page_size=settings.max_page_size
            )
        except ImportError:
            pass
        
        q = self.db.query(PublishedChat).filter(PublishedChat.is_public == True)
        
        if query and query.strip():
            search_term = f"%{query.strip().lower()}%"
            q = q.filter(
                or_(
                    func.lower(PublishedChat.title).like(search_term),
                    func.lower(PublishedChat.description).like(search_term),
                    func.lower(PublishedChat.tags).like(search_term)
                )
            )
        
        if tags:
            try:
                from ..utils import normalize_tags
                normalized_tags = normalize_tags(tags)
                if normalized_tags:
                    tag_filters = [
                        func.lower(PublishedChat.tags).like(f"%{tag}%")
                        for tag in normalized_tags
                    ]
                    q = q.filter(or_(*tag_filters))
            except ImportError:
                pass
        
        if user_id and user_id.strip():
            q = q.filter(PublishedChat.user_id == user_id.strip())
        
        sort_mapping = {
            "score": PublishedChat.score,
            "created_at": PublishedChat.created_at,
            "vote_count": PublishedChat.vote_count,
            "remix_count": PublishedChat.remix_count
        }
        
        sort_field = sort_mapping.get(sort_by, PublishedChat.score)
        order_func = desc(sort_field) if order == "desc" else asc(sort_field)
        
        if sort_by != "created_at":
            secondary_order = desc(PublishedChat.created_at) if order == "desc" else asc(PublishedChat.created_at)
            q = q.order_by(order_func, secondary_order)
        else:
            q = q.order_by(order_func)
        
        total = q.count()
        offset = (page - 1) * page_size
        chats = q.offset(offset).limit(page_size).all()
        
        return chats, total
    
    def get_top_chats(self, limit: int = 20) -> List[PublishedChat]:
        chats = self.db.query(PublishedChat).filter(
            PublishedChat.is_public == True
        ).order_by(
            desc(PublishedChat.score),
            desc(PublishedChat.created_at)
        ).limit(limit).all()
        
        return chats
    
    def get_user_vote(self, chat_id: str, user_id: str) -> Optional[ChatVote]:
        return self.db.query(ChatVote).filter(
            ChatVote.chat_id == chat_id,
            ChatVote.user_id == user_id
        ).first()
    
    def get_user_votes_batch(self, chat_ids: List[str], user_id: str) -> Dict[str, ChatVote]:
        if not chat_ids or not user_id:
            return {}
        
        votes = self.db.query(ChatVote).filter(
            ChatVote.chat_id.in_(chat_ids),
            ChatVote.user_id == user_id
        ).all()
        
        return {vote.chat_id: vote for vote in votes}
    
    def get_remixes(self, chat_id: str, limit: int = 20) -> List[ChatRemix]:
        chat = self.db.query(PublishedChat).filter(
            PublishedChat.id == chat_id
        ).first()
        
        if not chat:
            raise ChatNotFoundError(chat_id)
        
        remixes = self.db.query(ChatRemix).filter(
            ChatRemix.original_chat_id == chat_id
        ).order_by(
            desc(ChatRemix.created_at)
        ).limit(limit).all()
        
        return remixes
    
    def update_chat(
        self,
        chat_id: str,
        user_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = None
    ) -> PublishedChat:
        if not chat_id or not chat_id.strip():
            raise InvalidChatError("Chat ID cannot be empty")
        
        if not user_id or not user_id.strip():
            raise InvalidChatError("User ID cannot be empty")
        
        try:
            chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == chat_id.strip()
            ).first()
            
            if not chat:
                raise ChatNotFoundError(chat_id)
            
            if chat.user_id != user_id.strip():
                raise InvalidChatError("User is not the owner of this chat")
            
            if title is not None:
                if not title.strip():
                    raise InvalidChatError("Title cannot be empty")
                chat.title = title.strip()
            
            if description is not None:
                chat.description = description.strip() if description.strip() else None
            
            if tags is not None:
                valid_tags = [tag.strip().lower() for tag in tags if tag and tag.strip()]
                if valid_tags:
                    chat.tags = ",".join(valid_tags[:10])
                else:
                    chat.tags = None
            
            if is_public is not None:
                chat.is_public = is_public
            
            chat.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(chat)
            
            logger.info(f"Chat updated: {chat_id} by user: {user_id}")
            
            return chat
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update chat: {str(e)}")
    
    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        if not chat_id or not chat_id.strip():
            raise InvalidChatError("Chat ID cannot be empty")
        
        if not user_id or not user_id.strip():
            raise InvalidChatError("User ID cannot be empty")
        
        try:
            chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == chat_id.strip()
            ).first()
            
            if not chat:
                raise ChatNotFoundError(chat_id)
            
            if chat.user_id != user_id.strip():
                raise InvalidChatError("User is not the owner of this chat")
            
            self.db.query(ChatVote).filter(ChatVote.chat_id == chat_id).delete()
            self.db.query(ChatView).filter(ChatView.chat_id == chat_id).delete()
            
            self.db.delete(chat)
            self.db.commit()
            
            logger.info(f"Chat deleted: {chat_id} by user: {user_id}")
            
            return True
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete chat: {str(e)}")
    
    def feature_chat(self, chat_id: str, featured: bool = True) -> PublishedChat:
        if not chat_id or not chat_id.strip():
            raise InvalidChatError("Chat ID cannot be empty")
        
        try:
            chat = self.db.query(PublishedChat).filter(
                PublishedChat.id == chat_id.strip()
            ).first()
            
            if not chat:
                raise ChatNotFoundError(chat_id)
            
            chat.is_featured = featured
            chat.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(chat)
            
            logger.info(f"Chat {'featured' if featured else 'unfeatured'}: {chat_id}")
            
            return chat
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error featuring chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to feature chat: {str(e)}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        if not user_id or not user_id.strip():
            raise InvalidChatError("User ID cannot be empty")
        
        try:
            user_id = user_id.strip()
            
            total_chats = self.db.query(func.count(PublishedChat.id)).filter(
                PublishedChat.user_id == user_id
            ).scalar() or 0
            
            total_remixes = self.db.query(func.count(ChatRemix.id)).filter(
                ChatRemix.user_id == user_id
            ).scalar() or 0
            
            total_votes = self.db.query(func.count(ChatVote.id)).filter(
                ChatVote.user_id == user_id
            ).scalar() or 0
            
            profile_data = self.db.query(
                func.avg(PublishedChat.score).label('avg_score'),
                func.max(PublishedChat.score).label('max_score')
            ).filter(
                PublishedChat.user_id == user_id,
                PublishedChat.is_public == True
            ).first()
            
            average_score = float(profile_data.avg_score) if profile_data.avg_score else None
            
            top_chat = self.db.query(PublishedChat.id).filter(
                PublishedChat.user_id == user_id,
                PublishedChat.is_public == True
            ).order_by(desc(PublishedChat.score)).first()
            top_chat_id = top_chat.id if top_chat else None
            
            return {
                "user_id": user_id,
                "total_chats": total_chats,
                "total_remixes": total_remixes,
                "total_votes": total_votes,
                "average_score": average_score,
                "top_chat_id": top_chat_id
            }
        except Exception as e:
            logger.error(f"Error getting user profile {user_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get user profile: {str(e)}")
    
    def get_trending_chats(
        self,
        period: str = "day",
        limit: int = 20
    ) -> List[PublishedChat]:
        if period not in ("hour", "day", "week", "month"):
            raise InvalidChatError(f"Invalid period: {period}. Must be hour, day, week, or month")
        
        try:
            now = datetime.utcnow()
            
            if period == "hour":
                start_date = now - timedelta(hours=1)
            elif period == "day":
                start_date = now - timedelta(days=1)
            elif period == "week":
                start_date = now - timedelta(weeks=1)
            else:
                start_date = now - timedelta(days=30)
            
            chats = self.db.query(PublishedChat).filter(
                PublishedChat.is_public == True,
                PublishedChat.created_at >= start_date
            ).order_by(
                desc(PublishedChat.score),
                desc(PublishedChat.vote_count),
                desc(PublishedChat.view_count)
            ).limit(limit).all()
            
            return chats
        except (InvalidChatError):
            raise
        except Exception as e:
            logger.error(f"Error getting trending chats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get trending chats: {str(e)}")
    
    def get_analytics(
        self,
        period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            base_filter = PublishedChat.is_public == True
            if period_days:
                start_date = datetime.utcnow() - timedelta(days=period_days)
                base_filter = (PublishedChat.is_public == True) & (PublishedChat.created_at >= start_date)
            
            stats = self.db.query(
                func.count(PublishedChat.id).label('total_chats'),
                func.count(func.distinct(PublishedChat.user_id)).label('total_users'),
                func.sum(PublishedChat.vote_count).label('total_votes'),
                func.sum(PublishedChat.remix_count).label('total_remixes'),
                func.sum(PublishedChat.view_count).label('total_views'),
                func.avg(PublishedChat.score).label('avg_score')
            ).filter(base_filter).first()
            
            total_chats = stats.total_chats or 0
            total_users = stats.total_users or 0
            total_votes = stats.total_votes or 0
            total_remixes = stats.total_remixes or 0
            total_views = stats.total_views or 0
            average_score = float(stats.avg_score) if stats.avg_score else 0.0
            
            from collections import Counter
            
            all_tags = []
            chats_with_tags = self.db.query(PublishedChat.tags).filter(
                base_filter,
                PublishedChat.tags.isnot(None)
            ).all()
            for row in chats_with_tags:
                if row.tags:
                    all_tags.extend([tag.strip() for tag in row.tags.split(",") if tag.strip()])
            
            tag_counts = dict(Counter(all_tags))
            
            top_tags = sorted(
                [{"tag": tag, "count": count} for tag, count in tag_counts.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:10]
            
            return {
                "total_chats": total_chats,
                "total_users": total_users,
                "total_votes": total_votes,
                "total_remixes": total_remixes,
                "total_views": total_views,
                "average_score": round(average_score, 2),
                "top_tags": top_tags,
                "period_days": period_days
            }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get analytics: {str(e)}")
    
    def bulk_operation(
        self,
        chat_ids: List[str],
        operation: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not chat_ids:
            raise InvalidChatError("Chat IDs list cannot be empty")
        
        if operation not in ("delete", "feature", "unfeature", "make_public", "make_private"):
            raise InvalidChatError(f"Invalid operation: {operation}")
        
        if operation == "delete" and not user_id:
            raise InvalidChatError("User ID required for delete operation")
        
        successful = 0
        failed = 0
        failed_chat_ids = []
        errors = []
        
        try:
            for chat_id in chat_ids:
                try:
                    if operation == "delete":
                        self.delete_chat(chat_id, user_id)
                    elif operation == "feature":
                        self.feature_chat(chat_id, True)
                    elif operation == "unfeature":
                        self.feature_chat(chat_id, False)
                    elif operation == "make_public":
                        chat = self.get_chat(chat_id)
                        if chat:
                            chat.is_public = True
                            chat.updated_at = datetime.utcnow()
                            self.db.commit()
                    elif operation == "make_private":
                        chat = self.get_chat(chat_id)
                        if chat:
                            chat.is_public = False
                            chat.updated_at = datetime.utcnow()
                            self.db.commit()
                    
                    successful += 1
                except Exception as e:
                    failed += 1
                    failed_chat_ids.append(chat_id)
                    errors.append(f"Chat {chat_id}: {str(e)}")
                    logger.warning(f"Failed to {operation} chat {chat_id}: {e}")
            
            return {
                "operation": operation,
                "total_requested": len(chat_ids),
                "successful": successful,
                "failed": failed,
                "failed_chat_ids": failed_chat_ids,
                "errors": errors
            }
        except Exception as e:
            logger.error(f"Error in bulk operation {operation}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to perform bulk operation: {str(e)}")
    
    def get_chat_stats_detailed(self, chat_id: str) -> Dict[str, Any]:
        if not chat_id or not chat_id.strip():
            raise InvalidChatError("Chat ID cannot be empty")
        
        try:
            chat = self.get_chat(chat_id)
            
            if not chat:
                raise ChatNotFoundError(chat_id)
            
            from sqlalchemy import case
            
            vote_stats = self.db.query(
                func.sum(case((ChatVote.vote_type == "upvote", 1), else_=0)).label('upvotes'),
                func.sum(case((ChatVote.vote_type == "downvote", 1), else_=0)).label('downvotes')
            ).filter(
                ChatVote.chat_id == chat_id
            ).first()
            
            upvote_count = vote_stats.upvotes or 0
            downvote_count = vote_stats.downvotes or 0
            
            engagement_rate = None
            if chat.vote_count > 0:
                engagement_rate = round(chat.view_count / chat.vote_count, 2) if chat.vote_count > 0 else None
            
            rank = self.db.query(func.count(PublishedChat.id)).filter(
                PublishedChat.score > chat.score,
                PublishedChat.is_public == True
            ).scalar() + 1
            
            return {
                "chat_id": chat.id,
                "vote_count": chat.vote_count,
                "remix_count": chat.remix_count,
                "view_count": chat.view_count,
                "score": chat.score,
                "rank": rank,
                "upvote_count": upvote_count,
                "downvote_count": downvote_count,
                "engagement_rate": engagement_rate
            }
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            logger.error(f"Error getting detailed stats for chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to get chat stats: {str(e)}")
    
    def _process_chat_with_ai(self, chat: PublishedChat) -> None:
        """
        Process chat with AI services (moderation, sentiment, embeddings)
        
        This is called asynchronously after publishing to avoid blocking.
        """
        try:
            from ...config import settings
            if not settings.ai_enabled:
                return
            
            self._init_ai_services()
            
            # 1. Content moderation (check if content should be blocked)
            if self._moderation_service and settings.moderation_enabled:
                try:
                    moderation_result = self._moderation_service.moderate_chat(chat.id, chat=chat)
                    if moderation_result.is_toxic:
                        logger.warning(
                            f"Toxic content detected for chat {chat.id}: "
                            f"score={moderation_result.toxicity_score:.2f}"
                        )
                        # Optionally mark as not public or flag for review
                        # chat.is_public = False
                        # self.db.commit()
                except Exception as e:
                    logger.warning(f"Moderation failed for chat {chat.id}: {e}")
            
            # 2. Sentiment analysis
            if self._sentiment_service and settings.sentiment_enabled:
                try:
                    self._sentiment_service.analyze_chat_sentiment(chat.id, chat=chat)
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for chat {chat.id}: {e}")
            
            # 3. Generate embedding for semantic search
            if self._embedding_service:
                try:
                    self._embedding_service.create_or_update_embedding(chat.id, chat=chat)
                except Exception as e:
                    logger.warning(f"Embedding generation failed for chat {chat.id}: {e}")
            
        except Exception as e:
            logger.error(f"Error in AI processing for chat {chat.id}: {e}", exc_info=True)

