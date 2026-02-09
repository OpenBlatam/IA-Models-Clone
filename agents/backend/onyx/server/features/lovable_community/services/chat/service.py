"""
Chat Service (Modular)

Main service orchestrating chat operations using specialized modules.
"""

# Standard library imports
import logging
from typing import Optional, List, Dict, Any, Tuple

# Local application imports - Repositories
from ...repositories import (
    ChatRepository,
    RemixRepository,
    VoteRepository,
    ViewRepository,
)

# Local application imports - Models
from ...models import PublishedChat, ChatRemix, ChatVote, ChatView

# Local application imports - Exceptions
from ...exceptions import (
    ChatNotFoundError,
    InvalidChatError,
    DuplicateVoteError,
    RemixError,
    DatabaseError
)

# Local application imports - Helpers
from ...helpers import generate_id, get_current_timestamp

# Local application imports - Services
from ..ranking import RankingService

# Local application imports - Chat module components
from .validators import ChatValidator
from .processors import ChatAIProcessor
from .handlers import VoteHandler, ViewHandler, RemixHandler
from .managers import ScoreManager
from ...services.error_handling import log_and_raise_database_error

logger = logging.getLogger(__name__)


class ChatService:
    """
    Chat Service using Repository Pattern.
    
    All data access is delegated to repositories, following
    Dependency Inversion Principle.
    """
    
    def __init__(
        self,
        chat_repository: ChatRepository,
        remix_repository: RemixRepository,
        vote_repository: VoteRepository,
        view_repository: ViewRepository,
        ranking_service: Optional[RankingService] = None,
        validator: Optional[ChatValidator] = None,
        ai_processor: Optional[ChatAIProcessor] = None,
        score_manager: Optional[ScoreManager] = None
    ):
        """
        Initialize ChatService with dependencies.
        
        Uses Dependency Injection for all components following SOLID principles.
        
        Args:
            chat_repository: Repository for chat operations
            remix_repository: Repository for remix operations
            vote_repository: Repository for vote operations
            view_repository: Repository for view operations
            ranking_service: Service for ranking calculations
            validator: Validator instance (created if not provided)
            ai_processor: AI processor instance (created if not provided)
            score_manager: Score manager instance (created if not provided)
        """
        self.chat_repository = chat_repository
        self.remix_repository = remix_repository
        self.vote_repository = vote_repository
        self.view_repository = view_repository
        self.ranking_service = ranking_service or RankingService()
        
        self.validator = validator or ChatValidator()
        self.ai_processor = ai_processor or ChatAIProcessor(chat_repository)
        self.score_manager = score_manager or ScoreManager(self.ranking_service)
    
    def _get_chat_or_raise(self, chat_id: str) -> PublishedChat:
        """Get chat by ID or raise ChatNotFoundError."""
        chat = self.chat_repository.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundError(chat_id)
        return chat
    
    def _update_chat_score(
        self,
        chat: PublishedChat,
        vote_count: Optional[int] = None,
        remix_count: Optional[int] = None,
        view_count: Optional[int] = None
    ) -> float:
        """Calculate and update chat score."""
        return self.score_manager.calculate_score(chat, vote_count, remix_count, view_count)
    
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
        """
        Publish a new chat.
        
        Args:
            user_id: User ID
            title: Chat title
            chat_content: Chat content
            description: Optional description
            tags: Optional tags list
            is_public: Whether chat is public
            original_chat_id: Original chat ID if this is a remix
            
        Returns:
            Published chat
            
        Raises:
            InvalidChatError: If validation fails
            ChatNotFoundError: If original chat not found
            DatabaseError: If database operation fails
        """
        user_id = self.validator.validate_user_id(user_id)
        title = self.validator.validate_title(title)
        chat_content = self.validator.validate_chat_content(chat_content)
        
        if original_chat_id:
            self._get_chat_or_raise(original_chat_id)
        
        try:
            chat_id = generate_id()
            tags_str = self.validator.process_tags(tags)
            now = get_current_timestamp()
            
            chat = self.chat_repository.create(
                id=chat_id,
                user_id=user_id,
                title=title,
                description=description.strip() if description else None,
                chat_content=chat_content,
                tags=tags_str,
                is_public=is_public,
                original_chat_id=original_chat_id,
                created_at=now,
                updated_at=now
            )
            
            logger.info(f"Chat published: {chat_id} by user: {user_id}")
            
            try:
                self.ai_processor.process_chat(chat)
            except Exception as ai_error:
                logger.warning(f"AI processing failed for chat {chat_id}: {ai_error}")
            
            return chat
        except (InvalidChatError, ChatNotFoundError):
            raise
        except Exception as e:
            log_and_raise_database_error("publishing chat", e, {"chat_id": chat_id, "user_id": user_id})
    
    def get_chat(self, chat_id: str, user_id: Optional[str] = None) -> Optional[PublishedChat]:
        """
        Get a chat by ID.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID to record view
            
        Returns:
            Chat or None if not found
        """
        try:
            chat_id = self.validator.validate_chat_id(chat_id)
        except InvalidChatError:
            return None
        
        try:
            chat = self.chat_repository.get_by_id(chat_id)
            
            if chat and user_id:
                self.record_view(chat_id, user_id, chat)
            
            return chat
        except Exception as e:
            log_and_raise_database_error("getting chat", e, {"chat_id": chat_id})
    
    def record_view(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        chat: Optional[PublishedChat] = None
    ) -> None:
        """
        Record a view for a chat.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID
            chat: Optional chat object to avoid extra query
        """
        chat_id = self.validator.validate_chat_id(chat_id)
        
        try:
            if not chat:
                chat = self._get_chat_or_raise(chat_id)
            
            ViewHandler.create_view_record(self.view_repository, chat_id, user_id)
            
            new_view_count = chat.view_count + 1
            new_score = self._update_chat_score(chat, view_count=new_view_count)
            
            self.chat_repository.increment_view_count_and_score(chat_id, new_score)
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            log_and_raise_database_error("recording view", e, {"chat_id": chat_id, "user_id": user_id})
    
    def vote_chat(self, chat_id: str, user_id: str, vote_type: str) -> ChatVote:
        """
        Vote on a chat.
        
        Args:
            chat_id: Chat ID
            user_id: User ID
            vote_type: 'upvote' or 'downvote'
            
        Returns:
            Vote object
        """
        chat_id = self.validator.validate_chat_id(chat_id)
        user_id = self.validator.validate_user_id(user_id)
        VoteHandler.validate_vote_type(vote_type)
        
        try:
            chat = self._get_chat_or_raise(chat_id)
            existing_vote = self.vote_repository.get_user_vote(chat_id, user_id)
            
            if existing_vote:
                if existing_vote.vote_type == vote_type:
                    return existing_vote
                vote_increment, _ = VoteHandler.calculate_vote_increment(existing_vote, vote_type)
                self.vote_repository.update(existing_vote.id, vote_type=vote_type)
                vote = existing_vote
            else:
                vote_increment, _ = VoteHandler.calculate_vote_increment(None, vote_type)
                vote_id = generate_id()
                vote = self.vote_repository.create(
                    id=vote_id,
                    chat_id=chat_id,
                    user_id=user_id,
                    vote_type=vote_type,
                    created_at=get_current_timestamp()
                )
            
            from ...helpers.math_helpers import safe_increment
            
            new_vote_count = safe_increment(chat.vote_count, vote_increment)
            new_score = self._update_chat_score(chat, vote_count=new_vote_count)
            
            self.chat_repository.increment_vote_count_and_score(chat_id, vote_increment, new_score)
            
            logger.info(f"Vote recorded: {vote_type} on chat {chat_id} by user {user_id}")
            
            return vote
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            log_and_raise_database_error("voting chat", e, {"chat_id": chat_id, "user_id": user_id, "vote_type": vote_type})
    
    def remix_chat(
        self,
        original_chat_id: str,
        user_id: str,
        title: str,
        chat_content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[PublishedChat, ChatRemix]:
        """
        Create a remix of a chat.
        
        Args:
            original_chat_id: Original chat ID
            user_id: User ID creating remix
            title: Remix title
            chat_content: Remix content
            description: Optional description
            tags: Optional tags
            
        Returns:
            Tuple of (remix_chat, remix_record)
        """
        original_chat_id = self.validator.validate_chat_id(original_chat_id)
        user_id = self.validator.validate_user_id(user_id)
        title = self.validator.validate_title(title)
        chat_content = self.validator.validate_chat_content(chat_content)
        
        try:
            original_chat = self._get_chat_or_raise(original_chat_id)
            
            remix_chat = self.publish_chat(
                user_id=user_id,
                title=title,
                chat_content=chat_content,
                description=description,
                tags=tags,
                original_chat_id=original_chat_id
            )
            
            remix = RemixHandler.create_remix_record(
                self.remix_repository,
                original_chat_id,
                remix_chat.id,
                user_id
            )
            
            new_remix_count = original_chat.remix_count + 1
            new_score = self._update_chat_score(original_chat, remix_count=new_remix_count)
            
            self.chat_repository.increment_remix_count_and_score(original_chat_id, new_score)
            
            logger.info(f"Chat remixed: {remix_chat.id} from {original_chat_id} by user: {user_id}")
            
            return remix_chat, remix
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            logger.error(f"Error remixing chat {original_chat_id}: {e}", exc_info=True, extra={"original_chat_id": original_chat_id, "user_id": user_id})
            raise RemixError(str(e), original_chat_id) from e
    
    def search_chats(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        sort_by: str = "score",
        order: str = "desc",
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[PublishedChat], int]:
        """
        Search for chats.
        
        Args:
            query: Search query string
            tags: Filter by tags
            user_id: Filter by user ID
            sort_by: Sort field
            order: Sort order
            page: Page number
            page_size: Page size
            
        Returns:
            Tuple of (chats list, total count)
        """
        try:
            from ...config import settings
            from ...helpers.pagination import validate_and_calculate_pagination
            
            page, page_size, skip = validate_and_calculate_pagination(
                page,
                page_size,
                settings.max_page,
                settings.max_page_size
            )
            
            if query and query.strip():
                chats = self.chat_repository.search_by_query(query.strip(), skip, page_size)
                total = self.chat_repository.count_search_results(query=query.strip())
            elif tags:
                chats = self.chat_repository.get_by_tags(tags, skip, page_size)
                total = self.chat_repository.count_search_results(tags=tags)
            elif user_id:
                chats = self.chat_repository.get_by_user_id(user_id, skip, page_size)
                total = self.chat_repository.count_search_results(user_id=user_id)
            else:
                chats = self.chat_repository.get_public_chats(skip, page_size, sort_by, order)
                total = self.chat_repository.count(is_public=True)
            
            return chats, total
        except Exception as e:
            log_and_raise_database_error("searching chats", e, {"query": query, "tags": tags, "user_id": user_id})
    
    def get_top_chats(self, limit: int = 20) -> List[PublishedChat]:
        """Get top chats by score."""
        return self.chat_repository.get_public_chats(0, limit, "score", "desc")
    
    def get_user_vote(self, chat_id: str, user_id: str) -> Optional[ChatVote]:
        """Get user's vote for a chat."""
        return self.vote_repository.get_user_vote(chat_id, user_id)
    
    def get_user_votes_batch(
        self,
        chat_ids: List[str],
        user_id: str
    ) -> Dict[str, ChatVote]:
        """Get user's votes for multiple chats."""
        if not chat_ids or not user_id:
            return {}
        
        return self.vote_repository.get_user_votes_batch(chat_ids, user_id)
    
    def get_remixes(self, chat_id: str, limit: int = 20) -> List[ChatRemix]:
        """Get remixes of a chat."""
        chat_id = self.validator.validate_chat_id(chat_id)
        self._get_chat_or_raise(chat_id)
        return self.remix_repository.get_by_original_chat_id(chat_id)[:limit]
    
    def update_chat(
        self,
        chat_id: str,
        user_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = None
    ) -> PublishedChat:
        """Update a chat."""
        chat_id = self.validator.validate_chat_id(chat_id)
        user_id = self.validator.validate_user_id(user_id)
        
        try:
            chat = self._get_chat_or_raise(chat_id)
            self.validator.ensure_ownership(chat, user_id)
            
            update_data = {}
            if title is not None:
                update_data["title"] = self.validator.validate_title(title)
            
            if description is not None:
                update_data["description"] = description.strip() if description.strip() else None
            
            if tags is not None:
                update_data["tags"] = self.validator.process_tags(tags)
            
            if is_public is not None:
                update_data["is_public"] = is_public
            
            update_data["updated_at"] = get_current_timestamp()
            
            updated_chat = self.chat_repository.update(chat_id, **update_data)
            if not updated_chat:
                raise ChatNotFoundError(chat_id)
            
            logger.info(f"Chat updated: {chat_id} by user: {user_id}")
            return updated_chat
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            log_and_raise_database_error("updating chat", e, {"chat_id": chat_id, "user_id": user_id})
    
    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        """Delete a chat."""
        chat_id = self.validator.validate_chat_id(chat_id)
        user_id = self.validator.validate_user_id(user_id)
        
        try:
            chat = self._get_chat_or_raise(chat_id)
            self.validator.ensure_ownership(chat, user_id)
            
            self.vote_repository.delete_by_chat_id(chat_id)
            self.view_repository.delete_by_chat_id(chat_id)
            self.remix_repository.delete_by_original_chat_id(chat_id)
            
            result = self.chat_repository.delete(chat_id)
            
            logger.info(f"Chat deleted: {chat_id} by user: {user_id}")
            return result
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            log_and_raise_database_error("deleting chat", e, {"chat_id": chat_id, "user_id": user_id})
    
    def feature_chat(self, chat_id: str, featured: bool = True) -> PublishedChat:
        """Feature or unfeature a chat."""
        chat_id = self.validator.validate_chat_id(chat_id)
        
        try:
            updated_chat = self.chat_repository.update(
                chat_id,
                is_featured=featured,
                updated_at=get_current_timestamp()
            )
            if not updated_chat:
                raise ChatNotFoundError(chat_id)
            
            logger.info(f"Chat {'featured' if featured else 'unfeatured'}: {chat_id}")
            return updated_chat
        except (ChatNotFoundError, InvalidChatError):
            raise
        except Exception as e:
            log_and_raise_database_error("featuring chat", e, {"chat_id": chat_id, "featured": featured})
    
    def get_trending_chats(
        self,
        period: str = "day",
        limit: int = 20
    ) -> List[PublishedChat]:
        """Get trending chats."""
        if period not in ("hour", "day", "week", "month"):
            raise InvalidChatError(f"Invalid period: {period}")
        
        period_hours = {
            "hour": 1,
            "day": 24,
            "week": 168,
            "month": 720
        }
        
        return self.chat_repository.get_trending(period_hours[period], limit)
    
    def get_chat_rank(self, chat_id: str) -> int:
        """
        Get the rank of a chat based on its score.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Rank (1-based, where 1 is the highest score)
            
        Raises:
            ChatNotFoundError: If chat doesn't exist
        """
        chat_id = self.validator.validate_chat_id(chat_id)
        chat = self._get_chat_or_raise(chat_id)
        return self.chat_repository.get_rank_by_score(chat.score)
    
    def get_featured_chats(
        self,
        limit: int = 50,
        user_id: Optional[str] = None
    ) -> Tuple[List[PublishedChat], Dict[str, ChatVote]]:
        """
        Get featured chats with user votes if user_id provided.
        
        Args:
            limit: Maximum number of featured chats
            user_id: Optional user ID to get their votes
            
        Returns:
            Tuple of (chats list, user_votes dict)
        """
        chats = self.chat_repository.get_featured_chats(limit)
        
        user_votes = {}
        if user_id and chats:
            chat_ids = [chat.id for chat in chats]
            user_votes = self.vote_repository.get_user_votes_batch(chat_ids, user_id)
        
        return chats, user_votes

