"""
Export Service for data export operations.
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging

from ..repositories.chat_repository import ChatRepository
from ..repositories.bookmark_repository import BookmarkRepository
from ..exceptions import NotFoundError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.service_base import BaseService

logger = logging.getLogger(__name__)


class ExportService(BaseService):
    """Service for export operations."""
    
    def __init__(self, db: Session):
        """Initialize export service."""
        super().__init__(db)
        self.chat_repo = ChatRepository(db)
        self.bookmark_repo = BookmarkRepository(db)
    
    def export_chat(
        self,
        chat_id: str,
        include_stats: bool = True,
        include_comments: bool = False,
        include_votes: bool = False
    ) -> Dict[str, Any]:
        """Export a chat with optional related data."""
        chat = self.chat_repo.get_by_id(chat_id)
        
        if not chat:
            return None
        
        export_data = {
            "chat": self.serialize_model(chat),
            "exported_at": self.get_current_timestamp().isoformat()
        }
        
        if include_stats:
            export_data["stats"] = {
                "vote_count": chat.vote_count,
                "remix_count": chat.remix_count,
                "view_count": chat.view_count,
                "score": chat.score
            }
        
        if include_comments:
            from ..repositories.comment_repository import CommentRepository
            from ..constants import MAX_EXPORT_COMMENTS
            comment_repo = CommentRepository(self.db)
            comments, _ = comment_repo.get_by_chat(chat_id, page=1, page_size=MAX_EXPORT_COMMENTS)
            export_data["comments"] = self.serialize_list(comments)
        
        if include_votes:
            from ..repositories.vote_repository import VoteRepository
            from ..constants import MAX_EXPORT_VOTES
            vote_repo = VoteRepository(self.db)
            votes = vote_repo.get_by_chat(chat_id, limit=MAX_EXPORT_VOTES)
            export_data["votes"] = self.serialize_list(votes)
        
        return export_data
    
    @log_execution_time
    @handle_errors
    def export_chat_csv(self, chat_id: str) -> str:
        """
        Export a chat as CSV format.
        
        Args:
            chat_id: ID of chat to export
            
        Returns:
            CSV formatted string
            
        Raises:
            NotFoundError: If chat doesn't exist
        """
        self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        chat = self.chat_repo.get_by_id(chat_id)  # Need chat object for CSV
        
        csv_lines = []
        csv_lines.append("Field,Value")
        csv_lines.append(f"id,{chat.id}")
        csv_lines.append(f"title,{chat.title}")
        csv_lines.append(f"user_id,{chat.user_id}")
        csv_lines.append(f"score,{chat.score}")
        csv_lines.append(f"vote_count,{chat.vote_count}")
        csv_lines.append(f"remix_count,{chat.remix_count}")
        csv_lines.append(f"view_count,{chat.view_count}")
        
        return "\n".join(csv_lines)
    
    def export_user_data(
        self,
        user_id: str,
        include_chats: bool = True,
        include_comments: bool = True,
        include_bookmarks: bool = True
    ) -> Dict[str, Any]:
        """Export all user data."""
        export_data = {
            "user_id": user_id,
            "exported_at": self.get_current_timestamp().isoformat()
        }
        
        from ..constants import MAX_EXPORT_CHATS, MAX_EXPORT_COMMENTS, MAX_EXPORT_BOOKMARKS
        
        if include_chats:
            chats, _ = self.chat_repo.get_by_user_id(user_id, page=1, page_size=MAX_EXPORT_CHATS)
            export_data["chats"] = self.serialize_list(chats)
        
        if include_comments:
            from ..repositories.comment_repository import CommentRepository
            comment_repo = CommentRepository(self.db)
            comments = comment_repo.get_by_user(user_id, limit=MAX_EXPORT_COMMENTS)
            export_data["comments"] = self.serialize_list(comments)
        
        if include_bookmarks:
            bookmarks, _ = self.bookmark_repo.get_by_user(user_id, page=1, page_size=MAX_EXPORT_BOOKMARKS)
            export_data["bookmarks"] = self.serialize_list(bookmarks)
        
        return export_data
    
    def export_analytics_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Export analytics summary."""
        # Use repository to get chats with date range filtering
        if start_date or end_date:
            # Use repository date range method
            all_chats = self.chat_repo.get_by_date_range(
                start_date=start_date,
                end_date=end_date,
                is_public=True,
                limit=10000  # Large limit for export
            )
        else:
            # Get all public chats using repository
            all_chats, _ = self.chat_repo.get_all(
                page=1,
                page_size=10000,  # Large page size for export
                filters={"is_public": True}
            )
        
        from ..utils.statistics_helpers import (
            calculate_field_stats,
            count_by_condition
        )
        
        # Calculate statistics using helper
        stats = calculate_field_stats(
            all_chats,
            {
                'vote_count': {'type': 'sum'},
                'remix_count': {'type': 'sum'},
                'view_count': {'type': 'sum'},
                'score': {'type': 'avg', 'round': 2}
            }
        )
        
        summary = {
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "total_chats": len(all_chats),
            "total_votes": stats['vote_count'],
            "total_remixes": stats['remix_count'],
            "total_views": stats['view_count'],
            "featured_chats": count_by_condition(
                all_chats,
                lambda c: c.is_featured
            ),
            "average_score": stats['score'],
            "exported_at": self.get_current_timestamp().isoformat()
        }
        
        return summary






