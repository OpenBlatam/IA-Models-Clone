"""
Session Management for Recovery AI
"""

import uuid
import time
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


class Session:
    """User session"""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize session
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            metadata: Optional metadata
        """
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.data = {}
    
    def update_activity(self):
        """Update last activity time"""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """
        Check if session is expired
        
        Args:
            timeout_minutes: Timeout in minutes
        
        Returns:
            True if expired
        """
        elapsed = datetime.now() - self.last_activity
        return elapsed > timedelta(minutes=timeout_minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "data": self.data
        }


class SessionManager:
    """Manage user sessions"""
    
    def __init__(self, default_timeout_minutes: int = 30):
        """
        Initialize session manager
        
        Args:
            default_timeout_minutes: Default session timeout
        """
        self.sessions: Dict[str, Session] = {}
        self.default_timeout = default_timeout_minutes
        self.lock = Lock()
        logger.info("SessionManager initialized")
    
    def create_session(
        self,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new session
        
        Args:
            user_id: User identifier
            metadata: Optional metadata
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        with self.lock:
            session = Session(session_id, user_id, metadata)
            self.sessions[session_id] = session
        
        logger.info(f"Session created: {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def update_session(
        self,
        session_id: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Update session
        
        Args:
            session_id: Session identifier
            data: Optional data to update
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.update_activity()
                if data:
                    session.data.update(data)
    
    def delete_session(self, session_id: str):
        """Delete session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Session deleted: {session_id}")
    
    def cleanup_expired(self):
        """Cleanup expired sessions"""
        with self.lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(self.default_timeout)
            ]
            
            for sid in expired:
                del self.sessions[sid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for user"""
        with self.lock:
            return [
                session for session in self.sessions.values()
                if session.user_id == user_id
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self.lock:
            total = len(self.sessions)
            active = sum(
                1 for session in self.sessions.values()
                if not session.is_expired(self.default_timeout)
            )
            
            return {
                "total_sessions": total,
                "active_sessions": active,
                "expired_sessions": total - active
            }

