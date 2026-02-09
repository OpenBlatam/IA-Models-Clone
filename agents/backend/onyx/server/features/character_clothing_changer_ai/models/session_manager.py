"""
Session Manager for Flux2 Clothing Changer
===========================================

Advanced session management and tracking.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Session information."""
    session_id: str
    user_id: Optional[str]
    created_at: float
    last_activity: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}


class SessionManager:
    """Advanced session management system."""
    
    def __init__(
        self,
        session_timeout: float = 3600.0,  # 1 hour
        max_sessions_per_user: int = 10,
        enable_persistence: bool = True,
    ):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Session timeout in seconds
            max_sessions_per_user: Maximum sessions per user
            enable_persistence: Enable session persistence
        """
        self.session_timeout = session_timeout
        self.max_sessions_per_user = max_sessions_per_user
        self.enable_persistence = enable_persistence
        
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "expired_sessions": 0,
        }
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            user_id: Optional user ID
            ip_address: Optional IP address
            user_agent: Optional user agent
            metadata: Optional metadata
            
        Returns:
            Created session
        """
        session_id = str(uuid.uuid4())
        now = time.time()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )
        
        self.sessions[session_id] = session
        
        if user_id:
            # Limit sessions per user
            user_sessions = self.user_sessions[user_id]
            if len(user_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest = user_sessions[0]
                self.destroy_session(oldest)
                user_sessions.remove(oldest)
            
            user_sessions.append(session_id)
        
        self.stats["total_sessions"] += 1
        self.stats["active_sessions"] = len([s for s in self.sessions.values() if self.is_valid(s.session_id)])
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if expired
        if not self.is_valid(session_id):
            return None
        
        return session
    
    def update_session(
        self,
        session_id: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update session.
        
        Args:
            session_id: Session ID
            data: Optional data to update
            metadata: Optional metadata to update
            
        Returns:
            True if updated
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.last_activity = time.time()
        
        if data:
            session.data.update(data)
        
        if metadata:
            session.metadata.update(metadata)
        
        return True
    
    def is_valid(self, session_id: str) -> bool:
        """
        Check if session is valid.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if valid
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        elapsed = time.time() - session.last_activity
        
        return elapsed < self.session_timeout
    
    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if destroyed
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if session.user_id and session.user_id in self.user_sessions:
            if session_id in self.user_sessions[session.user_id]:
                self.user_sessions[session.user_id].remove(session_id)
        
        del self.sessions[session_id]
        self.stats["active_sessions"] = len([s for s in self.sessions.values() if self.is_valid(s.session_id)])
        
        logger.info(f"Destroyed session: {session_id}")
        return True
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        expired = []
        now = time.time()
        
        for session_id, session in self.sessions.items():
            elapsed = now - session.last_activity
            if elapsed >= self.session_timeout:
                expired.append(session_id)
        
        for session_id in expired:
            self.destroy_session(session_id)
            self.stats["expired_sessions"] += 1
        
        logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of sessions
        """
        session_ids = self.user_sessions.get(user_id, [])
        return [
            self.sessions[sid] for sid in session_ids
            if sid in self.sessions and self.is_valid(sid)
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            **self.stats,
            "total_sessions_stored": len(self.sessions),
            "unique_users": len(self.user_sessions),
            "average_sessions_per_user": (
                sum(len(sessions) for sessions in self.user_sessions.values()) / len(self.user_sessions)
                if self.user_sessions else 0.0
            ),
        }


