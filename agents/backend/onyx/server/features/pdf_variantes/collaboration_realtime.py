"""
PDF Variantes - Real-time Collaboration Engine
=============================================

Real-time collaboration features for PDF editing and annotation.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class CollaborationEventType(str, Enum):
    """Collaboration event types."""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ANNOTATION_ADDED = "annotation_added"
    ANNOTATION_MODIFIED = "annotation_modified"
    ANNOTATION_DELETED = "annotation_deleted"
    CURSOR_MOVED = "cursor_moved"
    TEXT_SELECTED = "text_selected"
    PAGE_CHANGED = "page_changed"
    COMMENT_ADDED = "comment_added"
    COMMENT_REPLIED = "comment_replied"
    DOCUMENT_MODIFIED = "document_modified"
    PERMISSION_CHANGED = "permission_changed"


class UserRole(str, Enum):
    """User roles in collaboration."""
    OWNER = "owner"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class CollaborationStatus(str, Enum):
    """Collaboration session status."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


@dataclass
class CollaborationUser:
    """Collaboration user."""
    user_id: str
    username: str
    email: str
    role: UserRole
    avatar_url: Optional[str] = None
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_online: bool = False
    current_page: int = 1
    cursor_position: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "avatar_url": self.avatar_url,
            "joined_at": self.joined_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_online": self.is_online,
            "current_page": self.current_page,
            "cursor_position": self.cursor_position
        }


@dataclass
class CollaborationEvent:
    """Collaboration event."""
    event_id: str
    event_type: CollaborationEventType
    user_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    page_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "page_number": self.page_number
        }


@dataclass
class CollaborationSession:
    """Collaboration session."""
    session_id: str
    document_id: str
    owner_id: str
    title: str
    description: Optional[str] = None
    status: CollaborationStatus = CollaborationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    max_participants: int = 10
    users: Dict[str, CollaborationUser] = field(default_factory=dict)
    events: List[CollaborationEvent] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "document_id": self.document_id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_participants": self.max_participants,
            "users": {uid: user.to_dict() for uid, user in self.users.items()},
            "events_count": len(self.events),
            "settings": self.settings
        }


class RealTimeCollaborationEngine:
    """Real-time collaboration engine."""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.event_handlers: Dict[CollaborationEventType, List[callable]] = {}
        self.websocket_connections: Dict[str, Set[str]] = {}  # session_id -> user_ids
        self.cleanup_task = None
        logger.info("Initialized Real-time Collaboration Engine")
    
    async def create_session(
        self,
        document_id: str,
        owner_id: str,
        title: str,
        description: Optional[str] = None,
        max_participants: int = 10,
        expires_in_hours: Optional[int] = None
    ) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        session = CollaborationSession(
            session_id=session_id,
            document_id=document_id,
            owner_id=owner_id,
            title=title,
            description=description,
            max_participants=max_participants,
            expires_at=expires_at
        )
        
        # Add owner as first user
        await self.add_user_to_session(session_id, owner_id, UserRole.OWNER)
        
        self.sessions[session_id] = session
        
        # Initialize user sessions tracking
        if owner_id not in self.user_sessions:
            self.user_sessions[owner_id] = set()
        self.user_sessions[owner_id].add(session_id)
        
        # Initialize websocket connections
        self.websocket_connections[session_id] = set()
        
        logger.info(f"Created collaboration session: {session_id}")
        return session
    
    async def add_user_to_session(
        self,
        session_id: str,
        user_id: str,
        role: UserRole,
        username: str = "Anonymous",
        email: str = ""
    ) -> bool:
        """Add user to collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if session is full
        if len(session.users) >= session.max_participants:
            logger.warning(f"Session {session_id} is full")
            return False
        
        # Check if user already in session
        if user_id in session.users:
            return True
        
        # Create user
        user = CollaborationUser(
            user_id=user_id,
            username=username,
            email=email,
            role=role
        )
        
        session.users[user_id] = user
        
        # Update user sessions tracking
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)
        
        # Create join event
        await self._create_event(
            session_id,
            CollaborationEventType.USER_JOINED,
            user_id,
            {"username": username, "role": role.value}
        )
        
        logger.info(f"Added user {user_id} to session {session_id}")
        return True
    
    async def remove_user_from_session(self, session_id: str, user_id: str) -> bool:
        """Remove user from collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return False
        
        # Create leave event
        await self._create_event(
            session_id,
            CollaborationEventType.USER_LEFT,
            user_id,
            {"username": session.users[user_id].username}
        )
        
        # Remove user
        del session.users[user_id]
        
        # Update user sessions tracking
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
        
        # Remove websocket connection
        if session_id in self.websocket_connections:
            self.websocket_connections[session_id].discard(user_id)
        
        logger.info(f"Removed user {user_id} from session {session_id}")
        return True
    
    async def update_user_activity(
        self,
        session_id: str,
        user_id: str,
        page_number: Optional[int] = None,
        cursor_position: Optional[Dict[str, Any]] = None
    ):
        """Update user activity."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return
        
        user = session.users[user_id]
        user.last_activity = datetime.utcnow()
        user.is_online = True
        
        if page_number is not None:
            user.current_page = page_number
        
        if cursor_position is not None:
            user.cursor_position = cursor_position
        
        # Create cursor moved event
        if cursor_position is not None:
            await self._create_event(
                session_id,
                CollaborationEventType.CURSOR_MOVED,
                user_id,
                {"cursor_position": cursor_position, "page_number": page_number},
                page_number
            )
    
    async def add_annotation(
        self,
        session_id: str,
        user_id: str,
        annotation_data: Dict[str, Any],
        page_number: int
    ) -> bool:
        """Add annotation to collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        
        # Check permissions
        if user.role == UserRole.VIEWER:
            return False
        
        # Create annotation event
        await self._create_event(
            session_id,
            CollaborationEventType.ANNOTATION_ADDED,
            user_id,
            annotation_data,
            page_number
        )
        
        logger.info(f"Added annotation by user {user_id} in session {session_id}")
        return True
    
    async def modify_annotation(
        self,
        session_id: str,
        user_id: str,
        annotation_id: str,
        modification_data: Dict[str, Any],
        page_number: int
    ) -> bool:
        """Modify annotation in collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        
        # Check permissions
        if user.role == UserRole.VIEWER:
            return False
        
        # Create modification event
        await self._create_event(
            session_id,
            CollaborationEventType.ANNOTATION_MODIFIED,
            user_id,
            {"annotation_id": annotation_id, "modifications": modification_data},
            page_number
        )
        
        logger.info(f"Modified annotation {annotation_id} by user {user_id}")
        return True
    
    async def delete_annotation(
        self,
        session_id: str,
        user_id: str,
        annotation_id: str,
        page_number: int
    ) -> bool:
        """Delete annotation from collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        
        # Check permissions
        if user.role == UserRole.VIEWER:
            return False
        
        # Create deletion event
        await self._create_event(
            session_id,
            CollaborationEventType.ANNOTATION_DELETED,
            user_id,
            {"annotation_id": annotation_id},
            page_number
        )
        
        logger.info(f"Deleted annotation {annotation_id} by user {user_id}")
        return True
    
    async def add_comment(
        self,
        session_id: str,
        user_id: str,
        comment_data: Dict[str, Any],
        page_number: int
    ) -> bool:
        """Add comment to collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return False
        
        # Create comment event
        await self._create_event(
            session_id,
            CollaborationEventType.COMMENT_ADDED,
            user_id,
            comment_data,
            page_number
        )
        
        logger.info(f"Added comment by user {user_id} in session {session_id}")
        return True
    
    async def _create_event(
        self,
        session_id: str,
        event_type: CollaborationEventType,
        user_id: str,
        data: Dict[str, Any],
        page_number: Optional[int] = None
    ):
        """Create collaboration event."""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            data=data,
            page_number=page_number
        )
        
        session = self.sessions[session_id]
        session.events.append(event)
        session.updated_at = datetime.utcnow()
        
        # Trigger event handlers
        await self._trigger_event_handlers(event)
        
        # Broadcast to websocket connections
        await self._broadcast_event(session_id, event)
    
    async def _trigger_event_handlers(self, event: CollaborationEvent):
        """Trigger registered event handlers."""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    async def _broadcast_event(self, session_id: str, event: CollaborationEvent):
        """Broadcast event to websocket connections."""
        # Mock implementation - would use actual websocket broadcasting
        connected_users = self.websocket_connections.get(session_id, set())
        
        for user_id in connected_users:
            # Send event to user's websocket connection
            logger.debug(f"Broadcasting event {event.event_id} to user {user_id}")
    
    def register_event_handler(self, event_type: CollaborationEventType, handler: callable):
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session."""
        return self.sessions.get(session_id)
    
    async def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Get user's collaboration sessions."""
        session_ids = self.user_sessions.get(user_id, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    async def get_session_events(
        self,
        session_id: str,
        limit: int = 100,
        event_types: Optional[List[CollaborationEventType]] = None
    ) -> List[CollaborationEvent]:
        """Get session events."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        events = session.events
        
        # Filter by event types if specified
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        # Return latest events
        return events[-limit:] if limit else events
    
    async def pause_session(self, session_id: str, user_id: str) -> bool:
        """Pause collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if user has permission (owner or editor)
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        if user.role not in [UserRole.OWNER, UserRole.EDITOR]:
            return False
        
        session.status = CollaborationStatus.PAUSED
        session.updated_at = datetime.utcnow()
        
        logger.info(f"Paused session {session_id}")
        return True
    
    async def resume_session(self, session_id: str, user_id: str) -> bool:
        """Resume collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if user has permission (owner or editor)
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        if user.role not in [UserRole.OWNER, UserRole.EDITOR]:
            return False
        
        session.status = CollaborationStatus.ACTIVE
        session.updated_at = datetime.utcnow()
        
        logger.info(f"Resumed session {session_id}")
        return True
    
    async def end_session(self, session_id: str, user_id: str) -> bool:
        """End collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if user has permission (owner only)
        if user_id not in session.users:
            return False
        
        user = session.users[user_id]
        if user.role != UserRole.OWNER:
            return False
        
        session.status = CollaborationStatus.ENDED
        session.updated_at = datetime.utcnow()
        
        logger.info(f"Ended session {session_id}")
        return True
    
    async def cleanup_expired_sessions(self):
        """Cleanup expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.expires_at and session.expires_at < current_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.end_session(session_id, self.sessions[session_id].owner_id)
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get collaboration session statistics."""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.status == CollaborationStatus.ACTIVE)
        total_users = sum(len(s.users) for s in self.sessions.values())
        total_events = sum(len(s.events) for s in self.sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "paused_sessions": total_sessions - active_sessions,
            "total_users": total_users,
            "total_events": total_events,
            "average_users_per_session": total_users / total_sessions if total_sessions > 0 else 0,
            "average_events_per_session": total_events / total_sessions if total_sessions > 0 else 0
        }


# Global instance
real_time_collaboration_engine = RealTimeCollaborationEngine()
