"""
Gamma App - Collaboration Engine
Advanced real-time collaboration features for content creation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4
import redis
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class CollaborationEventType(Enum):
    """Types of collaboration events"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CURSOR_UPDATE = "cursor_update"
    CONTENT_EDIT = "content_edit"
    SELECTION_CHANGE = "selection_change"
    COMMENT_ADDED = "comment_added"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"
    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"
    VERSION_UPDATE = "version_update"
    CONFLICT_RESOLUTION = "conflict_resolution"

class UserRole(Enum):
    """User roles in collaboration"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    COMMENTOR = "commentor"

@dataclass
class CollaborationUser:
    """User in collaboration session"""
    user_id: str
    username: str
    role: UserRole
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    is_typing: bool = False
    last_activity: datetime = None
    color: str = "#000000"

@dataclass
class CollaborationEvent:
    """Collaboration event"""
    id: str
    session_id: str
    user_id: str
    event_type: CollaborationEventType
    data: Dict[str, Any]
    timestamp: datetime
    version: int

@dataclass
class CollaborationSession:
    """Collaboration session"""
    id: str
    project_id: str
    content_id: str
    name: str
    owner_id: str
    users: Dict[str, CollaborationUser]
    content_version: int
    created_at: datetime
    last_activity: datetime
    settings: Dict[str, Any]

class CollaborationEngine:
    """
    Advanced collaboration engine for real-time content editing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize collaboration engine"""
        self.config = config or {}
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_colors: Dict[str, str] = {}
        self.redis_client = None
        
        # Initialize Redis for distributed collaboration
        self._init_redis()
        
        # Load user colors
        self._load_user_colors()
        
        logger.info("Collaboration Engine initialized successfully")

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 2),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for collaboration")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    def _load_user_colors(self):
        """Load user colors for collaboration"""
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
        self.user_colors = {str(i): colors[i % len(colors)] for i in range(100)}

    async def create_session(self, project_id: str, content_id: str, 
                           session_name: str, owner_id: str, 
                           owner_username: str) -> CollaborationSession:
        """Create new collaboration session"""
        try:
            session_id = str(uuid4())
            
            # Create owner user
            owner = CollaborationUser(
                user_id=owner_id,
                username=owner_username,
                role=UserRole.OWNER,
                color=self.user_colors.get(owner_id, "#FF6B6B"),
                last_activity=datetime.now()
            )
            
            # Create session
            session = CollaborationSession(
                id=session_id,
                project_id=project_id,
                content_id=content_id,
                name=session_name,
                owner_id=owner_id,
                users={owner_id: owner},
                content_version=0,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                settings={
                    "allow_editing": True,
                    "allow_comments": True,
                    "max_users": 10,
                    "auto_save_interval": 30
                }
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.active_connections[session_id] = []
            
            # Store in Redis
            if self.redis_client:
                await self._store_session_in_redis(session)
            
            logger.info(f"Created collaboration session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating collaboration session: {e}")
            raise

    async def join_session(self, session_id: str, user_id: str, 
                          username: str, role: UserRole = UserRole.EDITOR) -> bool:
        """Join collaboration session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Check if user already in session
            if user_id in session.users:
                return True
            
            # Check max users limit
            if len(session.users) >= session.settings.get("max_users", 10):
                return False
            
            # Create user
            user = CollaborationUser(
                user_id=user_id,
                username=username,
                role=role,
                color=self.user_colors.get(user_id, "#FF6B6B"),
                last_activity=datetime.now()
            )
            
            # Add user to session
            session.users[user_id] = user
            session.last_activity = datetime.now()
            
            # Broadcast user joined event
            await self._broadcast_event(session_id, CollaborationEvent(
                id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=CollaborationEventType.USER_JOINED,
                data={
                    "user": asdict(user),
                    "total_users": len(session.users)
                },
                timestamp=datetime.now(),
                version=session.content_version
            ))
            
            logger.info(f"User {user_id} joined session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining session: {e}")
            return False

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave collaboration session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Remove user from session
            if user_id in session.users:
                del session.users[user_id]
                session.last_activity = datetime.now()
                
                # Close user's WebSocket connections
                await self._close_user_connections(session_id, user_id)
                
                # Broadcast user left event
                await self._broadcast_event(session_id, CollaborationEvent(
                    id=str(uuid4()),
                    session_id=session_id,
                    user_id=user_id,
                    event_type=CollaborationEventType.USER_LEFT,
                    data={
                        "user_id": user_id,
                        "total_users": len(session.users)
                    },
                    timestamp=datetime.now(),
                    version=session.content_version
                ))
                
                # Clean up empty session
                if not session.users:
                    await self._cleanup_session(session_id)
                
                logger.info(f"User {user_id} left session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error leaving session: {e}")
            return False

    async def handle_websocket_connection(self, websocket: WebSocket, session_id: str, user_id: str):
        """Handle WebSocket connection for real-time collaboration"""
        await websocket.accept()
        
        try:
            # Add connection to session
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            self.active_connections[session_id].append(websocket)
            
            # Send current session state
            await self._send_session_state(websocket, session_id)
            
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_collaboration_message(session_id, user_id, message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid message format"
                    }))
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            # Remove connection
            if session_id in self.active_connections:
                self.active_connections[session_id] = [
                    conn for conn in self.active_connections[session_id] if conn != websocket
                ]

    async def _handle_collaboration_message(self, session_id: str, user_id: str, message: Dict[str, Any]):
        """Handle collaboration message"""
        try:
            event_type = message.get("type")
            data = message.get("data", {})
            
            session = self.active_sessions.get(session_id)
            if not session or user_id not in session.users:
                return
            
            # Update user activity
            session.users[user_id].last_activity = datetime.now()
            session.last_activity = datetime.now()
            
            # Handle different event types
            if event_type == "cursor_update":
                await self._handle_cursor_update(session_id, user_id, data)
            elif event_type == "content_edit":
                await self._handle_content_edit(session_id, user_id, data)
            elif event_type == "selection_change":
                await self._handle_selection_change(session_id, user_id, data)
            elif event_type == "comment_added":
                await self._handle_comment_added(session_id, user_id, data)
            elif event_type == "typing_start":
                await self._handle_typing_start(session_id, user_id, data)
            elif event_type == "typing_stop":
                await self._handle_typing_stop(session_id, user_id, data)
            elif event_type == "ping":
                await self._handle_ping(session_id, user_id, data)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error handling collaboration message: {e}")

    async def _handle_cursor_update(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle cursor position update"""
        session = self.active_sessions.get(session_id)
        if session and user_id in session.users:
            session.users[user_id].cursor_position = data.get("position")
            
            # Broadcast to other users
            await self._broadcast_event(session_id, CollaborationEvent(
                id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=CollaborationEventType.CURSOR_UPDATE,
                data=data,
                timestamp=datetime.now(),
                version=session.content_version
            ), exclude_user=user_id)

    async def _handle_content_edit(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle content edit"""
        session = self.active_sessions.get(session_id)
        if session and user_id in session.users:
            # Check if user has edit permissions
            user_role = session.users[user_id].role
            if user_role in [UserRole.OWNER, UserRole.EDITOR]:
                # Increment version
                session.content_version += 1
                
                # Broadcast edit to all users
                await self._broadcast_event(session_id, CollaborationEvent(
                    id=str(uuid4()),
                    session_id=session_id,
                    user_id=user_id,
                    event_type=CollaborationEventType.CONTENT_EDIT,
                    data={
                        **data,
                        "version": session.content_version
                    },
                    timestamp=datetime.now(),
                    version=session.content_version
                ))
                
                # Store in Redis
                if self.redis_client:
                    await self._store_content_version(session_id, session.content_version, data)

    async def _handle_selection_change(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle text selection change"""
        session = self.active_sessions.get(session_id)
        if session and user_id in session.users:
            session.users[user_id].selection = data.get("selection")
            
            # Broadcast to other users
            await self._broadcast_event(session_id, CollaborationEvent(
                id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=CollaborationEventType.SELECTION_CHANGE,
                data=data,
                timestamp=datetime.now(),
                version=session.content_version
            ), exclude_user=user_id)

    async def _handle_comment_added(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle comment addition"""
        session = self.active_sessions.get(session_id)
        if session and user_id in session.users:
            # Check if user has comment permissions
            user_role = session.users[user_id].role
            if user_role in [UserRole.OWNER, UserRole.EDITOR, UserRole.COMMENTOR]:
                # Broadcast comment to all users
                await self._broadcast_event(session_id, CollaborationEvent(
                    id=str(uuid4()),
                    session_id=session_id,
                    user_id=user_id,
                    event_type=CollaborationEventType.COMMENT_ADDED,
                    data=data,
                    timestamp=datetime.now(),
                    version=session.content_version
                ))

    async def _handle_typing_start(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle typing start"""
        session = self.active_sessions.get(session_id)
        if session and user_id in session.users:
            session.users[user_id].is_typing = True
            
            # Broadcast to other users
            await self._broadcast_event(session_id, CollaborationEvent(
                id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=CollaborationEventType.TYPING_START,
                data=data,
                timestamp=datetime.now(),
                version=session.content_version
            ), exclude_user=user_id)

    async def _handle_typing_stop(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle typing stop"""
        session = self.active_sessions.get(session_id)
        if session and user_id in session.users:
            session.users[user_id].is_typing = False
            
            # Broadcast to other users
            await self._broadcast_event(session_id, CollaborationEvent(
                id=str(uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=CollaborationEventType.TYPING_STOP,
                data=data,
                timestamp=datetime.now(),
                version=session.content_version
            ), exclude_user=user_id)

    async def _handle_ping(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle ping message"""
        # Send pong response
        await self._send_to_user(session_id, user_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })

    async def _broadcast_event(self, session_id: str, event: CollaborationEvent, 
                              exclude_user: Optional[str] = None):
        """Broadcast event to all users in session"""
        try:
            if session_id in self.active_connections:
                message = {
                    "type": event.event_type.value,
                    "data": event.data,
                    "user_id": event.user_id,
                    "timestamp": event.timestamp.isoformat(),
                    "version": event.version
                }
                
                for websocket in self.active_connections[session_id]:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        # Remove failed connection
                        self.active_connections[session_id] = [
                            conn for conn in self.active_connections[session_id] if conn != websocket
                        ]
                        
        except Exception as e:
            logger.error(f"Error broadcasting event: {e}")

    async def _send_to_user(self, session_id: str, user_id: str, message: Dict[str, Any]):
        """Send message to specific user"""
        try:
            if session_id in self.active_connections:
                for websocket in self.active_connections[session_id]:
                    try:
                        await websocket.send_text(json.dumps(message))
                        break  # Send to first available connection
                    except Exception as e:
                        logger.error(f"Error sending message to user: {e}")
                        
        except Exception as e:
            logger.error(f"Error sending message to user: {e}")

    async def _send_session_state(self, websocket: WebSocket, session_id: str):
        """Send current session state to new connection"""
        try:
            session = self.active_sessions.get(session_id)
            if session:
                state = {
                    "type": "session_state",
                    "data": {
                        "session_id": session.id,
                        "users": {uid: asdict(user) for uid, user in session.users.items()},
                        "content_version": session.content_version,
                        "settings": session.settings
                    }
                }
                await websocket.send_text(json.dumps(state))
        except Exception as e:
            logger.error(f"Error sending session state: {e}")

    async def _close_user_connections(self, session_id: str, user_id: str):
        """Close all connections for a user"""
        try:
            if session_id in self.active_connections:
                connections_to_close = []
                for websocket in self.active_connections[session_id]:
                    try:
                        await websocket.close()
                        connections_to_close.append(websocket)
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
                
                # Remove closed connections
                self.active_connections[session_id] = [
                    conn for conn in self.active_connections[session_id] 
                    if conn not in connections_to_close
                ]
        except Exception as e:
            logger.error(f"Error closing user connections: {e}")

    async def _cleanup_session(self, session_id: str):
        """Clean up empty session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            
            # Remove from Redis
            if self.redis_client:
                await self._remove_session_from_redis(session_id)
            
            logger.info(f"Cleaned up session: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

    async def _store_session_in_redis(self, session: CollaborationSession):
        """Store session in Redis"""
        try:
            if self.redis_client:
                key = f"collaboration:session:{session.id}"
                data = {
                    "session_id": session.id,
                    "project_id": session.project_id,
                    "content_id": session.content_id,
                    "name": session.name,
                    "owner_id": session.owner_id,
                    "content_version": session.content_version,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "settings": json.dumps(session.settings)
                }
                self.redis_client.hset(key, mapping=data)
                self.redis_client.expire(key, 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Error storing session in Redis: {e}")

    async def _remove_session_from_redis(self, session_id: str):
        """Remove session from Redis"""
        try:
            if self.redis_client:
                key = f"collaboration:session:{session_id}"
                self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error removing session from Redis: {e}")

    async def _store_content_version(self, session_id: str, version: int, data: Dict[str, Any]):
        """Store content version in Redis"""
        try:
            if self.redis_client:
                key = f"collaboration:content:{session_id}:{version}"
                self.redis_client.set(key, json.dumps(data), ex=86400)  # 24 hours
        except Exception as e:
            logger.error(f"Error storing content version: {e}")

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session"""
        return self.active_sessions.get(session_id)

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)

    def get_total_connections_count(self) -> int:
        """Get total count of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())

    async def cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=2)
            inactive_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.last_activity < cutoff_time:
                    inactive_sessions.append(session_id)
            
            for session_id in inactive_sessions:
                await self._cleanup_session(session_id)
            
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        except Exception as e:
            logger.error(f"Error cleaning up inactive sessions: {e}")



























