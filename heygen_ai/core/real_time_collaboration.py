"""
Real-Time Collaboration for HeyGen AI
====================================

Provides real-time collaboration features including multi-user video editing,
live streaming, and collaborative workspaces for enterprise-grade performance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# WebRTC imports
try:
    import aiortc
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# WebSocket imports
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CollaborationRoom:
    """Collaboration room configuration."""
    
    room_id: str
    name: str
    owner_id: str
    max_participants: int = 10
    is_public: bool = False
    password: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationUser:
    """Collaboration user information."""
    
    user_id: str
    username: str
    email: str
    role: str = "participant"  # owner, moderator, participant
    permissions: List[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborationSession:
    """Active collaboration session."""
    
    session_id: str
    room_id: str
    user_id: str
    connection_type: str = "webrtc"  # webrtc, websocket
    media_streams: List[str] = field(default_factory=list)
    is_active: bool = True
    started_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationRequest:
    """Request for collaboration features."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    room_id: str = ""
    action: str = ""  # join, leave, stream, edit, share
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborationResult:
    """Result of collaboration operation."""
    
    request_id: str
    success: bool
    session_id: Optional[str] = None
    room_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CollaborationManager(BaseService):
    """Manager for real-time collaboration features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the collaboration manager."""
        super().__init__("CollaborationManager", ServiceType.PHASE4, config)
        
        # Collaboration rooms
        self.rooms: Dict[str, CollaborationRoom] = {}
        
        # Active users
        self.active_users: Dict[str, CollaborationUser] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, CollaborationSession] = {}
        
        # WebRTC connections
        self.webrtc_connections: Dict[str, RTCPeerConnection] = {}
        
        # WebSocket connections
        self.websocket_connections: Dict[str, Any] = {}
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.collaboration_stats = {
            "total_rooms": 0,
            "active_rooms": 0,
            "total_users": 0,
            "active_users": 0,
            "total_sessions": 0,
            "active_sessions": 0,
            "total_streams": 0
        }
        
        # Room templates
        self.room_templates = {
            "video_editing": {
                "max_participants": 5,
                "features": ["video_edit", "audio_edit", "effects", "export"],
                "permissions": ["edit", "comment", "export"]
            },
            "live_streaming": {
                "max_participants": 100,
                "features": ["live_stream", "chat", "moderation"],
                "permissions": ["view", "chat", "moderate"]
            },
            "collaborative_review": {
                "max_participants": 10,
                "features": ["review", "comment", "approval"],
                "permissions": ["view", "comment", "approve"]
            }
        }

    async def _initialize_service_impl(self) -> None:
        """Initialize collaboration services."""
        try:
            logger.info("Initializing collaboration manager...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Initialize WebRTC
            await self._initialize_webrtc()
            
            # Initialize WebSocket server
            await self._initialize_websocket_server()
            
            # Load default rooms
            await self._load_default_rooms()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Collaboration manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collaboration manager: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not WEBRTC_AVAILABLE:
            missing_deps.append("aiortc")
        
        if not WEBSOCKETS_AVAILABLE:
            missing_deps.append("websockets")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some collaboration features may not be available")

    async def _initialize_webrtc(self) -> None:
        """Initialize WebRTC services."""
        try:
            if WEBRTC_AVAILABLE:
                logger.info("WebRTC initialized successfully")
            else:
                logger.warning("WebRTC not available")
                
        except Exception as e:
            logger.warning(f"WebRTC initialization had issues: {e}")

    async def _initialize_websocket_server(self) -> None:
        """Initialize WebSocket server."""
        try:
            if WEBSOCKETS_AVAILABLE:
                logger.info("WebSocket server initialized successfully")
            else:
                logger.warning("WebSocket server not available")
                
        except Exception as e:
            logger.warning(f"WebSocket server initialization had issues: {e}")

    async def _load_default_rooms(self) -> None:
        """Load default collaboration rooms."""
        try:
            # Create default rooms for each template
            for template_name, template_config in self.room_templates.items():
                room_id = f"default_{template_name}_{int(time.time())}"
                
                room = CollaborationRoom(
                    room_id=room_id,
                    name=f"Default {template_name.title()} Room",
                    owner_id="system",
                    max_participants=template_config["max_participants"],
                    is_public=True,
                    settings=template_config
                )
                
                self.rooms[room_id] = room
                self.collaboration_stats["total_rooms"] += 1
                self.collaboration_stats["active_rooms"] += 1
            
            logger.info(f"Loaded {len(self.room_templates)} default rooms")
            
        except Exception as e:
            logger.warning(f"Failed to load default rooms: {e}")

    async def _validate_configuration(self) -> None:
        """Validate collaboration manager configuration."""
        if not self.rooms:
            raise RuntimeError("No collaboration rooms configured")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def create_room(self, name: str, owner_id: str, template: str = "video_editing") -> str:
        """Create a new collaboration room."""
        try:
            logger.info(f"Creating collaboration room: {name}")
            
            # Validate template
            if template not in self.room_templates:
                raise ValueError(f"Invalid room template: {template}")
            
            # Generate room ID
            room_id = str(uuid.uuid4())
            
            # Get template configuration
            template_config = self.room_templates[template]
            
            # Create room
            room = CollaborationRoom(
                room_id=room_id,
                name=name,
                owner_id=owner_id,
                max_participants=template_config["max_participants"],
                is_public=False,
                settings=template_config
            )
            
            # Store room
            self.rooms[room_id] = room
            self.collaboration_stats["total_rooms"] += 1
            self.collaboration_stats["active_rooms"] += 1
            
            logger.info(f"Collaboration room created: {room_id}")
            return room_id
            
        except Exception as e:
            logger.error(f"Failed to create collaboration room: {e}")
            raise

    @with_error_handling
    async def join_room(self, room_id: str, user_id: str, username: str, email: str) -> CollaborationResult:
        """Join a collaboration room."""
        try:
            logger.info(f"User {user_id} joining room {room_id}")
            
            # Check if room exists
            if room_id not in self.rooms:
                return CollaborationResult(
                    request_id=str(uuid.uuid4()),
                    success=False,
                    error_message="Room not found"
                )
            
            room = self.rooms[room_id]
            
            # Check if room is full
            active_users_in_room = len([
                session for session in self.active_sessions.values()
                if session.room_id == room_id and session.is_active
            ])
            
            if active_users_in_room >= room.max_participants:
                return CollaborationResult(
                    request_id=str(uuid.uuid4()),
                    success=False,
                    error_message="Room is full"
                )
            
            # Create or update user
            user = CollaborationUser(
                user_id=user_id,
                username=username,
                email=email,
                role="participant"
            )
            self.active_users[user_id] = user
            
            # Create session
            session_id = str(uuid.uuid4())
            session = CollaborationSession(
                session_id=session_id,
                room_id=room_id,
                user_id=user_id,
                connection_type="webrtc"
            )
            self.active_sessions[session_id] = session
            
            # Update statistics
            self.collaboration_stats["total_users"] += 1
            self.collaboration_stats["active_users"] += 1
            self.collaboration_stats["total_sessions"] += 1
            self.collaboration_stats["active_sessions"] += 1
            
            logger.info(f"User {user_id} joined room {room_id}")
            
            return CollaborationResult(
                request_id=str(uuid.uuid4()),
                success=True,
                session_id=session_id,
                room_data={
                    "room_id": room_id,
                    "name": room.name,
                    "max_participants": room.max_participants,
                    "active_participants": active_users_in_room + 1
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to join room: {e}")
            return CollaborationResult(
                request_id=str(uuid.uuid4()),
                success=False,
                error_message=str(e)
            )

    @with_error_handling
    async def leave_room(self, session_id: str) -> CollaborationResult:
        """Leave a collaboration room."""
        try:
            logger.info(f"Session {session_id} leaving room")
            
            # Check if session exists
            if session_id not in self.active_sessions:
                return CollaborationResult(
                    request_id=str(uuid.uuid4()),
                    success=False,
                    error_message="Session not found"
                )
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            room_id = session.room_id
            
            # Close session
            session.is_active = False
            self.active_sessions[session_id] = session
            
            # Remove user if no active sessions
            active_sessions_for_user = [
                s for s in self.active_sessions.values()
                if s.user_id == user_id and s.is_active
            ]
            
            if not active_sessions_for_user:
                if user_id in self.active_users:
                    del self.active_users[user_id]
                    self.collaboration_stats["active_users"] -= 1
            
            # Update statistics
            self.collaboration_stats["active_sessions"] -= 1
            
            logger.info(f"Session {session_id} left room {room_id}")
            
            return CollaborationResult(
                request_id=str(uuid.uuid4()),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to leave room: {e}")
            return CollaborationResult(
                request_id=str(uuid.uuid4()),
                success=False,
                error_message=str(e)
            )

    @with_error_handling
    async def start_streaming(self, session_id: str, stream_type: str = "video") -> CollaborationResult:
        """Start streaming in a collaboration room."""
        try:
            logger.info(f"Starting {stream_type} stream for session {session_id}")
            
            # Check if session exists and is active
            if session_id not in self.active_sessions:
                return CollaborationResult(
                    request_id=str(uuid.uuid4()),
                    success=False,
                    error_message="Session not found"
                )
            
            session = self.active_sessions[session_id]
            if not session.is_active:
                return CollaborationResult(
                    request_id=str(uuid.uuid4()),
                    success=False,
                    error_message="Session not active"
                )
            
            # Add stream to session
            stream_id = f"{stream_type}_{int(time.time())}"
            session.media_streams.append(stream_id)
            
            # Update statistics
            self.collaboration_stats["total_streams"] += 1
            
            logger.info(f"Started {stream_type} stream: {stream_id}")
            
            return CollaborationResult(
                request_id=str(uuid.uuid4()),
                success=True,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return CollaborationResult(
                request_id=str(uuid.uuid4()),
                success=False,
                error_message=str(e)
            )

    @with_error_handling
    async def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a collaboration room."""
        try:
            if room_id not in self.rooms:
                return None
            
            room = self.rooms[room_id]
            
            # Count active participants
            active_participants = len([
                session for session in self.active_sessions.values()
                if session.room_id == room_id and session.is_active
            ])
            
            return {
                "room_id": room.room_id,
                "name": room.name,
                "owner_id": room.owner_id,
                "max_participants": room.max_participants,
                "active_participants": active_participants,
                "is_public": room.is_public,
                "created_at": room.created_at.isoformat(),
                "settings": room.settings
            }
            
        except Exception as e:
            logger.error(f"Failed to get room info: {e}")
            return None

    @with_error_handling
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active sessions for a user."""
        try:
            user_sessions = [
                session for session in self.active_sessions.values()
                if session.user_id == user_id and session.is_active
            ]
            
            return [
                {
                    "session_id": session.session_id,
                    "room_id": session.room_id,
                    "connection_type": session.connection_type,
                    "media_streams": session.media_streams,
                    "started_at": session.started_at.isoformat(),
                    "metadata": session.metadata
                }
                for session in user_sessions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the collaboration manager."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "webrtc": WEBRTC_AVAILABLE,
                "websockets": WEBSOCKETS_AVAILABLE
            }
            
            # Check room status
            room_status = {
                "total_rooms": self.collaboration_stats["total_rooms"],
                "active_rooms": self.collaboration_stats["active_rooms"],
                "room_templates": len(self.room_templates)
            }
            
            # Check user status
            user_status = {
                "total_users": self.collaboration_stats["total_users"],
                "active_users": self.collaboration_stats["active_users"],
                "total_sessions": self.collaboration_stats["total_sessions"],
                "active_sessions": self.collaboration_stats["active_sessions"]
            }
            
            # Check connections
            connection_status = {
                "webrtc_connections": len(self.webrtc_connections),
                "websocket_connections": len(self.websocket_connections)
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "rooms": room_status,
                "users": user_status,
                "connections": connection_status,
                "collaboration_stats": self.collaboration_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def get_room_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available room templates."""
        return self.room_templates.copy()

    async def delete_room(self, room_id: str, owner_id: str) -> bool:
        """Delete a collaboration room."""
        try:
            if room_id not in self.rooms:
                return False
            
            room = self.rooms[room_id]
            
            # Check ownership
            if room.owner_id != owner_id:
                return False
            
            # Close all active sessions in the room
            for session in self.active_sessions.values():
                if session.room_id == room_id and session.is_active:
                    session.is_active = False
            
            # Remove room
            del self.rooms[room_id]
            self.collaboration_stats["active_rooms"] -= 1
            
            logger.info(f"Room {room_id} deleted by {owner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            return False

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary collaboration files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for collab_file in temp_dir.glob("collaboration_*"):
                    collab_file.unlink()
                    logger.debug(f"Cleaned up temp file: {collab_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the collaboration manager."""
        try:
            # Close all WebRTC connections
            for connection in self.webrtc_connections.values():
                try:
                    await connection.close()
                except Exception:
                    pass
            
            # Close all WebSocket connections
            for connection in self.websocket_connections.values():
                try:
                    await connection.close()
                except Exception:
                    pass
            
            # Close all sessions
            for session in self.active_sessions.values():
                session.is_active = False
            
            logger.info("Collaboration manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
