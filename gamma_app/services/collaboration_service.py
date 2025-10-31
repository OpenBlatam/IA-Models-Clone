"""
Gamma App - Collaboration Service
Real-time collaboration features for content creation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from uuid import uuid4
import redis
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..api.models import CollaborationSession, CollaborationMessage, CollaborationEvent, SessionStatus

logger = logging.getLogger(__name__)

Base = declarative_base()

class CollaborationSessionDB(Base):
    """Database model for collaboration sessions"""
    __tablename__ = "collaboration_sessions"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    session_name = Column(String, nullable=False)
    creator_id = Column(String, nullable=False)
    participants = Column(JSON, default=list)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON, default=dict)

class CollaborationMessageDB(Base):
    """Database model for collaboration messages"""
    __tablename__ = "collaboration_messages"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    message_type = Column(String, nullable=False)
    content = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

@dataclass
class ActiveConnection:
    """Active WebSocket connection"""
    websocket: WebSocket
    user_id: str
    session_id: str
    connected_at: datetime
    last_ping: datetime

class CollaborationService:
    """
    Real-time collaboration service
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize collaboration service"""
        self.config = config or {}
        self.active_connections: Dict[str, List[ActiveConnection]] = {}
        self.user_sessions: Dict[str, Set[str]] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Redis for real-time features
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize database
        self._init_database()
        
        logger.info("Collaboration Service initialized successfully")

    def _init_database(self):
        """Initialize database connection"""
        try:
            database_url = self.config.get('database_url', 'sqlite:///collaboration.db')
            self.engine = create_engine(database_url)
            Base.metadata.create_all(self.engine)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.db_session = SessionLocal()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.db_session = None

    async def create_session(self, project_id: str, session_name: str, 
                           creator_id: str) -> CollaborationSession:
        """Create new collaboration session"""
        try:
            session_id = str(uuid4())
            
            session = CollaborationSession(
                id=session_id,
                project_id=project_id,
                session_name=session_name,
                creator_id=creator_id,
                participants=[creator_id],
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                settings={}
            )
            
            # Save to database
            if self.db_session:
                db_session = CollaborationSessionDB(
                    id=session_id,
                    project_id=project_id,
                    session_name=session_name,
                    creator_id=creator_id,
                    participants=[creator_id],
                    status="active",
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    settings={}
                )
                self.db_session.add(db_session)
                self.db_session.commit()
            
            # Initialize session data
            self.session_data[session_id] = {
                'content': {},
                'cursors': {},
                'selections': {},
                'comments': [],
                'version': 0
            }
            
            # Add creator to user sessions
            if creator_id not in self.user_sessions:
                self.user_sessions[creator_id] = set()
            self.user_sessions[creator_id].add(session_id)
            
            logger.info(f"Created collaboration session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating collaboration session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session by ID"""
        try:
            if self.db_session:
                db_session = self.db_session.query(CollaborationSessionDB).filter(
                    CollaborationSessionDB.id == session_id
                ).first()
                
                if db_session:
                    return CollaborationSession(
                        id=db_session.id,
                        project_id=db_session.project_id,
                        session_name=db_session.session_name,
                        creator_id=db_session.creator_id,
                        participants=db_session.participants or [],
                        status=SessionStatus(db_session.status),
                        created_at=db_session.created_at,
                        last_activity=db_session.last_activity,
                        settings=db_session.settings or {}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting collaboration session: {e}")
            return None

    async def join_session(self, session_id: str, user_id: str) -> bool:
        """Join collaboration session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Add user to participants if not already present
            if user_id not in session.participants:
                session.participants.append(user_id)
                
                # Update database
                if self.db_session:
                    db_session = self.db_session.query(CollaborationSessionDB).filter(
                        CollaborationSessionDB.id == session_id
                    ).first()
                    if db_session:
                        db_session.participants = session.participants
                        db_session.last_activity = datetime.now()
                        self.db_session.commit()
            
            # Add to user sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # Broadcast join event
            await self._broadcast_to_session(session_id, {
                'type': 'user_joined',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"User {user_id} joined session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining session: {e}")
            return False

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave collaboration session"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Remove user from participants
            if user_id in session.participants:
                session.participants.remove(user_id)
                
                # Update database
                if self.db_session:
                    db_session = self.db_session.query(CollaborationSessionDB).filter(
                        CollaborationSessionDB.id == session_id
                    ).first()
                    if db_session:
                        db_session.participants = session.participants
                        db_session.last_activity = datetime.now()
                        self.db_session.commit()
            
            # Remove from user sessions
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
            
            # Close WebSocket connections for this user in this session
            await self._close_user_connections(session_id, user_id)
            
            # Broadcast leave event
            await self._broadcast_to_session(session_id, {
                'type': 'user_left',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"User {user_id} left session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error leaving session: {e}")
            return False

    async def get_session_participants(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session participants with their status"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return []
            
            participants = []
            for user_id in session.participants:
                is_online = await self._is_user_online_in_session(session_id, user_id)
                participants.append({
                    'user_id': user_id,
                    'is_online': is_online,
                    'last_seen': datetime.now().isoformat()  # Would be actual last seen
                })
            
            return participants
            
        except Exception as e:
            logger.error(f"Error getting session participants: {e}")
            return []

    async def handle_websocket_connection(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection for real-time collaboration"""
        await websocket.accept()
        
        try:
            # Get user ID from connection (would be from authentication)
            user_id = await self._get_user_from_websocket(websocket)
            if not user_id:
                await websocket.close(code=1008, reason="Authentication required")
                return
            
            # Join session
            success = await self.join_session(session_id, user_id)
            if not success:
                await websocket.close(code=1008, reason="Session not found")
                return
            
            # Create active connection
            connection = ActiveConnection(
                websocket=websocket,
                user_id=user_id,
                session_id=session_id,
                connected_at=datetime.now(),
                last_ping=datetime.now()
            )
            
            # Add to active connections
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            self.active_connections[session_id].append(connection)
            
            # Send initial session state
            await self._send_session_state(websocket, session_id)
            
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_websocket_message(connection, message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Invalid message format'
                    }))
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            # Clean up connection
            await self._remove_connection(connection)
            await self.leave_session(session_id, user_id)

    async def _handle_websocket_message(self, connection: ActiveConnection, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get('type')
            data = message.get('data', {})
            
            if message_type == 'ping':
                connection.last_ping = datetime.now()
                await connection.websocket.send_text(json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
            
            elif message_type == 'cursor_update':
                await self._handle_cursor_update(connection, data)
            
            elif message_type == 'content_edit':
                await self._handle_content_edit(connection, data)
            
            elif message_type == 'selection':
                await self._handle_selection(connection, data)
            
            elif message_type == 'comment':
                await self._handle_comment(connection, data)
            
            elif message_type == 'typing':
                await self._handle_typing(connection, data)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _handle_cursor_update(self, connection: ActiveConnection, data: Dict[str, Any]):
        """Handle cursor position update"""
        session_id = connection.session_id
        user_id = connection.user_id
        
        # Update cursor position in session data
        if session_id in self.session_data:
            self.session_data[session_id]['cursors'][user_id] = {
                'position': data.get('position'),
                'timestamp': datetime.now().isoformat()
            }
        
        # Broadcast to other participants
        await self._broadcast_to_session(session_id, {
            'type': 'cursor_update',
            'user_id': user_id,
            'data': data
        }, exclude_user=user_id)

    async def _handle_content_edit(self, connection: ActiveConnection, data: Dict[str, Any]):
        """Handle content edit"""
        session_id = connection.session_id
        user_id = connection.user_id
        
        # Update content in session data
        if session_id in self.session_data:
            self.session_data[session_id]['content'].update(data.get('changes', {}))
            self.session_data[session_id]['version'] += 1
        
        # Save to database
        await self._save_collaboration_message(session_id, user_id, 'edit', data)
        
        # Broadcast to all participants
        await self._broadcast_to_session(session_id, {
            'type': 'content_edit',
            'user_id': user_id,
            'data': data,
            'version': self.session_data[session_id]['version']
        })

    async def _handle_selection(self, connection: ActiveConnection, data: Dict[str, Any]):
        """Handle text selection"""
        session_id = connection.session_id
        user_id = connection.user_id
        
        # Update selection in session data
        if session_id in self.session_data:
            self.session_data[session_id]['selections'][user_id] = {
                'selection': data.get('selection'),
                'timestamp': datetime.now().isoformat()
            }
        
        # Broadcast to other participants
        await self._broadcast_to_session(session_id, {
            'type': 'selection',
            'user_id': user_id,
            'data': data
        }, exclude_user=user_id)

    async def _handle_comment(self, connection: ActiveConnection, data: Dict[str, Any]):
        """Handle comment addition"""
        session_id = connection.session_id
        user_id = connection.user_id
        
        # Add comment to session data
        if session_id in self.session_data:
            comment = {
                'id': str(uuid4()),
                'user_id': user_id,
                'content': data.get('content'),
                'position': data.get('position'),
                'timestamp': datetime.now().isoformat()
            }
            self.session_data[session_id]['comments'].append(comment)
        
        # Save to database
        await self._save_collaboration_message(session_id, user_id, 'comment', data)
        
        # Broadcast to all participants
        await self._broadcast_to_session(session_id, {
            'type': 'comment',
            'user_id': user_id,
            'data': data
        })

    async def _handle_typing(self, connection: ActiveConnection, data: Dict[str, Any]):
        """Handle typing indicator"""
        session_id = connection.session_id
        user_id = connection.user_id
        
        # Broadcast typing indicator to other participants
        await self._broadcast_to_session(session_id, {
            'type': 'typing',
            'user_id': user_id,
            'data': data
        }, exclude_user=user_id)

    async def _send_session_state(self, websocket: WebSocket, session_id: str):
        """Send current session state to new connection"""
        try:
            if session_id in self.session_data:
                state = {
                    'type': 'session_state',
                    'data': {
                        'content': self.session_data[session_id]['content'],
                        'cursors': self.session_data[session_id]['cursors'],
                        'selections': self.session_data[session_id]['selections'],
                        'comments': self.session_data[session_id]['comments'],
                        'version': self.session_data[session_id]['version']
                    }
                }
                await websocket.send_text(json.dumps(state))
        except Exception as e:
            logger.error(f"Error sending session state: {e}")

    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], 
                                  exclude_user: Optional[str] = None):
        """Broadcast message to all participants in session"""
        try:
            if session_id in self.active_connections:
                for connection in self.active_connections[session_id]:
                    if exclude_user and connection.user_id == exclude_user:
                        continue
                    
                    try:
                        await connection.websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error sending message to {connection.user_id}: {e}")
                        # Remove failed connection
                        await self._remove_connection(connection)
        except Exception as e:
            logger.error(f"Error broadcasting to session: {e}")

    async def _remove_connection(self, connection: ActiveConnection):
        """Remove connection from active connections"""
        try:
            session_id = connection.session_id
            if session_id in self.active_connections:
                self.active_connections[session_id] = [
                    conn for conn in self.active_connections[session_id]
                    if conn != connection
                ]
                
                # Clean up empty session
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
        except Exception as e:
            logger.error(f"Error removing connection: {e}")

    async def _close_user_connections(self, session_id: str, user_id: str):
        """Close all connections for a user in a session"""
        try:
            if session_id in self.active_connections:
                connections_to_close = [
                    conn for conn in self.active_connections[session_id]
                    if conn.user_id == user_id
                ]
                
                for connection in connections_to_close:
                    try:
                        await connection.websocket.close()
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
                    await self._remove_connection(connection)
        except Exception as e:
            logger.error(f"Error closing user connections: {e}")

    async def _is_user_online_in_session(self, session_id: str, user_id: str) -> bool:
        """Check if user is online in session"""
        try:
            if session_id in self.active_connections:
                return any(
                    conn.user_id == user_id 
                    for conn in self.active_connections[session_id]
                )
            return False
        except Exception as e:
            logger.error(f"Error checking user online status: {e}")
            return False

    async def _get_user_from_websocket(self, websocket: WebSocket) -> Optional[str]:
        """Get user ID from WebSocket connection"""
        # This would extract user ID from authentication token
        # For now, return a placeholder
        return "user_123"

    async def _save_collaboration_message(self, session_id: str, user_id: str, 
                                        message_type: str, content: Dict[str, Any]):
        """Save collaboration message to database"""
        try:
            if self.db_session:
                message = CollaborationMessageDB(
                    id=str(uuid4()),
                    session_id=session_id,
                    user_id=user_id,
                    message_type=message_type,
                    content=content,
                    timestamp=datetime.now()
                )
                self.db_session.add(message)
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error saving collaboration message: {e}")

    async def get_session_history(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get session collaboration history"""
        try:
            if self.db_session:
                messages = self.db_session.query(CollaborationMessageDB).filter(
                    CollaborationMessageDB.session_id == session_id
                ).order_by(CollaborationMessageDB.timestamp.desc()).limit(limit).all()
                
                return [
                    {
                        'id': msg.id,
                        'user_id': msg.user_id,
                        'message_type': msg.message_type,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat()
                    }
                    for msg in messages
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []

    async def cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            if self.db_session:
                inactive_sessions = self.db_session.query(CollaborationSessionDB).filter(
                    CollaborationSessionDB.last_activity < cutoff_time,
                    CollaborationSessionDB.status == "active"
                ).all()
                
                for session in inactive_sessions:
                    session.status = "inactive"
                
                self.db_session.commit()
                logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        except Exception as e:
            logger.error(f"Error cleaning up inactive sessions: {e}")

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_connections)

    def get_total_connections_count(self) -> int:
        """Get total count of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())



























