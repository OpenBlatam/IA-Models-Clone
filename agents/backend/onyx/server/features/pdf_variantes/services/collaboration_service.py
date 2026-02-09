"""
Collaboration Service
Real-time collaboration features for PDF documents
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..models import (
    CollaborationInvite, Annotation, Feedback,
    RealTimeUpdate, WebSocketMessage, WebSocketResponse
)
from ..utils.config import Settings
from ..utils.cache_helpers import CacheManager

logger = logging.getLogger(__name__)

class CollaborationUser(BaseModel):
    """User in a collaboration session"""
    user_id: str
    username: str
    websocket: Optional[WebSocket] = None
    permissions: List[str] = []
    last_activity: datetime = datetime.utcnow()
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None

class CollaborationSession(BaseModel):
    """Collaboration session for a document"""
    document_id: str
    users: Dict[str, CollaborationUser] = {}
    annotations: List[Annotation] = []
    feedback: List[Feedback] = []
    created_at: datetime = datetime.utcnow()
    last_activity: datetime = datetime.utcnow()
    is_active: bool = True

class CollaborationService:
    """Service for real-time collaboration features"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_manager = CacheManager(settings)
        
        # Active collaboration sessions
        self.sessions: Dict[str, CollaborationSession] = {}
        
        # WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # User permissions cache
        self.user_permissions: Dict[str, Dict[str, List[str]]] = {}
    
    async def initialize(self):
        """Initialize the collaboration service"""
        try:
            await self.cache_manager.initialize()
            logger.info("Collaboration Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Collaboration Service: {e}")
            raise
    
    async def invite_collaborator(self, invite: CollaborationInvite, inviter_user_id: str) -> Dict[str, Any]:
        """Invite a collaborator to a document"""
        try:
            # Validate inviter permissions
            inviter_permissions = await self._get_user_permissions(inviter_user_id, invite.document_id)
            if "share" not in inviter_permissions:
                raise ValueError("User does not have permission to share document")
            
            # Create collaboration invite
            invite_id = str(UUID())
            invite_data = {
                "invite_id": invite_id,
                "document_id": invite.document_id,
                "invited_by": inviter_user_id,
                "invited_user": invite.invited_user,
                "permissions": invite.permissions,
                "expires_at": invite.expires_at,
                "accepted": False,
                "created_at": datetime.utcnow()
            }
            
            # Save invite
            await self.cache_manager.set(f"invite:{invite_id}", invite_data)
            
            # Send notification to invited user
            await self._send_notification(
                invite.invited_user,
                "collaboration_invite",
                {
                    "invite_id": invite_id,
                    "document_id": invite.document_id,
                    "invited_by": inviter_user_id,
                    "permissions": invite.permissions
                }
            )
            
            return {
                "success": True,
                "invite_id": invite_id,
                "message": "Collaboration invite sent successfully"
            }
            
        except Exception as e:
            logger.error(f"Error inviting collaborator: {e}")
            return {
                "success": False,
                "message": f"Failed to invite collaborator: {str(e)}"
            }
    
    async def connect_user(self, websocket: WebSocket, document_id: str, user_id: str):
        """Connect a user to a collaboration session"""
        try:
            await websocket.accept()
            
            # Get or create collaboration session
            session = await self._get_or_create_session(document_id)
            
            # Check user permissions
            permissions = await self._get_user_permissions(user_id, document_id)
            if not permissions:
                await websocket.close(code=1008, reason="No access to document")
                return
            
            # Create collaboration user
            user = CollaborationUser(
                user_id=user_id,
                username=await self._get_username(user_id),
                websocket=websocket,
                permissions=permissions,
                last_activity=datetime.utcnow()
            )
            
            # Add user to session
            session.users[user_id] = user
            self.active_connections[user_id] = websocket
            
            # Notify other users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "user_joined",
                    "user_id": user_id,
                    "username": user.username,
                    "timestamp": datetime.utcnow().isoformat()
                },
                exclude_user=user_id
            )
            
            # Send current session state to new user
            await self._send_session_state(websocket, session)
            
            logger.info(f"User {user_id} connected to document {document_id}")
            
        except Exception as e:
            logger.error(f"Error connecting user {user_id}: {e}")
            await websocket.close(code=1011, reason="Internal server error")
    
    async def disconnect_user(self, document_id: str, user_id: str):
        """Disconnect a user from a collaboration session"""
        try:
            # Remove from active connections
            if user_id in self.active_connections:
                del self.active_connections[user_id]
            
            # Remove from session
            if document_id in self.sessions:
                session = self.sessions[document_id]
                if user_id in session.users:
                    del session.users[user_id]
                    
                    # Notify other users
                    await self._broadcast_to_session(
                        document_id,
                        {
                            "type": "user_left",
                            "user_id": user_id,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        exclude_user=user_id
                    )
                    
                    # Clean up empty sessions
                    if not session.users:
                        del self.sessions[document_id]
            
            logger.info(f"User {user_id} disconnected from document {document_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting user {user_id}: {e}")
    
    async def handle_message(self, document_id: str, user_id: str, message: str):
        """Handle WebSocket message from user"""
        try:
            # Parse message
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await self._send_error(user_id, "Invalid JSON message")
                return
            
            # Validate message type
            message_type = data.get("type")
            if not message_type:
                await self._send_error(user_id, "Missing message type")
                return
            
            # Get session
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                await self._send_error(user_id, "Not in collaboration session")
                return
            
            # Update user activity
            session.users[user_id].last_activity = datetime.utcnow()
            session.last_activity = datetime.utcnow()
            
            # Handle different message types
            if message_type == "cursor_update":
                await self._handle_cursor_update(document_id, user_id, data)
            elif message_type == "selection_update":
                await self._handle_selection_update(document_id, user_id, data)
            elif message_type == "annotation_create":
                await self._handle_annotation_create(document_id, user_id, data)
            elif message_type == "annotation_update":
                await self._handle_annotation_update(document_id, user_id, data)
            elif message_type == "annotation_delete":
                await self._handle_annotation_delete(document_id, user_id, data)
            elif message_type == "feedback_create":
                await self._handle_feedback_create(document_id, user_id, data)
            elif message_type == "chat_message":
                await self._handle_chat_message(document_id, user_id, data)
            else:
                await self._send_error(user_id, f"Unknown message type: {message_type}")
            
        except Exception as e:
            logger.error(f"Error handling message from user {user_id}: {e}")
            await self._send_error(user_id, "Error processing message")
    
    async def _get_or_create_session(self, document_id: str) -> CollaborationSession:
        """Get or create a collaboration session"""
        if document_id not in self.sessions:
            self.sessions[document_id] = CollaborationSession(document_id=document_id)
        
        return self.sessions[document_id]
    
    async def _get_user_permissions(self, user_id: str, document_id: str) -> List[str]:
        """Get user permissions for a document"""
        try:
            # Check cache first
            cache_key = f"permissions:{user_id}:{document_id}"
            cached_permissions = await self.cache_manager.get(cache_key)
            if cached_permissions:
                return cached_permissions
            
            # TODO: Implement database lookup for permissions
            # For now, return default permissions
            permissions = ["view", "edit", "comment"]
            
            # Cache permissions
            await self.cache_manager.set(cache_key, permissions, ttl=3600)
            
            return permissions
            
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return []
    
    async def _get_username(self, user_id: str) -> str:
        """Get username for user ID"""
        try:
            # Check cache first
            cached_username = await self.cache_manager.get(f"username:{user_id}")
            if cached_username:
                return cached_username
            
            # TODO: Implement database lookup
            username = f"user_{user_id[:8]}"
            
            # Cache username
            await self.cache_manager.set(f"username:{user_id}", username, ttl=3600)
            
            return username
            
        except Exception as e:
            logger.error(f"Error getting username: {e}")
            return f"user_{user_id[:8]}"
    
    async def _broadcast_to_session(self, document_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast message to all users in a session"""
        try:
            session = self.sessions.get(document_id)
            if not session:
                return
            
            message_json = json.dumps(message)
            
            for user_id, user in session.users.items():
                if exclude_user and user_id == exclude_user:
                    continue
                
                if user.websocket:
                    try:
                        await user.websocket.send_text(message_json)
                    except Exception as e:
                        logger.error(f"Error sending message to user {user_id}: {e}")
                        # Remove disconnected user
                        await self.disconnect_user(document_id, user_id)
            
        except Exception as e:
            logger.error(f"Error broadcasting to session {document_id}: {e}")
    
    async def _send_session_state(self, websocket: WebSocket, session: CollaborationSession):
        """Send current session state to a user"""
        try:
            state = {
                "type": "session_state",
                "document_id": session.document_id,
                "users": [
                    {
                        "user_id": user_id,
                        "username": user.username,
                        "permissions": user.permissions,
                        "cursor_position": user.cursor_position,
                        "selection": user.selection
                    }
                    for user_id, user in session.users.items()
                ],
                "annotations": [annotation.dict() for annotation in session.annotations],
                "feedback": [feedback.dict() for feedback in session.feedback]
            }
            
            await websocket.send_text(json.dumps(state))
            
        except Exception as e:
            logger.error(f"Error sending session state: {e}")
    
    async def _send_error(self, user_id: str, error_message: str):
        """Send error message to user"""
        try:
            if user_id in self.active_connections:
                websocket = self.active_connections[user_id]
                error = {
                    "type": "error",
                    "message": error_message,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(error))
            
        except Exception as e:
            logger.error(f"Error sending error to user {user_id}: {e}")
    
    async def _send_notification(self, user_id: str, notification_type: str, data: Dict[str, Any]):
        """Send notification to user"""
        try:
            # TODO: Implement notification system
            logger.info(f"Notification for user {user_id}: {notification_type}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _handle_cursor_update(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle cursor position update"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            # Update cursor position
            session.users[user_id].cursor_position = data.get("position")
            
            # Broadcast to other users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "cursor_update",
                    "user_id": user_id,
                    "position": data.get("position"),
                    "timestamp": datetime.utcnow().isoformat()
                },
                exclude_user=user_id
            )
            
        except Exception as e:
            logger.error(f"Error handling cursor update: {e}")
    
    async def _handle_selection_update(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle text selection update"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            # Update selection
            session.users[user_id].selection = data.get("selection")
            
            # Broadcast to other users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "selection_update",
                    "user_id": user_id,
                    "selection": data.get("selection"),
                    "timestamp": datetime.utcnow().isoformat()
                },
                exclude_user=user_id
            )
            
        except Exception as e:
            logger.error(f"Error handling selection update: {e}")
    
    async def _handle_annotation_create(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle annotation creation"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            # Check permissions
            user_permissions = session.users[user_id].permissions
            if "comment" not in user_permissions:
                await self._send_error(user_id, "No permission to create annotations")
                return
            
            # Create annotation
            annotation = Annotation(
                annotation_id=str(UUID()),
                document_id=document_id,
                page_number=data.get("page_number"),
                content=data.get("content"),
                position=data.get("position", {}),
                created_by=user_id,
                annotation_type=data.get("type", "comment")
            )
            
            # Add to session
            session.annotations.append(annotation)
            
            # Broadcast to all users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "annotation_created",
                    "annotation": annotation.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling annotation create: {e}")
    
    async def _handle_annotation_update(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle annotation update"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            annotation_id = data.get("annotation_id")
            if not annotation_id:
                await self._send_error(user_id, "Missing annotation ID")
                return
            
            # Find annotation
            annotation = None
            for ann in session.annotations:
                if ann.annotation_id == annotation_id:
                    annotation = ann
                    break
            
            if not annotation:
                await self._send_error(user_id, "Annotation not found")
                return
            
            # Check permissions
            if annotation.created_by != user_id and "admin" not in session.users[user_id].permissions:
                await self._send_error(user_id, "No permission to update annotation")
                return
            
            # Update annotation
            annotation.content = data.get("content", annotation.content)
            annotation.updated_at = datetime.utcnow()
            
            # Broadcast to all users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "annotation_updated",
                    "annotation": annotation.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling annotation update: {e}")
    
    async def _handle_annotation_delete(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle annotation deletion"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            annotation_id = data.get("annotation_id")
            if not annotation_id:
                await self._send_error(user_id, "Missing annotation ID")
                return
            
            # Find and remove annotation
            annotation = None
            for i, ann in enumerate(session.annotations):
                if ann.annotation_id == annotation_id:
                    annotation = ann
                    del session.annotations[i]
                    break
            
            if not annotation:
                await self._send_error(user_id, "Annotation not found")
                return
            
            # Check permissions
            if annotation.created_by != user_id and "admin" not in session.users[user_id].permissions:
                await self._send_error(user_id, "No permission to delete annotation")
                return
            
            # Broadcast to all users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "annotation_deleted",
                    "annotation_id": annotation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling annotation delete: {e}")
    
    async def _handle_feedback_create(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle feedback creation"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            # Create feedback
            feedback = Feedback(
                feedback_id=str(UUID()),
                document_id=document_id,
                variant_id=data.get("variant_id"),
                type=data.get("type", "comment"),
                rating=data.get("rating"),
                comment=data.get("comment"),
                user_id=user_id
            )
            
            # Add to session
            session.feedback.append(feedback)
            
            # Broadcast to all users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "feedback_created",
                    "feedback": feedback.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling feedback create: {e}")
    
    async def _handle_chat_message(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle chat message"""
        try:
            session = self.sessions.get(document_id)
            if not session or user_id not in session.users:
                return
            
            # Broadcast chat message to all users
            await self._broadcast_to_session(
                document_id,
                {
                    "type": "chat_message",
                    "user_id": user_id,
                    "username": session.users[user_id].username,
                    "message": data.get("message"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
    
    async def cleanup(self):
        """Cleanup collaboration service"""
        try:
            # Close all WebSocket connections
            for user_id, websocket in self.active_connections.items():
                try:
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket for user {user_id}: {e}")
            
            # Clear sessions
            self.sessions.clear()
            self.active_connections.clear()
            
            await self.cache_manager.cleanup()
            
            logger.info("Collaboration Service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up Collaboration Service: {e}")
