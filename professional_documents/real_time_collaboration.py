"""
Real-Time Collaboration Service
==============================

Advanced real-time collaboration features for document editing and review.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class CollaborationAction(str, Enum):
    """Collaboration action types."""
    JOIN = "join"
    LEAVE = "leave"
    EDIT = "edit"
    COMMENT = "comment"
    HIGHLIGHT = "highlight"
    SUGGEST = "suggest"
    APPROVE = "approve"
    REJECT = "reject"
    CURSOR_MOVE = "cursor_move"
    SELECTION = "selection"


class UserPresence(str, Enum):
    """User presence status."""
    ONLINE = "online"
    EDITING = "editing"
    VIEWING = "viewing"
    AWAY = "away"
    OFFLINE = "offline"


@dataclass
class CollaborationUser:
    """User in collaboration session."""
    user_id: str
    username: str
    email: str
    role: str
    presence: UserPresence
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    last_activity: datetime = None
    color: str = "#3498db"


@dataclass
class CollaborationEvent:
    """Collaboration event."""
    event_id: str
    document_id: str
    user_id: str
    action: CollaborationAction
    timestamp: datetime
    data: Dict[str, Any]
    version: int


@dataclass
class DocumentComment:
    """Document comment."""
    comment_id: str
    document_id: str
    user_id: str
    content: str
    position: Dict[str, Any]
    created_at: datetime
    resolved: bool = False
    replies: List['DocumentComment'] = None


class RealTimeCollaborationService:
    """Real-time collaboration service for documents."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, WebSocketServerProtocol]] = {}  # document_id -> user_id -> websocket
        self.collaboration_users: Dict[str, Dict[str, CollaborationUser]] = {}  # document_id -> user_id -> user
        self.document_versions: Dict[str, int] = {}  # document_id -> version
        self.collaboration_events: Dict[str, List[CollaborationEvent]] = {}  # document_id -> events
        self.document_comments: Dict[str, List[DocumentComment]] = {}  # document_id -> comments
        self.operational_transforms: Dict[str, List[Dict[str, Any]]] = {}  # document_id -> operations
        
    async def join_document_session(
        self,
        document_id: str,
        user_id: str,
        username: str,
        email: str,
        role: str,
        websocket: WebSocketServerProtocol
    ) -> CollaborationUser:
        """Join a document collaboration session."""
        
        try:
            # Initialize document session if not exists
            if document_id not in self.active_sessions:
                self.active_sessions[document_id] = {}
                self.collaboration_users[document_id] = {}
                self.document_versions[document_id] = 0
                self.collaboration_events[document_id] = []
                self.document_comments[document_id] = []
                self.operational_transforms[document_id] = []
            
            # Create collaboration user
            user = CollaborationUser(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                presence=UserPresence.ONLINE,
                last_activity=datetime.now(),
                color=self._get_user_color(user_id)
            )
            
            # Add user to session
            self.active_sessions[document_id][user_id] = websocket
            self.collaboration_users[document_id][user_id] = user
            
            # Create join event
            event = CollaborationEvent(
                event_id=str(uuid4()),
                document_id=document_id,
                user_id=user_id,
                action=CollaborationAction.JOIN,
                timestamp=datetime.now(),
                data={
                    "username": username,
                    "role": role,
                    "color": user.color
                },
                version=self.document_versions[document_id]
            )
            
            self.collaboration_events[document_id].append(event)
            
            # Notify other users
            await self._broadcast_to_document(document_id, {
                "type": "user_joined",
                "user": {
                    "user_id": user_id,
                    "username": username,
                    "role": role,
                    "presence": user.presence.value,
                    "color": user.color
                },
                "timestamp": event.timestamp.isoformat()
            }, exclude_user=user_id)
            
            # Send current state to new user
            await self._send_document_state(websocket, document_id, user_id)
            
            logger.info(f"User {user_id} joined document {document_id}")
            
            return user
            
        except Exception as e:
            logger.error(f"Error joining document session: {str(e)}")
            raise
    
    async def leave_document_session(self, document_id: str, user_id: str):
        """Leave a document collaboration session."""
        
        try:
            if document_id not in self.active_sessions:
                return
            
            # Remove user from session
            if user_id in self.active_sessions[document_id]:
                del self.active_sessions[document_id][user_id]
            
            if user_id in self.collaboration_users[document_id]:
                user = self.collaboration_users[document_id][user_id]
                del self.collaboration_users[document_id][user_id]
                
                # Create leave event
                event = CollaborationEvent(
                    event_id=str(uuid4()),
                    document_id=document_id,
                    user_id=user_id,
                    action=CollaborationAction.LEAVE,
                    timestamp=datetime.now(),
                    data={
                        "username": user.username,
                        "role": user.role
                    },
                    version=self.document_versions[document_id]
                )
                
                self.collaboration_events[document_id].append(event)
                
                # Notify other users
                await self._broadcast_to_document(document_id, {
                    "type": "user_left",
                    "user_id": user_id,
                    "username": user.username,
                    "timestamp": event.timestamp.isoformat()
                })
            
            # Clean up empty sessions
            if not self.active_sessions[document_id]:
                del self.active_sessions[document_id]
                del self.collaboration_users[document_id]
                del self.document_versions[document_id]
                del self.collaboration_events[document_id]
                del self.document_comments[document_id]
                del self.operational_transforms[document_id]
            
            logger.info(f"User {user_id} left document {document_id}")
            
        except Exception as e:
            logger.error(f"Error leaving document session: {str(e)}")
    
    async def handle_collaboration_event(
        self,
        document_id: str,
        user_id: str,
        action: CollaborationAction,
        data: Dict[str, Any]
    ):
        """Handle collaboration event."""
        
        try:
            if document_id not in self.collaboration_users:
                return
            
            if user_id not in self.collaboration_users[document_id]:
                return
            
            # Update user activity
            user = self.collaboration_users[document_id][user_id]
            user.last_activity = datetime.now()
            
            # Handle different actions
            if action == CollaborationAction.EDIT:
                await self._handle_edit_action(document_id, user_id, data)
            elif action == CollaborationAction.COMMENT:
                await self._handle_comment_action(document_id, user_id, data)
            elif action == CollaborationAction.HIGHLIGHT:
                await self._handle_highlight_action(document_id, user_id, data)
            elif action == CollaborationAction.SUGGEST:
                await self._handle_suggest_action(document_id, user_id, data)
            elif action == CollaborationAction.CURSOR_MOVE:
                await self._handle_cursor_move_action(document_id, user_id, data)
            elif action == CollaborationAction.SELECTION:
                await self._handle_selection_action(document_id, user_id, data)
            
            # Create event record
            event = CollaborationEvent(
                event_id=str(uuid4()),
                document_id=document_id,
                user_id=user_id,
                action=action,
                timestamp=datetime.now(),
                data=data,
                version=self.document_versions[document_id]
            )
            
            self.collaboration_events[document_id].append(event)
            
            # Broadcast to other users
            await self._broadcast_to_document(document_id, {
                "type": "collaboration_event",
                "action": action.value,
                "user_id": user_id,
                "username": user.username,
                "data": data,
                "timestamp": event.timestamp.isoformat()
            }, exclude_user=user_id)
            
        except Exception as e:
            logger.error(f"Error handling collaboration event: {str(e)}")
    
    async def _handle_edit_action(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle document edit action."""
        
        # Apply operational transform
        operation = data.get("operation")
        if operation:
            await self._apply_operational_transform(document_id, operation, user_id)
        
        # Update document version
        self.document_versions[document_id] += 1
        
        # Update user presence
        user = self.collaboration_users[document_id][user_id]
        user.presence = UserPresence.EDITING
    
    async def _handle_comment_action(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle comment action."""
        
        comment = DocumentComment(
            comment_id=str(uuid4()),
            document_id=document_id,
            user_id=user_id,
            content=data.get("content", ""),
            position=data.get("position", {}),
            created_at=datetime.now()
        )
        
        self.document_comments[document_id].append(comment)
        
        # Update user presence
        user = self.collaboration_users[document_id][user_id]
        user.presence = UserPresence.EDITING
    
    async def _handle_highlight_action(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle highlight action."""
        
        # Store highlight data
        highlight_data = {
            "highlight_id": str(uuid4()),
            "user_id": user_id,
            "position": data.get("position", {}),
            "color": data.get("color", "#ffff00"),
            "timestamp": datetime.now().isoformat()
        }
        
        # This would typically be stored in a database
        # For now, we'll just broadcast the highlight
    
    async def _handle_suggest_action(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle suggestion action."""
        
        suggestion_data = {
            "suggestion_id": str(uuid4()),
            "user_id": user_id,
            "original_text": data.get("original_text", ""),
            "suggested_text": data.get("suggested_text", ""),
            "position": data.get("position", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store suggestion and broadcast
    
    async def _handle_cursor_move_action(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle cursor move action."""
        
        user = self.collaboration_users[document_id][user_id]
        user.cursor_position = data.get("position")
        user.presence = UserPresence.EDITING
    
    async def _handle_selection_action(self, document_id: str, user_id: str, data: Dict[str, Any]):
        """Handle text selection action."""
        
        user = self.collaboration_users[document_id][user_id]
        user.selection = data.get("selection")
        user.presence = UserPresence.EDITING
    
    async def _apply_operational_transform(
        self,
        document_id: str,
        operation: Dict[str, Any],
        user_id: str
    ):
        """Apply operational transform for conflict resolution."""
        
        # Store operation
        self.operational_transforms[document_id].append({
            "operation": operation,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "version": self.document_versions[document_id]
        })
        
        # Apply conflict resolution logic
        # This is a simplified version - real implementation would be more complex
        await self._resolve_operation_conflicts(document_id)
    
    async def _resolve_operation_conflicts(self, document_id: str):
        """Resolve conflicts between concurrent operations."""
        
        operations = self.operational_transforms[document_id]
        
        # Simple conflict resolution - in practice, this would be more sophisticated
        if len(operations) > 1:
            # Sort by timestamp and apply in order
            operations.sort(key=lambda x: x["timestamp"])
            
            # Apply transformations to resolve conflicts
            for i in range(1, len(operations)):
                current_op = operations[i]
                previous_ops = operations[:i]
                
                # Transform current operation against previous ones
                transformed_op = await self._transform_operation(current_op, previous_ops)
                operations[i] = transformed_op
    
    async def _transform_operation(
        self,
        operation: Dict[str, Any],
        previous_operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Transform operation against previous operations."""
        
        # Simplified transformation logic
        # Real implementation would handle complex text operations
        
        current_op = operation["operation"]
        
        for prev_op in previous_operations:
            prev_operation = prev_op["operation"]
            
            # Handle different operation types
            if current_op["type"] == "insert" and prev_operation["type"] == "insert":
                # Adjust position if needed
                if prev_operation["position"] <= current_op["position"]:
                    current_op["position"] += len(prev_operation["text"])
            
            elif current_op["type"] == "delete" and prev_operation["type"] == "insert":
                # Adjust delete range if needed
                if prev_operation["position"] < current_op["position"]:
                    current_op["position"] += len(prev_operation["text"])
            
            # Add more transformation rules as needed
        
        return operation
    
    async def _broadcast_to_document(
        self,
        document_id: str,
        message: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """Broadcast message to all users in document session."""
        
        if document_id not in self.active_sessions:
            return
        
        disconnected_users = []
        
        for user_id, websocket in self.active_sessions[document_id].items():
            if exclude_user and user_id == exclude_user:
                continue
            
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_users.append(user_id)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {str(e)}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.leave_document_session(document_id, user_id)
    
    async def _send_document_state(self, websocket: WebSocketServerProtocol, document_id: str, user_id: str):
        """Send current document state to user."""
        
        try:
            # Get current users
            users = []
            if document_id in self.collaboration_users:
                for uid, user in self.collaboration_users[document_id].items():
                    if uid != user_id:  # Don't include self
                        users.append({
                            "user_id": uid,
                            "username": user.username,
                            "role": user.role,
                            "presence": user.presence.value,
                            "color": user.color,
                            "cursor_position": user.cursor_position,
                            "selection": user.selection
                        })
            
            # Get comments
            comments = []
            if document_id in self.document_comments:
                for comment in self.document_comments[document_id]:
                    comments.append({
                        "comment_id": comment.comment_id,
                        "user_id": comment.user_id,
                        "content": comment.content,
                        "position": comment.position,
                        "created_at": comment.created_at.isoformat(),
                        "resolved": comment.resolved
                    })
            
            # Send state
            state_message = {
                "type": "document_state",
                "document_id": document_id,
                "version": self.document_versions.get(document_id, 0),
                "users": users,
                "comments": comments,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(state_message))
            
        except Exception as e:
            logger.error(f"Error sending document state: {str(e)}")
    
    def _get_user_color(self, user_id: str) -> str:
        """Get unique color for user."""
        
        colors = [
            "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
            "#9b59b6", "#1abc9c", "#34495e", "#e67e22",
            "#95a5a6", "#f1c40f", "#8e44ad", "#16a085"
        ]
        
        # Simple hash-based color assignment
        color_index = hash(user_id) % len(colors)
        return colors[color_index]
    
    async def get_document_collaborators(self, document_id: str) -> List[Dict[str, Any]]:
        """Get list of collaborators for a document."""
        
        if document_id not in self.collaboration_users:
            return []
        
        collaborators = []
        for user_id, user in self.collaboration_users[document_id].items():
            collaborators.append({
                "user_id": user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "presence": user.presence.value,
                "last_activity": user.last_activity.isoformat(),
                "color": user.color
            })
        
        return collaborators
    
    async def get_document_comments(self, document_id: str) -> List[Dict[str, Any]]:
        """Get comments for a document."""
        
        if document_id not in self.document_comments:
            return []
        
        comments = []
        for comment in self.document_comments[document_id]:
            comments.append({
                "comment_id": comment.comment_id,
                "user_id": comment.user_id,
                "content": comment.content,
                "position": comment.position,
                "created_at": comment.created_at.isoformat(),
                "resolved": comment.resolved,
                "replies": [
                    {
                        "comment_id": reply.comment_id,
                        "user_id": reply.user_id,
                        "content": reply.content,
                        "created_at": reply.created_at.isoformat()
                    }
                    for reply in (comment.replies or [])
                ]
            })
        
        return comments
    
    async def resolve_comment(self, document_id: str, comment_id: str, user_id: str) -> bool:
        """Resolve a comment."""
        
        if document_id not in self.document_comments:
            return False
        
        for comment in self.document_comments[document_id]:
            if comment.comment_id == comment_id:
                comment.resolved = True
                
                # Notify other users
                await self._broadcast_to_document(document_id, {
                    "type": "comment_resolved",
                    "comment_id": comment_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                return True
        
        return False
    
    async def get_collaboration_analytics(self, document_id: str) -> Dict[str, Any]:
        """Get collaboration analytics for a document."""
        
        if document_id not in self.collaboration_events:
            return {
                "total_events": 0,
                "active_users": 0,
                "total_comments": 0,
                "collaboration_score": 0
            }
        
        events = self.collaboration_events[document_id]
        comments = self.document_comments.get(document_id, [])
        active_users = len(self.collaboration_users.get(document_id, {}))
        
        # Calculate collaboration score
        edit_events = len([e for e in events if e.action == CollaborationAction.EDIT])
        comment_events = len([e for e in events if e.action == CollaborationAction.COMMENT])
        collaboration_score = min(100, (edit_events * 2 + comment_events * 3 + active_users * 10))
        
        return {
            "total_events": len(events),
            "active_users": active_users,
            "total_comments": len(comments),
            "resolved_comments": len([c for c in comments if c.resolved]),
            "edit_events": edit_events,
            "comment_events": comment_events,
            "collaboration_score": collaboration_score,
            "last_activity": max([e.timestamp for e in events]).isoformat() if events else None
        }



























