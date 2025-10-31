"""
Content Collaboration Engine - Advanced Team Collaboration and Content Management
============================================================================

This module provides comprehensive content collaboration capabilities including:
- Real-time collaborative editing
- Team management and role-based access control
- Content versioning and history tracking
- Comment and review system
- Approval workflows and notifications
- Content sharing and permissions
- Project management integration
- Communication and messaging
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
from collections import defaultdict, deque
import redis
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import boto3
from google.cloud import storage
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    EDITOR = "editor"
    WRITER = "writer"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    GUEST = "guest"

class ContentStatus(Enum):
    """Content status enumeration"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REJECTED = "rejected"

class NotificationType(Enum):
    """Notification type enumeration"""
    COMMENT = "comment"
    MENTION = "mention"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    CONTENT_UPDATED = "content_updated"
    DEADLINE_REMINDER = "deadline_reminder"
    ASSIGNMENT = "assignment"

class CollaborationAction(Enum):
    """Collaboration action enumeration"""
    CREATE = "create"
    EDIT = "edit"
    DELETE = "delete"
    COMMENT = "comment"
    APPROVE = "approve"
    REJECT = "reject"
    SHARE = "share"
    ASSIGN = "assign"

@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    email: str
    display_name: str
    avatar_url: str = ""
    role: UserRole = UserRole.VIEWER
    permissions: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    is_online: bool = False

@dataclass
class Team:
    """Team data structure"""
    team_id: str
    name: str
    description: str
    owner_id: str
    members: List[str] = field(default_factory=list)
    roles: Dict[str, UserRole] = field(default_factory=dict)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentVersion:
    """Content version data structure"""
    version_id: str
    content_id: str
    version_number: int
    content: str
    author_id: str
    changes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_current: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Comment:
    """Comment data structure"""
    comment_id: str
    content_id: str
    author_id: str
    content: str
    parent_comment_id: Optional[str] = None
    mentions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_resolved: bool = False
    reactions: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class Notification:
    """Notification data structure"""
    notification_id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    content_id: Optional[str] = None
    sender_id: Optional[str] = None
    is_read: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class CollaborationSession:
    """Collaboration session data structure"""
    session_id: str
    content_id: str
    participants: List[str] = field(default_factory=list)
    active_editors: Dict[str, str] = field(default_factory=dict)  # user_id -> cursor_position
    changes: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

class ContentCollaborationEngine:
    """
    Advanced Content Collaboration Engine
    
    Provides comprehensive team collaboration and content management capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Collaboration Engine"""
        self.config = config
        self.users = {}
        self.teams = {}
        self.content_versions = {}
        self.comments = {}
        self.notifications = {}
        self.collaboration_sessions = {}
        self.websocket_connections = {}
        self.redis_client = None
        self.database_engine = None
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_notification_system()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Content Collaboration Engine initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _initialize_notification_system(self):
        """Initialize notification system"""
        try:
            # Email configuration
            self.email_config = {
                "smtp_server": self.config.get("smtp_server", "smtp.gmail.com"),
                "smtp_port": self.config.get("smtp_port", 587),
                "username": self.config.get("email_username"),
                "password": self.config.get("email_password")
            }
            
            # Slack configuration
            if self.config.get("slack_bot_token"):
                self.slack_client = WebClient(token=self.config["slack_bot_token"])
            else:
                self.slack_client = None
            
            logger.info("Notification system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing notification system: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start notification processor
            notification_thread = threading.Thread(target=self._process_notifications, daemon=True)
            notification_thread.start()
            
            # Start session cleanup
            cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
            cleanup_thread.start()
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        try:
            user_id = str(uuid.uuid4())
            
            user = User(
                user_id=user_id,
                username=user_data["username"],
                email=user_data["email"],
                display_name=user_data["display_name"],
                avatar_url=user_data.get("avatar_url", ""),
                role=UserRole(user_data.get("role", "viewer")),
                permissions=user_data.get("permissions", []),
                preferences=user_data.get("preferences", {})
            )
            
            # Store user
            self.users[user_id] = user
            
            # Store in Redis for quick access
            if self.redis_client:
                self.redis_client.setex(f"user:{user_id}", 3600, json.dumps({
                    "user_id": user_id,
                    "username": user.username,
                    "display_name": user.display_name,
                    "role": user.role.value,
                    "is_online": user.is_online
                }))
            
            logger.info(f"User {user_id} created successfully")
            
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def create_team(self, team_data: Dict[str, Any], owner_id: str) -> Team:
        """Create a new team"""
        try:
            team_id = str(uuid.uuid4())
            
            team = Team(
                team_id=team_id,
                name=team_data["name"],
                description=team_data.get("description", ""),
                owner_id=owner_id,
                members=[owner_id],
                roles={owner_id: UserRole.ADMIN},
                permissions=team_data.get("permissions", {}),
                settings=team_data.get("settings", {})
            )
            
            # Store team
            self.teams[team_id] = team
            
            logger.info(f"Team {team_id} created successfully")
            
            return team
            
        except Exception as e:
            logger.error(f"Error creating team: {e}")
            raise
    
    async def add_team_member(self, team_id: str, user_id: str, role: UserRole) -> bool:
        """Add member to team"""
        try:
            if team_id not in self.teams:
                raise ValueError(f"Team {team_id} not found")
            
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            team = self.teams[team_id]
            
            # Add member
            if user_id not in team.members:
                team.members.append(user_id)
            
            # Set role
            team.roles[user_id] = role
            
            logger.info(f"User {user_id} added to team {team_id} with role {role.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding team member: {e}")
            return False
    
    async def create_content_version(self, content_id: str, content: str, 
                                   author_id: str, changes: List[str] = None) -> ContentVersion:
        """Create a new content version"""
        try:
            # Get current version number
            existing_versions = [
                v for v in self.content_versions.values() 
                if v.content_id == content_id
            ]
            version_number = len(existing_versions) + 1
            
            # Mark previous versions as not current
            for version in existing_versions:
                version.is_current = False
            
            # Create new version
            version_id = str(uuid.uuid4())
            version = ContentVersion(
                version_id=version_id,
                content_id=content_id,
                version_number=version_number,
                content=content,
                author_id=author_id,
                changes=changes or [],
                is_current=True
            )
            
            # Store version
            self.content_versions[version_id] = version
            
            # Store in Redis for quick access
            if self.redis_client:
                self.redis_client.setex(f"content_version:{version_id}", 3600, json.dumps({
                    "version_id": version_id,
                    "content_id": content_id,
                    "version_number": version_number,
                    "author_id": author_id,
                    "created_at": version.created_at.isoformat(),
                    "is_current": version.is_current
                }))
            
            logger.info(f"Content version {version_id} created for content {content_id}")
            
            return version
            
        except Exception as e:
            logger.error(f"Error creating content version: {e}")
            raise
    
    async def get_content_history(self, content_id: str) -> List[ContentVersion]:
        """Get content version history"""
        try:
            versions = [
                v for v in self.content_versions.values() 
                if v.content_id == content_id
            ]
            
            # Sort by version number
            versions.sort(key=lambda x: x.version_number, reverse=True)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting content history: {e}")
            return []
    
    async def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """Compare two content versions"""
        try:
            if version_id_1 not in self.content_versions or version_id_2 not in self.content_versions:
                raise ValueError("One or both versions not found")
            
            version_1 = self.content_versions[version_id_1]
            version_2 = self.content_versions[version_id_2]
            
            # Generate diff
            diff = list(difflib.unified_diff(
                version_1.content.splitlines(keepends=True),
                version_2.content.splitlines(keepends=True),
                fromfile=f"Version {version_1.version_number}",
                tofile=f"Version {version_2.version_number}"
            ))
            
            # Calculate statistics
            lines_added = len([line for line in diff if line.startswith('+') and not line.startswith('+++')])
            lines_removed = len([line for line in diff if line.startswith('-') and not line.startswith('---')])
            lines_modified = min(lines_added, lines_removed)
            
            return {
                "version_1": {
                    "version_id": version_1.version_id,
                    "version_number": version_1.version_number,
                    "author_id": version_1.author_id,
                    "created_at": version_1.created_at.isoformat()
                },
                "version_2": {
                    "version_id": version_2.version_id,
                    "version_number": version_2.version_number,
                    "author_id": version_2.author_id,
                    "created_at": version_2.created_at.isoformat()
                },
                "diff": ''.join(diff),
                "statistics": {
                    "lines_added": lines_added,
                    "lines_removed": lines_removed,
                    "lines_modified": lines_modified,
                    "total_changes": lines_added + lines_removed
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            raise
    
    async def add_comment(self, content_id: str, author_id: str, content: str, 
                         parent_comment_id: str = None) -> Comment:
        """Add comment to content"""
        try:
            comment_id = str(uuid.uuid4())
            
            # Extract mentions
            mentions = self._extract_mentions(content)
            
            comment = Comment(
                comment_id=comment_id,
                content_id=content_id,
                author_id=author_id,
                content=content,
                parent_comment_id=parent_comment_id,
                mentions=mentions
            )
            
            # Store comment
            self.comments[comment_id] = comment
            
            # Send notifications to mentioned users
            for mentioned_user in mentions:
                await self._send_notification(
                    mentioned_user,
                    NotificationType.MENTION,
                    "You were mentioned in a comment",
                    f"You were mentioned in a comment on content {content_id}",
                    content_id=content_id,
                    sender_id=author_id
                )
            
            # Send notification to content author
            await self._send_notification(
                author_id,
                NotificationType.COMMENT,
                "New comment added",
                f"A new comment was added to content {content_id}",
                content_id=content_id,
                sender_id=author_id
            )
            
            logger.info(f"Comment {comment_id} added to content {content_id}")
            
            return comment
            
        except Exception as e:
            logger.error(f"Error adding comment: {e}")
            raise
    
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract user mentions from content"""
        import re
        mentions = re.findall(r'@(\w+)', content)
        return mentions
    
    async def get_comments(self, content_id: str) -> List[Comment]:
        """Get all comments for content"""
        try:
            comments = [
                c for c in self.comments.values() 
                if c.content_id == content_id
            ]
            
            # Sort by creation time
            comments.sort(key=lambda x: x.created_at)
            
            return comments
            
        except Exception as e:
            logger.error(f"Error getting comments: {e}")
            return []
    
    async def start_collaboration_session(self, content_id: str, user_id: str) -> CollaborationSession:
        """Start a collaboration session"""
        try:
            # Check if session already exists
            existing_session = None
            for session in self.collaboration_sessions.values():
                if session.content_id == content_id:
                    existing_session = session
                    break
            
            if existing_session:
                # Add user to existing session
                if user_id not in existing_session.participants:
                    existing_session.participants.append(user_id)
                existing_session.last_activity = datetime.utcnow()
                return existing_session
            
            # Create new session
            session_id = str(uuid.uuid4())
            session = CollaborationSession(
                session_id=session_id,
                content_id=content_id,
                participants=[user_id]
            )
            
            # Store session
            self.collaboration_sessions[session_id] = session
            
            logger.info(f"Collaboration session {session_id} started for content {content_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error starting collaboration session: {e}")
            raise
    
    async def end_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """End a collaboration session"""
        try:
            if session_id not in self.collaboration_sessions:
                return False
            
            session = self.collaboration_sessions[session_id]
            
            # Remove user from session
            if user_id in session.participants:
                session.participants.remove(user_id)
            
            # Remove user from active editors
            if user_id in session.active_editors:
                del session.active_editors[user_id]
            
            # If no participants left, remove session
            if not session.participants:
                del self.collaboration_sessions[session_id]
            
            logger.info(f"User {user_id} left collaboration session {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ending collaboration session: {e}")
            return False
    
    async def handle_websocket_connection(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection for real-time collaboration"""
        try:
            await websocket.accept()
            self.websocket_connections[user_id] = websocket
            
            # Send welcome message
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            # Keep connection alive
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_websocket_message(user_id, message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            # Clean up connection
            if user_id in self.websocket_connections:
                del self.websocket_connections[user_id]
    
    async def _handle_websocket_message(self, user_id: str, message: Dict[str, Any]):
        """Handle WebSocket message"""
        try:
            message_type = message.get("type")
            
            if message_type == "join_session":
                await self._handle_join_session(user_id, message)
            elif message_type == "leave_session":
                await self._handle_leave_session(user_id, message)
            elif message_type == "cursor_update":
                await self._handle_cursor_update(user_id, message)
            elif message_type == "content_change":
                await self._handle_content_change(user_id, message)
            elif message_type == "typing":
                await self._handle_typing(user_id, message)
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _handle_join_session(self, user_id: str, message: Dict[str, Any]):
        """Handle join session message"""
        try:
            content_id = message.get("content_id")
            if content_id:
                session = await self.start_collaboration_session(content_id, user_id)
                
                # Notify other participants
                await self._broadcast_to_session(session.session_id, {
                    "type": "user_joined",
                    "user_id": user_id,
                    "participants": session.participants
                }, exclude_user=user_id)
                
        except Exception as e:
            logger.error(f"Error handling join session: {e}")
    
    async def _handle_leave_session(self, user_id: str, message: Dict[str, Any]):
        """Handle leave session message"""
        try:
            session_id = message.get("session_id")
            if session_id:
                await self.end_collaboration_session(session_id, user_id)
                
                # Notify other participants
                await self._broadcast_to_session(session_id, {
                    "type": "user_left",
                    "user_id": user_id
                })
                
        except Exception as e:
            logger.error(f"Error handling leave session: {e}")
    
    async def _handle_cursor_update(self, user_id: str, message: Dict[str, Any]):
        """Handle cursor update message"""
        try:
            session_id = message.get("session_id")
            cursor_position = message.get("cursor_position")
            
            if session_id in self.collaboration_sessions:
                session = self.collaboration_sessions[session_id]
                session.active_editors[user_id] = cursor_position
                
                # Broadcast cursor position to other users
                await self._broadcast_to_session(session_id, {
                    "type": "cursor_update",
                    "user_id": user_id,
                    "cursor_position": cursor_position
                }, exclude_user=user_id)
                
        except Exception as e:
            logger.error(f"Error handling cursor update: {e}")
    
    async def _handle_content_change(self, user_id: str, message: Dict[str, Any]):
        """Handle content change message"""
        try:
            session_id = message.get("session_id")
            content = message.get("content")
            change_type = message.get("change_type")
            
            if session_id in self.collaboration_sessions:
                session = self.collaboration_sessions[session_id]
                
                # Record change
                change = {
                    "user_id": user_id,
                    "change_type": change_type,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat()
                }
                session.changes.append(change)
                session.last_activity = datetime.utcnow()
                
                # Broadcast change to other users
                await self._broadcast_to_session(session_id, {
                    "type": "content_change",
                    "user_id": user_id,
                    "content": content,
                    "change_type": change_type
                }, exclude_user=user_id)
                
        except Exception as e:
            logger.error(f"Error handling content change: {e}")
    
    async def _handle_typing(self, user_id: str, message: Dict[str, Any]):
        """Handle typing indicator message"""
        try:
            session_id = message.get("session_id")
            is_typing = message.get("is_typing")
            
            if session_id in self.collaboration_sessions:
                # Broadcast typing indicator to other users
                await self._broadcast_to_session(session_id, {
                    "type": "typing",
                    "user_id": user_id,
                    "is_typing": is_typing
                }, exclude_user=user_id)
                
        except Exception as e:
            logger.error(f"Error handling typing indicator: {e}")
    
    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: str = None):
        """Broadcast message to all users in a session"""
        try:
            if session_id not in self.collaboration_sessions:
                return
            
            session = self.collaboration_sessions[session_id]
            
            for user_id in session.participants:
                if user_id != exclude_user and user_id in self.websocket_connections:
                    websocket = self.websocket_connections[user_id]
                    try:
                        await websocket.send_text(json.dumps(message))
                    except:
                        # Remove disconnected user
                        del self.websocket_connections[user_id]
                        
        except Exception as e:
            logger.error(f"Error broadcasting to session: {e}")
    
    async def _send_notification(self, user_id: str, notification_type: NotificationType, 
                               title: str, message: str, content_id: str = None, 
                               sender_id: str = None) -> Notification:
        """Send notification to user"""
        try:
            notification_id = str(uuid.uuid4())
            
            notification = Notification(
                notification_id=notification_id,
                user_id=user_id,
                type=notification_type,
                title=title,
                message=message,
                content_id=content_id,
                sender_id=sender_id
            )
            
            # Store notification
            self.notifications[notification_id] = notification
            
            # Send real-time notification via WebSocket
            if user_id in self.websocket_connections:
                websocket = self.websocket_connections[user_id]
                try:
                    await websocket.send_text(json.dumps({
                        "type": "notification",
                        "notification": {
                            "id": notification_id,
                            "type": notification_type.value,
                            "title": title,
                            "message": message,
                            "content_id": content_id,
                            "timestamp": notification.created_at.isoformat()
                        }
                    }))
                except:
                    pass
            
            # Send email notification
            if user_id in self.users:
                user = self.users[user_id]
                await self._send_email_notification(user.email, title, message)
            
            logger.info(f"Notification {notification_id} sent to user {user_id}")
            
            return notification
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            raise
    
    async def _send_email_notification(self, email: str, title: str, message: str):
        """Send email notification"""
        try:
            if not self.email_config.get("username"):
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config["username"]
            msg['To'] = email
            msg['Subject'] = title
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def get_user_notifications(self, user_id: str, limit: int = 50) -> List[Notification]:
        """Get user notifications"""
        try:
            notifications = [
                n for n in self.notifications.values() 
                if n.user_id == user_id
            ]
            
            # Sort by creation time (newest first)
            notifications.sort(key=lambda x: x.created_at, reverse=True)
            
            return notifications[:limit]
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark notification as read"""
        try:
            if notification_id in self.notifications:
                self.notifications[notification_id].is_read = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            return False
    
    def _process_notifications(self):
        """Process notifications in background"""
        while True:
            try:
                # Process pending notifications
                # This would typically involve sending emails, SMS, etc.
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error processing notifications: {e}")
                time.sleep(60)
    
    def _cleanup_sessions(self):
        """Clean up inactive sessions"""
        while True:
            try:
                current_time = datetime.utcnow()
                inactive_sessions = []
                
                for session_id, session in self.collaboration_sessions.items():
                    # Remove sessions inactive for more than 1 hour
                    if (current_time - session.last_activity).total_seconds() > 3600:
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    del self.collaboration_sessions[session_id]
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error cleaning up sessions: {e}")
                time.sleep(300)
    
    async def get_collaboration_analytics(self, team_id: str, time_period: str = "30d") -> Dict[str, Any]:
        """Get collaboration analytics for team"""
        try:
            if team_id not in self.teams:
                return {"error": "Team not found"}
            
            team = self.teams[team_id]
            
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Get team activity
            team_activity = {
                "total_members": len(team.members),
                "active_members": 0,
                "content_versions_created": 0,
                "comments_added": 0,
                "collaboration_sessions": 0,
                "notifications_sent": 0
            }
            
            # Count active members
            for user_id in team.members:
                if user_id in self.users:
                    user = self.users[user_id]
                    if (datetime.utcnow() - user.last_active).total_seconds() < 86400:  # Active in last 24 hours
                        team_activity["active_members"] += 1
            
            # Count content versions
            for version in self.content_versions.values():
                if (version.author_id in team.members and 
                    start_date <= version.created_at <= end_date):
                    team_activity["content_versions_created"] += 1
            
            # Count comments
            for comment in self.comments.values():
                if (comment.author_id in team.members and 
                    start_date <= comment.created_at <= end_date):
                    team_activity["comments_added"] += 1
            
            # Count collaboration sessions
            for session in self.collaboration_sessions.values():
                if any(user_id in team.members for user_id in session.participants):
                    team_activity["collaboration_sessions"] += 1
            
            # Count notifications
            for notification in self.notifications.values():
                if (notification.user_id in team.members and 
                    start_date <= notification.created_at <= end_date):
                    team_activity["notifications_sent"] += 1
            
            return {
                "team_id": team_id,
                "time_period": time_period,
                "activity": team_activity,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collaboration analytics: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Collaboration Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/collaborationdb",
            "redis_url": "redis://localhost:6379",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_username": "your-email@gmail.com",
            "email_password": "your-password",
            "slack_bot_token": "your-slack-bot-token"
        }
        
        engine = ContentCollaborationEngine(config)
        
        # Create users
        print("Creating users...")
        user1 = await engine.create_user({
            "username": "john_doe",
            "email": "john@example.com",
            "display_name": "John Doe",
            "role": "editor"
        })
        
        user2 = await engine.create_user({
            "username": "jane_smith",
            "email": "jane@example.com",
            "display_name": "Jane Smith",
            "role": "writer"
        })
        
        # Create team
        print("Creating team...")
        team = await engine.create_team({
            "name": "Content Team",
            "description": "Main content creation team"
        }, user1.user_id)
        
        # Add team member
        print("Adding team member...")
        await engine.add_team_member(team.team_id, user2.user_id, UserRole.WRITER)
        
        # Create content version
        print("Creating content version...")
        version = await engine.create_content_version(
            "content_001",
            "This is the initial content version.",
            user1.user_id,
            ["Initial content creation"]
        )
        
        # Add comment
        print("Adding comment...")
        comment = await engine.add_comment(
            "content_001",
            user2.user_id,
            "Great start! @john_doe, what do you think about adding more details?"
        )
        
        # Start collaboration session
        print("Starting collaboration session...")
        session = await engine.start_collaboration_session("content_001", user1.user_id)
        
        # Get content history
        print("Getting content history...")
        history = await engine.get_content_history("content_001")
        print(f"Content has {len(history)} versions")
        
        # Get comments
        print("Getting comments...")
        comments = await engine.get_comments("content_001")
        print(f"Content has {len(comments)} comments")
        
        # Get user notifications
        print("Getting user notifications...")
        notifications = await engine.get_user_notifications(user1.user_id)
        print(f"User has {len(notifications)} notifications")
        
        # Get collaboration analytics
        print("Getting collaboration analytics...")
        analytics = await engine.get_collaboration_analytics(team.team_id)
        print(f"Team activity: {analytics['activity']}")
        
        print("\nContent Collaboration Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























