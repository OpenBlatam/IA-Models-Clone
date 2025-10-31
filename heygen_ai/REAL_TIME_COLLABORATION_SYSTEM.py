#!/usr/bin/env python3
"""
🤝 HeyGen AI - Real-Time Collaboration System
=============================================

This module implements a comprehensive real-time collaboration system that enables
multiple users to work together on AI projects with live synchronization, conflict
resolution, and advanced collaboration features.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import asyncio
from concurrent.futures import ThreadPoolExecutor
import websockets
from websockets.server import serve
import aiohttp
from aiohttp import web, WSMsgType
import redis
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborationEventType(str, Enum):
    """Collaboration event types"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_UPDATED = "user_updated"
    CONTENT_CHANGED = "content_changed"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    COMMENT_ADDED = "comment_added"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    PERMISSION_CHANGED = "permission_changed"
    PROJECT_SHARED = "project_shared"

class UserRole(str, Enum):
    """User roles in collaboration"""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"

class ConflictResolutionStrategy(str, Enum):
    """Conflict resolution strategies"""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MANUAL_RESOLUTION = "manual_resolution"
    AUTOMATIC_MERGE = "automatic_merge"
    USER_PREFERENCE = "user_preference"

@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    email: str
    role: UserRole
    avatar_url: Optional[str] = None
    is_online: bool = False
    last_seen: datetime = field(default_factory=datetime.now)
    current_cursor: Optional[Dict[str, Any]] = None
    current_selection: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationEvent:
    """Collaboration event"""
    event_id: str
    event_type: CollaborationEventType
    user_id: str
    project_id: str
    timestamp: datetime
    data: Dict[str, Any]
    version: int = 1
    parent_event_id: Optional[str] = None
    conflict_resolution: Optional[ConflictResolutionStrategy] = None

@dataclass
class Project:
    """Collaboration project"""
    project_id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_public: bool = False
    collaborators: Dict[str, UserRole] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Comment:
    """Comment/annotation"""
    comment_id: str
    project_id: str
    user_id: str
    content: str
    position: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    replies: List[str] = field(default_factory=list)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConflictResolver:
    """Advanced conflict resolution system"""
    
    def __init__(self):
        self.conflict_history = []
        self.merge_strategies = {}
        self.auto_merge_enabled = True
    
    def detect_conflict(self, event1: CollaborationEvent, event2: CollaborationEvent) -> bool:
        """Detect if two events conflict"""
        if event1.event_type != event2.event_type:
            return False
        
        if event1.user_id == event2.user_id:
            return False
        
        # Check if events affect the same content
        if event1.event_type == CollaborationEventType.CONTENT_CHANGED:
            return self._detect_content_conflict(event1, event2)
        elif event1.event_type == CollaborationEventType.CURSOR_MOVED:
            return False  # Cursor movements don't conflict
        elif event1.event_type == CollaborationEventType.SELECTION_CHANGED:
            return False  # Selections don't conflict
        
        return False
    
    def _detect_content_conflict(self, event1: CollaborationEvent, event2: CollaborationEvent) -> bool:
        """Detect content conflict between two events"""
        # Check if events modify the same content area
        data1 = event1.data
        data2 = event2.data
        
        # Simple conflict detection based on content paths
        path1 = data1.get('path', '')
        path2 = data2.get('path', '')
        
        if path1 != path2:
            return False
        
        # Check timestamp proximity (events within 5 seconds might conflict)
        time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
        return time_diff < 5.0
    
    def resolve_conflict(self, event1: CollaborationEvent, event2: CollaborationEvent, 
                        strategy: ConflictResolutionStrategy) -> CollaborationEvent:
        """Resolve conflict between two events"""
        conflict_id = f"conflict_{event1.event_id}_{event2.event_id}"
        
        if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return event1 if event1.timestamp > event2.timestamp else event2
        
        elif strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            return event1 if event1.timestamp < event2.timestamp else event2
        
        elif strategy == ConflictResolutionStrategy.AUTOMATIC_MERGE:
            return self._automatic_merge(event1, event2)
        
        elif strategy == ConflictResolutionStrategy.USER_PREFERENCE:
            # This would involve user interaction
            return self._user_preference_resolution(event1, event2)
        
        else:
            # Default to last write wins
            return event1 if event1.timestamp > event2.timestamp else event2
    
    def _automatic_merge(self, event1: CollaborationEvent, event2: CollaborationEvent) -> CollaborationEvent:
        """Automatically merge conflicting events"""
        # Create merged event
        merged_data = event1.data.copy()
        
        # Merge data from both events
        for key, value in event2.data.items():
            if key in merged_data:
                if isinstance(merged_data[key], dict) and isinstance(value, dict):
                    merged_data[key].update(value)
                elif isinstance(merged_data[key], list) and isinstance(value, list):
                    merged_data[key].extend(value)
                else:
                    # Use the more recent value
                    merged_data[key] = value
            else:
                merged_data[key] = value
        
        # Create merged event
        merged_event = CollaborationEvent(
            event_id=f"merged_{event1.event_id}_{event2.event_id}",
            event_type=event1.event_type,
            user_id="system",  # System-generated merge
            project_id=event1.project_id,
            timestamp=datetime.now(),
            data=merged_data,
            version=max(event1.version, event2.version) + 1,
            parent_event_id=event1.event_id,
            conflict_resolution=ConflictResolutionStrategy.AUTOMATIC_MERGE
        )
        
        return merged_event
    
    def _user_preference_resolution(self, event1: CollaborationEvent, event2: CollaborationEvent) -> CollaborationEvent:
        """Resolve conflict based on user preference (placeholder)"""
        # In a real implementation, this would involve user interaction
        # For now, return the first event
        return event1

class RealTimeSync:
    """Real-time synchronization system"""
    
    def __init__(self):
        self.connections = {}
        self.event_queue = asyncio.Queue()
        self.sync_lock = asyncio.Lock()
        self.redis_client = None
        self.pubsub = None
    
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize real-time sync with Redis"""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.pubsub = self.redis_client.pubsub()
            logger.info("✅ Real-time sync initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory sync: {e}")
            self.redis_client = None
    
    async def add_connection(self, user_id: str, websocket: Any, project_id: str):
        """Add user connection"""
        async with self.sync_lock:
            if user_id not in self.connections:
                self.connections[user_id] = []
            
            self.connections[user_id].append({
                'websocket': websocket,
                'project_id': project_id,
                'connected_at': datetime.now()
            })
    
    async def remove_connection(self, user_id: str, websocket: Any):
        """Remove user connection"""
        async with self.sync_lock:
            if user_id in self.connections:
                self.connections[user_id] = [
                    conn for conn in self.connections[user_id] 
                    if conn['websocket'] != websocket
                ]
                
                if not self.connections[user_id]:
                    del self.connections[user_id]
    
    async def broadcast_event(self, event: CollaborationEvent, exclude_user: Optional[str] = None):
        """Broadcast event to all connected users"""
        if self.redis_client:
            # Use Redis pub/sub for distributed sync
            await self.redis_client.publish(
                f"project_{event.project_id}",
                json.dumps(event.__dict__, default=str)
            )
        else:
            # Use direct WebSocket broadcast
            await self._direct_broadcast(event, exclude_user)
    
    async def _direct_broadcast(self, event: CollaborationEvent, exclude_user: Optional[str] = None):
        """Direct WebSocket broadcast"""
        message = {
            'type': 'collaboration_event',
            'event': event.__dict__
        }
        
        for user_id, connections in self.connections.items():
            if exclude_user and user_id == exclude_user:
                continue
            
            for conn in connections:
                if conn['project_id'] == event.project_id:
                    try:
                        await conn['websocket'].send(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Failed to send message to {user_id}: {e}")
    
    async def subscribe_to_project(self, project_id: str, callback: Callable):
        """Subscribe to project events"""
        if self.redis_client:
            await self.pubsub.subscribe(f"project_{project_id}")
            
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        event = CollaborationEvent(**event_data)
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Failed to process Redis message: {e}")

class CommentSystem:
    """Advanced comment and annotation system"""
    
    def __init__(self):
        self.comments = {}
        self.comment_threads = {}
        self.mention_notifications = {}
    
    def add_comment(self, comment: Comment) -> str:
        """Add comment to project"""
        self.comments[comment.comment_id] = comment
        
        # Create thread if it doesn't exist
        thread_key = f"{comment.project_id}_{comment.position.get('path', 'root')}"
        if thread_key not in self.comment_threads:
            self.comment_threads[thread_key] = []
        
        self.comment_threads[thread_key].append(comment.comment_id)
        
        # Check for mentions
        self._process_mentions(comment)
        
        return comment.comment_id
    
    def _process_mentions(self, comment: Comment):
        """Process @mentions in comments"""
        content = comment.content
        mentions = []
        
        # Simple mention detection (look for @username)
        import re
        mention_pattern = r'@(\w+)'
        matches = re.findall(mention_pattern, content)
        
        for username in matches:
            mentions.append(username)
            # Store mention notification
            if username not in self.mention_notifications:
                self.mention_notifications[username] = []
            
            self.mention_notifications[username].append({
                'comment_id': comment.comment_id,
                'project_id': comment.project_id,
                'mentioned_by': comment.user_id,
                'timestamp': comment.created_at
            })
    
    def get_comments_for_position(self, project_id: str, position: Dict[str, Any]) -> List[Comment]:
        """Get comments for specific position"""
        thread_key = f"{project_id}_{position.get('path', 'root')}"
        comment_ids = self.comment_threads.get(thread_key, [])
        
        return [self.comments[cid] for cid in comment_ids if cid in self.comments]
    
    def resolve_comment(self, comment_id: str, user_id: str) -> bool:
        """Resolve comment"""
        if comment_id in self.comments:
            comment = self.comments[comment_id]
            comment.resolved = True
            comment.updated_at = datetime.now()
            return True
        return False

class RealTimeCollaborationSystem:
    """Main real-time collaboration system"""
    
    def __init__(self):
        self.users = {}
        self.projects = {}
        self.active_sessions = {}
        self.conflict_resolver = ConflictResolver()
        self.real_time_sync = RealTimeSync()
        self.comment_system = CommentSystem()
        self.event_history = {}
        self.permissions = {}
        self.initialized = False
    
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize collaboration system"""
        try:
            logger.info("🤝 Initializing Real-Time Collaboration System...")
            
            # Initialize components
            await self.real_time_sync.initialize(redis_url)
            
            self.initialized = True
            logger.info("✅ Real-Time Collaboration System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Collaboration System: {e}")
            raise
    
    def create_user(self, user_id: str, username: str, email: str, 
                   role: UserRole = UserRole.EDITOR) -> User:
        """Create new user"""
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            is_online=False
        )
        
        self.users[user_id] = user
        logger.info(f"✅ Created user: {username}")
        return user
    
    def create_project(self, project_id: str, name: str, description: str, 
                      owner_id: str, is_public: bool = False) -> Project:
        """Create new project"""
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            owner_id=owner_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_public=is_public
        )
        
        # Add owner as collaborator
        project.collaborators[owner_id] = UserRole.OWNER
        
        self.projects[project_id] = project
        logger.info(f"✅ Created project: {name}")
        return project
    
    async def join_project(self, user_id: str, project_id: str, websocket: Any) -> bool:
        """Join user to project"""
        if project_id not in self.projects:
            logger.error(f"Project {project_id} not found")
            return False
        
        if user_id not in self.users:
            logger.error(f"User {user_id} not found")
            return False
        
        project = self.projects[project_id]
        user = self.users[user_id]
        
        # Check permissions
        if not self._has_permission(user_id, project_id, "read"):
            logger.error(f"User {user_id} doesn't have permission to join project {project_id}")
            return False
        
        # Add to active sessions
        session_key = f"{user_id}_{project_id}"
        self.active_sessions[session_key] = {
            'user_id': user_id,
            'project_id': project_id,
            'websocket': websocket,
            'joined_at': datetime.now()
        }
        
        # Add connection to real-time sync
        await self.real_time_sync.add_connection(user_id, websocket, project_id)
        
        # Update user status
        user.is_online = True
        user.last_seen = datetime.now()
        
        # Broadcast user joined event
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.USER_JOINED,
            user_id=user_id,
            project_id=project_id,
            timestamp=datetime.now(),
            data={
                'username': user.username,
                'role': user.role.value,
                'avatar_url': user.avatar_url
            }
        )
        
        await self._process_event(event)
        
        logger.info(f"✅ User {user.username} joined project {project.name}")
        return True
    
    async def leave_project(self, user_id: str, project_id: str, websocket: Any):
        """Leave project"""
        session_key = f"{user_id}_{project_id}"
        
        if session_key in self.active_sessions:
            del self.active_sessions[session_key]
        
        # Remove from real-time sync
        await self.real_time_sync.remove_connection(user_id, websocket)
        
        # Update user status
        if user_id in self.users:
            self.users[user_id].is_online = False
        
        # Broadcast user left event
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.USER_LEFT,
            user_id=user_id,
            project_id=project_id,
            timestamp=datetime.now(),
            data={'username': self.users[user_id].username}
        )
        
        await self._process_event(event)
        
        logger.info(f"✅ User {user_id} left project {project_id}")
    
    async def handle_websocket_message(self, user_id: str, project_id: str, message: Dict[str, Any]):
        """Handle WebSocket message from client"""
        try:
            message_type = message.get('type')
            
            if message_type == 'cursor_move':
                await self._handle_cursor_move(user_id, project_id, message)
            elif message_type == 'content_change':
                await self._handle_content_change(user_id, project_id, message)
            elif message_type == 'selection_change':
                await self._handle_selection_change(user_id, project_id, message)
            elif message_type == 'add_comment':
                await self._handle_add_comment(user_id, project_id, message)
            elif message_type == 'resolve_comment':
                await self._handle_resolve_comment(user_id, project_id, message)
            else:
                logger.warning(f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Failed to handle WebSocket message: {e}")
    
    async def _handle_cursor_move(self, user_id: str, project_id: str, message: Dict[str, Any]):
        """Handle cursor movement"""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.CURSOR_MOVED,
            user_id=user_id,
            project_id=project_id,
            timestamp=datetime.now(),
            data=message.get('data', {})
        )
        
        await self._process_event(event)
    
    async def _handle_content_change(self, user_id: str, project_id: str, message: Dict[str, Any]):
        """Handle content change"""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.CONTENT_CHANGED,
            user_id=user_id,
            project_id=project_id,
            timestamp=datetime.now(),
            data=message.get('data', {})
        )
        
        await self._process_event(event)
    
    async def _handle_selection_change(self, user_id: str, project_id: str, message: Dict[str, Any]):
        """Handle selection change"""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.SELECTION_CHANGED,
            user_id=user_id,
            project_id=project_id,
            timestamp=datetime.now(),
            data=message.get('data', {})
        )
        
        await self._process_event(event)
    
    async def _handle_add_comment(self, user_id: str, project_id: str, message: Dict[str, Any]):
        """Handle add comment"""
        data = message.get('data', {})
        
        comment = Comment(
            comment_id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            content=data.get('content', ''),
            position=data.get('position', {}),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        comment_id = self.comment_system.add_comment(comment)
        
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.COMMENT_ADDED,
            user_id=user_id,
            project_id=project_id,
            timestamp=datetime.now(),
            data={
                'comment_id': comment_id,
                'content': comment.content,
                'position': comment.position
            }
        )
        
        await self._process_event(event)
    
    async def _handle_resolve_comment(self, user_id: str, project_id: str, message: Dict[str, Any]):
        """Handle resolve comment"""
        data = message.get('data', {})
        comment_id = data.get('comment_id')
        
        if self.comment_system.resolve_comment(comment_id, user_id):
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=CollaborationEventType.COMMENT_DELETED,
                user_id=user_id,
                project_id=project_id,
                timestamp=datetime.now(),
                data={'comment_id': comment_id}
            )
            
            await self._process_event(event)
    
    async def _process_event(self, event: CollaborationEvent):
        """Process collaboration event"""
        # Check for conflicts
        conflicts = self._detect_conflicts(event)
        
        if conflicts:
            # Resolve conflicts
            for conflict_event in conflicts:
                resolved_event = self.conflict_resolver.resolve_conflict(
                    event, conflict_event, ConflictResolutionStrategy.AUTOMATIC_MERGE
                )
                
                # Replace original event with resolved event
                event = resolved_event
        
        # Store event in history
        if event.project_id not in self.event_history:
            self.event_history[event.project_id] = []
        
        self.event_history[event.project_id].append(event)
        
        # Update project version
        if event.project_id in self.projects:
            self.projects[event.project_id].version += 1
            self.projects[event.project_id].updated_at = datetime.now()
        
        # Broadcast event
        await self.real_time_sync.broadcast_event(event, exclude_user=event.user_id)
    
    def _detect_conflicts(self, event: CollaborationEvent) -> List[CollaborationEvent]:
        """Detect conflicts with recent events"""
        if event.project_id not in self.event_history:
            return []
        
        recent_events = self.event_history[event.project_id][-10:]  # Check last 10 events
        conflicts = []
        
        for recent_event in recent_events:
            if self.conflict_resolver.detect_conflict(event, recent_event):
                conflicts.append(recent_event)
        
        return conflicts
    
    def _has_permission(self, user_id: str, project_id: str, permission: str) -> bool:
        """Check if user has permission for project"""
        if project_id not in self.projects:
            return False
        
        project = self.projects[project_id]
        
        # Owner has all permissions
        if project.owner_id == user_id:
            return True
        
        # Check collaborator permissions
        if user_id in project.collaborators:
            role = project.collaborators[user_id]
            
            if role == UserRole.ADMIN:
                return True
            elif role == UserRole.EDITOR:
                return permission in ['read', 'write', 'comment']
            elif role == UserRole.VIEWER:
                return permission in ['read', 'comment']
            elif role == UserRole.GUEST:
                return permission == 'read'
        
        return False
    
    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get project collaboration status"""
        if project_id not in self.projects:
            return {'error': 'Project not found'}
        
        project = self.projects[project_id]
        active_users = []
        
        for session_key, session in self.active_sessions.items():
            if session['project_id'] == project_id:
                user_id = session['user_id']
                if user_id in self.users:
                    user = self.users[user_id]
                    active_users.append({
                        'user_id': user_id,
                        'username': user.username,
                        'role': user.role.value,
                        'joined_at': session['joined_at'].isoformat(),
                        'cursor': user.current_cursor,
                        'selection': user.current_selection
                    })
        
        return {
            'project_id': project_id,
            'name': project.name,
            'version': project.version,
            'active_users': active_users,
            'total_collaborators': len(project.collaborators),
            'is_public': project.is_public,
            'last_updated': project.updated_at.isoformat()
        }
    
    async def shutdown(self):
        """Shutdown collaboration system"""
        self.initialized = False
        logger.info("✅ Real-Time Collaboration System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the real-time collaboration system"""
    print("🤝 HeyGen AI - Real-Time Collaboration System Demo")
    print("=" * 70)
    
    # Initialize collaboration system
    collaboration_system = RealTimeCollaborationSystem()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Collaboration System...")
        await collaboration_system.initialize()
        print("✅ Collaboration System initialized successfully")
        
        # Create demo users
        print("\n👥 Creating Demo Users...")
        
        users = [
            collaboration_system.create_user("user1", "Alice", "alice@example.com", UserRole.OWNER),
            collaboration_system.create_user("user2", "Bob", "bob@example.com", UserRole.EDITOR),
            collaboration_system.create_user("user3", "Charlie", "charlie@example.com", UserRole.VIEWER)
        ]
        
        for user in users:
            print(f"  ✅ Created user: {user.username} ({user.role.value})")
        
        # Create demo project
        print("\n📁 Creating Demo Project...")
        
        project = collaboration_system.create_project(
            project_id="proj1",
            name="AI Model Training Project",
            description="Collaborative AI model development",
            owner_id="user1",
            is_public=False
        )
        
        print(f"  ✅ Created project: {project.name}")
        
        # Add collaborators
        print("\n🤝 Adding Collaborators...")
        
        project.collaborators["user2"] = UserRole.EDITOR
        project.collaborators["user3"] = UserRole.VIEWER
        
        print(f"  ✅ Added {len(project.collaborators)} collaborators")
        
        # Simulate collaboration events
        print("\n🔄 Simulating Collaboration Events...")
        
        # Simulate user joining
        event1 = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.USER_JOINED,
            user_id="user1",
            project_id="proj1",
            timestamp=datetime.now(),
            data={'username': 'Alice', 'role': 'owner'}
        )
        
        await collaboration_system._process_event(event1)
        print("  ✅ User Alice joined project")
        
        # Simulate content change
        event2 = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.CONTENT_CHANGED,
            user_id="user1",
            project_id="proj1",
            timestamp=datetime.now(),
            data={
                'path': 'model.py',
                'content': 'class MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()',
                'change_type': 'insert'
            }
        )
        
        await collaboration_system._process_event(event2)
        print("  ✅ Content changed in model.py")
        
        # Simulate cursor movement
        event3 = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.CURSOR_MOVED,
            user_id="user2",
            project_id="proj1",
            timestamp=datetime.now(),
            data={
                'path': 'model.py',
                'line': 5,
                'column': 10
            }
        )
        
        await collaboration_system._process_event(event3)
        print("  ✅ User Bob moved cursor")
        
        # Simulate comment addition
        comment = Comment(
            comment_id=str(uuid.uuid4()),
            project_id="proj1",
            user_id="user2",
            content="This looks good! @user1",
            position={'path': 'model.py', 'line': 3},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        comment_id = collaboration_system.comment_system.add_comment(comment)
        
        event4 = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=CollaborationEventType.COMMENT_ADDED,
            user_id="user2",
            project_id="proj1",
            timestamp=datetime.now(),
            data={
                'comment_id': comment_id,
                'content': comment.content,
                'position': comment.position
            }
        )
        
        await collaboration_system._process_event(event4)
        print("  ✅ Comment added with mention")
        
        # Get project status
        print("\n📊 Project Status:")
        status = await collaboration_system.get_project_status("proj1")
        
        print(f"  📁 Project: {status['name']}")
        print(f"  🔢 Version: {status['version']}")
        print(f"  👥 Active Users: {len(status['active_users'])}")
        print(f"  🤝 Total Collaborators: {status['total_collaborators']}")
        print(f"  🔓 Public: {status['is_public']}")
        print(f"  ⏰ Last Updated: {status['last_updated']}")
        
        # Show event history
        print(f"\n📜 Event History: {len(collaboration_system.event_history.get('proj1', []))} events")
        
        # Show comments
        comments = collaboration_system.comment_system.get_comments_for_position(
            "proj1", {'path': 'model.py'}
        )
        print(f"💬 Comments: {len(comments)} comments")
        
        for comment in comments:
            print(f"  - {comment.content} (by {comment.user_id})")
        
        # Show mention notifications
        print(f"\n🔔 Mention Notifications:")
        for username, notifications in collaboration_system.comment_system.mention_notifications.items():
            print(f"  @{username}: {len(notifications)} mentions")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await collaboration_system.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


