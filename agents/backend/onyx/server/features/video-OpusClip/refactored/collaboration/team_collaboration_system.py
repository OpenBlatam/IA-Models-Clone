"""
Team Collaboration System

Advanced team collaboration features for the Ultimate Opus Clip system including
multi-user collaboration, project sharing, version control, and real-time editing.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Set, Tuple
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
import hashlib
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict
import websockets
from websockets.server import serve
import asyncio
import aiofiles

logger = structlog.get_logger("team_collaboration")

class UserRole(Enum):
    """User roles in the collaboration system."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"

class ProjectStatus(Enum):
    """Project status states."""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class CollaborationEvent(Enum):
    """Real-time collaboration events."""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    PROJECT_UPDATED = "project_updated"
    COMMENT_ADDED = "comment_added"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"
    VERSION_CREATED = "version_created"
    PERMISSION_CHANGED = "permission_changed"
    PROJECT_SHARED = "project_shared"

@dataclass
class User:
    """User information."""
    user_id: str
    username: str
    email: str
    display_name: str
    avatar_url: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_online: bool = False

@dataclass
class Project:
    """Project information."""
    project_id: str
    name: str
    description: str
    owner_id: str
    status: ProjectStatus = ProjectStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    video_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectMember:
    """Project member information."""
    project_id: str
    user_id: str
    role: UserRole
    joined_at: datetime = field(default_factory=datetime.now)
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True

@dataclass
class Comment:
    """Comment on project elements."""
    comment_id: str
    project_id: str
    user_id: str
    content: str
    element_type: str  # 'video', 'clip', 'timeline', etc.
    element_id: str
    timestamp: float  # Video timestamp
    position: Dict[str, float]  # x, y coordinates
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    replies: List[str] = field(default_factory=list)  # Comment IDs
    is_resolved: bool = False
    mentions: List[str] = field(default_factory=list)  # User IDs

@dataclass
class ProjectVersion:
    """Project version information."""
    version_id: str
    project_id: str
    version_number: int
    name: str
    description: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    changes: List[Dict[str, Any]] = field(default_factory=list)
    file_hash: str = ""
    file_size: int = 0
    is_auto_save: bool = False

@dataclass
class CollaborationSession:
    """Active collaboration session."""
    session_id: str
    project_id: str
    user_id: str
    joined_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    cursor_position: Optional[Dict[str, float]] = None
    is_editing: bool = False
    editing_element: Optional[str] = None

class UserManager:
    """Manages users and authentication."""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self.users: Dict[str, User] = {}
        self.online_users: Set[str] = set()
        self._init_database()
        self.logger = structlog.get_logger("user_manager")
    
    def _init_database(self):
        """Initialize user database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        display_name TEXT NOT NULL,
                        avatar_url TEXT,
                        role TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize user database: {e}")
    
    async def create_user(self, username: str, email: str, display_name: str, 
                         avatar_url: str = None, role: UserRole = UserRole.VIEWER) -> User:
        """Create a new user."""
        try:
            user_id = str(uuid.uuid4())
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                display_name=display_name,
                avatar_url=avatar_url,
                role=role
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (user_id, username, email, display_name, avatar_url, role)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, username, email, display_name, avatar_url, role.value))
                conn.commit()
            
            self.users[user_id] = user
            self.logger.info(f"Created user: {username}")
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            if user_id in self.users:
                return self.users[user_id]
            
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    user = User(
                        user_id=row[0],
                        username=row[1],
                        email=row[2],
                        display_name=row[3],
                        avatar_url=row[4],
                        role=UserRole(row[5]),
                        created_at=datetime.fromisoformat(row[6]),
                        last_active=datetime.fromisoformat(row[7])
                    )
                    self.users[user_id] = user
                    return user
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get user: {e}")
            return None
    
    async def update_user_activity(self, user_id: str):
        """Update user's last activity time."""
        try:
            if user_id in self.users:
                self.users[user_id].last_active = datetime.now()
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE users SET last_active = ? WHERE user_id = ?",
                        (datetime.now().isoformat(), user_id)
                    )
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to update user activity: {e}")
    
    async def set_user_online(self, user_id: str, is_online: bool = True):
        """Set user online/offline status."""
        try:
            if user_id in self.users:
                self.users[user_id].is_online = is_online
                
                if is_online:
                    self.online_users.add(user_id)
                else:
                    self.online_users.discard(user_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to set user online status: {e}")
    
    async def get_online_users(self) -> List[User]:
        """Get list of online users."""
        return [self.users[user_id] for user_id in self.online_users if user_id in self.users]

class ProjectManager:
    """Manages projects and project members."""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self.projects: Dict[str, Project] = {}
        self.project_members: Dict[str, List[ProjectMember]] = defaultdict(list)
        self._init_database()
        self.logger = structlog.get_logger("project_manager")
    
    def _init_database(self):
        """Initialize project database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Projects table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        project_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        owner_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        video_path TEXT,
                        thumbnail_url TEXT,
                        tags TEXT,
                        settings TEXT
                    )
                """)
                
                # Project members table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_members (
                        project_id TEXT,
                        user_id TEXT,
                        role TEXT NOT NULL,
                        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        permissions TEXT,
                        is_active BOOLEAN DEFAULT TRUE,
                        PRIMARY KEY (project_id, user_id)
                    )
                """)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize project database: {e}")
    
    async def create_project(self, name: str, description: str, owner_id: str, 
                           video_path: str = None, tags: List[str] = None) -> Project:
        """Create a new project."""
        try:
            project_id = str(uuid.uuid4())
            project = Project(
                project_id=project_id,
                name=name,
                description=description,
                owner_id=owner_id,
                video_path=video_path,
                tags=tags or []
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO projects (project_id, name, description, owner_id, status, video_path, tags, settings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id, name, description, owner_id, ProjectStatus.DRAFT.value,
                    video_path, json.dumps(tags or []), json.dumps({})
                ))
                
                # Add owner as project member
                conn.execute("""
                    INSERT INTO project_members (project_id, user_id, role, permissions)
                    VALUES (?, ?, ?, ?)
                """, (project_id, owner_id, UserRole.OWNER.value, json.dumps(["all"])))
                
                conn.commit()
            
            self.projects[project_id] = project
            self.project_members[project_id] = [ProjectMember(
                project_id=project_id,
                user_id=owner_id,
                role=UserRole.OWNER,
                permissions={"all"}
            )]
            
            self.logger.info(f"Created project: {name}")
            return project
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            raise
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        try:
            if project_id in self.projects:
                return self.projects[project_id]
            
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
                row = cursor.fetchone()
                
                if row:
                    project = Project(
                        project_id=row[0],
                        name=row[1],
                        description=row[2],
                        owner_id=row[3],
                        status=ProjectStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        updated_at=datetime.fromisoformat(row[6]),
                        video_path=row[7],
                        thumbnail_url=row[8],
                        tags=json.loads(row[9]) if row[9] else [],
                        settings=json.loads(row[10]) if row[10] else {}
                    )
                    self.projects[project_id] = project
                    return project
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get project: {e}")
            return None
    
    async def add_project_member(self, project_id: str, user_id: str, role: UserRole, 
                               permissions: Set[str] = None) -> bool:
        """Add a member to a project."""
        try:
            if permissions is None:
                permissions = self._get_default_permissions(role)
            
            member = ProjectMember(
                project_id=project_id,
                user_id=user_id,
                role=role,
                permissions=permissions
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO project_members (project_id, user_id, role, permissions, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (project_id, user_id, role.value, json.dumps(list(permissions)), True))
                conn.commit()
            
            # Update in-memory cache
            if project_id not in self.project_members:
                self.project_members[project_id] = []
            
            # Remove existing member if any
            self.project_members[project_id] = [
                m for m in self.project_members[project_id] if m.user_id != user_id
            ]
            
            self.project_members[project_id].append(member)
            
            self.logger.info(f"Added member {user_id} to project {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add project member: {e}")
            return False
    
    def _get_default_permissions(self, role: UserRole) -> Set[str]:
        """Get default permissions for a role."""
        permissions = {
            UserRole.OWNER: {"all"},
            UserRole.ADMIN: {"edit", "comment", "share", "manage_members"},
            UserRole.EDITOR: {"edit", "comment"},
            UserRole.VIEWER: {"view", "comment"},
            UserRole.GUEST: {"view"}
        }
        return permissions.get(role, {"view"})
    
    async def get_project_members(self, project_id: str) -> List[ProjectMember]:
        """Get project members."""
        try:
            if project_id in self.project_members:
                return self.project_members[project_id]
            
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM project_members WHERE project_id = ? AND is_active = TRUE",
                    (project_id,)
                )
                rows = cursor.fetchall()
                
                members = []
                for row in rows:
                    member = ProjectMember(
                        project_id=row[0],
                        user_id=row[1],
                        role=UserRole(row[2]),
                        joined_at=datetime.fromisoformat(row[3]),
                        permissions=set(json.loads(row[4])) if row[4] else set(),
                        is_active=bool(row[5])
                    )
                    members.append(member)
                
                self.project_members[project_id] = members
                return members
                
        except Exception as e:
            self.logger.error(f"Failed to get project members: {e}")
            return []
    
    async def check_permission(self, project_id: str, user_id: str, permission: str) -> bool:
        """Check if user has permission for project."""
        try:
            members = await self.get_project_members(project_id)
            
            for member in members:
                if member.user_id == user_id:
                    return "all" in member.permissions or permission in member.permissions
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check permission: {e}")
            return False

class CommentManager:
    """Manages comments and discussions."""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self.comments: Dict[str, Comment] = {}
        self._init_database()
        self.logger = structlog.get_logger("comment_manager")
    
    def _init_database(self):
        """Initialize comment database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS comments (
                        comment_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        element_type TEXT NOT NULL,
                        element_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        position TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        replies TEXT,
                        is_resolved BOOLEAN DEFAULT FALSE,
                        mentions TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize comment database: {e}")
    
    async def add_comment(self, project_id: str, user_id: str, content: str, 
                         element_type: str, element_id: str, timestamp: float,
                         position: Dict[str, float], mentions: List[str] = None) -> Comment:
        """Add a comment to a project element."""
        try:
            comment_id = str(uuid.uuid4())
            comment = Comment(
                comment_id=comment_id,
                project_id=project_id,
                user_id=user_id,
                content=content,
                element_type=element_type,
                element_id=element_id,
                timestamp=timestamp,
                position=position,
                mentions=mentions or []
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO comments (comment_id, project_id, user_id, content, element_type, 
                                        element_id, timestamp, position, replies, is_resolved, mentions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    comment_id, project_id, user_id, content, element_type,
                    element_id, timestamp, json.dumps(position), json.dumps([]),
                    False, json.dumps(mentions or [])
                ))
                conn.commit()
            
            self.comments[comment_id] = comment
            self.logger.info(f"Added comment: {comment_id}")
            return comment
            
        except Exception as e:
            self.logger.error(f"Failed to add comment: {e}")
            raise
    
    async def get_comments(self, project_id: str, element_id: str = None) -> List[Comment]:
        """Get comments for a project or element."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if element_id:
                    cursor = conn.execute(
                        "SELECT * FROM comments WHERE project_id = ? AND element_id = ? ORDER BY created_at",
                        (project_id, element_id)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM comments WHERE project_id = ? ORDER BY created_at",
                        (project_id,)
                    )
                
                rows = cursor.fetchall()
                comments = []
                
                for row in rows:
                    comment = Comment(
                        comment_id=row[0],
                        project_id=row[1],
                        user_id=row[2],
                        content=row[3],
                        element_type=row[4],
                        element_id=row[5],
                        timestamp=row[6],
                        position=json.loads(row[7]),
                        created_at=datetime.fromisoformat(row[8]),
                        updated_at=datetime.fromisoformat(row[9]),
                        replies=json.loads(row[10]) if row[10] else [],
                        is_resolved=bool(row[11]),
                        mentions=json.loads(row[12]) if row[12] else []
                    )
                    comments.append(comment)
                
                return comments
                
        except Exception as e:
            self.logger.error(f"Failed to get comments: {e}")
            return []

class VersionManager:
    """Manages project versions and version control."""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self.versions: Dict[str, List[ProjectVersion]] = defaultdict(list)
        self._init_database()
        self.logger = structlog.get_logger("version_manager")
    
    def _init_database(self):
        """Initialize version database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_versions (
                        version_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        version_number INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        created_by TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        changes TEXT,
                        file_hash TEXT,
                        file_size INTEGER,
                        is_auto_save BOOLEAN DEFAULT FALSE
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize version database: {e}")
    
    async def create_version(self, project_id: str, name: str, description: str,
                           created_by: str, changes: List[Dict[str, Any]] = None,
                           file_path: str = None, is_auto_save: bool = False) -> ProjectVersion:
        """Create a new project version."""
        try:
            # Get next version number
            version_number = await self._get_next_version_number(project_id)
            
            version_id = str(uuid.uuid4())
            
            # Calculate file hash and size if file provided
            file_hash = ""
            file_size = 0
            if file_path and Path(file_path).exists():
                file_hash = await self._calculate_file_hash(file_path)
                file_size = Path(file_path).stat().st_size
            
            version = ProjectVersion(
                version_id=version_id,
                project_id=project_id,
                version_number=version_number,
                name=name,
                description=description,
                created_by=created_by,
                changes=changes or [],
                file_hash=file_hash,
                file_size=file_size,
                is_auto_save=is_auto_save
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO project_versions (version_id, project_id, version_number, name, 
                                                description, created_by, changes, file_hash, file_size, is_auto_save)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version_id, project_id, version_number, name, description,
                    created_by, json.dumps(changes or []), file_hash, file_size, is_auto_save
                ))
                conn.commit()
            
            self.versions[project_id].append(version)
            self.logger.info(f"Created version {version_number} for project {project_id}")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {e}")
            raise
    
    async def _get_next_version_number(self, project_id: str) -> int:
        """Get next version number for project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT MAX(version_number) FROM project_versions WHERE project_id = ?",
                    (project_id,)
                )
                result = cursor.fetchone()
                return (result[0] or 0) + 1
                
        except Exception as e:
            self.logger.error(f"Failed to get next version number: {e}")
            return 1
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for integrity checking."""
        try:
            hash_md5 = hashlib.md5()
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(4096):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate file hash: {e}")
            return ""
    
    async def get_versions(self, project_id: str) -> List[ProjectVersion]:
        """Get all versions for a project."""
        try:
            if project_id in self.versions:
                return self.versions[project_id]
            
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM project_versions WHERE project_id = ? ORDER BY version_number",
                    (project_id,)
                )
                rows = cursor.fetchall()
                
                versions = []
                for row in rows:
                    version = ProjectVersion(
                        version_id=row[0],
                        project_id=row[1],
                        version_number=row[2],
                        name=row[3],
                        description=row[4],
                        created_by=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        changes=json.loads(row[7]) if row[7] else [],
                        file_hash=row[8],
                        file_size=row[9],
                        is_auto_save=bool(row[10])
                    )
                    versions.append(version)
                
                self.versions[project_id] = versions
                return versions
                
        except Exception as e:
            self.logger.error(f"Failed to get versions: {e}")
            return []

class RealTimeCollaboration:
    """Real-time collaboration features."""
    
    def __init__(self):
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.project_subscribers: Dict[str, Set[str]] = defaultdict(set)  # project_id -> user_ids
        self.logger = structlog.get_logger("realtime_collaboration")
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time collaboration."""
        try:
            async def handle_connection(websocket, path):
                await self._handle_websocket_connection(websocket, path)
            
            server = await serve(handle_connection, host, port)
            self.logger.info(f"WebSocket server started on {host}:{port}")
            return server
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connection."""
        try:
            user_id = None
            project_id = None
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("type")
                    
                    if event_type == "join_project":
                        user_id = data.get("user_id")
                        project_id = data.get("project_id")
                        
                        if user_id and project_id:
                            await self._join_project(websocket, user_id, project_id)
                    
                    elif event_type == "leave_project":
                        await self._leave_project(websocket, user_id, project_id)
                    
                    elif event_type == "cursor_update":
                        await self._update_cursor(websocket, user_id, project_id, data.get("position"))
                    
                    elif event_type == "edit_start":
                        await self._start_editing(websocket, user_id, project_id, data.get("element_id"))
                    
                    elif event_type == "edit_end":
                        await self._end_editing(websocket, user_id, project_id, data.get("element_id"))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))
            
        except websockets.exceptions.ConnectionClosed:
            await self._leave_project(websocket, user_id, project_id)
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
    
    async def _join_project(self, websocket, user_id: str, project_id: str):
        """Join a project for real-time collaboration."""
        try:
            session_id = str(uuid.uuid4())
            session = CollaborationSession(
                session_id=session_id,
                project_id=project_id,
                user_id=user_id
            )
            
            self.active_sessions[session_id] = session
            self.websocket_connections[session_id] = websocket
            self.project_subscribers[project_id].add(session_id)
            
            # Notify other users
            await self._broadcast_to_project(project_id, {
                "type": CollaborationEvent.USER_JOINED.value,
                "user_id": user_id,
                "session_id": session_id
            }, exclude_session=session_id)
            
            self.logger.info(f"User {user_id} joined project {project_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to join project: {e}")
    
    async def _leave_project(self, websocket, user_id: str, project_id: str):
        """Leave a project."""
        try:
            if not user_id or not project_id:
                return
            
            # Find and remove session
            session_to_remove = None
            for session_id, session in self.active_sessions.items():
                if session.user_id == user_id and session.project_id == project_id:
                    session_to_remove = session_id
                    break
            
            if session_to_remove:
                del self.active_sessions[session_to_remove]
                if session_to_remove in self.websocket_connections:
                    del self.websocket_connections[session_to_remove]
                self.project_subscribers[project_id].discard(session_to_remove)
                
                # Notify other users
                await self._broadcast_to_project(project_id, {
                    "type": CollaborationEvent.USER_LEFT.value,
                    "user_id": user_id
                })
                
                self.logger.info(f"User {user_id} left project {project_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to leave project: {e}")
    
    async def _update_cursor(self, websocket, user_id: str, project_id: str, position: Dict[str, float]):
        """Update user cursor position."""
        try:
            if not user_id or not project_id:
                return
            
            # Find session and update cursor
            for session in self.active_sessions.values():
                if session.user_id == user_id and session.project_id == project_id:
                    session.cursor_position = position
                    session.last_activity = datetime.now()
                    break
            
            # Broadcast cursor update to other users
            await self._broadcast_to_project(project_id, {
                "type": "cursor_update",
                "user_id": user_id,
                "position": position
            }, exclude_user=user_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update cursor: {e}")
    
    async def _start_editing(self, websocket, user_id: str, project_id: str, element_id: str):
        """Start editing an element."""
        try:
            if not user_id or not project_id:
                return
            
            # Find session and update editing status
            for session in self.active_sessions.values():
                if session.user_id == user_id and session.project_id == project_id:
                    session.is_editing = True
                    session.editing_element = element_id
                    session.last_activity = datetime.now()
                    break
            
            # Broadcast editing start to other users
            await self._broadcast_to_project(project_id, {
                "type": "edit_start",
                "user_id": user_id,
                "element_id": element_id
            }, exclude_user=user_id)
            
        except Exception as e:
            self.logger.error(f"Failed to start editing: {e}")
    
    async def _end_editing(self, websocket, user_id: str, project_id: str, element_id: str):
        """End editing an element."""
        try:
            if not user_id or not project_id:
                return
            
            # Find session and update editing status
            for session in self.active_sessions.values():
                if session.user_id == user_id and session.project_id == project_id:
                    session.is_editing = False
                    session.editing_element = None
                    session.last_activity = datetime.now()
                    break
            
            # Broadcast editing end to other users
            await self._broadcast_to_project(project_id, {
                "type": "edit_end",
                "user_id": user_id,
                "element_id": element_id
            }, exclude_user=user_id)
            
        except Exception as e:
            self.logger.error(f"Failed to end editing: {e}")
    
    async def _broadcast_to_project(self, project_id: str, message: Dict[str, Any], 
                                  exclude_session: str = None, exclude_user: str = None):
        """Broadcast message to all project subscribers."""
        try:
            if project_id not in self.project_subscribers:
                return
            
            for session_id in self.project_subscribers[project_id]:
                if exclude_session and session_id == exclude_session:
                    continue
                
                if exclude_user:
                    session = self.active_sessions.get(session_id)
                    if session and session.user_id == exclude_user:
                        continue
                
                websocket = self.websocket_connections.get(session_id)
                if websocket:
                    try:
                        await websocket.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        # Remove closed connection
                        self.project_subscribers[project_id].discard(session_id)
                        if session_id in self.active_sessions:
                            del self.active_sessions[session_id]
                        if session_id in self.websocket_connections:
                            del self.websocket_connections[session_id]
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast to project: {e}")

class TeamCollaborationSystem:
    """Main team collaboration system."""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.user_manager = UserManager(db_path)
        self.project_manager = ProjectManager(db_path)
        self.comment_manager = CommentManager(db_path)
        self.version_manager = VersionManager(db_path)
        self.realtime_collaboration = RealTimeCollaboration()
        
        self.logger = structlog.get_logger("team_collaboration_system")
    
    async def start(self, websocket_host: str = "localhost", websocket_port: int = 8765):
        """Start the collaboration system."""
        try:
            # Start WebSocket server
            await self.realtime_collaboration.start_websocket_server(websocket_host, websocket_port)
            
            self.logger.info("Team collaboration system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start collaboration system: {e}")
            raise
    
    async def create_user(self, username: str, email: str, display_name: str, 
                         avatar_url: str = None, role: UserRole = UserRole.VIEWER) -> User:
        """Create a new user."""
        return await self.user_manager.create_user(username, email, display_name, avatar_url, role)
    
    async def create_project(self, name: str, description: str, owner_id: str, 
                           video_path: str = None, tags: List[str] = None) -> Project:
        """Create a new project."""
        return await self.project_manager.create_project(name, description, owner_id, video_path, tags)
    
    async def add_project_member(self, project_id: str, user_id: str, role: UserRole) -> bool:
        """Add a member to a project."""
        return await self.project_manager.add_project_member(project_id, user_id, role)
    
    async def add_comment(self, project_id: str, user_id: str, content: str, 
                         element_type: str, element_id: str, timestamp: float,
                         position: Dict[str, float], mentions: List[str] = None) -> Comment:
        """Add a comment to a project element."""
        return await self.comment_manager.add_comment(
            project_id, user_id, content, element_type, element_id, timestamp, position, mentions
        )
    
    async def create_version(self, project_id: str, name: str, description: str,
                           created_by: str, changes: List[Dict[str, Any]] = None,
                           file_path: str = None) -> ProjectVersion:
        """Create a new project version."""
        return await self.version_manager.create_version(
            project_id, name, description, created_by, changes, file_path
        )
    
    async def get_project_collaborators(self, project_id: str) -> List[User]:
        """Get all collaborators for a project."""
        try:
            members = await self.project_manager.get_project_members(project_id)
            collaborators = []
            
            for member in members:
                user = await self.user_manager.get_user(member.user_id)
                if user:
                    collaborators.append(user)
            
            return collaborators
            
        except Exception as e:
            self.logger.error(f"Failed to get project collaborators: {e}")
            return []
    
    async def get_project_activity(self, project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent project activity."""
        try:
            activity = []
            
            # Get recent comments
            comments = await self.comment_manager.get_comments(project_id)
            for comment in comments[-10:]:  # Last 10 comments
                user = await self.user_manager.get_user(comment.user_id)
                activity.append({
                    "type": "comment",
                    "timestamp": comment.created_at,
                    "user": user.display_name if user else "Unknown",
                    "content": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content
                })
            
            # Get recent versions
            versions = await self.version_manager.get_versions(project_id)
            for version in versions[-5:]:  # Last 5 versions
                user = await self.user_manager.get_user(version.created_by)
                activity.append({
                    "type": "version",
                    "timestamp": version.created_at,
                    "user": user.display_name if user else "Unknown",
                    "content": f"Created version {version.version_number}: {version.name}"
                })
            
            # Sort by timestamp
            activity.sort(key=lambda x: x["timestamp"], reverse=True)
            return activity[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get project activity: {e}")
            return []

# Global team collaboration system instance
team_collaboration_system = TeamCollaborationSystem()

# Export classes
__all__ = [
    "TeamCollaborationSystem",
    "UserManager",
    "ProjectManager", 
    "CommentManager",
    "VersionManager",
    "RealTimeCollaboration",
    "User",
    "Project",
    "ProjectMember",
    "Comment",
    "ProjectVersion",
    "CollaborationSession",
    "UserRole",
    "ProjectStatus",
    "CollaborationEvent",
    "team_collaboration_system"
]


