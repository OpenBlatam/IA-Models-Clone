"""
Metaverse Integration
===================

Advanced metaverse integration for 3D document visualization and VR/AR collaboration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import base64
import numpy as np
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class MetaversePlatform(str, Enum):
    """Metaverse platform."""
    HORIZON_WORLDS = "horizon_worlds"
    VRCHAT = "vrchat"
    REC_ROOM = "rec_room"
    SPATIAL = "spatial"
    GATHER = "gather"
    CUSTOM = "custom"


class VRDeviceType(str, Enum):
    """VR device type."""
    OCULUS_QUEST = "oculus_quest"
    HTC_VIVE = "htc_vive"
    VALVE_INDEX = "valve_index"
    PLAYSTATION_VR = "playstation_vr"
    MIXED_REALITY = "mixed_reality"
    AR_GLASSES = "ar_glasses"


class Document3DType(str, Enum):
    """3D document type."""
    FLOATING_PANEL = "floating_panel"
    HOLOGRAPHIC_DISPLAY = "holographic_display"
    INTERACTIVE_CUBE = "interactive_cube"
    SPATIAL_WALL = "spatial_wall"
    VIRTUAL_DESK = "virtual_desk"
    IMMERSIVE_ROOM = "immersive_room"


@dataclass
class Vector3D:
    """3D vector."""
    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    """Quaternion for 3D rotation."""
    x: float
    y: float
    z: float
    w: float


@dataclass
class Transform3D:
    """3D transform."""
    position: Vector3D
    rotation: Quaternion
    scale: Vector3D


@dataclass
class MetaverseUser:
    """Metaverse user."""
    user_id: str
    username: str
    avatar_id: str
    platform: MetaversePlatform
    device_type: VRDeviceType
    position: Vector3D
    rotation: Quaternion
    is_online: bool = True
    last_seen: datetime = field(default_factory=datetime.now)
    permissions: List[str] = field(default_factory=list)


@dataclass
class Document3D:
    """3D document representation."""
    document_3d_id: str
    document_id: str
    document_type: Document3DType
    transform: Transform3D
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class MetaverseSession:
    """Metaverse collaboration session."""
    session_id: str
    world_id: str
    document_id: str
    participants: List[MetaverseUser] = field(default_factory=list)
    documents_3d: List[Document3D] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)


class MetaverseService:
    """Metaverse integration service."""
    
    def __init__(self):
        self.metaverse_sessions: Dict[str, MetaverseSession] = {}
        self.metaverse_users: Dict[str, MetaverseUser] = {}
        self.document_3d_objects: Dict[str, Document3D] = {}
        self.virtual_worlds: Dict[str, Dict[str, Any]] = {}
        self.avatar_templates: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_metaverse_platforms()
        self._initialize_avatar_templates()
    
    def _initialize_metaverse_platforms(self):
        """Initialize metaverse platform configurations."""
        
        self.virtual_worlds = {
            "document_collaboration_space": {
                "world_id": "doc_collab_001",
                "name": "Document Collaboration Space",
                "platform": MetaversePlatform.HORIZON_WORLDS,
                "max_users": 50,
                "features": ["3d_documents", "voice_chat", "gesture_controls", "spatial_audio"],
                "environment": "modern_office",
                "lighting": "natural",
                "ambient_sounds": ["office_ambience", "keyboard_typing"]
            },
            "presentation_theater": {
                "world_id": "presentation_001",
                "name": "Presentation Theater",
                "platform": MetaversePlatform.SPATIAL,
                "max_users": 100,
                "features": ["large_screens", "audience_seating", "presentation_tools"],
                "environment": "theater",
                "lighting": "stage",
                "ambient_sounds": ["audience_chatter", "presentation_audio"]
            },
            "creative_workspace": {
                "world_id": "creative_001",
                "name": "Creative Workspace",
                "platform": MetaversePlatform.REC_ROOM,
                "max_users": 25,
                "features": ["3d_modeling", "brainstorming_tools", "whiteboards"],
                "environment": "creative_studio",
                "lighting": "creative",
                "ambient_sounds": ["creative_music", "brainstorming"]
            }
        }
    
    def _initialize_avatar_templates(self):
        """Initialize avatar templates."""
        
        self.avatar_templates = {
            "professional": {
                "avatar_id": "prof_001",
                "name": "Professional",
                "appearance": {
                    "gender": "neutral",
                    "clothing": "business_casual",
                    "accessories": ["glasses", "watch"],
                    "colors": ["navy", "white", "gray"]
                },
                "animations": ["typing", "pointing", "nodding", "handshake"]
            },
            "creative": {
                "avatar_id": "creative_001",
                "name": "Creative",
                "appearance": {
                    "gender": "neutral",
                    "clothing": "casual_creative",
                    "accessories": ["headphones", "sketchbook"],
                    "colors": ["bright", "colorful", "artistic"]
                },
                "animations": ["sketching", "thinking", "excited", "collaborating"]
            },
            "technical": {
                "avatar_id": "tech_001",
                "name": "Technical",
                "appearance": {
                    "gender": "neutral",
                    "clothing": "tech_casual",
                    "accessories": ["vr_headset", "tablet"],
                    "colors": ["dark", "tech", "minimal"]
                },
                "animations": ["coding", "analyzing", "debugging", "explaining"]
            }
        }
    
    async def create_metaverse_user(
        self,
        user_id: str,
        username: str,
        platform: MetaversePlatform,
        device_type: VRDeviceType,
        avatar_template: str = "professional"
    ) -> MetaverseUser:
        """Create a metaverse user."""
        
        if avatar_template not in self.avatar_templates:
            avatar_template = "professional"
        
        avatar_config = self.avatar_templates[avatar_template]
        
        user = MetaverseUser(
            user_id=user_id,
            username=username,
            avatar_id=avatar_config["avatar_id"],
            platform=platform,
            device_type=device_type,
            position=Vector3D(0, 0, 0),
            rotation=Quaternion(0, 0, 0, 1)
        )
        
        self.metaverse_users[user_id] = user
        
        logger.info(f"Created metaverse user: {username} on {platform.value}")
        
        return user
    
    async def create_metaverse_session(
        self,
        document_id: str,
        world_id: str,
        creator_user_id: str,
        settings: Dict[str, Any] = None
    ) -> MetaverseSession:
        """Create a metaverse collaboration session."""
        
        if world_id not in self.virtual_worlds:
            raise ValueError(f"Virtual world {world_id} not found")
        
        world_config = self.virtual_worlds[world_id]
        
        # Get creator user
        if creator_user_id not in self.metaverse_users:
            raise ValueError(f"Metaverse user {creator_user_id} not found")
        
        creator = self.metaverse_users[creator_user_id]
        
        session = MetaverseSession(
            session_id=str(uuid4()),
            world_id=world_id,
            document_id=document_id,
            participants=[creator],
            settings=settings or {}
        )
        
        self.metaverse_sessions[session.session_id] = session
        
        # Create initial 3D document
        await self._create_initial_3d_document(session, document_id)
        
        logger.info(f"Created metaverse session: {session.session_id} in world {world_id}")
        
        return session
    
    async def _create_initial_3d_document(self, session: MetaverseSession, document_id: str):
        """Create initial 3D document representation."""
        
        # Create floating panel document
        document_3d = Document3D(
            document_3d_id=str(uuid4()),
            document_id=document_id,
            document_type=Document3DType.FLOATING_PANEL,
            transform=Transform3D(
                position=Vector3D(0, 1.5, -2),
                rotation=Quaternion(0, 0, 0, 1),
                scale=Vector3D(2, 1.5, 0.1)
            ),
            content="Document content will be loaded here",
            metadata={
                "interactive": True,
                "collaborative": True,
                "resizable": True,
                "movable": True
            }
        )
        
        session.documents_3d.append(document_3d)
        self.document_3d_objects[document_3d.document_3d_id] = document_3d
    
    async def join_metaverse_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """Join a metaverse session."""
        
        if session_id not in self.metaverse_sessions:
            return False
        
        if user_id not in self.metaverse_users:
            return False
        
        session = self.metaverse_sessions[session_id]
        user = self.metaverse_users[user_id]
        
        # Check if user is already in session
        if any(p.user_id == user_id for p in session.participants):
            return True
        
        # Check world capacity
        world_config = self.virtual_worlds[session.world_id]
        if len(session.participants) >= world_config["max_users"]:
            return False
        
        # Add user to session
        session.participants.append(user)
        
        # Position user in world
        await self._position_user_in_world(user, session)
        
        logger.info(f"User {user.username} joined session {session_id}")
        
        return True
    
    async def _position_user_in_world(self, user: MetaverseUser, session: MetaverseSession):
        """Position user in virtual world."""
        
        # Simple positioning logic - in production, use more sophisticated algorithms
        participant_count = len(session.participants)
        
        # Arrange users in a circle
        angle = (participant_count - 1) * (2 * math.pi / 8)  # Max 8 users in circle
        radius = 2.0
        
        user.position = Vector3D(
            x=radius * math.cos(angle),
            y=0,
            z=radius * math.sin(angle)
        )
        
        # Face center
        user.rotation = Quaternion(
            x=0,
            y=math.sin(angle / 2),
            z=0,
            w=math.cos(angle / 2)
        )
    
    async def update_user_position(
        self,
        user_id: str,
        position: Vector3D,
        rotation: Quaternion
    ) -> bool:
        """Update user position in metaverse."""
        
        if user_id not in self.metaverse_users:
            return False
        
        user = self.metaverse_users[user_id]
        user.position = position
        user.rotation = rotation
        user.last_seen = datetime.now()
        
        return True
    
    async def create_3d_document(
        self,
        session_id: str,
        document_id: str,
        document_type: Document3DType,
        transform: Transform3D,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Document3D:
        """Create 3D document representation."""
        
        if session_id not in self.metaverse_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.metaverse_sessions[session_id]
        
        document_3d = Document3D(
            document_3d_id=str(uuid4()),
            document_id=document_id,
            document_type=document_type,
            transform=transform,
            content=content,
            metadata=metadata or {}
        )
        
        session.documents_3d.append(document_3d)
        self.document_3d_objects[document_3d.document_3d_id] = document_3d
        
        logger.info(f"Created 3D document: {document_3d.document_3d_id}")
        
        return document_3d
    
    async def interact_with_3d_document(
        self,
        document_3d_id: str,
        user_id: str,
        interaction_type: str,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle interaction with 3D document."""
        
        if document_3d_id not in self.document_3d_objects:
            raise ValueError(f"3D document {document_3d_id} not found")
        
        if user_id not in self.metaverse_users:
            raise ValueError(f"User {user_id} not found")
        
        document_3d = self.document_3d_objects[document_3d_id]
        user = self.metaverse_users[user_id]
        
        # Record interaction
        interaction = {
            "interaction_id": str(uuid4()),
            "user_id": user_id,
            "username": user.username,
            "interaction_type": interaction_type,
            "interaction_data": interaction_data,
            "timestamp": datetime.now().isoformat()
        }
        
        document_3d.interactions.append(interaction)
        
        # Handle different interaction types
        if interaction_type == "select":
            return await self._handle_document_selection(document_3d, user, interaction_data)
        elif interaction_type == "edit":
            return await self._handle_document_editing(document_3d, user, interaction_data)
        elif interaction_type == "move":
            return await self._handle_document_movement(document_3d, user, interaction_data)
        elif interaction_type == "resize":
            return await self._handle_document_resize(document_3d, user, interaction_data)
        elif interaction_type == "comment":
            return await self._handle_document_comment(document_3d, user, interaction_data)
        else:
            return {"status": "unknown_interaction", "interaction": interaction}
    
    async def _handle_document_selection(
        self,
        document_3d: Document3D,
        user: MetaverseUser,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document selection."""
        
        return {
            "status": "selected",
            "document_id": document_3d.document_id,
            "user": user.username,
            "selection_data": interaction_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_document_editing(
        self,
        document_3d: Document3D,
        user: MetaverseUser,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document editing."""
        
        # Update document content
        if "content" in interaction_data:
            document_3d.content = interaction_data["content"]
        
        return {
            "status": "edited",
            "document_id": document_3d.document_id,
            "user": user.username,
            "changes": interaction_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_document_movement(
        self,
        document_3d: Document3D,
        user: MetaverseUser,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document movement."""
        
        # Update document position
        if "position" in interaction_data:
            pos = interaction_data["position"]
            document_3d.transform.position = Vector3D(pos["x"], pos["y"], pos["z"])
        
        if "rotation" in interaction_data:
            rot = interaction_data["rotation"]
            document_3d.transform.rotation = Quaternion(rot["x"], rot["y"], rot["z"], rot["w"])
        
        return {
            "status": "moved",
            "document_id": document_3d.document_id,
            "user": user.username,
            "new_transform": {
                "position": {
                    "x": document_3d.transform.position.x,
                    "y": document_3d.transform.position.y,
                    "z": document_3d.transform.position.z
                },
                "rotation": {
                    "x": document_3d.transform.rotation.x,
                    "y": document_3d.transform.rotation.y,
                    "z": document_3d.transform.rotation.z,
                    "w": document_3d.transform.rotation.w
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_document_resize(
        self,
        document_3d: Document3D,
        user: MetaverseUser,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document resizing."""
        
        # Update document scale
        if "scale" in interaction_data:
            scale = interaction_data["scale"]
            document_3d.transform.scale = Vector3D(scale["x"], scale["y"], scale["z"])
        
        return {
            "status": "resized",
            "document_id": document_3d.document_id,
            "user": user.username,
            "new_scale": {
                "x": document_3d.transform.scale.x,
                "y": document_3d.transform.scale.y,
                "z": document_3d.transform.scale.z
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_document_comment(
        self,
        document_3d: Document3D,
        user: MetaverseUser,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document commenting."""
        
        comment = {
            "comment_id": str(uuid4()),
            "user_id": user.user_id,
            "username": user.username,
            "content": interaction_data.get("content", ""),
            "position": interaction_data.get("position", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "commented",
            "document_id": document_3d.document_id,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current session state."""
        
        if session_id not in self.metaverse_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.metaverse_sessions[session_id]
        world_config = self.virtual_worlds[session.world_id]
        
        return {
            "session_id": session_id,
            "world_id": session.world_id,
            "world_name": world_config["name"],
            "platform": world_config["platform"].value,
            "participants": [
                {
                    "user_id": p.user_id,
                    "username": p.username,
                    "avatar_id": p.avatar_id,
                    "position": {
                        "x": p.position.x,
                        "y": p.position.y,
                        "z": p.position.z
                    },
                    "rotation": {
                        "x": p.rotation.x,
                        "y": p.rotation.y,
                        "z": p.rotation.z,
                        "w": p.rotation.w
                    },
                    "is_online": p.is_online,
                    "last_seen": p.last_seen.isoformat()
                }
                for p in session.participants
            ],
            "documents_3d": [
                {
                    "document_3d_id": d.document_3d_id,
                    "document_id": d.document_id,
                    "document_type": d.document_type.value,
                    "transform": {
                        "position": {
                            "x": d.transform.position.x,
                            "y": d.transform.position.y,
                            "z": d.transform.position.z
                        },
                        "rotation": {
                            "x": d.transform.rotation.x,
                            "y": d.transform.rotation.y,
                            "z": d.transform.rotation.z,
                            "w": d.transform.rotation.w
                        },
                        "scale": {
                            "x": d.transform.scale.x,
                            "y": d.transform.scale.y,
                            "z": d.transform.scale.z
                        }
                    },
                    "content": d.content,
                    "metadata": d.metadata,
                    "interaction_count": len(d.interactions)
                }
                for d in session.documents_3d
            ],
            "session_settings": session.settings,
            "started_at": session.started_at.isoformat(),
            "is_active": session.is_active
        }
    
    async def leave_metaverse_session(self, session_id: str, user_id: str) -> bool:
        """Leave a metaverse session."""
        
        if session_id not in self.metaverse_sessions:
            return False
        
        session = self.metaverse_sessions[session_id]
        
        # Remove user from participants
        session.participants = [p for p in session.participants if p.user_id != user_id]
        
        # If no participants left, end session
        if not session.participants:
            session.is_active = False
        
        logger.info(f"User {user_id} left session {session_id}")
        
        return True
    
    async def get_metaverse_analytics(self) -> Dict[str, Any]:
        """Get metaverse analytics."""
        
        total_sessions = len(self.metaverse_sessions)
        active_sessions = len([s for s in self.metaverse_sessions.values() if s.is_active])
        total_users = len(self.metaverse_users)
        online_users = len([u for u in self.metaverse_users.values() if u.is_online])
        
        # Platform distribution
        platform_distribution = defaultdict(int)
        for user in self.metaverse_users.values():
            platform_distribution[user.platform.value] += 1
        
        # Device type distribution
        device_distribution = defaultdict(int)
        for user in self.metaverse_users.values():
            device_distribution[user.device_type.value] += 1
        
        # Document type distribution
        document_type_distribution = defaultdict(int)
        for doc_3d in self.document_3d_objects.values():
            document_type_distribution[doc_3d.document_type.value] += 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_users": total_users,
            "online_users": online_users,
            "platform_distribution": dict(platform_distribution),
            "device_distribution": dict(device_distribution),
            "document_type_distribution": dict(document_type_distribution),
            "total_3d_documents": len(self.document_3d_objects),
            "total_interactions": sum(len(doc.interactions) for doc in self.document_3d_objects.values()),
            "average_session_duration": self._calculate_average_session_duration()
        }
    
    def _calculate_average_session_duration(self) -> float:
        """Calculate average session duration."""
        
        completed_sessions = [
            s for s in self.metaverse_sessions.values()
            if not s.is_active and s.started_at
        ]
        
        if not completed_sessions:
            return 0.0
        
        total_duration = sum(
            (datetime.now() - s.started_at).total_seconds()
            for s in completed_sessions
        )
        
        return total_duration / len(completed_sessions)
    
    async def create_immersive_presentation(
        self,
        session_id: str,
        document_id: str,
        presentation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create immersive presentation in metaverse."""
        
        if session_id not in self.metaverse_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.metaverse_sessions[session_id]
        
        # Create presentation theater environment
        presentation_3d = Document3D(
            document_3d_id=str(uuid4()),
            document_id=document_id,
            document_type=Document3DType.IMMERSIVE_ROOM,
            transform=Transform3D(
                position=Vector3D(0, 0, 0),
                rotation=Quaternion(0, 0, 0, 1),
                scale=Vector3D(10, 6, 8)  # Large presentation space
            ),
            content=presentation_data.get("content", ""),
            metadata={
                "presentation_mode": True,
                "slides": presentation_data.get("slides", []),
                "current_slide": 0,
                "auto_advance": presentation_data.get("auto_advance", False),
                "presenter_controls": True
            }
        )
        
        session.documents_3d.append(presentation_3d)
        self.document_3d_objects[presentation_3d.document_3d_id] = presentation_3d
        
        return {
            "presentation_id": presentation_3d.document_3d_id,
            "session_id": session_id,
            "document_id": document_id,
            "presentation_type": "immersive_room",
            "created_at": datetime.now().isoformat()
        }
    
    async def advance_presentation_slide(
        self,
        presentation_id: str,
        user_id: str,
        direction: str = "next"
    ) -> Dict[str, Any]:
        """Advance presentation slide."""
        
        if presentation_id not in self.document_3d_objects:
            raise ValueError(f"Presentation {presentation_id} not found")
        
        presentation = self.document_3d_objects[presentation_id]
        
        if not presentation.metadata.get("presentation_mode"):
            raise ValueError("Document is not in presentation mode")
        
        slides = presentation.metadata.get("slides", [])
        current_slide = presentation.metadata.get("current_slide", 0)
        
        if direction == "next" and current_slide < len(slides) - 1:
            current_slide += 1
        elif direction == "previous" and current_slide > 0:
            current_slide -= 1
        
        presentation.metadata["current_slide"] = current_slide
        
        return {
            "presentation_id": presentation_id,
            "current_slide": current_slide,
            "total_slides": len(slides),
            "slide_content": slides[current_slide] if slides else None,
            "advanced_by": user_id,
            "direction": direction,
            "timestamp": datetime.now().isoformat()
        }



























