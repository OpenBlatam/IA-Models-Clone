"""
Metaverse AR/VR Service for Gamma App
====================================

Advanced service for Metaverse, Augmented Reality, and Virtual Reality
capabilities including 3D content creation, spatial computing, and
immersive experiences.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import math
from enum import Enum

logger = logging.getLogger(__name__)

class VRDeviceType(str, Enum):
    """Types of VR devices supported."""
    OCULUS_QUEST = "oculus_quest"
    OCULUS_RIFT = "oculus_rift"
    HTC_VIVE = "htc_vive"
    VALVE_INDEX = "valve_index"
    PLAYSTATION_VR = "playstation_vr"
    WINDOWS_MR = "windows_mr"
    CARDBOARD = "cardboard"
    GEAR_VR = "gear_vr"

class ARPlatform(str, Enum):
    """AR platforms supported."""
    ARKIT = "arkit"
    ARCORE = "arcore"
    WINDOWS_MR = "windows_mr"
    HOLOLENS = "hololens"
    MAGIC_LEAP = "magic_leap"
    WEBXR = "webxr"

class ContentType(str, Enum):
    """Types of 3D content."""
    MODEL_3D = "3d_model"
    ANIMATION = "animation"
    ENVIRONMENT = "environment"
    AVATAR = "avatar"
    INTERACTIVE_OBJECT = "interactive_object"
    UI_ELEMENT = "ui_element"
    AUDIO_SPATIAL = "audio_spatial"
    HAPTIC_FEEDBACK = "haptic_feedback"

class InteractionType(str, Enum):
    """Types of interactions in VR/AR."""
    GRAB = "grab"
    POINT = "point"
    GESTURE = "gesture"
    VOICE = "voice"
    EYE_TRACKING = "eye_tracking"
    HAND_TRACKING = "hand_tracking"
    CONTROLLER = "controller"
    TOUCH = "touch"

@dataclass
class VRDevice:
    """VR device information."""
    device_id: str
    device_type: VRDeviceType
    name: str
    resolution: Tuple[int, int]
    refresh_rate: int
    field_of_view: float
    tracking_type: str
    controllers: List[str]
    is_connected: bool = False
    battery_level: Optional[int] = None
    last_seen: datetime = field(default_factory=datetime.now)

@dataclass
class ARSession:
    """AR session information."""
    session_id: str
    platform: ARPlatform
    device_info: Dict[str, Any]
    tracking_state: str
    lighting_estimation: Dict[str, Any]
    plane_detection: List[Dict[str, Any]]
    anchors: List[str]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VR3DContent:
    """3D content for VR/AR."""
    content_id: str
    content_type: ContentType
    name: str
    description: str
    file_path: str
    file_format: str
    file_size: int
    dimensions: Tuple[float, float, float]
    vertices_count: int
    textures: List[str]
    materials: List[str]
    animations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VRScene:
    """VR/AR scene definition."""
    scene_id: str
    name: str
    description: str
    content_objects: List[str]
    lighting: Dict[str, Any]
    physics: Dict[str, Any]
    audio: Dict[str, Any]
    interactions: List[Dict[str, Any]]
    environment_settings: Dict[str, Any]
    is_published: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VRInteraction:
    """VR/AR interaction definition."""
    interaction_id: str
    interaction_type: InteractionType
    target_object: str
    trigger_conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    feedback: Dict[str, Any]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VRUser:
    """VR/AR user information."""
    user_id: str
    avatar_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    hand_positions: Dict[str, Tuple[float, float, float]]
    eye_tracking: Dict[str, Any]
    voice_commands: List[str]
    interaction_history: List[str]
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)

class MetaverseARVRService:
    """Service for Metaverse, AR, and VR capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vr_devices: Dict[str, VRDevice] = {}
        self.ar_sessions: Dict[str, ARSession] = {}
        self.vr_content: Dict[str, VR3DContent] = {}
        self.vr_scenes: Dict[str, VRScene] = {}
        self.vr_interactions: Dict[str, VRInteraction] = {}
        self.vr_users: Dict[str, VRUser] = {}
        self.active_scenes: Dict[str, List[str]] = {}  # scene_id -> user_ids
        
        # Initialize default content
        self._initialize_default_content()
        
        logger.info("MetaverseARVRService initialized")
    
    async def register_vr_device(self, device_info: Dict[str, Any]) -> str:
        """Register a VR device."""
        try:
            device_id = str(uuid.uuid4())
            device = VRDevice(
                device_id=device_id,
                device_type=VRDeviceType(device_info.get("device_type", "oculus_quest")),
                name=device_info.get("name", "Unknown VR Device"),
                resolution=tuple(device_info.get("resolution", [1920, 1080])),
                refresh_rate=device_info.get("refresh_rate", 90),
                field_of_view=device_info.get("field_of_view", 110.0),
                tracking_type=device_info.get("tracking_type", "inside_out"),
                controllers=device_info.get("controllers", []),
                is_connected=True
            )
            
            self.vr_devices[device_id] = device
            logger.info(f"VR device registered: {device_id}")
            return device_id
            
        except Exception as e:
            logger.error(f"Error registering VR device: {e}")
            raise
    
    async def start_ar_session(self, platform: ARPlatform, device_info: Dict[str, Any]) -> str:
        """Start an AR session."""
        try:
            session_id = str(uuid.uuid4())
            session = ARSession(
                session_id=session_id,
                platform=platform,
                device_info=device_info,
                tracking_state="initializing",
                lighting_estimation={},
                plane_detection=[],
                anchors=[]
            )
            
            self.ar_sessions[session_id] = session
            
            # Simulate session initialization
            await asyncio.sleep(1)
            session.tracking_state = "tracking"
            
            logger.info(f"AR session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting AR session: {e}")
            raise
    
    async def upload_3d_content(self, content_info: Dict[str, Any]) -> str:
        """Upload 3D content for VR/AR."""
        try:
            content_id = str(uuid.uuid4())
            content = VR3DContent(
                content_id=content_id,
                content_type=ContentType(content_info.get("content_type", "3d_model")),
                name=content_info.get("name", "Untitled 3D Content"),
                description=content_info.get("description", ""),
                file_path=content_info.get("file_path", ""),
                file_format=content_info.get("file_format", "gltf"),
                file_size=content_info.get("file_size", 0),
                dimensions=tuple(content_info.get("dimensions", [1.0, 1.0, 1.0])),
                vertices_count=content_info.get("vertices_count", 0),
                textures=content_info.get("textures", []),
                materials=content_info.get("materials", []),
                animations=content_info.get("animations", [])
            )
            
            self.vr_content[content_id] = content
            logger.info(f"3D content uploaded: {content_id}")
            return content_id
            
        except Exception as e:
            logger.error(f"Error uploading 3D content: {e}")
            raise
    
    async def create_vr_scene(self, scene_info: Dict[str, Any]) -> str:
        """Create a VR/AR scene."""
        try:
            scene_id = str(uuid.uuid4())
            scene = VRScene(
                scene_id=scene_id,
                name=scene_info.get("name", "Untitled Scene"),
                description=scene_info.get("description", ""),
                content_objects=scene_info.get("content_objects", []),
                lighting=scene_info.get("lighting", {
                    "ambient_light": {"color": [1.0, 1.0, 1.0], "intensity": 0.3},
                    "directional_light": {"color": [1.0, 1.0, 1.0], "intensity": 1.0, "direction": [0, -1, 0]}
                }),
                physics=scene_info.get("physics", {
                    "gravity": [0, -9.81, 0],
                    "collision_detection": True,
                    "physics_engine": "bullet"
                }),
                audio=scene_info.get("audio", {
                    "spatial_audio": True,
                    "reverb": "medium",
                    "ambient_sound": None
                }),
                interactions=scene_info.get("interactions", []),
                environment_settings=scene_info.get("environment_settings", {
                    "skybox": "default",
                    "fog": False,
                    "weather": "clear"
                })
            )
            
            self.vr_scenes[scene_id] = scene
            logger.info(f"VR scene created: {scene_id}")
            return scene_id
            
        except Exception as e:
            logger.error(f"Error creating VR scene: {e}")
            raise
    
    async def add_interaction_to_scene(self, scene_id: str, interaction_info: Dict[str, Any]) -> str:
        """Add interaction to a VR scene."""
        try:
            if scene_id not in self.vr_scenes:
                raise ValueError(f"Scene {scene_id} not found")
            
            interaction_id = str(uuid.uuid4())
            interaction = VRInteraction(
                interaction_id=interaction_id,
                interaction_type=InteractionType(interaction_info.get("interaction_type", "grab")),
                target_object=interaction_info.get("target_object", ""),
                trigger_conditions=interaction_info.get("trigger_conditions", {}),
                actions=interaction_info.get("actions", []),
                feedback=interaction_info.get("feedback", {})
            )
            
            self.vr_interactions[interaction_id] = interaction
            self.vr_scenes[scene_id].interactions.append(interaction_id)
            
            logger.info(f"Interaction added to scene {scene_id}: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error adding interaction to scene: {e}")
            raise
    
    async def join_vr_scene(self, scene_id: str, user_id: str, avatar_info: Dict[str, Any]) -> bool:
        """Join a VR scene."""
        try:
            if scene_id not in self.vr_scenes:
                raise ValueError(f"Scene {scene_id} not found")
            
            # Create or update VR user
            if user_id not in self.vr_users:
                self.vr_users[user_id] = VRUser(
                    user_id=user_id,
                    avatar_id=avatar_info.get("avatar_id", "default"),
                    position=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    hand_positions={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
                    eye_tracking={},
                    voice_commands=[],
                    interaction_history=[]
                )
            
            # Add user to scene
            if scene_id not in self.active_scenes:
                self.active_scenes[scene_id] = []
            
            if user_id not in self.active_scenes[scene_id]:
                self.active_scenes[scene_id].append(user_id)
            
            logger.info(f"User {user_id} joined VR scene {scene_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining VR scene: {e}")
            return False
    
    async def update_user_position(self, user_id: str, position: Tuple[float, float, float], 
                                 rotation: Tuple[float, float, float, float]) -> bool:
        """Update user position in VR."""
        try:
            if user_id not in self.vr_users:
                return False
            
            self.vr_users[user_id].position = position
            self.vr_users[user_id].rotation = rotation
            self.vr_users[user_id].last_activity = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating user position: {e}")
            return False
    
    async def process_hand_tracking(self, user_id: str, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process hand tracking data."""
        try:
            if user_id not in self.vr_users:
                return {"error": "User not found"}
            
            # Update hand positions
            if "left_hand" in hand_data:
                self.vr_users[user_id].hand_positions["left"] = tuple(hand_data["left_hand"]["position"])
            if "right_hand" in hand_data:
                self.vr_users[user_id].hand_positions["right"] = tuple(hand_data["right_hand"]["position"])
            
            # Process gestures
            gestures = []
            if hand_data.get("left_hand", {}).get("gesture") == "point":
                gestures.append("left_point")
            if hand_data.get("right_hand", {}).get("gesture") == "grab":
                gestures.append("right_grab")
            
            return {
                "gestures_detected": gestures,
                "hand_positions": self.vr_users[user_id].hand_positions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing hand tracking: {e}")
            return {"error": str(e)}
    
    async def process_voice_command(self, user_id: str, command: str) -> Dict[str, Any]:
        """Process voice command in VR."""
        try:
            if user_id not in self.vr_users:
                return {"error": "User not found"}
            
            # Add to voice commands history
            self.vr_users[user_id].voice_commands.append(command)
            
            # Process command
            command_lower = command.lower()
            response = {"command": command, "response": "Command processed"}
            
            if "move" in command_lower:
                response["action"] = "movement"
            elif "grab" in command_lower:
                response["action"] = "interaction"
            elif "teleport" in command_lower:
                response["action"] = "teleportation"
            elif "menu" in command_lower:
                response["action"] = "ui_interaction"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return {"error": str(e)}
    
    async def get_scene_users(self, scene_id: str) -> List[Dict[str, Any]]:
        """Get users in a VR scene."""
        try:
            if scene_id not in self.active_scenes:
                return []
            
            users = []
            for user_id in self.active_scenes[scene_id]:
                if user_id in self.vr_users:
                    user = self.vr_users[user_id]
                    users.append({
                        "user_id": user_id,
                        "avatar_id": user.avatar_id,
                        "position": user.position,
                        "rotation": user.rotation,
                        "last_activity": user.last_activity.isoformat()
                    })
            
            return users
            
        except Exception as e:
            logger.error(f"Error getting scene users: {e}")
            return []
    
    async def get_available_content(self, content_type: Optional[ContentType] = None) -> List[Dict[str, Any]]:
        """Get available 3D content."""
        try:
            content_list = []
            for content in self.vr_content.values():
                if content_type is None or content.content_type == content_type:
                    content_list.append({
                        "content_id": content.content_id,
                        "name": content.name,
                        "description": content.description,
                        "content_type": content.content_type.value,
                        "file_format": content.file_format,
                        "file_size": content.file_size,
                        "dimensions": content.dimensions,
                        "created_at": content.created_at.isoformat()
                    })
            
            return content_list
            
        except Exception as e:
            logger.error(f"Error getting available content: {e}")
            return []
    
    async def get_scene_info(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Get VR scene information."""
        try:
            if scene_id not in self.vr_scenes:
                return None
            
            scene = self.vr_scenes[scene_id]
            users = await self.get_scene_users(scene_id)
            
            return {
                "scene_id": scene.scene_id,
                "name": scene.name,
                "description": scene.description,
                "content_objects": scene.content_objects,
                "lighting": scene.lighting,
                "physics": scene.physics,
                "audio": scene.audio,
                "interactions": scene.interactions,
                "environment_settings": scene.environment_settings,
                "active_users": len(users),
                "users": users,
                "is_published": scene.is_published,
                "created_at": scene.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting scene info: {e}")
            return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get Metaverse AR/VR service statistics."""
        try:
            total_content = len(self.vr_content)
            total_scenes = len(self.vr_scenes)
            total_users = len(self.vr_users)
            active_sessions = len([s for s in self.ar_sessions.values() if s.is_active])
            connected_devices = len([d for d in self.vr_devices.values() if d.is_connected])
            
            # Content type distribution
            content_type_stats = {}
            for content in self.vr_content.values():
                content_type = content.content_type.value
                content_type_stats[content_type] = content_type_stats.get(content_type, 0) + 1
            
            # Platform distribution
            platform_stats = {}
            for session in self.ar_sessions.values():
                platform = session.platform.value
                platform_stats[platform] = platform_stats.get(platform, 0) + 1
            
            return {
                "total_vr_devices": len(self.vr_devices),
                "connected_devices": connected_devices,
                "total_ar_sessions": len(self.ar_sessions),
                "active_ar_sessions": active_sessions,
                "total_3d_content": total_content,
                "total_vr_scenes": total_scenes,
                "total_vr_users": total_users,
                "content_type_distribution": content_type_stats,
                "platform_distribution": platform_stats,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def _initialize_default_content(self):
        """Initialize default 3D content."""
        try:
            # Default 3D models
            default_models = [
                {
                    "content_type": "3d_model",
                    "name": "Default Cube",
                    "description": "A simple cube for testing",
                    "file_path": "/default/cube.gltf",
                    "file_format": "gltf",
                    "file_size": 1024,
                    "dimensions": [1.0, 1.0, 1.0],
                    "vertices_count": 24
                },
                {
                    "content_type": "3d_model",
                    "name": "Default Sphere",
                    "description": "A simple sphere for testing",
                    "file_path": "/default/sphere.gltf",
                    "file_format": "gltf",
                    "file_size": 2048,
                    "dimensions": [1.0, 1.0, 1.0],
                    "vertices_count": 48
                },
                {
                    "content_type": "avatar",
                    "name": "Default Avatar",
                    "description": "Basic human avatar",
                    "file_path": "/default/avatar.gltf",
                    "file_format": "gltf",
                    "file_size": 5120,
                    "dimensions": [0.6, 1.8, 0.4],
                    "vertices_count": 120
                }
            ]
            
            for model_info in default_models:
                content_id = str(uuid.uuid4())
                content = VR3DContent(
                    content_id=content_id,
                    content_type=ContentType(model_info["content_type"]),
                    name=model_info["name"],
                    description=model_info["description"],
                    file_path=model_info["file_path"],
                    file_format=model_info["file_format"],
                    file_size=model_info["file_size"],
                    dimensions=tuple(model_info["dimensions"]),
                    vertices_count=model_info["vertices_count"],
                    textures=[],
                    materials=[],
                    animations=[]
                )
                self.vr_content[content_id] = content
            
            logger.info("Default 3D content initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default content: {e}")