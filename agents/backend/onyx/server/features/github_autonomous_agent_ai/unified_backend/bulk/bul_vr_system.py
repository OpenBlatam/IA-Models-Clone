"""
BUL - Business Universal Language (Virtual Reality System)
==========================================================

Advanced Virtual Reality system with immersive experiences and spatial computing.
"""

import asyncio
import logging
import json
import time
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge
import numpy as np
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
from OpenGL.arrays import vbo
import cv2
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_vr.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
VR_SESSIONS = Counter('bul_vr_sessions_total', 'Total VR sessions', ['session_type', 'status'])
VR_OBJECTS = Gauge('bul_vr_objects_total', 'Total VR objects', ['object_type'])
VR_INTERACTIONS = Counter('bul_vr_interactions_total', 'Total VR interactions', ['interaction_type'])
VR_PERFORMANCE = Histogram('bul_vr_frame_time_seconds', 'VR frame rendering time')

class VRSessionType(str, Enum):
    """VR session type enumeration."""
    IMMERSIVE = "immersive"
    COLLABORATIVE = "collaborative"
    TRAINING = "training"
    SIMULATION = "simulation"
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    THERAPY = "therapy"
    ARCHITECTURE = "architecture"

class VRObjectType(str, Enum):
    """VR object type enumeration."""
    MODEL_3D = "3d_model"
    TEXTURE = "texture"
    ANIMATION = "animation"
    SOUND = "sound"
    LIGHT = "light"
    PARTICLE = "particle"
    UI_ELEMENT = "ui_element"
    INTERACTIVE = "interactive"

class VRInteractionType(str, Enum):
    """VR interaction type enumeration."""
    GRAB = "grab"
    RELEASE = "release"
    TOUCH = "touch"
    GAZE = "gaze"
    VOICE = "voice"
    GESTURE = "gesture"
    TELEPORT = "teleport"
    SCALE = "scale"

class VRSessionStatus(str, Enum):
    """VR session status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"

# Database Models
class VRSession(Base):
    __tablename__ = "vr_sessions"
    
    id = Column(String, primary_key=True)
    session_name = Column(String, nullable=False)
    session_type = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    environment_id = Column(String)
    status = Column(String, default=VRSessionStatus.ACTIVE)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    duration = Column(Float, default=0.0)
    frame_rate = Column(Float, default=90.0)
    resolution = Column(String, default="1920x1080")
    fov = Column(Float, default=110.0)
    tracking_enabled = Column(Boolean, default=True)
    hand_tracking = Column(Boolean, default=True)
    eye_tracking = Column(Boolean, default=False)
    haptic_feedback = Column(Boolean, default=True)
    spatial_audio = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")

class VREnvironment(Base):
    __tablename__ = "vr_environments"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    environment_type = Column(String, default="custom")
    skybox_texture = Column(String)
    lighting_preset = Column(String, default="default")
    physics_enabled = Column(Boolean, default=True)
    gravity = Column(Float, default=-9.81)
    ambient_light = Column(Float, default=0.3)
    directional_light = Column(Float, default=1.0)
    fog_enabled = Column(Boolean, default=False)
    fog_density = Column(Float, default=0.01)
    fog_color = Column(String, default="#808080")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")

class VRObject(Base):
    __tablename__ = "vr_objects"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    object_type = Column(String, nullable=False)
    model_path = Column(String)
    texture_path = Column(String)
    position_x = Column(Float, default=0.0)
    position_y = Column(Float, default=0.0)
    position_z = Column(Float, default=0.0)
    rotation_x = Column(Float, default=0.0)
    rotation_y = Column(Float, default=0.0)
    rotation_z = Column(Float, default=0.0)
    scale_x = Column(Float, default=1.0)
    scale_y = Column(Float, default=1.0)
    scale_z = Column(Float, default=1.0)
    is_interactive = Column(Boolean, default=False)
    is_visible = Column(Boolean, default=True)
    physics_enabled = Column(Boolean, default=False)
    mass = Column(Float, default=1.0)
    friction = Column(Float, default=0.5)
    restitution = Column(Float, default=0.3)
    environment_id = Column(String, ForeignKey("vr_environments.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")
    
    # Relationships
    environment = relationship("VREnvironment")

class VRInteraction(Base):
    __tablename__ = "vr_interactions"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("vr_sessions.id"))
    object_id = Column(String, ForeignKey("vr_objects.id"))
    interaction_type = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    hand_position_x = Column(Float)
    hand_position_y = Column(Float)
    hand_position_z = Column(Float)
    gaze_position_x = Column(Float)
    gaze_position_y = Column(Float)
    gaze_position_z = Column(Float)
    interaction_data = Column(Text, default="{}")
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("VRSession")
    object = relationship("VRObject")

class VRUser(Base):
    __tablename__ = "vr_users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    avatar_model = Column(String)
    height = Column(Float, default=1.75)
    arm_span = Column(Float, default=1.8)
    dominant_hand = Column(String, default="right")
    comfort_settings = Column(Text, default="{}")
    accessibility_settings = Column(Text, default="{}")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# VR Configuration
VR_CONFIG = {
    "default_frame_rate": 90,
    "default_resolution": "1920x1080",
    "default_fov": 110.0,
    "max_session_duration": 3600,  # 1 hour
    "hand_tracking_confidence": 0.7,
    "eye_tracking_accuracy": 0.95,
    "haptic_intensity_range": [0.0, 1.0],
    "spatial_audio_distance": 10.0,
    "physics_timestep": 1.0/60.0,
    "collision_detection": True,
    "raycast_distance": 100.0,
    "teleport_distance": 5.0,
    "comfort_settings": {
        "snap_turning": True,
        "comfort_vignette": True,
        "height_adjustment": True,
        "boundary_guardian": True
    },
    "accessibility_settings": {
        "text_to_speech": True,
        "voice_commands": True,
        "one_handed_mode": False,
        "color_blind_support": True
    }
}

class AdvancedVRSystem:
    """Advanced Virtual Reality system with comprehensive features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Virtual Reality System",
            description="Advanced Virtual Reality system with immersive experiences and spatial computing",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # VR components
        self.active_sessions: Dict[str, VRSession] = {}
        self.environments: Dict[str, VREnvironment] = {}
        self.objects: Dict[str, VRObject] = {}
        self.websocket_connections: List[WebSocket] = []
        
        # VR rendering components
        self.pygame_initialized = False
        self.opengl_initialized = False
        self.mediapipe_hands = None
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        self.initialize_vr_components()
        
        logger.info("Advanced VR System initialized")
    
    def setup_middleware(self):
        """Setup VR middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup VR API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with VR system information."""
            return {
                "message": "BUL Virtual Reality System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Immersive VR Experiences",
                    "Spatial Computing",
                    "Hand Tracking",
                    "Eye Tracking",
                    "Haptic Feedback",
                    "Spatial Audio",
                    "Physics Simulation",
                    "Collaborative VR"
                ],
                "session_types": [session_type.value for session_type in VRSessionType],
                "object_types": [object_type.value for object_type in VRObjectType],
                "interaction_types": [interaction_type.value for interaction_type in VRInteractionType],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/sessions/create", tags=["Sessions"])
        async def create_vr_session(session_request: dict):
            """Create new VR session."""
            try:
                # Validate request
                required_fields = ["session_name", "session_type", "user_id"]
                if not all(field in session_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                session_name = session_request["session_name"]
                session_type = session_request["session_type"]
                user_id = session_request["user_id"]
                
                # Create VR session
                session = VRSession(
                    id=f"session_{int(time.time())}",
                    session_name=session_name,
                    session_type=session_type,
                    user_id=user_id,
                    environment_id=session_request.get("environment_id"),
                    frame_rate=session_request.get("frame_rate", VR_CONFIG["default_frame_rate"]),
                    resolution=session_request.get("resolution", VR_CONFIG["default_resolution"]),
                    fov=session_request.get("fov", VR_CONFIG["default_fov"]),
                    tracking_enabled=session_request.get("tracking_enabled", True),
                    hand_tracking=session_request.get("hand_tracking", True),
                    eye_tracking=session_request.get("eye_tracking", False),
                    haptic_feedback=session_request.get("haptic_feedback", True),
                    spatial_audio=session_request.get("spatial_audio", True),
                    metadata=json.dumps(session_request.get("metadata", {}))
                )
                
                self.db.add(session)
                self.db.commit()
                
                # Add to active sessions
                self.active_sessions[session.id] = session
                
                VR_SESSIONS.labels(session_type=session_type, status="active").inc()
                
                return {
                    "message": "VR session created successfully",
                    "session_id": session.id,
                    "session_name": session.session_name,
                    "session_type": session.session_type,
                    "status": session.status
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating VR session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/sessions", tags=["Sessions"])
        async def get_vr_sessions():
            """Get all VR sessions."""
            try:
                sessions = self.db.query(VRSession).all()
                
                return {
                    "sessions": [
                        {
                            "id": session.id,
                            "session_name": session.session_name,
                            "session_type": session.session_type,
                            "user_id": session.user_id,
                            "environment_id": session.environment_id,
                            "status": session.status,
                            "start_time": session.start_time.isoformat(),
                            "end_time": session.end_time.isoformat() if session.end_time else None,
                            "duration": session.duration,
                            "frame_rate": session.frame_rate,
                            "resolution": session.resolution,
                            "fov": session.fov,
                            "tracking_enabled": session.tracking_enabled,
                            "hand_tracking": session.hand_tracking,
                            "eye_tracking": session.eye_tracking,
                            "haptic_feedback": session.haptic_feedback,
                            "spatial_audio": session.spatial_audio,
                            "metadata": json.loads(session.metadata)
                        }
                        for session in sessions
                    ],
                    "total": len(sessions)
                }
                
            except Exception as e:
                logger.error(f"Error getting VR sessions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/environments/create", tags=["Environments"])
        async def create_vr_environment(environment_request: dict):
            """Create VR environment."""
            try:
                # Validate request
                required_fields = ["name"]
                if not all(field in environment_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = environment_request["name"]
                
                # Create VR environment
                environment = VREnvironment(
                    id=f"env_{int(time.time())}",
                    name=name,
                    description=environment_request.get("description"),
                    environment_type=environment_request.get("environment_type", "custom"),
                    skybox_texture=environment_request.get("skybox_texture"),
                    lighting_preset=environment_request.get("lighting_preset", "default"),
                    physics_enabled=environment_request.get("physics_enabled", True),
                    gravity=environment_request.get("gravity", -9.81),
                    ambient_light=environment_request.get("ambient_light", 0.3),
                    directional_light=environment_request.get("directional_light", 1.0),
                    fog_enabled=environment_request.get("fog_enabled", False),
                    fog_density=environment_request.get("fog_density", 0.01),
                    fog_color=environment_request.get("fog_color", "#808080"),
                    metadata=json.dumps(environment_request.get("metadata", {}))
                )
                
                self.db.add(environment)
                self.db.commit()
                
                # Add to environments cache
                self.environments[environment.id] = environment
                
                return {
                    "message": "VR environment created successfully",
                    "environment_id": environment.id,
                    "name": environment.name,
                    "environment_type": environment.environment_type
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating VR environment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/environments", tags=["Environments"])
        async def get_vr_environments():
            """Get all VR environments."""
            try:
                environments = self.db.query(VREnvironment).filter(VREnvironment.is_active == True).all()
                
                return {
                    "environments": [
                        {
                            "id": env.id,
                            "name": env.name,
                            "description": env.description,
                            "environment_type": env.environment_type,
                            "skybox_texture": env.skybox_texture,
                            "lighting_preset": env.lighting_preset,
                            "physics_enabled": env.physics_enabled,
                            "gravity": env.gravity,
                            "ambient_light": env.ambient_light,
                            "directional_light": env.directional_light,
                            "fog_enabled": env.fog_enabled,
                            "fog_density": env.fog_density,
                            "fog_color": env.fog_color,
                            "metadata": json.loads(env.metadata),
                            "created_at": env.created_at.isoformat()
                        }
                        for env in environments
                    ],
                    "total": len(environments)
                }
                
            except Exception as e:
                logger.error(f"Error getting VR environments: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/objects/create", tags=["Objects"])
        async def create_vr_object(object_request: dict):
            """Create VR object."""
            try:
                # Validate request
                required_fields = ["name", "object_type", "environment_id"]
                if not all(field in object_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = object_request["name"]
                object_type = object_request["object_type"]
                environment_id = object_request["environment_id"]
                
                # Create VR object
                vr_object = VRObject(
                    id=f"obj_{int(time.time())}",
                    name=name,
                    object_type=object_type,
                    model_path=object_request.get("model_path"),
                    texture_path=object_request.get("texture_path"),
                    position_x=object_request.get("position_x", 0.0),
                    position_y=object_request.get("position_y", 0.0),
                    position_z=object_request.get("position_z", 0.0),
                    rotation_x=object_request.get("rotation_x", 0.0),
                    rotation_y=object_request.get("rotation_y", 0.0),
                    rotation_z=object_request.get("rotation_z", 0.0),
                    scale_x=object_request.get("scale_x", 1.0),
                    scale_y=object_request.get("scale_y", 1.0),
                    scale_z=object_request.get("scale_z", 1.0),
                    is_interactive=object_request.get("is_interactive", False),
                    is_visible=object_request.get("is_visible", True),
                    physics_enabled=object_request.get("physics_enabled", False),
                    mass=object_request.get("mass", 1.0),
                    friction=object_request.get("friction", 0.5),
                    restitution=object_request.get("restitution", 0.3),
                    environment_id=environment_id,
                    metadata=json.dumps(object_request.get("metadata", {}))
                )
                
                self.db.add(vr_object)
                self.db.commit()
                
                # Add to objects cache
                self.objects[vr_object.id] = vr_object
                
                VR_OBJECTS.labels(object_type=object_type).inc()
                
                return {
                    "message": "VR object created successfully",
                    "object_id": vr_object.id,
                    "name": vr_object.name,
                    "object_type": vr_object.object_type,
                    "environment_id": vr_object.environment_id
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating VR object: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/objects", tags=["Objects"])
        async def get_vr_objects(environment_id: str = None):
            """Get VR objects."""
            try:
                query = self.db.query(VRObject)
                
                if environment_id:
                    query = query.filter(VRObject.environment_id == environment_id)
                
                objects = query.filter(VRObject.is_visible == True).all()
                
                return {
                    "objects": [
                        {
                            "id": obj.id,
                            "name": obj.name,
                            "object_type": obj.object_type,
                            "model_path": obj.model_path,
                            "texture_path": obj.texture_path,
                            "position": {
                                "x": obj.position_x,
                                "y": obj.position_y,
                                "z": obj.position_z
                            },
                            "rotation": {
                                "x": obj.rotation_x,
                                "y": obj.rotation_y,
                                "z": obj.rotation_z
                            },
                            "scale": {
                                "x": obj.scale_x,
                                "y": obj.scale_y,
                                "z": obj.scale_z
                            },
                            "is_interactive": obj.is_interactive,
                            "is_visible": obj.is_visible,
                            "physics_enabled": obj.physics_enabled,
                            "mass": obj.mass,
                            "friction": obj.friction,
                            "restitution": obj.restitution,
                            "environment_id": obj.environment_id,
                            "metadata": json.loads(obj.metadata),
                            "created_at": obj.created_at.isoformat()
                        }
                        for obj in objects
                    ],
                    "total": len(objects)
                }
                
            except Exception as e:
                logger.error(f"Error getting VR objects: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/interactions/record", tags=["Interactions"])
        async def record_vr_interaction(interaction_request: dict):
            """Record VR interaction."""
            try:
                # Validate request
                required_fields = ["session_id", "object_id", "interaction_type", "user_id"]
                if not all(field in interaction_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                session_id = interaction_request["session_id"]
                object_id = interaction_request["object_id"]
                interaction_type = interaction_request["interaction_type"]
                user_id = interaction_request["user_id"]
                
                # Create VR interaction
                interaction = VRInteraction(
                    id=f"interaction_{int(time.time())}",
                    session_id=session_id,
                    object_id=object_id,
                    interaction_type=interaction_type,
                    user_id=user_id,
                    hand_position_x=interaction_request.get("hand_position_x"),
                    hand_position_y=interaction_request.get("hand_position_y"),
                    hand_position_z=interaction_request.get("hand_position_z"),
                    gaze_position_x=interaction_request.get("gaze_position_x"),
                    gaze_position_y=interaction_request.get("gaze_position_y"),
                    gaze_position_z=interaction_request.get("gaze_position_z"),
                    interaction_data=json.dumps(interaction_request.get("interaction_data", {}))
                )
                
                self.db.add(interaction)
                self.db.commit()
                
                VR_INTERACTIONS.labels(interaction_type=interaction_type).inc()
                
                return {
                    "message": "VR interaction recorded successfully",
                    "interaction_id": interaction.id,
                    "session_id": session_id,
                    "object_id": object_id,
                    "interaction_type": interaction_type
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error recording VR interaction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/interactions", tags=["Interactions"])
        async def get_vr_interactions(session_id: str = None, limit: int = 100):
            """Get VR interactions."""
            try:
                query = self.db.query(VRInteraction)
                
                if session_id:
                    query = query.filter(VRInteraction.session_id == session_id)
                
                interactions = query.order_by(VRInteraction.timestamp.desc()).limit(limit).all()
                
                return {
                    "interactions": [
                        {
                            "id": interaction.id,
                            "session_id": interaction.session_id,
                            "object_id": interaction.object_id,
                            "interaction_type": interaction.interaction_type,
                            "user_id": interaction.user_id,
                            "hand_position": {
                                "x": interaction.hand_position_x,
                                "y": interaction.hand_position_y,
                                "z": interaction.hand_position_z
                            } if interaction.hand_position_x is not None else None,
                            "gaze_position": {
                                "x": interaction.gaze_position_x,
                                "y": interaction.gaze_position_y,
                                "z": interaction.gaze_position_z
                            } if interaction.gaze_position_x is not None else None,
                            "interaction_data": json.loads(interaction.interaction_data),
                            "timestamp": interaction.timestamp.isoformat()
                        }
                        for interaction in interactions
                    ],
                    "total": len(interactions)
                }
                
            except Exception as e:
                logger.error(f"Error getting VR interactions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time VR data."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and process VR data
                    data = await websocket.receive_text()
                    await self.process_vr_data(data)
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
        
        @self.app.get("/dashboard", tags=["Dashboard"])
        async def get_vr_dashboard():
            """Get VR system dashboard."""
            try:
                # Get statistics
                total_sessions = self.db.query(VRSession).count()
                active_sessions = self.db.query(VRSession).filter(VRSession.status == VRSessionStatus.ACTIVE).count()
                total_environments = self.db.query(VREnvironment).count()
                total_objects = self.db.query(VRObject).count()
                total_interactions = self.db.query(VRInteraction).count()
                
                # Get session type distribution
                session_types = {}
                for session_type in VRSessionType:
                    count = self.db.query(VRSession).filter(VRSession.session_type == session_type.value).count()
                    session_types[session_type.value] = count
                
                # Get object type distribution
                object_types = {}
                for object_type in VRObjectType:
                    count = self.db.query(VRObject).filter(VRObject.object_type == object_type.value).count()
                    object_types[object_type.value] = count
                
                # Get recent interactions
                recent_interactions = self.db.query(VRInteraction).order_by(
                    VRInteraction.timestamp.desc()
                ).limit(10).all()
                
                return {
                    "summary": {
                        "total_sessions": total_sessions,
                        "active_sessions": active_sessions,
                        "total_environments": total_environments,
                        "total_objects": total_objects,
                        "total_interactions": total_interactions
                    },
                    "session_type_distribution": session_types,
                    "object_type_distribution": object_types,
                    "recent_interactions": [
                        {
                            "id": interaction.id,
                            "session_id": interaction.session_id,
                            "object_id": interaction.object_id,
                            "interaction_type": interaction.interaction_type,
                            "user_id": interaction.user_id,
                            "timestamp": interaction.timestamp.isoformat()
                        }
                        for interaction in recent_interactions
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default VR data."""
        try:
            # Create sample environments
            sample_environments = [
                {
                    "name": "Office Environment",
                    "description": "Professional office space for meetings and collaboration",
                    "environment_type": "office",
                    "lighting_preset": "bright",
                    "physics_enabled": True,
                    "ambient_light": 0.4,
                    "directional_light": 1.2
                },
                {
                    "name": "Nature Forest",
                    "description": "Peaceful forest environment for relaxation and meditation",
                    "environment_type": "nature",
                    "lighting_preset": "natural",
                    "physics_enabled": True,
                    "ambient_light": 0.2,
                    "directional_light": 0.8,
                    "fog_enabled": True,
                    "fog_density": 0.005
                },
                {
                    "name": "Space Station",
                    "description": "Futuristic space station for training and simulation",
                    "environment_type": "space",
                    "lighting_preset": "artificial",
                    "physics_enabled": True,
                    "gravity": 0.0,
                    "ambient_light": 0.1,
                    "directional_light": 0.6
                }
            ]
            
            for env_data in sample_environments:
                environment = VREnvironment(
                    id=f"env_{env_data['name'].lower().replace(' ', '_')}",
                    name=env_data["name"],
                    description=env_data["description"],
                    environment_type=env_data["environment_type"],
                    lighting_preset=env_data["lighting_preset"],
                    physics_enabled=env_data["physics_enabled"],
                    gravity=env_data.get("gravity", -9.81),
                    ambient_light=env_data["ambient_light"],
                    directional_light=env_data["directional_light"],
                    fog_enabled=env_data.get("fog_enabled", False),
                    fog_density=env_data.get("fog_density", 0.01),
                    is_active=True
                )
                
                self.db.add(environment)
                self.environments[environment.id] = environment
            
            self.db.commit()
            logger.info("Default VR data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default VR data: {e}")
    
    def initialize_vr_components(self):
        """Initialize VR rendering and tracking components."""
        try:
            # Initialize Pygame for VR rendering
            pygame.init()
            self.pygame_initialized = True
            
            # Initialize OpenGL
            pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
            self.opengl_initialized = True
            
            # Initialize MediaPipe for hand tracking
            self.mediapipe_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=VR_CONFIG["hand_tracking_confidence"],
                min_tracking_confidence=VR_CONFIG["hand_tracking_confidence"]
            )
            
            logger.info("VR components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VR components: {e}")
    
    async def process_vr_data(self, data: str):
        """Process real-time VR data from WebSocket."""
        try:
            vr_data = json.loads(data)
            data_type = vr_data.get("type")
            
            if data_type == "hand_tracking":
                await self.process_hand_tracking(vr_data)
            elif data_type == "eye_tracking":
                await self.process_eye_tracking(vr_data)
            elif data_type == "head_tracking":
                await self.process_head_tracking(vr_data)
            elif data_type == "interaction":
                await self.process_interaction_data(vr_data)
            elif data_type == "performance":
                await self.process_performance_data(vr_data)
            
        except Exception as e:
            logger.error(f"Error processing VR data: {e}")
    
    async def process_hand_tracking(self, data: dict):
        """Process hand tracking data."""
        try:
            session_id = data.get("session_id")
            hand_data = data.get("hand_data", {})
            
            # Process hand landmarks
            left_hand = hand_data.get("left_hand")
            right_hand = hand_data.get("right_hand")
            
            # Update hand positions for interactions
            if left_hand:
                await self.update_hand_position(session_id, "left", left_hand)
            if right_hand:
                await self.update_hand_position(session_id, "right", right_hand)
            
        except Exception as e:
            logger.error(f"Error processing hand tracking: {e}")
    
    async def process_eye_tracking(self, data: dict):
        """Process eye tracking data."""
        try:
            session_id = data.get("session_id")
            eye_data = data.get("eye_data", {})
            
            # Process gaze direction and pupil data
            gaze_direction = eye_data.get("gaze_direction")
            pupil_size = eye_data.get("pupil_size")
            
            # Update gaze tracking for interactions
            if gaze_direction:
                await self.update_gaze_direction(session_id, gaze_direction)
            
        except Exception as e:
            logger.error(f"Error processing eye tracking: {e}")
    
    async def process_head_tracking(self, data: dict):
        """Process head tracking data."""
        try:
            session_id = data.get("session_id")
            head_data = data.get("head_data", {})
            
            # Process head position and rotation
            position = head_data.get("position")
            rotation = head_data.get("rotation")
            
            # Update head tracking for spatial audio and comfort settings
            if position and rotation:
                await self.update_head_tracking(session_id, position, rotation)
            
        except Exception as e:
            logger.error(f"Error processing head tracking: {e}")
    
    async def process_interaction_data(self, data: dict):
        """Process interaction data."""
        try:
            session_id = data.get("session_id")
            interaction_type = data.get("interaction_type")
            object_id = data.get("object_id")
            user_id = data.get("user_id")
            
            # Record interaction
            interaction_request = {
                "session_id": session_id,
                "object_id": object_id,
                "interaction_type": interaction_type,
                "user_id": user_id,
                "interaction_data": data.get("interaction_data", {})
            }
            
            # Add hand/gaze positions if available
            if "hand_position" in data:
                hand_pos = data["hand_position"]
                interaction_request.update({
                    "hand_position_x": hand_pos.get("x"),
                    "hand_position_y": hand_pos.get("y"),
                    "hand_position_z": hand_pos.get("z")
                })
            
            if "gaze_position" in data:
                gaze_pos = data["gaze_position"]
                interaction_request.update({
                    "gaze_position_x": gaze_pos.get("x"),
                    "gaze_position_y": gaze_pos.get("y"),
                    "gaze_position_z": gaze_pos.get("z")
                })
            
            # Record interaction in database
            interaction = VRInteraction(
                id=f"interaction_{int(time.time())}",
                session_id=session_id,
                object_id=object_id,
                interaction_type=interaction_type,
                user_id=user_id,
                hand_position_x=interaction_request.get("hand_position_x"),
                hand_position_y=interaction_request.get("hand_position_y"),
                hand_position_z=interaction_request.get("hand_position_z"),
                gaze_position_x=interaction_request.get("gaze_position_x"),
                gaze_position_y=interaction_request.get("gaze_position_y"),
                gaze_position_z=interaction_request.get("gaze_position_z"),
                interaction_data=json.dumps(interaction_request.get("interaction_data", {}))
            )
            
            self.db.add(interaction)
            self.db.commit()
            
            VR_INTERACTIONS.labels(interaction_type=interaction_type).inc()
            
        except Exception as e:
            logger.error(f"Error processing interaction data: {e}")
    
    async def process_performance_data(self, data: dict):
        """Process performance data."""
        try:
            session_id = data.get("session_id")
            frame_time = data.get("frame_time")
            fps = data.get("fps")
            gpu_usage = data.get("gpu_usage")
            cpu_usage = data.get("cpu_usage")
            
            # Record performance metrics
            if frame_time:
                VR_PERFORMANCE.observe(frame_time)
            
            # Update session performance
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                # Update session metadata with performance data
                metadata = json.loads(session.metadata)
                metadata["performance"] = {
                    "fps": fps,
                    "frame_time": frame_time,
                    "gpu_usage": gpu_usage,
                    "cpu_usage": cpu_usage,
                    "timestamp": datetime.now().isoformat()
                }
                session.metadata = json.dumps(metadata)
                self.db.commit()
            
        except Exception as e:
            logger.error(f"Error processing performance data: {e}")
    
    async def update_hand_position(self, session_id: str, hand: str, hand_data: dict):
        """Update hand position for interactions."""
        try:
            # Process hand landmarks and update interaction capabilities
            landmarks = hand_data.get("landmarks", [])
            
            if landmarks:
                # Calculate hand center position
                center_x = sum(point["x"] for point in landmarks) / len(landmarks)
                center_y = sum(point["y"] for point in landmarks) / len(landmarks)
                center_z = sum(point["z"] for point in landmarks) / len(landmarks)
                
                # Check for interactions with nearby objects
                await self.check_hand_interactions(session_id, hand, center_x, center_y, center_z)
            
        except Exception as e:
            logger.error(f"Error updating hand position: {e}")
    
    async def update_gaze_direction(self, session_id: str, gaze_direction: dict):
        """Update gaze direction for interactions."""
        try:
            # Process gaze direction and update interaction capabilities
            x = gaze_direction.get("x", 0)
            y = gaze_direction.get("y", 0)
            z = gaze_direction.get("z", 0)
            
            # Check for gaze-based interactions
            await self.check_gaze_interactions(session_id, x, y, z)
            
        except Exception as e:
            logger.error(f"Error updating gaze direction: {e}")
    
    async def update_head_tracking(self, session_id: str, position: dict, rotation: dict):
        """Update head tracking for spatial audio and comfort."""
        try:
            # Update spatial audio based on head position
            await self.update_spatial_audio(session_id, position, rotation)
            
            # Update comfort settings based on head movement
            await self.update_comfort_settings(session_id, position, rotation)
            
        except Exception as e:
            logger.error(f"Error updating head tracking: {e}")
    
    async def check_hand_interactions(self, session_id: str, hand: str, x: float, y: float, z: float):
        """Check for hand interactions with VR objects."""
        try:
            # Get session environment
            session = self.active_sessions.get(session_id)
            if not session or not session.environment_id:
                return
            
            # Get objects in environment
            objects = self.db.query(VRObject).filter(
                VRObject.environment_id == session.environment_id,
                VRObject.is_interactive == True,
                VRObject.is_visible == True
            ).all()
            
            # Check for interactions with nearby objects
            for obj in objects:
                distance = math.sqrt(
                    (x - obj.position_x) ** 2 +
                    (y - obj.position_y) ** 2 +
                    (z - obj.position_z) ** 2
                )
                
                # If hand is close enough to object, trigger interaction
                if distance < 0.1:  # 10cm threshold
                    await self.trigger_hand_interaction(session_id, obj.id, hand, x, y, z)
            
        except Exception as e:
            logger.error(f"Error checking hand interactions: {e}")
    
    async def check_gaze_interactions(self, session_id: str, x: float, y: float, z: float):
        """Check for gaze interactions with VR objects."""
        try:
            # Get session environment
            session = self.active_sessions.get(session_id)
            if not session or not session.environment_id:
                return
            
            # Get objects in environment
            objects = self.db.query(VRObject).filter(
                VRObject.environment_id == session.environment_id,
                VRObject.is_interactive == True,
                VRObject.is_visible == True
            ).all()
            
            # Check for gaze interactions with objects
            for obj in objects:
                # Calculate gaze-object intersection
                if await self.check_gaze_object_intersection(x, y, z, obj):
                    await self.trigger_gaze_interaction(session_id, obj.id, x, y, z)
            
        except Exception as e:
            logger.error(f"Error checking gaze interactions: {e}")
    
    async def check_gaze_object_intersection(self, gaze_x: float, gaze_y: float, gaze_z: float, obj: VRObject) -> bool:
        """Check if gaze ray intersects with VR object."""
        try:
            # Simple bounding box intersection test
            # In a real implementation, this would use proper ray-object intersection
            distance = math.sqrt(
                (gaze_x - obj.position_x) ** 2 +
                (gaze_y - obj.position_y) ** 2 +
                (gaze_z - obj.position_z) ** 2
            )
            
            # Check if gaze is within object bounds
            return distance < (obj.scale_x + obj.scale_y + obj.scale_z) / 3
            
        except Exception:
            return False
    
    async def trigger_hand_interaction(self, session_id: str, object_id: str, hand: str, x: float, y: float, z: float):
        """Trigger hand interaction with VR object."""
        try:
            # Record hand interaction
            interaction_request = {
                "session_id": session_id,
                "object_id": object_id,
                "interaction_type": VRInteractionType.GRAB,
                "user_id": "user",  # Would be actual user ID
                "hand_position_x": x,
                "hand_position_y": y,
                "hand_position_z": z,
                "interaction_data": {"hand": hand, "action": "grab"}
            }
            
            # Process interaction
            await self.process_interaction_data({"type": "interaction", **interaction_request})
            
        except Exception as e:
            logger.error(f"Error triggering hand interaction: {e}")
    
    async def trigger_gaze_interaction(self, session_id: str, object_id: str, x: float, y: float, z: float):
        """Trigger gaze interaction with VR object."""
        try:
            # Record gaze interaction
            interaction_request = {
                "session_id": session_id,
                "object_id": object_id,
                "interaction_type": VRInteractionType.GAZE,
                "user_id": "user",  # Would be actual user ID
                "gaze_position_x": x,
                "gaze_position_y": y,
                "gaze_position_z": z,
                "interaction_data": {"action": "gaze"}
            }
            
            # Process interaction
            await self.process_interaction_data({"type": "interaction", **interaction_request})
            
        except Exception as e:
            logger.error(f"Error triggering gaze interaction: {e}")
    
    async def update_spatial_audio(self, session_id: str, position: dict, rotation: dict):
        """Update spatial audio based on head position."""
        try:
            # Update spatial audio parameters
            # In a real implementation, this would update 3D audio engine
            logger.debug(f"Updating spatial audio for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error updating spatial audio: {e}")
    
    async def update_comfort_settings(self, session_id: str, position: dict, rotation: dict):
        """Update comfort settings based on head movement."""
        try:
            # Update comfort settings like snap turning, vignette, etc.
            # In a real implementation, this would adjust VR comfort features
            logger.debug(f"Updating comfort settings for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error updating comfort settings: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8010, debug: bool = False):
        """Run the VR system."""
        logger.info(f"Starting VR System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL VR System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run VR system
    system = AdvancedVRSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
