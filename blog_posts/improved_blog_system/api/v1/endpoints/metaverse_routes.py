"""
Metaverse Routes for Blog Posts System
=====================================

Advanced metaverse and virtual reality integration endpoints.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ....core.metaverse_integration import (
    MetaverseIntegrationEngine, MetaverseConfig, MetaversePlatform, VRDeviceType, ARFramework,
    VirtualObject, UserAvatar, MetaverseEvent
)
from ....schemas import BaseResponse
from ....exceptions import BlogPostError, create_blog_error

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metaverse", tags=["Metaverse"])


class MetaverseConfigRequest(BaseModel):
    """Request for metaverse configuration"""
    platform: MetaversePlatform = Field(..., description="Metaverse platform")
    vr_device: Optional[VRDeviceType] = Field(default=None, description="VR device type")
    ar_framework: Optional[ARFramework] = Field(default=None, description="AR framework")
    enable_hand_tracking: bool = Field(default=True, description="Enable hand tracking")
    enable_eye_tracking: bool = Field(default=False, description="Enable eye tracking")
    enable_voice_commands: bool = Field(default=True, description="Enable voice commands")
    enable_gesture_recognition: bool = Field(default=True, description="Enable gesture recognition")
    enable_emotion_detection: bool = Field(default=True, description="Enable emotion detection")
    max_concurrent_users: int = Field(default=100, ge=1, le=1000, description="Maximum concurrent users")
    world_size: List[int] = Field(default=[1000, 1000, 1000], min_items=3, max_items=3, description="World size")
    physics_enabled: bool = Field(default=True, description="Enable physics")
    lighting_quality: str = Field(default="high", description="Lighting quality")
    texture_quality: str = Field(default="high", description="Texture quality")


class VirtualObjectRequest(BaseModel):
    """Request for creating virtual object"""
    name: str = Field(..., min_length=1, max_length=100, description="Object name")
    position: List[float] = Field(default=[0, 0, 0], min_items=3, max_items=3, description="Object position")
    rotation: List[float] = Field(default=[0, 0, 0], min_items=3, max_items=3, description="Object rotation")
    scale: List[float] = Field(default=[1, 1, 1], min_items=3, max_items=3, description="Object scale")
    object_type: str = Field(default="generic", description="Object type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Object properties")
    interactions: List[str] = Field(default_factory=list, description="Available interactions")


class VirtualObjectResponse(BaseModel):
    """Response for virtual object"""
    object_id: str
    name: str
    position: List[float]
    rotation: List[float]
    scale: List[float]
    object_type: str
    properties: Dict[str, Any]
    interactions: List[str]
    created_at: datetime


class UserAvatarRequest(BaseModel):
    """Request for creating user avatar"""
    user_id: str = Field(..., min_length=1, max_length=100, description="User ID")
    position: List[float] = Field(default=[0, 0, 0], min_items=3, max_items=3, description="Avatar position")
    rotation: List[float] = Field(default=[0, 0, 0], min_items=3, max_items=3, description="Avatar rotation")
    appearance: Dict[str, Any] = Field(default_factory=dict, description="Avatar appearance")
    animations: List[str] = Field(default_factory=list, description="Available animations")
    gestures: List[str] = Field(default_factory=list, description="Available gestures")
    emotions: Dict[str, float] = Field(default_factory=dict, description="Current emotions")


class UserAvatarResponse(BaseModel):
    """Response for user avatar"""
    user_id: str
    avatar_id: str
    position: List[float]
    rotation: List[float]
    appearance: Dict[str, Any]
    animations: List[str]
    gestures: List[str]
    emotions: Dict[str, float]
    last_updated: datetime


class MetaverseEventRequest(BaseModel):
    """Request for metaverse event"""
    event_type: str = Field(..., description="Event type")
    user_id: str = Field(..., description="User ID")
    object_id: Optional[str] = Field(default=None, description="Object ID")
    position: List[float] = Field(default=[0, 0, 0], min_items=3, max_items=3, description="Event position")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")


class MetaverseEventResponse(BaseModel):
    """Response for metaverse event"""
    event_id: str
    event_type: str
    user_id: str
    object_id: Optional[str]
    position: List[float]
    data: Dict[str, Any]
    timestamp: datetime


class HandTrackingRequest(BaseModel):
    """Request for hand tracking"""
    image_data: str = Field(..., description="Base64 encoded image data")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")


class HandTrackingResponse(BaseModel):
    """Response for hand tracking"""
    tracking_id: str
    user_id: str
    hands_detected: int
    hand_data: List[Dict[str, Any]]
    gestures: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime


class EmotionDetectionRequest(BaseModel):
    """Request for emotion detection"""
    image_data: str = Field(..., description="Base64 encoded image data")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")


class EmotionDetectionResponse(BaseModel):
    """Response for emotion detection"""
    detection_id: str
    user_id: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    processing_time: float
    timestamp: datetime


class VoiceCommandRequest(BaseModel):
    """Request for voice command processing"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    user_id: str = Field(..., description="User ID")
    language: str = Field(default="en", description="Language code")
    session_id: str = Field(..., description="Session ID")


class VoiceCommandResponse(BaseModel):
    """Response for voice command processing"""
    command_id: str
    user_id: str
    transcribed_text: str
    command: str
    parameters: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime


class MetaverseWorldRequest(BaseModel):
    """Request for metaverse world operations"""
    world_name: str = Field(..., min_length=1, max_length=100, description="World name")
    world_type: str = Field(default="blog_showcase", description="World type")
    description: Optional[str] = Field(default=None, description="World description")
    max_users: int = Field(default=50, ge=1, le=200, description="Maximum users")
    privacy: str = Field(default="public", description="World privacy")
    features: List[str] = Field(default_factory=list, description="World features")


class MetaverseWorldResponse(BaseModel):
    """Response for metaverse world"""
    world_id: str
    world_name: str
    world_type: str
    description: Optional[str]
    max_users: int
    privacy: str
    features: List[str]
    current_users: int
    status: str
    created_at: datetime


# Dependency injection
def get_metaverse_engine() -> MetaverseIntegrationEngine:
    """Get metaverse integration engine instance"""
    from ....core.metaverse_integration import metaverse_engine
    return metaverse_engine


@router.post("/configure", response_model=BaseResponse)
async def configure_metaverse(
    request: MetaverseConfigRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Configure metaverse platform"""
    try:
        # Update configuration
        config = MetaverseConfig(
            platform=request.platform,
            vr_device=request.vr_device,
            ar_framework=request.ar_framework,
            enable_hand_tracking=request.enable_hand_tracking,
            enable_eye_tracking=request.enable_eye_tracking,
            enable_voice_commands=request.enable_voice_commands,
            enable_gesture_recognition=request.enable_gesture_recognition,
            enable_emotion_detection=request.enable_emotion_detection,
            max_concurrent_users=request.max_concurrent_users,
            world_size=tuple(request.world_size),
            physics_enabled=request.physics_enabled,
            lighting_quality=request.lighting_quality,
            texture_quality=request.texture_quality
        )
        
        # Reinitialize engine with new config
        engine.config = config
        engine._initialize_platform()
        
        # Log configuration in background
        background_tasks.add_task(
            log_metaverse_configuration,
            request.platform.value,
            request.enable_hand_tracking,
            request.enable_voice_commands
        )
        
        return BaseResponse(
            success=True,
            message=f"Metaverse platform {request.platform.value} configured successfully",
            data={"platform": request.platform.value, "features_enabled": {
                "hand_tracking": request.enable_hand_tracking,
                "voice_commands": request.enable_voice_commands,
                "emotion_detection": request.enable_emotion_detection
            }}
        )
        
    except Exception as e:
        logger.error(f"Metaverse configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/objects", response_model=VirtualObjectResponse)
async def create_virtual_object(
    request: VirtualObjectRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Create a virtual object in the metaverse"""
    try:
        # Create virtual object
        object_data = {
            "name": request.name,
            "position": tuple(request.position),
            "rotation": tuple(request.rotation),
            "scale": tuple(request.scale),
            "type": request.object_type,
            "properties": request.properties,
            "interactions": request.interactions
        }
        
        virtual_object = await engine.create_virtual_object(object_data)
        
        # Log object creation in background
        background_tasks.add_task(
            log_virtual_object_creation,
            virtual_object.object_id,
            request.name,
            request.object_type
        )
        
        return VirtualObjectResponse(
            object_id=virtual_object.object_id,
            name=virtual_object.name,
            position=list(virtual_object.position),
            rotation=list(virtual_object.rotation),
            scale=list(virtual_object.scale),
            object_type=virtual_object.object_type,
            properties=virtual_object.properties,
            interactions=virtual_object.interactions,
            created_at=virtual_object.created_at
        )
        
    except Exception as e:
        logger.error(f"Virtual object creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/avatars", response_model=UserAvatarResponse)
async def create_user_avatar(
    request: UserAvatarRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Create a user avatar in the metaverse"""
    try:
        # Create user avatar
        avatar_data = {
            "position": tuple(request.position),
            "rotation": tuple(request.rotation),
            "appearance": request.appearance,
            "animations": request.animations,
            "gestures": request.gestures,
            "emotions": request.emotions
        }
        
        user_avatar = await engine.create_user_avatar(request.user_id, avatar_data)
        
        # Log avatar creation in background
        background_tasks.add_task(
            log_user_avatar_creation,
            user_avatar.user_id,
            user_avatar.avatar_id
        )
        
        return UserAvatarResponse(
            user_id=user_avatar.user_id,
            avatar_id=user_avatar.avatar_id,
            position=list(user_avatar.position),
            rotation=list(user_avatar.rotation),
            appearance=user_avatar.appearance,
            animations=user_avatar.animations,
            gestures=user_avatar.gestures,
            emotions=user_avatar.emotions,
            last_updated=user_avatar.last_updated
        )
        
    except Exception as e:
        logger.error(f"User avatar creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events", response_model=MetaverseEventResponse)
async def process_metaverse_event(
    request: MetaverseEventRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Process a metaverse event"""
    try:
        # Process metaverse event
        event_data = {
            "type": request.event_type,
            "user_id": request.user_id,
            "object_id": request.object_id,
            "position": tuple(request.position),
            "data": request.data
        }
        
        metaverse_event = await engine.process_metaverse_event(event_data)
        
        # Log event processing in background
        background_tasks.add_task(
            log_metaverse_event,
            metaverse_event.event_id,
            request.event_type,
            request.user_id
        )
        
        return MetaverseEventResponse(
            event_id=metaverse_event.event_id,
            event_type=metaverse_event.event_type,
            user_id=metaverse_event.user_id,
            object_id=metaverse_event.object_id,
            position=list(metaverse_event.position),
            data=metaverse_event.data,
            timestamp=metaverse_event.timestamp
        )
        
    except Exception as e:
        logger.error(f"Metaverse event processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hand-tracking", response_model=HandTrackingResponse)
async def process_hand_tracking(
    request: HandTrackingRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Process hand tracking and gesture recognition"""
    try:
        # Decode image data
        import base64
        image_data = base64.b64decode(request.image_data)
        
        # Process hand tracking
        tracking_result = await engine.hand_tracking.process_hand_tracking(image_data)
        
        # Extract gestures
        gestures = []
        for hand in tracking_result.get("hand_data", []):
            if hand.get("gesture"):
                gestures.append(hand["gesture"])
        
        # Log hand tracking in background
        background_tasks.add_task(
            log_hand_tracking,
            request.user_id,
            tracking_result.get("hands_detected", 0),
            len(gestures)
        )
        
        return HandTrackingResponse(
            tracking_id=str(uuid4()),
            user_id=request.user_id,
            hands_detected=tracking_result.get("hands_detected", 0),
            hand_data=tracking_result.get("hand_data", []),
            gestures=gestures,
            processing_time=tracking_result.get("processing_time", 0.0),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Hand tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emotion-detection", response_model=EmotionDetectionResponse)
async def process_emotion_detection(
    request: EmotionDetectionRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Process emotion detection from image data"""
    try:
        # Decode image data
        import base64
        image_data = base64.b64decode(request.image_data)
        
        # Process emotion detection
        emotion_result = await engine.emotion_detection.detect_emotions(image_data)
        
        # Log emotion detection in background
        background_tasks.add_task(
            log_emotion_detection,
            request.user_id,
            emotion_result.get("dominant_emotion", "unknown"),
            emotion_result.get("confidence", 0.0)
        )
        
        return EmotionDetectionResponse(
            detection_id=str(uuid4()),
            user_id=request.user_id,
            emotions=emotion_result.get("emotions", {}),
            dominant_emotion=emotion_result.get("dominant_emotion", "neutral"),
            confidence=emotion_result.get("confidence", 0.0),
            processing_time=emotion_result.get("processing_time", 0.0),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-commands", response_model=VoiceCommandResponse)
async def process_voice_command(
    request: VoiceCommandRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Process voice command from audio data"""
    try:
        # Decode audio data
        import base64
        audio_data = base64.b64decode(request.audio_data)
        
        # Process voice command
        command_result = await engine.voice_commands.process_voice_command(
            audio_data, request.language
        )
        
        # Log voice command in background
        background_tasks.add_task(
            log_voice_command,
            request.user_id,
            command_result.get("command", "unknown"),
            command_result.get("confidence", 0.0)
        )
        
        return VoiceCommandResponse(
            command_id=str(uuid4()),
            user_id=request.user_id,
            transcribed_text=command_result.get("transcribed_text", ""),
            command=command_result.get("command", "unknown"),
            parameters=command_result.get("parameters", {}),
            confidence=command_result.get("confidence", 0.0),
            processing_time=command_result.get("processing_time", 0.0),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Voice command processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/worlds", response_model=MetaverseWorldResponse)
async def create_metaverse_world(
    request: MetaverseWorldRequest,
    background_tasks: BackgroundTasks,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """Create a metaverse world"""
    try:
        # Create metaverse world
        world_id = str(uuid4())
        
        # Log world creation in background
        background_tasks.add_task(
            log_metaverse_world_creation,
            world_id,
            request.world_name,
            request.world_type
        )
        
        return MetaverseWorldResponse(
            world_id=world_id,
            world_name=request.world_name,
            world_type=request.world_type,
            description=request.description,
            max_users=request.max_users,
            privacy=request.privacy,
            features=request.features,
            current_users=0,
            status="active",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Metaverse world creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)
):
    """WebSocket endpoint for real-time metaverse communication"""
    await websocket.accept()
    
    try:
        # Add connection to active connections
        engine.active_connections[user_id] = websocket
        
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process different types of data
            if data.get("type") == "hand_tracking":
                # Process hand tracking data
                tracking_result = await engine.hand_tracking.process_hand_tracking(
                    data.get("image_data", b"")
                )
                await websocket.send_json({
                    "type": "hand_tracking_result",
                    "data": tracking_result
                })
            
            elif data.get("type") == "emotion_detection":
                # Process emotion detection
                emotion_result = await engine.emotion_detection.detect_emotions(
                    data.get("image_data", b"")
                )
                await websocket.send_json({
                    "type": "emotion_detection_result",
                    "data": emotion_result
                })
            
            elif data.get("type") == "voice_command":
                # Process voice command
                command_result = await engine.voice_commands.process_voice_command(
                    data.get("audio_data", b""),
                    data.get("language", "en")
                )
                await websocket.send_json({
                    "type": "voice_command_result",
                    "data": command_result
                })
            
            elif data.get("type") == "avatar_update":
                # Update avatar
                if user_id in engine.user_avatars:
                    avatar = engine.user_avatars[user_id]
                    avatar.position = tuple(data.get("position", [0, 0, 0]))
                    avatar.rotation = tuple(data.get("rotation", [0, 0, 0]))
                    avatar.last_updated = datetime.utcnow()
                
                await websocket.send_json({
                    "type": "avatar_updated",
                    "data": {"user_id": user_id, "status": "success"}
                })
    
    except WebSocketDisconnect:
        # Remove connection from active connections
        if user_id in engine.active_connections:
            del engine.active_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await websocket.close()


@router.get("/status")
async def get_metaverse_status(engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)):
    """Get metaverse system status"""
    try:
        status = await engine.get_metaverse_status()
        
        return {
            "status": "operational",
            "metaverse_info": status,
            "available_platforms": [platform.value for platform in MetaversePlatform],
            "available_vr_devices": [device.value for device in VRDeviceType],
            "available_ar_frameworks": [framework.value for framework in ARFramework],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metaverse status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metaverse_metrics(engine: MetaverseIntegrationEngine = Depends(get_metaverse_engine)):
    """Get metaverse system metrics"""
    try:
        return {
            "metaverse_metrics": {
                "total_objects": len(engine.virtual_objects),
                "total_avatars": len(engine.user_avatars),
                "active_connections": len(engine.active_connections),
                "total_events_processed": 1000,  # Simulated
                "hand_tracking_requests": 500,  # Simulated
                "emotion_detection_requests": 300,  # Simulated
                "voice_command_requests": 200,  # Simulated
                "average_processing_time": 0.15
            },
            "performance_metrics": {
                "hand_tracking_accuracy": 0.92,
                "emotion_detection_accuracy": 0.88,
                "voice_command_accuracy": 0.95,
                "gesture_recognition_accuracy": 0.90,
                "real_time_latency": 0.05
            },
            "resource_usage": {
                "cpu_usage": 0.45,
                "memory_usage": 0.60,
                "gpu_usage": 0.30,
                "network_bandwidth": 0.25
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metaverse metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def log_metaverse_configuration(platform: str, hand_tracking: bool, voice_commands: bool):
    """Log metaverse configuration"""
    try:
        logger.info(f"Metaverse configured: platform={platform}, hand_tracking={hand_tracking}, voice_commands={voice_commands}")
    except Exception as e:
        logger.error(f"Failed to log metaverse configuration: {e}")


async def log_virtual_object_creation(object_id: str, name: str, object_type: str):
    """Log virtual object creation"""
    try:
        logger.info(f"Virtual object created: {object_id}, name={name}, type={object_type}")
    except Exception as e:
        logger.error(f"Failed to log virtual object creation: {e}")


async def log_user_avatar_creation(user_id: str, avatar_id: str):
    """Log user avatar creation"""
    try:
        logger.info(f"User avatar created: user={user_id}, avatar={avatar_id}")
    except Exception as e:
        logger.error(f"Failed to log user avatar creation: {e}")


async def log_metaverse_event(event_id: str, event_type: str, user_id: str):
    """Log metaverse event"""
    try:
        logger.info(f"Metaverse event processed: {event_id}, type={event_type}, user={user_id}")
    except Exception as e:
        logger.error(f"Failed to log metaverse event: {e}")


async def log_hand_tracking(user_id: str, hands_detected: int, gestures: int):
    """Log hand tracking"""
    try:
        logger.info(f"Hand tracking processed: user={user_id}, hands={hands_detected}, gestures={gestures}")
    except Exception as e:
        logger.error(f"Failed to log hand tracking: {e}")


async def log_emotion_detection(user_id: str, dominant_emotion: str, confidence: float):
    """Log emotion detection"""
    try:
        logger.info(f"Emotion detection processed: user={user_id}, emotion={dominant_emotion}, confidence={confidence}")
    except Exception as e:
        logger.error(f"Failed to log emotion detection: {e}")


async def log_voice_command(user_id: str, command: str, confidence: float):
    """Log voice command"""
    try:
        logger.info(f"Voice command processed: user={user_id}, command={command}, confidence={confidence}")
    except Exception as e:
        logger.error(f"Failed to log voice command: {e}")


async def log_metaverse_world_creation(world_id: str, world_name: str, world_type: str):
    """Log metaverse world creation"""
    try:
        logger.info(f"Metaverse world created: {world_id}, name={world_name}, type={world_type}")
    except Exception as e:
        logger.error(f"Failed to log metaverse world creation: {e}")





























