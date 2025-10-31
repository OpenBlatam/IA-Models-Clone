"""
Advanced AR/VR API endpoints
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_ar_vr_service import AdvancedARVRService, ARVRDeviceType, ARVRContentType, ARVRInteractionType, ARVRTrackingType
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateARVRSceneRequest(BaseModel):
    """Request model for creating an AR/VR scene."""
    name: str = Field(..., description="Scene name")
    description: str = Field(..., description="Scene description")
    scene_type: str = Field(..., description="Scene type")
    device_type: str = Field(..., description="Device type")
    tracking_type: str = Field(..., description="Tracking type")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="Scene configuration")


class AddARVRObjectRequest(BaseModel):
    """Request model for adding an AR/VR object."""
    scene_id: str = Field(..., description="Scene ID")
    name: str = Field(..., description="Object name")
    object_type: str = Field(..., description="Object type")
    content_type: str = Field(..., description="Content type")
    position: Dict[str, float] = Field(..., description="Object position")
    rotation: Dict[str, float] = Field(..., description="Object rotation")
    scale: Dict[str, float] = Field(..., description="Object scale")
    content_data: Optional[Dict[str, Any]] = Field(default=None, description="Content data")


class StartARVRSessionRequest(BaseModel):
    """Request model for starting an AR/VR session."""
    scene_id: str = Field(..., description="Scene ID")
    device_id: str = Field(..., description="Device ID")
    device_type: str = Field(..., description="Device type")
    session_configuration: Optional[Dict[str, Any]] = Field(default=None, description="Session configuration")


class TrackARVRInteractionRequest(BaseModel):
    """Request model for tracking AR/VR interaction."""
    session_id: str = Field(..., description="Session ID")
    interaction_type: str = Field(..., description="Interaction type")
    interaction_data: Dict[str, Any] = Field(..., description="Interaction data")
    position: Optional[Dict[str, float]] = Field(default=None, description="Interaction position")
    rotation: Optional[Dict[str, float]] = Field(default=None, description="Interaction rotation")


async def get_ar_vr_service(session: DatabaseSessionDep) -> AdvancedARVRService:
    """Get AR/VR service instance."""
    return AdvancedARVRService(session)


@router.post("/scenes", response_model=Dict[str, Any])
async def create_ar_vr_scene(
    request: CreateARVRSceneRequest = Depends(),
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new AR/VR scene."""
    try:
        # Convert enums
        try:
            device_type_enum = ARVRDeviceType(request.device_type.lower())
            tracking_type_enum = ARVRTrackingType(request.tracking_type.lower())
        except ValueError as e:
            raise ValidationError(f"Invalid enum value: {e}")
        
        result = await ar_vr_service.create_ar_vr_scene(
            name=request.name,
            description=request.description,
            scene_type=request.scene_type,
            user_id=str(current_user.id),
            device_type=device_type_enum,
            tracking_type=tracking_type_enum,
            configuration=request.configuration
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AR/VR scene created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create AR/VR scene"
        )


@router.post("/objects", response_model=Dict[str, Any])
async def add_ar_vr_object(
    request: AddARVRObjectRequest = Depends(),
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Add an object to an AR/VR scene."""
    try:
        # Convert content type to enum
        try:
            content_type_enum = ARVRContentType(request.content_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid content type: {request.content_type}")
        
        result = await ar_vr_service.add_ar_vr_object(
            scene_id=request.scene_id,
            name=request.name,
            object_type=request.object_type,
            content_type=content_type_enum,
            position=request.position,
            rotation=request.rotation,
            scale=request.scale,
            content_data=request.content_data
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AR/VR object added successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add AR/VR object"
        )


@router.post("/sessions", response_model=Dict[str, Any])
async def start_ar_vr_session(
    request: StartARVRSessionRequest = Depends(),
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Start an AR/VR session."""
    try:
        # Convert device type to enum
        try:
            device_type_enum = ARVRDeviceType(request.device_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid device type: {request.device_type}")
        
        result = await ar_vr_service.start_ar_vr_session(
            scene_id=request.scene_id,
            user_id=str(current_user.id),
            device_id=request.device_id,
            device_type=device_type_enum,
            session_configuration=request.session_configuration
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AR/VR session started successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start AR/VR session"
        )


@router.post("/interactions", response_model=Dict[str, Any])
async def track_ar_vr_interaction(
    request: TrackARVRInteractionRequest = Depends(),
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Track AR/VR interaction."""
    try:
        # Convert interaction type to enum
        try:
            interaction_type_enum = ARVRInteractionType(request.interaction_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid interaction type: {request.interaction_type}")
        
        result = await ar_vr_service.track_ar_vr_interaction(
            session_id=request.session_id,
            interaction_type=interaction_type_enum,
            interaction_data=request.interaction_data,
            position=request.position,
            rotation=request.rotation
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AR/VR interaction tracked successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track AR/VR interaction"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_ar_vr_analytics(
    scene_id: Optional[str] = Query(default=None, description="Scene ID"),
    user_id: Optional[str] = Query(default=None, description="User ID"),
    device_type: Optional[str] = Query(default=None, description="Device type"),
    time_period: str = Query(default="24_hours", description="Time period"),
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Get AR/VR analytics."""
    try:
        result = await ar_vr_service.get_ar_vr_analytics(
            scene_id=scene_id,
            user_id=user_id,
            device_type=device_type,
            time_period=time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "AR/VR analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR/VR analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_ar_vr_stats(
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Get AR/VR system statistics."""
    try:
        result = await ar_vr_service.get_ar_vr_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "AR/VR statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR/VR statistics"
        )


@router.get("/device-types", response_model=Dict[str, Any])
async def get_ar_vr_device_types():
    """Get available AR/VR device types."""
    device_types = {
        "vr_headset": {
            "name": "VR Headset",
            "description": "Virtual Reality headset device",
            "icon": "🥽",
            "capabilities": ["vr", "head_tracking", "hand_tracking", "spatial_audio"],
            "platforms": ["pc", "console", "mobile", "standalone"]
        },
        "ar_glasses": {
            "name": "AR Glasses",
            "description": "Augmented Reality glasses device",
            "icon": "👓",
            "capabilities": ["ar", "see_through", "hand_tracking", "voice_control"],
            "platforms": ["standalone", "mobile", "pc"]
        },
        "ar_phone": {
            "name": "AR Phone",
            "description": "Smartphone with AR capabilities",
            "icon": "📱",
            "capabilities": ["ar", "camera", "sensors", "touch"],
            "platforms": ["ios", "android"]
        },
        "ar_tablet": {
            "name": "AR Tablet",
            "description": "Tablet with AR capabilities",
            "icon": "📱",
            "capabilities": ["ar", "camera", "sensors", "touch", "large_screen"],
            "platforms": ["ios", "android", "windows"]
        },
        "mixed_reality": {
            "name": "Mixed Reality",
            "description": "Mixed Reality device",
            "icon": "🌐",
            "capabilities": ["ar", "vr", "hand_tracking", "spatial_mapping"],
            "platforms": ["windows", "hololens"]
        },
        "hololens": {
            "name": "HoloLens",
            "description": "Microsoft HoloLens device",
            "icon": "🥽",
            "capabilities": ["ar", "hand_tracking", "voice_control", "spatial_mapping"],
            "platforms": ["windows"]
        },
        "oculus": {
            "name": "Oculus",
            "description": "Oculus VR device",
            "icon": "🥽",
            "capabilities": ["vr", "hand_tracking", "spatial_audio", "room_scale"],
            "platforms": ["pc", "standalone"]
        },
        "vive": {
            "name": "HTC Vive",
            "description": "HTC Vive VR device",
            "icon": "🥽",
            "capabilities": ["vr", "room_scale", "hand_tracking", "spatial_audio"],
            "platforms": ["pc"]
        },
        "quest": {
            "name": "Oculus Quest",
            "description": "Oculus Quest standalone VR device",
            "icon": "🥽",
            "capabilities": ["vr", "hand_tracking", "spatial_audio", "wireless"],
            "platforms": ["standalone"]
        },
        "rift": {
            "name": "Oculus Rift",
            "description": "Oculus Rift PC VR device",
            "icon": "🥽",
            "capabilities": ["vr", "room_scale", "hand_tracking", "spatial_audio"],
            "platforms": ["pc"]
        },
        "gear_vr": {
            "name": "Gear VR",
            "description": "Samsung Gear VR device",
            "icon": "🥽",
            "capabilities": ["vr", "mobile", "controller"],
            "platforms": ["mobile"]
        },
        "daydream": {
            "name": "Daydream",
            "description": "Google Daydream VR device",
            "icon": "🥽",
            "capabilities": ["vr", "mobile", "controller"],
            "platforms": ["mobile"]
        },
        "cardboard": {
            "name": "Cardboard",
            "description": "Google Cardboard VR device",
            "icon": "🥽",
            "capabilities": ["vr", "mobile", "basic"],
            "platforms": ["mobile"]
        },
        "windows_mr": {
            "name": "Windows Mixed Reality",
            "description": "Windows Mixed Reality device",
            "icon": "🥽",
            "capabilities": ["vr", "ar", "hand_tracking", "spatial_mapping"],
            "platforms": ["windows"]
        },
        "psvr": {
            "name": "PlayStation VR",
            "description": "PlayStation VR device",
            "icon": "🥽",
            "capabilities": ["vr", "console", "controller"],
            "platforms": ["console"]
        },
        "valve_index": {
            "name": "Valve Index",
            "description": "Valve Index VR device",
            "icon": "🥽",
            "capabilities": ["vr", "high_fidelity", "finger_tracking"],
            "platforms": ["pc"]
        },
        "pico": {
            "name": "Pico",
            "description": "Pico VR device",
            "icon": "🥽",
            "capabilities": ["vr", "standalone", "hand_tracking"],
            "platforms": ["standalone"]
        },
        "htc_focus": {
            "name": "HTC Focus",
            "description": "HTC Focus VR device",
            "icon": "🥽",
            "capabilities": ["vr", "standalone", "hand_tracking"],
            "platforms": ["standalone"]
        },
        "lenovo_mirage": {
            "name": "Lenovo Mirage",
            "description": "Lenovo Mirage VR device",
            "icon": "🥽",
            "capabilities": ["vr", "standalone", "hand_tracking"],
            "platforms": ["standalone"]
        },
        "samsung_odyssey": {
            "name": "Samsung Odyssey",
            "description": "Samsung Odyssey VR device",
            "icon": "🥽",
            "capabilities": ["vr", "high_resolution", "hand_tracking"],
            "platforms": ["windows"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "device_types": device_types,
            "total_types": len(device_types)
        },
        "message": "AR/VR device types retrieved successfully"
    }


@router.get("/content-types", response_model=Dict[str, Any])
async def get_ar_vr_content_types():
    """Get available AR/VR content types."""
    content_types = {
        "model_3d": {
            "name": "3D Model",
            "description": "Three-dimensional model",
            "icon": "🎲",
            "formats": ["obj", "fbx", "gltf", "glb", "dae", "3ds", "blend"],
            "use_cases": ["visualization", "gaming", "education", "training"]
        },
        "texture": {
            "name": "Texture",
            "description": "Surface texture for 3D models",
            "icon": "🎨",
            "formats": ["jpg", "png", "tga", "dds", "ktx", "astc"],
            "use_cases": ["rendering", "visualization", "gaming", "art"]
        },
        "audio": {
            "name": "Audio",
            "description": "Spatial audio content",
            "icon": "🔊",
            "formats": ["wav", "mp3", "ogg", "aac", "flac"],
            "use_cases": ["spatial_audio", "narration", "music", "sound_effects"]
        },
        "video": {
            "name": "Video",
            "description": "Video content for AR/VR",
            "icon": "🎥",
            "formats": ["mp4", "avi", "mov", "mkv", "webm"],
            "use_cases": ["360_video", "ar_video", "vr_video", "streaming"]
        },
        "image": {
            "name": "Image",
            "description": "Image content for AR/VR",
            "icon": "🖼️",
            "formats": ["jpg", "png", "tga", "dds", "ktx"],
            "use_cases": ["ar_markers", "textures", "ui", "backgrounds"]
        },
        "animation": {
            "name": "Animation",
            "description": "Animated content",
            "icon": "🎬",
            "formats": ["fbx", "gltf", "dae", "blend", "maya"],
            "use_cases": ["character_animation", "object_animation", "ui_animation"]
        },
        "interactive": {
            "name": "Interactive",
            "description": "Interactive AR/VR content",
            "icon": "🎮",
            "formats": ["unity", "unreal", "webxr", "native"],
            "use_cases": ["gaming", "education", "training", "simulation"]
        },
        "game": {
            "name": "Game",
            "description": "AR/VR game content",
            "icon": "🎮",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["entertainment", "gaming", "competition", "social"]
        },
        "simulation": {
            "name": "Simulation",
            "description": "AR/VR simulation content",
            "icon": "🧪",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["training", "education", "research", "testing"]
        },
        "education": {
            "name": "Education",
            "description": "Educational AR/VR content",
            "icon": "📚",
            "formats": ["unity", "unreal", "webxr", "native"],
            "use_cases": ["learning", "training", "demonstration", "exploration"]
        },
        "training": {
            "name": "Training",
            "description": "Training AR/VR content",
            "icon": "🎯",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["skill_development", "safety_training", "procedure_training"]
        },
        "marketing": {
            "name": "Marketing",
            "description": "Marketing AR/VR content",
            "icon": "📢",
            "formats": ["unity", "unreal", "webxr", "native"],
            "use_cases": ["product_demo", "brand_experience", "advertising"]
        },
        "entertainment": {
            "name": "Entertainment",
            "description": "Entertainment AR/VR content",
            "icon": "🎭",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["movies", "concerts", "theater", "experiences"]
        },
        "architecture": {
            "name": "Architecture",
            "description": "Architectural AR/VR content",
            "icon": "🏗️",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["building_visualization", "space_planning", "design_review"]
        },
        "medical": {
            "name": "Medical",
            "description": "Medical AR/VR content",
            "icon": "🏥",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["surgery_training", "anatomy_study", "therapy", "diagnosis"]
        },
        "engineering": {
            "name": "Engineering",
            "description": "Engineering AR/VR content",
            "icon": "⚙️",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["design_review", "assembly_training", "maintenance", "testing"]
        },
        "art": {
            "name": "Art",
            "description": "Artistic AR/VR content",
            "icon": "🎨",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["digital_art", "sculpture", "painting", "exhibitions"]
        },
        "music": {
            "name": "Music",
            "description": "Musical AR/VR content",
            "icon": "🎵",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["concerts", "music_creation", "instrument_training", "listening"]
        },
        "sports": {
            "name": "Sports",
            "description": "Sports AR/VR content",
            "icon": "⚽",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["training", "analysis", "fan_experience", "fitness"]
        },
        "travel": {
            "name": "Travel",
            "description": "Travel AR/VR content",
            "icon": "✈️",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["virtual_tours", "destination_preview", "cultural_experience"]
        },
        "shopping": {
            "name": "Shopping",
            "description": "Shopping AR/VR content",
            "icon": "🛒",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["product_visualization", "virtual_store", "try_before_buy"]
        },
        "social": {
            "name": "Social",
            "description": "Social AR/VR content",
            "icon": "👥",
            "formats": ["unity", "unreal", "native", "webxr"],
            "use_cases": ["virtual_meetings", "social_gaming", "shared_experiences"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "content_types": content_types,
            "total_types": len(content_types)
        },
        "message": "AR/VR content types retrieved successfully"
    }


@router.get("/interaction-types", response_model=Dict[str, Any])
async def get_ar_vr_interaction_types():
    """Get available AR/VR interaction types."""
    interaction_types = {
        "gaze": {
            "name": "Gaze",
            "description": "Eye gaze-based interaction",
            "icon": "👁️",
            "precision": "high",
            "comfort": "high",
            "speed": "medium"
        },
        "gesture": {
            "name": "Gesture",
            "description": "Hand gesture-based interaction",
            "icon": "✋",
            "precision": "medium",
            "comfort": "high",
            "speed": "medium"
        },
        "voice": {
            "name": "Voice",
            "description": "Voice command-based interaction",
            "icon": "🗣️",
            "precision": "medium",
            "comfort": "very_high",
            "speed": "fast"
        },
        "touch": {
            "name": "Touch",
            "description": "Touch-based interaction",
            "icon": "👆",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "controller": {
            "name": "Controller",
            "description": "Controller-based interaction",
            "icon": "🎮",
            "precision": "high",
            "comfort": "medium",
            "speed": "fast"
        },
        "hand_tracking": {
            "name": "Hand Tracking",
            "description": "Direct hand tracking interaction",
            "icon": "✋",
            "precision": "high",
            "comfort": "high",
            "speed": "medium"
        },
        "eye_tracking": {
            "name": "Eye Tracking",
            "description": "Eye movement-based interaction",
            "icon": "👁️",
            "precision": "very_high",
            "comfort": "high",
            "speed": "fast"
        },
        "head_tracking": {
            "name": "Head Tracking",
            "description": "Head movement-based interaction",
            "icon": "👤",
            "precision": "medium",
            "comfort": "high",
            "speed": "fast"
        },
        "body_tracking": {
            "name": "Body Tracking",
            "description": "Full body movement tracking",
            "icon": "🏃",
            "precision": "medium",
            "comfort": "medium",
            "speed": "medium"
        },
        "face_tracking": {
            "name": "Face Tracking",
            "description": "Facial expression tracking",
            "icon": "😊",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "object_tracking": {
            "name": "Object Tracking",
            "description": "Physical object tracking",
            "icon": "📦",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "marker_tracking": {
            "name": "Marker Tracking",
            "description": "Visual marker tracking",
            "icon": "🎯",
            "precision": "very_high",
            "comfort": "high",
            "speed": "fast"
        },
        "plane_tracking": {
            "name": "Plane Tracking",
            "description": "Surface plane tracking",
            "icon": "📐",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "image_tracking": {
            "name": "Image Tracking",
            "description": "Image recognition tracking",
            "icon": "🖼️",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "surface_tracking": {
            "name": "Surface Tracking",
            "description": "Surface detection and tracking",
            "icon": "🏠",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "light_tracking": {
            "name": "Light Tracking",
            "description": "Light source tracking",
            "icon": "💡",
            "precision": "medium",
            "comfort": "high",
            "speed": "fast"
        },
        "depth_tracking": {
            "name": "Depth Tracking",
            "description": "Depth information tracking",
            "icon": "📏",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "slam": {
            "name": "SLAM",
            "description": "Simultaneous Localization and Mapping",
            "icon": "🗺️",
            "precision": "high",
            "comfort": "high",
            "speed": "medium"
        },
        "vio": {
            "name": "VIO",
            "description": "Visual-Inertial Odometry",
            "icon": "📹",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "gps": {
            "name": "GPS",
            "description": "Global Positioning System",
            "icon": "📍",
            "precision": "medium",
            "comfort": "high",
            "speed": "fast"
        },
        "compass": {
            "name": "Compass",
            "description": "Magnetic compass tracking",
            "icon": "🧭",
            "precision": "medium",
            "comfort": "high",
            "speed": "fast"
        },
        "accelerometer": {
            "name": "Accelerometer",
            "description": "Acceleration sensor tracking",
            "icon": "⚡",
            "precision": "medium",
            "comfort": "high",
            "speed": "fast"
        },
        "gyroscope": {
            "name": "Gyroscope",
            "description": "Rotation sensor tracking",
            "icon": "🌀",
            "precision": "high",
            "comfort": "high",
            "speed": "fast"
        },
        "magnetometer": {
            "name": "Magnetometer",
            "description": "Magnetic field sensor tracking",
            "icon": "🧲",
            "precision": "medium",
            "comfort": "high",
            "speed": "fast"
        }
    }
    
    return {
        "success": True,
        "data": {
            "interaction_types": interaction_types,
            "total_types": len(interaction_types)
        },
        "message": "AR/VR interaction types retrieved successfully"
    }


@router.get("/tracking-types", response_model=Dict[str, Any])
async def get_ar_vr_tracking_types():
    """Get available AR/VR tracking types."""
    tracking_types = {
        "inside_out": {
            "name": "Inside-Out Tracking",
            "description": "Device-based tracking using onboard sensors",
            "icon": "📱",
            "accuracy": "high",
            "latency": "low",
            "setup": "easy"
        },
        "outside_in": {
            "name": "Outside-In Tracking",
            "description": "External sensor-based tracking",
            "icon": "📡",
            "accuracy": "very_high",
            "latency": "very_low",
            "setup": "complex"
        },
        "marker_based": {
            "name": "Marker-Based Tracking",
            "description": "Tracking using visual markers",
            "icon": "🎯",
            "accuracy": "high",
            "latency": "low",
            "setup": "medium"
        },
        "markerless": {
            "name": "Markerless Tracking",
            "description": "Tracking without visual markers",
            "icon": "👁️",
            "accuracy": "medium",
            "latency": "medium",
            "setup": "easy"
        },
        "slam": {
            "name": "SLAM",
            "description": "Simultaneous Localization and Mapping",
            "icon": "🗺️",
            "accuracy": "high",
            "latency": "medium",
            "setup": "medium"
        },
        "vio": {
            "name": "VIO",
            "description": "Visual-Inertial Odometry",
            "icon": "📹",
            "accuracy": "high",
            "latency": "low",
            "setup": "medium"
        },
        "gps": {
            "name": "GPS",
            "description": "Global Positioning System",
            "icon": "📍",
            "accuracy": "medium",
            "latency": "low",
            "setup": "easy"
        },
        "compass": {
            "name": "Compass",
            "description": "Magnetic compass tracking",
            "icon": "🧭",
            "accuracy": "medium",
            "latency": "low",
            "setup": "easy"
        },
        "imu": {
            "name": "IMU",
            "description": "Inertial Measurement Unit",
            "icon": "📱",
            "accuracy": "medium",
            "latency": "very_low",
            "setup": "easy"
        },
        "camera": {
            "name": "Camera",
            "description": "Camera-based tracking",
            "icon": "📷",
            "accuracy": "high",
            "latency": "medium",
            "setup": "medium"
        },
        "depth": {
            "name": "Depth",
            "description": "Depth sensor tracking",
            "icon": "📏",
            "accuracy": "high",
            "latency": "medium",
            "setup": "medium"
        },
        "lidar": {
            "name": "LiDAR",
            "description": "Light Detection and Ranging",
            "icon": "📡",
            "accuracy": "very_high",
            "latency": "medium",
            "setup": "complex"
        },
        "structured_light": {
            "name": "Structured Light",
            "description": "Structured light pattern tracking",
            "icon": "💡",
            "accuracy": "high",
            "latency": "medium",
            "setup": "medium"
        },
        "time_of_flight": {
            "name": "Time of Flight",
            "description": "Time of flight sensor tracking",
            "icon": "⏱️",
            "accuracy": "high",
            "latency": "low",
            "setup": "medium"
        },
        "stereo": {
            "name": "Stereo",
            "description": "Stereo camera tracking",
            "icon": "👁️",
            "accuracy": "high",
            "latency": "medium",
            "setup": "medium"
        },
        "monocular": {
            "name": "Monocular",
            "description": "Single camera tracking",
            "icon": "📷",
            "accuracy": "medium",
            "latency": "low",
            "setup": "easy"
        },
        "multi_camera": {
            "name": "Multi-Camera",
            "description": "Multiple camera tracking",
            "icon": "📹",
            "accuracy": "very_high",
            "latency": "medium",
            "setup": "complex"
        },
        "fusion": {
            "name": "Sensor Fusion",
            "description": "Multiple sensor fusion tracking",
            "icon": "🔗",
            "accuracy": "very_high",
            "latency": "low",
            "setup": "complex"
        }
    }
    
    return {
        "success": True,
        "data": {
            "tracking_types": tracking_types,
            "total_types": len(tracking_types)
        },
        "message": "AR/VR tracking types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_ar_vr_health(
    ar_vr_service: AdvancedARVRService = Depends(get_ar_vr_service),
    current_user: CurrentUserDep = Depends()
):
    """Get AR/VR system health status."""
    try:
        # Get AR/VR stats
        stats = await ar_vr_service.get_ar_vr_stats()
        
        # Calculate health metrics
        total_scenes = stats["data"].get("total_scenes", 0)
        total_objects = stats["data"].get("total_objects", 0)
        total_sessions = stats["data"].get("total_sessions", 0)
        total_interactions = stats["data"].get("total_interactions", 0)
        active_sessions = stats["data"].get("active_sessions", 0)
        scenes_by_type = stats["data"].get("scenes_by_type", {})
        objects_by_content_type = stats["data"].get("objects_by_content_type", {})
        sessions_by_device_type = stats["data"].get("sessions_by_device_type", {})
        
        # Calculate health score
        health_score = 100
        
        # Check scene diversity
        if len(scenes_by_type) < 2:
            health_score -= 20
        elif len(scenes_by_type) > 10:
            health_score -= 5
        
        # Check content diversity
        if len(objects_by_content_type) < 3:
            health_score -= 25
        elif len(objects_by_content_type) > 15:
            health_score -= 5
        
        # Check device diversity
        if len(sessions_by_device_type) < 2:
            health_score -= 20
        elif len(sessions_by_device_type) > 8:
            health_score -= 5
        
        # Check session activity
        if total_sessions > 0:
            interactions_per_session = total_interactions / total_sessions
            if interactions_per_session < 5:
                health_score -= 30
            elif interactions_per_session > 1000:
                health_score -= 10
        
        # Check active sessions
        if total_sessions > 0:
            active_ratio = active_sessions / total_sessions
            if active_ratio < 0.1:
                health_score -= 25
            elif active_ratio > 0.8:
                health_score -= 5
        
        # Check object distribution
        if total_scenes > 0:
            objects_per_scene = total_objects / total_scenes
            if objects_per_scene < 1:
                health_score -= 20
            elif objects_per_scene > 100:
                health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_scenes": total_scenes,
                "total_objects": total_objects,
                "total_sessions": total_sessions,
                "total_interactions": total_interactions,
                "active_sessions": active_sessions,
                "scene_diversity": len(scenes_by_type),
                "content_diversity": len(objects_by_content_type),
                "device_diversity": len(sessions_by_device_type),
                "interactions_per_session": interactions_per_session if total_sessions > 0 else 0,
                "active_ratio": active_ratio if total_sessions > 0 else 0,
                "objects_per_scene": objects_per_scene if total_scenes > 0 else 0,
                "scenes_by_type": scenes_by_type,
                "objects_by_content_type": objects_by_content_type,
                "sessions_by_device_type": sessions_by_device_type,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "AR/VR health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR/VR health status"
        )
























