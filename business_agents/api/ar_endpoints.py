"""
Augmented Reality API Endpoints
===============================

REST API endpoints for augmented reality integration,
AR content management, and immersive interactions.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.augmented_reality_service import (
    AugmentedRealityService, ARDeviceType, ARContentType, ARInteractionType, ARQuality,
    ARDevice, ARContent, ARSession, ARInteraction
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/ar", tags=["Augmented Reality"])

# Pydantic models
class ARDeviceRegistrationRequest(BaseModel):
    name: str = Field(..., description="Device name")
    device_type: str = Field(..., description="Device type")
    capabilities: List[str] = Field(default_factory=list, description="Device capabilities")
    resolution: List[int] = Field(..., description="Device resolution [width, height]")
    field_of_view: float = Field(..., description="Field of view in degrees")
    tracking_accuracy: float = Field(..., description="Tracking accuracy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Device metadata")

class ARContentCreateRequest(BaseModel):
    content_type: str = Field(..., description="Content type")
    title: str = Field(..., description="Content title")
    description: str = Field(..., description="Content description")
    data: Any = Field(..., description="Content data")
    position: List[float] = Field([0, 0, 0], description="3D position [x, y, z]")
    rotation: List[float] = Field([0, 0, 0], description="3D rotation [x, y, z]")
    scale: List[float] = Field([1, 1, 1], description="3D scale [x, y, z]")
    quality: str = Field("high", description="Content quality")
    interactive: bool = Field(True, description="Interactive content")

class ARSessionStartRequest(BaseModel):
    device_id: str = Field(..., description="AR device ID")
    user_id: str = Field(..., description="User ID")
    content_list: List[str] = Field(default_factory=list, description="Content IDs to load")

class ARInteractionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    interaction_type: str = Field(..., description="Interaction type")
    data: Dict[str, Any] = Field(..., description="Interaction data")
    confidence: float = Field(1.0, description="Interaction confidence")

class WorkflowVisualizationRequest(BaseModel):
    workflow_data: Dict[str, Any] = Field(..., description="Workflow data")
    device_id: str = Field(..., description="Target device ID")

class DataVisualizationRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Data to visualize")
    visualization_type: str = Field(..., description="Visualization type")
    device_id: str = Field(..., description="Target device ID")

# Global AR service instance
ar_service = None

def get_ar_service() -> AugmentedRealityService:
    """Get global AR service instance."""
    global ar_service
    if ar_service is None:
        ar_service = AugmentedRealityService({
            "augmented_reality": {
                "max_devices": 100,
                "max_content_per_session": 50,
                "gesture_recognition_enabled": True,
                "face_detection_enabled": True,
                "pose_estimation_enabled": True,
                "spatial_mapping_enabled": True,
                "real_time_processing": True
            }
        })
    return ar_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_ar_service(
    current_user: User = Depends(require_permission("ar:manage"))
):
    """Initialize the augmented reality service."""
    
    ar_service = get_ar_service()
    
    try:
        await ar_service.initialize()
        return {"message": "Augmented Reality Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize AR service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_ar_status(
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get augmented reality service status."""
    
    ar_service = get_ar_service()
    
    try:
        status = await ar_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR status: {str(e)}")

@router.post("/devices/register", response_model=Dict[str, str])
async def register_ar_device(
    request: ARDeviceRegistrationRequest,
    current_user: User = Depends(require_permission("ar:manage"))
):
    """Register a new AR device."""
    
    ar_service = get_ar_service()
    
    try:
        # Convert string to enum
        device_type = ARDeviceType(request.device_type)
        
        # Create AR device
        device = ARDevice(
            device_id="",  # Will be generated
            name=request.name,
            device_type=device_type,
            capabilities=request.capabilities,
            resolution=tuple(request.resolution),
            field_of_view=request.field_of_view,
            tracking_accuracy=request.tracking_accuracy,
            battery_level=100.0,
            connection_status="connected",
            last_seen=datetime.utcnow(),
            metadata=request.metadata
        )
        
        # Register device
        device_id = await ar_service.register_ar_device(device)
        
        return {
            "message": "AR device registered successfully",
            "device_id": device_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register AR device: {str(e)}")

@router.delete("/devices/{device_id}", response_model=Dict[str, str])
async def unregister_ar_device(
    device_id: str,
    current_user: User = Depends(require_permission("ar:manage"))
):
    """Unregister an AR device."""
    
    ar_service = get_ar_service()
    
    try:
        success = await ar_service.unregister_ar_device(device_id)
        
        if success:
            return {"message": f"AR device {device_id} unregistered successfully"}
        else:
            raise HTTPException(status_code=404, detail="AR device not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unregister AR device: {str(e)}")

@router.get("/devices", response_model=List[Dict[str, Any]])
async def get_ar_devices(
    device_type: Optional[str] = Query(None, description="Filter by device type"),
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get AR devices."""
    
    ar_service = get_ar_service()
    
    try:
        # Convert string to enum if provided
        device_type_enum = ARDeviceType(device_type) if device_type else None
        
        # Get devices
        devices = await ar_service.get_ar_devices(device_type_enum)
        
        result = []
        for device in devices:
            device_dict = {
                "device_id": device.device_id,
                "name": device.name,
                "device_type": device.device_type.value,
                "capabilities": device.capabilities,
                "resolution": list(device.resolution),
                "field_of_view": device.field_of_view,
                "tracking_accuracy": device.tracking_accuracy,
                "battery_level": device.battery_level,
                "connection_status": device.connection_status,
                "last_seen": device.last_seen.isoformat(),
                "metadata": device.metadata
            }
            result.append(device_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR devices: {str(e)}")

@router.get("/devices/{device_id}", response_model=Dict[str, Any])
async def get_ar_device(
    device_id: str,
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get specific AR device."""
    
    ar_service = get_ar_service()
    
    try:
        device = await ar_service.get_ar_device(device_id)
        
        if not device:
            raise HTTPException(status_code=404, detail="AR device not found")
        
        return {
            "device_id": device.device_id,
            "name": device.name,
            "device_type": device.device_type.value,
            "capabilities": device.capabilities,
            "resolution": list(device.resolution),
            "field_of_view": device.field_of_view,
            "tracking_accuracy": device.tracking_accuracy,
            "battery_level": device.battery_level,
            "connection_status": device.connection_status,
            "last_seen": device.last_seen.isoformat(),
            "metadata": device.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR device: {str(e)}")

@router.post("/content/create", response_model=Dict[str, Any])
async def create_ar_content(
    request: ARContentCreateRequest,
    current_user: User = Depends(require_permission("ar:create"))
):
    """Create AR content."""
    
    ar_service = get_ar_service()
    
    try:
        # Convert string to enum
        content_type = ARContentType(request.content_type)
        quality = ARQuality(request.quality)
        
        # Create AR content
        content = await ar_service.create_ar_content(
            content_type=content_type,
            title=request.title,
            description=request.description,
            data=request.data,
            position=tuple(request.position),
            rotation=tuple(request.rotation),
            scale=tuple(request.scale),
            quality=quality,
            interactive=request.interactive
        )
        
        return {
            "content_id": content.content_id,
            "content_type": content.content_type.value,
            "title": content.title,
            "description": content.description,
            "position": list(content.position),
            "rotation": list(content.rotation),
            "scale": list(content.scale),
            "quality": content.quality.value,
            "interactive": content.interactive,
            "created_at": content.created_at.isoformat(),
            "metadata": content.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create AR content: {str(e)}")

@router.get("/content", response_model=List[Dict[str, Any]])
async def get_ar_content_list(
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get AR content list."""
    
    ar_service = get_ar_service()
    
    try:
        # Convert string to enum if provided
        content_type_enum = ARContentType(content_type) if content_type else None
        
        # Get content
        content_list = await ar_service.get_ar_content_list(content_type_enum)
        
        result = []
        for content in content_list:
            content_dict = {
                "content_id": content.content_id,
                "content_type": content.content_type.value,
                "title": content.title,
                "description": content.description,
                "position": list(content.position),
                "rotation": list(content.rotation),
                "scale": list(content.scale),
                "quality": content.quality.value,
                "interactive": content.interactive,
                "created_at": content.created_at.isoformat(),
                "metadata": content.metadata
            }
            result.append(content_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR content: {str(e)}")

@router.get("/content/{content_id}", response_model=Dict[str, Any])
async def get_ar_content(
    content_id: str,
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get specific AR content."""
    
    ar_service = get_ar_service()
    
    try:
        content = await ar_service.get_ar_content(content_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="AR content not found")
        
        return {
            "content_id": content.content_id,
            "content_type": content.content_type.value,
            "title": content.title,
            "description": content.description,
            "data": content.data,
            "position": list(content.position),
            "rotation": list(content.rotation),
            "scale": list(content.scale),
            "quality": content.quality.value,
            "interactive": content.interactive,
            "created_at": content.created_at.isoformat(),
            "metadata": content.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR content: {str(e)}")

@router.post("/sessions/start", response_model=Dict[str, Any])
async def start_ar_session(
    request: ARSessionStartRequest,
    current_user: User = Depends(require_permission("ar:execute"))
):
    """Start AR session."""
    
    ar_service = get_ar_service()
    
    try:
        # Start AR session
        session = await ar_service.start_ar_session(
            device_id=request.device_id,
            user_id=request.user_id,
            content_list=request.content_list
        )
        
        return {
            "session_id": session.session_id,
            "device_id": session.device_id,
            "user_id": session.user_id,
            "content_list": session.content_list,
            "start_time": session.start_time.isoformat(),
            "metadata": session.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start AR session: {str(e)}")

@router.post("/sessions/{session_id}/end", response_model=Dict[str, str])
async def end_ar_session(
    session_id: str,
    current_user: User = Depends(require_permission("ar:execute"))
):
    """End AR session."""
    
    ar_service = get_ar_service()
    
    try:
        success = await ar_service.end_ar_session(session_id)
        
        if success:
            return {"message": f"AR session {session_id} ended successfully"}
        else:
            raise HTTPException(status_code=404, detail="AR session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end AR session: {str(e)}")

@router.get("/sessions", response_model=List[Dict[str, Any]])
async def get_ar_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get AR sessions."""
    
    ar_service = get_ar_service()
    
    try:
        sessions = await ar_service.get_ar_sessions(user_id)
        
        result = []
        for session in sessions:
            session_dict = {
                "session_id": session.session_id,
                "device_id": session.device_id,
                "user_id": session.user_id,
                "content_list": session.content_list,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "duration": session.duration,
                "interactions": session.interactions,
                "performance_metrics": session.performance_metrics,
                "metadata": session.metadata
            }
            result.append(session_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR sessions: {str(e)}")

@router.post("/interactions/record", response_model=Dict[str, Any])
async def record_ar_interaction(
    request: ARInteractionRequest,
    current_user: User = Depends(require_permission("ar:execute"))
):
    """Record AR interaction."""
    
    ar_service = get_ar_service()
    
    try:
        # Convert string to enum
        interaction_type = ARInteractionType(request.interaction_type)
        
        # Record interaction
        interaction = await ar_service.record_ar_interaction(
            session_id=request.session_id,
            interaction_type=interaction_type,
            data=request.data,
            confidence=request.confidence
        )
        
        return {
            "interaction_id": interaction.interaction_id,
            "session_id": interaction.session_id,
            "interaction_type": interaction.interaction_type.value,
            "timestamp": interaction.timestamp.isoformat(),
            "data": interaction.data,
            "confidence": interaction.confidence,
            "metadata": interaction.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record AR interaction: {str(e)}")

@router.get("/interactions", response_model=List[Dict[str, Any]])
async def get_ar_interactions(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get AR interactions."""
    
    ar_service = get_ar_service()
    
    try:
        interactions = await ar_service.get_ar_interactions(session_id)
        
        result = []
        for interaction in interactions:
            interaction_dict = {
                "interaction_id": interaction.interaction_id,
                "session_id": interaction.session_id,
                "interaction_type": interaction.interaction_type.value,
                "timestamp": interaction.timestamp.isoformat(),
                "data": interaction.data,
                "confidence": interaction.confidence,
                "metadata": interaction.metadata
            }
            result.append(interaction_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR interactions: {str(e)}")

@router.post("/processing/gesture", response_model=Dict[str, Any])
async def process_gesture_recognition(
    file: UploadFile = File(...),
    current_user: User = Depends(require_permission("ar:process"))
):
    """Process gesture recognition from image."""
    
    ar_service = get_ar_service()
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Process gesture recognition
        result = await ar_service.process_gesture(image_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process gesture: {str(e)}")

@router.post("/processing/face", response_model=Dict[str, Any])
async def process_face_detection(
    file: UploadFile = File(...),
    current_user: User = Depends(require_permission("ar:process"))
):
    """Process face detection from image."""
    
    ar_service = get_ar_service()
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Process face detection
        result = await ar_service.process_face_detection(image_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process face detection: {str(e)}")

@router.post("/processing/pose", response_model=Dict[str, Any])
async def process_pose_estimation(
    file: UploadFile = File(...),
    current_user: User = Depends(require_permission("ar:process"))
):
    """Process pose estimation from image."""
    
    ar_service = get_ar_service()
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Process pose estimation
        result = await ar_service.process_pose_estimation(image_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process pose estimation: {str(e)}")

@router.post("/visualizations/workflow", response_model=Dict[str, Any])
async def create_workflow_visualization(
    request: WorkflowVisualizationRequest,
    current_user: User = Depends(require_permission("ar:create"))
):
    """Create AR workflow visualization."""
    
    ar_service = get_ar_service()
    
    try:
        # Create workflow visualization
        content = await ar_service.create_workflow_visualization(
            workflow_data=request.workflow_data,
            device_id=request.device_id
        )
        
        return {
            "content_id": content.content_id,
            "content_type": content.content_type.value,
            "title": content.title,
            "description": content.description,
            "data": content.data,
            "position": list(content.position),
            "rotation": list(content.rotation),
            "scale": list(content.scale),
            "quality": content.quality.value,
            "interactive": content.interactive,
            "created_at": content.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow visualization: {str(e)}")

@router.post("/visualizations/data", response_model=Dict[str, Any])
async def create_data_visualization(
    request: DataVisualizationRequest,
    current_user: User = Depends(require_permission("ar:create"))
):
    """Create AR data visualization."""
    
    ar_service = get_ar_service()
    
    try:
        # Create data visualization
        content = await ar_service.create_data_visualization(
            data=request.data,
            visualization_type=request.visualization_type,
            device_id=request.device_id
        )
        
        return {
            "content_id": content.content_id,
            "content_type": content.content_type.value,
            "title": content.title,
            "description": content.description,
            "data": content.data,
            "position": list(content.position),
            "rotation": list(content.rotation),
            "scale": list(content.scale),
            "quality": content.quality.value,
            "interactive": content.interactive,
            "created_at": content.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create data visualization: {str(e)}")

@router.get("/device-types", response_model=List[Dict[str, Any]])
async def get_ar_device_types(
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get available AR device types."""
    
    try:
        device_types = [
            {
                "type": "hololens",
                "name": "Microsoft HoloLens",
                "description": "Mixed reality headset with spatial computing",
                "capabilities": ["hand_tracking", "eye_tracking", "voice", "spatial_mapping"],
                "resolution": [2048, 1080],
                "field_of_view": 52.0
            },
            {
                "type": "magic_leap",
                "name": "Magic Leap",
                "description": "Spatial computing platform",
                "capabilities": ["hand_tracking", "eye_tracking", "voice", "spatial_mapping"],
                "resolution": [1440, 1760],
                "field_of_view": 70.0
            },
            {
                "type": "oculus",
                "name": "Meta Oculus",
                "description": "Virtual reality headset",
                "capabilities": ["hand_tracking", "voice", "spatial_mapping"],
                "resolution": [1832, 1920],
                "field_of_view": 90.0
            },
            {
                "type": "mobile_ar",
                "name": "Mobile AR",
                "description": "Mobile device augmented reality",
                "capabilities": ["hand_tracking", "face_detection", "object_recognition"],
                "resolution": [1920, 1080],
                "field_of_view": 60.0
            },
            {
                "type": "web_ar",
                "name": "Web AR",
                "description": "Browser-based augmented reality",
                "capabilities": ["hand_tracking", "face_detection", "object_recognition"],
                "resolution": [1280, 720],
                "field_of_view": 45.0
            }
        ]
        
        return device_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR device types: {str(e)}")

@router.get("/content-types", response_model=List[Dict[str, Any]])
async def get_ar_content_types(
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get available AR content types."""
    
    try:
        content_types = [
            {
                "type": "workflow_visualization",
                "name": "Workflow Visualization",
                "description": "Interactive 3D workflow visualization",
                "use_cases": ["process_management", "workflow_analysis", "training"]
            },
            {
                "type": "data_visualization",
                "name": "Data Visualization",
                "description": "Interactive 3D data visualization",
                "use_cases": ["analytics", "reporting", "data_exploration"]
            },
            {
                "type": "interactive_3d",
                "name": "Interactive 3D",
                "description": "Interactive 3D objects and models",
                "use_cases": ["product_demo", "training", "simulation"]
            },
            {
                "type": "annotation",
                "name": "Annotation",
                "description": "AR annotations and labels",
                "use_cases": ["guidance", "documentation", "training"]
            },
            {
                "type": "guidance",
                "name": "Guidance",
                "description": "Step-by-step AR guidance",
                "use_cases": ["maintenance", "assembly", "training"]
            },
            {
                "type": "collaboration",
                "name": "Collaboration",
                "description": "Multi-user AR collaboration",
                "use_cases": ["remote_meetings", "design_review", "training"]
            }
        ]
        
        return content_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR content types: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_ar_analytics(
    current_user: User = Depends(require_permission("ar:view"))
):
    """Get AR analytics."""
    
    ar_service = get_ar_service()
    
    try:
        # Get service status
        status = await ar_service.get_service_status()
        
        # Get sessions
        sessions = await ar_service.get_ar_sessions()
        
        # Get interactions
        interactions = await ar_service.get_ar_interactions()
        
        # Calculate analytics
        analytics = {
            "total_devices": status.get("total_devices", 0),
            "connected_devices": status.get("connected_devices", 0),
            "total_content": status.get("total_content", 0),
            "active_sessions": status.get("active_sessions", 0),
            "total_sessions": status.get("total_sessions", 0),
            "total_interactions": status.get("total_interactions", 0),
            "average_session_duration": sum(s.duration or 0 for s in sessions) / max(len(sessions), 1),
            "interaction_types": {},
            "device_usage": {},
            "content_usage": {},
            "user_engagement": "high",
            "gesture_recognition_enabled": status.get("gesture_recognition_enabled", False),
            "face_detection_enabled": status.get("face_detection_enabled", False),
            "pose_estimation_enabled": status.get("pose_estimation_enabled", False),
            "spatial_mapping_enabled": status.get("spatial_mapping_enabled", False),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Calculate interaction types
        for interaction in interactions:
            interaction_type = interaction.interaction_type.value
            if interaction_type not in analytics["interaction_types"]:
                analytics["interaction_types"][interaction_type] = 0
            analytics["interaction_types"][interaction_type] += 1
            
        # Calculate device usage
        for session in sessions:
            device_id = session.device_id
            if device_id not in analytics["device_usage"]:
                analytics["device_usage"][device_id] = 0
            analytics["device_usage"][device_id] += 1
            
        # Calculate content usage
        for session in sessions:
            for content_id in session.content_list:
                if content_id not in analytics["content_usage"]:
                    analytics["content_usage"][content_id] = 0
                analytics["content_usage"][content_id] += 1
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AR analytics: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def ar_health_check():
    """AR service health check."""
    
    ar_service = get_ar_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(ar_service, 'ar_devices') and len(ar_service.ar_devices) >= 0
        
        # Get service status
        status = await ar_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "total_devices": status.get("total_devices", 0),
            "connected_devices": status.get("connected_devices", 0),
            "total_content": status.get("total_content", 0),
            "active_sessions": status.get("active_sessions", 0),
            "gesture_recognition_enabled": status.get("gesture_recognition_enabled", False),
            "face_detection_enabled": status.get("face_detection_enabled", False),
            "pose_estimation_enabled": status.get("pose_estimation_enabled", False),
            "spatial_mapping_enabled": status.get("spatial_mapping_enabled", False),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }




























