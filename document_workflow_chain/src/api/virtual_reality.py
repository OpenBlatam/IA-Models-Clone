"""
Virtual Reality API - Advanced Implementation
===========================================

FastAPI endpoints for virtual reality operations.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.virtual_reality_service import (
    vr_service,
    VRDeviceType,
    VRExperienceType,
    VRInteractionType
)

logger = logging.getLogger(__name__)

# Pydantic models
class VRDeviceRegistration(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    device_type: VRDeviceType = Field(..., description="Type of VR device")
    device_name: str = Field(..., description="Name of the device")
    capabilities: List[str] = Field(..., description="Device capabilities")
    tracking_types: List[str] = Field(..., description="Supported tracking types")
    location: Dict[str, float] = Field(..., description="Device location coordinates")
    device_info: Dict[str, Any] = Field(default_factory=dict, description="Additional device information")

class VRSessionCreation(BaseModel):
    device_id: str = Field(..., description="Device ID for the session")
    session_name: str = Field(..., description="Name of the VR session")
    experience_type: VRExperienceType = Field(..., description="Type of VR experience")
    session_config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")

class VREnvironmentCreation(BaseModel):
    environment_name: str = Field(..., description="Name of the VR environment")
    environment_type: str = Field(..., description="Type of environment")
    environment_data: Dict[str, Any] = Field(..., description="Environment data")
    physics_config: Dict[str, Any] = Field(default_factory=dict, description="Physics configuration")

class VRAvatarCreation(BaseModel):
    avatar_name: str = Field(..., description="Name of the VR avatar")
    avatar_type: str = Field(..., description="Type of avatar")
    avatar_data: Dict[str, Any] = Field(..., description="Avatar data")
    customization: Dict[str, Any] = Field(default_factory=dict, description="Avatar customization")

class VRWorkflowCreation(BaseModel):
    workflow_name: str = Field(..., description="Name of the VR workflow")
    workflow_type: str = Field(..., description="Type of workflow")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow triggers")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow conditions")

class VRWorkflowExecution(BaseModel):
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    session_id: str = Field(..., description="ID of the VR session")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")

class VRInteractionTracking(BaseModel):
    interaction_type: VRInteractionType = Field(..., description="Type of VR interaction")
    interaction_data: Dict[str, Any] = Field(..., description="Interaction data")

class VRRoomCreation(BaseModel):
    room_name: str = Field(..., description="Name of the VR room")
    room_type: str = Field(..., description="Type of room")
    room_config: Dict[str, Any] = Field(..., description="Room configuration")
    max_occupants: int = Field(default=10, description="Maximum number of occupants")

class VRRoomJoin(BaseModel):
    room_id: str = Field(..., description="ID of the VR room")
    user_id: str = Field(..., description="ID of the user joining")

# Create router
router = APIRouter(prefix="/vr", tags=["Virtual Reality"])

@router.post("/devices/register")
async def register_vr_device(device_data: VRDeviceRegistration) -> Dict[str, Any]:
    """Register a new VR device"""
    try:
        device_id = await vr_service.register_vr_device(
            device_id=device_data.device_id,
            device_type=device_data.device_type,
            device_name=device_data.device_name,
            capabilities=device_data.capabilities,
            tracking_types=device_data.tracking_types,
            location=device_data.location,
            device_info=device_data.device_info
        )
        
        return {
            "success": True,
            "device_id": device_id,
            "message": "VR device registered successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to register VR device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/create")
async def create_vr_session(session_data: VRSessionCreation) -> Dict[str, Any]:
    """Create a new VR session"""
    try:
        session_id = await vr_service.create_vr_session(
            device_id=session_data.device_id,
            session_name=session_data.session_name,
            experience_type=session_data.experience_type,
            session_config=session_data.session_config
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "VR session created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create VR session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/environments/create")
async def create_vr_environment(environment_data: VREnvironmentCreation) -> Dict[str, Any]:
    """Create a VR environment"""
    try:
        environment_id = await vr_service.create_vr_environment(
            environment_name=environment_data.environment_name,
            environment_type=environment_data.environment_type,
            environment_data=environment_data.environment_data,
            physics_config=environment_data.physics_config
        )
        
        return {
            "success": True,
            "environment_id": environment_id,
            "message": "VR environment created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create VR environment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/avatars/create")
async def create_vr_avatar(avatar_data: VRAvatarCreation) -> Dict[str, Any]:
    """Create a VR avatar"""
    try:
        avatar_id = await vr_service.create_vr_avatar(
            avatar_name=avatar_data.avatar_name,
            avatar_type=avatar_data.avatar_type,
            avatar_data=avatar_data.avatar_data,
            customization=avatar_data.customization
        )
        
        return {
            "success": True,
            "avatar_id": avatar_id,
            "message": "VR avatar created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create VR avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/create")
async def create_vr_workflow(workflow_data: VRWorkflowCreation) -> Dict[str, Any]:
    """Create a VR workflow"""
    try:
        workflow_id = await vr_service.create_vr_workflow(
            workflow_name=workflow_data.workflow_name,
            workflow_type=workflow_data.workflow_type,
            steps=workflow_data.steps,
            triggers=workflow_data.triggers,
            conditions=workflow_data.conditions
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "VR workflow created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create VR workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/execute")
async def execute_vr_workflow(execution_data: VRWorkflowExecution) -> Dict[str, Any]:
    """Execute a VR workflow"""
    try:
        result = await vr_service.execute_vr_workflow(
            workflow_id=execution_data.workflow_id,
            session_id=execution_data.session_id,
            context=execution_data.context
        )
        
        return {
            "success": True,
            "result": result,
            "message": "VR workflow executed successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to execute VR workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/interactions")
async def track_vr_interaction(
    session_id: str,
    interaction_data: VRInteractionTracking
) -> Dict[str, Any]:
    """Track VR interaction"""
    try:
        interaction_id = await vr_service.track_vr_interaction(
            session_id=session_id,
            interaction_type=interaction_data.interaction_type,
            interaction_data=interaction_data.interaction_data
        )
        
        return {
            "success": True,
            "interaction_id": interaction_id,
            "message": "VR interaction tracked successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to track VR interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rooms/create")
async def create_vr_room(room_data: VRRoomCreation) -> Dict[str, Any]:
    """Create a VR room for social experiences"""
    try:
        room_id = await vr_service.create_vr_room(
            room_name=room_data.room_name,
            room_type=room_data.room_type,
            room_config=room_data.room_config,
            max_occupants=room_data.max_occupants
        )
        
        return {
            "success": True,
            "room_id": room_id,
            "message": "VR room created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create VR room: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rooms/join")
async def join_vr_room(join_data: VRRoomJoin) -> Dict[str, Any]:
    """Join a VR room"""
    try:
        result = await vr_service.join_vr_room(
            room_id=join_data.room_id,
            session_id="",  # This should be passed from the session context
            user_id=join_data.user_id
        )
        
        return {
            "success": True,
            "result": result,
            "message": "Joined VR room successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to join VR room: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/end")
async def end_vr_session(session_id: str) -> Dict[str, Any]:
    """End VR session"""
    try:
        result = await vr_service.end_vr_session(session_id=session_id)
        
        return {
            "success": True,
            "result": result,
            "message": "VR session ended successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to end VR session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/analytics")
async def get_vr_session_analytics(session_id: str) -> Dict[str, Any]:
    """Get VR session analytics"""
    try:
        analytics = await vr_service.get_vr_session_analytics(session_id=session_id)
        
        if analytics is None:
            raise HTTPException(status_code=404, detail="VR session not found")
        
        return {
            "success": True,
            "analytics": analytics,
            "message": "VR session analytics retrieved successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get VR session analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_vr_stats() -> Dict[str, Any]:
    """Get VR service statistics"""
    try:
        stats = await vr_service.get_vr_stats()
        
        return {
            "success": True,
            "stats": stats,
            "message": "VR statistics retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices")
async def get_vr_devices() -> Dict[str, Any]:
    """Get all registered VR devices"""
    try:
        devices = list(vr_service.vr_devices.values())
        
        return {
            "success": True,
            "devices": devices,
            "count": len(devices),
            "message": "VR devices retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def get_vr_sessions() -> Dict[str, Any]:
    """Get all VR sessions"""
    try:
        sessions = list(vr_service.vr_sessions.values())
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "message": "VR sessions retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environments")
async def get_vr_environments() -> Dict[str, Any]:
    """Get all VR environments"""
    try:
        environments = list(vr_service.vr_environments.values())
        
        return {
            "success": True,
            "environments": environments,
            "count": len(environments),
            "message": "VR environments retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR environments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/avatars")
async def get_vr_avatars() -> Dict[str, Any]:
    """Get all VR avatars"""
    try:
        avatars = list(vr_service.vr_avatars.values())
        
        return {
            "success": True,
            "avatars": avatars,
            "count": len(avatars),
            "message": "VR avatars retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR avatars: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def get_vr_workflows() -> Dict[str, Any]:
    """Get all VR workflows"""
    try:
        workflows = list(vr_service.vr_workflows.values())
        
        return {
            "success": True,
            "workflows": workflows,
            "count": len(workflows),
            "message": "VR workflows retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rooms")
async def get_vr_rooms() -> Dict[str, Any]:
    """Get all VR rooms"""
    try:
        rooms = list(vr_service.vr_rooms.values())
        
        return {
            "success": True,
            "rooms": rooms,
            "count": len(rooms),
            "message": "VR rooms retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get VR rooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def vr_health_check() -> Dict[str, Any]:
    """VR service health check"""
    try:
        stats = await vr_service.get_vr_stats()
        
        return {
            "success": True,
            "status": "healthy",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "VR service is healthy"
        }
    
    except Exception as e:
        logger.error(f"VR service health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "message": "VR service is unhealthy"
        }
