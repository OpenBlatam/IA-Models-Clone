"""
Metaverse API - Ultimate Advanced Implementation
==============================================

FastAPI endpoints for metaverse operations.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.metaverse_service import (
    metaverse_service,
    MetaverseWorldType,
    DigitalTwinType,
    MetaverseInteractionType
)

logger = logging.getLogger(__name__)

# Pydantic models
class MetaverseWorldCreation(BaseModel):
    world_name: str = Field(..., description="Name of the metaverse world")
    world_type: MetaverseWorldType = Field(..., description="Type of metaverse world")
    world_config: Dict[str, Any] = Field(..., description="World configuration")
    creator_id: str = Field(..., description="ID of the world creator")

class DigitalTwinCreation(BaseModel):
    twin_name: str = Field(..., description="Name of the digital twin")
    twin_type: DigitalTwinType = Field(..., description="Type of digital twin")
    twin_data: Dict[str, Any] = Field(..., description="Digital twin data")
    world_id: str = Field(..., description="ID of the metaverse world")
    owner_id: str = Field(..., description="ID of the twin owner")

class MetaverseUserRegistration(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username in the metaverse")
    avatar_config: Dict[str, Any] = Field(..., description="Avatar configuration")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

class MetaverseWorldJoin(BaseModel):
    world_id: str = Field(..., description="ID of the metaverse world")
    user_id: str = Field(..., description="ID of the user joining")
    session_config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")

class MetaverseInteractionTracking(BaseModel):
    interaction_type: MetaverseInteractionType = Field(..., description="Type of metaverse interaction")
    interaction_data: Dict[str, Any] = Field(..., description="Interaction data")

class MetaverseAssetCreation(BaseModel):
    asset_name: str = Field(..., description="Name of the metaverse asset")
    asset_type: str = Field(..., description="Type of asset")
    asset_data: Dict[str, Any] = Field(..., description="Asset data")
    world_id: str = Field(..., description="ID of the metaverse world")
    creator_id: str = Field(..., description="ID of the asset creator")

class MetaverseEventCreation(BaseModel):
    event_name: str = Field(..., description="Name of the metaverse event")
    event_type: str = Field(..., description="Type of event")
    event_data: Dict[str, Any] = Field(..., description="Event data")
    world_id: str = Field(..., description="ID of the metaverse world")
    organizer_id: str = Field(..., description="ID of the event organizer")

# Create router
router = APIRouter(prefix="/metaverse", tags=["Metaverse"])

@router.post("/worlds/create")
async def create_metaverse_world(world_data: MetaverseWorldCreation) -> Dict[str, Any]:
    """Create a new metaverse world"""
    try:
        world_id = await metaverse_service.create_metaverse_world(
            world_name=world_data.world_name,
            world_type=world_data.world_type,
            world_config=world_data.world_config,
            creator_id=world_data.creator_id
        )
        
        return {
            "success": True,
            "world_id": world_id,
            "message": "Metaverse world created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create metaverse world: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/digital-twins/create")
async def create_digital_twin(twin_data: DigitalTwinCreation) -> Dict[str, Any]:
    """Create a digital twin"""
    try:
        twin_id = await metaverse_service.create_digital_twin(
            twin_name=twin_data.twin_name,
            twin_type=twin_data.twin_type,
            twin_data=twin_data.twin_data,
            world_id=twin_data.world_id,
            owner_id=twin_data.owner_id
        )
        
        return {
            "success": True,
            "twin_id": twin_id,
            "message": "Digital twin created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create digital twin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/register")
async def register_metaverse_user(user_data: MetaverseUserRegistration) -> Dict[str, Any]:
    """Register a metaverse user"""
    try:
        user_id = await metaverse_service.register_metaverse_user(
            user_id=user_data.user_id,
            username=user_data.username,
            avatar_config=user_data.avatar_config,
            preferences=user_data.preferences
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "Metaverse user registered successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to register metaverse user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/worlds/join")
async def join_metaverse_world(join_data: MetaverseWorldJoin) -> Dict[str, Any]:
    """Join a metaverse world"""
    try:
        session_id = await metaverse_service.join_metaverse_world(
            world_id=join_data.world_id,
            user_id=join_data.user_id,
            session_config=join_data.session_config
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Joined metaverse world successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to join metaverse world: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/interactions")
async def track_metaverse_interaction(
    session_id: str,
    interaction_data: MetaverseInteractionTracking
) -> Dict[str, Any]:
    """Track metaverse interaction"""
    try:
        interaction_id = await metaverse_service.track_metaverse_interaction(
            session_id=session_id,
            interaction_type=interaction_data.interaction_type,
            interaction_data=interaction_data.interaction_data
        )
        
        return {
            "success": True,
            "interaction_id": interaction_id,
            "message": "Metaverse interaction tracked successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to track metaverse interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assets/create")
async def create_metaverse_asset(asset_data: MetaverseAssetCreation) -> Dict[str, Any]:
    """Create a metaverse asset"""
    try:
        asset_id = await metaverse_service.create_metaverse_asset(
            asset_name=asset_data.asset_name,
            asset_type=asset_data.asset_type,
            asset_data=asset_data.asset_data,
            world_id=asset_data.world_id,
            creator_id=asset_data.creator_id
        )
        
        return {
            "success": True,
            "asset_id": asset_id,
            "message": "Metaverse asset created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create metaverse asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events/create")
async def create_metaverse_event(event_data: MetaverseEventCreation) -> Dict[str, Any]:
    """Create a metaverse event"""
    try:
        event_id = await metaverse_service.create_metaverse_event(
            event_name=event_data.event_name,
            event_type=event_data.event_type,
            event_data=event_data.event_data,
            world_id=event_data.world_id,
            organizer_id=event_data.organizer_id
        )
        
        return {
            "success": True,
            "event_id": event_id,
            "message": "Metaverse event created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create metaverse event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/leave")
async def leave_metaverse_world(session_id: str) -> Dict[str, Any]:
    """Leave a metaverse world"""
    try:
        result = await metaverse_service.leave_metaverse_world(session_id=session_id)
        
        return {
            "success": True,
            "result": result,
            "message": "Left metaverse world successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to leave metaverse world: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/worlds/{world_id}/analytics")
async def get_metaverse_world_analytics(world_id: str) -> Dict[str, Any]:
    """Get metaverse world analytics"""
    try:
        analytics = await metaverse_service.get_metaverse_world_analytics(world_id=world_id)
        
        if analytics is None:
            raise HTTPException(status_code=404, detail="Metaverse world not found")
        
        return {
            "success": True,
            "analytics": analytics,
            "message": "Metaverse world analytics retrieved successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metaverse world analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_metaverse_stats() -> Dict[str, Any]:
    """Get metaverse service statistics"""
    try:
        stats = await metaverse_service.get_metaverse_stats()
        
        return {
            "success": True,
            "stats": stats,
            "message": "Metaverse statistics retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get metaverse stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/worlds")
async def get_metaverse_worlds() -> Dict[str, Any]:
    """Get all metaverse worlds"""
    try:
        worlds = list(metaverse_service.metaverse_worlds.values())
        
        return {
            "success": True,
            "worlds": worlds,
            "count": len(worlds),
            "message": "Metaverse worlds retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get metaverse worlds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/digital-twins")
async def get_digital_twins() -> Dict[str, Any]:
    """Get all digital twins"""
    try:
        twins = list(metaverse_service.digital_twins.values())
        
        return {
            "success": True,
            "twins": twins,
            "count": len(twins),
            "message": "Digital twins retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get digital twins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
async def get_metaverse_users() -> Dict[str, Any]:
    """Get all metaverse users"""
    try:
        users = list(metaverse_service.metaverse_users.values())
        
        return {
            "success": True,
            "users": users,
            "count": len(users),
            "message": "Metaverse users retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get metaverse users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def get_metaverse_sessions() -> Dict[str, Any]:
    """Get all metaverse sessions"""
    try:
        sessions = list(metaverse_service.metaverse_sessions.values())
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "message": "Metaverse sessions retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get metaverse sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assets")
async def get_metaverse_assets() -> Dict[str, Any]:
    """Get all metaverse assets"""
    try:
        assets = list(metaverse_service.metaverse_assets.values())
        
        return {
            "success": True,
            "assets": assets,
            "count": len(assets),
            "message": "Metaverse assets retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get metaverse assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events")
async def get_metaverse_events() -> Dict[str, Any]:
    """Get all metaverse events"""
    try:
        events = list(metaverse_service.metaverse_events.values())
        
        return {
            "success": True,
            "events": events,
            "count": len(events),
            "message": "Metaverse events retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get metaverse events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def metaverse_health_check() -> Dict[str, Any]:
    """Metaverse service health check"""
    try:
        stats = await metaverse_service.get_metaverse_stats()
        
        return {
            "success": True,
            "status": "healthy",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Metaverse service is healthy"
        }
    
    except Exception as e:
        logger.error(f"Metaverse service health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Metaverse service is unhealthy"
        }
