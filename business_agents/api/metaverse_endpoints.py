"""
Metaverse API Endpoints
=======================

REST API endpoints for metaverse integration,
virtual world management, and immersive experiences.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.metaverse_service import (
    MetaverseService, MetaversePlatform, VirtualWorldType, AvatarType, InteractionType, VirtualAssetType,
    VirtualWorld, VirtualAvatar, VirtualAsset, MetaverseEvent, MetaverseInteraction
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/metaverse", tags=["Metaverse"])

# Pydantic models
class VirtualWorldCreateRequest(BaseModel):
    name: str = Field(..., description="World name")
    world_type: str = Field(..., description="World type")
    platform: str = Field(..., description="Metaverse platform")
    description: str = Field(..., description="World description")
    capacity: int = Field(50, description="Maximum number of users")
    template_id: Optional[str] = Field(None, description="Template ID to use")

class VirtualAvatarCreateRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Avatar name")
    avatar_type: str = Field(..., description="Avatar type")
    world_id: str = Field(..., description="World ID")
    template_id: Optional[str] = Field(None, description="Template ID to use")

class VirtualAssetCreateRequest(BaseModel):
    name: str = Field(..., description="Asset name")
    asset_type: str = Field(..., description="Asset type")
    owner_id: str = Field(..., description="Owner ID")
    world_id: str = Field(..., description="World ID")
    location: List[float] = Field(..., description="3D location [x, y, z]")
    template_id: Optional[str] = Field(None, description="Template ID to use")

class MetaverseEventCreateRequest(BaseModel):
    world_id: str = Field(..., description="World ID")
    name: str = Field(..., description="Event name")
    description: str = Field(..., description="Event description")
    event_type: str = Field(..., description="Event type")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    max_attendees: int = Field(50, description="Maximum number of attendees")

class MetaverseInteractionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    world_id: str = Field(..., description="World ID")
    interaction_type: str = Field(..., description="Interaction type")
    data: Dict[str, Any] = Field(..., description="Interaction data")
    target_id: Optional[str] = Field(None, description="Target ID")
    duration: float = Field(0.0, description="Interaction duration")

class WorldJoinRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    world_id: str = Field(..., description="World ID")

# Global metaverse service instance
metaverse_service = None

def get_metaverse_service() -> MetaverseService:
    """Get global metaverse service instance."""
    global metaverse_service
    if metaverse_service is None:
        metaverse_service = MetaverseService({
            "metaverse": {
                "max_worlds": 1000,
                "max_users_per_world": 100,
                "max_assets_per_world": 10000,
                "event_duration_hours": 24,
                "interaction_timeout": 300,
                "world_persistence": True,
                "asset_persistence": True
            }
        })
    return metaverse_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_metaverse_service(
    current_user: User = Depends(require_permission("metaverse:manage"))
):
    """Initialize the metaverse service."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        await metaverse_service.initialize()
        return {"message": "Metaverse Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize metaverse service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_metaverse_status(
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get metaverse service status."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        status = await metaverse_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse status: {str(e)}")

@router.post("/worlds/create", response_model=Dict[str, Any])
async def create_virtual_world(
    request: VirtualWorldCreateRequest,
    current_user: User = Depends(require_permission("metaverse:create"))
):
    """Create a new virtual world."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Convert string to enum
        world_type = VirtualWorldType(request.world_type)
        platform = MetaversePlatform(request.platform)
        
        # Create virtual world
        world = await metaverse_service.create_virtual_world(
            name=request.name,
            world_type=world_type,
            platform=platform,
            description=request.description,
            capacity=request.capacity,
            template_id=request.template_id
        )
        
        return {
            "world_id": world.world_id,
            "name": world.name,
            "world_type": world.world_type.value,
            "platform": world.platform.value,
            "description": world.description,
            "capacity": world.capacity,
            "current_users": world.current_users,
            "location": list(world.location),
            "size": list(world.size),
            "assets": world.assets,
            "rules": world.rules,
            "created_at": world.created_at.isoformat(),
            "metadata": world.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create virtual world: {str(e)}")

@router.get("/worlds", response_model=List[Dict[str, Any]])
async def get_virtual_worlds(
    world_type: Optional[str] = Query(None, description="Filter by world type"),
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get virtual worlds."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Convert string to enum if provided
        world_type_enum = VirtualWorldType(world_type) if world_type else None
        
        # Get worlds
        worlds = await metaverse_service.get_virtual_worlds(world_type_enum)
        
        result = []
        for world in worlds:
            world_dict = {
                "world_id": world.world_id,
                "name": world.name,
                "world_type": world.world_type.value,
                "platform": world.platform.value,
                "description": world.description,
                "capacity": world.capacity,
                "current_users": world.current_users,
                "location": list(world.location),
                "size": list(world.size),
                "assets": world.assets,
                "rules": world.rules,
                "created_at": world.created_at.isoformat(),
                "metadata": world.metadata
            }
            result.append(world_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtual worlds: {str(e)}")

@router.get("/worlds/{world_id}", response_model=Dict[str, Any])
async def get_virtual_world(
    world_id: str,
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get specific virtual world."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        worlds = await metaverse_service.get_virtual_worlds()
        world = next((w for w in worlds if w.world_id == world_id), None)
        
        if not world:
            raise HTTPException(status_code=404, detail="Virtual world not found")
        
        return {
            "world_id": world.world_id,
            "name": world.name,
            "world_type": world.world_type.value,
            "platform": world.platform.value,
            "description": world.description,
            "capacity": world.capacity,
            "current_users": world.current_users,
            "location": list(world.location),
            "size": list(world.size),
            "assets": world.assets,
            "rules": world.rules,
            "created_at": world.created_at.isoformat(),
            "metadata": world.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtual world: {str(e)}")

@router.post("/worlds/join", response_model=Dict[str, str])
async def join_world(
    request: WorldJoinRequest,
    current_user: User = Depends(require_permission("metaverse:execute"))
):
    """Join a virtual world."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        success = await metaverse_service.join_world(request.user_id, request.world_id)
        
        if success:
            return {"message": f"Successfully joined world {request.world_id}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to join world (capacity full or world not found)")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to join world: {str(e)}")

@router.post("/worlds/leave", response_model=Dict[str, str])
async def leave_world(
    request: WorldJoinRequest,
    current_user: User = Depends(require_permission("metaverse:execute"))
):
    """Leave a virtual world."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        success = await metaverse_service.leave_world(request.user_id, request.world_id)
        
        if success:
            return {"message": f"Successfully left world {request.world_id}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to leave world")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to leave world: {str(e)}")

@router.get("/worlds/{world_id}/analytics", response_model=Dict[str, Any])
async def get_world_analytics(
    world_id: str,
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get world analytics."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        analytics = await metaverse_service.get_world_analytics(world_id)
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get world analytics: {str(e)}")

@router.post("/avatars/create", response_model=Dict[str, Any])
async def create_virtual_avatar(
    request: VirtualAvatarCreateRequest,
    current_user: User = Depends(require_permission("metaverse:create"))
):
    """Create a new virtual avatar."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Convert string to enum
        avatar_type = AvatarType(request.avatar_type)
        
        # Create virtual avatar
        avatar = await metaverse_service.create_virtual_avatar(
            user_id=request.user_id,
            name=request.name,
            avatar_type=avatar_type,
            world_id=request.world_id,
            template_id=request.template_id
        )
        
        return {
            "avatar_id": avatar.avatar_id,
            "user_id": avatar.user_id,
            "name": avatar.name,
            "avatar_type": avatar.avatar_type.value,
            "appearance": avatar.appearance,
            "location": list(avatar.location),
            "world_id": avatar.world_id,
            "status": avatar.status,
            "last_seen": avatar.last_seen.isoformat(),
            "metadata": avatar.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create virtual avatar: {str(e)}")

@router.get("/avatars", response_model=List[Dict[str, Any]])
async def get_virtual_avatars(
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get virtual avatars."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        avatars = await metaverse_service.get_virtual_avatars(world_id)
        
        result = []
        for avatar in avatars:
            avatar_dict = {
                "avatar_id": avatar.avatar_id,
                "user_id": avatar.user_id,
                "name": avatar.name,
                "avatar_type": avatar.avatar_type.value,
                "appearance": avatar.appearance,
                "location": list(avatar.location),
                "world_id": avatar.world_id,
                "status": avatar.status,
                "last_seen": avatar.last_seen.isoformat(),
                "metadata": avatar.metadata
            }
            result.append(avatar_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtual avatars: {str(e)}")

@router.post("/assets/create", response_model=Dict[str, Any])
async def create_virtual_asset(
    request: VirtualAssetCreateRequest,
    current_user: User = Depends(require_permission("metaverse:create"))
):
    """Create a new virtual asset."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Convert string to enum
        asset_type = VirtualAssetType(request.asset_type)
        
        # Create virtual asset
        asset = await metaverse_service.create_virtual_asset(
            name=request.name,
            asset_type=asset_type,
            owner_id=request.owner_id,
            world_id=request.world_id,
            location=tuple(request.location),
            template_id=request.template_id
        )
        
        return {
            "asset_id": asset.asset_id,
            "name": asset.name,
            "asset_type": asset.asset_type.value,
            "owner_id": asset.owner_id,
            "world_id": asset.world_id,
            "location": list(asset.location),
            "properties": asset.properties,
            "value": asset.value,
            "created_at": asset.created_at.isoformat(),
            "metadata": asset.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create virtual asset: {str(e)}")

@router.get("/assets", response_model=List[Dict[str, Any]])
async def get_virtual_assets(
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get virtual assets."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        assets = await metaverse_service.get_virtual_assets(world_id)
        
        result = []
        for asset in assets:
            asset_dict = {
                "asset_id": asset.asset_id,
                "name": asset.name,
                "asset_type": asset.asset_type.value,
                "owner_id": asset.owner_id,
                "world_id": asset.world_id,
                "location": list(asset.location),
                "properties": asset.properties,
                "value": asset.value,
                "created_at": asset.created_at.isoformat(),
                "metadata": asset.metadata
            }
            result.append(asset_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get virtual assets: {str(e)}")

@router.post("/events/create", response_model=Dict[str, Any])
async def create_metaverse_event(
    request: MetaverseEventCreateRequest,
    current_user: User = Depends(require_permission("metaverse:create"))
):
    """Create a metaverse event."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Create metaverse event
        event = await metaverse_service.create_metaverse_event(
            world_id=request.world_id,
            name=request.name,
            description=request.description,
            event_type=request.event_type,
            start_time=request.start_time,
            end_time=request.end_time,
            max_attendees=request.max_attendees
        )
        
        return {
            "event_id": event.event_id,
            "world_id": event.world_id,
            "name": event.name,
            "description": event.description,
            "event_type": event.event_type,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat(),
            "attendees": event.attendees,
            "max_attendees": event.max_attendees,
            "status": event.status,
            "metadata": event.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create metaverse event: {str(e)}")

@router.get("/events", response_model=List[Dict[str, Any]])
async def get_metaverse_events(
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get metaverse events."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        events = await metaverse_service.get_metaverse_events(world_id)
        
        result = []
        for event in events:
            event_dict = {
                "event_id": event.event_id,
                "world_id": event.world_id,
                "name": event.name,
                "description": event.description,
                "event_type": event.event_type,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat(),
                "attendees": event.attendees,
                "max_attendees": event.max_attendees,
                "status": event.status,
                "metadata": event.metadata
            }
            result.append(event_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse events: {str(e)}")

@router.post("/interactions/record", response_model=Dict[str, Any])
async def record_metaverse_interaction(
    request: MetaverseInteractionRequest,
    current_user: User = Depends(require_permission("metaverse:execute"))
):
    """Record metaverse interaction."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Convert string to enum
        interaction_type = InteractionType(request.interaction_type)
        
        # Record interaction
        interaction = await metaverse_service.record_metaverse_interaction(
            user_id=request.user_id,
            world_id=request.world_id,
            interaction_type=interaction_type,
            data=request.data,
            target_id=request.target_id,
            duration=request.duration
        )
        
        return {
            "interaction_id": interaction.interaction_id,
            "user_id": interaction.user_id,
            "world_id": interaction.world_id,
            "interaction_type": interaction.interaction_type.value,
            "target_id": interaction.target_id,
            "data": interaction.data,
            "timestamp": interaction.timestamp.isoformat(),
            "duration": interaction.duration,
            "metadata": interaction.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metaverse interaction: {str(e)}")

@router.get("/interactions", response_model=List[Dict[str, Any]])
async def get_metaverse_interactions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
    interaction_type: Optional[str] = Query(None, description="Filter by interaction type"),
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get metaverse interactions."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Convert string to enum if provided
        interaction_type_enum = InteractionType(interaction_type) if interaction_type else None
        
        interactions = await metaverse_service.get_metaverse_interactions(
            user_id, world_id, interaction_type_enum
        )
        
        result = []
        for interaction in interactions:
            interaction_dict = {
                "interaction_id": interaction.interaction_id,
                "user_id": interaction.user_id,
                "world_id": interaction.world_id,
                "interaction_type": interaction.interaction_type.value,
                "target_id": interaction.target_id,
                "data": interaction.data,
                "timestamp": interaction.timestamp.isoformat(),
                "duration": interaction.duration,
                "metadata": interaction.metadata
            }
            result.append(interaction_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse interactions: {str(e)}")

@router.get("/platforms", response_model=List[Dict[str, Any]])
async def get_metaverse_platforms(
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get available metaverse platforms."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        platforms = [
            {
                "platform": "decentraland",
                "name": "Decentraland",
                "type": "blockchain",
                "currency": "MANA",
                "features": ["land_ownership", "nft_assets", "smart_contracts"],
                "description": "Decentralized virtual world built on Ethereum"
            },
            {
                "platform": "sandbox",
                "name": "The Sandbox",
                "type": "blockchain",
                "currency": "SAND",
                "features": ["land_ownership", "nft_assets", "voxel_creation"],
                "description": "Virtual world where players can build, own, and monetize their gaming experiences"
            },
            {
                "platform": "vrchat",
                "name": "VRChat",
                "type": "social",
                "currency": "VRC+",
                "features": ["avatar_creation", "world_creation", "social_interaction"],
                "description": "Online virtual world platform with user-generated content"
            },
            {
                "platform": "horizon_worlds",
                "name": "Horizon Worlds",
                "type": "vr",
                "currency": "Meta",
                "features": ["world_creation", "avatar_creation", "vr_interaction"],
                "description": "Virtual reality platform for creating and exploring virtual worlds"
            },
            {
                "platform": "spatial",
                "name": "Spatial",
                "type": "business",
                "currency": "SPATIAL",
                "features": ["business_meetings", "3d_models", "collaboration"],
                "description": "Virtual events and meetings platform for business"
            },
            {
                "platform": "gather",
                "name": "Gather",
                "type": "business",
                "currency": "GATHER",
                "features": ["video_calls", "spatial_audio", "business_meetings"],
                "description": "Virtual office and event platform with spatial video"
            }
        ]
        
        return platforms
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse platforms: {str(e)}")

@router.get("/world-types", response_model=List[Dict[str, Any]])
async def get_world_types(
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get available world types."""
    
    try:
        world_types = [
            {
                "type": "business_office",
                "name": "Business Office",
                "description": "Virtual office space for business meetings and collaboration",
                "capacity": 50,
                "features": ["meeting_rooms", "workstations", "collaboration_spaces"]
            },
            {
                "type": "conference_room",
                "name": "Conference Room",
                "description": "Professional conference room for meetings and presentations",
                "capacity": 20,
                "features": ["presentation_screen", "whiteboard", "video_conferencing"]
            },
            {
                "type": "training_center",
                "name": "Training Center",
                "description": "Comprehensive training center with classrooms and practical areas",
                "capacity": 100,
                "features": ["classrooms", "labs", "auditorium", "breakout_rooms"]
            },
            {
                "type": "showroom",
                "name": "Product Showroom",
                "description": "Interactive product showroom for demonstrations and sales",
                "capacity": 30,
                "features": ["display_cases", "product_models", "interactive_screens"]
            },
            {
                "type": "event_space",
                "name": "Event Space",
                "description": "Large venue for virtual events and conferences",
                "capacity": 500,
                "features": ["stage", "seating", "audio_visual", "networking_areas"]
            },
            {
                "type": "collaboration_hub",
                "name": "Collaboration Hub",
                "description": "Creative space for team collaboration and brainstorming",
                "capacity": 25,
                "features": ["whiteboards", "sticky_notes", "idea_boards", "breakout_spaces"]
            }
        ]
        
        return world_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get world types: {str(e)}")

@router.get("/templates", response_model=Dict[str, Any])
async def get_templates(
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get available templates."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        return {
            "world_templates": metaverse_service.world_templates,
            "avatar_templates": metaverse_service.avatar_templates,
            "asset_templates": metaverse_service.asset_templates
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_metaverse_analytics(
    current_user: User = Depends(require_permission("metaverse:view"))
):
    """Get metaverse analytics."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Get service status
        status = await metaverse_service.get_service_status()
        
        # Get worlds
        worlds = await metaverse_service.get_virtual_worlds()
        
        # Get interactions
        interactions = await metaverse_service.get_metaverse_interactions()
        
        # Calculate analytics
        analytics = {
            "total_worlds": status.get("total_worlds", 0),
            "total_avatars": status.get("total_avatars", 0),
            "total_assets": status.get("total_assets", 0),
            "total_events": status.get("total_events", 0),
            "active_events": status.get("active_events", 0),
            "total_interactions": status.get("total_interactions", 0),
            "world_types": {},
            "platforms": {},
            "interaction_types": {},
            "average_world_utilization": 0.0,
            "most_active_worlds": [],
            "user_engagement": "high",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Calculate world types
        for world in worlds:
            world_type = world.world_type.value
            if world_type not in analytics["world_types"]:
                analytics["world_types"][world_type] = 0
            analytics["world_types"][world_type] += 1
            
        # Calculate platforms
        for world in worlds:
            platform = world.platform.value
            if platform not in analytics["platforms"]:
                analytics["platforms"][platform] = 0
            analytics["platforms"][platform] += 1
            
        # Calculate interaction types
        for interaction in interactions:
            interaction_type = interaction.interaction_type.value
            if interaction_type not in analytics["interaction_types"]:
                analytics["interaction_types"][interaction_type] = 0
            analytics["interaction_types"][interaction_type] += 1
            
        # Calculate average world utilization
        if worlds:
            total_utilization = sum((w.current_users / w.capacity) * 100 for w in worlds)
            analytics["average_world_utilization"] = total_utilization / len(worlds)
            
        # Get most active worlds
        world_activity = {}
        for interaction in interactions:
            world_id = interaction.world_id
            if world_id not in world_activity:
                world_activity[world_id] = 0
            world_activity[world_id] += 1
            
        analytics["most_active_worlds"] = sorted(
            world_activity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse analytics: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def metaverse_health_check():
    """Metaverse service health check."""
    
    metaverse_service = get_metaverse_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(metaverse_service, 'virtual_worlds') and len(metaverse_service.virtual_worlds) >= 0
        
        # Get service status
        status = await metaverse_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "total_worlds": status.get("total_worlds", 0),
            "total_avatars": status.get("total_avatars", 0),
            "total_assets": status.get("total_assets", 0),
            "total_events": status.get("total_events", 0),
            "active_events": status.get("active_events", 0),
            "available_platforms": status.get("available_platforms", 0),
            "world_templates": status.get("world_templates", 0),
            "avatar_templates": status.get("avatar_templates", 0),
            "asset_templates": status.get("asset_templates", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }



























