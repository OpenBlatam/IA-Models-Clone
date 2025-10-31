"""
Metaverse API routes for Facebook Posts API
Virtual reality, augmented reality, and immersive content experiences
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import get_request_id
from ..services.metaverse_service import (
    get_metaverse_service, MetaversePlatform, ContentType, InteractionType,
    VirtualWorld, Avatar, VirtualEvent, MetaverseInteraction
)
from ..services.security_service import get_security_service
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/metaverse", tags=["Metaverse"])

# Security scheme
security = HTTPBearer()


# Virtual World Management Routes

@router.post(
    "/worlds",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Virtual world created successfully"},
        400: {"description": "Invalid world data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Virtual world creation error"}
    },
    summary="Create virtual world",
    description="Create a virtual world in the metaverse"
)
@timed("metaverse_create_world")
async def create_virtual_world(
    name: str = Query(..., description="World name"),
    description: str = Query(..., description="World description"),
    platform: str = Query("mock", description="Metaverse platform"),
    coordinates: Dict[str, float] = Query(..., description="World coordinates"),
    capacity: int = Query(100, description="Maximum capacity", ge=1, le=1000),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Create a virtual world"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not name or not description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name and description are required"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Create virtual world
        world = await metaverse_service.create_virtual_world(
            name=name,
            description=description,
            platform=metaverse_platform,
            coordinates=coordinates,
            capacity=capacity
        )
        
        logger.info(
            "Virtual world created",
            world_id=world.id,
            name=name,
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Virtual world created successfully",
            "world": {
                "id": world.id,
                "name": world.name,
                "description": world.description,
                "platform": world.platform.value,
                "world_url": world.world_url,
                "coordinates": world.coordinates,
                "capacity": world.capacity,
                "current_users": world.current_users,
                "created_at": world.created_at.isoformat()
            },
            "request_id": request_id,
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Virtual world creation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Virtual world creation failed: {str(e)}"
        )


@router.get(
    "/worlds/{world_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Virtual world retrieved successfully"},
        404: {"description": "Virtual world not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Virtual world retrieval error"}
    },
    summary="Get virtual world",
    description="Get virtual world by ID"
)
@timed("metaverse_get_world")
async def get_virtual_world(
    world_id: str = Path(..., description="World ID"),
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get virtual world by ID"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Get virtual world
        world = await metaverse_service.get_virtual_world(world_id, metaverse_platform)
        
        if not world:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Virtual world not found"
            )
        
        logger.info(
            "Virtual world retrieved",
            world_id=world_id,
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Virtual world retrieved successfully",
            "world": {
                "id": world.id,
                "name": world.name,
                "description": world.description,
                "platform": world.platform.value,
                "world_url": world.world_url,
                "coordinates": world.coordinates,
                "capacity": world.capacity,
                "current_users": world.current_users,
                "tags": world.tags,
                "created_at": world.created_at.isoformat()
            },
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Virtual world retrieval failed",
            world_id=world_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Virtual world retrieval failed: {str(e)}"
        )


@router.get(
    "/worlds",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Virtual worlds retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Virtual worlds retrieval error"}
    },
    summary="List virtual worlds",
    description="List all virtual worlds"
)
@timed("metaverse_list_worlds")
async def list_virtual_worlds(
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """List virtual worlds"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # List virtual worlds
        worlds = await metaverse_service.list_virtual_worlds(metaverse_platform)
        
        worlds_data = []
        for world in worlds:
            worlds_data.append({
                "id": world.id,
                "name": world.name,
                "description": world.description,
                "platform": world.platform.value,
                "world_url": world.world_url,
                "coordinates": world.coordinates,
                "capacity": world.capacity,
                "current_users": world.current_users,
                "tags": world.tags,
                "created_at": world.created_at.isoformat()
            })
        
        logger.info(
            "Virtual worlds listed",
            count=len(worlds),
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Virtual worlds retrieved successfully",
            "worlds": worlds_data,
            "total_count": len(worlds),
            "platform": platform,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Virtual worlds listing failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Virtual worlds listing failed: {str(e)}"
        )


# Avatar Management Routes

@router.post(
    "/avatars",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Avatar created successfully"},
        400: {"description": "Invalid avatar data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Avatar creation error"}
    },
    summary="Create avatar",
    description="Create an avatar in the metaverse"
)
@timed("metaverse_create_avatar")
async def create_avatar(
    name: str = Query(..., description="Avatar name"),
    user_id: str = Query(..., description="User ID"),
    platform: str = Query("mock", description="Metaverse platform"),
    avatar_data: Dict[str, Any] = Query(..., description="Avatar data"),
    customizations: Dict[str, Any] = Query(..., description="Avatar customizations"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Create an avatar"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not name or not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name and user ID are required"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Create avatar
        avatar = await metaverse_service.create_avatar(
            name=name,
            user_id=user_id,
            platform=metaverse_platform,
            avatar_data=avatar_data,
            customizations=customizations
        )
        
        logger.info(
            "Avatar created",
            avatar_id=avatar.id,
            name=name,
            user_id=user_id,
            platform=platform,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Avatar created successfully",
            "avatar": {
                "id": avatar.id,
                "name": avatar.name,
                "user_id": avatar.user_id,
                "platform": avatar.platform.value,
                "avatar_data": avatar.avatar_data,
                "customizations": avatar.customizations,
                "accessories": avatar.accessories,
                "created_at": avatar.created_at.isoformat()
            },
            "request_id": request_id,
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Avatar creation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Avatar creation failed: {str(e)}"
        )


@router.get(
    "/avatars/{avatar_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Avatar retrieved successfully"},
        404: {"description": "Avatar not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Avatar retrieval error"}
    },
    summary="Get avatar",
    description="Get avatar by ID"
)
@timed("metaverse_get_avatar")
async def get_avatar(
    avatar_id: str = Path(..., description="Avatar ID"),
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get avatar by ID"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Get avatar
        avatar = await metaverse_service.get_avatar(avatar_id, metaverse_platform)
        
        if not avatar:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Avatar not found"
            )
        
        logger.info(
            "Avatar retrieved",
            avatar_id=avatar_id,
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Avatar retrieved successfully",
            "avatar": {
                "id": avatar.id,
                "name": avatar.name,
                "user_id": avatar.user_id,
                "platform": avatar.platform.value,
                "avatar_data": avatar.avatar_data,
                "customizations": avatar.customizations,
                "accessories": avatar.accessories,
                "created_at": avatar.created_at.isoformat()
            },
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Avatar retrieval failed",
            avatar_id=avatar_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Avatar retrieval failed: {str(e)}"
        )


@router.put(
    "/avatars/{avatar_id}/customize",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Avatar customized successfully"},
        400: {"description": "Invalid customization data"},
        401: {"description": "Unauthorized"},
        404: {"description": "Avatar not found"},
        500: {"description": "Avatar customization error"}
    },
    summary="Customize avatar",
    description="Customize an avatar"
)
@timed("metaverse_customize_avatar")
async def customize_avatar(
    avatar_id: str = Path(..., description="Avatar ID"),
    customizations: Dict[str, Any] = Query(..., description="Avatar customizations"),
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Customize an avatar"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Customize avatar
        success = await metaverse_service.customize_avatar(avatar_id, customizations, metaverse_platform)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Avatar not found"
            )
        
        logger.info(
            "Avatar customized",
            avatar_id=avatar_id,
            customizations=customizations,
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Avatar customized successfully",
            "avatar_id": avatar_id,
            "customizations": customizations,
            "platform": platform,
            "request_id": request_id,
            "customized_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Avatar customization failed",
            avatar_id=avatar_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Avatar customization failed: {str(e)}"
        )


# Virtual Event Management Routes

@router.post(
    "/events",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Virtual event created successfully"},
        400: {"description": "Invalid event data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Virtual event creation error"}
    },
    summary="Create virtual event",
    description="Create a virtual event in the metaverse"
)
@timed("metaverse_create_event")
async def create_virtual_event(
    title: str = Query(..., description="Event title"),
    description: str = Query(..., description="Event description"),
    platform: str = Query("mock", description="Metaverse platform"),
    world_id: str = Query("", description="World ID"),
    start_time: str = Query(..., description="Start time (ISO format)"),
    end_time: str = Query(..., description="End time (ISO format)"),
    max_attendees: int = Query(100, description="Maximum attendees", ge=1, le=1000),
    event_type: str = Query("social", description="Event type"),
    host_id: str = Query("", description="Host ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Create a virtual event"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not title or not description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title and description are required"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Parse datetime strings
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid datetime format. Use ISO format (e.g., 2024-01-01T12:00:00Z)"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Create virtual event
        event = await metaverse_service.create_virtual_event(
            title=title,
            description=description,
            platform=metaverse_platform,
            world_id=world_id,
            start_time=start_dt,
            end_time=end_dt,
            max_attendees=max_attendees,
            event_type=event_type,
            host_id=host_id
        )
        
        logger.info(
            "Virtual event created",
            event_id=event.id,
            title=title,
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Virtual event created successfully",
            "event": {
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "platform": event.platform.value,
                "world_id": event.world_id,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat(),
                "max_attendees": event.max_attendees,
                "current_attendees": event.current_attendees,
                "event_type": event.event_type,
                "host_id": event.host_id,
                "created_at": event.created_at.isoformat()
            },
            "request_id": request_id,
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Virtual event creation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Virtual event creation failed: {str(e)}"
        )


@router.get(
    "/events",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Virtual events retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Virtual events retrieval error"}
    },
    summary="List virtual events",
    description="List all virtual events"
)
@timed("metaverse_list_events")
async def list_virtual_events(
    platform: str = Query("mock", description="Metaverse platform"),
    event_type: str = Query(None, description="Event type filter"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """List virtual events"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # List virtual events
        events = await metaverse_service.list_virtual_events(metaverse_platform, event_type)
        
        events_data = []
        for event in events:
            events_data.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "platform": event.platform.value,
                "world_id": event.world_id,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat(),
                "max_attendees": event.max_attendees,
                "current_attendees": event.current_attendees,
                "event_type": event.event_type,
                "host_id": event.host_id,
                "created_at": event.created_at.isoformat()
            })
        
        logger.info(
            "Virtual events listed",
            count=len(events),
            platform=platform,
            event_type=event_type,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Virtual events retrieved successfully",
            "events": events_data,
            "total_count": len(events),
            "platform": platform,
            "event_type": event_type,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Virtual events listing failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Virtual events listing failed: {str(e)}"
        )


# Metaverse Interaction Routes

@router.post(
    "/interactions",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Metaverse interaction recorded successfully"},
        400: {"description": "Invalid interaction data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Metaverse interaction recording error"}
    },
    summary="Record metaverse interaction",
    description="Record a metaverse interaction"
)
@timed("metaverse_record_interaction")
async def record_metaverse_interaction(
    user_id: str = Query(..., description="User ID"),
    world_id: str = Query(..., description="World ID"),
    interaction_type: str = Query(..., description="Interaction type"),
    coordinates: Dict[str, float] = Query(..., description="Coordinates"),
    duration: float = Query(0.0, description="Interaction duration"),
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Record a metaverse interaction"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id or not world_id or not interaction_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID, world ID, and interaction type are required"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Validate interaction type
        try:
            interaction_type_enum = InteractionType(interaction_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interaction type. Valid types: {[t.value for t in InteractionType]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Record interaction
        interaction_id = await metaverse_service.record_interaction(
            user_id=user_id,
            world_id=world_id,
            interaction_type=interaction_type_enum,
            coordinates=coordinates,
            duration=duration,
            platform=metaverse_platform
        )
        
        logger.info(
            "Metaverse interaction recorded",
            interaction_id=interaction_id,
            user_id=user_id,
            world_id=world_id,
            interaction_type=interaction_type,
            platform=platform,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Metaverse interaction recorded successfully",
            "interaction": {
                "id": interaction_id,
                "user_id": user_id,
                "world_id": world_id,
                "interaction_type": interaction_type,
                "coordinates": coordinates,
                "duration": duration,
                "platform": platform,
                "timestamp": datetime.now().isoformat()
            },
            "request_id": request_id,
            "recorded_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Metaverse interaction recording failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metaverse interaction recording failed: {str(e)}"
        )


# Metaverse Analytics Routes

@router.get(
    "/analytics/user/{user_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "User analytics retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "User analytics retrieval error"}
    },
    summary="Get user analytics",
    description="Get user metaverse analytics"
)
@timed("metaverse_get_user_analytics")
async def get_user_analytics(
    user_id: str = Path(..., description="User ID"),
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get user metaverse analytics"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Get user analytics
        analytics = await metaverse_service.analyze_user_interactions(user_id, metaverse_platform)
        
        logger.info(
            "User analytics retrieved",
            user_id=user_id,
            platform=platform,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "User analytics retrieved successfully",
            "analytics": analytics,
            "user_id": user_id,
            "platform": platform,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "User analytics retrieval failed",
            user_id=user_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User analytics retrieval failed: {str(e)}"
        )


@router.get(
    "/analytics/world/{world_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "World analytics retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "World analytics retrieval error"}
    },
    summary="Get world analytics",
    description="Get world metaverse analytics"
)
@timed("metaverse_get_world_analytics")
async def get_world_analytics(
    world_id: str = Path(..., description="World ID"),
    platform: str = Query("mock", description="Metaverse platform"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get world metaverse analytics"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Get world analytics
        analytics = await metaverse_service.analyze_world_popularity(world_id, metaverse_platform)
        
        logger.info(
            "World analytics retrieved",
            world_id=world_id,
            platform=platform,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "World analytics retrieved successfully",
            "analytics": analytics,
            "world_id": world_id,
            "platform": platform,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "World analytics retrieval failed",
            world_id=world_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"World analytics retrieval failed: {str(e)}"
        )


# Metaverse Content Generation Routes

@router.post(
    "/content/generate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Metaverse content generated successfully"},
        400: {"description": "Invalid content data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Metaverse content generation error"}
    },
    summary="Generate metaverse content",
    description="Generate metaverse content"
)
@timed("metaverse_generate_content")
async def generate_metaverse_content(
    content_type: str = Query(..., description="Content type"),
    description: str = Query(..., description="Content description"),
    platform: str = Query("mock", description="Metaverse platform"),
    metadata: Dict[str, Any] = Query(..., description="Content metadata"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Generate metaverse content"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not content_type or not description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content type and description are required"
            )
        
        # Validate platform
        try:
            metaverse_platform = MetaversePlatform(platform)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Valid platforms: {[p.value for p in MetaversePlatform]}"
            )
        
        # Validate content type
        try:
            content_type_enum = ContentType(content_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type. Valid types: {[t.value for t in ContentType]}"
            )
        
        # Get metaverse service
        metaverse_service = get_metaverse_service()
        
        # Generate content
        content = await metaverse_service.generate_metaverse_content(
            content_type=content_type_enum,
            description=description,
            platform=metaverse_platform,
            metadata=metadata
        )
        
        logger.info(
            "Metaverse content generated",
            content_id=content["id"],
            content_type=content_type,
            platform=platform,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Metaverse content generated successfully",
            "content": content,
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Metaverse content generation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metaverse content generation failed: {str(e)}"
        )


# Export router
__all__ = ["router"]





























