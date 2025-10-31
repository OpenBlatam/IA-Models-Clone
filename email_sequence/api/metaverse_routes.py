"""
Metaverse Routes for Email Sequence System

This module provides API endpoints for metaverse integration including
virtual reality, augmented reality, and immersive marketing experiences.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse
from ..core.metaverse_integration import (
    metaverse_integration_engine,
    MetaversePlatform,
    MetaverseExperienceType,
    MetaverseDeviceType
)
from ..core.dependencies import get_current_user
from ..core.exceptions import MetaverseError

logger = logging.getLogger(__name__)

# Metaverse router
metaverse_router = APIRouter(
    prefix="/api/v1/metaverse",
    tags=["Metaverse"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


@metaverse_router.post("/experiences")
async def create_metaverse_experience(
    name: str,
    experience_type: MetaverseExperienceType,
    platform: MetaversePlatform,
    description: str,
    assets: Optional[List[Dict[str, Any]]] = None,
    interactions: Optional[List[Dict[str, Any]]] = None
):
    """
    Create a metaverse experience.
    
    Args:
        name: Experience name
        experience_type: Type of experience
        platform: Metaverse platform
        description: Experience description
        assets: List of assets
        interactions: List of interactions
        
    Returns:
        Experience creation result
    """
    try:
        experience_id = await metaverse_integration_engine.create_metaverse_experience(
            name=name,
            experience_type=experience_type,
            platform=platform,
            description=description,
            assets=assets,
            interactions=interactions
        )
        
        return {
            "status": "success",
            "experience_id": experience_id,
            "message": "Metaverse experience created successfully"
        }
        
    except MetaverseError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating metaverse experience: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.post("/virtual-stores")
async def create_virtual_store(
    store_name: str,
    products: List[Dict[str, Any]],
    platform: MetaversePlatform = MetaversePlatform.DECENTRALAND
):
    """
    Create a virtual store experience.
    
    Args:
        store_name: Name of the virtual store
        products: List of products to showcase
        platform: Metaverse platform
        
    Returns:
        Virtual store creation result
    """
    try:
        experience_id = await metaverse_integration_engine.create_virtual_store(
            store_name=store_name,
            products=products,
            platform=platform
        )
        
        return {
            "status": "success",
            "experience_id": experience_id,
            "store_name": store_name,
            "products_count": len(products),
            "platform": platform.value,
            "message": "Virtual store created successfully"
        }
        
    except MetaverseError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating virtual store: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.post("/ar-showcases")
async def create_ar_product_showcase(
    product_data: Dict[str, Any],
    device_type: MetaverseDeviceType = MetaverseDeviceType.MOBILE_AR
):
    """
    Create an AR product showcase.
    
    Args:
        product_data: Product information
        device_type: AR device type
        
    Returns:
        AR showcase creation result
    """
    try:
        experience_id = await metaverse_integration_engine.create_ar_product_showcase(
            product_data=product_data,
            device_type=device_type
        )
        
        return {
            "status": "success",
            "experience_id": experience_id,
            "product_name": product_data.get("name", "Product"),
            "device_type": device_type.value,
            "message": "AR product showcase created successfully"
        }
        
    except MetaverseError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating AR product showcase: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.post("/vr-brand-experiences")
async def create_vr_brand_experience(
    brand_data: Dict[str, Any],
    platform: MetaversePlatform = MetaversePlatform.VRChat
):
    """
    Create a VR brand experience.
    
    Args:
        brand_data: Brand information
        platform: VR platform
        
    Returns:
        VR brand experience creation result
    """
    try:
        experience_id = await metaverse_integration_engine.create_vr_brand_experience(
            brand_data=brand_data,
            platform=platform
        )
        
        return {
            "status": "success",
            "experience_id": experience_id,
            "brand_name": brand_data.get("name", "Brand"),
            "platform": platform.value,
            "message": "VR brand experience created successfully"
        }
        
    except MetaverseError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating VR brand experience: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.post("/track-interaction")
async def track_user_interaction(
    user_id: str,
    experience_id: str,
    interaction_type: str,
    interaction_data: Dict[str, Any]
):
    """
    Track user interaction in metaverse.
    
    Args:
        user_id: User identifier
        experience_id: Experience identifier
        interaction_type: Type of interaction
        interaction_data: Interaction data
        
    Returns:
        Interaction tracking result
    """
    try:
        await metaverse_integration_engine.track_user_interaction(
            user_id=user_id,
            experience_id=experience_id,
            interaction_type=interaction_type,
            interaction_data=interaction_data
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "experience_id": experience_id,
            "interaction_type": interaction_type,
            "message": "User interaction tracked successfully"
        }
        
    except MetaverseError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking user interaction: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/experiences")
async def list_metaverse_experiences():
    """
    List all metaverse experiences.
    
    Returns:
        List of metaverse experiences
    """
    try:
        experiences = []
        for experience_id, experience in metaverse_integration_engine.metaverse_experiences.items():
            experiences.append({
                "experience_id": experience_id,
                "name": experience.name,
                "experience_type": experience.experience_type.value,
                "platform": experience.platform.value,
                "description": experience.description,
                "assets_count": len(experience.assets),
                "interactions_count": len(experience.interactions),
                "is_active": experience.is_active,
                "created_at": experience.created_at.isoformat(),
                "updated_at": experience.updated_at.isoformat()
            })
        
        return {
            "status": "success",
            "experiences": experiences,
            "total_experiences": len(experiences)
        }
        
    except Exception as e:
        logger.error(f"Error listing metaverse experiences: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/experiences/{experience_id}")
async def get_metaverse_experience(experience_id: str):
    """
    Get metaverse experience details.
    
    Args:
        experience_id: Experience identifier
        
    Returns:
        Experience details
    """
    try:
        if experience_id not in metaverse_integration_engine.metaverse_experiences:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metaverse experience not found")
        
        experience = metaverse_integration_engine.metaverse_experiences[experience_id]
        
        return {
            "status": "success",
            "experience": {
                "experience_id": experience_id,
                "name": experience.name,
                "experience_type": experience.experience_type.value,
                "platform": experience.platform.value,
                "description": experience.description,
                "assets": [
                    {
                        "asset_id": asset.asset_id,
                        "name": asset.name,
                        "asset_type": asset.asset_type,
                        "platform": asset.platform.value,
                        "file_path": asset.file_path,
                        "metadata": asset.metadata,
                        "tags": asset.tags
                    }
                    for asset in experience.assets
                ],
                "interactions": experience.interactions,
                "is_active": experience.is_active,
                "created_at": experience.created_at.isoformat(),
                "updated_at": experience.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metaverse experience: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/users")
async def list_metaverse_users():
    """
    List all metaverse users.
    
    Returns:
        List of metaverse users
    """
    try:
        users = []
        for user_id, user in metaverse_integration_engine.metaverse_users.items():
            users.append({
                "user_id": user_id,
                "avatar_id": user.avatar_id,
                "platform": user.platform.value,
                "device_type": user.device_type.value,
                "location": user.location,
                "preferences": user.preferences,
                "interactions_count": len(user.interaction_history),
                "created_at": user.created_at.isoformat(),
                "last_active": user.last_active.isoformat()
            })
        
        return {
            "status": "success",
            "users": users,
            "total_users": len(users)
        }
        
    except Exception as e:
        logger.error(f"Error listing metaverse users: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/sessions")
async def list_active_sessions():
    """
    List active metaverse sessions.
    
    Returns:
        List of active sessions
    """
    try:
        sessions = []
        for session_key, session in metaverse_integration_engine.active_sessions.items():
            sessions.append({
                "session_key": session_key,
                "user_id": session["user_id"],
                "experience_id": session["experience_id"],
                "start_time": session["start_time"].isoformat(),
                "interactions_count": len(session["interactions"]),
                "duration_seconds": (datetime.utcnow() - session["start_time"]).total_seconds()
            })
        
        return {
            "status": "success",
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing active sessions: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/assets")
async def list_metaverse_assets():
    """
    List all metaverse assets.
    
    Returns:
        List of metaverse assets
    """
    try:
        assets = []
        for asset_id, asset in metaverse_integration_engine.metaverse_assets.items():
            assets.append({
                "asset_id": asset_id,
                "name": asset.name,
                "asset_type": asset.asset_type,
                "platform": asset.platform.value,
                "file_path": asset.file_path,
                "file_size": asset.file_size,
                "optimization_level": asset.optimization_level,
                "tags": asset.tags,
                "metadata": asset.metadata,
                "created_at": asset.created_at.isoformat()
            })
        
        return {
            "status": "success",
            "assets": assets,
            "total_assets": len(assets)
        }
        
    except Exception as e:
        logger.error(f"Error listing metaverse assets: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/analytics")
async def get_metaverse_analytics():
    """
    Get metaverse analytics.
    
    Returns:
        Metaverse analytics data
    """
    try:
        analytics = await metaverse_integration_engine.get_metaverse_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting metaverse analytics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/platforms")
async def list_metaverse_platforms():
    """
    List supported metaverse platforms.
    
    Returns:
        List of supported platforms
    """
    try:
        platforms = [
            {
                "platform": platform.value,
                "name": platform.value.replace("_", " ").title(),
                "supported": True
            }
            for platform in MetaversePlatform
        ]
        
        return {
            "status": "success",
            "platforms": platforms,
            "total_platforms": len(platforms)
        }
        
    except Exception as e:
        logger.error(f"Error listing metaverse platforms: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/experience-types")
async def list_experience_types():
    """
    List supported experience types.
    
    Returns:
        List of supported experience types
    """
    try:
        experience_types = [
            {
                "type": exp_type.value,
                "name": exp_type.value.replace("_", " ").title(),
                "description": f"{exp_type.value.replace('_', ' ').title()} experience"
            }
            for exp_type in MetaverseExperienceType
        ]
        
        return {
            "status": "success",
            "experience_types": experience_types,
            "total_types": len(experience_types)
        }
        
    except Exception as e:
        logger.error(f"Error listing experience types: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/device-types")
async def list_device_types():
    """
    List supported device types.
    
    Returns:
        List of supported device types
    """
    try:
        device_types = [
            {
                "type": device_type.value,
                "name": device_type.value.replace("_", " ").title(),
                "category": "VR" if "vr" in device_type.value else "AR" if "ar" in device_type.value else "Other"
            }
            for device_type in MetaverseDeviceType
        ]
        
        return {
            "status": "success",
            "device_types": device_types,
            "total_types": len(device_types)
        }
        
    except Exception as e:
        logger.error(f"Error listing device types: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@metaverse_router.get("/capabilities")
async def get_metaverse_capabilities():
    """
    Get metaverse capabilities.
    
    Returns:
        Metaverse capabilities information
    """
    try:
        capabilities = {
            "vr_enabled": metaverse_integration_engine.vr_enabled,
            "ar_enabled": metaverse_integration_engine.ar_enabled,
            "webxr_enabled": metaverse_integration_engine.webxr_enabled,
            "3d_processing_enabled": metaverse_integration_engine.3d_processing_enabled,
            "supported_platforms": [platform.value for platform in MetaversePlatform],
            "supported_experience_types": [exp_type.value for exp_type in MetaverseExperienceType],
            "supported_device_types": [device_type.value for device_type in MetaverseDeviceType],
            "total_experiences": len(metaverse_integration_engine.metaverse_experiences),
            "total_users": len(metaverse_integration_engine.metaverse_users),
            "total_assets": len(metaverse_integration_engine.metaverse_assets),
            "active_sessions": len(metaverse_integration_engine.active_sessions)
        }
        
        return {
            "status": "success",
            "capabilities": capabilities
        }
        
    except Exception as e:
        logger.error(f"Error getting metaverse capabilities: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# Error handlers for metaverse routes
@metaverse_router.exception_handler(MetaverseError)
async def metaverse_error_handler(request, exc):
    """Handle metaverse errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=f"Metaverse error: {exc.message}",
            error_code="METAVERSE_ERROR"
        ).dict()
    )





























