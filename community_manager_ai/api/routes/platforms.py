"""
Platforms API Routes
====================

Endpoints para gestión de conexiones a plataformas sociales.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/platforms", tags=["platforms"])


class PlatformConnect(BaseModel):
    platform: str
    credentials: Dict[str, str]


class PlatformResponse(BaseModel):
    platform: str
    connected: bool
    connected_at: Optional[str] = None


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


@router.post("/connect", response_model=PlatformResponse)
async def connect_platform(
    connection: PlatformConnect,
    manager = Depends(get_community_manager)
):
    """Conectar una plataforma social"""
    try:
        success = manager.connect_platform(
            platform=connection.platform,
            credentials=connection.credentials
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Error conectando a {connection.platform}"
            )
        
        return PlatformResponse(
            platform=connection.platform,
            connected=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[str])
async def get_connected_platforms(
    manager = Depends(get_community_manager)
):
    """Obtener lista de plataformas conectadas"""
    try:
        platforms = manager.social_connector.get_connected_platforms()
        return platforms
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{platform}")
async def disconnect_platform(
    platform: str,
    manager = Depends(get_community_manager)
):
    """Desconectar una plataforma"""
    try:
        success = manager.social_connector.disconnect(platform)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Plataforma {platform} no está conectada"
            )
        return {"status": "disconnected", "platform": platform}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

