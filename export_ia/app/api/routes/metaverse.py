"""
Metaverse API Routes - Rutas API para sistema de Metaverso
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
import logging

from ..metaverse.metaverse_engine import MetaverseEngine, WorldType, AvatarType, InteractionType, AssetType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/metaverse", tags=["Metaverse"])

# Instancia global del motor de metaverso
metaverse_engine = MetaverseEngine()


# Modelos Pydantic
class CreateWorldRequest(BaseModel):
    name: str
    world_type: str
    description: str
    owner_id: str
    max_users: int = 100
    is_public: bool = True
    password: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class CreateAvatarRequest(BaseModel):
    user_id: str
    name: str
    avatar_type: str
    appearance: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, float]] = None
    stats: Optional[Dict[str, Any]] = None


class CreateAssetRequest(BaseModel):
    name: str
    asset_type: str
    owner_id: str
    world_id: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    properties: Optional[Dict[str, Any]] = None
    is_nft: bool = False
    nft_contract: Optional[str] = None
    nft_token_id: Optional[str] = None
    price: Optional[str] = None


class RecordInteractionRequest(BaseModel):
    user_id: str
    avatar_id: str
    world_id: str
    interaction_type: str
    target_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# Rutas de Mundos Virtuales
@router.post("/worlds")
async def create_virtual_world(request: CreateWorldRequest):
    """Crear mundo virtual."""
    try:
        world_type = WorldType(request.world_type)
        
        world_id = await metaverse_engine.create_world(
            name=request.name,
            world_type=world_type,
            description=request.description,
            owner_id=request.owner_id,
            max_users=request.max_users,
            is_public=request.is_public,
            password=request.password,
            settings=request.settings
        )
        
        world = metaverse_engine.worlds[world_id]
        
        return {
            "world_id": world_id,
            "name": world.name,
            "world_type": world.world_type.value,
            "description": world.description,
            "max_users": world.max_users,
            "owner_id": world.owner_id,
            "is_public": world.is_public,
            "created_at": world.created_at.isoformat(),
            "success": True,
            "message": "Mundo virtual creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear mundo virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worlds")
async def get_virtual_worlds(
    world_type: Optional[str] = Query(None, description="Tipo de mundo"),
    is_public: Optional[bool] = Query(None, description="Mundos públicos"),
    limit: int = Query(100, description="Límite de mundos")
):
    """Obtener mundos virtuales."""
    try:
        worlds = []
        
        for world_id, world in metaverse_engine.worlds.items():
            # Filtrar por tipo de mundo
            if world_type and world.world_type.value != world_type:
                continue
            
            # Filtrar por público/privado
            if is_public is not None and world.is_public != is_public:
                continue
            
            worlds.append({
                "world_id": world_id,
                "name": world.name,
                "world_type": world.world_type.value,
                "description": world.description,
                "max_users": world.max_users,
                "current_users": world.current_users,
                "owner_id": world.owner_id,
                "is_public": world.is_public,
                "created_at": world.created_at.isoformat(),
                "last_activity": world.last_activity.isoformat()
            })
        
        # Ordenar por actividad reciente
        worlds.sort(key=lambda x: x["last_activity"], reverse=True)
        
        return {
            "worlds": worlds[:limit],
            "count": len(worlds[:limit]),
            "total_worlds": len(metaverse_engine.worlds),
            "filters": {
                "world_type": world_type,
                "is_public": is_public
            },
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener mundos virtuales: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worlds/{world_id}")
async def get_virtual_world(world_id: str):
    """Obtener mundo virtual específico."""
    try:
        world_info = await metaverse_engine.get_world_info(world_id)
        
        return {
            "world": world_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener mundo virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/worlds/{world_id}")
async def delete_virtual_world(world_id: str):
    """Eliminar mundo virtual."""
    try:
        if world_id not in metaverse_engine.worlds:
            raise HTTPException(status_code=404, detail="Mundo no encontrado")
        
        world = metaverse_engine.worlds[world_id]
        del metaverse_engine.worlds[world_id]
        
        # Limpiar activos del mundo
        world_assets = [
            asset_id for asset_id, asset in metaverse_engine.assets.items()
            if asset.world_id == world_id
        ]
        for asset_id in world_assets:
            del metaverse_engine.assets[asset_id]
        
        metaverse_engine.stats["total_worlds"] -= 1
        
        return {
            "world_id": world_id,
            "name": world.name,
            "success": True,
            "message": "Mundo virtual eliminado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar mundo virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Avatares
@router.post("/avatars")
async def create_virtual_avatar(request: CreateAvatarRequest):
    """Crear avatar virtual."""
    try:
        avatar_type = AvatarType(request.avatar_type)
        
        avatar_id = await metaverse_engine.create_avatar(
            user_id=request.user_id,
            name=request.name,
            avatar_type=avatar_type,
            appearance=request.appearance,
            position=request.position,
            stats=request.stats
        )
        
        avatar = metaverse_engine.avatars[avatar_id]
        
        return {
            "avatar_id": avatar_id,
            "user_id": avatar.user_id,
            "name": avatar.name,
            "avatar_type": avatar.avatar_type.value,
            "appearance": avatar.appearance,
            "position": avatar.position,
            "stats": avatar.stats,
            "created_at": avatar.created_at.isoformat(),
            "success": True,
            "message": "Avatar virtual creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear avatar virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/avatars")
async def get_virtual_avatars(
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    avatar_type: Optional[str] = Query(None, description="Tipo de avatar"),
    limit: int = Query(100, description="Límite de avatares")
):
    """Obtener avatares virtuales."""
    try:
        avatars = []
        
        for avatar_id, avatar in metaverse_engine.avatars.items():
            # Filtrar por usuario
            if user_id and avatar.user_id != user_id:
                continue
            
            # Filtrar por tipo de avatar
            if avatar_type and avatar.avatar_type.value != avatar_type:
                continue
            
            avatars.append({
                "avatar_id": avatar_id,
                "user_id": avatar.user_id,
                "name": avatar.name,
                "avatar_type": avatar.avatar_type.value,
                "appearance": avatar.appearance,
                "position": avatar.position,
                "stats": avatar.stats,
                "created_at": avatar.created_at.isoformat(),
                "last_active": avatar.last_active.isoformat()
            })
        
        # Ordenar por actividad reciente
        avatars.sort(key=lambda x: x["last_active"], reverse=True)
        
        return {
            "avatars": avatars[:limit],
            "count": len(avatars[:limit]),
            "total_avatars": len(metaverse_engine.avatars),
            "filters": {
                "user_id": user_id,
                "avatar_type": avatar_type
            },
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener avatares virtuales: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/avatars/{avatar_id}")
async def get_virtual_avatar(avatar_id: str):
    """Obtener avatar virtual específico."""
    try:
        if avatar_id not in metaverse_engine.avatars:
            raise HTTPException(status_code=404, detail="Avatar no encontrado")
        
        avatar = metaverse_engine.avatars[avatar_id]
        
        return {
            "avatar": {
                "avatar_id": avatar_id,
                "user_id": avatar.user_id,
                "name": avatar.name,
                "avatar_type": avatar.avatar_type.value,
                "appearance": avatar.appearance,
                "position": avatar.position,
                "rotation": avatar.rotation,
                "scale": avatar.scale,
                "animations": avatar.animations,
                "inventory": avatar.inventory,
                "stats": avatar.stats,
                "created_at": avatar.created_at.isoformat(),
                "last_active": avatar.last_active.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener avatar virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Activos Virtuales
@router.post("/assets")
async def create_virtual_asset(request: CreateAssetRequest):
    """Crear activo virtual."""
    try:
        asset_type = AssetType(request.asset_type)
        price = Decimal(request.price) if request.price else None
        
        asset_id = await metaverse_engine.create_asset(
            name=request.name,
            asset_type=asset_type,
            owner_id=request.owner_id,
            world_id=request.world_id,
            position=request.position,
            properties=request.properties,
            is_nft=request.is_nft,
            nft_contract=request.nft_contract,
            nft_token_id=request.nft_token_id,
            price=price
        )
        
        asset = metaverse_engine.assets[asset_id]
        
        return {
            "asset_id": asset_id,
            "name": asset.name,
            "asset_type": asset.asset_type.value,
            "owner_id": asset.owner_id,
            "world_id": asset.world_id,
            "position": asset.position,
            "properties": asset.properties,
            "is_nft": asset.is_nft,
            "nft_contract": asset.nft_contract,
            "nft_token_id": asset.nft_token_id,
            "price": str(asset.price) if asset.price else None,
            "created_at": asset.created_at.isoformat(),
            "success": True,
            "message": "Activo virtual creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear activo virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets")
async def get_virtual_assets(
    owner_id: Optional[str] = Query(None, description="ID del propietario"),
    world_id: Optional[str] = Query(None, description="ID del mundo"),
    asset_type: Optional[str] = Query(None, description="Tipo de activo"),
    is_nft: Optional[bool] = Query(None, description="Activos NFT"),
    limit: int = Query(100, description="Límite de activos")
):
    """Obtener activos virtuales."""
    try:
        assets = []
        
        for asset_id, asset in metaverse_engine.assets.items():
            # Filtrar por propietario
            if owner_id and asset.owner_id != owner_id:
                continue
            
            # Filtrar por mundo
            if world_id and asset.world_id != world_id:
                continue
            
            # Filtrar por tipo de activo
            if asset_type and asset.asset_type.value != asset_type:
                continue
            
            # Filtrar por NFT
            if is_nft is not None and asset.is_nft != is_nft:
                continue
            
            assets.append({
                "asset_id": asset_id,
                "name": asset.name,
                "asset_type": asset.asset_type.value,
                "owner_id": asset.owner_id,
                "world_id": asset.world_id,
                "position": asset.position,
                "properties": asset.properties,
                "is_nft": asset.is_nft,
                "nft_contract": asset.nft_contract,
                "nft_token_id": asset.nft_token_id,
                "price": str(asset.price) if asset.price else None,
                "created_at": asset.created_at.isoformat()
            })
        
        # Ordenar por fecha de creación
        assets.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "assets": assets[:limit],
            "count": len(assets[:limit]),
            "total_assets": len(metaverse_engine.assets),
            "filters": {
                "owner_id": owner_id,
                "world_id": world_id,
                "asset_type": asset_type,
                "is_nft": is_nft
            },
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener activos virtuales: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Interacciones
@router.post("/interactions")
async def record_metaverse_interaction(request: RecordInteractionRequest):
    """Registrar interacción en el metaverso."""
    try:
        interaction_type = InteractionType(request.interaction_type)
        
        interaction_id = await metaverse_engine.record_interaction(
            user_id=request.user_id,
            avatar_id=request.avatar_id,
            world_id=request.world_id,
            interaction_type=interaction_type,
            target_id=request.target_id,
            data=request.data
        )
        
        return {
            "interaction_id": interaction_id,
            "user_id": request.user_id,
            "avatar_id": request.avatar_id,
            "world_id": request.world_id,
            "interaction_type": request.interaction_type,
            "target_id": request.target_id,
            "success": True,
            "message": "Interacción registrada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al registrar interacción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interactions")
async def get_metaverse_interactions(
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    world_id: Optional[str] = Query(None, description="ID del mundo"),
    interaction_type: Optional[str] = Query(None, description="Tipo de interacción"),
    limit: int = Query(100, description="Límite de interacciones")
):
    """Obtener interacciones del metaverso."""
    try:
        interactions = []
        
        for interaction_id, interaction in metaverse_engine.interactions.items():
            # Filtrar por usuario
            if user_id and interaction.user_id != user_id:
                continue
            
            # Filtrar por mundo
            if world_id and interaction.world_id != world_id:
                continue
            
            # Filtrar por tipo de interacción
            if interaction_type and interaction.interaction_type.value != interaction_type:
                continue
            
            interactions.append({
                "interaction_id": interaction_id,
                "user_id": interaction.user_id,
                "avatar_id": interaction.avatar_id,
                "world_id": interaction.world_id,
                "interaction_type": interaction.interaction_type.value,
                "target_id": interaction.target_id,
                "data": interaction.data,
                "timestamp": interaction.timestamp.isoformat()
            })
        
        # Ordenar por timestamp descendente
        interactions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "interactions": interactions[:limit],
            "count": len(interactions[:limit]),
            "total_interactions": len(metaverse_engine.interactions),
            "filters": {
                "user_id": user_id,
                "world_id": world_id,
                "interaction_type": interaction_type
            },
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener interacciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Estadísticas
@router.get("/stats")
async def get_metaverse_stats():
    """Obtener estadísticas del metaverso."""
    try:
        stats = await metaverse_engine.get_metaverse_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas del metaverso: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def metaverse_health_check():
    """Verificar salud del sistema de metaverso."""
    try:
        health = await metaverse_engine.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de metaverso: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/world-types")
async def get_world_types():
    """Obtener tipos de mundos disponibles."""
    return {
        "world_types": [
            {
                "value": world_type.value,
                "name": world_type.name,
                "description": f"Tipo de mundo {world_type.value}"
            }
            for world_type in WorldType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/avatar-types")
async def get_avatar_types():
    """Obtener tipos de avatares disponibles."""
    return {
        "avatar_types": [
            {
                "value": avatar_type.value,
                "name": avatar_type.name,
                "description": f"Tipo de avatar {avatar_type.value}"
            }
            for avatar_type in AvatarType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/asset-types")
async def get_asset_types():
    """Obtener tipos de activos disponibles."""
    return {
        "asset_types": [
            {
                "value": asset_type.value,
                "name": asset_type.name,
                "description": f"Tipo de activo {asset_type.value}"
            }
            for asset_type in AssetType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/interaction-types")
async def get_interaction_types():
    """Obtener tipos de interacciones disponibles."""
    return {
        "interaction_types": [
            {
                "value": interaction_type.value,
                "name": interaction_type.name,
                "description": f"Tipo de interacción {interaction_type.value}"
            }
            for interaction_type in InteractionType
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo
@router.post("/examples/create-virtual-space")
async def create_virtual_space_example():
    """Ejemplo: Crear espacio virtual completo."""
    try:
        # Crear mundo virtual
        world_id = await metaverse_engine.create_world(
            name="Espacio Virtual de Ejemplo",
            world_type=WorldType.SOCIAL,
            description="Un espacio virtual para demostrar las capacidades del metaverso",
            owner_id="demo_user",
            max_users=50,
            is_public=True,
            settings={"theme": "futuristic", "gravity": 9.8}
        )
        
        # Crear avatar
        avatar_id = await metaverse_engine.create_avatar(
            user_id="demo_user",
            name="Avatar Demo",
            avatar_type=AvatarType.HUMAN,
            appearance={"hair_color": "brown", "eye_color": "blue", "height": 1.75},
            position={"x": 0, "y": 0, "z": 0},
            stats={"level": 1, "experience": 0, "health": 100}
        )
        
        # Crear activos virtuales
        furniture_id = await metaverse_engine.create_asset(
            name="Silla Virtual",
            asset_type=AssetType.OBJECT,
            owner_id="demo_user",
            world_id=world_id,
            position={"x": 2, "y": 0, "z": 1},
            properties={"material": "wood", "color": "brown", "interactable": True}
        )
        
        nft_art_id = await metaverse_engine.create_asset(
            name="Arte Digital NFT",
            asset_type=AssetType.NFT,
            owner_id="demo_user",
            world_id=world_id,
            position={"x": 0, "y": 2, "z": 0},
            is_nft=True,
            nft_contract="0x1234567890abcdef",
            nft_token_id="1",
            price=Decimal("0.1")
        )
        
        # Registrar interacciones
        interaction_id = await metaverse_engine.record_interaction(
            user_id="demo_user",
            avatar_id=avatar_id,
            world_id=world_id,
            interaction_type=InteractionType.CHAT,
            data={"message": "¡Hola desde el metaverso!"}
        )
        
        world = metaverse_engine.worlds[world_id]
        avatar = metaverse_engine.avatars[avatar_id]
        
        return {
            "virtual_space": {
                "world": {
                    "world_id": world_id,
                    "name": world.name,
                    "world_type": world.world_type.value,
                    "description": world.description,
                    "max_users": world.max_users,
                    "created_at": world.created_at.isoformat()
                },
                "avatar": {
                    "avatar_id": avatar_id,
                    "name": avatar.name,
                    "avatar_type": avatar.avatar_type.value,
                    "appearance": avatar.appearance,
                    "position": avatar.position,
                    "stats": avatar.stats
                },
                "assets": {
                    "furniture": furniture_id,
                    "nft_art": nft_art_id
                },
                "interaction": interaction_id
            },
            "success": True,
            "message": "Espacio virtual de ejemplo creado exitosamente",
            "description": "Se creó un mundo virtual completo con avatar, activos e interacciones",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de espacio virtual: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples/metaverse-dashboard")
async def get_metaverse_dashboard():
    """Ejemplo: Obtener datos para dashboard del metaverso."""
    try:
        # Obtener estadísticas generales
        stats = await metaverse_engine.get_metaverse_stats()
        
        # Obtener mundos populares
        popular_worlds = []
        for world_id, world in metaverse_engine.worlds.items():
            popular_worlds.append({
                "world_id": world_id,
                "name": world.name,
                "world_type": world.world_type.value,
                "current_users": world.current_users,
                "max_users": world.max_users,
                "last_activity": world.last_activity.isoformat()
            })
        
        # Ordenar por usuarios activos
        popular_worlds.sort(key=lambda x: x["current_users"], reverse=True)
        popular_worlds = popular_worlds[:10]  # Top 10
        
        # Obtener avatares activos
        active_avatars = []
        for avatar_id, avatar in metaverse_engine.avatars.items():
            active_avatars.append({
                "avatar_id": avatar_id,
                "name": avatar.name,
                "avatar_type": avatar.avatar_type.value,
                "user_id": avatar.user_id,
                "last_active": avatar.last_active.isoformat()
            })
        
        # Ordenar por actividad reciente
        active_avatars.sort(key=lambda x: x["last_active"], reverse=True)
        active_avatars = active_avatars[:20]  # Top 20
        
        # Obtener activos NFT
        nft_assets = []
        for asset_id, asset in metaverse_engine.assets.items():
            if asset.is_nft:
                nft_assets.append({
                    "asset_id": asset_id,
                    "name": asset.name,
                    "asset_type": asset.asset_type.value,
                    "owner_id": asset.owner_id,
                    "world_id": asset.world_id,
                    "price": str(asset.price) if asset.price else None,
                    "nft_contract": asset.nft_contract,
                    "nft_token_id": asset.nft_token_id
                })
        
        # Ordenar por precio
        nft_assets.sort(key=lambda x: float(x["price"]) if x["price"] else 0, reverse=True)
        nft_assets = nft_assets[:10]  # Top 10 NFT
        
        return {
            "dashboard": {
                "stats": stats,
                "popular_worlds": popular_worlds,
                "active_avatars": active_avatars,
                "nft_assets": nft_assets,
                "world_types_distribution": stats.get("world_types_distribution", {}),
                "avatar_types_distribution": stats.get("avatar_types_distribution", {}),
                "asset_types_distribution": stats.get("asset_types_distribution", {})
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener dashboard del metaverso: {e}")
        raise HTTPException(status_code=500, detail=str(e))




