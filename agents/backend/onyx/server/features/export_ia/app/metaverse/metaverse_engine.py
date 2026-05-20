"""
Metaverse Engine - Motor de Metaverso y Realidad Virtual
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import hashlib
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class WorldType(Enum):
    """Tipos de mundos virtuales."""
    GAME = "game"
    SOCIAL = "social"
    EDUCATIONAL = "educational"
    BUSINESS = "business"
    ENTERTAINMENT = "entertainment"
    ART = "art"
    MUSIC = "music"
    SPORTS = "sports"
    SHOPPING = "shopping"
    CUSTOM = "custom"


class AvatarType(Enum):
    """Tipos de avatares."""
    HUMAN = "human"
    ANIMAL = "animal"
    ROBOT = "robot"
    FANTASY = "fantasy"
    ABSTRACT = "abstract"
    CUSTOM = "custom"


class InteractionType(Enum):
    """Tipos de interacciones."""
    CHAT = "chat"
    GESTURE = "gesture"
    EMOTE = "emote"
    TOUCH = "touch"
    VOICE = "voice"
    MOVEMENT = "movement"
    OBJECT_MANIPULATION = "object_manipulation"
    COLLABORATION = "collaboration"


class AssetType(Enum):
    """Tipos de activos virtuales."""
    AVATAR = "avatar"
    OBJECT = "object"
    BUILDING = "building"
    LANDSCAPE = "landscape"
    VEHICLE = "vehicle"
    WEAPON = "weapon"
    CLOTHING = "clothing"
    ACCESSORY = "accessory"
    NFT = "nft"
    CURRENCY = "currency"


@dataclass
class VirtualWorld:
    """Mundo virtual."""
    world_id: str
    name: str
    world_type: WorldType
    description: str
    max_users: int = 100
    current_users: int = 0
    owner_id: str = ""
    is_public: bool = True
    password: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class Avatar:
    """Avatar virtual."""
    avatar_id: str
    user_id: str
    name: str
    avatar_type: AvatarType
    appearance: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, float] = field(default_factory=dict)
    rotation: Dict[str, float] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)
    animations: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class VirtualAsset:
    """Activo virtual."""
    asset_id: str
    name: str
    asset_type: AssetType
    owner_id: str
    world_id: Optional[str] = None
    position: Dict[str, float] = field(default_factory=dict)
    rotation: Dict[str, float] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_nft: bool = False
    nft_contract: Optional[str] = None
    nft_token_id: Optional[str] = None
    price: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MetaverseInteraction:
    """Interacción en el metaverso."""
    interaction_id: str
    user_id: str
    avatar_id: str
    world_id: str
    interaction_type: InteractionType
    target_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MetaverseEngine:
    """
    Motor de Metaverso y Realidad Virtual.
    """
    
    def __init__(self, config_directory: str = "metaverse_config"):
        """Inicializar motor de metaverso."""
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        
        # Mundos virtuales
        self.worlds: Dict[str, VirtualWorld] = {}
        self.avatars: Dict[str, Avatar] = {}
        self.assets: Dict[str, VirtualAsset] = {}
        self.interactions: Dict[str, MetaverseInteraction] = {}
        
        # Configuración
        self.max_worlds = 1000
        self.max_avatars_per_user = 5
        self.max_assets_per_world = 10000
        self.interaction_history_days = 30
        
        # Estadísticas
        self.stats = {
            "total_worlds": 0,
            "total_avatars": 0,
            "total_assets": 0,
            "total_interactions": 0,
            "active_users": 0,
            "total_online_time": 0,
            "start_time": datetime.now()
        }
        
        # Inicializar sistemas
        self._initialize_systems()
        
        logger.info("MetaverseEngine inicializado")
    
    async def initialize(self):
        """Inicializar el motor de metaverso."""
        try:
            # Cargar datos existentes
            self._load_worlds()
            self._load_avatars()
            self._load_assets()
            
            # Iniciar sistemas de monitoreo
            asyncio.create_task(self._monitor_worlds())
            asyncio.create_task(self._process_interactions())
            
            logger.info("MetaverseEngine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar MetaverseEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de metaverso."""
        try:
            # Guardar datos
            await self._save_worlds()
            await self._save_avatars()
            await self._save_assets()
            
            logger.info("MetaverseEngine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar MetaverseEngine: {e}")
    
    def _initialize_systems(self):
        """Inicializar sistemas del metaverso."""
        try:
            # Inicializar sistemas de renderizado, física, etc.
            self.physics_engine = None  # Se integraría con un motor de física
            self.rendering_engine = None  # Se integraría con un motor de renderizado
            self.audio_engine = None  # Se integraría con un motor de audio
            
            logger.info("Sistemas del metaverso inicializados")
            
        except Exception as e:
            logger.error(f"Error al inicializar sistemas: {e}")
    
    async def _monitor_worlds(self):
        """Monitorear mundos virtuales."""
        while True:
            try:
                for world_id, world in self.worlds.items():
                    # Actualizar estadísticas del mundo
                    world.current_users = min(world.current_users, world.max_users)
                    
                    # Verificar actividad
                    if (datetime.now() - world.last_activity).total_seconds() > 3600:
                        world.current_users = max(0, world.current_users - 1)
                
                # Actualizar estadísticas globales
                self.stats["active_users"] = sum(
                    world.current_users for world in self.worlds.values()
                )
                
                await asyncio.sleep(60)  # Monitorear cada minuto
                
            except Exception as e:
                logger.error(f"Error en monitoreo de mundos: {e}")
                await asyncio.sleep(300)
    
    async def _process_interactions(self):
        """Procesar interacciones del metaverso."""
        while True:
            try:
                # Procesar interacciones recientes
                recent_interactions = [
                    interaction for interaction in self.interactions.values()
                    if (datetime.now() - interaction.timestamp).total_seconds() < 300
                ]
                
                for interaction in recent_interactions:
                    # Procesar diferentes tipos de interacciones
                    await self._handle_interaction(interaction)
                
                # Limpiar interacciones antiguas
                cutoff_time = datetime.now() - timedelta(days=self.interaction_history_days)
                self.interactions = {
                    interaction_id: interaction
                    for interaction_id, interaction in self.interactions.items()
                    if interaction.timestamp > cutoff_time
                }
                
                await asyncio.sleep(10)  # Procesar cada 10 segundos
                
            except Exception as e:
                logger.error(f"Error en procesamiento de interacciones: {e}")
                await asyncio.sleep(30)
    
    async def _handle_interaction(self, interaction: MetaverseInteraction):
        """Manejar interacción específica."""
        try:
            if interaction.interaction_type == InteractionType.CHAT:
                # Procesar chat
                await self._process_chat_interaction(interaction)
                
            elif interaction.interaction_type == InteractionType.MOVEMENT:
                # Procesar movimiento
                await self._process_movement_interaction(interaction)
                
            elif interaction.interaction_type == InteractionType.OBJECT_MANIPULATION:
                # Procesar manipulación de objetos
                await self._process_object_interaction(interaction)
                
            elif interaction.interaction_type == InteractionType.COLLABORATION:
                # Procesar colaboración
                await self._process_collaboration_interaction(interaction)
                
        except Exception as e:
            logger.error(f"Error al manejar interacción: {e}")
    
    async def _process_chat_interaction(self, interaction: MetaverseInteraction):
        """Procesar interacción de chat."""
        # Implementar lógica de chat
        pass
    
    async def _process_movement_interaction(self, interaction: MetaverseInteraction):
        """Procesar interacción de movimiento."""
        # Implementar lógica de movimiento
        pass
    
    async def _process_object_interaction(self, interaction: MetaverseInteraction):
        """Procesar interacción con objetos."""
        # Implementar lógica de manipulación de objetos
        pass
    
    async def _process_collaboration_interaction(self, interaction: MetaverseInteraction):
        """Procesar interacción de colaboración."""
        # Implementar lógica de colaboración
        pass
    
    def _load_worlds(self):
        """Cargar mundos existentes."""
        try:
            worlds_file = self.config_directory / "worlds.json"
            if worlds_file.exists():
                with open(worlds_file, 'r') as f:
                    worlds_data = json.load(f)
                
                for world_id, data in worlds_data.items():
                    data['world_type'] = WorldType(data['world_type'])
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['last_activity'] = datetime.fromisoformat(data['last_activity'])
                    
                    self.worlds[world_id] = VirtualWorld(**data)
                
                logger.info(f"Cargados {len(self.worlds)} mundos virtuales")
                
        except Exception as e:
            logger.error(f"Error al cargar mundos: {e}")
    
    async def _save_worlds(self):
        """Guardar mundos."""
        try:
            worlds_file = self.config_directory / "worlds.json"
            
            worlds_data = {}
            for world_id, world in self.worlds.items():
                data = world.__dict__.copy()
                data['world_type'] = data['world_type'].value
                data['created_at'] = data['created_at'].isoformat()
                data['last_activity'] = data['last_activity'].isoformat()
                worlds_data[world_id] = data
            
            with open(worlds_file, 'w') as f:
                json.dump(worlds_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar mundos: {e}")
    
    def _load_avatars(self):
        """Cargar avatares existentes."""
        try:
            avatars_file = self.config_directory / "avatars.json"
            if avatars_file.exists():
                with open(avatars_file, 'r') as f:
                    avatars_data = json.load(f)
                
                for avatar_id, data in avatars_data.items():
                    data['avatar_type'] = AvatarType(data['avatar_type'])
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['last_active'] = datetime.fromisoformat(data['last_active'])
                    
                    self.avatars[avatar_id] = Avatar(**data)
                
                logger.info(f"Cargados {len(self.avatars)} avatares")
                
        except Exception as e:
            logger.error(f"Error al cargar avatares: {e}")
    
    async def _save_avatars(self):
        """Guardar avatares."""
        try:
            avatars_file = self.config_directory / "avatars.json"
            
            avatars_data = {}
            for avatar_id, avatar in self.avatars.items():
                data = avatar.__dict__.copy()
                data['avatar_type'] = data['avatar_type'].value
                data['created_at'] = data['created_at'].isoformat()
                data['last_active'] = data['last_active'].isoformat()
                avatars_data[avatar_id] = data
            
            with open(avatars_file, 'w') as f:
                json.dump(avatars_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar avatares: {e}")
    
    def _load_assets(self):
        """Cargar activos existentes."""
        try:
            assets_file = self.config_directory / "assets.json"
            if assets_file.exists():
                with open(assets_file, 'r') as f:
                    assets_data = json.load(f)
                
                for asset_id, data in assets_data.items():
                    data['asset_type'] = AssetType(data['asset_type'])
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    if data.get('price'):
                        data['price'] = Decimal(str(data['price']))
                    
                    self.assets[asset_id] = VirtualAsset(**data)
                
                logger.info(f"Cargados {len(self.assets)} activos virtuales")
                
        except Exception as e:
            logger.error(f"Error al cargar activos: {e}")
    
    async def _save_assets(self):
        """Guardar activos."""
        try:
            assets_file = self.config_directory / "assets.json"
            
            assets_data = {}
            for asset_id, asset in self.assets.items():
                data = asset.__dict__.copy()
                data['asset_type'] = data['asset_type'].value
                data['created_at'] = data['created_at'].isoformat()
                if data.get('price'):
                    data['price'] = str(data['price'])
                assets_data[asset_id] = data
            
            with open(assets_file, 'w') as f:
                json.dump(assets_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar activos: {e}")
    
    async def create_world(
        self,
        name: str,
        world_type: WorldType,
        description: str,
        owner_id: str,
        max_users: int = 100,
        is_public: bool = True,
        password: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """Crear mundo virtual."""
        try:
            if len(self.worlds) >= self.max_worlds:
                raise ValueError("Límite máximo de mundos alcanzado")
            
            world_id = str(uuid.uuid4())
            
            world = VirtualWorld(
                world_id=world_id,
                name=name,
                world_type=world_type,
                description=description,
                max_users=max_users,
                owner_id=owner_id,
                is_public=is_public,
                password=password,
                settings=settings or {}
            )
            
            self.worlds[world_id] = world
            self.stats["total_worlds"] += 1
            
            logger.info(f"Mundo virtual creado: {name} ({world_type.value})")
            return world_id
            
        except Exception as e:
            logger.error(f"Error al crear mundo virtual: {e}")
            raise
    
    async def create_avatar(
        self,
        user_id: str,
        name: str,
        avatar_type: AvatarType,
        appearance: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, float]] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """Crear avatar virtual."""
        try:
            # Verificar límite de avatares por usuario
            user_avatars = [
                avatar for avatar in self.avatars.values()
                if avatar.user_id == user_id
            ]
            
            if len(user_avatars) >= self.max_avatars_per_user:
                raise ValueError(f"Límite máximo de avatares por usuario alcanzado ({self.max_avatars_per_user})")
            
            avatar_id = str(uuid.uuid4())
            
            avatar = Avatar(
                avatar_id=avatar_id,
                user_id=user_id,
                name=name,
                avatar_type=avatar_type,
                appearance=appearance or {},
                position=position or {"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                scale={"x": 1, "y": 1, "z": 1},
                stats=stats or {}
            )
            
            self.avatars[avatar_id] = avatar
            self.stats["total_avatars"] += 1
            
            logger.info(f"Avatar virtual creado: {name} ({avatar_type.value})")
            return avatar_id
            
        except Exception as e:
            logger.error(f"Error al crear avatar virtual: {e}")
            raise
    
    async def create_asset(
        self,
        name: str,
        asset_type: AssetType,
        owner_id: str,
        world_id: Optional[str] = None,
        position: Optional[Dict[str, float]] = None,
        properties: Optional[Dict[str, Any]] = None,
        is_nft: bool = False,
        nft_contract: Optional[str] = None,
        nft_token_id: Optional[str] = None,
        price: Optional[Decimal] = None
    ) -> str:
        """Crear activo virtual."""
        try:
            # Verificar límite de activos por mundo
            if world_id:
                world_assets = [
                    asset for asset in self.assets.values()
                    if asset.world_id == world_id
                ]
                
                if len(world_assets) >= self.max_assets_per_world:
                    raise ValueError(f"Límite máximo de activos por mundo alcanzado ({self.max_assets_per_world})")
            
            asset_id = str(uuid.uuid4())
            
            asset = VirtualAsset(
                asset_id=asset_id,
                name=name,
                asset_type=asset_type,
                owner_id=owner_id,
                world_id=world_id,
                position=position or {"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                scale={"x": 1, "y": 1, "z": 1},
                properties=properties or {},
                is_nft=is_nft,
                nft_contract=nft_contract,
                nft_token_id=nft_token_id,
                price=price
            )
            
            self.assets[asset_id] = asset
            self.stats["total_assets"] += 1
            
            logger.info(f"Activo virtual creado: {name} ({asset_type.value})")
            return asset_id
            
        except Exception as e:
            logger.error(f"Error al crear activo virtual: {e}")
            raise
    
    async def record_interaction(
        self,
        user_id: str,
        avatar_id: str,
        world_id: str,
        interaction_type: InteractionType,
        target_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Registrar interacción en el metaverso."""
        try:
            interaction_id = str(uuid.uuid4())
            
            interaction = MetaverseInteraction(
                interaction_id=interaction_id,
                user_id=user_id,
                avatar_id=avatar_id,
                world_id=world_id,
                interaction_type=interaction_type,
                target_id=target_id,
                data=data or {}
            )
            
            self.interactions[interaction_id] = interaction
            self.stats["total_interactions"] += 1
            
            # Actualizar actividad del mundo
            if world_id in self.worlds:
                self.worlds[world_id].last_activity = datetime.now()
            
            # Actualizar actividad del avatar
            if avatar_id in self.avatars:
                self.avatars[avatar_id].last_active = datetime.now()
            
            logger.debug(f"Interacción registrada: {interaction_type.value}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error al registrar interacción: {e}")
            raise
    
    async def get_world_info(self, world_id: str) -> Dict[str, Any]:
        """Obtener información de mundo virtual."""
        try:
            if world_id not in self.worlds:
                raise ValueError(f"Mundo {world_id} no encontrado")
            
            world = self.worlds[world_id]
            
            # Obtener avatares en el mundo
            avatars_in_world = [
                avatar for avatar in self.avatars.values()
                if avatar.avatar_id in [interaction.avatar_id for interaction in self.interactions.values()
                                      if interaction.world_id == world_id and
                                      (datetime.now() - interaction.timestamp).total_seconds() < 300]
            ]
            
            # Obtener activos del mundo
            world_assets = [
                asset for asset in self.assets.values()
                if asset.world_id == world_id
            ]
            
            return {
                "world_id": world_id,
                "name": world.name,
                "world_type": world.world_type.value,
                "description": world.description,
                "max_users": world.max_users,
                "current_users": world.current_users,
                "owner_id": world.owner_id,
                "is_public": world.is_public,
                "settings": world.settings,
                "created_at": world.created_at.isoformat(),
                "last_activity": world.last_activity.isoformat(),
                "avatars_count": len(avatars_in_world),
                "assets_count": len(world_assets),
                "nft_assets_count": len([asset for asset in world_assets if asset.is_nft])
            }
            
        except Exception as e:
            logger.error(f"Error al obtener información de mundo: {e}")
            raise
    
    async def get_metaverse_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del metaverso."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "worlds_count": len(self.worlds),
            "avatars_count": len(self.avatars),
            "assets_count": len(self.assets),
            "interactions_count": len(self.interactions),
            "nft_assets_count": len([asset for asset in self.assets.values() if asset.is_nft]),
            "world_types_distribution": {
                world_type.value: sum(
                    1 for world in self.worlds.values()
                    if world.world_type == world_type
                )
                for world_type in WorldType
            },
            "avatar_types_distribution": {
                avatar_type.value: sum(
                    1 for avatar in self.avatars.values()
                    if avatar.avatar_type == avatar_type
                )
                for avatar_type in AvatarType
            },
            "asset_types_distribution": {
                asset_type.value: sum(
                    1 for asset in self.assets.values()
                    if asset.asset_type == asset_type
                )
                for asset_type in AssetType
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de metaverso."""
        try:
            return {
                "status": "healthy",
                "worlds_count": len(self.worlds),
                "avatars_count": len(self.avatars),
                "assets_count": len(self.assets),
                "interactions_count": len(self.interactions),
                "active_users": self.stats["active_users"],
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de metaverso: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




