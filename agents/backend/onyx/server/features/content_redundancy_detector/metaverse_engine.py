"""
Metaverse Engine for Advanced Metaverse Processing
Motor de Metaverso para procesamiento avanzado de metaverso ultra-optimizado
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from datetime import datetime, timedelta
import statistics
import random
import math

logger = logging.getLogger(__name__)


class MetaverseType(Enum):
    """Tipos de metaverso"""
    VIRTUAL_WORLD = "virtual_world"
    GAMING = "gaming"
    SOCIAL = "social"
    EDUCATIONAL = "educational"
    BUSINESS = "business"
    ENTERTAINMENT = "entertainment"
    RETAIL = "retail"
    REAL_ESTATE = "real_estate"
    ART = "art"
    MUSIC = "music"
    SPORTS = "sports"
    HEALTHCARE = "healthcare"
    TRAVEL = "travel"
    WORKPLACE = "workplace"
    CUSTOM = "custom"


class AvatarType(Enum):
    """Tipos de avatares"""
    HUMAN = "human"
    ANIMAL = "animal"
    ROBOT = "robot"
    FANTASY = "fantasy"
    ABSTRACT = "abstract"
    CUSTOM = "custom"


class EnvironmentType(Enum):
    """Tipos de entornos"""
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    CARTOON = "cartoon"
    SCI_FI = "sci_fi"
    FANTASY = "fantasy"
    MINIMALIST = "minimalist"
    SURREAL = "surreal"
    CUSTOM = "custom"


@dataclass
class MetaverseWorld:
    """Mundo del metaverso"""
    id: str
    name: str
    description: str
    metaverse_type: MetaverseType
    environment_type: EnvironmentType
    max_users: int
    current_users: int
    world_size: Tuple[float, float, float]  # width, height, depth
    spawn_point: Tuple[float, float, float]  # x, y, z
    physics_enabled: bool
    gravity: float
    lighting: Dict[str, Any]
    weather: Dict[str, Any]
    audio: Dict[str, Any]
    created_at: float
    last_modified: float
    metadata: Dict[str, Any]


@dataclass
class Avatar:
    """Avatar del metaverso"""
    id: str
    user_id: str
    name: str
    avatar_type: AvatarType
    appearance: Dict[str, Any]
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    animations: List[str]
    accessories: List[Dict[str, Any]]
    world_id: str
    created_at: float
    last_updated: float
    metadata: Dict[str, Any]


@dataclass
class MetaverseObject:
    """Objeto del metaverso"""
    id: str
    name: str
    object_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    mesh_data: Dict[str, Any]
    texture_data: Dict[str, Any]
    physics_properties: Dict[str, Any]
    interactive: bool
    world_id: str
    created_at: float
    last_modified: float
    metadata: Dict[str, Any]


@dataclass
class MetaverseEvent:
    """Evento del metaverso"""
    id: str
    event_type: str
    world_id: str
    user_id: Optional[str]
    object_id: Optional[str]
    position: Tuple[float, float, float]
    data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]


class MetaverseRenderer:
    """Renderizador del metaverso"""
    
    def __init__(self):
        self.render_engines = {
            "webgl": self._render_webgl,
            "webgpu": self._render_webgpu,
            "vulkan": self._render_vulkan,
            "directx": self._render_directx,
            "opengl": self._render_opengl,
            "metal": self._render_metal
        }
    
    async def render_world(self, world: MetaverseWorld, render_engine: str = "webgl") -> Dict[str, Any]:
        """Renderizar mundo del metaverso"""
        try:
            renderer = self.render_engines.get(render_engine)
            if not renderer:
                raise ValueError(f"Unsupported render engine: {render_engine}")
            
            return await renderer(world)
            
        except Exception as e:
            logger.error(f"Error rendering metaverse world: {e}")
            raise
    
    async def _render_webgl(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Renderizar con WebGL"""
        return {
            "render_engine": "webgl",
            "fps": random.uniform(30, 60),
            "draw_calls": random.randint(100, 1000),
            "triangles": random.randint(10000, 100000),
            "textures_loaded": random.randint(10, 100),
            "shaders_compiled": random.randint(5, 20),
            "memory_usage": random.uniform(50, 500),  # MB
            "render_time": random.uniform(10, 50)  # ms
        }
    
    async def _render_webgpu(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Renderizar con WebGPU"""
        return {
            "render_engine": "webgpu",
            "fps": random.uniform(40, 80),
            "draw_calls": random.randint(200, 2000),
            "triangles": random.randint(20000, 200000),
            "textures_loaded": random.randint(20, 200),
            "shaders_compiled": random.randint(10, 40),
            "memory_usage": random.uniform(100, 800),  # MB
            "render_time": random.uniform(5, 30)  # ms
        }
    
    async def _render_vulkan(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Renderizar con Vulkan"""
        return {
            "render_engine": "vulkan",
            "fps": random.uniform(50, 120),
            "draw_calls": random.randint(500, 5000),
            "triangles": random.randint(50000, 500000),
            "textures_loaded": random.randint(50, 500),
            "shaders_compiled": random.randint(20, 100),
            "memory_usage": random.uniform(200, 1000),  # MB
            "render_time": random.uniform(2, 20)  # ms
        }
    
    async def _render_directx(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Renderizar con DirectX"""
        return {
            "render_engine": "directx",
            "fps": random.uniform(45, 100),
            "draw_calls": random.randint(300, 3000),
            "triangles": random.randint(30000, 300000),
            "textures_loaded": random.randint(30, 300),
            "shaders_compiled": random.randint(15, 60),
            "memory_usage": random.uniform(150, 700),  # MB
            "render_time": random.uniform(3, 25)  # ms
        }
    
    async def _render_opengl(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Renderizar con OpenGL"""
        return {
            "render_engine": "opengl",
            "fps": random.uniform(35, 70),
            "draw_calls": random.randint(150, 1500),
            "triangles": random.randint(15000, 150000),
            "textures_loaded": random.randint(25, 250),
            "shaders_compiled": random.randint(12, 50),
            "memory_usage": random.uniform(80, 600),  # MB
            "render_time": random.uniform(8, 40)  # ms
        }
    
    async def _render_metal(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Renderizar con Metal"""
        return {
            "render_engine": "metal",
            "fps": random.uniform(55, 110),
            "draw_calls": random.randint(400, 4000),
            "triangles": random.randint(40000, 400000),
            "textures_loaded": random.randint(40, 400),
            "shaders_compiled": random.randint(18, 80),
            "memory_usage": random.uniform(180, 900),  # MB
            "render_time": random.uniform(2, 18)  # ms
        }


class MetaversePhysics:
    """Física del metaverso"""
    
    def __init__(self):
        self.physics_engines = {
            "bullet": self._physics_bullet,
            "havok": self._physics_havok,
            "physx": self._physics_physx,
            "box2d": self._physics_box2d,
            "cannon": self._physics_cannon
        }
    
    async def simulate_physics(self, world: MetaverseWorld, physics_engine: str = "bullet") -> Dict[str, Any]:
        """Simular física del metaverso"""
        try:
            engine = self.physics_engines.get(physics_engine)
            if not engine:
                raise ValueError(f"Unsupported physics engine: {physics_engine}")
            
            return await engine(world)
            
        except Exception as e:
            logger.error(f"Error simulating metaverse physics: {e}")
            raise
    
    async def _physics_bullet(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Simular física con Bullet"""
        return {
            "physics_engine": "bullet",
            "collision_checks": random.randint(100, 1000),
            "rigid_bodies": random.randint(10, 100),
            "constraints": random.randint(0, 50),
            "simulation_time": random.uniform(1, 10),  # ms
            "memory_usage": random.uniform(10, 100),  # MB
            "cpu_usage": random.uniform(5, 30)  # %
        }
    
    async def _physics_havok(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Simular física con Havok"""
        return {
            "physics_engine": "havok",
            "collision_checks": random.randint(200, 2000),
            "rigid_bodies": random.randint(20, 200),
            "constraints": random.randint(0, 100),
            "simulation_time": random.uniform(0.5, 5),  # ms
            "memory_usage": random.uniform(20, 200),  # MB
            "cpu_usage": random.uniform(10, 40)  # %
        }
    
    async def _physics_physx(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Simular física con PhysX"""
        return {
            "physics_engine": "physx",
            "collision_checks": random.randint(150, 1500),
            "rigid_bodies": random.randint(15, 150),
            "constraints": random.randint(0, 75),
            "simulation_time": random.uniform(0.8, 8),  # ms
            "memory_usage": random.uniform(15, 150),  # MB
            "cpu_usage": random.uniform(8, 35)  # %
        }
    
    async def _physics_box2d(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Simular física con Box2D"""
        return {
            "physics_engine": "box2d",
            "collision_checks": random.randint(50, 500),
            "rigid_bodies": random.randint(5, 50),
            "constraints": random.randint(0, 25),
            "simulation_time": random.uniform(2, 15),  # ms
            "memory_usage": random.uniform(5, 50),  # MB
            "cpu_usage": random.uniform(3, 20)  # %
        }
    
    async def _physics_cannon(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Simular física con Cannon.js"""
        return {
            "physics_engine": "cannon",
            "collision_checks": random.randint(80, 800),
            "rigid_bodies": random.randint(8, 80),
            "constraints": random.randint(0, 40),
            "simulation_time": random.uniform(1.5, 12),  # ms
            "memory_usage": random.uniform(8, 80),  # MB
            "cpu_usage": random.uniform(4, 25)  # %
        }


class MetaverseAudio:
    """Audio del metaverso"""
    
    def __init__(self):
        self.audio_engines = {
            "webaudio": self._audio_webaudio,
            "openal": self._audio_openal,
            "fmod": self._audio_fmod,
            "wwise": self._audio_wwise,
            "xaudio": self._audio_xaudio
        }
    
    async def process_audio(self, world: MetaverseWorld, audio_engine: str = "webaudio") -> Dict[str, Any]:
        """Procesar audio del metaverso"""
        try:
            engine = self.audio_engines.get(audio_engine)
            if not engine:
                raise ValueError(f"Unsupported audio engine: {audio_engine}")
            
            return await engine(world)
            
        except Exception as e:
            logger.error(f"Error processing metaverse audio: {e}")
            raise
    
    async def _audio_webaudio(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Procesar audio con WebAudio"""
        return {
            "audio_engine": "webaudio",
            "active_sources": random.randint(5, 50),
            "sample_rate": 44100,
            "buffer_size": 1024,
            "latency": random.uniform(10, 50),  # ms
            "cpu_usage": random.uniform(2, 15),  # %
            "memory_usage": random.uniform(5, 50)  # MB
        }
    
    async def _audio_openal(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Procesar audio con OpenAL"""
        return {
            "audio_engine": "openal",
            "active_sources": random.randint(10, 100),
            "sample_rate": 48000,
            "buffer_size": 512,
            "latency": random.uniform(5, 25),  # ms
            "cpu_usage": random.uniform(3, 20),  # %
            "memory_usage": random.uniform(10, 100)  # MB
        }
    
    async def _audio_fmod(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Procesar audio con FMOD"""
        return {
            "audio_engine": "fmod",
            "active_sources": random.randint(15, 150),
            "sample_rate": 48000,
            "buffer_size": 256,
            "latency": random.uniform(3, 15),  # ms
            "cpu_usage": random.uniform(5, 25),  # %
            "memory_usage": random.uniform(15, 150)  # MB
        }
    
    async def _audio_wwise(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Procesar audio con Wwise"""
        return {
            "audio_engine": "wwise",
            "active_sources": random.randint(20, 200),
            "sample_rate": 48000,
            "buffer_size": 128,
            "latency": random.uniform(2, 10),  # ms
            "cpu_usage": random.uniform(8, 30),  # %
            "memory_usage": random.uniform(20, 200)  # MB
        }
    
    async def _audio_xaudio(self, world: MetaverseWorld) -> Dict[str, Any]:
        """Procesar audio con XAudio2"""
        return {
            "audio_engine": "xaudio",
            "active_sources": random.randint(12, 120),
            "sample_rate": 48000,
            "buffer_size": 384,
            "latency": random.uniform(4, 20),  # ms
            "cpu_usage": random.uniform(4, 22),  # %
            "memory_usage": random.uniform(12, 120)  # MB
        }


class MetaverseEngine:
    """Motor principal del metaverso"""
    
    def __init__(self):
        self.worlds: Dict[str, MetaverseWorld] = {}
        self.avatars: Dict[str, Avatar] = {}
        self.objects: Dict[str, MetaverseObject] = {}
        self.events: Dict[str, MetaverseEvent] = {}
        self.renderer = MetaverseRenderer()
        self.physics = MetaversePhysics()
        self.audio = MetaverseAudio()
        self.is_running = False
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar motor del metaverso"""
        try:
            self.is_running = True
            logger.info("Metaverse engine started")
        except Exception as e:
            logger.error(f"Error starting metaverse engine: {e}")
            raise
    
    async def stop(self):
        """Detener motor del metaverso"""
        try:
            self.is_running = False
            logger.info("Metaverse engine stopped")
        except Exception as e:
            logger.error(f"Error stopping metaverse engine: {e}")
    
    async def create_metaverse_world(self, world_info: Dict[str, Any]) -> str:
        """Crear mundo del metaverso"""
        world_id = f"world_{uuid.uuid4().hex[:8]}"
        
        world = MetaverseWorld(
            id=world_id,
            name=world_info["name"],
            description=world_info.get("description", ""),
            metaverse_type=MetaverseType(world_info["metaverse_type"]),
            environment_type=EnvironmentType(world_info["environment_type"]),
            max_users=world_info.get("max_users", 100),
            current_users=0,
            world_size=tuple(world_info.get("world_size", [1000, 1000, 1000])),
            spawn_point=tuple(world_info.get("spawn_point", [0, 0, 0])),
            physics_enabled=world_info.get("physics_enabled", True),
            gravity=world_info.get("gravity", -9.81),
            lighting=world_info.get("lighting", {}),
            weather=world_info.get("weather", {}),
            audio=world_info.get("audio", {}),
            created_at=time.time(),
            last_modified=time.time(),
            metadata=world_info.get("metadata", {})
        )
        
        async with self._lock:
            self.worlds[world_id] = world
        
        logger.info(f"Metaverse world created: {world_id} ({world.name})")
        return world_id
    
    async def create_avatar(self, avatar_info: Dict[str, Any]) -> str:
        """Crear avatar del metaverso"""
        avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
        
        avatar = Avatar(
            id=avatar_id,
            user_id=avatar_info["user_id"],
            name=avatar_info["name"],
            avatar_type=AvatarType(avatar_info["avatar_type"]),
            appearance=avatar_info.get("appearance", {}),
            position=tuple(avatar_info.get("position", [0, 0, 0])),
            rotation=tuple(avatar_info.get("rotation", [0, 0, 0])),
            scale=tuple(avatar_info.get("scale", [1, 1, 1])),
            animations=avatar_info.get("animations", []),
            accessories=avatar_info.get("accessories", []),
            world_id=avatar_info["world_id"],
            created_at=time.time(),
            last_updated=time.time(),
            metadata=avatar_info.get("metadata", {})
        )
        
        async with self._lock:
            self.avatars[avatar_id] = avatar
        
        logger.info(f"Metaverse avatar created: {avatar_id} ({avatar.name})")
        return avatar_id
    
    async def create_metaverse_object(self, object_info: Dict[str, Any]) -> str:
        """Crear objeto del metaverso"""
        object_id = f"object_{uuid.uuid4().hex[:8]}"
        
        obj = MetaverseObject(
            id=object_id,
            name=object_info["name"],
            object_type=object_info["object_type"],
            position=tuple(object_info.get("position", [0, 0, 0])),
            rotation=tuple(object_info.get("rotation", [0, 0, 0])),
            scale=tuple(object_info.get("scale", [1, 1, 1])),
            mesh_data=object_info.get("mesh_data", {}),
            texture_data=object_info.get("texture_data", {}),
            physics_properties=object_info.get("physics_properties", {}),
            interactive=object_info.get("interactive", False),
            world_id=object_info["world_id"],
            created_at=time.time(),
            last_modified=time.time(),
            metadata=object_info.get("metadata", {})
        )
        
        async with self._lock:
            self.objects[object_id] = obj
        
        logger.info(f"Metaverse object created: {object_id} ({obj.name})")
        return object_id
    
    async def render_world(self, world_id: str, render_engine: str = "webgl") -> Dict[str, Any]:
        """Renderizar mundo del metaverso"""
        if world_id not in self.worlds:
            raise ValueError(f"Metaverse world {world_id} not found")
        
        world = self.worlds[world_id]
        return await self.renderer.render_world(world, render_engine)
    
    async def simulate_physics(self, world_id: str, physics_engine: str = "bullet") -> Dict[str, Any]:
        """Simular física del metaverso"""
        if world_id not in self.worlds:
            raise ValueError(f"Metaverse world {world_id} not found")
        
        world = self.worlds[world_id]
        return await self.physics.simulate_physics(world, physics_engine)
    
    async def process_audio(self, world_id: str, audio_engine: str = "webaudio") -> Dict[str, Any]:
        """Procesar audio del metaverso"""
        if world_id not in self.worlds:
            raise ValueError(f"Metaverse world {world_id} not found")
        
        world = self.worlds[world_id]
        return await self.audio.process_audio(world, audio_engine)
    
    async def get_world_info(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información del mundo del metaverso"""
        if world_id not in self.worlds:
            return None
        
        world = self.worlds[world_id]
        return {
            "id": world.id,
            "name": world.name,
            "description": world.description,
            "metaverse_type": world.metaverse_type.value,
            "environment_type": world.environment_type.value,
            "max_users": world.max_users,
            "current_users": world.current_users,
            "world_size": world.world_size,
            "spawn_point": world.spawn_point,
            "physics_enabled": world.physics_enabled,
            "gravity": world.gravity,
            "created_at": world.created_at,
            "last_modified": world.last_modified
        }
    
    async def get_avatar_info(self, avatar_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información del avatar"""
        if avatar_id not in self.avatars:
            return None
        
        avatar = self.avatars[avatar_id]
        return {
            "id": avatar.id,
            "user_id": avatar.user_id,
            "name": avatar.name,
            "avatar_type": avatar.avatar_type.value,
            "position": avatar.position,
            "rotation": avatar.rotation,
            "scale": avatar.scale,
            "world_id": avatar.world_id,
            "created_at": avatar.created_at,
            "last_updated": avatar.last_updated
        }
    
    async def get_object_info(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información del objeto"""
        if object_id not in self.objects:
            return None
        
        obj = self.objects[object_id]
        return {
            "id": obj.id,
            "name": obj.name,
            "object_type": obj.object_type,
            "position": obj.position,
            "rotation": obj.rotation,
            "scale": obj.scale,
            "interactive": obj.interactive,
            "world_id": obj.world_id,
            "created_at": obj.created_at,
            "last_modified": obj.last_modified
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "worlds": {
                "total": len(self.worlds),
                "by_type": {
                    metaverse_type.value: sum(1 for w in self.worlds.values() if w.metaverse_type == metaverse_type)
                    for metaverse_type in MetaverseType
                },
                "by_environment": {
                    env_type.value: sum(1 for w in self.worlds.values() if w.environment_type == env_type)
                    for env_type in EnvironmentType
                }
            },
            "avatars": {
                "total": len(self.avatars),
                "by_type": {
                    avatar_type.value: sum(1 for a in self.avatars.values() if a.avatar_type == avatar_type)
                    for avatar_type in AvatarType
                }
            },
            "objects": len(self.objects),
            "events": len(self.events)
        }


# Instancia global del motor del metaverso
metaverse_engine = MetaverseEngine()


# Router para endpoints del motor del metaverso
metaverse_router = APIRouter()


@metaverse_router.post("/metaverse/worlds")
async def create_metaverse_world_endpoint(world_data: dict):
    """Crear mundo del metaverso"""
    try:
        world_id = await metaverse_engine.create_metaverse_world(world_data)
        
        return {
            "message": "Metaverse world created successfully",
            "world_id": world_id,
            "name": world_data["name"],
            "metaverse_type": world_data["metaverse_type"],
            "environment_type": world_data["environment_type"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metaverse type or environment type: {e}")
    except Exception as e:
        logger.error(f"Error creating metaverse world: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create metaverse world: {str(e)}")


@metaverse_router.get("/metaverse/worlds")
async def get_metaverse_worlds_endpoint():
    """Obtener mundos del metaverso"""
    try:
        worlds = metaverse_engine.worlds
        return {
            "worlds": [
                {
                    "id": world.id,
                    "name": world.name,
                    "description": world.description,
                    "metaverse_type": world.metaverse_type.value,
                    "environment_type": world.environment_type.value,
                    "max_users": world.max_users,
                    "current_users": world.current_users,
                    "created_at": world.created_at
                }
                for world in worlds.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting metaverse worlds: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse worlds: {str(e)}")


@metaverse_router.get("/metaverse/worlds/{world_id}")
async def get_metaverse_world_endpoint(world_id: str):
    """Obtener mundo del metaverso específico"""
    try:
        info = await metaverse_engine.get_world_info(world_id)
        
        if info:
            return info
        else:
            raise HTTPException(status_code=404, detail="Metaverse world not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metaverse world: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse world: {str(e)}")


@metaverse_router.post("/metaverse/avatars")
async def create_avatar_endpoint(avatar_data: dict):
    """Crear avatar del metaverso"""
    try:
        avatar_id = await metaverse_engine.create_avatar(avatar_data)
        
        return {
            "message": "Metaverse avatar created successfully",
            "avatar_id": avatar_id,
            "name": avatar_data["name"],
            "avatar_type": avatar_data["avatar_type"],
            "world_id": avatar_data["world_id"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid avatar type: {e}")
    except Exception as e:
        logger.error(f"Error creating metaverse avatar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create metaverse avatar: {str(e)}")


@metaverse_router.get("/metaverse/avatars/{avatar_id}")
async def get_avatar_endpoint(avatar_id: str):
    """Obtener avatar específico"""
    try:
        info = await metaverse_engine.get_avatar_info(avatar_id)
        
        if info:
            return info
        else:
            raise HTTPException(status_code=404, detail="Metaverse avatar not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metaverse avatar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse avatar: {str(e)}")


@metaverse_router.post("/metaverse/objects")
async def create_metaverse_object_endpoint(object_data: dict):
    """Crear objeto del metaverso"""
    try:
        object_id = await metaverse_engine.create_metaverse_object(object_data)
        
        return {
            "message": "Metaverse object created successfully",
            "object_id": object_id,
            "name": object_data["name"],
            "object_type": object_data["object_type"],
            "world_id": object_data["world_id"]
        }
        
    except Exception as e:
        logger.error(f"Error creating metaverse object: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create metaverse object: {str(e)}")


@metaverse_router.get("/metaverse/objects/{object_id}")
async def get_metaverse_object_endpoint(object_id: str):
    """Obtener objeto específico"""
    try:
        info = await metaverse_engine.get_object_info(object_id)
        
        if info:
            return info
        else:
            raise HTTPException(status_code=404, detail="Metaverse object not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metaverse object: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse object: {str(e)}")


@metaverse_router.post("/metaverse/worlds/{world_id}/render")
async def render_world_endpoint(world_id: str, render_data: dict):
    """Renderizar mundo del metaverso"""
    try:
        render_engine = render_data.get("render_engine", "webgl")
        result = await metaverse_engine.render_world(world_id, render_engine)
        
        return {
            "message": "World rendered successfully",
            "world_id": world_id,
            "render_result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error rendering metaverse world: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render metaverse world: {str(e)}")


@metaverse_router.post("/metaverse/worlds/{world_id}/physics")
async def simulate_physics_endpoint(world_id: str, physics_data: dict):
    """Simular física del metaverso"""
    try:
        physics_engine = physics_data.get("physics_engine", "bullet")
        result = await metaverse_engine.simulate_physics(world_id, physics_engine)
        
        return {
            "message": "Physics simulated successfully",
            "world_id": world_id,
            "physics_result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error simulating metaverse physics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate metaverse physics: {str(e)}")


@metaverse_router.post("/metaverse/worlds/{world_id}/audio")
async def process_audio_endpoint(world_id: str, audio_data: dict):
    """Procesar audio del metaverso"""
    try:
        audio_engine = audio_data.get("audio_engine", "webaudio")
        result = await metaverse_engine.process_audio(world_id, audio_engine)
        
        return {
            "message": "Audio processed successfully",
            "world_id": world_id,
            "audio_result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing metaverse audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process metaverse audio: {str(e)}")


@metaverse_router.get("/metaverse/stats")
async def get_metaverse_stats_endpoint():
    """Obtener estadísticas del motor del metaverso"""
    try:
        stats = await metaverse_engine.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting metaverse stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse stats: {str(e)}")


# Funciones de utilidad para integración
async def start_metaverse_engine():
    """Iniciar motor del metaverso"""
    await metaverse_engine.start()


async def stop_metaverse_engine():
    """Detener motor del metaverso"""
    await metaverse_engine.stop()


async def create_metaverse_world(world_info: Dict[str, Any]) -> str:
    """Crear mundo del metaverso"""
    return await metaverse_engine.create_metaverse_world(world_info)


async def create_avatar(avatar_info: Dict[str, Any]) -> str:
    """Crear avatar del metaverso"""
    return await metaverse_engine.create_avatar(avatar_info)


async def get_metaverse_engine_stats() -> Dict[str, Any]:
    """Obtener estadísticas del motor del metaverso"""
    return await metaverse_engine.get_system_stats()


logger.info("Metaverse engine module loaded successfully")

