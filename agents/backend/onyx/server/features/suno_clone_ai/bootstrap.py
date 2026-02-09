"""
Bootstrap Module
Inicializa la aplicación con arquitectura modular
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI

from config.settings import settings
from modules.registry import get_module_registry
from modules.base import ModuleConfig
from modules.music_generation import MusicGenerationModule
from modules.audio_processing import AudioProcessingModule
from modules.search import SearchModule

logger = logging.getLogger(__name__)


def bootstrap_application(app: FastAPI) -> Dict[str, Any]:
    """
    Inicializa la aplicación con todos los módulos
    
    Args:
        app: Aplicación FastAPI
        
    Returns:
        Diccionario con módulos inicializados
    """
    registry = get_module_registry()
    
    # Registrar módulos
    registry.register(
        MusicGenerationModule,
        ModuleConfig(
            name="music_generation",
            version="1.0.0",
            enabled=True,
            dependencies=[],
            config=settings.__dict__
        )
    )
    
    registry.register(
        AudioProcessingModule,
        ModuleConfig(
            name="audio_processing",
            version="1.0.0",
            enabled=True,
            dependencies=[],
            config=settings.__dict__
        )
    )
    
    registry.register(
        SearchModule,
        ModuleConfig(
            name="search",
            version="1.0.0",
            enabled=True,
            dependencies=[],
            config=settings.__dict__
        )
    )
    
    logger.info("Modules registered successfully")
    
    return {
        "registry": registry,
        "modules": registry.get_all_modules()
    }


async def initialize_modules(registry) -> bool:
    """Inicializa todos los módulos registrados"""
    try:
        await registry.initialize_all()
        logger.info("All modules initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize modules: {e}", exc_info=True)
        return False


async def shutdown_modules(registry):
    """Cierra todos los módulos"""
    try:
        await registry.shutdown_all()
        logger.info("All modules shut down")
    except Exception as e:
        logger.error(f"Error shutting down modules: {e}", exc_info=True)















