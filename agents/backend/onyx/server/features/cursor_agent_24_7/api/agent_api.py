"""
Agent API - REST API para control del agente
============================================

API REST para controlar el agente 24/7 con endpoints simples.

Este módulo ahora usa una arquitectura modular:
- api/models/: Modelos Pydantic
- api/routes/: Rutas HTTP organizadas por dominio
- api/templates/: Templates HTML
- api/app_config.py: Configuración de la aplicación
"""

import logging
from typing import Optional
from fastapi import FastAPI
import uvicorn

# Usar structlog para logging estructurado
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from ..core.agent import CursorAgent
from .app_config import create_app


class AgentAPI:
    """
    Clase wrapper para la API del agente.
    
    Proporciona una interfaz simple para crear y ejecutar
    el servidor API del agente.
    """
    
    def __init__(
        self,
        agent: Optional[CursorAgent] = None,
        host: str = "0.0.0.0",
        port: int = 8024
    ) -> None:
        """
        Inicializar API del agente.
        
        Args:
            agent: Instancia del agente. Si es None, se crea una nueva.
            host: Host donde escuchar (default: "0.0.0.0").
            port: Puerto donde escuchar (default: 8024).
        """
        self.agent = agent
        self.host = host
        self.port = port
        self.app: FastAPI = create_app(self.agent)
        logger.info(f"AgentAPI initialized on {host}:{port}")
    
    async def run(self) -> None:
        """
        Ejecutar servidor API.
        
        Inicia el servidor uvicorn con la aplicación FastAPI.
        """
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        logger.info(f"Starting API server on {self.host}:{self.port}")
        await server.serve()


# Función de conveniencia para compatibilidad hacia atrás
def create_app(agent: Optional[CursorAgent] = None) -> FastAPI:
    """
    Crear aplicación FastAPI (función de conveniencia).
    
    Args:
        agent: Instancia del agente. Si es None, se crea una nueva.
    
    Returns:
        Aplicación FastAPI configurada.
    """
    from .app_config import create_app as _create_app
    return _create_app(agent)
