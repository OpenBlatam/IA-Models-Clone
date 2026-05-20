"""
MCP Server - Servidor principal del Model Context Protocol
==========================================================

Servidor minimal que expone conectores estandarizados para acceso
a recursos (archivos, DB, APIs) de forma segura y observable.

Refactorizado con arquitectura modular:
- models/: Modelos de request/response
- handlers/: Handlers de endpoints
- services/: Lógica de negocio
- routes/: Configuración de rutas
- dependencies/: Dependencias de FastAPI
"""

import logging
from typing import Optional, TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from fastapi import FastAPI

from .connectors import ConnectorRegistry
from .manifests import ManifestRegistry
from .security import MCPSecurityManager
from .observability import MCPObservability
from .routes import create_router

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Servidor MCP principal.
    
    Expone endpoints para:
    - Listar recursos disponibles
    - Consultar recursos específicos
    - Operaciones sobre recursos (read, write, query, etc.)
    
    Gestiona la aplicación FastAPI y todos los componentes necesarios
    para el funcionamiento del servidor MCP.
    """
    
    def __init__(
        self,
        connector_registry: ConnectorRegistry,
        manifest_registry: ManifestRegistry,
        security_manager: MCPSecurityManager,
        observability: Optional[MCPObservability] = None,
        enable_rate_limiting: bool = True,
        enable_caching: bool = True,
        enable_cors: bool = True,
        cors_origins: Optional[List[str]] = None,
    ) -> None:
        """
        Inicializar servidor MCP.
        
        Args:
            connector_registry: Registry de conectores (requerido).
            manifest_registry: Registry de manifests (requerido).
            security_manager: Gestor de seguridad (requerido).
            observability: Gestor de observabilidad (opcional).
            enable_rate_limiting: Habilitar rate limiting (default: True).
            enable_caching: Habilitar cache (default: True).
            enable_cors: Habilitar CORS (default: True).
            cors_origins: Orígenes permitidos para CORS (opcional).
        
        Raises:
            ValueError: Si algún parámetro requerido es None o inválido.
            RuntimeError: Si hay error al crear la aplicación FastAPI.
        """
        if connector_registry is None:
            raise ValueError("connector_registry cannot be None")
        if manifest_registry is None:
            raise ValueError("manifest_registry cannot be None")
        if security_manager is None:
            raise ValueError("security_manager cannot be None")
        
        self.connector_registry: ConnectorRegistry = connector_registry
        self.manifest_registry: ManifestRegistry = manifest_registry
        self.security_manager: MCPSecurityManager = security_manager
        self.observability: Optional[MCPObservability] = (
            observability or MCPObservability()
        )
        self.enable_rate_limiting: bool = enable_rate_limiting
        self.enable_caching: bool = enable_caching
        self.enable_cors: bool = enable_cors
        self.cors_origins: Optional[List[str]] = cors_origins
        
        # Crear aplicación FastAPI con rutas modulares
        try:
            self.app: "FastAPI" = create_router(
                connector_registry=self.connector_registry,
                manifest_registry=self.manifest_registry,
                security_manager=self.security_manager,
                observability=self.observability
            )
            logger.info("MCP Server initialized successfully")
        except Exception as e:
            logger.error(f"Error creating MCP server: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create MCP server: {e}") from e
    
    def get_app(self) -> "FastAPI":
        """
        Retorna la aplicación FastAPI.
        
        Returns:
            Instancia de FastAPI configurada con todas las rutas y middleware.
        
        Raises:
            RuntimeError: Si la aplicación no está inicializada.
        """
        if not hasattr(self, 'app') or self.app is None:
            raise RuntimeError("MCP Server application not initialized")
        return self.app
    
    def get_info(self) -> Dict[str, Any]:
        """
        Retorna información del servidor.
        
        Returns:
            Diccionario con información del servidor incluyendo:
            - Número de conectores registrados
            - Número de manifests registrados
            - Características habilitadas
            - Estado de observabilidad
        """
        info = {
            "rate_limiting_enabled": self.enable_rate_limiting,
            "caching_enabled": self.enable_caching,
            "cors_enabled": self.enable_cors,
            "observability_enabled": self.observability is not None,
        }
        
        try:
            # Obtener conteo de conectores
            if hasattr(self.connector_registry, 'count'):
                info["connectors_count"] = self.connector_registry.count()
            elif hasattr(self.connector_registry, 'list_connectors'):
                connectors = self.connector_registry.list_connectors()
                info["connectors_count"] = len(connectors) if connectors else 0
            else:
                info["connectors_count"] = 0
                logger.warning("ConnectorRegistry does not have count or list_connectors method")
        except Exception as e:
            logger.warning(f"Error getting connectors count: {e}")
            info["connectors_count"] = 0
            info["connectors_count_error"] = str(e)
        
        try:
            # Obtener conteo de manifests
            if hasattr(self.manifest_registry, 'count'):
                info["manifests_count"] = self.manifest_registry.count()
            elif hasattr(self.manifest_registry, 'list_resource_ids'):
                manifests = self.manifest_registry.list_resource_ids()
                info["manifests_count"] = len(manifests) if manifests else 0
            elif hasattr(self.manifest_registry, 'get_all'):
                manifests = self.manifest_registry.get_all()
                info["manifests_count"] = len(manifests) if manifests else 0
            else:
                info["manifests_count"] = 0
                logger.warning("ManifestRegistry does not have count, list_resource_ids, or get_all method")
        except Exception as e:
            logger.warning(f"Error getting manifests count: {e}")
            info["manifests_count"] = 0
            info["manifests_count_error"] = str(e)
        
        return info
