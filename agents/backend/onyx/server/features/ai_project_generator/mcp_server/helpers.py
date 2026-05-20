"""
MCP Helpers - Funciones helper para configuración rápida
=========================================================

Funciones de utilidad para crear y configurar servidores MCP de forma rápida
y con valores por defecto sensatos.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .server import MCPServer
from .connectors import (
    ConnectorRegistry,
    FileSystemConnector,
    DatabaseConnector,
    APIConnector
)
from .manifests import ManifestRegistry, ManifestLoader
from .security import MCPSecurityManager
from .observability import MCPObservability
from .rate_limiter import RateLimiter
from .cache import MCPCache

logger = logging.getLogger(__name__)


def create_mcp_server(
    secret_key: str,
    manifests_path: Optional[str] = None,
    enable_rate_limiting: bool = True,
    enable_caching: bool = True,
    enable_cors: bool = True,
    cors_origins: Optional[List[str]] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    otlp_endpoint: Optional[str] = None,
) -> MCPServer:
    """
    Crea un servidor MCP con configuración por defecto.
    
    Args:
        secret_key: Clave secreta para JWT (requerida, mínimo 32 caracteres).
        manifests_path: Ruta a directorio con manifests (opcional).
        enable_rate_limiting: Habilitar rate limiting (default: True).
        enable_caching: Habilitar cache (default: True).
        enable_cors: Habilitar CORS (default: True).
        cors_origins: Orígenes permitidos para CORS (default: None = todos).
        enable_tracing: Habilitar tracing (default: True).
        enable_metrics: Habilitar métricas (default: True).
        otlp_endpoint: Endpoint OTLP para tracing (opcional).
    
    Returns:
        MCPServer configurado y listo para usar.
    
    Raises:
        ValueError: Si secret_key está vacío, es inválido, o los parámetros son inválidos.
        RuntimeError: Si hay error al crear el servidor.
    """
    # Validar secret_key
    if not secret_key or not isinstance(secret_key, str):
        raise ValueError("secret_key must be a non-empty string")
    secret_key = secret_key.strip()
    if not secret_key:
        raise ValueError("secret_key cannot be empty or whitespace only")
    if len(secret_key) < 32:
        logger.warning("secret_key is shorter than 32 characters, consider using a longer key for security")
    
    # Validar parámetros booleanos
    if not isinstance(enable_rate_limiting, bool):
        raise ValueError("enable_rate_limiting must be a boolean")
    if not isinstance(enable_caching, bool):
        raise ValueError("enable_caching must be a boolean")
    if not isinstance(enable_cors, bool):
        raise ValueError("enable_cors must be a boolean")
    if not isinstance(enable_tracing, bool):
        raise ValueError("enable_tracing must be a boolean")
    if not isinstance(enable_metrics, bool):
        raise ValueError("enable_metrics must be a boolean")
    
    # Validar cors_origins
    if cors_origins is not None:
        if not isinstance(cors_origins, list):
            raise ValueError("cors_origins must be a list or None")
        for origin in cors_origins:
            if not isinstance(origin, str) or not origin.strip():
                raise ValueError("All cors_origins must be non-empty strings")
    
    # Validar otlp_endpoint
    if otlp_endpoint is not None:
        if not isinstance(otlp_endpoint, str) or not otlp_endpoint.strip():
            raise ValueError("otlp_endpoint must be a non-empty string or None")
    
    try:
        # Crear registries
        connector_registry = ConnectorRegistry()
        manifest_registry = ManifestRegistry()
        
        # Registrar conectores por defecto
        try:
            connector_registry.register("filesystem", FileSystemConnector())
            connector_registry.register("database", DatabaseConnector())
            connector_registry.register("api", APIConnector())
            logger.debug("Default connectors registered")
        except Exception as e:
            logger.warning(f"Error registering default connectors: {e}")
        
        # Cargar manifests si se proporciona path
        if manifests_path:
            if not isinstance(manifests_path, str):
                raise ValueError(f"manifests_path must be a string, got {type(manifests_path)}")
            
            manifests_path_obj = Path(manifests_path)
            if not manifests_path_obj.exists():
                logger.warning(f"Manifests path does not exist: {manifests_path}")
            elif not manifests_path_obj.is_dir():
                logger.warning(f"Manifests path is not a directory: {manifests_path}")
            else:
                try:
                    ManifestLoader.register_from_directory(
                        manifest_registry,
                        str(manifests_path_obj)
                    )
                    logger.info(f"Manifests loaded from {manifests_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not load manifests from {manifests_path}: {e}",
                        exc_info=True
                    )
        
        # Crear security manager
        try:
            security_manager = MCPSecurityManager(secret_key=secret_key.strip())
        except Exception as e:
            logger.error(f"Error creating security manager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create security manager: {e}") from e
        
        # Crear observabilidad
        try:
            observability = MCPObservability(
                enable_tracing=enable_tracing,
                enable_metrics=enable_metrics,
                otlp_endpoint=otlp_endpoint,
            )
        except Exception as e:
            logger.warning(f"Error creating observability: {e}", exc_info=True)
            observability = None
        
        # Crear servidor
        try:
            server = MCPServer(
                connector_registry=connector_registry,
                manifest_registry=manifest_registry,
                security_manager=security_manager,
                observability=observability,
                enable_rate_limiting=enable_rate_limiting,
                enable_caching=enable_caching,
                enable_cors=enable_cors,
                cors_origins=cors_origins,
            )
            logger.info("MCP Server created successfully")
            return server
        except Exception as e:
            logger.error(f"Error creating MCP server: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create MCP server: {e}") from e
    
    except Exception as e:
        logger.error(f"Unexpected error in create_mcp_server: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create MCP server: {e}") from e


def setup_rate_limits(
    rate_limiter: RateLimiter,
    default_per_minute: int = 60,
    default_per_hour: int = 1000,
    resource_limits: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """
    Configura límites de rate limiting.
    
    Args:
        rate_limiter: Instancia de RateLimiter.
        default_per_minute: Límite por defecto por minuto (default: 60).
        default_per_hour: Límite por defecto por hora (default: 1000).
        resource_limits: Diccionario con límites específicos por recurso.
            Ejemplo: {"resource1": {"per_minute": 30, "per_hour": 500}}
    
    Raises:
        ValueError: Si los parámetros son inválidos.
        TypeError: Si rate_limiter no es una instancia de RateLimiter.
    """
    if rate_limiter is None:
        raise ValueError("rate_limiter cannot be None")
    
    if not isinstance(rate_limiter, RateLimiter):
        raise TypeError(f"rate_limiter must be an instance of RateLimiter, got {type(rate_limiter)}")
    
    if default_per_minute < 1:
        raise ValueError("default_per_minute must be at least 1")
    
    if default_per_hour < default_per_minute:
        raise ValueError("default_per_hour must be >= default_per_minute")
    
    try:
        if resource_limits:
            for resource_id, limits in resource_limits.items():
                if not isinstance(resource_id, str) or not resource_id.strip():
                    logger.warning(f"Invalid resource_id: {resource_id}, skipping")
                    continue
                
                if not isinstance(limits, dict):
                    logger.warning(f"Invalid limits for {resource_id}, skipping")
                    continue
                
                per_minute = limits.get("per_minute", default_per_minute)
                per_hour = limits.get("per_hour", default_per_hour)
                
                if per_minute < 1 or per_hour < 1:
                    logger.warning(
                        f"Invalid limits for {resource_id}: "
                        f"per_minute={per_minute}, per_hour={per_hour}, skipping"
                    )
                    continue
                
                rate_limiter.set_limit(
                    resource_id,
                    requests_per_minute=per_minute,
                    requests_per_hour=per_hour,
                )
                logger.debug(f"Rate limit set for {resource_id}: {per_minute}/min, {per_hour}/hour")
        
        logger.info("Rate limits configured successfully")
    
    except Exception as e:
        logger.error(f"Error setting up rate limits: {e}", exc_info=True)
        raise RuntimeError(f"Failed to setup rate limits: {e}") from e


def setup_cache(
    cache: MCPCache,
    default_ttl: int = 300,
    resource_ttls: Optional[Dict[str, int]] = None,
) -> None:
    """
    Configura cache con TTLs personalizados.
    
    Args:
        cache: Instancia de MCPCache.
        default_ttl: TTL por defecto en segundos (default: 300).
        resource_ttls: Diccionario con TTLs específicos por recurso.
            Ejemplo: {"resource1": 600, "resource2": 1200}
    
    Raises:
        ValueError: Si los parámetros son inválidos.
        TypeError: Si cache no es una instancia de MCPCache.
    """
    if cache is None:
        raise ValueError("cache cannot be None")
    
    if not isinstance(cache, MCPCache):
        raise TypeError(f"cache must be an instance of MCPCache, got {type(cache)}")
    
    if default_ttl < 1:
        raise ValueError("default_ttl must be at least 1 second")
    
    try:
        cache.default_ttl = default_ttl
        logger.debug(f"Default TTL set to {default_ttl} seconds")
        
        if resource_ttls:
            for resource_id, ttl in resource_ttls.items():
                if not isinstance(resource_id, str) or not resource_id.strip():
                    logger.warning(f"Invalid resource_id: {resource_id}, skipping")
                    continue
                
                if not isinstance(ttl, int) or ttl < 1:
                    logger.warning(f"Invalid TTL for {resource_id}: {ttl}, skipping")
                    continue
                
                # Los TTLs por recurso se aplican en el manifest metadata
                # Esta función solo configura el default
                logger.debug(f"Resource TTL for {resource_id}: {ttl} seconds (applied via manifest)")
        
        logger.info("Cache configured successfully")
    
    except Exception as e:
        logger.error(f"Error setting up cache: {e}", exc_info=True)
        raise RuntimeError(f"Failed to setup cache: {e}") from e
