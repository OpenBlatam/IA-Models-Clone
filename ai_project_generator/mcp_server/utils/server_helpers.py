"""
Server Helpers - Utilidades para inicialización y configuración del servidor
============================================================================

Funciones helper para facilitar la creación y configuración del servidor MCP.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..server import MCPServer
from ..connectors import ConnectorRegistry, FileSystemConnector, DatabaseConnector, APIConnector
from ..manifests import ManifestRegistry, ManifestLoader
from ..security import MCPSecurityManager
from ..observability import MCPObservability
from ..config import MCPSettings, load_settings

logger = logging.getLogger(__name__)


def create_server_from_config(
    config: Optional[MCPSettings] = None,
    config_file: Optional[str] = None,
    use_env: bool = True
) -> MCPServer:
    """
    Crear servidor MCP desde configuración.
    
    Args:
        config: Configuración MCPSettings (opcional)
        config_file: Ruta a archivo de configuración (opcional)
        use_env: Si usar variables de entorno (default: True)
    
    Returns:
        MCPServer configurado
    
    Raises:
        ValueError: Si la configuración es inválida
        RuntimeError: Si hay error al crear el servidor
    """
    # Cargar configuración
    if config is None:
        config = load_settings(file_path=config_file, use_env=use_env, validate=True)
    
    # Crear componentes
    connector_registry = ConnectorRegistry()
    manifest_registry = ManifestRegistry()
    
    # Cargar manifests si hay path configurado
    if config.manifests_path:
        loader = ManifestLoader()
        manifests = loader.load_from_directory(config.manifests_path)
        for manifest in manifests:
            manifest_registry.register(manifest)
    
    # Crear security manager
    security_manager = MCPSecurityManager(
        secret_key=config.security.secret_key,
        token_expire_minutes=config.security.token_expire_minutes,
        algorithm=config.security.algorithm
    )
    
    # Crear observability (opcional)
    observability = None
    if config.observability.enabled:
        observability = MCPObservability(
            tracing_enabled=config.observability.tracing_enabled,
            metrics_enabled=config.observability.metrics_enabled,
            otlp_endpoint=config.observability.otlp_endpoint
        )
    
    # Crear servidor
    server = MCPServer(
        connector_registry=connector_registry,
        manifest_registry=manifest_registry,
        security_manager=security_manager,
        observability=observability,
        enable_rate_limiting=config.rate_limiting.enabled,
        enable_caching=config.cache.enabled,
        enable_cors=config.cors.enabled,
        cors_origins=config.cors.allowed_origins if config.cors.enabled else None
    )
    
    logger.info("MCP Server created from configuration")
    return server


def create_default_connectors(
    connector_registry: ConnectorRegistry,
    filesystem_paths: Optional[List[str]] = None,
    database_urls: Optional[Dict[str, str]] = None,
    api_endpoints: Optional[Dict[str, str]] = None
) -> None:
    """
    Crear y registrar conectores por defecto.
    
    Args:
        connector_registry: Registry de conectores
        filesystem_paths: Lista de paths para filesystem connectors (opcional)
        database_urls: Diccionario de {name: url} para database connectors (opcional)
        api_endpoints: Diccionario de {name: endpoint} para API connectors (opcional)
    """
    # Filesystem connectors
    if filesystem_paths:
        for path in filesystem_paths:
            try:
                connector = FileSystemConnector(base_path=path)
                connector_registry.register(connector)
                logger.info(f"Registered filesystem connector: {path}")
            except Exception as e:
                logger.warning(f"Failed to register filesystem connector {path}: {e}")
    
    # Database connectors
    if database_urls:
        for name, url in database_urls.items():
            try:
                connector = DatabaseConnector(connection_string=url)
                connector_registry.register(connector, name=name)
                logger.info(f"Registered database connector: {name}")
            except Exception as e:
                logger.warning(f"Failed to register database connector {name}: {e}")
    
    # API connectors
    if api_endpoints:
        for name, endpoint in api_endpoints.items():
            try:
                connector = APIConnector(base_url=endpoint)
                connector_registry.register(connector, name=name)
                logger.info(f"Registered API connector: {name}")
            except Exception as e:
                logger.warning(f"Failed to register API connector {name}: {e}")


def setup_logging_from_config(config: MCPSettings) -> None:
    """
    Configurar logging desde configuración.
    
    Args:
        config: Configuración MCPSettings
    """
    from .logging_helpers import setup_logging
    
    setup_logging(
        level=config.observability.log_level,
        format_string=config.observability.log_format,
        include_timestamp=True,
        include_module=True,
        structured=False
    )
    
    logger.info(f"Logging configured: level={config.observability.log_level}")


def validate_server_config(config: MCPSettings) -> List[str]:
    """
    Validar configuración del servidor.
    
    Args:
        config: Configuración a validar
    
    Returns:
        Lista de warnings (vacía si todo está bien)
    """
    warnings = []
    
    # Validar secret key
    if len(config.security.secret_key) < 32:
        warnings.append("Secret key is shorter than 32 characters - consider using a longer key")
    
    if config.security.secret_key == "change-me-in-production":
        warnings.append("Using default secret key - change in production!")
    
    # Validar HTTPS
    if not config.security.require_https and not config.server.debug:
        warnings.append("HTTPS not required in production - consider enabling require_https")
    
    # Validar manifests path
    if config.manifests_path:
        path = Path(config.manifests_path)
        if not path.exists():
            warnings.append(f"Manifests path does not exist: {config.manifests_path}")
        elif not path.is_dir():
            warnings.append(f"Manifests path is not a directory: {config.manifests_path}")
    
    # Validar observability
    if config.observability.enabled and config.observability.tracing_enabled:
        if not config.observability.otlp_endpoint:
            warnings.append("Tracing enabled but OTLP endpoint not configured")
    
    return warnings


def print_server_info(server: MCPServer) -> None:
    """
    Imprimir información del servidor.
    
    Args:
        server: Instancia de MCPServer
    """
    print("\n" + "=" * 60)
    print("MCP Server Information")
    print("=" * 60)
    print(f"Connectors: {len(server.connector_registry.list_connectors())}")
    print(f"Resources: {len(server.manifest_registry.list_resources())}")
    print(f"Rate Limiting: {'Enabled' if server.enable_rate_limiting else 'Disabled'}")
    print(f"Caching: {'Enabled' if server.enable_caching else 'Disabled'}")
    print(f"CORS: {'Enabled' if server.enable_cors else 'Disabled'}")
    print(f"Observability: {'Enabled' if server.observability else 'Disabled'}")
    print("=" * 60 + "\n")


def get_server_summary(server: MCPServer) -> Dict[str, Any]:
    """
    Obtener resumen del servidor.
    
    Args:
        server: Instancia de MCPServer
    
    Returns:
        Diccionario con resumen del servidor
    """
    connectors = server.connector_registry.list_connectors()
    resources = server.manifest_registry.list_resources()
    
    return {
        "connectors": {
            "count": len(connectors),
            "types": list(set(c.get_connector_type() for c in connectors))
        },
        "resources": {
            "count": len(resources),
            "by_type": {}
        },
        "features": {
            "rate_limiting": server.enable_rate_limiting,
            "caching": server.enable_caching,
            "cors": server.enable_cors,
            "observability": server.observability is not None
        }
    }

