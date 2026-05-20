"""
Route configuration for MCP Server
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any

from fastapi import FastAPI, Depends, Request, status, Path, Body
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..connectors import ConnectorRegistry
from ..manifests import ManifestRegistry
from ..security import MCPSecurityManager
from ..observability import MCPObservability
from ..services import ResourceService, OperationService
from ..handlers import (
    list_resources,
    get_resource,
    query_resource,
    health_check,
    get_metrics,
    get_stats,
    get_server_info,
    get_version,
    get_prometheus_metrics
)
from ..models import MCPRequest
from ..middleware import create_cors_middleware

logger = logging.getLogger(__name__)


def create_router(
    connector_registry: ConnectorRegistry,
    manifest_registry: ManifestRegistry,
    security_manager: MCPSecurityManager,
    observability: Optional[MCPObservability] = None
) -> FastAPI:
    """
    Create and configure FastAPI router with all MCP routes.
    
    Args:
        connector_registry: Registry of connectors (debe ser no None)
        manifest_registry: Registry of resource manifests (debe ser no None)
        security_manager: Security manager (debe ser no None)
        observability: Observability manager (opcional)
        
    Returns:
        Configured FastAPI application
        
    Raises:
        ValueError: Si algún parámetro requerido es None o inválido
        RuntimeError: Si hay error al configurar la aplicación
    """
    # Validar parámetros requeridos
    if connector_registry is None:
        raise ValueError("connector_registry cannot be None")
    if manifest_registry is None:
        raise ValueError("manifest_registry cannot be None")
    if security_manager is None:
        raise ValueError("security_manager cannot be None")
    
    try:
        # Tags metadata para OpenAPI
        tags_metadata = [
            {
                "name": "Resources",
                "description": "Operations for managing and querying MCP resources. Resources represent data sources like filesystems, databases, or APIs.",
                "externalDocs": {
                    "description": "MCP Resources Documentation",
                    "url": "https://docs.example.com/mcp/resources"
                }
            },
            {
                "name": "Operations",
                "description": "Execute operations (read, write, query, etc.) on MCP resources. Operations are executed through connectors.",
            },
            {
                "name": "System",
                "description": "System endpoints for health checks, metrics, statistics, and server information.",
            }
        ]
        
        app = FastAPI(
            title="MCP Server",
            description="""
            Model Context Protocol Server - Provides standardized access to resources.
            
            ## Overview
            
            The MCP Server provides a unified interface for accessing various data sources
            (filesystems, databases, APIs) through standardized connectors.
            
            ## Authentication
            
            All endpoints require JWT Bearer token authentication. Include the token in the
            Authorization header: `Authorization: Bearer <token>`
            
            ## Resources
            
            Resources are data sources registered in the MCP server. Each resource has:
            - A unique resource ID
            - A connector type (filesystem, database, api)
            - Supported operations
            - Access permissions
            
            ## Operations
            
            Operations are actions performed on resources:
            - **read**: Read data from a resource
            - **write**: Write data to a resource
            - **query**: Query/search a resource
            - **list**: List items in a resource
            
            ## Features
            
            - 🔒 **Security**: JWT authentication and scope-based authorization
            - 📊 **Observability**: Metrics, tracing, and structured logging
            - 🚀 **Performance**: Caching, rate limiting, and connection pooling
            - 🔧 **Flexibility**: Pluggable connectors and extensible architecture
            """,
            version="1.0.0",
            docs_url="/mcp/v1/docs",
            redoc_url="/mcp/v1/redoc",
            openapi_url="/mcp/v1/openapi.json",
            openapi_tags=tags_metadata,
            contact={
                "name": "MCP Server Support",
                "email": "support@example.com",
            },
            license_info={
                "name": "Proprietary",
            },
        )
        
        # Personalizar schema OpenAPI
        from ..utils.openapi_helpers import customize_openapi_schema
        customize_openapi_schema(
            app,
            servers=[
                {
                    "url": "http://localhost:8020",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            tags_metadata=tags_metadata
        )
    except Exception as e:
        logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create FastAPI application: {e}") from e
    
    # Setup exception handlers and middleware
    _setup_exception_handlers(app, observability)
    _setup_middleware(app)
    
    # Setup CORS (configurable via environment variables)
    import os
    cors_origins_str = os.getenv("MCP_CORS_ORIGINS", "*")
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
    if cors_origins:
        create_cors_middleware(
            app,
            allow_origins=cors_origins,
            allow_credentials=os.getenv("MCP_CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
            max_age=int(os.getenv("MCP_CORS_MAX_AGE", "3600"))
        )
    
    # Create services
    resource_service = ResourceService(
        connector_registry=connector_registry,
        manifest_registry=manifest_registry,
        security_manager=security_manager
    )
    
    operation_service = OperationService(
        resource_service=resource_service,
        observability=observability
    )
    
    # Create auth dependency with security manager
    security_bearer = HTTPBearer()
    
    async def get_current_user_dependency(
        credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
    ) -> Dict[str, Any]:
        """
        Dependency to get current user with security manager.
        
        Args:
            credentials: HTTP Bearer credentials from request
            
        Returns:
            User dictionary with authentication information
            
        Raises:
            HTTPException: Si el token es inválido o expirado
        """
        from fastapi import HTTPException, status
        
        if not credentials or not credentials.credentials:
            logger.warning("Missing or empty credentials")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify token with security manager
        try:
            user = await security_manager.verify_token(credentials.credentials)
            if not user or not isinstance(user, dict):
                logger.warning("Invalid user data from token verification")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error verifying token: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Dependency injection helpers
    def get_resource_service() -> ResourceService:
        return resource_service
    
    def get_operation_service() -> OperationService:
        return operation_service
    
    def get_manifest_registry() -> ManifestRegistry:
        return manifest_registry
    
    def get_connector_registry() -> ConnectorRegistry:
        return connector_registry
    
    def get_observability() -> Optional[MCPObservability]:
        return observability
    
    def get_security_manager() -> MCPSecurityManager:
        return security_manager
    
    # Setup routes
    setup_routes(
        app,
        get_resource_service=get_resource_service,
        get_operation_service=get_operation_service,
        get_manifest_registry=get_manifest_registry,
        get_connector_registry=get_connector_registry,
        get_observability=get_observability
    )
    
    return app


def setup_routes(
    app: FastAPI,
    get_resource_service: Callable[[], ResourceService],
    get_operation_service: Callable[[], OperationService],
    get_manifest_registry: Callable[[], ManifestRegistry],
    get_connector_registry: Callable[[], ConnectorRegistry],
    get_observability: Callable[[], Optional[MCPObservability]]
) -> None:
    """
    Setup all MCP routes on the FastAPI app
    
    Args:
        app: FastAPI application instance
        get_resource_service: Dependency function for resource service
        get_operation_service: Dependency function for operation service
        get_manifest_registry: Dependency function for manifest registry
        get_connector_registry: Dependency function for connector registry
        get_observability: Dependency function for observability
    """
    # Resources endpoints
    @app.get(
        "/mcp/v1/resources",
        tags=["Resources"],
        summary="List available resources",
        description="List all resources available to the authenticated user",
        response_description="List of available resources",
        responses={
            200: {
                "description": "List of resources",
                "content": {
                    "application/json": {
                        "example": [
                            {
                                "resource_id": "filesystem:/data",
                                "name": "Data Filesystem",
                                "connector_type": "filesystem",
                                "supported_operations": ["read", "list"]
                            }
                        ]
                    }
                }
            },
            401: {"description": "Unauthorized - Invalid or missing token"},
            500: {"description": "Internal server error"}
        }
    )
    async def list_resources_endpoint(
        user: dict = Depends(get_current_user_dependency),
        resource_service: ResourceService = Depends(get_resource_service),
        observability: Optional[MCPObservability] = Depends(get_observability)
    ):
        """
        List all resources available to the authenticated user.
        
        Returns a list of resource manifests that the user has access to,
        including resource metadata and supported operations.
        """
        return await list_resources(user, resource_service, observability)
    
    @app.get(
        "/mcp/v1/resources/{resource_id}",
        tags=["Resources"],
        summary="Get resource information",
        description="Get detailed information about a specific resource",
        response_description="Resource manifest",
        responses={
            200: {
                "description": "Resource information",
                "content": {
                    "application/json": {
                        "example": {
                            "resource_id": "filesystem:/data",
                            "name": "Data Filesystem",
                            "connector_type": "filesystem",
                            "description": "Access to data filesystem",
                            "supported_operations": ["read", "list", "write"]
                        }
                    }
                }
            },
            401: {"description": "Unauthorized - Invalid or missing token"},
            403: {"description": "Forbidden - User doesn't have access to this resource"},
            404: {"description": "Resource not found"},
            500: {"description": "Internal server error"}
        }
    )
    async def get_resource_endpoint(
        resource_id: str = Path(..., description="ID of the resource to retrieve", example="filesystem:/data"),
        user: dict = Depends(get_current_user_dependency),
        resource_service: ResourceService = Depends(get_resource_service)
    ):
        """
        Get detailed information about a specific resource.
        
        Returns the complete manifest for the specified resource,
        including metadata, supported operations, and access permissions.
        """
        return await get_resource(resource_id, user, resource_service)
    
    # Operations endpoint
    @app.post(
        "/mcp/v1/resources/{resource_id}/query",
        tags=["Operations"],
        summary="Execute operation on resource",
        description="Execute an operation (read, write, query, etc.) on a specific resource",
        response_description="Operation result",
        responses={
            200: {
                "description": "Operation successful",
                "content": {
                    "application/json": {
                        "example": {
                            "success": True,
                            "data": {"content": "file contents"},
                            "metadata": {
                                "resource_id": "filesystem:/data",
                                "operation": "read",
                                "connector_type": "filesystem"
                            },
                            "timestamp": "2024-01-01T00:00:00"
                        }
                    }
                }
            },
            400: {"description": "Bad request - Invalid parameters or request format"},
            401: {"description": "Unauthorized - Invalid or missing token"},
            403: {"description": "Forbidden - User doesn't have permission for this operation"},
            404: {"description": "Resource not found"},
            422: {"description": "Validation error - Invalid request data"},
            500: {"description": "Internal server error"}
        }
    )
    async def query_resource_endpoint(
        resource_id: str = Path(..., description="ID of the resource", example="filesystem:/data"),
        request: MCPRequest = Body(..., description="Operation request with operation and parameters"),
        user: dict = Depends(get_current_user_dependency),
        operation_service: OperationService = Depends(get_operation_service),
        manifest_registry: ManifestRegistry = Depends(get_manifest_registry)
    ):
        """
        Execute an operation on a resource.
        
        Performs the specified operation (read, write, query, list, etc.) on the resource
        with the provided parameters. The operation must be supported by the resource's connector.
        """
        return await query_resource(resource_id, request, user, operation_service, manifest_registry)
    
    # Health check
    @app.get(
        "/mcp/v1/health",
        tags=["System"],
        summary="Health check",
        description="Check the health status of the MCP server and its components",
        response_description="Health status information",
        responses={
            200: {
                "description": "Health status",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "timestamp": "2024-01-01T00:00:00",
                            "resources_count": 5,
                            "connectors_count": 3,
                            "connector_health": {
                                "filesystem": True,
                                "database": True,
                                "api": True
                            }
                        }
                    }
                }
            }
        }
    )
    async def health_check_endpoint(
        connector_registry: ConnectorRegistry = Depends(get_connector_registry),
        manifest_registry: ManifestRegistry = Depends(get_manifest_registry)
    ) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns the health status of the server, including:
        - Overall status (healthy, degraded, unhealthy)
        - Resource and connector counts
        - Individual connector health status
        """
        return await health_check(connector_registry, manifest_registry)
    
    # Metrics endpoint
    @app.get(
        "/mcp/v1/metrics",
        tags=["System"],
        summary="Get server metrics",
        description="Get current metrics and statistics from the MCP server",
        response_description="Server metrics",
        responses={
            200: {
                "description": "Metrics data",
                "content": {
                    "application/json": {
                        "example": {
                            "timestamp": "2024-01-01T00:00:00",
                            "server": {"version": "1.0.0", "status": "running"},
                            "observability": {"enabled": True},
                            "connectors": {"count": 3, "types": ["filesystem", "database", "api"]},
                            "resources": {"count": 5}
                        }
                    }
                }
            }
        }
    )
    async def metrics_endpoint(
        observability: Optional[MCPObservability] = Depends(get_observability),
        connector_registry: ConnectorRegistry = Depends(get_connector_registry),
        manifest_registry: ManifestRegistry = Depends(get_manifest_registry)
    ) -> Dict[str, Any]:
        """
        Get server metrics.
        
        Returns current metrics including:
        - Server status and version
        - Observability metrics (if enabled)
        - Connector and resource counts
        - System performance metrics
        """
        return await get_metrics(observability, connector_registry, manifest_registry)
    
    # Stats endpoint
    @app.get(
        "/mcp/v1/stats",
        tags=["System"],
        summary="Get detailed statistics",
        description="Get detailed statistics and performance data from the server",
        response_description="Detailed statistics",
        responses={
            200: {
                "description": "Statistics data",
                "content": {
                    "application/json": {
                        "example": {
                            "timestamp": "2024-01-01T00:00:00",
                            "summary": {},
                            "observability": {"enabled": True, "metrics_data": {}}
                        }
                    }
                }
            }
        }
    )
    async def stats_endpoint(
        resource_service: ResourceService = Depends(get_resource_service),
        observability: Optional[MCPObservability] = Depends(get_observability)
    ) -> Dict[str, Any]:
        """
        Get detailed server statistics.
        
        Returns comprehensive statistics including:
        - Observability statistics (tracing, metrics)
        - Resource usage statistics
        - Performance metrics
        - Historical data (if available)
        """
        return await get_stats(resource_service, observability)
    
    # Info endpoint
    @app.get(
        "/mcp/v1/info",
        tags=["System"],
        summary="Get server information",
        description="Get server information, capabilities, and supported features",
        response_description="Server information",
        responses={
            200: {
                "description": "Server information",
                "content": {
                    "application/json": {
                        "example": {
                            "server": {
                                "name": "MCP Server",
                                "version": "1.0.0",
                                "protocol": "MCP v1",
                                "timestamp": "2024-01-01T00:00:00"
                            },
                            "capabilities": {
                                "connectors": ["filesystem", "database", "api"],
                                "operations": ["read", "write", "query", "list"]
                            },
                            "resources": {"total": 5, "by_type": {"filesystem": 3, "database": 2}}
                        }
                    }
                }
            }
        }
    )
    async def info_endpoint(
        connector_registry: ConnectorRegistry = Depends(get_connector_registry),
        manifest_registry: ManifestRegistry = Depends(get_manifest_registry),
        security_manager: MCPSecurityManager = Depends(get_security_manager)
    ) -> Dict[str, Any]:
        """
        Get server information and capabilities.
        
        Returns comprehensive information about the server including:
        - Server version and protocol
        - Available connectors and their types
        - Supported operations
        - Resource statistics
        - Security configuration
        """
        return await get_server_info(connector_registry, manifest_registry, security_manager)
    
    # Version endpoint
    @app.get(
        "/mcp/v1/version",
        tags=["System"],
        summary="Get server version",
        description="Get version information for the MCP server",
        response_description="Version information",
        responses={
            200: {
                "description": "Version information",
                "content": {
                    "application/json": {
                        "example": {
                            "version": "1.0.0",
                            "author": "Blatam Academy",
                            "license": "Proprietary",
                            "protocol": "MCP v1",
                            "api_version": "v1"
                        }
                    }
                }
            }
        }
    )
    async def version_endpoint() -> Dict[str, Any]:
        """
        Get server version information.
        
        Returns version, author, license, and protocol information
        for the MCP server.
        """
        return await get_version()
    
    # Prometheus metrics endpoint
    @app.get(
        "/mcp/v1/metrics/prometheus",
        tags=["System"],
        summary="Prometheus metrics",
        description="Get metrics in Prometheus format for scraping",
        response_description="Metrics in Prometheus format",
        responses={
            200: {
                "description": "Prometheus metrics",
                "content": {
                    "text/plain": {
                        "example": "# TYPE mcp_requests_total counter\nmcp_requests_total{resource_id=\"test\",operation=\"read\"} 42\n"
                    }
                }
            },
            500: {"description": "Error generating metrics"}
        }
    )
    async def prometheus_metrics_endpoint(
        observability: Optional[MCPObservability] = Depends(get_observability)
    ) -> Response:
        """
        Get metrics in Prometheus format.
        
        This endpoint exposes metrics in Prometheus format for scraping.
        Compatible with Prometheus server and other monitoring tools.
        """
        if not observability:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(
                content="# Observability not enabled\n",
                status_code=503,
                media_type="text/plain"
            )
        return await get_prometheus_metrics(observability)


def _setup_exception_handlers(app: FastAPI, observability: Optional[MCPObservability]) -> None:
    """
    Setup global exception handlers for FastAPI application.
    
    Args:
        app: FastAPI application instance
        observability: Observability manager (opcional)
    """
    if app is None:
        raise ValueError("app cannot be None")
    
    @app.exception_handler(status.HTTP_401_UNAUTHORIZED)
    async def unauthorized_handler(request: Request, exc: Exception):
        """Handle unauthorized errors with detailed logging"""
        path = str(request.url.path) if request else "unknown"
        method = request.method if request else "unknown"
        
        logger.warning(
            f"Unauthorized access attempt: {method} {path}",
            extra={"path": path, "method": method}
        )
        
        if observability:
            try:
                observability.record_error("unauthorized", path=path, method=method)
            except Exception as e:
                logger.warning(f"Failed to record observability error: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error": "Authentication required",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": path,
            }
        )
    
    @app.exception_handler(status.HTTP_403_FORBIDDEN)
    async def forbidden_handler(request: Request, exc: Exception):
        """Handle forbidden errors with detailed logging"""
        path = str(request.url.path) if request else "unknown"
        method = request.method if request else "unknown"
        
        logger.warning(
            f"Forbidden access attempt: {method} {path}",
            extra={"path": path, "method": method}
        )
        
        if observability:
            try:
                observability.record_error("forbidden", path=path, method=method)
            except Exception as e:
                logger.warning(f"Failed to record observability error: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "error": "Access denied",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": path,
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed information"""
        path = str(request.url.path) if request else "unknown"
        method = request.method if request else "unknown"
        errors = exc.errors() if hasattr(exc, 'errors') else []
        
        logger.warning(
            f"Validation error: {method} {path} - {len(errors)} error(s)",
            extra={"path": path, "method": method, "error_count": len(errors)}
        )
        
        if observability:
            try:
                observability.record_error(
                    "validation_error",
                    errors=errors,
                    path=path,
                    method=method
                )
            except Exception as e:
                logger.warning(f"Failed to record observability error: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Validation error",
                "details": errors,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": path,
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions with comprehensive logging"""
        path = str(request.url.path) if request else "unknown"
        method = request.method if request else "unknown"
        error_type = type(exc).__name__
        error_message = str(exc)
        
        logger.error(
            f"Unhandled exception: {error_type} - {error_message}",
            exc_info=True,
            extra={
                "path": path,
                "method": method,
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        
        if observability:
            try:
                observability.record_error(
                    "unhandled_exception",
                    error=error_message,
                    error_type=error_type,
                    path=path,
                    method=method
                )
            except Exception as e:
                logger.warning(f"Failed to record observability error: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": path,
                "error_type": error_type,
            }
        )


def _setup_middleware(app: FastAPI) -> None:
    """
    Setup global middleware for FastAPI application.
    
    Configura middleware para:
    - Request ID tracking
    - Process time measurement
    - Security headers
    - Request logging
    
    Args:
        app: FastAPI application instance
        
    Raises:
        ValueError: Si app es None
    """
    if app is None:
        raise ValueError("app cannot be None")
    
    # Request ID middleware
    @app.middleware("http")
    async def add_request_id_header(request: Request, call_next):
        """
        Add request ID to response headers and request state.
        
        Genera un UUID único para cada request y lo agrega a:
        - Request state (para logging)
        - Response headers (X-Request-ID)
        """
        import uuid
        request_id = str(uuid.uuid4())
        try:
            # Store request ID in request state for logging
            request.state.request_id = request_id
            response = await call_next(request)
            try:
                response.headers["X-Request-ID"] = request_id
            except Exception as e:
                logger.warning(f"Failed to add request ID header: {e}")
            return response
        except Exception as e:
            logger.error(f"Error in request ID middleware: {e}", exc_info=True)
            raise
    
    # Process time middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """
        Add process time to response headers.
        
        Mide el tiempo de procesamiento de la request y lo agrega
        al header X-Process-Time en milisegundos.
        """
        import time
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            try:
                response.headers["X-Process-Time"] = f"{process_time:.4f}"
            except Exception as e:
                logger.warning(f"Failed to add process time header: {e}")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error in process time middleware: {e}",
                exc_info=True,
                extra={"duration_ms": process_time * 1000}
            )
            raise
    
    # Security headers middleware (usar el mejorado del módulo)
    from ..middleware.security import create_security_middleware
    create_security_middleware(app)

