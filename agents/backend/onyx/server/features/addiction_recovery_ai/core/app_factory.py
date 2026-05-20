"""
Application factory for creating and configuring FastAPI app
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config.app_config import get_config
from core.lifespan import lifespan
from core.middleware_config import setup_middleware
from core.routes_config import setup_routes
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        FastAPI: Configured application instance
    """
    config = get_config()
    
    setup_logging(
        level=config.log_level,
        format_string=config.log_format
    )
    
    app = FastAPI(
        title="Addiction Recovery AI API",
        description="Sistema de IA para ayudar a dejar adicciones (cigarrillos, alcohol, drogas y otras dependencias)",
        version=config.app_version,
        lifespan=lifespan
    )
    
    customize_openapi_schema(app)
    setup_cors(app, config)
    setup_middleware(app, config)
    setup_routes(app)
    setup_root_endpoint(app)
    
    return app


def customize_openapi_schema(app: FastAPI) -> None:
    """Customize OpenAPI schema"""
    try:
        from api.openapi_customization import customize_openapi_schema
        app.openapi = lambda: customize_openapi_schema(app)
    except ImportError:
        try:
            from .api.openapi_customization import customize_openapi_schema
            app.openapi = lambda: customize_openapi_schema(app)
        except ImportError:
            logger.debug("OpenAPI customization not available")


def setup_cors(app: FastAPI, config) -> None:
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_root_endpoint(app: FastAPI) -> None:
    """Setup root endpoint with auto-generated endpoint list"""
    
    @app.get("/")
    async def root():
        """Root endpoint with service information and auto-generated endpoint list"""
        from fastapi.openapi.utils import get_openapi
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        endpoints = {}
        paths = openapi_schema.get("paths", {})
        
        for path, methods in paths.items():
            if path == "/":
                continue
            
            path_parts = path.strip("/").split("/")
            if len(path_parts) > 0:
                category = path_parts[0] if path_parts[0] else "root"
                
                if category not in endpoints:
                    endpoints[category] = {}
                
                for method, details in methods.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        operation_id = details.get("operationId", "")
                        summary = details.get("summary", "")
                        
                        endpoint_key = operation_id or f"{method.lower()}_{path_parts[-1] if path_parts else 'root'}"
                        endpoints[category][endpoint_key] = {
                            "method": method.upper(),
                            "path": path,
                            "summary": summary
                        }
        
        return {
            "service": "Addiction Recovery AI",
            "version": app.version,
            "status": "running",
            "description": "Sistema de IA para ayudar a dejar adicciones",
            "endpoints": endpoints,
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }

