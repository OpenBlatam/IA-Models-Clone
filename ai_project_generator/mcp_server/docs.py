"""
MCP API Documentation - Generación automática de documentación
===============================================================
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)


class APIDocumentation:
    """
    Generador de documentación automática para MCP
    
    Genera documentación OpenAPI/Swagger y ReDoc automáticamente.
    """
    
    def __init__(self, app: Any):
        """
        Args:
            app: Aplicación FastAPI
        """
        self.app = app
        self.router = APIRouter(prefix="/mcp/v1/docs", tags=["documentation"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Configura rutas de documentación"""
        
        @self.router.get("/")
        async def docs_index() -> HTMLResponse:
            """Página principal de documentación"""
            html = self._generate_docs_html()
            return HTMLResponse(content=html)
        
        @self.router.get("/openapi.json")
        async def openapi_json() -> Dict[str, Any]:
            """OpenAPI schema en JSON"""
            return self.app.openapi()
        
        @self.router.get("/swagger")
        async def swagger_ui() -> HTMLResponse:
            """Swagger UI"""
            from fastapi.openapi.docs import get_swagger_ui_html
            return get_swagger_ui_html(
                openapi_url="/mcp/v1/docs/openapi.json",
                title="MCP Server API Documentation",
            )
        
        @self.router.get("/redoc")
        async def redoc_ui() -> HTMLResponse:
            """ReDoc UI"""
            from fastapi.openapi.docs import get_redoc_html
            return get_redoc_html(
                openapi_url="/mcp/v1/docs/openapi.json",
                title="MCP Server API Documentation",
            )
        
        @self.router.get("/resources")
        async def resources_docs() -> Dict[str, Any]:
            """Documentación de recursos disponibles"""
            # Implementar con acceso a manifest registry
            return {
                "resources": [],
                "message": "Resource documentation - implement with manifest registry",
            }
    
    def _generate_docs_html(self) -> str:
        """Genera HTML de documentación"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Server API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .doc-link { display: block; margin: 20px 0; padding: 15px; 
                   background: #f5f5f5; border-radius: 5px; text-decoration: none; 
                   color: #2196F3; }
        .doc-link:hover { background: #e0e0e0; }
    </style>
</head>
<body>
    <h1>MCP Server API Documentation</h1>
    <h2>Available Documentation</h2>
    <a href="/mcp/v1/docs/swagger" class="doc-link">Swagger UI</a>
    <a href="/mcp/v1/docs/redoc" class="doc-link">ReDoc</a>
    <a href="/mcp/v1/docs/openapi.json" class="doc-link">OpenAPI JSON Schema</a>
    <a href="/mcp/v1/docs/resources" class="doc-link">Resources Documentation</a>
</body>
</html>
        """
    
    def get_router(self) -> APIRouter:
        """Retorna el router de documentación"""
        return self.router
    
    def add_resource_documentation(
        self,
        resource_id: str,
        description: str,
        examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Agrega documentación para un recurso
        
        Args:
            resource_id: ID del recurso
            description: Descripción del recurso
            examples: Ejemplos de uso (opcional)
        """
        # Implementar almacenamiento de documentación
        logger.info(f"Added documentation for resource {resource_id}")

