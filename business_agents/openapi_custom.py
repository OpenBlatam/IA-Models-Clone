"""Custom OpenAPI/Swagger documentation improvements."""
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def custom_openapi(app) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema
    
    from .settings import settings
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version="1.0.0",
        description="""
        Ultimate Quantum AI ML NLP Benchmark API
        
        ## Features
        - Modular architecture with plugin system
        - Real-time WebSocket support
        - Webhook event system
        - Batch processing capabilities
        - Background task management
        - Advanced caching and rate limiting
        
        ## Authentication
        Set `ENFORCE_AUTH=true` and provide either:
        - `X-API-Key` header with your API key
        - `Authorization: Bearer <JWT>` header
        
        ## Rate Limiting
        - Default: 100 requests/minute per IP
        - Enable distributed rate limiting with Redis: `USE_DISTRIBUTED_RATE_LIMIT=true`
        """,
        routes=app.routes,
    )
    
    # Add custom metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "email": "support@example.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]
    
    # Add security schemes
    if settings.enforce_auth:
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        openapi_schema["security"] = [
            {"ApiKeyAuth": []},
            {"BearerAuth": []}
        ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


