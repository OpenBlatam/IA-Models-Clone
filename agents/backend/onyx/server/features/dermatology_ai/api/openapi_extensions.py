"""
OpenAPI/Swagger Extensions
Enhanced API documentation with examples and schemas
"""

from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def custom_openapi(app) -> Dict[str, Any]:
    """
    Custom OpenAPI schema with enhanced documentation
    
    Usage:
        app.openapi = lambda: custom_openapi(app)
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "analysis",
            "description": "Skin analysis operations. Upload images for AI-powered analysis.",
            "externalDocs": {
                "description": "Analysis Guide",
                "url": "https://docs.example.com/analysis"
            }
        },
        {
            "name": "recommendations",
            "description": "Product recommendations based on skin analysis.",
        },
        {
            "name": "health",
            "description": "Health check and system status endpoints.",
        },
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "https://auth.example.com/authorize",
                    "tokenUrl": "https://auth.example.com/token",
                    "scopes": {
                        "read": "Read access",
                        "write": "Write access",
                    }
                }
            }
        },
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Add examples to schemas
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        schemas = openapi_schema["components"]["schemas"]
        
        # Add example to Analysis schema
        if "Analysis" in schemas:
            schemas["Analysis"]["example"] = {
                "id": "analysis-123",
                "user_id": "user-123",
                "status": "completed",
                "metrics": {
                    "overall_score": 75.0,
                    "texture_score": 80.0,
                    "hydration_score": 70.0,
                },
                "created_at": "2025-01-01T00:00:00Z"
            }
        
        # Add example to Recommendation schema
        if "Recommendation" in schemas:
            schemas["Recommendation"]["example"] = {
                "product_id": "product-123",
                "product_name": "Moisturizing Cream",
                "category": "skincare",
                "priority": 1,
                "reason": "Suitable for combination skin",
                "confidence": 0.85
            }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "https://api.example.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.example.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8006",
            "description": "Local development server"
        }
    ]
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "email": "api-support@example.com",
        "url": "https://support.example.com"
    }
    
    # Add license
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema















