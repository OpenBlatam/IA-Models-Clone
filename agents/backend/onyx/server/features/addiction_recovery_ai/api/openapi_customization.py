"""
OpenAPI customization
Custom OpenAPI schema generation
"""

from typing import Dict, Any
from fastapi.openapi.utils import get_openapi


def customize_openapi_schema(
    app,
    title: str = "Addiction Recovery AI API",
    version: str = "3.3.0",
    description: str = "Sistema de IA para ayudar a dejar adicciones"
) -> Dict[str, Any]:
    """
    Customize OpenAPI schema
    
    Args:
        app: FastAPI application instance
        title: API title
        version: API version
        description: API description
    
    Returns:
        Customized OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=title,
        version=version,
        description=description,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "Assessment",
            "description": "Addiction assessment and evaluation endpoints"
        },
        {
            "name": "Progress",
            "description": "Progress tracking and statistics endpoints"
        },
        {
            "name": "Relapse",
            "description": "Relapse prevention and risk assessment endpoints"
        },
        {
            "name": "Support",
            "description": "Support and motivation endpoints"
        },
        {
            "name": "Analytics",
            "description": "Analytics and reporting endpoints"
        },
        {
            "name": "Health",
            "description": "Health check endpoints"
        }
    ]
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8018",
            "description": "Development server"
        },
        {
            "url": "https://api.addictionrecovery.ai",
            "description": "Production server"
        }
    ]
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "email": "support@addictionrecovery.ai"
    }
    
    # Add license information
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

