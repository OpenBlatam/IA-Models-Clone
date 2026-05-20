"""
OpenAPI Extensions for Color Grading API
=========================================

Enhanced OpenAPI documentation with examples and schemas.
"""

from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def custom_openapi(app) -> Dict[str, Any]:
    """
    Custom OpenAPI schema with enhanced documentation.
    
    Args:
        app: FastAPI application
        
    Returns:
        OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Color Grading AI TruthGPT API",
        version="1.0.0",
        description="""
        ## Color Grading AI TruthGPT API
        
        Automatic color grading API with OpenRouter and TruthGPT integration.
        
        ### Features
        
        - **Video & Image Processing**: Apply color grading to videos and images
        - **Templates**: Use pre-defined color grading templates
        - **Color Matching**: Match colors from reference images/videos
        - **Text-to-Color**: Generate color parameters from text descriptions
        - **Batch Processing**: Process multiple files in parallel
        - **Quality Analysis**: Analyze video quality metrics
        - **Presets**: Save and reuse custom color grading presets
        
        ### Authentication
        
        Currently no authentication required. For production, implement API keys.
        
        ### Rate Limiting
        
        - 100 requests per minute per IP address
        - Check `X-RateLimit-Remaining` header for remaining requests
        
        ### Examples
        
        See individual endpoint documentation for examples.
        """,
        routes=app.routes,
    )
    
    # Add custom examples
    openapi_schema["components"]["examples"] = {
        "ColorParams": {
            "summary": "Color Grading Parameters",
            "value": {
                "brightness": 0.1,
                "contrast": 1.2,
                "saturation": 1.1,
                "color_balance": {
                    "r": 0.1,
                    "g": 0.0,
                    "b": -0.1
                }
            }
        },
        "GradingRequest": {
            "summary": "Video Grading Request",
            "value": {
                "template_name": "Cinematic Warm",
                "description": "warm cinematic look with high contrast"
            }
        }
    }
    
    # Add tags
    openapi_schema["tags"] = [
        {
            "name": "Grading",
            "description": "Color grading operations for videos and images"
        },
        {
            "name": "Analysis",
            "description": "Media analysis and quality metrics"
        },
        {
            "name": "Templates",
            "description": "Color grading templates management"
        },
        {
            "name": "Presets",
            "description": "User-created presets management"
        },
        {
            "name": "Batch",
            "description": "Batch processing operations"
        },
        {
            "name": "Health",
            "description": "Health check and system status"
        },
        {
            "name": "Tasks",
            "description": "Async task management"
        },
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema




