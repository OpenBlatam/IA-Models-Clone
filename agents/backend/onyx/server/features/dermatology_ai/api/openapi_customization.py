"""
OpenAPI/Swagger Customization
Enhances API documentation with examples and better descriptions
"""

from fastapi.openapi.utils import get_openapi
from typing import Dict, Any


def custom_openapi(app) -> Dict[str, Any]:
    """
    Customize OpenAPI schema
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Customized OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Dermatology AI API",
        version="7.1.0",
        description="""
        # Dermatology AI API Documentation
        
        AI-powered skin analysis and recommendation system built with Hexagonal Architecture.
        
        ## Features
        
        - **Skin Image Analysis**: Upload images for AI-powered analysis
        - **Condition Detection**: Automatic detection of skin conditions
        - **Metrics Calculation**: Comprehensive skin health metrics
        - **Personalized Recommendations**: Product recommendations based on analysis
        
        ## Authentication
        
        All endpoints require authentication via OAuth2 Bearer token.
        
        ## Rate Limiting
        
        - Default: 10 requests per second
        - Burst: 20 requests
        - Rate limit headers included in responses
        
        ## Error Handling
        
        - **400**: Validation errors
        - **401**: Authentication required
        - **404**: Resource not found
        - **429**: Rate limit exceeded
        - **500**: Internal server error
        
        ## Architecture
        
        This API follows Hexagonal Architecture principles:
        - **Domain Layer**: Business logic and entities
        - **Application Layer**: Use cases and orchestration
        - **Infrastructure Layer**: External adapters and implementations
        
        ## Health Checks
        
        - `GET /health` - Basic health check
        - `GET /health/ready` - Readiness check
        - `GET /health/live` - Liveness check
        - `GET /health/detailed` - Detailed component status
        - `GET /health/metrics` - Prometheus metrics
        """,
        routes=app.routes,
    )
    
    # Add custom examples
    openapi_schema["components"]["schemas"]["AnalysisResponse"] = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "example": True},
            "analysis_id": {"type": "string", "example": "550e8400-e29b-41d4-a716-446655440000"},
            "status": {"type": "string", "example": "completed"},
            "metrics": {
                "type": "object",
                "example": {
                    "overall_score": 75.5,
                    "hydration_score": 68.0,
                    "texture_score": 72.0,
                    "elasticity_score": 70.0
                }
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "example": "acne"},
                        "confidence": {"type": "number", "example": 0.85},
                        "severity": {"type": "string", "example": "moderate"}
                    }
                }
            }
        }
    }
    
    openapi_schema["components"]["schemas"]["RecommendationResponse"] = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "example": True},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "example": "hydrating-serum-001"},
                        "product_name": {"type": "string", "example": "Hydrating Serum"},
                        "category": {"type": "string", "example": "serum"},
                        "priority": {"type": "integer", "example": 1},
                        "reason": {"type": "string", "example": "Low hydration score detected"},
                        "confidence": {"type": "number", "example": 0.85},
                        "usage_frequency": {"type": "string", "example": "daily"}
                    }
                }
            }
        }
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "/oauth/authorize",
                    "tokenUrl": "/oauth/token",
                    "scopes": {
                        "read": "Read access",
                        "write": "Write access"
                    }
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema















