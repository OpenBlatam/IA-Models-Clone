"""
OpenAPI Extensions
=================

Extensions and customizations for OpenAPI/Swagger documentation.
"""

from typing import Dict, Any


def get_openapi_tags() -> list:
    """
    Get OpenAPI tags for API documentation.
    
    Returns:
        List of tag dictionaries
    """
    return [
        {
            "name": "Clothing Change",
            "description": "Operations for changing character clothing with AI"
        },
        {
            "name": "Face Swap",
            "description": "Operations for face swapping in images"
        },
        {
            "name": "Batch Operations",
            "description": "Batch processing operations"
        },
        {
            "name": "Workflow Management",
            "description": "Workflow status and management operations"
        },
        {
            "name": "Metrics",
            "description": "Service metrics and analytics"
        },
        {
            "name": "Cache",
            "description": "Cache management operations"
        },
        {
            "name": "Rate Limiting",
            "description": "Rate limiting operations"
        },
        {
            "name": "Webhooks",
            "description": "Webhook registration and management"
        },
        {
            "name": "Health",
            "description": "Health check endpoints"
        }
    ]


def get_openapi_info() -> Dict[str, Any]:
    """
    Get OpenAPI info for API documentation.
    
    Returns:
        Dictionary with API info
    """
    return {
        "title": "Character Clothing Changer AI API",
        "description": """
        API for changing character clothing and face swapping using AI.
        
        Features:
        - Clothing change with OpenRouter and TruthGPT optimization
        - Face swap in inpainting images
        - Batch processing
        - Workflow management
        - Metrics and analytics
        - Webhook notifications
        - Rate limiting
        - Caching
        
        ## Authentication
        
        Currently no authentication required. API key authentication can be enabled.
        
        ## Rate Limiting
        
        Default rate limit: 100 requests per minute per client.
        Rate limit headers are included in all responses.
        
        ## Webhooks
        
        Register webhooks to receive notifications on workflow completion or failure.
        Webhooks support HMAC-SHA256 signatures for security.
        
        ## Error Handling
        
        All errors follow a consistent format:
        ```json
        {
            "success": false,
            "error": "Error message",
            "error_code": "ERROR_CODE",
            "timestamp": "2024-01-01T00:00:00"
        }
        ```
        """,
        "version": "1.0.0",
        "contact": {
            "name": "API Support",
            "email": "support@example.com"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    }


def get_openapi_servers() -> list:
    """
    Get OpenAPI servers for API documentation.
    
    Returns:
        List of server dictionaries
    """
    return [
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]


def get_openapi_responses() -> Dict[str, Dict[str, Any]]:
    """
    Get common OpenAPI response schemas.
    
    Returns:
        Dictionary of common responses
    """
    return {
        "400": {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Invalid request parameters"},
                            "error_code": {"type": "string", "example": "VALIDATION_ERROR"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        },
        "429": {
            "description": "Too Many Requests",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string", "example": "Rate limit exceeded"},
                            "retry_after": {"type": "integer", "example": 60}
                        }
                    }
                }
            },
            "headers": {
                "X-RateLimit-Limit": {
                    "schema": {"type": "integer"},
                    "description": "Rate limit per window"
                },
                "X-RateLimit-Remaining": {
                    "schema": {"type": "integer"},
                    "description": "Remaining requests in current window"
                },
                "X-RateLimit-Reset": {
                    "schema": {"type": "integer"},
                    "description": "Seconds until rate limit resets"
                }
            }
        },
        "500": {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "example": False},
                            "error": {"type": "string", "example": "Internal server error"},
                            "error_type": {"type": "string", "example": "internal_error"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }
    }

