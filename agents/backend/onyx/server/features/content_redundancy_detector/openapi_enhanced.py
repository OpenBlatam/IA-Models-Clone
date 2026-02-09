"""
Enhanced OpenAPI Documentation
Advanced API documentation with examples, schemas, and interactive features
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List
import json
import time

# Enhanced OpenAPI schema configuration
OPENAPI_TAGS = [
    {
        "name": "Content Analysis",
        "description": "Advanced content analysis and redundancy detection",
        "externalDocs": {
            "description": "Content Analysis Guide",
            "url": "https://docs.example.com/content-analysis"
        }
    },
    {
        "name": "Similarity Detection",
        "description": "Text similarity comparison and matching",
        "externalDocs": {
            "description": "Similarity Detection Guide",
            "url": "https://docs.example.com/similarity-detection"
        }
    },
    {
        "name": "Quality Assessment",
        "description": "Content quality analysis and scoring",
        "externalDocs": {
            "description": "Quality Assessment Guide",
            "url": "https://docs.example.com/quality-assessment"
        }
    },
    {
        "name": "Batch Processing",
        "description": "Bulk content processing and analysis",
        "externalDocs": {
            "description": "Batch Processing Guide",
            "url": "https://docs.example.com/batch-processing"
        }
    },
    {
        "name": "Webhooks",
        "description": "Event notifications and real-time updates",
        "externalDocs": {
            "description": "Webhooks Guide",
            "url": "https://docs.example.com/webhooks"
        }
    },
    {
        "name": "Health & Monitoring",
        "description": "System health checks and monitoring endpoints",
        "externalDocs": {
            "description": "Monitoring Guide",
            "url": "https://docs.example.com/monitoring"
        }
    },
    {
        "name": "API Modular",
        "description": "Modular API endpoints with advanced features",
        "externalDocs": {
            "description": "Modular API Guide",
            "url": "https://docs.example.com/modular-api"
        }
    }
]

# Enhanced examples for API documentation
API_EXAMPLES = {
    "content_analysis": {
        "basic": {
            "summary": "Basic Content Analysis",
            "description": "Simple content analysis request",
            "value": {
                "content": "This is a sample text for analysis. It contains multiple sentences to demonstrate the redundancy detection capabilities.",
                "analysis_type": "redundancy",
                "language": "en",
                "threshold": 0.7
            }
        },
        "comprehensive": {
            "summary": "Comprehensive Analysis",
            "description": "Full analysis with all options enabled",
            "value": {
                "content": "Advanced content analysis with multiple features enabled for comprehensive evaluation.",
                "analysis_type": "comprehensive",
                "language": "auto",
                "threshold": 0.8,
                "include_metadata": True,
                "cache_result": True,
                "priority": 8
            }
        },
        "multilingual": {
            "summary": "Multilingual Content",
            "description": "Analysis of content in different languages",
            "value": {
                "content": "Este es un texto de ejemplo en español para demostrar las capacidades de detección de redundancia.",
                "analysis_type": "redundancy",
                "language": "es",
                "threshold": 0.75
            }
        }
    },
    "similarity_comparison": {
        "basic": {
            "summary": "Basic Similarity Check",
            "description": "Simple text similarity comparison",
            "value": {
                "text1": "The quick brown fox jumps over the lazy dog",
                "text2": "A fast brown fox leaps over the sleepy dog",
                "algorithm": "cosine",
                "threshold": 0.7
            }
        },
        "advanced": {
            "summary": "Advanced Similarity Analysis",
            "description": "Advanced similarity with multiple algorithms",
            "value": {
                "text1": "Machine learning is a subset of artificial intelligence",
                "text2": "AI includes machine learning as one of its components",
                "algorithm": "hybrid",
                "threshold": 0.8,
                "normalize": True,
                "case_sensitive": False
            }
        }
    },
    "quality_assessment": {
        "basic": {
            "summary": "Basic Quality Check",
            "description": "Simple content quality assessment",
            "value": {
                "content": "This is a well-written article with proper grammar and clear structure.",
                "quality_metrics": ["readability", "grammar"],
                "target_audience": "general"
            }
        },
        "comprehensive": {
            "summary": "Comprehensive Quality Analysis",
            "description": "Full quality assessment with all metrics",
            "value": {
                "content": "A comprehensive analysis of content quality requires multiple evaluation criteria including readability, grammar, coherence, completeness, and clarity.",
                "quality_metrics": ["readability", "grammar", "coherence", "completeness", "clarity"],
                "target_audience": "academic",
                "min_score": 0.8
            }
        }
    },
    "batch_processing": {
        "small_batch": {
            "summary": "Small Batch Processing",
            "description": "Processing a small batch of content items",
            "value": {
                "items": [
                    {"content": "First item for analysis", "type": "text"},
                    {"content": "Second item for analysis", "type": "text"},
                    {"content": "Third item for analysis", "type": "text"}
                ],
                "batch_size": 3,
                "priority": 5
            }
        },
        "large_batch": {
            "summary": "Large Batch Processing",
            "description": "Processing a large batch with callback",
            "value": {
                "items": [
                    {"content": f"Content item {i}", "id": f"item_{i}"} 
                    for i in range(1, 21)
                ],
                "batch_size": 10,
                "priority": 7,
                "callback_url": "https://api.example.com/webhooks/batch-complete"
            }
        }
    }
}

# Response examples
RESPONSE_EXAMPLES = {
    "success": {
        "summary": "Successful Response",
        "description": "Standard successful API response",
        "value": {
            "success": True,
            "data": {
                "analysis_id": "analysis_123456789",
                "redundancy_score": 0.75,
                "similarity_threshold": 0.7,
                "is_redundant": True,
                "confidence": 0.92,
                "processing_time": 0.245,
                "metadata": {
                    "language": "en",
                    "word_count": 25,
                    "character_count": 150,
                    "analysis_type": "redundancy"
                }
            },
            "error": None,
            "timestamp": 1640995200.0,
            "message": "Analysis completed successfully"
        }
    },
    "error": {
        "summary": "Error Response",
        "description": "Standard error response format",
        "value": {
            "success": False,
            "data": None,
            "error": {
                "message": "Validation error",
                "status_code": 400,
                "type": "ValidationError",
                "detail": {
                    "field": "content",
                    "issue": "Content cannot be empty"
                }
            },
            "timestamp": 1640995200.0,
            "message": "Request validation failed"
        }
    },
    "rate_limited": {
        "summary": "Rate Limited Response",
        "description": "Rate limit exceeded response",
        "value": {
            "success": False,
            "data": None,
            "error": {
                "message": "Rate limit exceeded",
                "status_code": 429,
                "type": "RateLimitError",
                "detail": {
                    "retry_after": 60,
                    "limit": 100,
                    "window": "minute"
                }
            },
            "timestamp": 1640995200.0,
            "message": "Too many requests, please try again later"
        }
    }
}


def create_enhanced_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Create enhanced OpenAPI schema with advanced features"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    # Get base OpenAPI schema
    openapi_schema = get_openapi(
        title="Content Redundancy Detector API",
        version="2.0.0",
        description="""
        # Content Redundancy Detector API
        
        Advanced AI-powered content analysis and redundancy detection API with microservices architecture.
        
        ## Features
        
        - **Content Analysis**: Detect redundant content using advanced ML algorithms
        - **Similarity Detection**: Compare text similarity with multiple algorithms
        - **Quality Assessment**: Evaluate content quality across multiple metrics
        - **Batch Processing**: Process large volumes of content efficiently
        - **Real-time Webhooks**: Get instant notifications for analysis completion
        - **Advanced Caching**: Redis-powered intelligent caching for optimal performance
        - **Rate Limiting**: Smart rate limiting with multiple strategies
        - **Health Monitoring**: Comprehensive health checks and monitoring
        
        ## Authentication
        
        The API uses OAuth2 with JWT tokens for authentication. Include the token in the Authorization header:
        
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ## Rate Limiting
        
        - **Free Tier**: 100 requests/minute, 1,000 requests/hour
        - **Premium Tier**: 500 requests/minute, 10,000 requests/hour
        - **Enterprise**: Custom limits available
        
        ## Response Format
        
        All responses follow a consistent format:
        
        ```json
        {
          "success": true,
          "data": { ... },
          "error": null,
          "timestamp": 1640995200.0,
          "message": "Operation completed successfully"
        }
        ```
        
        ## Error Handling
        
        Errors are returned with appropriate HTTP status codes and detailed error information:
        
        - `400 Bad Request`: Invalid request data
        - `401 Unauthorized`: Authentication required
        - `403 Forbidden`: Insufficient permissions
        - `429 Too Many Requests`: Rate limit exceeded
        - `500 Internal Server Error`: Server error
        
        ## Webhooks
        
        Subscribe to real-time events for analysis completion, batch processing, and system updates.
        
        ## Support
        
        - Documentation: [https://docs.example.com](https://docs.example.com)
        - Support: [support@example.com](mailto:support@example.com)
        - Status: [https://status.example.com](https://status.example.com)
        """,
        routes=app.routes,
        tags=OPENAPI_TAGS
    )
    
    # Enhance the schema
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "url": "https://docs.example.com",
        "email": "support@example.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png",
        "altText": "Content Redundancy Detector"
    }
    
    # Add servers
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
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "https://auth.example.com/oauth/authorize",
                    "tokenUrl": "https://auth.example.com/oauth/token",
                    "scopes": {
                        "read": "Read access to content analysis",
                        "write": "Write access to content analysis",
                        "admin": "Administrative access"
                    }
                },
                "clientCredentials": {
                    "tokenUrl": "https://auth.example.com/oauth/token",
                    "scopes": {
                        "read": "Read access to content analysis",
                        "write": "Write access to content analysis"
                    }
                }
            }
        },
        "ApiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [
        {"OAuth2": ["read"]},
        {"ApiKey": []}
    ]
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Complete API Documentation",
        "url": "https://docs.example.com/api"
    }
    
    # Add examples to paths
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                # Add examples based on path
                if "analyze" in path:
                    operation["requestBody"] = operation.get("requestBody", {})
                    if "content" in operation["requestBody"]:
                        for content_type, content in operation["requestBody"]["content"].items():
                            if "application/json" in content_type:
                                content["examples"] = API_EXAMPLES["content_analysis"]
                
                elif "similarity" in path:
                    operation["requestBody"] = operation.get("requestBody", {})
                    if "content" in operation["requestBody"]:
                        for content_type, content in operation["requestBody"]["content"].items():
                            if "application/json" in content_type:
                                content["examples"] = API_EXAMPLES["similarity_comparison"]
                
                elif "quality" in path:
                    operation["requestBody"] = operation.get("requestBody", {})
                    if "content" in operation["requestBody"]:
                        for content_type, content in operation["requestBody"]["content"].items():
                            if "application/json" in content_type:
                                content["examples"] = API_EXAMPLES["quality_assessment"]
                
                elif "batch" in path:
                    operation["requestBody"] = operation.get("requestBody", {})
                    if "content" in operation["requestBody"]:
                        for content_type, content in operation["requestBody"]["content"].items():
                            if "application/json" in content_type:
                                content["examples"] = API_EXAMPLES["batch_processing"]
                
                # Add response examples
                if "responses" in operation:
                    for status_code, response in operation["responses"].items():
                        if "content" in response:
                            for content_type, content in response["content"].items():
                                if "application/json" in content_type:
                                    if status_code.startswith("2"):
                                        content["examples"] = RESPONSE_EXAMPLES["success"]
                                    elif status_code == "429":
                                        content["examples"] = RESPONSE_EXAMPLES["rate_limited"]
                                    else:
                                        content["examples"] = RESPONSE_EXAMPLES["error"]
    
    # Add custom extensions
    openapi_schema["x-api-version"] = "2.0.0"
    openapi_schema["x-rate-limit"] = {
        "free": "100/minute, 1000/hour",
        "premium": "500/minute, 10000/hour",
        "enterprise": "custom"
    }
    openapi_schema["x-features"] = [
        "content-analysis",
        "similarity-detection",
        "quality-assessment",
        "batch-processing",
        "real-time-webhooks",
        "advanced-caching",
        "rate-limiting",
        "health-monitoring"
    ]
    
    app.openapi_schema = openapi_schema
    return openapi_schema


def create_enhanced_docs_html(app: FastAPI) -> HTMLResponse:
    """Create enhanced Swagger UI HTML with custom styling"""
    
    openapi_schema = create_enhanced_openapi_schema(app)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Content Redundancy Detector API</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
        <style>
            .swagger-ui .topbar {{
                background-color: #2c3e50;
            }}
            .swagger-ui .topbar .download-url-wrapper {{
                display: none;
            }}
            .swagger-ui .info .title {{
                color: #2c3e50;
            }}
            .swagger-ui .scheme-container {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
            }}
            .custom-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }}
            .custom-header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .custom-header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="custom-header">
            <h1>Content Redundancy Detector API</h1>
            <p>Advanced AI-powered content analysis and redundancy detection</p>
        </div>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script>
            SwaggerUIBundle({{
                url: '/openapi.json',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.presets.standalone
                ],
                layout: "StandaloneLayout",
                deepLinking: true,
                showExtensions: true,
                showCommonExtensions: true,
                tryItOutEnabled: true,
                requestInterceptor: function(request) {{
                    // Add custom headers or modify requests
                    return request;
                }},
                responseInterceptor: function(response) {{
                    // Handle responses
                    return response;
                }},
                onComplete: function() {{
                    console.log('Swagger UI loaded');
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


def create_enhanced_redoc_html(app: FastAPI) -> HTMLResponse:
    """Create enhanced ReDoc HTML with custom styling"""
    
    openapi_schema = create_enhanced_openapi_schema(app)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Content Redundancy Detector API - ReDoc</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        <style>
            body {{
                margin: 0;
                padding: 0;
            }}
            .custom-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 0;
            }}
            .custom-header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-family: 'Montserrat', sans-serif;
            }}
            .custom-header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
                font-family: 'Roboto', sans-serif;
            }}
        </style>
    </head>
    <body>
        <div class="custom-header">
            <h1>Content Redundancy Detector API</h1>
            <p>Advanced AI-powered content analysis and redundancy detection</p>
        </div>
        <redoc spec-url='/openapi.json'></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)





