"""
OpenAPI Schema Generation
=========================

Enhanced OpenAPI schema generation with custom features.
"""

import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from ..schemas.common_schemas import (
    ErrorResponse, SuccessResponse, PaginationResponse,
    HealthCheckResponse, SystemInfoResponse, MetricsResponse
)
from ..schemas.agent_schemas import (
    AgentResponse, AgentListResponse, CapabilityExecutionResponse
)
from ..schemas.workflow_schemas import (
    WorkflowResponse, WorkflowListResponse, WorkflowExecutionResponse
)
from ..schemas.document_schemas import (
    DocumentResponse, DocumentListResponse, DocumentGenerationResponse
)

class CustomOpenAPIGenerator:
    """Custom OpenAPI schema generator with enhanced features."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.custom_schemas = {}
        self.custom_examples = {}
        self.custom_responses = {}
    
    def add_custom_schema(self, name: str, schema: Dict[str, Any]):
        """Add a custom schema to the OpenAPI spec."""
        self.custom_schemas[name] = schema
    
    def add_custom_example(self, name: str, example: Dict[str, Any]):
        """Add a custom example to the OpenAPI spec."""
        self.custom_examples[name] = example
    
    def add_custom_response(self, name: str, response: Dict[str, Any]):
        """Add a custom response to the OpenAPI spec."""
        self.custom_responses[name] = response
    
    def generate_openapi_schema(self) -> Dict[str, Any]:
        """Generate enhanced OpenAPI schema."""
        # Get base OpenAPI schema
        openapi_schema = get_openapi(
            title="Business Agents System API",
            version="1.0.0",
            description=self._get_api_description(),
            routes=self.app.routes,
            tags=self._get_tags_metadata()
        )
        
        # Enhance the schema
        self._enhance_info(openapi_schema)
        self._enhance_servers(openapi_schema)
        self._enhance_security(openapi_schema)
        self._enhance_components(openapi_schema)
        self._enhance_paths(openapi_schema)
        self._add_custom_schemas(openapi_schema)
        self._add_custom_examples(openapi_schema)
        self._add_custom_responses(openapi_schema)
        
        return openapi_schema
    
    def _get_api_description(self) -> str:
        """Get comprehensive API description."""
        return """
# Business Agents System API

A comprehensive agent system for all business areas with workflow management and document generation.

## Features

- **Agent Management**: Create, manage, and execute business agents across different domains
- **Workflow Engine**: Design and execute complex business workflows
- **Document Generation**: Generate business documents in multiple formats
- **Real-time Monitoring**: Track system performance and agent executions
- **Caching & Performance**: Redis-based caching for improved performance
- **Rate Limiting**: Built-in rate limiting and request throttling

## Business Areas

- Marketing: Campaign planning, brand management, content strategy
- Sales: Lead qualification, customer acquisition, sales processes
- Operations: Process optimization, workflow automation
- HR: Employee lifecycle, recruitment, performance management
- Finance: Budget analysis, financial reporting, cost optimization
- Legal: Contract review, compliance monitoring, document analysis
- Technical: System documentation, API specifications, technical guides
- Content: Content creation, editorial workflows, publishing

## Authentication

The API uses API key authentication. Include your API key in the `X-API-Key` header:

```
X-API-Key: your-api-key-here
```

## Rate Limiting

API requests are rate limited to prevent abuse:
- **Default**: 100 requests per minute per IP
- **Authenticated**: 1000 requests per minute per API key
- **Burst**: Up to 10 requests per second

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Error Handling

The API uses standard HTTP status codes and returns structured error responses:

```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": {
    "field": "email",
    "issue": "Invalid email format"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

## Pagination

List endpoints support pagination:

```
GET /api/agents?page=1&size=10&sort_by=name&sort_order=asc
```

Response includes pagination metadata:

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "size": 10,
  "pages": 10,
  "has_next": true,
  "has_prev": false
}
```

## Webhooks

The system supports webhooks for real-time notifications:

- **Agent Execution Complete**: Notified when agent capabilities finish
- **Workflow Status Change**: Notified when workflow status changes
- **Document Generated**: Notified when document generation completes
- **System Alerts**: Notified of system health issues

## SDKs

Official SDKs are available for:
- Python
- JavaScript/TypeScript
- Java
- C#

## Support

For API support and questions:
- Documentation: https://docs.business-agents.com
- Support: support@business-agents.com
- Status: https://status.business-agents.com
        """
    
    def _get_tags_metadata(self) -> List[Dict[str, Any]]:
        """Get tags metadata for OpenAPI."""
        return [
            {
                "name": "Agents",
                "description": "Business agent management and execution",
                "externalDocs": {
                    "description": "Agent Documentation",
                    "url": "https://docs.business-agents.com/agents"
                }
            },
            {
                "name": "Workflows",
                "description": "Workflow design, execution, and management",
                "externalDocs": {
                    "description": "Workflow Documentation", 
                    "url": "https://docs.business-agents.com/workflows"
                }
            },
            {
                "name": "Documents",
                "description": "Document generation and management",
                "externalDocs": {
                    "description": "Document Documentation",
                    "url": "https://docs.business-agents.com/documents"
                }
            },
            {
                "name": "System",
                "description": "System information, health checks, and metrics",
                "externalDocs": {
                    "description": "System Documentation",
                    "url": "https://docs.business-agents.com/system"
                }
            },
            {
                "name": "Health",
                "description": "Health check and system status endpoints"
            }
        ]
    
    def _enhance_info(self, schema: Dict[str, Any]):
        """Enhance the info section."""
        schema["info"].update({
            "title": "Business Agents System API",
            "version": "1.0.0",
            "description": self._get_api_description(),
            "termsOfService": "https://business-agents.com/terms",
            "contact": {
                "name": "Business Agents Support",
                "url": "https://business-agents.com/support",
                "email": "support@business-agents.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            },
            "x-logo": {
                "url": "https://business-agents.com/logo.png",
                "altText": "Business Agents Logo"
            }
        })
    
    def _enhance_servers(self, schema: Dict[str, Any]):
        """Add server configurations."""
        schema["servers"] = [
            {
                "url": "https://api.business-agents.com/v1",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.business-agents.com/v1", 
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ]
    
    def _enhance_security(self, schema: Dict[str, Any]):
        """Add security schemes."""
        schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for authentication"
            }
        }
        
        # Add default security
        schema["security"] = [
            {"ApiKeyAuth": []},
            {"BearerAuth": []}
        ]
    
    def _enhance_components(self, schema: Dict[str, Any]):
        """Enhance components section."""
        # Add common response schemas
        schema["components"]["responses"] = {
            "ValidationError": {
                "description": "Validation error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "validation_error",
                            "message": "Invalid input data",
                            "details": {
                                "field": "email",
                                "issue": "Invalid email format"
                            },
                            "timestamp": "2024-01-01T00:00:00Z",
                            "request_id": "req_123456"
                        }
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "not_found",
                            "message": "Resource not found",
                            "timestamp": "2024-01-01T00:00:00Z",
                            "request_id": "req_123456"
                        }
                    }
                }
            },
            "RateLimitExceeded": {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "rate_limit_exceeded",
                            "message": "Too many requests",
                            "details": {
                                "limit": 100,
                                "window": "1 minute",
                                "retry_after": 60
                            },
                            "timestamp": "2024-01-01T00:00:00Z",
                            "request_id": "req_123456"
                        }
                    }
                }
            }
        }
        
        # Add common parameter schemas
        schema["components"]["parameters"] = {
            "PageParam": {
                "name": "page",
                "in": "query",
                "description": "Page number",
                "required": False,
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                }
            },
            "SizeParam": {
                "name": "size", 
                "in": "query",
                "description": "Page size",
                "required": False,
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            },
            "SortByParam": {
                "name": "sort_by",
                "in": "query",
                "description": "Sort field",
                "required": False,
                "schema": {
                    "type": "string"
                }
            },
            "SortOrderParam": {
                "name": "sort_order",
                "in": "query", 
                "description": "Sort order",
                "required": False,
                "schema": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "default": "asc"
                }
            }
        }
    
    def _enhance_paths(self, schema: Dict[str, Any]):
        """Enhance paths with additional metadata."""
        for path, methods in schema.get("paths", {}).items():
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    # Add common parameters to list endpoints
                    if method == "get" and "list" in operation.get("operationId", "").lower():
                        if "parameters" not in operation:
                            operation["parameters"] = []
                        
                        operation["parameters"].extend([
                            {"$ref": "#/components/parameters/PageParam"},
                            {"$ref": "#/components/parameters/SizeParam"},
                            {"$ref": "#/components/parameters/SortByParam"},
                            {"$ref": "#/components/parameters/SortOrderParam"}
                        ])
                    
                    # Add common responses
                    if "responses" not in operation:
                        operation["responses"] = {}
                    
                    # Add error responses
                    operation["responses"].update({
                        "400": {"$ref": "#/components/responses/ValidationError"},
                        "404": {"$ref": "#/components/responses/NotFound"},
                        "429": {"$ref": "#/components/responses/RateLimitExceeded"}
                    })
    
    def _add_custom_schemas(self, schema: Dict[str, Any]):
        """Add custom schemas."""
        if "components" not in schema:
            schema["components"] = {}
        if "schemas" not in schema["components"]:
            schema["components"]["schemas"] = {}
        
        schema["components"]["schemas"].update(self.custom_schemas)
    
    def _add_custom_examples(self, schema: Dict[str, Any]):
        """Add custom examples."""
        if "components" not in schema:
            schema["components"] = {}
        if "examples" not in schema["components"]:
            schema["components"]["examples"] = {}
        
        schema["components"]["examples"].update(self.custom_examples)
    
    def _add_custom_responses(self, schema: Dict[str, Any]):
        """Add custom responses."""
        if "components" not in schema:
            schema["components"] = {}
        if "responses" not in schema["components"]:
            schema["components"]["responses"] = {}
        
        schema["components"]["responses"].update(self.custom_responses)

def get_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Get enhanced OpenAPI schema for the application."""
    generator = CustomOpenAPIGenerator(app)
    return generator.generate_openapi_schema()
