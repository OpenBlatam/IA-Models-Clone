"""
Documentation Module

Comprehensive API documentation with:
- Interactive API documentation
- OpenAPI schema generation
- Request/response examples
- Error code documentation
- Performance metrics documentation
"""

from .api_documentation import (
    APIDocumentationConfig,
    create_custom_openapi,
    setup_documentation_routes,
    APIExamplesGenerator
)

__all__ = [
    'APIDocumentationConfig',
    'create_custom_openapi',
    'setup_documentation_routes',
    'APIExamplesGenerator'
]