"""
Common domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.common import (
        ErrorResponse,
        SuccessResponse,
        PaginatedResponse
    )
    
    def register_schemas():
        register_schema("common", "ErrorResponse", ErrorResponse)
        register_schema("common", "SuccessResponse", SuccessResponse)
        register_schema("common", "PaginatedResponse", PaginatedResponse)
except ImportError:
    pass



