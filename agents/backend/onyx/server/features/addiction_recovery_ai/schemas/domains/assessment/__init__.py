"""
Assessment domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.assessment import (
        AssessmentRequest,
        AssessmentResponse,
        ProfileResponse,
        UpdateProfileRequest
    )
    
    def register_schemas():
        register_schema("assessment", "AssessmentRequest", AssessmentRequest)
        register_schema("assessment", "AssessmentResponse", AssessmentResponse)
        register_schema("assessment", "ProfileResponse", ProfileResponse)
        register_schema("assessment", "UpdateProfileRequest", UpdateProfileRequest)
except ImportError:
    pass



