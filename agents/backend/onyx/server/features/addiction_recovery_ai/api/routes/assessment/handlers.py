"""
Request handlers for assessment endpoints
Pure functions for processing requests
"""

try:
    from schemas.assessment import AssessmentRequest, AssessmentResponse, ProfileResponse, UpdateProfileRequest
    from schemas.common import SuccessResponse
    from dependencies import AddictionAnalyzerDep
except ImportError:
    from ...schemas.assessment import AssessmentRequest, AssessmentResponse, ProfileResponse, UpdateProfileRequest
    from ...schemas.common import SuccessResponse
    from ...dependencies import AddictionAnalyzerDep


async def process_assessment(
    request: AssessmentRequest,
    analyzer: AddictionAnalyzerDep
) -> AssessmentResponse:
    """Process addiction assessment"""
    try:
        from core.addiction_analyzer_functions import assess_addiction
        from .transformers import transform_assessment_request_to_dict, transform_analysis_to_response
    except ImportError:
        from ...core.addiction_analyzer_functions import assess_addiction
        from .transformers import transform_assessment_request_to_dict, transform_analysis_to_response
    
    # Transform request to dict (RORO pattern)
    assessment_data = transform_assessment_request_to_dict(request)
    
    # Use functional approach instead of class method
    analysis = assess_addiction(assessment_data)
    
    # Transform analysis to response (RORO pattern)
    return transform_analysis_to_response(analysis, request)


async def get_user_profile(user_id: str) -> ProfileResponse:
    """Get user profile"""
    # TODO: Implement database lookup
    return ProfileResponse(
        user_id=user_id,
        email=None,
        name=None,
        addiction_type=None,
        days_sober=None,
        created_at=None,
        updated_at=None
    )


async def update_user_profile(request: UpdateProfileRequest) -> SuccessResponse:
    """Update user profile"""
    # TODO: Implement database update
    return SuccessResponse(message="Perfil actualizado exitosamente")

