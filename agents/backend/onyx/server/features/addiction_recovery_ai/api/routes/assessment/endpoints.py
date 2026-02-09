"""
Assessment endpoints - Ultra modular implementation
"""

from fastapi import APIRouter, status

try:
    from schemas.assessment import (
        AssessmentRequest,
        AssessmentResponse,
        ProfileResponse,
        UpdateProfileRequest
    )
    from schemas.common import ErrorResponse, SuccessResponse
    from dependencies import AddictionAnalyzerDep
    from .validators import validate_assessment_request, validate_profile_update
    from .handlers import process_assessment, get_user_profile, update_user_profile
except ImportError:
    from ...schemas.assessment import (
        AssessmentRequest,
        AssessmentResponse,
        ProfileResponse,
        UpdateProfileRequest
    )
    from ...schemas.common import ErrorResponse, SuccessResponse
    from ...dependencies import AddictionAnalyzerDep
    from .validators import validate_assessment_request, validate_profile_update
    from .handlers import process_assessment, get_user_profile, update_user_profile

router = APIRouter(prefix="/assessment", tags=["Assessment"])


@router.post(
    "/assess",
    response_model=AssessmentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def assess_addiction(
    request: AssessmentRequest,
    analyzer: AddictionAnalyzerDep
) -> AssessmentResponse:
    """
    Evalúa una adicción y proporciona análisis completo
    
    - **addiction_type**: Tipo de adicción (smoking, alcohol, drugs, etc.)
    - **severity**: Nivel de severidad (low, moderate, high, severe)
    - **frequency**: Frecuencia de uso (daily, weekly, monthly, occasional)
    """
    await validate_assessment_request(request)
    return await process_assessment(request, analyzer)


@router.get(
    "/profile/{user_id}",
    response_model=ProfileResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_profile(user_id: str) -> ProfileResponse:
    """
    Obtiene perfil del usuario
    
    - **user_id**: ID único del usuario
    """
    from utils.validators import validate_user_id
    
    validate_user_id(user_id)
    return await get_user_profile(user_id)


@router.post(
    "/update-profile",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_profile(request: UpdateProfileRequest) -> SuccessResponse:
    """
    Actualiza perfil del usuario
    
    - **email**: Email del usuario (opcional)
    - **name**: Nombre del usuario (opcional)
    - **addiction_type**: Tipo de adicción (opcional)
    """
    await validate_profile_update(request)
    return await update_user_profile(request)

