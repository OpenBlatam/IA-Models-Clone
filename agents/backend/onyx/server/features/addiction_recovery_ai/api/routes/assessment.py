"""
Assessment routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict

try:
    from schemas.assessment import (
        AssessmentRequest,
        AssessmentResponse,
        ProfileResponse,
        UpdateProfileRequest
    )
    from schemas.common import ErrorResponse, SuccessResponse
    from dependencies import AddictionAnalyzerDep
    from utils.validators import AddictionTypeValidator
except ImportError:
    from ...schemas.assessment import (
        AssessmentRequest,
        AssessmentResponse,
        ProfileResponse,
        UpdateProfileRequest
    )
    from ...schemas.common import ErrorResponse, SuccessResponse
    from ...dependencies import AddictionAnalyzerDep
    from ...utils.validators import AddictionTypeValidator

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
    # Guard clause: Validate addiction type early
    if not AddictionTypeValidator.validate_type(request.addiction_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de adicción no válido. Tipos válidos: {AddictionTypeValidator.get_valid_types()}"
        )
    
    # Process assessment
    try:
        assessment_data = request.model_dump()
        analysis = analyzer.assess_addiction(assessment_data)
        
        # Transform to response model
        return AssessmentResponse(
            assessment_id=analysis.get("assessment_id", ""),
            addiction_type=request.addiction_type,
            severity_score=analysis.get("severity_score", 0.0),
            risk_level=analysis.get("risk_level", "unknown"),
            recommendations=analysis.get("recommendations", []),
            next_steps=analysis.get("next_steps", [])
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en evaluación: {str(e)}"
        )


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
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    # TODO: Implement database lookup
    # For now, return placeholder
    return ProfileResponse(
        user_id=user_id,
        email=None,
        name=None,
        addiction_type=None,
        days_sober=None,
        created_at=None,
        updated_at=None
    )


@router.post(
    "/update-profile",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_profile(
    request: UpdateProfileRequest
) -> SuccessResponse:
    """
    Actualiza perfil del usuario
    
    - **email**: Email del usuario (opcional)
    - **name**: Nombre del usuario (opcional)
    - **addiction_type**: Tipo de adicción (opcional)
    """
    # Guard clause: At least one field must be provided
    if not any([request.email, request.name, request.addiction_type, request.additional_info]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Al menos un campo debe ser proporcionado para actualizar"
        )
    
    # TODO: Implement database update
    return SuccessResponse(
        message="Perfil actualizado exitosamente"
    )

