"""
Relapse prevention routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, status

try:
    from schemas.relapse import (
        RelapseRiskCheckRequest,
        RelapseRiskResponse,
        CopingStrategiesRequest,
        CopingStrategiesResponse,
        EmergencyPlanRequest,
        EmergencyPlanResponse
    )
    from schemas.common import ErrorResponse
    from dependencies import RelapsePreventionDep
except ImportError:
    from ...schemas.relapse import (
        RelapseRiskCheckRequest,
        RelapseRiskResponse,
        CopingStrategiesRequest,
        CopingStrategiesResponse,
        EmergencyPlanRequest,
        EmergencyPlanResponse
    )
    from ...schemas.common import ErrorResponse
    from ...dependencies import RelapsePreventionDep

router = APIRouter(prefix="/relapse", tags=["Relapse Prevention"])


@router.post(
    "/check-risk",
    response_model=RelapseRiskResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def check_relapse_risk(
    request: RelapseRiskCheckRequest,
    relapse_prevention: RelapsePreventionDep
) -> RelapseRiskResponse:
    """
    Evalúa riesgo de recaída
    
    - **user_id**: ID del usuario
    - **days_sober**: Días sobrio
    - **stress_level**: Nivel de estrés (0-10)
    - **support_level**: Nivel de apoyo (0-10)
    - **triggers**: Lista de triggers actuales
    """
    # Guard clause: Validate stress and support levels
    if request.stress_level < 0 or request.stress_level > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stress_level debe estar entre 0 y 10"
        )
    
    if request.support_level < 0 or request.support_level > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="support_level debe estar entre 0 y 10"
        )
    
    # Process risk assessment
    try:
        current_state = {
            "stress_level": request.stress_level,
            "support_level": request.support_level,
            "triggers": request.triggers,
            "isolation": request.isolation,
            "negative_thinking": request.negative_thinking,
            "romanticizing": request.romanticizing,
            "skipping_support": request.skipping_support
        }
        
        risk_analysis = relapse_prevention.check_relapse_risk(
            request.user_id,
            request.days_sober,
            current_state,
            None
        )
        
        return RelapseRiskResponse(
            user_id=request.user_id,
            risk_level=risk_analysis.get("risk_level", "unknown"),
            risk_score=risk_analysis.get("risk_score", 0.0),
            risk_factors=risk_analysis.get("risk_factors", []),
            protective_factors=risk_analysis.get("protective_factors", []),
            recommendations=risk_analysis.get("recommendations", [])
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluando riesgo: {str(e)}"
        )


@router.post(
    "/coping-strategies",
    response_model=CopingStrategiesResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_coping_strategies(
    request: CopingStrategiesRequest,
    relapse_prevention: RelapsePreventionDep
) -> CopingStrategiesResponse:
    """
    Obtiene estrategias de afrontamiento para una situación
    
    - **situation**: Descripción de la situación actual
    - **trigger_type**: Tipo de trigger (opcional)
    """
    # Guard clause: Validate situation is provided
    if not request.situation or not request.situation.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="situation es requerido"
        )
    
    try:
        strategies = relapse_prevention.get_coping_strategies(
            request.situation,
            request.trigger_type
        )
        
        return CopingStrategiesResponse(
            situation=request.situation,
            trigger_type=request.trigger_type,
            strategies=strategies.get("strategies", []),
            immediate_actions=strategies.get("immediate_actions", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estrategias: {str(e)}"
        )


@router.post(
    "/emergency-plan",
    response_model=EmergencyPlanResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def generate_emergency_plan(
    request: EmergencyPlanRequest,
    relapse_prevention: RelapsePreventionDep
) -> EmergencyPlanResponse:
    """
    Genera plan de emergencia
    
    - **user_id**: ID del usuario
    - **current_situation**: Detalles de la situación actual
    """
    # Guard clause: Validate user_id
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    # Guard clause: Validate current_situation
    if not request.current_situation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="current_situation es requerido"
        )
    
    try:
        plan = relapse_prevention.generate_emergency_plan(
            request.user_id,
            request.current_situation
        )
        
        return EmergencyPlanResponse(
            user_id=request.user_id,
            plan_id=plan.get("plan_id", ""),
            immediate_steps=plan.get("immediate_steps", []),
            contacts=plan.get("contacts", []),
            resources=plan.get("resources", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando plan de emergencia: {str(e)}"
        )

