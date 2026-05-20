"""
Assessment routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

try:
    from core.addiction_analyzer import AddictionAnalyzer
    from utils.validators import AddictionTypeValidator
except ImportError:
    from ...core.addiction_analyzer import AddictionAnalyzer
    from ...utils.validators import AddictionTypeValidator

router = APIRouter()

analyzer = AddictionAnalyzer()


class AssessmentRequest(BaseModel):
    addiction_type: str
    severity: str
    frequency: str
    duration_years: Optional[float] = None
    daily_cost: Optional[float] = None
    triggers: List[str] = []
    motivations: List[str] = []
    previous_attempts: int = 0
    support_system: bool = False
    medical_conditions: List[str] = []
    additional_info: Optional[str] = None


@router.post("/assess")
async def assess_addiction(request: AssessmentRequest):
    """Evalúa una adicción y proporciona análisis completo"""
    try:
        if not AddictionTypeValidator.validate_type(request.addiction_type):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de adicción no válido. Tipos válidos: {AddictionTypeValidator.get_valid_types()}"
            )
        
        assessment_data = request.dict()
        analysis = analyzer.assess_addiction(assessment_data)
        
        return JSONResponse(content=analysis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en evaluación: {str(e)}")


@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Obtiene perfil del usuario"""
    return JSONResponse(content={
        "user_id": user_id,
        "message": "Perfil del usuario (implementar con base de datos)",
        "status": "success"
    })


@router.post("/update-profile")
async def update_profile(
    user_id: str = Body(...),
    profile_data: dict = Body(...)
):
    """Actualiza perfil del usuario"""
    try:
        return JSONResponse(content={
            "user_id": user_id,
            "profile": profile_data,
            "status": "success",
            "message": "Perfil actualizado"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando perfil: {str(e)}")



