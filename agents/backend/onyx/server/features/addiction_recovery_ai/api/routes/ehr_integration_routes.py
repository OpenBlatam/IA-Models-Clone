"""
EHR integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.ehr_integration_service import EHRIntegrationService
except ImportError:
    from ...services.ehr_integration_service import EHRIntegrationService

router = APIRouter()

ehr_integration = EHRIntegrationService()


@router.post("/ehr/connect")
async def connect_ehr_system(
    user_id: str = Body(...),
    ehr_system: str = Body(...),
    connection_credentials: Dict = Body(...)
):
    """Conecta sistema EHR"""
    try:
        connection = ehr_integration.connect_ehr_system(
            user_id, ehr_system, connection_credentials
        )
        return JSONResponse(content=connection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando EHR: {str(e)}")


@router.get("/ehr/medical-history/{user_id}")
async def get_medical_history(
    user_id: str,
    ehr_system: str = Query(...)
):
    """Obtiene historial médico"""
    try:
        history = ehr_integration.get_medical_history(user_id, ehr_system)
        return JSONResponse(content=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {str(e)}")



