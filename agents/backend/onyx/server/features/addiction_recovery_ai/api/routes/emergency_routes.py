"""
Emergency routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.emergency_service import EmergencyService
except ImportError:
    from ...services.emergency_service import EmergencyService

router = APIRouter()

emergency = EmergencyService()


@router.post("/emergency/contact")
async def create_emergency_contact(
    user_id: str = Body(...),
    name: str = Body(...),
    relationship: str = Body(...),
    phone: str = Body(...),
    email: Optional[str] = Body(None),
    is_primary: bool = Body(False)
):
    """Crea un contacto de emergencia"""
    try:
        contact = emergency.create_emergency_contact(
            user_id, name, relationship, phone, email, is_primary
        )
        return JSONResponse(content=contact)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando contacto: {str(e)}")


@router.get("/emergency/contacts/{user_id}")
async def get_emergency_contacts(user_id: str):
    """Obtiene contactos de emergencia del usuario"""
    try:
        contacts = emergency.get_emergency_contacts(user_id)
        return JSONResponse(content={
            "user_id": user_id,
            "contacts": contacts,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo contactos: {str(e)}")


@router.post("/emergency/trigger")
async def trigger_emergency(
    user_id: str = Body(...),
    risk_level: str = Body(...),
    situation: str = Body(...)
):
    """Activa protocolo de emergencia"""
    try:
        protocol = emergency.trigger_emergency_protocol(user_id, risk_level, situation)
        return JSONResponse(content=protocol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activando protocolo: {str(e)}")


@router.get("/emergency/resources")
async def get_crisis_resources(location: Optional[str] = Query(None)):
    """Obtiene recursos de crisis"""
    try:
        resources = emergency.get_crisis_resources(location)
        return JSONResponse(content={
            "resources": resources,
            "location": location,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recursos: {str(e)}")



