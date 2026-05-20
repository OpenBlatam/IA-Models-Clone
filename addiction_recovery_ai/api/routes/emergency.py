"""
Emergency services routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, Query, status

try:
    from schemas.emergency import (
        CreateEmergencyContactRequest,
        EmergencyContactResponse,
        EmergencyContactsListResponse,
        TriggerEmergencyRequest,
        EmergencyProtocolResponse,
        CrisisResourcesResponse
    )
    from schemas.common import ErrorResponse
    from dependencies import EmergencyServiceDep
except ImportError:
    from ...schemas.emergency import (
        CreateEmergencyContactRequest,
        EmergencyContactResponse,
        EmergencyContactsListResponse,
        TriggerEmergencyRequest,
        EmergencyProtocolResponse,
        CrisisResourcesResponse
    )
    from ...schemas.common import ErrorResponse
    from ...dependencies import EmergencyServiceDep

router = APIRouter(prefix="/emergency", tags=["Emergency Services"])


@router.post(
    "/contact",
    response_model=EmergencyContactResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_emergency_contact(
    request: CreateEmergencyContactRequest,
    emergency: EmergencyServiceDep
) -> EmergencyContactResponse:
    """
    Crea un contacto de emergencia
    
    - **user_id**: ID del usuario
    - **name**: Nombre del contacto
    - **relationship**: Relación con el usuario
    - **phone**: Número de teléfono
    - **email**: Email (opcional)
    - **is_primary**: Si es contacto principal
    """
    # Guard clause: Validate required fields
    if not request.name or not request.name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="name es requerido"
        )
    
    if not request.phone or not request.phone.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="phone es requerido"
        )
    
    try:
        contact_data = emergency.create_emergency_contact(
            request.user_id,
            request.name,
            request.relationship,
            request.phone,
            request.email,
            request.is_primary
        )
        
        return EmergencyContactResponse(
            contact_id=contact_data.get("contact_id", ""),
            user_id=request.user_id,
            name=request.name,
            relationship=request.relationship,
            phone=request.phone,
            email=request.email,
            is_primary=request.is_primary
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando contacto: {str(e)}"
        )


@router.get(
    "/contacts/{user_id}",
    response_model=EmergencyContactsListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_emergency_contacts(
    user_id: str,
    emergency: EmergencyServiceDep
) -> EmergencyContactsListResponse:
    """
    Obtiene contactos de emergencia del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        contacts_data = emergency.get_emergency_contacts(user_id)
        
        contacts = [
            EmergencyContactResponse(
                contact_id=c.get("contact_id", ""),
                user_id=user_id,
                name=c.get("name", ""),
                relationship=c.get("relationship", ""),
                phone=c.get("phone", ""),
                email=c.get("email"),
                is_primary=c.get("is_primary", False),
                created_at=c.get("created_at")
            )
            for c in contacts_data
        ]
        
        primary_contact = next((c for c in contacts if c.is_primary), None)
        
        return EmergencyContactsListResponse(
            user_id=user_id,
            contacts=contacts,
            primary_contact=primary_contact
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo contactos: {str(e)}"
        )


@router.post(
    "/trigger",
    response_model=EmergencyProtocolResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def trigger_emergency(
    request: TriggerEmergencyRequest,
    emergency: EmergencyServiceDep
) -> EmergencyProtocolResponse:
    """
    Activa protocolo de emergencia
    
    - **user_id**: ID del usuario
    - **risk_level**: Nivel de riesgo (low, moderate, high, critical)
    - **situation**: Descripción de la situación
    """
    # Guard clause: Validate required fields
    if not request.situation or not request.situation.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="situation es requerido"
        )
    
    try:
        protocol_data = emergency.trigger_emergency_protocol(
            request.user_id,
            request.risk_level,
            request.situation
        )
        
        return EmergencyProtocolResponse(
            protocol_id=protocol_data.get("protocol_id", ""),
            user_id=request.user_id,
            risk_level=request.risk_level,
            situation=request.situation,
            actions_taken=protocol_data.get("actions_taken", []),
            contacts_notified=protocol_data.get("contacts_notified", []),
            resources_provided=protocol_data.get("resources_provided", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activando protocolo: {str(e)}"
        )


@router.get(
    "/resources",
    response_model=CrisisResourcesResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_crisis_resources(
    location: str = Query(None, description="Location filter"),
    emergency: EmergencyServiceDep
) -> CrisisResourcesResponse:
    """
    Obtiene recursos de crisis
    
    - **location**: Ubicación para filtrar recursos (opcional)
    """
    try:
        resources_data = emergency.get_crisis_resources(location)
        
        from schemas.emergency import CrisisResourceResponse
        resources = [
            CrisisResourceResponse(
                resource_id=r.get("resource_id", ""),
                name=r.get("name", ""),
                type=r.get("type", ""),
                phone=r.get("phone"),
                website=r.get("website"),
                location=r.get("location"),
                available_24_7=r.get("available_24_7", False)
            )
            for r in resources_data
        ]
        
        return CrisisResourcesResponse(
            resources=resources,
            location=location,
            total=len(resources)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo recursos: {str(e)}"
        )

