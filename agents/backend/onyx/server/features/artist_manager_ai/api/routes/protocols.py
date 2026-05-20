"""
Protocols API Routes
====================

Endpoints para gestión de protocolos.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

router = APIRouter(prefix="/protocols", tags=["protocols"])


class ProtocolCreate(BaseModel):
    title: str
    description: str
    category: str
    priority: str
    rules: List[str]
    do_s: List[str] = None
    dont_s: List[str] = None
    context: Optional[str] = None
    applicable_events: List[str] = None
    notes: Optional[str] = None


class ProtocolComplianceCreate(BaseModel):
    is_compliant: bool
    event_id: Optional[str] = None
    notes: Optional[str] = None
    violations: List[str] = None


def get_artist_manager(artist_id: str):
    """Dependency para obtener ArtistManager."""
    from ...core.artist_manager import ArtistManager
    from ...core.protocol_manager import Protocol, ProtocolCategory, ProtocolPriority
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    manager = ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key)
    return manager, Protocol, ProtocolCategory, ProtocolPriority


@router.post("/{artist_id}", response_model=Dict[str, Any])
async def create_protocol(artist_id: str, protocol: ProtocolCreate):
    """Crear nuevo protocolo."""
    try:
        manager, Protocol, ProtocolCategory, ProtocolPriority = get_artist_manager(artist_id)
        
        import uuid
        protocol_id = str(uuid.uuid4())
        
        category = ProtocolCategory(protocol.category) if protocol.category in [c.value for c in ProtocolCategory] else ProtocolCategory.GENERAL
        priority = ProtocolPriority(protocol.priority) if protocol.priority in [p.value for p in ProtocolPriority] else ProtocolPriority.MEDIUM
        
        protocol_obj = Protocol(
            id=protocol_id,
            title=protocol.title,
            description=protocol.description,
            category=category,
            priority=priority,
            rules=protocol.rules,
            do_s=protocol.do_s or [],
            dont_s=protocol.dont_s or [],
            context=protocol.context,
            applicable_events=protocol.applicable_events or [],
            notes=protocol.notes
        )
        
        created_protocol = manager.protocols.add_protocol(protocol_obj)
        return created_protocol.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}", response_model=List[Dict[str, Any]])
async def get_protocols(
    artist_id: str,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    event_id: Optional[str] = None
):
    """Obtener protocolos."""
    try:
        manager, _, ProtocolCategory, ProtocolPriority = get_artist_manager(artist_id)
        
        if event_id:
            protocols = manager.protocols.get_protocols_for_event(event_id)
        elif category:
            cat = ProtocolCategory(category)
            protocols = manager.protocols.get_protocols_by_category(cat)
        elif priority:
            pri = ProtocolPriority(priority)
            protocols = manager.protocols.get_protocols_by_priority(pri)
        else:
            protocols = manager.protocols.get_all_protocols()
        
        return [p.to_dict() for p in protocols]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/{protocol_id}", response_model=Dict[str, Any])
async def get_protocol(artist_id: str, protocol_id: str):
    """Obtener protocolo específico."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        protocol = manager.protocols.get_protocol(protocol_id)
        if not protocol:
            raise HTTPException(status_code=404, detail="Protocol not found")
        return protocol.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{artist_id}/{protocol_id}", response_model=Dict[str, Any])
async def update_protocol(artist_id: str, protocol_id: str, protocol_update: Dict[str, Any]):
    """Actualizar protocolo."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        updated_protocol = manager.protocols.update_protocol(protocol_id, **protocol_update)
        return updated_protocol.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{artist_id}/{protocol_id}")
async def delete_protocol(artist_id: str, protocol_id: str):
    """Eliminar protocolo."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        success = manager.protocols.delete_protocol(protocol_id)
        if not success:
            raise HTTPException(status_code=404, detail="Protocol not found")
        return {"status": "deleted", "protocol_id": protocol_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/{protocol_id}/compliance", response_model=Dict[str, Any])
async def record_compliance(artist_id: str, protocol_id: str, compliance: ProtocolComplianceCreate):
    """Registrar cumplimiento de protocolo."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        compliance_record = manager.protocols.record_compliance(
            protocol_id=protocol_id,
            is_compliant=compliance.is_compliant,
            event_id=compliance.event_id,
            notes=compliance.notes,
            violations=compliance.violations or []
        )
        return compliance_record.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/events/{event_id}/check-compliance", response_model=Dict[str, Any])
async def check_event_compliance(artist_id: str, event_id: str):
    """Verificar cumplimiento de protocolos para un evento usando IA."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        compliance_report = await manager.check_protocol_compliance(event_id)
        return compliance_report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/{protocol_id}/compliance-rate", response_model=Dict[str, Any])
async def get_compliance_rate(artist_id: str, protocol_id: str):
    """Obtener tasa de cumplimiento de protocolo."""
    try:
        manager, _, _, _ = get_artist_manager(artist_id)
        rate = manager.protocols.get_compliance_rate(protocol_id)
        return {
            "protocol_id": protocol_id,
            "compliance_rate": rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




