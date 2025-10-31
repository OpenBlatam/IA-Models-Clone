"""
API Routes for Omniverse Engine
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.omniverse_engine import (
    omniverse_engine,
    OmniverseLevel,
    MultiverseType,
    RealityCreationType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/omniverse", tags=["Omniverse"])


# Request Models
class OmniverseFieldRequest(BaseModel):
    name: str = Field(..., description="Name of the omniverse field")
    level: str = Field(..., description="Omniverse manipulation level")
    power: float = Field(..., ge=0.0, le=1000.0, description="Field power")
    radius: float = Field(..., ge=0.0, le=100000.0, description="Field radius")
    duration: float = Field(..., ge=0.0, le=86400.0, description="Field duration in seconds")
    effects: List[str] = Field(..., description="List of omniverse effects")


class MultiverseManagementRequest(BaseModel):
    name: str = Field(..., description="Name of the multiverse portal")
    source_multiverse: str = Field(..., description="Source multiverse")
    target_multiverse: str = Field(..., description="Target multiverse")
    portal_type: str = Field(..., description="Type of multiverse portal")
    stability_target: float = Field(..., ge=0.0, le=1.0, description="Target stability")


class RealityCreationRequest(BaseModel):
    name: str = Field(..., description="Name of the reality")
    reality_type: str = Field(..., description="Type of reality")
    dimensions: int = Field(..., ge=1, le=26, description="Number of dimensions")
    laws_of_physics: Dict[str, Any] = Field(..., description="Laws of physics")
    constants: Dict[str, float] = Field(..., description="Physical constants")


# Omniverse Routes
@router.post("/create-omniverse-field")
async def create_omniverse_field(request: OmniverseFieldRequest):
    """Create an omniverse manipulation field"""
    try:
        # Validate omniverse level
        try:
            level = OmniverseLevel(request.level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid omniverse level: {request.level}")
        
        result = await omniverse_engine.create_omniverse_field(
            name=request.name,
            level=level,
            power=request.power,
            radius=request.radius,
            duration=request.duration,
            effects=request.effects
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "omniverse_level": result.omniverse_level.value,
            "affected_area": result.affected_area,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating omniverse field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manage-multiverse")
async def manage_multiverse(request: MultiverseManagementRequest):
    """Manage multiverses and create portals"""
    try:
        # Validate multiverse type
        try:
            portal_type = MultiverseType(request.portal_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid multiverse type: {request.portal_type}")
        
        result = await omniverse_engine.manage_multiverse(
            name=request.name,
            source_multiverse=request.source_multiverse,
            target_multiverse=request.target_multiverse,
            portal_type=portal_type,
            stability_target=request.stability_target
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "multiverse_type": result.multiverse_type.value,
            "portals_created": result.portals_created,
            "stability_achieved": result.stability_achieved,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error managing multiverse: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-reality")
async def create_reality(request: RealityCreationRequest):
    """Create a new reality"""
    try:
        # Validate reality type
        try:
            reality_type = RealityCreationType(request.reality_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid reality type: {request.reality_type}")
        
        result = await omniverse_engine.create_reality(
            name=request.name,
            reality_type=reality_type,
            dimensions=request.dimensions,
            laws_of_physics=request.laws_of_physics,
            constants=request.constants
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "reality_id": result.reality_id,
            "reality_type": result.reality_type.value,
            "dimensions_created": result.dimensions_created,
            "laws_established": result.laws_established,
            "entities_created": result.entities_created,
            "energy_consumed": result.energy_consumed,
            "creation_time": result.creation_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating reality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_omniverse_status():
    """Get current omniverse status"""
    try:
        status = await omniverse_engine.get_omniverse_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting omniverse status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/omniverse-fields")
async def get_omniverse_fields():
    """Get all omniverse fields"""
    try:
        fields = await omniverse_engine.get_omniverse_fields()
        return {"omniverse_fields": fields}
        
    except Exception as e:
        logger.error(f"Error getting omniverse fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multiverse-portals")
async def get_multiverse_portals():
    """Get all multiverse portals"""
    try:
        portals = await omniverse_engine.get_multiverse_portals()
        return {"multiverse_portals": portals}
        
    except Exception as e:
        logger.error(f"Error getting multiverse portals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realities")
async def get_realities():
    """Get all created realities"""
    try:
        realities = await omniverse_engine.get_realities()
        return {"realities": realities}
        
    except Exception as e:
        logger.error(f"Error getting realities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/omniverse-history")
async def get_omniverse_history():
    """Get omniverse manipulation history"""
    try:
        history = await omniverse_engine.get_omniverse_history()
        return {"omniverse_history": history}
        
    except Exception as e:
        logger.error(f"Error getting omniverse history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multiverse-history")
async def get_multiverse_history():
    """Get multiverse management history"""
    try:
        history = await omniverse_engine.get_multiverse_history()
        return {"multiverse_history": history}
        
    except Exception as e:
        logger.error(f"Error getting multiverse history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reality-history")
async def get_reality_history():
    """Get reality creation history"""
    try:
        history = await omniverse_engine.get_reality_history()
        return {"reality_history": history}
        
    except Exception as e:
        logger.error(f"Error getting reality history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    """Get omniverse capabilities"""
    try:
        capabilities = await omniverse_engine.get_capabilities()
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for Omniverse Engine"""
    try:
        status = await omniverse_engine.get_omniverse_status()
        return {
            "status": "healthy",
            "engine": "Omniverse Engine",
            "omniverse_fields": status["omniverse_fields"],
            "multiverse_portals": status["multiverse_portals"],
            "realities": status["realities"],
            "omniverse_energy": status["omniverse_energy"],
            "timestamp": status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

















