"""
API Routes for Reality Manipulation Engine
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.reality_manipulation_engine import (
    reality_manipulation_engine,
    RealityLevel,
    DimensionType,
    UniverseType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reality", tags=["Reality Manipulation"])


# Request Models
class RealityFieldRequest(BaseModel):
    name: str = Field(..., description="Name of the reality field")
    level: str = Field(..., description="Reality manipulation level")
    strength: float = Field(..., ge=0.0, le=100.0, description="Field strength")
    radius: float = Field(..., ge=0.0, le=10000.0, description="Field radius")
    duration: float = Field(..., ge=0.0, le=86400.0, description="Field duration in seconds")
    effects: List[str] = Field(..., description="List of reality effects")


class DimensionManipulationRequest(BaseModel):
    name: str = Field(..., description="Name of the dimension portal")
    source_dimension: str = Field(..., description="Source dimension")
    target_dimension: str = Field(..., description="Target dimension")
    portal_type: str = Field(..., description="Type of dimension portal")
    stability_target: float = Field(..., ge=0.0, le=1.0, description="Target stability")


class UniverseCreationRequest(BaseModel):
    name: str = Field(..., description="Name of the universe")
    universe_type: str = Field(..., description="Type of universe")
    dimensions: int = Field(..., ge=1, le=11, description="Number of dimensions")
    laws_of_physics: Dict[str, Any] = Field(..., description="Laws of physics")
    constants: Dict[str, float] = Field(..., description="Physical constants")


# Reality Manipulation Routes
@router.post("/create-reality-field")
async def create_reality_field(request: RealityFieldRequest):
    """Create a reality manipulation field"""
    try:
        # Validate reality level
        try:
            level = RealityLevel(request.level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid reality level: {request.level}")
        
        result = await reality_manipulation_engine.create_reality_field(
            name=request.name,
            level=level,
            strength=request.strength,
            radius=request.radius,
            duration=request.duration,
            effects=request.effects
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "reality_level": result.reality_level.value,
            "affected_area": result.affected_area,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating reality field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manipulate-dimension")
async def manipulate_dimension(request: DimensionManipulationRequest):
    """Manipulate dimensions and create portals"""
    try:
        # Validate dimension type
        try:
            portal_type = DimensionType(request.portal_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid dimension type: {request.portal_type}")
        
        result = await reality_manipulation_engine.manipulate_dimension(
            name=request.name,
            source_dimension=request.source_dimension,
            target_dimension=request.target_dimension,
            portal_type=portal_type,
            stability_target=request.stability_target
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "dimension_type": result.dimension_type.value,
            "portals_created": result.portals_created,
            "stability_achieved": result.stability_achieved,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error manipulating dimension: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-universe")
async def create_universe(request: UniverseCreationRequest):
    """Create a new universe"""
    try:
        # Validate universe type
        try:
            universe_type = UniverseType(request.universe_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid universe type: {request.universe_type}")
        
        result = await reality_manipulation_engine.create_universe(
            name=request.name,
            universe_type=universe_type,
            dimensions=request.dimensions,
            laws_of_physics=request.laws_of_physics,
            constants=request.constants
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "universe_id": result.universe_id,
            "universe_type": result.universe_type.value,
            "dimensions_created": result.dimensions_created,
            "laws_established": result.laws_established,
            "entities_created": result.entities_created,
            "energy_consumed": result.energy_consumed,
            "creation_time": result.creation_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating universe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_reality_status():
    """Get current reality manipulation status"""
    try:
        status = await reality_manipulation_engine.get_reality_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting reality status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reality-fields")
async def get_reality_fields():
    """Get all reality fields"""
    try:
        fields = await reality_manipulation_engine.get_reality_fields()
        return {"reality_fields": fields}
        
    except Exception as e:
        logger.error(f"Error getting reality fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dimension-portals")
async def get_dimension_portals():
    """Get all dimension portals"""
    try:
        portals = await reality_manipulation_engine.get_dimension_portals()
        return {"dimension_portals": portals}
        
    except Exception as e:
        logger.error(f"Error getting dimension portals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/universes")
async def get_universes():
    """Get all created universes"""
    try:
        universes = await reality_manipulation_engine.get_universes()
        return {"universes": universes}
        
    except Exception as e:
        logger.error(f"Error getting universes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/manipulation-history")
async def get_manipulation_history():
    """Get reality manipulation history"""
    try:
        history = await reality_manipulation_engine.get_manipulation_history()
        return {"manipulation_history": history}
        
    except Exception as e:
        logger.error(f"Error getting manipulation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dimension-history")
async def get_dimension_history():
    """Get dimension manipulation history"""
    try:
        history = await reality_manipulation_engine.get_dimension_history()
        return {"dimension_history": history}
        
    except Exception as e:
        logger.error(f"Error getting dimension history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/universe-history")
async def get_universe_history():
    """Get universe creation history"""
    try:
        history = await reality_manipulation_engine.get_universe_history()
        return {"universe_history": history}
        
    except Exception as e:
        logger.error(f"Error getting universe history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    """Get reality manipulation capabilities"""
    try:
        capabilities = await reality_manipulation_engine.get_capabilities()
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for Reality Manipulation Engine"""
    try:
        status = await reality_manipulation_engine.get_reality_status()
        return {
            "status": "healthy",
            "engine": "Reality Manipulation Engine",
            "reality_fields": status["reality_fields"],
            "dimension_portals": status["dimension_portals"],
            "universes": status["universes"],
            "energy_reserves": status["energy_reserves"],
            "timestamp": status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

















