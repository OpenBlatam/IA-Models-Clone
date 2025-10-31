"""
API Routes for Infinite Creation Engine
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.infinite_creation_engine import (
    infinite_creation_engine,
    CreationLevel,
    ExistenceType,
    GenerationType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/infinite-creation", tags=["Infinite Creation"])


# Request Models
class InfiniteCreationFieldRequest(BaseModel):
    name: str = Field(..., description="Name of the infinite creation field")
    level: str = Field(..., description="Infinite creation level")
    power: float = Field(..., ge=0.0, le=10000.0, description="Field power")
    radius: float = Field(..., ge=0.0, le=1000000.0, description="Field radius")
    duration: float = Field(..., ge=0.0, le=86400.0, description="Field duration in seconds")
    effects: List[str] = Field(..., description="List of infinite creation effects")


class ExistenceManagementRequest(BaseModel):
    name: str = Field(..., description="Name of the existence portal")
    source_existence: str = Field(..., description="Source existence")
    target_existence: str = Field(..., description="Target existence")
    existence_type: str = Field(..., description="Type of existence")
    stability_target: float = Field(..., ge=0.0, le=1.0, description="Target stability")


class EntityGenerationRequest(BaseModel):
    name: str = Field(..., description="Name of the entity")
    generation_type: str = Field(..., description="Type of generation")
    properties: Dict[str, Any] = Field(..., description="Entity properties")
    capabilities: List[str] = Field(..., description="Entity capabilities")


# Infinite Creation Routes
@router.post("/create-infinite-field")
async def create_infinite_field(request: InfiniteCreationFieldRequest):
    """Create an infinite creation field"""
    try:
        # Validate creation level
        try:
            level = CreationLevel(request.level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid creation level: {request.level}")
        
        result = await infinite_creation_engine.create_infinite_field(
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
            "creation_level": result.creation_level.value,
            "entities_created": result.entities_created,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating infinite creation field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manage-existence")
async def manage_existence(request: ExistenceManagementRequest):
    """Manage existence and create portals"""
    try:
        # Validate existence type
        try:
            existence_type = ExistenceType(request.existence_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid existence type: {request.existence_type}")
        
        result = await infinite_creation_engine.manage_existence(
            name=request.name,
            source_existence=request.source_existence,
            target_existence=request.target_existence,
            existence_type=existence_type,
            stability_target=request.stability_target
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "existence_type": result.existence_type.value,
            "portals_created": result.portals_created,
            "stability_achieved": result.stability_achieved,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error managing existence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-entity")
async def generate_entity(request: EntityGenerationRequest):
    """Generate infinite entities"""
    try:
        # Validate generation type
        try:
            generation_type = GenerationType(request.generation_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid generation type: {request.generation_type}")
        
        result = await infinite_creation_engine.generate_entity(
            name=request.name,
            generation_type=generation_type,
            properties=request.properties,
            capabilities=request.capabilities
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "generation_type": result.generation_type.value,
            "entities_generated": result.entities_generated,
            "energy_consumed": result.energy_consumed,
            "generation_time": result.generation_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_infinite_status():
    """Get current infinite creation status"""
    try:
        status = await infinite_creation_engine.get_infinite_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting infinite creation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/creation-fields")
async def get_creation_fields():
    """Get all creation fields"""
    try:
        fields = await infinite_creation_engine.get_creation_fields()
        return {"creation_fields": fields}
        
    except Exception as e:
        logger.error(f"Error getting creation fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/existence-portals")
async def get_existence_portals():
    """Get all existence portals"""
    try:
        portals = await infinite_creation_engine.get_existence_portals()
        return {"existence_portals": portals}
        
    except Exception as e:
        logger.error(f"Error getting existence portals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generated-entities")
async def get_generated_entities():
    """Get all generated entities"""
    try:
        entities = await infinite_creation_engine.get_generated_entities()
        return {"generated_entities": entities}
        
    except Exception as e:
        logger.error(f"Error getting generated entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/creation-history")
async def get_creation_history():
    """Get infinite creation history"""
    try:
        history = await infinite_creation_engine.get_creation_history()
        return {"creation_history": history}
        
    except Exception as e:
        logger.error(f"Error getting creation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/existence-history")
async def get_existence_history():
    """Get existence management history"""
    try:
        history = await infinite_creation_engine.get_existence_history()
        return {"existence_history": history}
        
    except Exception as e:
        logger.error(f"Error getting existence history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generation-history")
async def get_generation_history():
    """Get generation history"""
    try:
        history = await infinite_creation_engine.get_generation_history()
        return {"generation_history": history}
        
    except Exception as e:
        logger.error(f"Error getting generation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    """Get infinite creation capabilities"""
    try:
        capabilities = await infinite_creation_engine.get_capabilities()
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for Infinite Creation Engine"""
    try:
        status = await infinite_creation_engine.get_infinite_status()
        return {
            "status": "healthy",
            "engine": "Infinite Creation Engine",
            "creation_fields": status["creation_fields"],
            "existence_portals": status["existence_portals"],
            "generated_entities": status["generated_entities"],
            "infinite_energy": status["infinite_energy"],
            "timestamp": status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

















