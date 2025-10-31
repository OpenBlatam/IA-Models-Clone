"""
API Routes for Transcendence Technology Engine
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.transcendence_technology_engine import (
    transcendence_technology_engine,
    TranscendenceLevel,
    EnlightenmentStage,
    AscensionType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcendence", tags=["Transcendence Technology"])


# Request Models
class TranscendenceFieldRequest(BaseModel):
    name: str = Field(..., description="Name of the transcendence field")
    level: str = Field(..., description="Transcendence level")
    power: float = Field(..., ge=0.0, le=100.0, description="Field power")
    radius: float = Field(..., ge=0.0, le=10000.0, description="Field radius")
    duration: float = Field(..., ge=0.0, le=86400.0, description="Field duration in seconds")
    effects: List[str] = Field(..., description="List of transcendence effects")


class EnlightenmentRequest(BaseModel):
    name: str = Field(..., description="Name of the enlightenment process")
    stage: str = Field(..., description="Enlightenment stage")
    techniques: List[str] = Field(..., description="List of enlightenment techniques")
    duration: float = Field(..., ge=0.0, le=86400.0, description="Process duration in seconds")


class AscensionPortalRequest(BaseModel):
    name: str = Field(..., description="Name of the ascension portal")
    ascension_type: str = Field(..., description="Type of ascension")
    destination: str = Field(..., description="Ascension destination")
    energy_required: float = Field(..., ge=0.0, le=100000.0, description="Energy required")
    stability_target: float = Field(..., ge=0.0, le=1.0, description="Target stability")


# Transcendence Technology Routes
@router.post("/create-transcendence-field")
async def create_transcendence_field(request: TranscendenceFieldRequest):
    """Create a transcendence field"""
    try:
        # Validate transcendence level
        try:
            level = TranscendenceLevel(request.level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid transcendence level: {request.level}")
        
        result = await transcendence_technology_engine.create_transcendence_field(
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
            "transcendence_level": result.transcendence_level.value,
            "power_achieved": result.power_achieved,
            "energy_consumed": result.energy_consumed,
            "side_effects": result.side_effects,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating transcendence field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initiate-enlightenment")
async def initiate_enlightenment(request: EnlightenmentRequest):
    """Initiate enlightenment process"""
    try:
        # Validate enlightenment stage
        try:
            stage = EnlightenmentStage(request.stage)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid enlightenment stage: {request.stage}")
        
        result = await transcendence_technology_engine.initiate_enlightenment(
            name=request.name,
            stage=stage,
            techniques=request.techniques,
            duration=request.duration
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "enlightenment_stage": result.enlightenment_stage.value,
            "progress_achieved": result.progress_achieved,
            "techniques_used": result.techniques_used,
            "energy_consumed": result.energy_consumed,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initiating enlightenment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-ascension-portal")
async def create_ascension_portal(request: AscensionPortalRequest):
    """Create ascension portal"""
    try:
        # Validate ascension type
        try:
            ascension_type = AscensionType(request.ascension_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid ascension type: {request.ascension_type}")
        
        result = await transcendence_technology_engine.create_ascension_portal(
            name=request.name,
            ascension_type=ascension_type,
            destination=request.destination,
            energy_required=request.energy_required,
            stability_target=request.stability_target
        )
        
        return {
            "success": result.success,
            "operation_id": result.operation_id,
            "ascension_type": result.ascension_type.value,
            "destination_reached": result.destination_reached,
            "energy_consumed": result.energy_consumed,
            "stability_achieved": result.stability_achieved,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating ascension portal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_transcendence_status():
    """Get current transcendence status"""
    try:
        status = await transcendence_technology_engine.get_transcendence_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting transcendence status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcendence-fields")
async def get_transcendence_fields():
    """Get all transcendence fields"""
    try:
        fields = await transcendence_technology_engine.get_transcendence_fields()
        return {"transcendence_fields": fields}
        
    except Exception as e:
        logger.error(f"Error getting transcendence fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enlightenment-processes")
async def get_enlightenment_processes():
    """Get all enlightenment processes"""
    try:
        processes = await transcendence_technology_engine.get_enlightenment_processes()
        return {"enlightenment_processes": processes}
        
    except Exception as e:
        logger.error(f"Error getting enlightenment processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ascension-portals")
async def get_ascension_portals():
    """Get all ascension portals"""
    try:
        portals = await transcendence_technology_engine.get_ascension_portals()
        return {"ascension_portals": portals}
        
    except Exception as e:
        logger.error(f"Error getting ascension portals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcendence-history")
async def get_transcendence_history():
    """Get transcendence history"""
    try:
        history = await transcendence_technology_engine.get_transcendence_history()
        return {"transcendence_history": history}
        
    except Exception as e:
        logger.error(f"Error getting transcendence history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enlightenment-history")
async def get_enlightenment_history():
    """Get enlightenment history"""
    try:
        history = await transcendence_technology_engine.get_enlightenment_history()
        return {"enlightenment_history": history}
        
    except Exception as e:
        logger.error(f"Error getting enlightenment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ascension-history")
async def get_ascension_history():
    """Get ascension history"""
    try:
        history = await transcendence_technology_engine.get_ascension_history()
        return {"ascension_history": history}
        
    except Exception as e:
        logger.error(f"Error getting ascension history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    """Get transcendence capabilities"""
    try:
        capabilities = await transcendence_technology_engine.get_capabilities()
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for Transcendence Technology Engine"""
    try:
        status = await transcendence_technology_engine.get_transcendence_status()
        return {
            "status": "healthy",
            "engine": "Transcendence Technology Engine",
            "transcendence_fields": status["transcendence_fields"],
            "enlightenment_processes": status["enlightenment_processes"],
            "ascension_portals": status["ascension_portals"],
            "cosmic_energy": status["cosmic_energy"],
            "timestamp": status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

















