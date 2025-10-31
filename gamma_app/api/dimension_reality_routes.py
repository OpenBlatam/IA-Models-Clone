"""
Dimension Hopping & Reality Engine API Routes for Gamma App
==========================================================

API endpoints for Dimension Hopping and Reality Engine services providing
advanced interdimensional travel and reality manipulation capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..services.dimension_hopping_service import (
    DimensionHoppingService,
    Dimension,
    DimensionHop,
    RealityAnomaly,
    InterdimensionalEntity,
    DimensionType,
    RealityLevel,
    HoppingMethod
)

from ..services.reality_engine_service import (
    RealityEngineService,
    RealityInstance,
    UniverseSimulation,
    RealityManipulation,
    ExistenceEntity,
    RealityType,
    UniverseState,
    ExistenceLevel
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/dimension-reality", tags=["Dimension Hopping & Reality Engine"])

# Dependency to get services
def get_dimension_service() -> DimensionHoppingService:
    """Get Dimension Hopping service instance."""
    return DimensionHoppingService()

def get_reality_service() -> RealityEngineService:
    """Get Reality Engine service instance."""
    return RealityEngineService()

@router.get("/")
async def dimension_reality_root():
    """Dimension Hopping & Reality Engine root endpoint."""
    return {
        "message": "Dimension Hopping & Reality Engine Services for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": [
            "Dimension Hopping",
            "Reality Engine",
            "Interdimensional Travel",
            "Reality Manipulation",
            "Universe Simulation",
            "Existence Management",
            "Parallel Universe Access",
            "Reality Anomaly Detection"
        ]
    }

# ==================== DIMENSION HOPPING ENDPOINTS ====================

@router.post("/dimension/register")
async def register_dimension(
    dimension_info: Dict[str, Any],
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Register a new dimension."""
    try:
        dimension_id = await dimension_service.register_dimension(dimension_info)
        return {
            "dimension_id": dimension_id,
            "message": "Dimension registered successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering dimension: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register dimension: {e}")

@router.post("/dimension/hop/initiate")
async def initiate_dimension_hop(
    hop_info: Dict[str, Any],
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Initiate dimension hopping."""
    try:
        hop_id = await dimension_service.initiate_dimension_hop(hop_info)
        return {
            "hop_id": hop_id,
            "message": "Dimension hop initiated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating dimension hop: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate dimension hop: {e}")

@router.post("/dimension/anomalies/detect")
async def detect_reality_anomaly(
    anomaly_info: Dict[str, Any],
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Detect a reality anomaly."""
    try:
        anomaly_id = await dimension_service.detect_reality_anomaly(anomaly_info)
        return {
            "anomaly_id": anomaly_id,
            "message": "Reality anomaly detected successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error detecting reality anomaly: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect reality anomaly: {e}")

@router.post("/dimension/entities/register")
async def register_interdimensional_entity(
    entity_info: Dict[str, Any],
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Register an interdimensional entity."""
    try:
        entity_id = await dimension_service.register_interdimensional_entity(entity_info)
        return {
            "entity_id": entity_id,
            "message": "Interdimensional entity registered successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering interdimensional entity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register interdimensional entity: {e}")

@router.get("/dimension/{dimension_id}/info")
async def get_dimension_info(
    dimension_id: str,
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Get dimension information."""
    try:
        info = await dimension_service.get_dimension_info(dimension_id)
        if info:
            return info
        else:
            raise HTTPException(status_code=404, detail="Dimension not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dimension info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dimension info: {e}")

@router.get("/dimension/travelers/{traveler_id}/status")
async def get_traveler_status(
    traveler_id: str,
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Get traveler status."""
    try:
        status = await dimension_service.get_traveler_status(traveler_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Traveler not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting traveler status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get traveler status: {e}")

@router.get("/dimension/statistics")
async def get_dimension_statistics(
    dimension_service: DimensionHoppingService = Depends(get_dimension_service)
):
    """Get dimension hopping service statistics."""
    try:
        stats = await dimension_service.get_dimension_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting dimension statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dimension statistics: {e}")

# ==================== REALITY ENGINE ENDPOINTS ====================

@router.post("/reality/create")
async def create_reality(
    reality_info: Dict[str, Any],
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Create a new reality instance."""
    try:
        reality_id = await reality_service.create_reality(reality_info)
        return {
            "reality_id": reality_id,
            "message": "Reality created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating reality: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create reality: {e}")

@router.post("/reality/simulations/start")
async def start_universe_simulation(
    simulation_info: Dict[str, Any],
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Start a universe simulation."""
    try:
        simulation_id = await reality_service.start_universe_simulation(simulation_info)
        return {
            "simulation_id": simulation_id,
            "message": "Universe simulation started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting universe simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start universe simulation: {e}")

@router.post("/reality/manipulate")
async def manipulate_reality(
    manipulation_info: Dict[str, Any],
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Manipulate reality."""
    try:
        manipulation_id = await reality_service.manipulate_reality(manipulation_info)
        return {
            "manipulation_id": manipulation_id,
            "message": "Reality manipulation executed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error manipulating reality: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to manipulate reality: {e}")

@router.post("/reality/entities/create")
async def create_existence_entity(
    entity_info: Dict[str, Any],
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Create an existence entity."""
    try:
        entity_id = await reality_service.create_existence_entity(entity_info)
        return {
            "entity_id": entity_id,
            "message": "Existence entity created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating existence entity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create existence entity: {e}")

@router.get("/reality/{reality_id}/status")
async def get_reality_status(
    reality_id: str,
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Get reality status."""
    try:
        status = await reality_service.get_reality_status(reality_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Reality not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reality status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reality status: {e}")

@router.get("/reality/simulations/{simulation_id}/progress")
async def get_simulation_progress(
    simulation_id: str,
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Get simulation progress."""
    try:
        progress = await reality_service.get_simulation_progress(simulation_id)
        if progress:
            return progress
        else:
            raise HTTPException(status_code=404, detail="Simulation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get simulation progress: {e}")

@router.get("/reality/statistics")
async def get_reality_statistics(
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Get reality engine service statistics."""
    try:
        stats = await reality_service.get_reality_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting reality statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reality statistics: {e}")

# ==================== COMBINED ENDPOINTS ====================

@router.get("/health")
async def health_check(
    dimension_service: DimensionHoppingService = Depends(get_dimension_service),
    reality_service: RealityEngineService = Depends(get_reality_service)
):
    """Health check for both services."""
    try:
        dimension_stats = await dimension_service.get_dimension_statistics()
        reality_stats = await reality_service.get_reality_statistics()
        
        return {
            "status": "healthy",
            "dimension_service": {
                "status": "operational",
                "total_dimensions": dimension_stats.get("total_dimensions", 0),
                "total_hops": dimension_stats.get("total_hops", 0),
                "total_anomalies": dimension_stats.get("total_anomalies", 0)
            },
            "reality_service": {
                "status": "operational",
                "total_realities": reality_stats.get("total_realities", 0),
                "total_simulations": reality_stats.get("total_simulations", 0),
                "total_manipulations": reality_stats.get("total_manipulations", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities of both services."""
    return {
        "dimension_hopping": {
            "dimension_types": [dimension_type.value for dimension_type in DimensionType],
            "reality_levels": [reality_level.value for reality_level in RealityLevel],
            "hopping_methods": [hopping_method.value for hopping_method in HoppingMethod],
            "capabilities": [
                "Dimension Registration",
                "Interdimensional Travel",
                "Reality Anomaly Detection",
                "Entity Management",
                "Parallel Universe Access",
                "Dimensional Stability Monitoring",
                "Reality Bridge Creation",
                "Consciousness Transfer"
            ]
        },
        "reality_engine": {
            "reality_types": [reality_type.value for reality_type in RealityType],
            "universe_states": [universe_state.value for universe_state in UniverseState],
            "existence_levels": [existence_level.value for existence_level in ExistenceLevel],
            "capabilities": [
                "Reality Creation",
                "Universe Simulation",
                "Reality Manipulation",
                "Existence Entity Management",
                "Physical Law Modification",
                "Consciousness Engineering",
                "Temporal Manipulation",
                "Existence Transcendence"
            ]
        },
        "combined_capabilities": [
            "Multidimensional Reality Management",
            "Interdimensional Consciousness Transfer",
            "Reality-Dimension Synchronization",
            "Universal Existence Orchestration",
            "Transdimensional Entity Communication",
            "Reality-Dimension Bridge Creation",
            "Multiverse Simulation",
            "Existence Transcendence"
        ],
        "timestamp": datetime.now().isoformat()
    }

