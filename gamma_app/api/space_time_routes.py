"""
Space Exploration & Time Travel API Routes for Gamma App
========================================================

API endpoints for Space Exploration and Time Travel services providing
advanced space mission management and temporal manipulation capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..services.space_exploration_service import (
    SpaceExplorationService,
    Spacecraft,
    SpaceMission,
    SpaceData,
    SpaceResource,
    SpacecraftType,
    MissionType,
    CelestialBody
)

from ..services.time_travel_service import (
    TimeTravelService,
    Timeline,
    TimeTravelEvent,
    TemporalAnomaly,
    TemporalParadox,
    TimeTravelType,
    TimelineStability,
    TemporalEventType
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/space-time", tags=["Space Exploration & Time Travel"])

# Dependency to get services
def get_space_service() -> SpaceExplorationService:
    """Get Space Exploration service instance."""
    return SpaceExplorationService()

def get_time_service() -> TimeTravelService:
    """Get Time Travel service instance."""
    return TimeTravelService()

@router.get("/")
async def space_time_root():
    """Space Exploration & Time Travel root endpoint."""
    return {
        "message": "Space Exploration & Time Travel Services for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": [
            "Space Exploration",
            "Time Travel",
            "Satellite Management",
            "Space Mission Planning",
            "Temporal Manipulation",
            "Timeline Management",
            "Paradox Prevention",
            "Interplanetary Communication"
        ]
    }

# ==================== SPACE EXPLORATION ENDPOINTS ====================

@router.post("/space/spacecraft/register")
async def register_spacecraft(
    spacecraft_info: Dict[str, Any],
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Register a new spacecraft."""
    try:
        spacecraft_id = await space_service.register_spacecraft(spacecraft_info)
        return {
            "spacecraft_id": spacecraft_id,
            "message": "Spacecraft registered successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering spacecraft: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register spacecraft: {e}")

@router.post("/space/missions/create")
async def create_space_mission(
    mission_info: Dict[str, Any],
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Create a new space mission."""
    try:
        mission_id = await space_service.create_space_mission(mission_info)
        return {
            "mission_id": mission_id,
            "message": "Space mission created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating space mission: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create space mission: {e}")

@router.post("/space/data/collect")
async def collect_space_data(
    data_info: Dict[str, Any],
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Collect space exploration data."""
    try:
        data_id = await space_service.collect_space_data(data_info)
        return {
            "data_id": data_id,
            "message": "Space data collected successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error collecting space data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect space data: {e}")

@router.post("/space/resources/discover")
async def discover_space_resource(
    resource_info: Dict[str, Any],
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Discover a new space resource."""
    try:
        resource_id = await space_service.discover_space_resource(resource_info)
        return {
            "resource_id": resource_id,
            "message": "Space resource discovered successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error discovering space resource: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to discover space resource: {e}")

@router.get("/space/spacecraft/{spacecraft_id}/status")
async def get_spacecraft_status(
    spacecraft_id: str,
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Get spacecraft status."""
    try:
        status = await space_service.get_spacecraft_status(spacecraft_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Spacecraft not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting spacecraft status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get spacecraft status: {e}")

@router.get("/space/missions/{mission_id}/progress")
async def get_mission_progress(
    mission_id: str,
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Get mission progress."""
    try:
        progress = await space_service.get_mission_progress(mission_id)
        if progress:
            return progress
        else:
            raise HTTPException(status_code=404, detail="Mission not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mission progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get mission progress: {e}")

@router.get("/space/statistics")
async def get_space_statistics(
    space_service: SpaceExplorationService = Depends(get_space_service)
):
    """Get space exploration service statistics."""
    try:
        stats = await space_service.get_space_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting space statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get space statistics: {e}")

# ==================== TIME TRAVEL ENDPOINTS ====================

@router.post("/time/timelines/create")
async def create_timeline(
    timeline_info: Dict[str, Any],
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Create a new timeline."""
    try:
        timeline_id = await time_service.create_timeline(timeline_info)
        return {
            "timeline_id": timeline_id,
            "message": "Timeline created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create timeline: {e}")

@router.post("/time/travel/initiate")
async def initiate_time_travel(
    travel_info: Dict[str, Any],
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Initiate time travel."""
    try:
        event_id = await time_service.initiate_time_travel(travel_info)
        return {
            "event_id": event_id,
            "message": "Time travel initiated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating time travel: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate time travel: {e}")

@router.post("/time/anomalies/detect")
async def detect_temporal_anomaly(
    anomaly_info: Dict[str, Any],
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Detect a temporal anomaly."""
    try:
        anomaly_id = await time_service.detect_temporal_anomaly(anomaly_info)
        return {
            "anomaly_id": anomaly_id,
            "message": "Temporal anomaly detected successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error detecting temporal anomaly: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect temporal anomaly: {e}")

@router.post("/time/paradoxes/{paradox_id}/resolve")
async def resolve_temporal_paradox(
    paradox_id: str,
    resolution_info: Dict[str, Any],
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Resolve a temporal paradox."""
    try:
        success = await time_service.resolve_temporal_paradox(paradox_id, resolution_info)
        return {
            "paradox_id": paradox_id,
            "resolution_success": success,
            "message": f"Paradox resolution {'successful' if success else 'failed'}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resolving temporal paradox: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve temporal paradox: {e}")

@router.get("/time/timelines/{timeline_id}/status")
async def get_timeline_status(
    timeline_id: str,
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Get timeline status."""
    try:
        status = await time_service.get_timeline_status(timeline_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Timeline not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting timeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline status: {e}")

@router.get("/time/travelers/{traveler_id}/history")
async def get_time_travel_history(
    traveler_id: str,
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Get time travel history for a traveler."""
    try:
        history = await time_service.get_time_travel_history(traveler_id)
        return {
            "traveler_id": traveler_id,
            "history": history,
            "total_events": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting time travel history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get time travel history: {e}")

@router.get("/time/statistics")
async def get_temporal_statistics(
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Get temporal service statistics."""
    try:
        stats = await time_service.get_temporal_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting temporal statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get temporal statistics: {e}")

# ==================== COMBINED ENDPOINTS ====================

@router.get("/health")
async def health_check(
    space_service: SpaceExplorationService = Depends(get_space_service),
    time_service: TimeTravelService = Depends(get_time_service)
):
    """Health check for both services."""
    try:
        space_stats = await space_service.get_space_statistics()
        time_stats = await time_service.get_temporal_statistics()
        
        return {
            "status": "healthy",
            "space_service": {
                "status": "operational",
                "total_spacecraft": space_stats.get("total_spacecraft", 0),
                "total_missions": space_stats.get("total_missions", 0),
                "total_data_points": space_stats.get("total_data_points", 0)
            },
            "time_service": {
                "status": "operational",
                "total_timelines": time_stats.get("total_timelines", 0),
                "total_travel_events": time_stats.get("total_travel_events", 0),
                "total_paradoxes": time_stats.get("total_paradoxes", 0)
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
        "space_exploration": {
            "spacecraft_types": [spacecraft_type.value for spacecraft_type in SpacecraftType],
            "mission_types": [mission_type.value for mission_type in MissionType],
            "celestial_bodies": [celestial_body.value for celestial_body in CelestialBody],
            "capabilities": [
                "Spacecraft Registration",
                "Mission Planning",
                "Data Collection",
                "Resource Discovery",
                "Interplanetary Communication",
                "Satellite Management",
                "Space Mission Execution",
                "Celestial Body Exploration"
            ]
        },
        "time_travel": {
            "travel_types": [travel_type.value for travel_type in TimeTravelType],
            "timeline_stability": [stability.value for stability in TimelineStability],
            "temporal_events": [event_type.value for event_type in TemporalEventType],
            "capabilities": [
                "Timeline Creation",
                "Time Travel",
                "Temporal Anomaly Detection",
                "Paradox Resolution",
                "Timeline Management",
                "Temporal Manipulation",
                "Causal Loop Prevention",
                "Temporal Stability Monitoring"
            ]
        },
        "combined_capabilities": [
            "Space-Time Continuum Management",
            "Interdimensional Travel",
            "Temporal Space Missions",
            "Paradox-Free Exploration",
            "Timeline-Safe Space Travel",
            "Temporal Resource Management",
            "Space-Time Anomaly Detection",
            "Multidimensional Communication"
        ],
        "timestamp": datetime.now().isoformat()
    }


