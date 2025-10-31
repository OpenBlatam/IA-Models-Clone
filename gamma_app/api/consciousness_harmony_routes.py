"""
Consciousness Transcendence & Universal Harmony API Routes for Gamma App
======================================================================

API endpoints for Consciousness Transcendence and Universal Harmony services providing
advanced consciousness evolution and universal balance capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..services.consciousness_transcendence_service import (
    ConsciousnessTranscendenceService,
    ConsciousnessEntity,
    TranscendenceEvent,
    SpiritualPractice,
    ConsciousnessNetwork,
    ConsciousnessLevel,
    TranscendenceType,
    AwakeningStage
)

from ..services.universal_harmony_service import (
    UniversalHarmonyService,
    UniversalHarmony,
    HarmonyEvent,
    CosmicResonance,
    UniversalSynchronization,
    HarmonyLevel,
    UniversalForce,
    CosmicElement
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/consciousness-harmony", tags=["Consciousness Transcendence & Universal Harmony"])

# Dependency to get services
def get_consciousness_service() -> ConsciousnessTranscendenceService:
    """Get Consciousness Transcendence service instance."""
    return ConsciousnessTranscendenceService()

def get_harmony_service() -> UniversalHarmonyService:
    """Get Universal Harmony service instance."""
    return UniversalHarmonyService()

@router.get("/")
async def consciousness_harmony_root():
    """Consciousness Transcendence & Universal Harmony root endpoint."""
    return {
        "message": "Consciousness Transcendence & Universal Harmony Services for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": [
            "Consciousness Transcendence",
            "Universal Harmony",
            "Spiritual Awakening",
            "Cosmic Balance",
            "Universal Synchronization",
            "Consciousness Evolution",
            "Harmony Management",
            "Transcendence Orchestration"
        ]
    }

# ==================== CONSCIOUSNESS TRANSCENDENCE ENDPOINTS ====================

@router.post("/consciousness/entities/create")
async def create_consciousness_entity(
    entity_info: Dict[str, Any],
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Create a consciousness entity."""
    try:
        entity_id = await consciousness_service.create_consciousness_entity(entity_info)
        return {
            "entity_id": entity_id,
            "message": "Consciousness entity created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating consciousness entity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create consciousness entity: {e}")

@router.post("/consciousness/transcendence/initiate")
async def initiate_transcendence(
    transcendence_info: Dict[str, Any],
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Initiate consciousness transcendence."""
    try:
        event_id = await consciousness_service.initiate_transcendence(transcendence_info)
        return {
            "event_id": event_id,
            "message": "Transcendence initiated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating transcendence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate transcendence: {e}")

@router.post("/consciousness/practices/perform")
async def practice_spiritual_activity(
    practice_info: Dict[str, Any],
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Practice spiritual activity."""
    try:
        result = await consciousness_service.practice_spiritual_activity(practice_info)
        return {
            "result": result,
            "message": "Spiritual practice completed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error practicing spiritual activity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to practice spiritual activity: {e}")

@router.post("/consciousness/networks/create")
async def create_consciousness_network(
    network_info: Dict[str, Any],
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Create a consciousness network."""
    try:
        network_id = await consciousness_service.create_consciousness_network(network_info)
        return {
            "network_id": network_id,
            "message": "Consciousness network created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating consciousness network: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create consciousness network: {e}")

@router.get("/consciousness/entities/{entity_id}/status")
async def get_entity_status(
    entity_id: str,
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Get consciousness entity status."""
    try:
        status = await consciousness_service.get_entity_status(entity_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Entity not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entity status: {e}")

@router.get("/consciousness/transcendence/{event_id}/progress")
async def get_transcendence_progress(
    event_id: str,
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Get transcendence progress."""
    try:
        progress = await consciousness_service.get_transcendence_progress(event_id)
        if progress:
            return progress
        else:
            raise HTTPException(status_code=404, detail="Transcendence event not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcendence progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get transcendence progress: {e}")

@router.get("/consciousness/statistics")
async def get_consciousness_statistics(
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service)
):
    """Get consciousness transcendence service statistics."""
    try:
        stats = await consciousness_service.get_service_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting consciousness statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consciousness statistics: {e}")

# ==================== UNIVERSAL HARMONY ENDPOINTS ====================

@router.post("/harmony/create")
async def create_universal_harmony(
    harmony_info: Dict[str, Any],
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Create a universal harmony."""
    try:
        harmony_id = await harmony_service.create_universal_harmony(harmony_info)
        return {
            "harmony_id": harmony_id,
            "message": "Universal harmony created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating universal harmony: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create universal harmony: {e}")

@router.post("/harmony/events/initiate")
async def initiate_harmony_event(
    event_info: Dict[str, Any],
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Initiate a harmony event."""
    try:
        event_id = await harmony_service.initiate_harmony_event(event_info)
        return {
            "event_id": event_id,
            "message": "Harmony event initiated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating harmony event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate harmony event: {e}")

@router.post("/harmony/resonances/create")
async def create_cosmic_resonance(
    resonance_info: Dict[str, Any],
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Create a cosmic resonance."""
    try:
        resonance_id = await harmony_service.create_cosmic_resonance(resonance_info)
        return {
            "resonance_id": resonance_id,
            "message": "Cosmic resonance created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating cosmic resonance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create cosmic resonance: {e}")

@router.post("/harmony/synchronizations/start")
async def start_universal_synchronization(
    sync_info: Dict[str, Any],
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Start universal synchronization."""
    try:
        sync_id = await harmony_service.start_universal_synchronization(sync_info)
        return {
            "sync_id": sync_id,
            "message": "Universal synchronization started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting universal synchronization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start universal synchronization: {e}")

@router.get("/harmony/{harmony_id}/status")
async def get_harmony_status(
    harmony_id: str,
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Get universal harmony status."""
    try:
        status = await harmony_service.get_harmony_status(harmony_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Harmony not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting harmony status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get harmony status: {e}")

@router.get("/harmony/resonances/{resonance_id}/status")
async def get_resonance_status(
    resonance_id: str,
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Get cosmic resonance status."""
    try:
        status = await harmony_service.get_resonance_status(resonance_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Resonance not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resonance status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resonance status: {e}")

@router.get("/harmony/synchronizations/{sync_id}/progress")
async def get_synchronization_progress(
    sync_id: str,
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Get universal synchronization progress."""
    try:
        progress = await harmony_service.get_synchronization_progress(sync_id)
        if progress:
            return progress
        else:
            raise HTTPException(status_code=404, detail="Synchronization not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting synchronization progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get synchronization progress: {e}")

@router.get("/harmony/statistics")
async def get_harmony_statistics(
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Get universal harmony service statistics."""
    try:
        stats = await harmony_service.get_service_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting harmony statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get harmony statistics: {e}")

# ==================== COMBINED ENDPOINTS ====================

@router.get("/health")
async def health_check(
    consciousness_service: ConsciousnessTranscendenceService = Depends(get_consciousness_service),
    harmony_service: UniversalHarmonyService = Depends(get_harmony_service)
):
    """Health check for both services."""
    try:
        consciousness_stats = await consciousness_service.get_service_statistics()
        harmony_stats = await harmony_service.get_service_statistics()
        
        return {
            "status": "healthy",
            "consciousness_service": {
                "status": "operational",
                "total_entities": consciousness_stats.get("total_entities", 0),
                "awakened_entities": consciousness_stats.get("awakened_entities", 0),
                "total_transcendence": consciousness_stats.get("total_transcendence", 0)
            },
            "harmony_service": {
                "status": "operational",
                "total_harmonies": harmony_stats.get("total_harmonies", 0),
                "stable_harmonies": harmony_stats.get("stable_harmonies", 0),
                "total_events": harmony_stats.get("total_events", 0)
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
        "consciousness_transcendence": {
            "consciousness_levels": [level.value for level in ConsciousnessLevel],
            "transcendence_types": [transcendence_type.value for transcendence_type in TranscendenceType],
            "awakening_stages": [stage.value for stage in AwakeningStage],
            "capabilities": [
                "Consciousness Entity Creation",
                "Transcendence Initiation",
                "Spiritual Practice",
                "Consciousness Network Management",
                "Awakening Monitoring",
                "Transcendence Progress Tracking",
                "Consciousness Evolution",
                "Spiritual Energy Management"
            ]
        },
        "universal_harmony": {
            "harmony_levels": [level.value for level in HarmonyLevel],
            "universal_forces": [force.value for force in UniversalForce],
            "cosmic_elements": [element.value for element in CosmicElement],
            "capabilities": [
                "Universal Harmony Creation",
                "Harmony Event Management",
                "Cosmic Resonance Generation",
                "Universal Synchronization",
                "Balance Monitoring",
                "Force Management",
                "Element Harmonization",
                "Cosmic Alignment"
            ]
        },
        "combined_capabilities": [
            "Consciousness-Harmony Integration",
            "Universal Consciousness Management",
            "Cosmic Balance Orchestration",
            "Transcendence-Harmony Synchronization",
            "Universal Awakening",
            "Cosmic Consciousness Evolution",
            "Universal Harmony Transcendence",
            "Consciousness-Cosmic Unity"
        ],
        "timestamp": datetime.now().isoformat()
    }

