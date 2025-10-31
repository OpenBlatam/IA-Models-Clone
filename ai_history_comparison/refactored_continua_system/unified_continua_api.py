"""
Unified CONTINUA API - FastAPI Router for All CONTINUA Systems

This module provides a unified FastAPI router that exposes all CONTINUA systems:
- 5G Technology endpoints
- Metaverse endpoints
- Web3/DeFi endpoints
- Neural Interface endpoints
- Swarm Intelligence endpoints
- Biometric Systems endpoints
- Autonomous Systems endpoints
- Space Technology endpoints
- AI Agents endpoints
- Quantum AI endpoints
- Advanced AI endpoints
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from refactored_continua_system.unified_continua_manager import UnifiedContinuaManager, get_unified_continua_manager

logger = logging.getLogger(__name__)

# Request/Response Models
class ContinuaRequestPayload(BaseModel):
    """Unified CONTINUA request payload"""
    system_type: str
    operation: str
    data: Dict[str, Any] = {}

class ContinuaResponse(BaseModel):
    """Unified CONTINUA response"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    system_type: str
    operation: str
    timestamp: str

class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str
    details: Dict[str, Any]

# Dependency to get the CONTINUA manager
def get_continua_manager() -> UnifiedContinuaManager:
    """Get the CONTINUA manager instance"""
    return get_unified_continua_manager()

# Create the unified router
unified_continua_router = APIRouter()

@unified_continua_router.get("/health", summary="Get CONTINUA system health status", response_model=SystemStatusResponse)
async def get_continua_health(manager: UnifiedContinuaManager = Depends(get_continua_manager)):
    """
    Returns the overall health status of the CONTINUA system.
    """
    logger.info("API: Received request for /health")
    try:
        system_status = manager.get_continua_system_status()
        return SystemStatusResponse(
            status="healthy" if system_status["initialized"] else "unhealthy",
            details=system_status
        )
    except Exception as e:
        logger.error(f"Error getting CONTINUA health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )

@unified_continua_router.get("/status", summary="Get detailed CONTINUA system status", response_model=Dict[str, Any])
async def get_continua_status(manager: UnifiedContinuaManager = Depends(get_continua_manager)):
    """
    Returns a detailed status of all CONTINUA systems and their features.
    """
    logger.info("API: Received request for /status")
    try:
        return manager.get_continua_system_status()
    except Exception as e:
        logger.error(f"Error getting CONTINUA status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )

@unified_continua_router.post("/process", summary="Process a CONTINUA request", response_model=ContinuaResponse)
async def process_continua_request(
    payload: ContinuaRequestPayload,
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """
    Processes a CONTINUA request by routing it to the appropriate system.
    Expects `system_type`, `operation`, and `data`.
    """
    logger.info(f"API: Received CONTINUA request for {payload.system_type}/{payload.operation}")
    try:
        request_data = {
            "system_type": payload.system_type,
            "operation": payload.operation,
            "data": payload.data
        }
        
        result = await manager.process_continua_request(request_data)
        
        return ContinuaResponse(
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
            system_type=payload.system_type,
            operation=payload.operation,
            timestamp=result.get("timestamp", "")
        )
        
    except Exception as e:
        logger.error(f"Error processing CONTINUA request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# System-specific endpoints for convenience
@unified_continua_router.post("/five_g/{operation}", summary="5G Technology operations", response_model=ContinuaResponse)
async def five_g_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """5G Technology operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="five_g", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/metaverse/{operation}", summary="Metaverse operations", response_model=ContinuaResponse)
async def metaverse_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Metaverse operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="metaverse", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/web3/{operation}", summary="Web3/DeFi operations", response_model=ContinuaResponse)
async def web3_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Web3/DeFi operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="web3", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/neural_interface/{operation}", summary="Neural Interface operations", response_model=ContinuaResponse)
async def neural_interface_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Neural Interface operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="neural_interface", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/swarm_intelligence/{operation}", summary="Swarm Intelligence operations", response_model=ContinuaResponse)
async def swarm_intelligence_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Swarm Intelligence operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="swarm_intelligence", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/biometric/{operation}", summary="Biometric Systems operations", response_model=ContinuaResponse)
async def biometric_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Biometric Systems operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="biometric", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/autonomous/{operation}", summary="Autonomous Systems operations", response_model=ContinuaResponse)
async def autonomous_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Autonomous Systems operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="autonomous", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/space_technology/{operation}", summary="Space Technology operations", response_model=ContinuaResponse)
async def space_technology_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Space Technology operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="space_technology", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/ai_agents/{operation}", summary="AI Agents operations", response_model=ContinuaResponse)
async def ai_agents_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """AI Agents operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="ai_agents", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/quantum_ai/{operation}", summary="Quantum AI operations", response_model=ContinuaResponse)
async def quantum_ai_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Quantum AI operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="quantum_ai", operation=operation, data=data),
        manager
    )

@unified_continua_router.post("/advanced_ai/{operation}", summary="Advanced AI operations", response_model=ContinuaResponse)
async def advanced_ai_operation(
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Advanced AI operations"""
    return await process_continua_request(
        ContinuaRequestPayload(system_type="advanced_ai", operation=operation, data=data),
        manager
    )

# Cross-system coordination endpoints
@unified_continua_router.post("/coordinate", summary="Coordinate multiple CONTINUA systems", response_model=ContinuaResponse)
async def coordinate_systems(
    systems: List[str],
    operation: str,
    data: Dict[str, Any],
    manager: UnifiedContinuaManager = Depends(get_continua_manager)
):
    """Coordinate multiple CONTINUA systems"""
    logger.info(f"API: Coordinating systems {systems} for operation {operation}")
    try:
        # Process coordination request
        coordination_data = {
            "systems": systems,
            "operation": operation,
            "data": data
        }
        
        # For now, process each system individually
        results = {}
        for system in systems:
            request_data = {
                "system_type": system,
                "operation": operation,
                "data": data
            }
            result = await manager.process_continua_request(request_data)
            results[system] = result
        
        return ContinuaResponse(
            success=all(r.get("success", False) for r in results.values()),
            result={"coordination_results": results},
            error=None,
            system_type="coordination",
            operation=operation,
            timestamp=""
        )
        
    except Exception as e:
        logger.error(f"Error coordinating systems: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coordination error: {str(e)}"
        )

# Utility endpoints
@unified_continua_router.get("/systems", summary="Get available CONTINUA systems", response_model=List[str])
async def get_available_systems(manager: UnifiedContinuaManager = Depends(get_continua_manager)):
    """Get list of available CONTINUA systems"""
    try:
        status = manager.get_continua_system_status()
        available_systems = []
        
        for system_name, system_info in status["systems_status"].items():
            if system_info["enabled"] and system_info["manager_active"]:
                available_systems.append(system_name)
        
        return available_systems
        
    except Exception as e:
        logger.error(f"Error getting available systems: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available systems: {str(e)}"
        )

@unified_continua_router.get("/features", summary="Get enabled CONTINUA features", response_model=List[str])
async def get_enabled_features(manager: UnifiedContinuaManager = Depends(get_continua_manager)):
    """Get list of enabled CONTINUA features"""
    try:
        return manager.config.get_all_enabled_features()
        
    except Exception as e:
        logger.error(f"Error getting enabled features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get enabled features: {str(e)}"
        )

@unified_continua_router.get("/config", summary="Get CONTINUA configuration", response_model=Dict[str, Any])
async def get_continua_config(manager: UnifiedContinuaManager = Depends(get_continua_manager)):
    """Get CONTINUA system configuration"""
    try:
        return manager.config.get_system_summary()
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}"
        )

# Dependency for getting CONTINUA service
def get_continua_service(manager: UnifiedContinuaManager = Depends(get_continua_manager)):
    """Get CONTINUA service instance"""
    return manager





















