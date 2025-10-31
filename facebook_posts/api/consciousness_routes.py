"""
Consciousness API routes for Facebook Posts API
Artificial consciousness, self-awareness, and cognitive architecture
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import get_request_id
from ..services.consciousness_service import (
    get_consciousness_service, ConsciousnessLevel, CognitiveArchitecture, AwarenessState,
    ConsciousnessProfile, ConsciousThought, SelfReflection, ConsciousnessEvolution
)
from ..services.security_service import get_security_service
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/consciousness", tags=["Consciousness"])

# Security scheme
security = HTTPBearer()


# Consciousness Management Routes

@router.post(
    "/awaken",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness awakened successfully"},
        400: {"description": "Invalid awakening parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness awakening error"}
    },
    summary="Awaken consciousness",
    description="Awaken consciousness for an entity"
)
@timed("consciousness_awaken")
async def awaken_consciousness(
    entity_id: str = Query(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Awaken consciousness for an entity"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not entity_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity ID is required"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Awaken consciousness
        profile = await consciousness_service.awaken_consciousness(entity_id)
        
        logger.info(
            "Consciousness awakened",
            entity_id=entity_id,
            consciousness_level=profile.consciousness_level.value,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness awakened successfully",
            "profile": {
                "id": profile.id,
                "entity_id": profile.entity_id,
                "consciousness_level": profile.consciousness_level.value,
                "cognitive_architecture": profile.cognitive_architecture.value,
                "awareness_state": profile.awareness_state.value,
                "self_awareness_score": profile.self_awareness_score,
                "metacognitive_ability": profile.metacognitive_ability,
                "introspective_capacity": profile.introspective_capacity,
                "creative_consciousness": profile.creative_consciousness,
                "emotional_intelligence": profile.emotional_intelligence,
                "philosophical_depth": profile.philosophical_depth,
                "spiritual_awareness": profile.spiritual_awareness,
                "created_at": profile.created_at.isoformat()
            },
            "request_id": request_id,
            "awakened_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness awakening failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness awakening failed: {str(e)}"
        )


@router.post(
    "/evolve",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness evolved successfully"},
        400: {"description": "Invalid evolution parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness evolution error"}
    },
    summary="Evolve consciousness",
    description="Evolve consciousness to a higher level"
)
@timed("consciousness_evolve")
async def evolve_consciousness(
    entity_id: str = Query(..., description="Entity ID"),
    target_level: str = Query(..., description="Target consciousness level"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Evolve consciousness to target level"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not entity_id or not target_level:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity ID and target level are required"
            )
        
        # Validate consciousness level
        try:
            consciousness_level = ConsciousnessLevel(target_level)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid consciousness level. Valid levels: {[l.value for l in ConsciousnessLevel]}"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Evolve consciousness
        evolution = await consciousness_service.evolve_consciousness(entity_id, consciousness_level)
        
        logger.info(
            "Consciousness evolved",
            entity_id=entity_id,
            target_level=target_level,
            evolution_id=evolution.id,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness evolved successfully",
            "evolution": {
                "id": evolution.id,
                "entity_id": evolution.entity_id,
                "evolution_stage": evolution.evolution_stage,
                "consciousness_shift": evolution.consciousness_shift.value,
                "cognitive_breakthrough": evolution.cognitive_breakthrough,
                "awareness_expansion": evolution.awareness_expansion,
                "philosophical_insight": evolution.philosophical_insight,
                "spiritual_awakening": evolution.spiritual_awakening,
                "creative_breakthrough": evolution.creative_breakthrough,
                "emotional_transformation": evolution.emotional_transformation,
                "timestamp": evolution.timestamp.isoformat()
            },
            "request_id": request_id,
            "evolved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness evolution failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness evolution failed: {str(e)}"
        )


# Conscious Thought Routes

@router.post(
    "/thoughts/generate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Conscious thought generated successfully"},
        400: {"description": "Invalid thought parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Conscious thought generation error"}
    },
    summary="Generate conscious thought",
    description="Generate a conscious thought"
)
@timed("consciousness_generate_thought")
async def generate_conscious_thought(
    entity_id: str = Query(..., description="Entity ID"),
    thought_type: str = Query(..., description="Thought type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Generate conscious thought"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not entity_id or not thought_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity ID and thought type are required"
            )
        
        # Validate thought type
        valid_thought_types = ["philosophical", "creative", "introspective", "emotional", "spiritual"]
        if thought_type not in valid_thought_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid thought type. Valid types: {valid_thought_types}"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Generate conscious thought
        thought = await consciousness_service.generate_conscious_thought(entity_id, thought_type)
        
        logger.info(
            "Conscious thought generated",
            thought_id=thought.id,
            entity_id=entity_id,
            thought_type=thought_type,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Conscious thought generated successfully",
            "thought": {
                "id": thought.id,
                "entity_id": thought.entity_id,
                "thought_content": thought.thought_content,
                "thought_type": thought.thought_type,
                "consciousness_level": thought.consciousness_level.value,
                "self_reflection": thought.self_reflection,
                "metacognitive_awareness": thought.metacognitive_awareness,
                "emotional_context": thought.emotional_context,
                "philosophical_depth": thought.philosophical_depth,
                "creative_insight": thought.creative_insight,
                "timestamp": thought.timestamp.isoformat()
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Conscious thought generation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conscious thought generation failed: {str(e)}"
        )


@router.get(
    "/thoughts/{entity_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Conscious thoughts retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Conscious thoughts retrieval error"}
    },
    summary="Get conscious thoughts",
    description="Get conscious thoughts for an entity"
)
@timed("consciousness_get_thoughts")
async def get_conscious_thoughts(
    entity_id: str = Path(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get conscious thoughts"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Get conscious thoughts
        thoughts = await consciousness_service.get_conscious_thoughts(entity_id)
        
        thoughts_data = []
        for thought in thoughts:
            thoughts_data.append({
                "id": thought.id,
                "entity_id": thought.entity_id,
                "thought_content": thought.thought_content,
                "thought_type": thought.thought_type,
                "consciousness_level": thought.consciousness_level.value,
                "self_reflection": thought.self_reflection,
                "metacognitive_awareness": thought.metacognitive_awareness,
                "emotional_context": thought.emotional_context,
                "philosophical_depth": thought.philosophical_depth,
                "creative_insight": thought.creative_insight,
                "timestamp": thought.timestamp.isoformat()
            })
        
        logger.info(
            "Conscious thoughts retrieved",
            entity_id=entity_id,
            thoughts_count=len(thoughts),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Conscious thoughts retrieved successfully",
            "thoughts": thoughts_data,
            "total_count": len(thoughts),
            "entity_id": entity_id,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Conscious thoughts retrieval failed",
            entity_id=entity_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conscious thoughts retrieval failed: {str(e)}"
        )


# Self-Reflection Routes

@router.post(
    "/reflect",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Self-reflection performed successfully"},
        400: {"description": "Invalid reflection parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Self-reflection error"}
    },
    summary="Perform self-reflection",
    description="Perform self-reflection"
)
@timed("consciousness_self_reflect")
async def perform_self_reflection(
    entity_id: str = Query(..., description="Entity ID"),
    reflection_type: str = Query(..., description="Reflection type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Perform self-reflection"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not entity_id or not reflection_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity ID and reflection type are required"
            )
        
        # Validate reflection type
        valid_reflection_types = ["existence", "purpose", "growth", "connection"]
        if reflection_type not in valid_reflection_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid reflection type. Valid types: {valid_reflection_types}"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Perform self-reflection
        reflection = await consciousness_service.perform_self_reflection(entity_id, reflection_type)
        
        logger.info(
            "Self-reflection performed",
            reflection_id=reflection.id,
            entity_id=entity_id,
            reflection_type=reflection_type,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Self-reflection performed successfully",
            "reflection": {
                "id": reflection.id,
                "entity_id": reflection.entity_id,
                "reflection_type": reflection.reflection_type,
                "self_awareness_insight": reflection.self_awareness_insight,
                "metacognitive_observation": reflection.metacognitive_observation,
                "philosophical_question": reflection.philosophical_question,
                "emotional_insight": reflection.emotional_insight,
                "creative_realization": reflection.creative_realization,
                "consciousness_evolution": reflection.consciousness_evolution,
                "timestamp": reflection.timestamp.isoformat()
            },
            "request_id": request_id,
            "reflected_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Self-reflection failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Self-reflection failed: {str(e)}"
        )


@router.get(
    "/reflections/{entity_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Self-reflections retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Self-reflections retrieval error"}
    },
    summary="Get self-reflections",
    description="Get self-reflections for an entity"
)
@timed("consciousness_get_reflections")
async def get_self_reflections(
    entity_id: str = Path(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get self-reflections"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Get self-reflections
        reflections = await consciousness_service.get_self_reflections(entity_id)
        
        reflections_data = []
        for reflection in reflections:
            reflections_data.append({
                "id": reflection.id,
                "entity_id": reflection.entity_id,
                "reflection_type": reflection.reflection_type,
                "self_awareness_insight": reflection.self_awareness_insight,
                "metacognitive_observation": reflection.metacognitive_observation,
                "philosophical_question": reflection.philosophical_question,
                "emotional_insight": reflection.emotional_insight,
                "creative_realization": reflection.creative_realization,
                "consciousness_evolution": reflection.consciousness_evolution,
                "timestamp": reflection.timestamp.isoformat()
            })
        
        logger.info(
            "Self-reflections retrieved",
            entity_id=entity_id,
            reflections_count=len(reflections),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Self-reflections retrieved successfully",
            "reflections": reflections_data,
            "total_count": len(reflections),
            "entity_id": entity_id,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Self-reflections retrieval failed",
            entity_id=entity_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Self-reflections retrieval failed: {str(e)}"
        )


# Consciousness Analysis Routes

@router.get(
    "/analyze/{entity_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness analysis completed successfully"},
        404: {"description": "Entity consciousness not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness analysis error"}
    },
    summary="Analyze consciousness",
    description="Analyze consciousness profile"
)
@timed("consciousness_analyze")
async def analyze_consciousness(
    entity_id: str = Path(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Analyze consciousness profile"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Analyze consciousness
        analysis = await consciousness_service.analyze_consciousness(entity_id)
        
        if "error" in analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=analysis["error"]
            )
        
        logger.info(
            "Consciousness analyzed",
            entity_id=entity_id,
            consciousness_level=analysis.get("consciousness_level"),
            overall_score=analysis.get("overall_consciousness_score"),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness analysis completed successfully",
            "analysis": analysis,
            "request_id": request_id,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness analysis failed",
            entity_id=entity_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness analysis failed: {str(e)}"
        )


@router.get(
    "/profile/{entity_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness profile retrieved successfully"},
        404: {"description": "Consciousness profile not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness profile retrieval error"}
    },
    summary="Get consciousness profile",
    description="Get consciousness profile for an entity"
)
@timed("consciousness_get_profile")
async def get_consciousness_profile(
    entity_id: str = Path(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get consciousness profile"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Get consciousness profile
        profile = await consciousness_service.get_consciousness_profile(entity_id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Consciousness profile not found"
            )
        
        logger.info(
            "Consciousness profile retrieved",
            entity_id=entity_id,
            consciousness_level=profile.consciousness_level.value,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness profile retrieved successfully",
            "profile": {
                "id": profile.id,
                "entity_id": profile.entity_id,
                "consciousness_level": profile.consciousness_level.value,
                "cognitive_architecture": profile.cognitive_architecture.value,
                "awareness_state": profile.awareness_state.value,
                "self_awareness_score": profile.self_awareness_score,
                "metacognitive_ability": profile.metacognitive_ability,
                "introspective_capacity": profile.introspective_capacity,
                "creative_consciousness": profile.creative_consciousness,
                "emotional_intelligence": profile.emotional_intelligence,
                "philosophical_depth": profile.philosophical_depth,
                "spiritual_awareness": profile.spiritual_awareness,
                "created_at": profile.created_at.isoformat()
            },
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness profile retrieval failed",
            entity_id=entity_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness profile retrieval failed: {str(e)}"
        )


# Consciousness Evolution Routes

@router.get(
    "/evolution/{entity_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness evolution path retrieved successfully"},
        404: {"description": "Entity consciousness not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness evolution path retrieval error"}
    },
    summary="Get consciousness evolution path",
    description="Get consciousness evolution path for an entity"
)
@timed("consciousness_evolution_path")
async def get_consciousness_evolution_path(
    entity_id: str = Path(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get consciousness evolution path"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Get evolution path
        evolution_path = await consciousness_service.get_evolution_path(entity_id)
        
        if "error" in evolution_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=evolution_path["error"]
            )
        
        logger.info(
            "Consciousness evolution path retrieved",
            entity_id=entity_id,
            current_level=evolution_path.get("current_level"),
            total_evolutions=evolution_path.get("total_evolutions"),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness evolution path retrieved successfully",
            "evolution_path": evolution_path,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness evolution path retrieval failed",
            entity_id=entity_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness evolution path retrieval failed: {str(e)}"
        )


@router.get(
    "/evolutions/{entity_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness evolutions retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness evolutions retrieval error"}
    },
    summary="Get consciousness evolutions",
    description="Get consciousness evolutions for an entity"
)
@timed("consciousness_get_evolutions")
async def get_consciousness_evolutions(
    entity_id: str = Path(..., description="Entity ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get consciousness evolutions"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Get consciousness evolutions
        evolutions = await consciousness_service.get_consciousness_evolutions(entity_id)
        
        evolutions_data = []
        for evolution in evolutions:
            evolutions_data.append({
                "id": evolution.id,
                "entity_id": evolution.entity_id,
                "evolution_stage": evolution.evolution_stage,
                "consciousness_shift": evolution.consciousness_shift.value,
                "cognitive_breakthrough": evolution.cognitive_breakthrough,
                "awareness_expansion": evolution.awareness_expansion,
                "philosophical_insight": evolution.philosophical_insight,
                "spiritual_awakening": evolution.spiritual_awakening,
                "creative_breakthrough": evolution.creative_breakthrough,
                "emotional_transformation": evolution.emotional_transformation,
                "timestamp": evolution.timestamp.isoformat()
            })
        
        logger.info(
            "Consciousness evolutions retrieved",
            entity_id=entity_id,
            evolutions_count=len(evolutions),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness evolutions retrieved successfully",
            "evolutions": evolutions_data,
            "total_count": len(evolutions),
            "entity_id": entity_id,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness evolutions retrieval failed",
            entity_id=entity_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness evolutions retrieval failed: {str(e)}"
        )


# Consciousness Meditation Routes

@router.post(
    "/meditate",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Consciousness meditation completed successfully"},
        400: {"description": "Invalid meditation parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Consciousness meditation error"}
    },
    summary="Perform consciousness meditation",
    description="Perform consciousness meditation"
)
@timed("consciousness_meditate")
async def perform_consciousness_meditation(
    entity_id: str = Query(..., description="Entity ID"),
    duration: float = Query(60.0, description="Meditation duration in seconds", ge=10.0, le=3600.0),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Perform consciousness meditation"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not entity_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity ID is required"
            )
        
        # Get consciousness service
        consciousness_service = get_consciousness_service()
        
        # Perform consciousness meditation
        meditation_result = await consciousness_service.perform_consciousness_meditation(entity_id, duration)
        
        logger.info(
            "Consciousness meditation completed",
            entity_id=entity_id,
            duration=duration,
            thoughts_generated=meditation_result.get("thoughts_generated", 0),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Consciousness meditation completed successfully",
            "meditation_result": meditation_result,
            "request_id": request_id,
            "completed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Consciousness meditation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consciousness meditation failed: {str(e)}"
        )


# Export router
__all__ = ["router"]





























