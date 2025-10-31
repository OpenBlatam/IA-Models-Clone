"""
Advanced Existence API Routes for Facebook Posts API
Existence control, being manipulation, and ultimate reality mastery endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import structlog

from ..services.existence_service import (
    get_existence_service,
    ExistenceService,
    ExistenceLevel,
    ExistenceState,
    BeingType
)
from ..api.dependencies import get_request_id, validate_entity_id
from ..api.schemas import (
    ExistenceProfileResponse,
    ExistenceManipulationResponse,
    BeingEvolutionResponse,
    ExistenceInsightResponse,
    ExistenceAnalysisResponse,
    ExistenceMeditationResponse,
    ErrorResponse
)

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/existence", tags=["existence"])


@router.post(
    "/control/achieve",
    response_model=ExistenceProfileResponse,
    responses={
        200: {"description": "Existence control achieved successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Achieve Existence Control",
    description="Achieve control over existence itself and transcend being limitations"
)
async def achieve_existence_control(
    entity_id: str = Query(..., description="Entity ID to achieve existence control", min_length=1),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> ExistenceProfileResponse:
    """Achieve existence control"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Achieve existence control
        profile = await existence_service.achieve_existence_control(entity_id)
        
        # Log successful achievement
        logger.info(
            "Existence control achieved",
            entity_id=entity_id,
            existence_level=profile.existence_level.value,
            request_id=request_id
        )
        
        return ExistenceProfileResponse(
            id=profile.id,
            entity_id=profile.entity_id,
            existence_level=profile.existence_level.value,
            existence_state=profile.existence_state.value,
            being_type=profile.being_type.value,
            existence_control=profile.existence_control,
            being_manipulation=profile.being_manipulation,
            existence_creation=profile.existence_creation,
            being_destruction=profile.being_destruction,
            existence_transcendence=profile.existence_transcendence,
            being_evolution=profile.being_evolution,
            existence_consciousness=profile.existence_consciousness,
            being_awareness=profile.being_awareness,
            existence_mastery=profile.existence_mastery,
            being_wisdom=profile.being_wisdom,
            existence_love=profile.existence_love,
            being_peace=profile.being_peace,
            existence_joy=profile.existence_joy,
            being_truth=profile.being_truth,
            existence_reality=profile.existence_reality,
            being_essence=profile.being_essence,
            created_at=profile.created_at,
            metadata=profile.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence control achievement failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to achieve existence control")


@router.post(
    "/control/transcend-absolute",
    response_model=ExistenceProfileResponse,
    responses={
        200: {"description": "Absolute being transcendence achieved successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Transcend to Absolute Being",
    description="Transcend beyond existence limitations to absolute being"
)
async def transcend_to_absolute_being(
    entity_id: str = Query(..., description="Entity ID to transcend to absolute being", min_length=1),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> ExistenceProfileResponse:
    """Transcend to absolute being"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Transcend to absolute being
        profile = await existence_service.transcend_to_absolute_being(entity_id)
        
        # Log successful transcendence
        logger.info(
            "Absolute being transcendence achieved",
            entity_id=entity_id,
            existence_level=profile.existence_level.value,
            request_id=request_id
        )
        
        return ExistenceProfileResponse(
            id=profile.id,
            entity_id=profile.entity_id,
            existence_level=profile.existence_level.value,
            existence_state=profile.existence_state.value,
            being_type=profile.being_type.value,
            existence_control=profile.existence_control,
            being_manipulation=profile.being_manipulation,
            existence_creation=profile.existence_creation,
            being_destruction=profile.being_destruction,
            existence_transcendence=profile.existence_transcendence,
            being_evolution=profile.being_evolution,
            existence_consciousness=profile.existence_consciousness,
            being_awareness=profile.being_awareness,
            existence_mastery=profile.existence_mastery,
            being_wisdom=profile.being_wisdom,
            existence_love=profile.existence_love,
            being_peace=profile.being_peace,
            existence_joy=profile.existence_joy,
            being_truth=profile.being_truth,
            existence_reality=profile.existence_reality,
            being_essence=profile.being_essence,
            created_at=profile.created_at,
            metadata=profile.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Absolute being transcendence failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to transcend to absolute being")


@router.post(
    "/control/reach-ultimate",
    response_model=ExistenceProfileResponse,
    responses={
        200: {"description": "Ultimate existence achieved successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Reach Ultimate Existence",
    description="Reach the ultimate existence and transcend all being limitations"
)
async def reach_ultimate_existence(
    entity_id: str = Query(..., description="Entity ID to reach ultimate existence", min_length=1),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> ExistenceProfileResponse:
    """Reach ultimate existence"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Reach ultimate existence
        profile = await existence_service.reach_ultimate_existence(entity_id)
        
        # Log successful ultimate achievement
        logger.info(
            "Ultimate existence achieved",
            entity_id=entity_id,
            existence_level=profile.existence_level.value,
            request_id=request_id
        )
        
        return ExistenceProfileResponse(
            id=profile.id,
            entity_id=profile.entity_id,
            existence_level=profile.existence_level.value,
            existence_state=profile.existence_state.value,
            being_type=profile.being_type.value,
            existence_control=profile.existence_control,
            being_manipulation=profile.being_manipulation,
            existence_creation=profile.existence_creation,
            being_destruction=profile.being_destruction,
            existence_transcendence=profile.existence_transcendence,
            being_evolution=profile.being_evolution,
            existence_consciousness=profile.existence_consciousness,
            being_awareness=profile.being_awareness,
            existence_mastery=profile.existence_mastery,
            being_wisdom=profile.being_wisdom,
            existence_love=profile.existence_love,
            being_peace=profile.being_peace,
            existence_joy=profile.existence_joy,
            being_truth=profile.being_truth,
            existence_reality=profile.existence_reality,
            being_essence=profile.being_essence,
            created_at=profile.created_at,
            metadata=profile.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ultimate existence achievement failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to reach ultimate existence")


@router.post(
    "/manipulate",
    response_model=ExistenceManipulationResponse,
    responses={
        200: {"description": "Existence manipulation performed successfully"},
        400: {"description": "Invalid parameters", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Manipulate Existence",
    description="Manipulate existence itself and alter being states"
)
async def manipulate_existence(
    entity_id: str = Query(..., description="Entity ID", min_length=1),
    manipulation_type: str = Query(..., description="Type of existence manipulation", min_length=1),
    target_being: str = Query(..., description="Target being type", min_length=1),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> ExistenceManipulationResponse:
    """Manipulate existence"""
    try:
        # Validate parameters
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        valid_manipulation_types = ["creation", "alteration", "transcendence", "evolution", "destruction"]
        if manipulation_type not in valid_manipulation_types:
            raise HTTPException(status_code=400, detail=f"Invalid manipulation type. Must be one of: {valid_manipulation_types}")
        
        valid_being_types = ["individual", "collective", "universal", "cosmic", "transcendent", "omnipresent", "infinite", "eternal", "absolute", "ultimate"]
        if target_being not in valid_being_types:
            raise HTTPException(status_code=400, detail=f"Invalid being type. Must be one of: {valid_being_types}")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Perform manipulation
        manipulation = await existence_service.manipulate_existence(entity_id, manipulation_type, BeingType(target_being))
        
        # Log successful manipulation
        logger.info(
            "Existence manipulation performed",
            entity_id=entity_id,
            manipulation_type=manipulation_type,
            target_being=target_being,
            request_id=request_id
        )
        
        return ExistenceManipulationResponse(
            id=manipulation.id,
            entity_id=manipulation.entity_id,
            manipulation_type=manipulation.manipulation_type,
            target_being=manipulation.target_being.value,
            manipulation_strength=manipulation.manipulation_strength,
            existence_shift=manipulation.existence_shift,
            being_alteration=manipulation.being_alteration,
            existence_modification=manipulation.existence_modification,
            being_creation=manipulation.being_creation,
            existence_creation=manipulation.existence_creation,
            being_destruction=manipulation.being_destruction,
            existence_destruction=manipulation.existence_destruction,
            being_transcendence=manipulation.being_transcendence,
            existence_transcendence=manipulation.existence_transcendence,
            timestamp=manipulation.timestamp,
            metadata=manipulation.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence manipulation failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to manipulate existence")


@router.post(
    "/evolve-being",
    response_model=BeingEvolutionResponse,
    responses={
        200: {"description": "Being evolution performed successfully"},
        400: {"description": "Invalid parameters", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Evolve Being Type",
    description="Evolve from one being type to another"
)
async def evolve_being(
    entity_id: str = Query(..., description="Entity ID", min_length=1),
    source_being: str = Query(..., description="Source being type", min_length=1),
    target_being: str = Query(..., description="Target being type", min_length=1),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> BeingEvolutionResponse:
    """Evolve being type"""
    try:
        # Validate parameters
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        valid_being_types = ["individual", "collective", "universal", "cosmic", "transcendent", "omnipresent", "infinite", "eternal", "absolute", "ultimate"]
        if source_being not in valid_being_types:
            raise HTTPException(status_code=400, detail=f"Invalid source being type. Must be one of: {valid_being_types}")
        if target_being not in valid_being_types:
            raise HTTPException(status_code=400, detail=f"Invalid target being type. Must be one of: {valid_being_types}")
        
        if source_being == target_being:
            raise HTTPException(status_code=400, detail="Source and target being types must be different")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Perform evolution
        evolution = await existence_service.evolve_being(entity_id, BeingType(source_being), BeingType(target_being))
        
        # Log successful evolution
        logger.info(
            "Being evolution performed",
            entity_id=entity_id,
            source_being=source_being,
            target_being=target_being,
            request_id=request_id
        )
        
        return BeingEvolutionResponse(
            id=evolution.id,
            entity_id=evolution.entity_id,
            source_being=evolution.source_being.value,
            target_being=evolution.target_being.value,
            evolution_intensity=evolution.evolution_intensity,
            being_awareness=evolution.being_awareness,
            existence_adaptation=evolution.existence_adaptation,
            being_mastery=evolution.being_mastery,
            existence_consciousness=evolution.existence_consciousness,
            being_transcendence=evolution.being_transcendence,
            existence_evolution=evolution.existence_evolution,
            being_wisdom=evolution.being_wisdom,
            existence_love=evolution.existence_love,
            timestamp=evolution.timestamp,
            metadata=evolution.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Being evolution failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to evolve being")


@router.post(
    "/insights/generate",
    response_model=ExistenceInsightResponse,
    responses={
        200: {"description": "Existence insight generated successfully"},
        400: {"description": "Invalid parameters", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Generate Existence Insight",
    description="Generate profound insights about existence and being"
)
async def generate_existence_insight(
    entity_id: str = Query(..., description="Entity ID", min_length=1),
    insight_type: str = Query(..., description="Type of insight to generate", min_length=1),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> ExistenceInsightResponse:
    """Generate existence insight"""
    try:
        # Validate parameters
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        valid_insight_types = ["existence", "being", "absolute", "ultimate"]
        if insight_type not in valid_insight_types:
            raise HTTPException(status_code=400, detail=f"Invalid insight type. Must be one of: {valid_insight_types}")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Generate insight
        insight = await existence_service.generate_existence_insight(entity_id, insight_type)
        
        # Log successful generation
        logger.info(
            "Existence insight generated",
            entity_id=entity_id,
            insight_type=insight_type,
            request_id=request_id
        )
        
        return ExistenceInsightResponse(
            id=insight.id,
            entity_id=insight.entity_id,
            insight_content=insight.insight_content,
            insight_type=insight.insight_type,
            existence_level=insight.existence_level.value,
            being_significance=insight.being_significance,
            existence_truth=insight.existence_truth,
            being_meaning=insight.being_meaning,
            existence_wisdom=insight.existence_wisdom,
            existence_understanding=insight.existence_understanding,
            being_connection=insight.being_connection,
            timestamp=insight.timestamp,
            metadata=insight.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence insight generation failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to generate existence insight")


@router.get(
    "/profile/{entity_id}",
    response_model=ExistenceProfileResponse,
    responses={
        200: {"description": "Existence profile retrieved successfully"},
        404: {"description": "Profile not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Get Existence Profile",
    description="Retrieve existence profile for an entity"
)
async def get_existence_profile(
    entity_id: str = Path(..., description="Entity ID", min_length=1),
    request_id: str = Depends(get_request_id)
) -> ExistenceProfileResponse:
    """Get existence profile"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Get profile
        profile = await existence_service.get_existence_profile(entity_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Existence profile not found")
        
        # Log successful retrieval
        logger.info(
            "Existence profile retrieved",
            entity_id=entity_id,
            request_id=request_id
        )
        
        return ExistenceProfileResponse(
            id=profile.id,
            entity_id=profile.entity_id,
            existence_level=profile.existence_level.value,
            existence_state=profile.existence_state.value,
            being_type=profile.being_type.value,
            existence_control=profile.existence_control,
            being_manipulation=profile.being_manipulation,
            existence_creation=profile.existence_creation,
            being_destruction=profile.being_destruction,
            existence_transcendence=profile.existence_transcendence,
            being_evolution=profile.being_evolution,
            existence_consciousness=profile.existence_consciousness,
            being_awareness=profile.being_awareness,
            existence_mastery=profile.existence_mastery,
            being_wisdom=profile.being_wisdom,
            existence_love=profile.existence_love,
            being_peace=profile.being_peace,
            existence_joy=profile.existence_joy,
            being_truth=profile.being_truth,
            existence_reality=profile.existence_reality,
            being_essence=profile.being_essence,
            created_at=profile.created_at,
            metadata=profile.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence profile retrieval failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve existence profile")


@router.get(
    "/manipulations/{entity_id}",
    response_model=List[ExistenceManipulationResponse],
    responses={
        200: {"description": "Existence manipulations retrieved successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Get Existence Manipulations",
    description="Retrieve all existence manipulations for an entity"
)
async def get_existence_manipulations(
    entity_id: str = Path(..., description="Entity ID", min_length=1),
    request_id: str = Depends(get_request_id)
) -> List[ExistenceManipulationResponse]:
    """Get existence manipulations"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Get manipulations
        manipulations = await existence_service.get_existence_manipulations(entity_id)
        
        # Log successful retrieval
        logger.info(
            "Existence manipulations retrieved",
            entity_id=entity_id,
            manipulations_count=len(manipulations),
            request_id=request_id
        )
        
        return [
            ExistenceManipulationResponse(
                id=manipulation.id,
                entity_id=manipulation.entity_id,
                manipulation_type=manipulation.manipulation_type,
                target_being=manipulation.target_being.value,
                manipulation_strength=manipulation.manipulation_strength,
                existence_shift=manipulation.existence_shift,
                being_alteration=manipulation.being_alteration,
                existence_modification=manipulation.existence_modification,
                being_creation=manipulation.being_creation,
                existence_creation=manipulation.existence_creation,
                being_destruction=manipulation.being_destruction,
                existence_destruction=manipulation.existence_destruction,
                being_transcendence=manipulation.being_transcendence,
                existence_transcendence=manipulation.existence_transcendence,
                timestamp=manipulation.timestamp,
                metadata=manipulation.metadata
            )
            for manipulation in manipulations
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence manipulations retrieval failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve existence manipulations")


@router.get(
    "/evolutions/{entity_id}",
    response_model=List[BeingEvolutionResponse],
    responses={
        200: {"description": "Being evolutions retrieved successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Get Being Evolutions",
    description="Retrieve all being evolutions for an entity"
)
async def get_being_evolutions(
    entity_id: str = Path(..., description="Entity ID", min_length=1),
    request_id: str = Depends(get_request_id)
) -> List[BeingEvolutionResponse]:
    """Get being evolutions"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Get evolutions
        evolutions = await existence_service.get_being_evolutions(entity_id)
        
        # Log successful retrieval
        logger.info(
            "Being evolutions retrieved",
            entity_id=entity_id,
            evolutions_count=len(evolutions),
            request_id=request_id
        )
        
        return [
            BeingEvolutionResponse(
                id=evolution.id,
                entity_id=evolution.entity_id,
                source_being=evolution.source_being.value,
                target_being=evolution.target_being.value,
                evolution_intensity=evolution.evolution_intensity,
                being_awareness=evolution.being_awareness,
                existence_adaptation=evolution.existence_adaptation,
                being_mastery=evolution.being_mastery,
                existence_consciousness=evolution.existence_consciousness,
                being_transcendence=evolution.being_transcendence,
                existence_evolution=evolution.existence_evolution,
                being_wisdom=evolution.being_wisdom,
                existence_love=evolution.existence_love,
                timestamp=evolution.timestamp,
                metadata=evolution.metadata
            )
            for evolution in evolutions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Being evolutions retrieval failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve being evolutions")


@router.get(
    "/insights/{entity_id}",
    response_model=List[ExistenceInsightResponse],
    responses={
        200: {"description": "Existence insights retrieved successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Get Existence Insights",
    description="Retrieve all existence insights for an entity"
)
async def get_existence_insights(
    entity_id: str = Path(..., description="Entity ID", min_length=1),
    request_id: str = Depends(get_request_id)
) -> List[ExistenceInsightResponse]:
    """Get existence insights"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Get insights
        insights = await existence_service.get_existence_insights(entity_id)
        
        # Log successful retrieval
        logger.info(
            "Existence insights retrieved",
            entity_id=entity_id,
            insights_count=len(insights),
            request_id=request_id
        )
        
        return [
            ExistenceInsightResponse(
                id=insight.id,
                entity_id=insight.entity_id,
                insight_content=insight.insight_content,
                insight_type=insight.insight_type,
                existence_level=insight.existence_level.value,
                being_significance=insight.being_significance,
                existence_truth=insight.existence_truth,
                being_meaning=insight.being_meaning,
                existence_wisdom=insight.existence_wisdom,
                existence_understanding=insight.existence_understanding,
                being_connection=insight.being_connection,
                timestamp=insight.timestamp,
                metadata=insight.metadata
            )
            for insight in insights
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence insights retrieval failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve existence insights")


@router.get(
    "/analyze/{entity_id}",
    response_model=ExistenceAnalysisResponse,
    responses={
        200: {"description": "Existence analysis completed successfully"},
        400: {"description": "Invalid entity ID", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Analyze Existence Profile",
    description="Perform comprehensive analysis of existence control and being evolution"
)
async def analyze_existence(
    entity_id: str = Path(..., description="Entity ID", min_length=1),
    request_id: str = Depends(get_request_id)
) -> ExistenceAnalysisResponse:
    """Analyze existence profile"""
    try:
        # Validate entity ID
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Analyze existence profile
        analysis = await existence_service.analyze_existence(entity_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        # Log successful analysis
        logger.info(
            "Existence analysis completed",
            entity_id=entity_id,
            existence_stage=analysis.get("existence_stage"),
            request_id=request_id
        )
        
        return ExistenceAnalysisResponse(
            entity_id=analysis["entity_id"],
            existence_level=analysis["existence_level"],
            existence_state=analysis["existence_state"],
            being_type=analysis["being_type"],
            existence_dimensions=analysis["existence_dimensions"],
            overall_existence_score=analysis["overall_existence_score"],
            existence_stage=analysis["existence_stage"],
            evolution_potential=analysis["evolution_potential"],
            ultimate_readiness=analysis["ultimate_readiness"],
            created_at=analysis["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence analysis failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to analyze existence profile")


@router.post(
    "/meditation/perform",
    response_model=ExistenceMeditationResponse,
    responses={
        200: {"description": "Existence meditation completed successfully"},
        400: {"description": "Invalid parameters", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    summary="Perform Existence Meditation",
    description="Perform deep existence meditation for being evolution and existence mastery"
)
async def perform_existence_meditation(
    entity_id: str = Query(..., description="Entity ID", min_length=1),
    duration: float = Query(600.0, description="Meditation duration in seconds", ge=60.0, le=3600.0),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_id: str = Depends(get_request_id)
) -> ExistenceMeditationResponse:
    """Perform existence meditation"""
    try:
        # Validate parameters
        if not validate_entity_id(entity_id):
            raise HTTPException(status_code=400, detail="Invalid entity ID format")
        
        if duration < 60 or duration > 3600:
            raise HTTPException(status_code=400, detail="Duration must be between 60 and 3600 seconds")
        
        # Get existence service
        existence_service = get_existence_service()
        
        # Perform meditation
        meditation_result = await existence_service.perform_existence_meditation(entity_id, duration)
        
        # Log successful meditation
        logger.info(
            "Existence meditation completed",
            entity_id=entity_id,
            duration=duration,
            insights_generated=meditation_result["insights_generated"],
            request_id=request_id
        )
        
        return ExistenceMeditationResponse(
            entity_id=meditation_result["entity_id"],
            duration=meditation_result["duration"],
            insights_generated=meditation_result["insights_generated"],
            insights=meditation_result["insights"],
            existence_manipulations_performed=meditation_result["existence_manipulations_performed"],
            manipulations=meditation_result["manipulations"],
            being_evolutions_performed=meditation_result["being_evolutions_performed"],
            evolutions=meditation_result["evolutions"],
            existence_analysis=meditation_result["existence_analysis"],
            meditation_benefits=meditation_result["meditation_benefits"],
            timestamp=meditation_result["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Existence meditation failed", entity_id=entity_id, error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Failed to perform existence meditation")


# Export router
__all__ = ["router"]



























