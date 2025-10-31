"""
FastAPI Routes for Email Sequence System

This module provides comprehensive API endpoints for managing email sequences,
following FastAPI best practices with proper async/await patterns, error handling,
and dependency injection.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .schemas import (
    SequenceCreateRequest,
    SequenceUpdateRequest,
    SequenceResponse,
    SequenceListResponse,
    SequenceSearchRequest,
    CampaignCreateRequest,
    CampaignResponse,
    TemplateCreateRequest,
    TemplateResponse,
    SubscriberCreateRequest,
    SubscriberResponse,
    BulkSubscriberCreateRequest,
    BulkOperationResponse,
    AnalyticsResponse,
    HealthCheckResponse,
    ErrorResponse
)
from ..core.email_sequence_engine import EmailSequenceEngine
from ..services.langchain_service import LangChainEmailService
from ..services.delivery_service import EmailDeliveryService
from ..services.analytics_service import EmailAnalyticsService
from ..core.dependencies import get_engine, get_current_user, get_database
from ..core.exceptions import (
    SequenceNotFoundError,
    InvalidSequenceError,
    SubscriberNotFoundError,
    TemplateNotFoundError,
    CampaignNotFoundError
)

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Router
email_sequence_router = APIRouter(
    prefix="/api/v1/email-sequences",
    tags=["Email Sequences"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


# Dependency functions
async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Get current user ID from JWT token"""
    # In a real implementation, you would decode the JWT token
    # and extract the user ID
    return "user_123"  # Placeholder


async def validate_sequence_access(
    sequence_id: UUID,
    user_id: str = Depends(get_current_user_id),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> None:
    """Validate user has access to sequence"""
    # Check if sequence exists and user has access
    if sequence_id not in engine.active_sequences:
        raise HTTPException(
            status_code=404,
            detail=f"Sequence {sequence_id} not found"
        )
    # Add additional access control logic here


# Health Check
@email_sequence_router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the email sequence service"
)
async def health_check(
    engine: EmailSequenceEngine = Depends(get_engine)
) -> HealthCheckResponse:
    """Health check endpoint"""
    try:
        stats = engine.get_stats()
        services = {
            "engine": "healthy" if stats["status"] == "running" else "unhealthy",
            "langchain": "healthy",  # Add actual health check
            "delivery": "healthy",   # Add actual health check
            "analytics": "healthy"   # Add actual health check
        }
        
        overall_status = "healthy" if all(
            status == "healthy" for status in services.values()
        ) else "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="2.0.0",
            services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


# Sequence Management
@email_sequence_router.post(
    "",
    response_model=SequenceResponse,
    status_code=201,
    summary="Create Email Sequence",
    description="Create a new email sequence with AI-powered content generation"
)
async def create_sequence(
    request: SequenceCreateRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> SequenceResponse:
    """Create a new email sequence"""
    try:
        # Generate sequence using LangChain
        result = await engine.create_sequence(
            name=request.name,
            target_audience=request.target_audience,
            goals=request.goals,
            tone=request.tone
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.message
            )
        
        sequence = result.data["sequence"]
        
        # Update sequence with additional request data
        sequence.description = request.description
        sequence.personalization_enabled = request.personalization_enabled
        sequence.ab_testing_enabled = request.ab_testing_enabled
        sequence.tracking_enabled = request.tracking_enabled
        sequence.tags = request.tags
        sequence.category = request.category
        sequence.priority = request.priority
        
        # Add background task for analytics setup
        background_tasks.add_task(
            setup_sequence_analytics,
            sequence.id,
            user_id
        )
        
        return SequenceResponse(
            id=sequence.id,
            name=sequence.name,
            description=sequence.description,
            status=sequence.status,
            steps=sequence.steps,
            triggers=sequence.triggers,
            personalization_enabled=sequence.personalization_enabled,
            personalization_variables=sequence.personalization_variables,
            ab_testing_enabled=sequence.ab_testing_enabled,
            ab_test_variants=sequence.ab_test_variants,
            tracking_enabled=sequence.tracking_enabled,
            conversion_tracking=sequence.conversion_tracking,
            max_duration_days=sequence.max_duration_days,
            timezone=sequence.timezone,
            tags=sequence.tags,
            category=sequence.category,
            priority=sequence.priority,
            created_at=sequence.created_at,
            updated_at=sequence.updated_at,
            activated_at=sequence.activated_at,
            completed_at=sequence.completed_at,
            total_subscribers=sequence.total_subscribers,
            active_subscribers=sequence.active_subscribers,
            completed_subscribers=sequence.completed_subscribers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating sequence: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create sequence"
        )


@email_sequence_router.get(
    "",
    response_model=SequenceListResponse,
    summary="List Email Sequences",
    description="Get a paginated list of email sequences with optional filtering"
)
async def list_sequences(
    search_request: SequenceSearchRequest = Depends(),
    user_id: str = Depends(get_current_user_id),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> SequenceListResponse:
    """List email sequences with filtering and pagination"""
    try:
        # Filter sequences based on search criteria
        sequences = list(engine.active_sequences.values())
        
        # Apply filters
        if search_request.status:
            sequences = [s for s in sequences if s.status == search_request.status]
        
        if search_request.category:
            sequences = [s for s in sequences if s.category == search_request.category]
        
        if search_request.tags:
            sequences = [
                s for s in sequences 
                if any(tag in s.tags for tag in search_request.tags)
            ]
        
        if search_request.created_after:
            sequences = [
                s for s in sequences 
                if s.created_at >= search_request.created_after
            ]
        
        if search_request.created_before:
            sequences = [
                s for s in sequences 
                if s.created_at <= search_request.created_before
            ]
        
        # Apply pagination
        total_count = len(sequences)
        start_idx = search_request.offset
        end_idx = start_idx + search_request.limit
        paginated_sequences = sequences[start_idx:end_idx]
        
        # Convert to response format
        sequence_responses = [
            SequenceResponse(
                id=seq.id,
                name=seq.name,
                description=seq.description,
                status=seq.status,
                steps=seq.steps,
                triggers=seq.triggers,
                personalization_enabled=seq.personalization_enabled,
                personalization_variables=seq.personalization_variables,
                ab_testing_enabled=seq.ab_testing_enabled,
                ab_test_variants=seq.ab_test_variants,
                tracking_enabled=seq.tracking_enabled,
                conversion_tracking=seq.conversion_tracking,
                max_duration_days=seq.max_duration_days,
                timezone=seq.timezone,
                tags=seq.tags,
                category=seq.category,
                priority=seq.priority,
                created_at=seq.created_at,
                updated_at=seq.updated_at,
                activated_at=seq.activated_at,
                completed_at=seq.completed_at,
                total_subscribers=seq.total_subscribers,
                active_subscribers=seq.active_subscribers,
                completed_subscribers=seq.completed_subscribers
            )
            for seq in paginated_sequences
        ]
        
        return SequenceListResponse(
            sequences=sequence_responses,
            total_count=total_count,
            limit=search_request.limit,
            offset=search_request.offset,
            has_more=end_idx < total_count
        )
        
    except Exception as e:
        logger.error(f"Error listing sequences: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list sequences"
        )


@email_sequence_router.get(
    "/{sequence_id}",
    response_model=SequenceResponse,
    summary="Get Email Sequence",
    description="Get a specific email sequence by ID"
)
async def get_sequence(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(validate_sequence_access)
) -> SequenceResponse:
    """Get a specific email sequence"""
    try:
        engine = await get_engine()
        sequence = engine.active_sequences[sequence_id]
        
        return SequenceResponse(
            id=sequence.id,
            name=sequence.name,
            description=sequence.description,
            status=sequence.status,
            steps=sequence.steps,
            triggers=sequence.triggers,
            personalization_enabled=sequence.personalization_enabled,
            personalization_variables=sequence.personalization_variables,
            ab_testing_enabled=sequence.ab_testing_enabled,
            ab_test_variants=sequence.ab_test_variants,
            tracking_enabled=sequence.tracking_enabled,
            conversion_tracking=sequence.conversion_tracking,
            max_duration_days=sequence.max_duration_days,
            timezone=sequence.timezone,
            tags=sequence.tags,
            category=sequence.category,
            priority=sequence.priority,
            created_at=sequence.created_at,
            updated_at=sequence.updated_at,
            activated_at=sequence.activated_at,
            completed_at=sequence.completed_at,
            total_subscribers=sequence.total_subscribers,
            active_subscribers=sequence.active_subscribers,
            completed_subscribers=sequence.completed_subscribers
        )
        
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Sequence {sequence_id} not found"
        )
    except Exception as e:
        logger.error(f"Error getting sequence {sequence_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get sequence"
        )


@email_sequence_router.put(
    "/{sequence_id}",
    response_model=SequenceResponse,
    summary="Update Email Sequence",
    description="Update an existing email sequence"
)
async def update_sequence(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    request: SequenceUpdateRequest = ...,
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(validate_sequence_access),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> SequenceResponse:
    """Update an email sequence"""
    try:
        sequence = engine.active_sequences[sequence_id]
        
        # Update fields if provided
        if request.name is not None:
            sequence.name = request.name
        if request.description is not None:
            sequence.description = request.description
        if request.status is not None:
            sequence.status = request.status
        if request.personalization_enabled is not None:
            sequence.personalization_enabled = request.personalization_enabled
        if request.tracking_enabled is not None:
            sequence.tracking_enabled = request.tracking_enabled
        if request.tags is not None:
            sequence.tags = request.tags
        if request.category is not None:
            sequence.category = request.category
        if request.priority is not None:
            sequence.priority = request.priority
        
        sequence.updated_at = datetime.utcnow()
        
        return SequenceResponse(
            id=sequence.id,
            name=sequence.name,
            description=sequence.description,
            status=sequence.status,
            steps=sequence.steps,
            triggers=sequence.triggers,
            personalization_enabled=sequence.personalization_enabled,
            personalization_variables=sequence.personalization_variables,
            ab_testing_enabled=sequence.ab_testing_enabled,
            ab_test_variants=sequence.ab_test_variants,
            tracking_enabled=sequence.tracking_enabled,
            conversion_tracking=sequence.conversion_tracking,
            max_duration_days=sequence.max_duration_days,
            timezone=sequence.timezone,
            tags=sequence.tags,
            category=sequence.category,
            priority=sequence.priority,
            created_at=sequence.created_at,
            updated_at=sequence.updated_at,
            activated_at=sequence.activated_at,
            completed_at=sequence.completed_at,
            total_subscribers=sequence.total_subscribers,
            active_subscribers=sequence.active_subscribers,
            completed_subscribers=sequence.completed_subscribers
        )
        
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Sequence {sequence_id} not found"
        )
    except Exception as e:
        logger.error(f"Error updating sequence {sequence_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update sequence"
        )


@email_sequence_router.delete(
    "/{sequence_id}",
    status_code=204,
    summary="Delete Email Sequence",
    description="Delete an email sequence"
)
async def delete_sequence(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(validate_sequence_access),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> None:
    """Delete an email sequence"""
    try:
        if sequence_id not in engine.active_sequences:
            raise HTTPException(
                status_code=404,
                detail=f"Sequence {sequence_id} not found"
            )
        
        del engine.active_sequences[sequence_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting sequence {sequence_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete sequence"
        )


# Sequence Actions
@email_sequence_router.post(
    "/{sequence_id}/activate",
    response_model=SequenceResponse,
    summary="Activate Sequence",
    description="Activate an email sequence to start sending emails"
)
async def activate_sequence(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(validate_sequence_access),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> SequenceResponse:
    """Activate an email sequence"""
    try:
        result = await engine.activate_sequence(sequence_id)
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.message
            )
        
        sequence = engine.active_sequences[sequence_id]
        
        return SequenceResponse(
            id=sequence.id,
            name=sequence.name,
            description=sequence.description,
            status=sequence.status,
            steps=sequence.steps,
            triggers=sequence.triggers,
            personalization_enabled=sequence.personalization_enabled,
            personalization_variables=sequence.personalization_variables,
            ab_testing_enabled=sequence.ab_testing_enabled,
            ab_test_variants=sequence.ab_test_variants,
            tracking_enabled=sequence.tracking_enabled,
            conversion_tracking=sequence.conversion_tracking,
            max_duration_days=sequence.max_duration_days,
            timezone=sequence.timezone,
            tags=sequence.tags,
            category=sequence.category,
            priority=sequence.priority,
            created_at=sequence.created_at,
            updated_at=sequence.updated_at,
            activated_at=sequence.activated_at,
            completed_at=sequence.completed_at,
            total_subscribers=sequence.total_subscribers,
            active_subscribers=sequence.active_subscribers,
            completed_subscribers=sequence.completed_subscribers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating sequence {sequence_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to activate sequence"
        )


@email_sequence_router.get(
    "/{sequence_id}/analytics",
    response_model=AnalyticsResponse,
    summary="Get Sequence Analytics",
    description="Get analytics data for an email sequence"
)
async def get_sequence_analytics(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(validate_sequence_access),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> AnalyticsResponse:
    """Get analytics for an email sequence"""
    try:
        result = await engine.get_sequence_analytics(sequence_id)
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.message
            )
        
        return AnalyticsResponse(
            sequence_id=sequence_id,
            metrics=result.data,
            time_range={
                "start": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
                "end": datetime.utcnow()
            },
            generated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for sequence {sequence_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get sequence analytics"
        )


# Subscriber Management
@email_sequence_router.post(
    "/{sequence_id}/subscribers",
    response_model=BulkOperationResponse,
    status_code=201,
    summary="Add Subscribers to Sequence",
    description="Add multiple subscribers to an email sequence"
)
async def add_subscribers_to_sequence(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    request: BulkSubscriberCreateRequest = ...,
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(validate_sequence_access),
    engine: EmailSequenceEngine = Depends(get_engine)
) -> BulkOperationResponse:
    """Add subscribers to an email sequence"""
    try:
        # Convert request to Subscriber objects
        subscribers = []
        for sub_data in request.subscribers:
            from ..models.subscriber import Subscriber
            subscriber = Subscriber(
                email=sub_data.email,
                first_name=sub_data.first_name,
                last_name=sub_data.last_name,
                phone=sub_data.phone,
                company=sub_data.company,
                job_title=sub_data.job_title,
                custom_fields=sub_data.custom_fields,
                tags=sub_data.tags,
                source=sub_data.source
            )
            subscribers.append(subscriber)
        
        result = await engine.add_subscribers_to_sequence(sequence_id, subscribers)
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.message
            )
        
        return BulkOperationResponse(
            total_items=len(subscribers),
            successful_items=result.data.get("subscribers_added", 0),
            failed_items=len(subscribers) - result.data.get("subscribers_added", 0),
            errors=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding subscribers to sequence {sequence_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to add subscribers to sequence"
        )


# Background Tasks
async def setup_sequence_analytics(sequence_id: UUID, user_id: str) -> None:
    """Background task to setup analytics for a sequence"""
    try:
        # Setup analytics tracking for the sequence
        logger.info(f"Setting up analytics for sequence {sequence_id}")
        # Implementation would go here
    except Exception as e:
        logger.error(f"Error setting up analytics for sequence {sequence_id}: {e}")


# Error Handlers
@email_sequence_router.exception_handler(SequenceNotFoundError)
async def sequence_not_found_handler(request, exc):
    """Handle sequence not found errors"""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error=f"Sequence {exc.sequence_id} not found",
            error_code="SEQUENCE_NOT_FOUND"
        ).dict()
    )


@email_sequence_router.exception_handler(InvalidSequenceError)
async def invalid_sequence_handler(request, exc):
    """Handle invalid sequence errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.message,
            error_code="INVALID_SEQUENCE",
            details=exc.details
        ).dict()
    )































