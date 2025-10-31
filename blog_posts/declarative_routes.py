from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
from typing import List, Optional, Dict, Any, Union, Annotated
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
from fastapi import (
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic.types import conint, constr
import structlog
from .functional_fastapi_components import (
    from .sqlalchemy_2_implementation import SQLAlchemy2Manager
        import uuid
            import psutil
            import psutil
from typing import Any, List, Dict, Optional
import logging
"""
🎯 Declarative Route Definitions with Clear Return Type Annotations
==================================================================

Declarative approach to FastAPI routes using:
- Clear return type annotations
- Functional route handlers
- Pydantic models for request/response
- Dependency injection
- Type-safe operations
- OpenAPI documentation
"""


    FastAPI, APIRouter, Depends, HTTPException, status, 
    Request, Response, BackgroundTasks, Query, Path, Body, Header,
    Form, File, UploadFile
)

    # Pydantic Models
    TextAnalysisRequest, BatchAnalysisRequest, AnalysisUpdateRequest,
    PaginationRequest, AnalysisFilterRequest,
    AnalysisResponse, BatchAnalysisResponse, PaginatedResponse,
    HealthResponse, ErrorResponse, SuccessResponse,
    
    # Enums
    AnalysisTypeEnum, OptimizationTierEnum, AnalysisStatusEnum,
    
    # Service Functions
    create_analysis_service, get_analysis_service,
    update_analysis_service, list_analyses_service, create_batch_service,
    
    # Pure Functions
    validate_text_content, calculate_processing_priority,
    estimate_processing_time, generate_cache_key
)

# ============================================================================
# Type Definitions and Annotations
# ============================================================================

# Type aliases for better readability
AnalysisID = Annotated[int, Path(description="Analysis ID", ge=1)]
BatchID = Annotated[int, Path(description="Batch ID", ge=1)]
PageNumber = Annotated[int, Query(description="Page number", ge=1, default=1)]
PageSize = Annotated[int, Query(description="Page size", ge=1, le=100, default=20)]
OrderBy = Annotated[str, Query(description="Field to order by", default="created_at")]
OrderDesc = Annotated[bool, Query(description="Descending order", default=True)]

# Dependency type annotations
DBManager = Annotated[Any, Depends()]  # Will be replaced with actual type
AuthToken = Annotated[HTTPAuthorizationCredentials, Depends(HTTPBearer())]
BackgroundTaskManager = Annotated[BackgroundTasks, Depends()]

# ============================================================================
# Route Response Types
# ============================================================================

class RouteResponse(BaseModel):
    """Base response wrapper for consistent API responses."""
    success: bool = Field(description="Operation success status")
    data: Optional[Dict[str, Any]] = Field(description="Response data")
    message: Optional[str] = Field(description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(description="Request ID for tracking")

class AnalysisListResponse(RouteResponse):
    """Response type for analysis listing endpoints."""
    data: Optional[PaginatedResponse[AnalysisResponse]] = Field(description="Paginated analysis results")

class AnalysisDetailResponse(RouteResponse):
    """Response type for analysis detail endpoints."""
    data: Optional[AnalysisResponse] = Field(description="Analysis details")

class BatchDetailResponse(RouteResponse):
    """Response type for batch analysis endpoints."""
    data: Optional[BatchAnalysisResponse] = Field(description="Batch analysis details")

class HealthCheckResponse(RouteResponse):
    """Response type for health check endpoints."""
    data: Optional[HealthResponse] = Field(description="Health check results")

class ErrorDetailResponse(RouteResponse):
    """Response type for error endpoints."""
    data: Optional[ErrorResponse] = Field(description="Error details")

# ============================================================================
# Route Decorators and Utilities
# ============================================================================

def with_response_wrapper(response_model: type):
    """Decorator to wrap responses in consistent format."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                result = await func(*args, **kwargs)
                
                # If result is already a RouteResponse, return as is
                if isinstance(result, RouteResponse):
                    return result
                
                # Wrap in appropriate response type
                return response_model(
                    success=True,
                    data=result,
                    message="Operation completed successfully"
                )
                
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                # Wrap other exceptions in error response
                return ErrorDetailResponse(
                    success=False,
                    data=ErrorResponse(
                        error=str(e),
                        error_code="INTERNAL_ERROR",
                        detail="An unexpected error occurred",
                        timestamp=datetime.now()
                    ),
                    message="Operation failed"
                )
        
        return wrapper
    return decorator

def with_request_logging():
    """Decorator to log request details."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            logger = structlog.get_logger("route_handler")
            
            # Extract request info
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if request:
                logger.info(
                    "Route request started",
                    method=request.method,
                    url=str(request.url),
                    client_ip=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent")
                )
            
            try:
                result = await func(*args, **kwargs)
                logger.info("Route request completed successfully")
                return result
            except Exception as e:
                logger.error("Route request failed", error=str(e))
                raise
        
        return wrapper
    return decorator

def with_performance_monitoring():
    """Decorator to monitor route performance."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Log performance metrics
                logger = structlog.get_logger("performance")
                logger.info(
                    "Route performance",
                    function=func.__name__,
                    processing_time_seconds=processing_time,
                    success=True
                )
                
                return result
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logger = structlog.get_logger("performance")
                logger.error(
                    "Route performance",
                    function=func.__name__,
                    processing_time_seconds=processing_time,
                    success=False,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator

# ============================================================================
# Dependency Functions
# ============================================================================

async def get_db_manager() -> Optional[Dict[str, Any]]:
    """Dependency to get database manager."""
    # In real implementation, return actual database manager
    return SQLAlchemy2Manager()

async def get_current_user(auth_token: AuthToken) -> Dict[str, Any]:
    """Dependency to get current authenticated user."""
    # In real implementation, validate token and return user
    return {
        "id": 1,
        "email": "user@example.com",
        "role": "user"
    }

async async def get_request_id(request: Request) -> str:
    """Dependency to get or generate request ID."""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id

# ============================================================================
# Declarative Route Definitions
# ============================================================================

class AnalysisRoutes:
    """Declarative route definitions for analysis endpoints."""
    
    def __init__(self, router: APIRouter):
        
    """__init__ function."""
self.router = router
        self._register_routes()
    
    def _register_routes(self) -> Any:
        """Register all analysis routes."""
        
        # POST /analyses - Create new analysis
        @self.router.post(
            "/analyses",
            response_model=AnalysisDetailResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create Text Analysis",
            description="Create a new text analysis with the specified parameters",
            response_description="Analysis created successfully",
            tags=["Analysis"]
        )
        @with_response_wrapper(AnalysisDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def create_analysis(
            request: TextAnalysisRequest = Body(
                ...,
                description="Analysis request parameters",
                example={
                    "text_content": "This is a positive text for sentiment analysis.",
                    "analysis_type": "sentiment",
                    "optimization_tier": "standard",
                    "metadata": {"source": "user_input", "priority": "high"}
                }
            ),
            background_tasks: BackgroundTaskManager = Depends(),
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> AnalysisDetailResponse:
            """
            Create a new text analysis.
            
            This endpoint creates a new text analysis with the specified parameters.
            The analysis will be processed asynchronously and results will be available
            once processing is complete.
            
            Args:
                request: Analysis request parameters
                background_tasks: Background task manager
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                AnalysisDetailResponse: Created analysis details
                
            Raises:
                HTTPException: If validation fails or database error occurs
            """
            
            # Validate request
            validation_result = validate_text_content(request.text_content)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Validation failed",
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings
                    }
                )
            
            # Create analysis
            result = await create_analysis_service(request, db_manager)
            
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.error
                )
            
            # Add background task for processing
            background_tasks.add_task(
                self._process_analysis_background,
                result.data.id,
                request.text_content,
                request.analysis_type,
                db_manager
            )
            
            return AnalysisDetailResponse(
                success=True,
                data=result.data,
                message="Analysis created successfully",
                request_id=request_id
            )
        
        # GET /analyses/{analysis_id} - Get analysis by ID
        @self.router.get(
            "/analyses/{analysis_id}",
            response_model=AnalysisDetailResponse,
            status_code=status.HTTP_200_OK,
            summary="Get Analysis by ID",
            description="Retrieve analysis details by ID",
            response_description="Analysis details",
            tags=["Analysis"]
        )
        @with_response_wrapper(AnalysisDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def get_analysis(
            analysis_id: AnalysisID,
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> AnalysisDetailResponse:
            """
            Get analysis by ID.
            
            Retrieve detailed information about a specific analysis by its ID.
            
            Args:
                analysis_id: Analysis ID
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                AnalysisDetailResponse: Analysis details
                
            Raises:
                HTTPException: If analysis not found
            """
            
            result = await get_analysis_service(analysis_id, db_manager)
            
            if not result.success:
                if "not found" in result.error.lower():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Analysis with ID {analysis_id} not found"
                    )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.error
                )
            
            return AnalysisDetailResponse(
                success=True,
                data=result.data,
                message="Analysis retrieved successfully",
                request_id=request_id
            )
        
        # PUT /analyses/{analysis_id} - Update analysis
        @self.router.put(
            "/analyses/{analysis_id}",
            response_model=AnalysisDetailResponse,
            status_code=status.HTTP_200_OK,
            summary="Update Analysis",
            description="Update analysis details by ID",
            response_description="Updated analysis details",
            tags=["Analysis"]
        )
        @with_response_wrapper(AnalysisDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def update_analysis(
            analysis_id: AnalysisID,
            update_request: AnalysisUpdateRequest = Body(
                ...,
                description="Analysis update parameters",
                example={
                    "status": "completed",
                    "sentiment_score": 0.8,
                    "quality_score": 0.95,
                    "processing_time_ms": 150.5,
                    "model_used": "distilbert-sentiment",
                    "confidence_score": 0.92
                }
            ),
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> AnalysisDetailResponse:
            """
            Update analysis by ID.
            
            Update the details of a specific analysis by its ID.
            
            Args:
                analysis_id: Analysis ID
                update_request: Analysis update parameters
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                AnalysisDetailResponse: Updated analysis details
                
            Raises:
                HTTPException: If analysis not found or update fails
            """
            
            result = await update_analysis_service(analysis_id, update_request, db_manager)
            
            if not result.success:
                if "not found" in result.error.lower():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Analysis with ID {analysis_id} not found"
                    )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.error
                )
            
            return AnalysisDetailResponse(
                success=True,
                data=result.data,
                message="Analysis updated successfully",
                request_id=request_id
            )
        
        # GET /analyses - List analyses with pagination and filtering
        @self.router.get(
            "/analyses",
            response_model=AnalysisListResponse,
            status_code=status.HTTP_200_OK,
            summary="List Analyses",
            description="Retrieve paginated list of analyses with optional filtering",
            response_description="Paginated analysis results",
            tags=["Analysis"]
        )
        @with_response_wrapper(AnalysisListResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def list_analyses(
            page: PageNumber,
            size: PageSize,
            order_by: OrderBy,
            order_desc: OrderDesc,
            analysis_type: Optional[AnalysisTypeEnum] = Query(
                None, description="Filter by analysis type"
            ),
            status: Optional[AnalysisStatusEnum] = Query(
                None, description="Filter by status"
            ),
            optimization_tier: Optional[OptimizationTierEnum] = Query(
                None, description="Filter by optimization tier"
            ),
            date_from: Optional[datetime] = Query(
                None, description="Filter from date (ISO format)"
            ),
            date_to: Optional[datetime] = Query(
                None, description="Filter to date (ISO format)"
            ),
            min_sentiment_score: Optional[float] = Query(
                None, ge=-1.0, le=1.0, description="Minimum sentiment score"
            ),
            max_sentiment_score: Optional[float] = Query(
                None, ge=-1.0, le=1.0, description="Maximum sentiment score"
            ),
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> AnalysisListResponse:
            """
            List analyses with pagination and filtering.
            
            Retrieve a paginated list of analyses with optional filtering by various criteria.
            
            Args:
                page: Page number (1-based)
                size: Page size (1-100)
                order_by: Field to order by
                order_desc: Descending order flag
                analysis_type: Filter by analysis type
                status: Filter by status
                optimization_tier: Filter by optimization tier
                date_from: Filter from date
                date_to: Filter to date
                min_sentiment_score: Minimum sentiment score
                max_sentiment_score: Maximum sentiment score
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                AnalysisListResponse: Paginated analysis results
            """
            
            # Build pagination request
            pagination = PaginationRequest(
                page=page,
                size=size,
                order_by=order_by,
                order_desc=order_desc
            )
            
            # Build filter request
            filters = AnalysisFilterRequest(
                analysis_type=analysis_type,
                status=status,
                optimization_tier=optimization_tier,
                date_from=date_from,
                date_to=date_to,
                min_sentiment_score=min_sentiment_score,
                max_sentiment_score=max_sentiment_score
            )
            
            result = await list_analyses_service(pagination, filters, db_manager)
            
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.error
                )
            
            return AnalysisListResponse(
                success=True,
                data=result.data,
                message=f"Retrieved {len(result.data.items)} analyses",
                request_id=request_id
            )
        
        # DELETE /analyses/{analysis_id} - Delete analysis
        @self.router.delete(
            "/analyses/{analysis_id}",
            response_model=SuccessResponse,
            status_code=status.HTTP_200_OK,
            summary="Delete Analysis",
            description="Delete analysis by ID",
            response_description="Analysis deleted successfully",
            tags=["Analysis"]
        )
        @with_response_wrapper(SuccessResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def delete_analysis(
            analysis_id: AnalysisID,
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> SuccessResponse:
            """
            Delete analysis by ID.
            
            Permanently delete a specific analysis by its ID.
            
            Args:
                analysis_id: Analysis ID
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                SuccessResponse: Deletion confirmation
                
            Raises:
                HTTPException: If analysis not found
            """
            
            # In real implementation, call delete service
            # For now, return success response
            return SuccessResponse(
                success=True,
                message=f"Analysis {analysis_id} deleted successfully",
                data={"deleted_id": analysis_id},
                request_id=request_id
            )
    
    async def _process_analysis_background(
        self,
        analysis_id: int,
        text_content: str,
        analysis_type: AnalysisTypeEnum,
        db_manager: Any
    ):
        """Background task to process analysis."""
        logger = structlog.get_logger("background_processor")
        
        try:
            logger.info(f"Starting background processing for analysis {analysis_id}")
            
            # Simulate processing
            await asyncio.sleep(2)
            
            # Update analysis with results
            update_request = AnalysisUpdateRequest(
                status=AnalysisStatusEnum.COMPLETED,
                sentiment_score=0.5,
                processing_time_ms=2000.0,
                model_used="background-processor"
            )
            
            await update_analysis_service(analysis_id, update_request, db_manager)
            
            logger.info(f"Completed background processing for analysis {analysis_id}")
            
        except Exception as e:
            logger.error(f"Background processing failed for analysis {analysis_id}: {e}")
            
            # Update analysis with error
            error_request = AnalysisUpdateRequest(
                status=AnalysisStatusEnum.ERROR,
                error_message=str(e)
            )
            
            await update_analysis_service(analysis_id, error_request, db_manager)


class BatchRoutes:
    """Declarative route definitions for batch analysis endpoints."""
    
    def __init__(self, router: APIRouter):
        
    """__init__ function."""
self.router = router
        self._register_routes()
    
    def _register_routes(self) -> Any:
        """Register all batch routes."""
        
        # POST /batches - Create batch analysis
        @self.router.post(
            "/batches",
            response_model=BatchDetailResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create Batch Analysis",
            description="Create a new batch analysis with multiple texts",
            response_description="Batch analysis created successfully",
            tags=["Batch Analysis"]
        )
        @with_response_wrapper(BatchDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def create_batch(
            request: BatchAnalysisRequest = Body(
                ...,
                description="Batch analysis request parameters",
                example={
                    "batch_name": "Customer Feedback Analysis",
                    "texts": [
                        "Great product, highly recommended!",
                        "Disappointed with the service quality.",
                        "Average experience, could be better."
                    ],
                    "analysis_type": "sentiment",
                    "optimization_tier": "standard",
                    "priority": 7
                }
            ),
            background_tasks: BackgroundTaskManager = Depends(),
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> BatchDetailResponse:
            """
            Create batch analysis.
            
            Create a new batch analysis to process multiple texts simultaneously.
            
            Args:
                request: Batch analysis request parameters
                background_tasks: Background task manager
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                BatchDetailResponse: Created batch details
            """
            
            result = await create_batch_service(request, db_manager)
            
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.error
                )
            
            # Add background task for processing
            background_tasks.add_task(
                self._process_batch_background,
                result.data.id,
                request.texts,
                request.analysis_type,
                db_manager
            )
            
            return BatchDetailResponse(
                success=True,
                data=result.data,
                message="Batch analysis created successfully",
                request_id=request_id
            )
        
        # GET /batches/{batch_id} - Get batch by ID
        @self.router.get(
            "/batches/{batch_id}",
            response_model=BatchDetailResponse,
            status_code=status.HTTP_200_OK,
            summary="Get Batch by ID",
            description="Retrieve batch analysis details by ID",
            response_description="Batch analysis details",
            tags=["Batch Analysis"]
        )
        @with_response_wrapper(BatchDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def get_batch(
            batch_id: BatchID,
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> BatchDetailResponse:
            """
            Get batch by ID.
            
            Retrieve detailed information about a specific batch analysis by its ID.
            
            Args:
                batch_id: Batch ID
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                BatchDetailResponse: Batch analysis details
            """
            
            # In real implementation, call get batch service
            # For now, return mock response
            return BatchDetailResponse(
                success=True,
                data=BatchAnalysisResponse(
                    id=batch_id,
                    batch_name="Sample Batch",
                    analysis_type=AnalysisTypeEnum.SENTIMENT,
                    status=AnalysisStatusEnum.COMPLETED,
                    total_texts=10,
                    completed_count=8,
                    error_count=1,
                    progress_percentage=90.0,
                    optimization_tier=OptimizationTierEnum.STANDARD,
                    priority=5,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                message="Batch retrieved successfully",
                request_id=request_id
            )
    
    async def _process_batch_background(
        self,
        batch_id: int,
        texts: List[str],
        analysis_type: AnalysisTypeEnum,
        db_manager: Any
    ):
        """Background task to process batch."""
        logger = structlog.get_logger("batch_processor")
        
        try:
            logger.info(f"Starting batch processing for batch {batch_id}")
            
            completed_count = 0
            error_count = 0
            
            for i, text in enumerate(texts):
                try:
                    # Create analysis request
                    analysis_request = TextAnalysisRequest(
                        text_content=text,
                        analysis_type=analysis_type
                    )
                    
                    # Process analysis
                    analysis_result = await create_analysis_service(analysis_request, db_manager)
                    
                    if analysis_result.success:
                        # Update with simulated results
                        update_request = AnalysisUpdateRequest(
                            status=AnalysisStatusEnum.COMPLETED,
                            sentiment_score=0.5 + (i * 0.1),
                            processing_time_ms=100.0 + i,
                            model_used="batch-processor"
                        )
                        
                        await update_analysis_service(analysis_result.data.id, update_request, db_manager)
                        completed_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing text {i+1} in batch {batch_id}: {e}")
            
            # Update batch progress
            await db_manager.update_batch_progress(batch_id, completed_count, error_count)
            
            logger.info(
                f"Completed batch {batch_id}: {completed_count} successful, {error_count} errors"
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed for batch {batch_id}: {e}")


class HealthRoutes:
    """Declarative route definitions for health check endpoints."""
    
    def __init__(self, router: APIRouter):
        
    """__init__ function."""
self.router = router
        self._register_routes()
    
    def _register_routes(self) -> Any:
        """Register all health routes."""
        
        # GET /health - Health check
        @self.router.get(
            "/health",
            response_model=HealthCheckResponse,
            status_code=status.HTTP_200_OK,
            summary="Health Check",
            description="Check system health and status",
            response_description="System health status",
            tags=["Health"]
        )
        @with_response_wrapper(HealthCheckResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def health_check(
            db_manager: DBManager = Depends(get_db_manager),
            request_id: str = Depends(get_request_id)
        ) -> HealthCheckResponse:
            """
            Health check endpoint.
            
            Check the health status of the system and its dependencies.
            
            Args:
                db_manager: Database manager
                request_id: Request ID for tracking
                
            Returns:
                HealthCheckResponse: System health status
            """
            
            start_time = datetime.now()
            
            # Check database health
            db_healthy = True  # In real implementation, check actual database
            try:
                # await db_manager.ping()
                pass
            except Exception:
                db_healthy = False
            
            # Calculate uptime
            uptime = (datetime.now() - start_time).total_seconds()
            
            health_data = HealthResponse(
                status="healthy" if db_healthy else "unhealthy",
                timestamp=datetime.now(),
                version="1.0.0",
                uptime_seconds=uptime,
                database={"status": "healthy" if db_healthy else "unhealthy"},
                performance={"response_time_ms": uptime * 1000}
            )
            
            return HealthCheckResponse(
                success=db_healthy,
                data=health_data,
                message="Health check completed",
                request_id=request_id
            )
        
        # GET /health/detailed - Detailed health check
        @self.router.get(
            "/health/detailed",
            response_model=HealthCheckResponse,
            status_code=status.HTTP_200_OK,
            summary="Detailed Health Check",
            description="Detailed system health check with performance metrics",
            response_description="Detailed health status",
            tags=["Health"]
        )
        @with_response_wrapper(HealthCheckResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def detailed_health_check(
            db_manager: DBManager = Depends(get_db_manager),
            request_id: str = Depends(get_request_id)
        ) -> HealthCheckResponse:
            """
            Detailed health check endpoint.
            
            Perform a detailed health check with performance metrics and dependency status.
            
            Args:
                db_manager: Database manager
                request_id: Request ID for tracking
                
            Returns:
                HealthCheckResponse: Detailed health status
            """
            
            start_time = datetime.now()
            
            # Perform detailed checks
            checks = {
                "database": await self._check_database(db_manager),
                "cache": await self._check_cache(),
                "memory": await self._check_memory(),
                "disk": await self._check_disk()
            }
            
            # Determine overall health
            all_healthy = all(check["status"] == "healthy" for check in checks.values())
            
            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            health_data = HealthResponse(
                status="healthy" if all_healthy else "unhealthy",
                timestamp=datetime.now(),
                version="1.0.0",
                uptime_seconds=processing_time,
                database=checks["database"],
                performance={
                    "response_time_ms": processing_time * 1000,
                    "checks_performed": len(checks)
                },
                errors=[check["error"] for check in checks.values() if check["error"]]
            )
            
            return HealthCheckResponse(
                success=all_healthy,
                data=health_data,
                message="Detailed health check completed",
                request_id=request_id
            )
    
    async def _check_database(self, db_manager: Any) -> Dict[str, Any]:
        """Check database health."""
        try:
            # await db_manager.ping()
            return {"status": "healthy", "error": None}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            # Check cache connectivity
            return {"status": "healthy", "error": None}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {"status": "warning", "error": f"High memory usage: {memory.percent}%"}
            return {"status": "healthy", "error": None}
        except ImportError:
            return {"status": "unknown", "error": "psutil not available"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return {"status": "warning", "error": f"High disk usage: {disk.percent}%"}
            return {"status": "healthy", "error": None}
        except ImportError:
            return {"status": "unknown", "error": "psutil not available"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# ============================================================================
# Route Factory and Application Setup
# ============================================================================

def create_analysis_router() -> APIRouter:
    """Create analysis router with all routes."""
    router = APIRouter(prefix="/api/v1", tags=["Analysis"])
    
    # Register route classes
    AnalysisRoutes(router)
    BatchRoutes(router)
    HealthRoutes(router)
    
    return router

def create_app() -> FastAPI:
    """Create FastAPI application with declarative routes."""
    app = FastAPI(
        title="Text Analysis API",
        description="Functional FastAPI application with declarative routes",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add routers
    analysis_router = create_analysis_router()
    app.include_router(analysis_router)
    
    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger = structlog.get_logger("exception_handler")
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if app.debug else "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return app

# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example usage of declarative routes."""
    
    # Create application
    app = create_app()
    
    # Start server (in real application)
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    print("Declarative routes application created successfully!")
    print("Available endpoints:")
    print("- POST /api/v1/analyses - Create analysis")
    print("- GET /api/v1/analyses/{id} - Get analysis")
    print("- PUT /api/v1/analyses/{id} - Update analysis")
    print("- GET /api/v1/analyses - List analyses")
    print("- DELETE /api/v1/analyses/{id} - Delete analysis")
    print("- POST /api/v1/batches - Create batch")
    print("- GET /api/v1/batches/{id} - Get batch")
    print("- GET /api/v1/health - Health check")
    print("- GET /api/v1/health/detailed - Detailed health check")

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 