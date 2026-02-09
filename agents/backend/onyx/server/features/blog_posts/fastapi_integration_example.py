from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import (
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import structlog
from sqlalchemy_2_implementation import (
from typing import Any, List, Dict, Optional
"""
🚀 FastAPI + SQLAlchemy 2.0 Integration Example
===============================================

Practical example showing how to integrate SQLAlchemy 2.0 with FastAPI
for the Blatam Academy NLP system.
"""


    FastAPI, Depends, HTTPException, status, Request, Response,
    BackgroundTasks, Query, Path, Body, Header
)

    DatabaseConfig, SQLAlchemy2Manager,
    TextAnalysisCreate, TextAnalysisUpdate, BatchAnalysisCreate,
    TextAnalysisResponse, BatchAnalysisResponse,
    AnalysisType, AnalysisStatus, OptimizationTier,
    DatabaseError, ValidationError
)

# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseModel):
    """Application settings."""
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/nlp_db",
        env="DATABASE_URL"
    )
    api_title: str = Field(default="Blatam Academy NLP API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

# ============================================================================
# Request/Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    database: Dict[str, Any]
    performance: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None

class AnalysisListResponse(BaseModel):
    """Paginated analysis list response."""
    items: List[TextAnalysisResponse]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool

class BatchCreateRequest(BaseModel):
    """Batch creation request."""
    batch_name: str = Field(..., min_length=1, max_length=200)
    analysis_type: AnalysisType
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    optimization_tier: OptimizationTier = OptimizationTier.STANDARD
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch_name": "Sample Batch",
                "analysis_type": "sentiment",
                "texts": [
                    "This is a positive text.",
                    "This is a negative text.",
                    "This is a neutral text."
                ],
                "optimization_tier": "standard"
            }
        }
    )

# ============================================================================
# Database Management
# ============================================================================

class DatabaseManager:
    """Database manager singleton."""
    
    def __init__(self) -> Any:
        self.settings = Settings()
        self._db_manager: Optional[SQLAlchemy2Manager] = None
    
    async def get_db(self) -> SQLAlchemy2Manager:
        """Get database manager instance."""
        if self._db_manager is None:
            config = DatabaseConfig(
                url=self.settings.database_url,
                pool_size=20,
                max_overflow=30,
                enable_caching=True
            )
            self._db_manager = SQLAlchemy2Manager(config)
            await self._db_manager.initialize()
        
        return self._db_manager
    
    async def cleanup(self) -> Any:
        """Cleanup database manager."""
        if self._db_manager:
            await self._db_manager.cleanup()
            self._db_manager = None

# Global database manager
db_manager = DatabaseManager()

async def get_db() -> SQLAlchemy2Manager:
    """Database dependency."""
    return await db_manager.get_db()

# ============================================================================
# Middleware
# ============================================================================

class RequestLoggingMiddleware:
    """Request logging middleware."""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
        self.logger = structlog.get_logger(__name__)
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            start_time = datetime.now()
            request_id = f"req_{start_time.timestamp()}"
            
            # Log request start
            self.logger.info(
                "Request started",
                request_id=request_id,
                method=scope["method"],
                path=scope["path"]
            )
            
            async def send_with_logging(message) -> Any:
                if message["type"] == "http.response.start":
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    self.logger.info(
                        "Request completed",
                        request_id=request_id,
                        status_code=message["status"],
                        duration=duration
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_with_logging)
        else:
            await self.app(scope, receive, send)

# ============================================================================
# Error Handling
# ============================================================================

async def database_exception_handler(request: Request, exc: DatabaseError):
    """Handle database exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Database operation failed",
            detail=str(exc),
            timestamp=datetime.now(),
            request_id=request.headers.get("X-Request-ID")
        ).model_dump()
    )

async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
            timestamp=datetime.now(),
            request_id=request.headers.get("X-Request-ID")
        ).model_dump()
    )

# ============================================================================
# API Routes
# ============================================================================

app = FastAPI(
    title="Blatam Academy NLP API",
    version="1.0.0",
    description="NLP Analysis API with SQLAlchemy 2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(RequestLoggingMiddleware)

# Add exception handlers
app.add_exception_handler(DatabaseError, database_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)

@app.get("/health", response_model=HealthResponse)
async def health_check(db: SQLAlchemy2Manager = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database health
        health = await db.health_check()
        metrics = await db.get_performance_metrics()
        
        return HealthResponse(
            status="healthy" if health.is_healthy else "unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database={
                "is_healthy": health.is_healthy,
                "connection_count": health.connection_count,
                "pool_size": health.pool_size,
                "avg_query_time": health.avg_query_time
            },
            performance=metrics
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/analyses", response_model=TextAnalysisResponse)
async def create_analysis(
    data: TextAnalysisCreate,
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """Create a new text analysis."""
    try:
        analysis = await db.create_text_analysis(data)
        return analysis
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@app.get("/analyses/{analysis_id}", response_model=TextAnalysisResponse)
async def get_analysis(
    analysis_id: int = Path(..., description="Analysis ID", ge=1),
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """Get analysis by ID."""
    analysis = await db.get_text_analysis(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    return analysis

@app.put("/analyses/{analysis_id}", response_model=TextAnalysisResponse)
async def update_analysis(
    analysis_id: int = Path(..., description="Analysis ID", ge=1),
    data: TextAnalysisUpdate = Body(...),
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """Update analysis results."""
    try:
        analysis = await db.update_text_analysis(analysis_id, data)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        return analysis
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )

@app.get("/analyses", response_model=AnalysisListResponse)
async def list_analyses(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    analysis_type: Optional[AnalysisType] = Query(None, description="Filter by analysis type"),
    status: Optional[AnalysisStatus] = Query(None, description="Filter by status"),
    order_by: str = Query("created_at", description="Order by field"),
    order_desc: bool = Query(True, description="Descending order"),
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """List analyses with filtering and pagination."""
    try:
        # Calculate offset
        offset = (page - 1) * size
        
        # Build filters
        filter_params = {}
        if analysis_type:
            filter_params["analysis_type"] = analysis_type
        if status:
            filter_params["status"] = status
        
        # Get analyses
        analyses = await db.list_text_analyses(
            limit=size,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc,
            **filter_params
        )
        
        # For simplicity, we'll assume total count
        # In a real implementation, you'd get this from the database
        total = len(analyses) + (page - 1) * size  # Approximate
        
        return AnalysisListResponse(
            items=analyses,
            total=total,
            page=page,
            size=size,
            has_next=len(analyses) == size,
            has_prev=page > 1
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analyses: {str(e)}"
        )

@app.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: int = Path(..., description="Analysis ID", ge=1),
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """Delete analysis by ID."""
    success = await db.delete_text_analysis(analysis_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    return {"message": "Analysis deleted successfully"}

@app.post("/batches", response_model=BatchAnalysisResponse)
async def create_batch(
    data: BatchCreateRequest,
    background_tasks: BackgroundTasks,
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """Create batch analysis with multiple texts."""
    try:
        # Create batch
        batch_data = BatchAnalysisCreate(
            batch_name=data.batch_name,
            analysis_type=data.analysis_type,
            optimization_tier=data.optimization_tier
        )
        batch = await db.create_batch_analysis(batch_data)
        
        # Add background task to process texts
        background_tasks.add_task(
            process_batch_texts,
            batch.id,
            data.texts,
            data.analysis_type,
            db
        )
        
        return batch
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch: {str(e)}"
        )

@app.get("/batches/{batch_id}", response_model=BatchAnalysisResponse)
async def get_batch(
    batch_id: int = Path(..., description="Batch ID", ge=1),
    db: SQLAlchemy2Manager = Depends(get_db)
):
    """Get batch analysis by ID."""
    batch = await db.get_batch_analysis(batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )
    return batch

@app.get("/metrics")
async def get_metrics(db: SQLAlchemy2Manager = Depends(get_db)):
    """Get performance metrics."""
    try:
        metrics = await db.get_performance_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )

# ============================================================================
# Background Tasks
# ============================================================================

async def process_batch_texts(
    batch_id: int,
    texts: List[str],
    analysis_type: AnalysisType,
    db: SQLAlchemy2Manager
):
    """Background task to process batch texts."""
    logger = structlog.get_logger(__name__)
    
    try:
        completed_count = 0
        error_count = 0
        
        logger.info(f"Starting batch processing for batch {batch_id}")
        
        for i, text in enumerate(texts):
            try:
                # Create analysis
                analysis_data = TextAnalysisCreate(
                    text_content=text,
                    analysis_type=analysis_type
                )
                analysis = await db.create_text_analysis(analysis_data)
                
                # Simulate NLP processing
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Update with results (simulated)
                update_data = TextAnalysisUpdate(
                    status=AnalysisStatus.COMPLETED,
                    sentiment_score=0.5 + (i * 0.1),  # Simulated score
                    processing_time_ms=100.0 + i,
                    model_used="test-model"
                )
                await db.update_text_analysis(analysis.id, update_data)
                
                completed_count += 1
                
                logger.info(f"Processed text {i+1}/{len(texts)} in batch {batch_id}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing text {i+1} in batch {batch_id}: {e}")
        
        # Update batch progress
        await db.update_batch_progress(batch_id, completed_count, error_count)
        
        logger.info(
            f"Completed batch {batch_id}: {completed_count} successful, {error_count} errors"
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing {batch_id}: {e}")

# ============================================================================
# Application Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logging.info("Starting FastAPI application...")
    
    # Initialize database
    await db_manager.get_db()
    
    logging.info("FastAPI application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logging.info("Shutting down FastAPI application...")
    
    # Cleanup database
    await db_manager.cleanup()
    
    logging.info("FastAPI application shut down successfully")

# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example usage of the FastAPI application."""
    
    # This would be used for testing or demonstration
    print("FastAPI + SQLAlchemy 2.0 Integration Example")
    print("=" * 50)
    
    # Example API calls would go here
    # In a real scenario, you'd use a test client
    
    print("✅ Application configured successfully")
    print("📚 API Documentation available at /docs")
    print("🔍 Health check available at /health")
    print("📊 Metrics available at /metrics")

# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    # Run development server
    uvicorn.run(
        "fastapi_integration_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 