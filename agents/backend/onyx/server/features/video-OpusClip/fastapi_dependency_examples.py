"""
🔧 FastAPI Dependency Injection Examples for Video-OpusClip

This module provides comprehensive examples of using FastAPI's dependency
injection system for managing state and shared resources in the Video-OpusClip
AI video processing system.

Examples include:
- Basic dependency injection patterns
- Advanced dependency scoping
- Error handling and recovery
- Performance optimization
- Testing and mocking
- Health monitoring
- Security and validation
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable
from contextlib import asynccontextmanager
from functools import wraps

import torch
import torch.nn as nn
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from pydantic import BaseModel, Field, validator
import structlog

# Import Video-OpusClip components
from fastapi_dependency_injection import (
    DependencyContainer, AppConfig, get_app_config,
    get_dependency_container, set_dependency_container,
    create_app, add_health_endpoints,
    singleton_dependency, cached_dependency, inject_dependencies
)

# Configure logging
logger = structlog.get_logger(__name__)

# =============================================================================
# Request/Response Models
# =============================================================================

class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    video_url: str = Field(..., description="URL of the video to process")
    processing_type: str = Field(default="caption", description="Type of processing")
    max_length: int = Field(default=100, ge=1, le=500, description="Maximum output length")
    quality: str = Field(default="medium", description="Processing quality")
    
    @validator("video_url")
    def validate_video_url(cls, v):
        """Validate video URL format and security."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format")
        return v
    
    @validator("quality")
    def validate_quality(cls, v):
        """Validate processing quality."""
        if v not in ["low", "medium", "high"]:
            raise ValueError("Quality must be low, medium, or high")
        return v

class VideoProcessingResponse(BaseModel):
    """Response model for video processing."""
    success: bool = Field(..., description="Whether processing was successful")
    request_id: str = Field(..., description="Unique request ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Processing result")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for processing")
    confidence_score: float = Field(..., description="Confidence score")

class BatchProcessingRequest(BaseModel):
    """Request model for batch video processing."""
    videos: List[VideoProcessingRequest] = Field(..., description="List of videos to process")
    batch_size: int = Field(default=10, ge=1, le=50, description="Batch size")
    parallel: bool = Field(default=True, description="Process in parallel")

class BatchProcessingResponse(BaseModel):
    """Response model for batch video processing."""
    success: bool = Field(..., description="Whether batch processing was successful")
    batch_id: str = Field(..., description="Unique batch ID")
    results: List[VideoProcessingResponse] = Field(..., description="Processing results")
    total_time: float = Field(..., description="Total processing time")
    success_count: int = Field(..., description="Number of successful processes")
    error_count: int = Field(..., description="Number of failed processes")

# =============================================================================
# Mock Models and Services
# =============================================================================

class MockVideoProcessor(nn.Module):
    """Mock video processor for demonstration."""
    
    def __init__(self, model_name: str = "video_processor"):
        super().__init__()
        self.model_name = model_name
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    async def process_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process video asynchronously."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock processing
        input_tensor = torch.randn(1, 3, 224, 224)
        result = self.forward(input_tensor)
        
        return {
            "processed_frames": 30,
            "features": result.tolist(),
            "confidence": 0.95
        }

class MockCaptionGenerator(nn.Module):
    """Mock caption generator for demonstration."""
    
    def __init__(self, model_name: str = "caption_generator"):
        super().__init__()
        self.model_name = model_name
        self.encoder = nn.Linear(100, 256)
        self.decoder = nn.Linear(256, 1000)  # Vocabulary size
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded
    
    async def generate_caption(self, video_features: List[float], max_length: int) -> str:
        """Generate caption asynchronously."""
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        # Mock caption generation
        captions = [
            "A person walking in a park",
            "A car driving on the highway",
            "A bird flying in the sky",
            "A cat playing with a ball",
            "A sunset over the ocean"
        ]
        
        import random
        return random.choice(captions)

# =============================================================================
# Example 1: Basic Dependency Injection
# =============================================================================

def example_basic_dependency_injection():
    """Example 1: Basic dependency injection patterns."""
    
    print("🎯 Example 1: Basic Dependency Injection")
    print("-" * 40)
    
    app = FastAPI(title="Basic Dependency Injection Example")
    
    # Basic dependency function
    async def get_database_session():
        """Get database session dependency."""
        container = get_dependency_container()
        async with container.db_manager.get_session() as session:
            yield session
    
    # Basic dependency function with error handling
    async def get_model_dependency(model_name: str):
        """Get model dependency with error handling."""
        container = get_dependency_container()
        model = await container.model_manager.get_model(model_name)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model {model_name} not available"
            )
        
        return model
    
    @app.post("/video/process/basic")
    async def process_video_basic(
        request: VideoProcessingRequest,
        model: nn.Module = Depends(get_model_dependency("video_processor")),
        db_session = Depends(get_database_session)
    ):
        """Basic video processing with dependency injection."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Process video using injected model
            result = await model.process_video({
                "url": request.video_url,
                "type": request.processing_type,
                "max_length": request.max_length
            })
            
            processing_time = time.time() - start_time
            
            return VideoProcessingResponse(
                success=True,
                request_id=request_id,
                result=result,
                processing_time=processing_time,
                model_used=model.model_name,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    return app

# =============================================================================
# Example 2: Advanced Dependency Scoping
# =============================================================================

def example_advanced_dependency_scoping():
    """Example 2: Advanced dependency scoping patterns."""
    
    print("🔄 Example 2: Advanced Dependency Scoping")
    print("-" * 40)
    
    app = FastAPI(title="Advanced Dependency Scoping Example")
    
    # Singleton dependency (shared across all requests)
    @singleton_dependency
    async def get_performance_optimizer():
        """Singleton performance optimizer."""
        container = get_dependency_container()
        return container.performance_optimizer
    
    # Cached dependency (cached for TTL)
    @cached_dependency(ttl=300)
    async def get_model_config(model_name: str):
        """Cached model configuration."""
        # Simulate loading configuration
        await asyncio.sleep(0.01)
        return {
            "model_name": model_name,
            "version": "1.0.0",
            "device": "cpu",
            "batch_size": 32
        }
    
    # Request-scoped dependency (new instance per request)
    async def get_request_logger(request: Request):
        """Request-scoped logger."""
        return {
            "request_id": str(uuid.uuid4()),
            "user_agent": request.headers.get("user-agent", ""),
            "ip": request.client.host if request.client else "unknown"
        }
    
    # Session-scoped dependency (shared within user session)
    async def get_user_session(user_id: str):
        """Session-scoped user session."""
        # In real implementation, this would be tied to user session
        return {
            "user_id": user_id,
            "session_id": str(uuid.uuid4()),
            "preferences": {"quality": "high", "language": "en"}
        }
    
    @app.post("/video/process/advanced")
    async def process_video_advanced(
        request: VideoProcessingRequest,
        optimizer = Depends(get_performance_optimizer),
        model_config = Depends(lambda: get_model_config("video_processor")),
        request_logger = Depends(get_request_logger),
        user_session = Depends(lambda: get_user_session("user123"))
    ):
        """Advanced video processing with different dependency scopes."""
        
        start_time = time.time()
        
        # Log request
        logger.info("Processing video", **request_logger)
        
        # Use user preferences
        quality = user_session["preferences"]["quality"]
        
        # Use performance optimizer
        with optimizer.optimize_context():
            # Simulate processing
            await asyncio.sleep(0.1)
            
            result = {
                "processed_frames": 30,
                "quality": quality,
                "model_config": model_config
            }
        
        processing_time = time.time() - start_time
        
        return VideoProcessingResponse(
            success=True,
            request_id=request_logger["request_id"],
            result=result,
            processing_time=processing_time,
            model_used=model_config["model_name"],
            confidence_score=0.95
        )
    
    return app

# =============================================================================
# Example 3: Error Handling and Recovery
# =============================================================================

def example_error_handling_and_recovery():
    """Example 3: Error handling and recovery patterns."""
    
    print("🛡️ Example 3: Error Handling and Recovery")
    print("-" * 40)
    
    app = FastAPI(title="Error Handling and Recovery Example")
    
    # Dependency with retry logic
    async def get_model_with_retry(model_name: str, max_retries: int = 3):
        """Get model with retry logic."""
        container = get_dependency_container()
        
        for attempt in range(max_retries):
            try:
                model = await container.model_manager.get_model(model_name)
                if model:
                    return model
                raise ValueError(f"Model {model_name} not found")
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load model {model_name} after {max_retries} attempts"
                    )
                logger.warning(f"Model load attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
    
    # Dependency with fallback
    async def get_cache_with_fallback():
        """Get cache with fallback to in-memory."""
        container = get_dependency_container()
        
        try:
            cache = await container.cache_manager.get_client()
            if cache:
                return cache
        except Exception as e:
            logger.warning(f"Cache unavailable, using fallback: {e}")
        
        # Fallback to in-memory cache
        return {"type": "memory", "data": {}}
    
    # Dependency with health check
    async def get_healthy_database():
        """Get database only if healthy."""
        container = get_dependency_container()
        
        if not await container.db_manager.health_check():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is unhealthy"
            )
        
        async with container.db_manager.get_session() as session:
            yield session
    
    @app.post("/video/process/robust")
    async def process_video_robust(
        request: VideoProcessingRequest,
        model: nn.Module = Depends(lambda: get_model_with_retry("video_processor")),
        cache = Depends(get_cache_with_fallback),
        db_session = Depends(get_healthy_database)
    ):
        """Robust video processing with error handling."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Check cache first
            cache_key = f"video:{request.video_url}"
            if cache.get("type") == "memory" and cache_key in cache.get("data", {}):
                cached_result = cache["data"][cache_key]
                return VideoProcessingResponse(
                    success=True,
                    request_id=request_id,
                    result=cached_result,
                    processing_time=0.001,  # Very fast from cache
                    model_used="cached",
                    confidence_score=0.99
                )
            
            # Process video
            result = await model.process_video({
                "url": request.video_url,
                "type": request.processing_type,
                "max_length": request.max_length
            })
            
            # Cache result
            if cache.get("type") == "memory":
                cache["data"][cache_key] = result
            
            processing_time = time.time() - start_time
            
            return VideoProcessingResponse(
                success=True,
                request_id=request_id,
                result=result,
                processing_time=processing_time,
                model_used=model.model_name,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    return app

# =============================================================================
# Example 4: Performance Optimization
# =============================================================================

def example_performance_optimization():
    """Example 4: Performance optimization with dependencies."""
    
    print("⚡ Example 4: Performance Optimization")
    print("-" * 40)
    
    app = FastAPI(title="Performance Optimization Example")
    
    # Dependency with performance monitoring
    async def get_model_with_monitoring(model_name: str):
        """Get model with performance monitoring."""
        start_time = time.time()
        
        try:
            container = get_dependency_container()
            model = await container.model_manager.get_model(model_name)
            
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model {model_name} not available"
                )
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded in {load_time:.3f}s")
            
            return model
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Model {model_name} failed to load in {load_time:.3f}s: {e}")
            raise
    
    # Dependency with connection pooling
    async def get_optimized_database():
        """Get database with connection pooling."""
        container = get_dependency_container()
        
        # Use connection pool
        async with container.db_manager.get_session() as session:
            yield session
    
    # Dependency with batch processing
    async def get_batch_processor():
        """Get batch processor for efficient processing."""
        container = get_dependency_container()
        return {
            "optimizer": container.performance_optimizer,
            "mixed_precision": container.mixed_precision_manager,
            "gradient_accumulator": container.gradient_accumulator
        }
    
    @app.post("/video/process/optimized")
    async def process_video_optimized(
        request: VideoProcessingRequest,
        model: nn.Module = Depends(lambda: get_model_with_monitoring("video_processor")),
        db_session = Depends(get_optimized_database),
        batch_processor = Depends(get_batch_processor)
    ):
        """Optimized video processing with performance monitoring."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Use performance optimizations
            with batch_processor["optimizer"].optimize_context():
                with batch_processor["mixed_precision"].autocast():
                    with batch_processor["gradient_accumulator"].accumulate_gradients():
                        # Process video
                        result = await model.process_video({
                            "url": request.video_url,
                            "type": request.processing_type,
                            "max_length": request.max_length
                        })
            
            processing_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(
                "Video processing completed",
                request_id=request_id,
                processing_time=processing_time,
                model_used=model.model_name,
                optimizations_applied=["mixed_precision", "gradient_accumulation"]
            )
            
            return VideoProcessingResponse(
                success=True,
                request_id=request_id,
                result=result,
                processing_time=processing_time,
                model_used=model.model_name,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Video processing failed",
                request_id=request_id,
                processing_time=processing_time,
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    return app

# =============================================================================
# Example 5: Batch Processing
# =============================================================================

def example_batch_processing():
    """Example 5: Batch processing with dependencies."""
    
    print("📦 Example 5: Batch Processing")
    print("-" * 40)
    
    app = FastAPI(title="Batch Processing Example")
    
    # Dependency for batch processing
    async def get_batch_processor():
        """Get batch processor with all necessary components."""
        container = get_dependency_container()
        return {
            "model": await container.model_manager.get_model("video_processor"),
            "optimizer": container.performance_optimizer,
            "logger": container.training_logger,
            "error_handler": container.error_handler
        }
    
    # Background task for batch processing
    async def process_batch_background(
        batch_id: str,
        videos: List[VideoProcessingRequest],
        batch_processor: Dict[str, Any]
    ):
        """Background task for batch processing."""
        
        results = []
        success_count = 0
        error_count = 0
        
        # Start batch logging
        await batch_processor["logger"].start_batch_session(batch_id)
        
        try:
            for i, video_request in enumerate(videos):
                try:
                    start_time = time.time()
                    
                    # Process video
                    result = await batch_processor["model"].process_video({
                        "url": video_request.video_url,
                        "type": video_request.processing_type,
                        "max_length": video_request.max_length
                    })
                    
                    processing_time = time.time() - start_time
                    
                    video_result = VideoProcessingResponse(
                        success=True,
                        request_id=f"{batch_id}_{i}",
                        result=result,
                        processing_time=processing_time,
                        model_used=batch_processor["model"].model_name,
                        confidence_score=result.get("confidence", 0.0)
                    )
                    
                    results.append(video_result)
                    success_count += 1
                    
                    # Log progress
                    await batch_processor["logger"].log_batch_progress(
                        batch_id, i + 1, len(videos), success_count, error_count
                    )
                    
                except Exception as e:
                    error_count += 1
                    await batch_processor["error_handler"].handle_error(e)
                    
                    # Add error result
                    video_result = VideoProcessingResponse(
                        success=False,
                        request_id=f"{batch_id}_{i}",
                        result={"error": str(e)},
                        processing_time=0.0,
                        model_used="error",
                        confidence_score=0.0
                    )
                    results.append(video_result)
            
            # Log batch completion
            await batch_processor["logger"].log_batch_completion(
                batch_id, success_count, error_count
            )
            
        except Exception as e:
            await batch_processor["error_handler"].handle_error(e)
            await batch_processor["logger"].log_batch_error(batch_id, str(e))
    
    @app.post("/video/process/batch", response_model=BatchProcessingResponse)
    async def process_video_batch(
        request: BatchProcessingRequest,
        background_tasks: BackgroundTasks,
        batch_processor = Depends(get_batch_processor)
    ):
        """Batch video processing with background tasks."""
        
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Validate batch size
        if len(request.videos) > request.batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {len(request.videos)} exceeds maximum {request.batch_size}"
            )
        
        # Start background processing
        background_tasks.add_task(
            process_batch_background,
            batch_id,
            request.videos,
            batch_processor
        )
        
        total_time = time.time() - start_time
        
        return BatchProcessingResponse(
            success=True,
            batch_id=batch_id,
            results=[],  # Will be populated by background task
            total_time=total_time,
            success_count=0,
            error_count=0
        )
    
    @app.get("/video/batch/{batch_id}/status")
    async def get_batch_status(batch_id: str):
        """Get batch processing status."""
        
        container = get_dependency_container()
        
        # Get batch status from logger
        status = await container.training_logger.get_batch_status(batch_id)
        
        return {
            "batch_id": batch_id,
            "status": status,
            "timestamp": time.time()
        }
    
    return app

# =============================================================================
# Example 6: Security and Validation
# =============================================================================

def example_security_and_validation():
    """Example 6: Security and validation patterns."""
    
    print("🔒 Example 6: Security and Validation")
    print("-" * 40)
    
    app = FastAPI(title="Security and Validation Example")
    
    # OAuth2 scheme for authentication
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    # Dependency for authentication
    async def get_current_user(
        token: str = Depends(oauth2_scheme),
        security_scopes: SecurityScopes = SecurityScopes()
    ) -> Dict[str, Any]:
        """Get current authenticated user."""
        # In real implementation, validate JWT token
        # For demo, return mock user
        return {
            "user_id": "user123",
            "email": "user@example.com",
            "scopes": ["video:process", "video:read"],
            "is_active": True
        }
    
    # Dependency for authorization
    def require_scope(required_scope: str):
        """Require specific scope for endpoint access."""
        def dependency(user: Dict[str, Any] = Depends(get_current_user)):
            if required_scope not in user.get("scopes", []):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {required_scope}"
                )
            return user
        return dependency
    
    # Dependency for rate limiting
    async def check_rate_limit(user: Dict[str, Any] = Depends(get_current_user)):
        """Check rate limit for user."""
        container = get_dependency_container()
        cache = await container.cache_manager.get_client()
        
        if cache:
            key = f"rate_limit:{user['user_id']}"
            current_count = await cache.get(key, 0)
            
            if int(current_count) >= 10:  # 10 requests per minute
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # Increment counter
            await cache.incr(key)
            await cache.expire(key, 60)  # 1 minute TTL
        
        return user
    
    # Dependency for input validation
    async def validate_video_input(request: VideoProcessingRequest):
        """Validate video input."""
        # Check file size (simulated)
        if len(request.video_url) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video URL too long"
            )
        
        # Check processing type
        allowed_types = ["caption", "transcription", "analysis"]
        if request.processing_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing type. Allowed: {allowed_types}"
            )
        
        return request
    
    @app.post("/video/process/secure")
    async def process_video_secure(
        request: VideoProcessingRequest,
        user: Dict[str, Any] = Depends(require_scope("video:process")),
        rate_limited_user: Dict[str, Any] = Depends(check_rate_limit),
        validated_request: VideoProcessingRequest = Depends(validate_video_input),
        model: nn.Module = Depends(lambda: get_model_with_retry("video_processor"))
    ):
        """Secure video processing with authentication and validation."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Process video
            result = await model.process_video({
                "url": validated_request.video_url,
                "type": validated_request.processing_type,
                "max_length": validated_request.max_length,
                "user_id": user["user_id"]
            })
            
            processing_time = time.time() - start_time
            
            # Log secure processing
            logger.info(
                "Secure video processing completed",
                user_id=user["user_id"],
                request_id=request_id,
                processing_time=processing_time
            )
            
            return VideoProcessingResponse(
                success=True,
                request_id=request_id,
                result=result,
                processing_time=processing_time,
                model_used=model.model_name,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(
                "Secure video processing failed",
                user_id=user["user_id"],
                request_id=request_id,
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    return app

# =============================================================================
# Example 7: Testing and Mocking
# =============================================================================

def example_testing_and_mocking():
    """Example 7: Testing and mocking patterns."""
    
    print("🧪 Example 7: Testing and Mocking")
    print("-" * 40)
    
    app = FastAPI(title="Testing and Mocking Example")
    
    # Dependency that can be easily mocked
    async def get_testable_model(model_name: str = "video_processor"):
        """Get model that can be easily mocked in tests."""
        container = get_dependency_container()
        model = await container.model_manager.get_model(model_name)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model {model_name} not available"
            )
        
        return model
    
    # Dependency with configuration for testing
    async def get_testable_database(test_mode: bool = False):
        """Get database with test mode support."""
        if test_mode:
            # Return mock database for testing
            return {"type": "mock", "data": {}}
        
        container = get_dependency_container()
        async with container.db_manager.get_session() as session:
            yield session
    
    @app.post("/video/process/testable")
    async def process_video_testable(
        request: VideoProcessingRequest,
        model: nn.Module = Depends(get_testable_model),
        db_session = Depends(get_testable_database)
    ):
        """Testable video processing endpoint."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Process video
            result = await model.process_video({
                "url": request.video_url,
                "type": request.processing_type,
                "max_length": request.max_length
            })
            
            processing_time = time.time() - start_time
            
            return VideoProcessingResponse(
                success=True,
                request_id=request_id,
                result=result,
                processing_time=processing_time,
                model_used=model.model_name,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    return app

# =============================================================================
# Example 8: Health Monitoring
# =============================================================================

def example_health_monitoring():
    """Example 8: Health monitoring patterns."""
    
    print("🏥 Example 8: Health Monitoring")
    print("-" * 40)
    
    app = FastAPI(title="Health Monitoring Example")
    
    # Dependency for health monitoring
    async def get_health_monitor():
        """Get health monitor for all dependencies."""
        container = get_dependency_container()
        return {
            "database": container.db_manager,
            "cache": container.cache_manager,
            "models": container.model_manager,
            "services": container.service_manager
        }
    
    # Dependency that checks health before use
    async def get_healthy_model(model_name: str):
        """Get model only if all dependencies are healthy."""
        health_monitor = await get_health_monitor()
        
        # Check all dependencies
        checks = {
            "database": await health_monitor["database"].health_check(),
            "cache": await health_monitor["cache"].health_check(),
            "models": await health_monitor["models"].health_check(),
            "services": await health_monitor["services"].health_check()
        }
        
        # If any dependency is unhealthy, raise error
        unhealthy_deps = [name for name, healthy in checks.items() if not healthy]
        if unhealthy_deps:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Unhealthy dependencies: {unhealthy_deps}"
            )
        
        # Get model
        container = get_dependency_container()
        model = await container.model_manager.get_model(model_name)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model {model_name} not available"
            )
        
        return model
    
    @app.post("/video/process/healthy")
    async def process_video_healthy(
        request: VideoProcessingRequest,
        model: nn.Module = Depends(lambda: get_healthy_model("video_processor")),
        health_monitor = Depends(get_health_monitor)
    ):
        """Video processing with health monitoring."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Process video
            result = await model.process_video({
                "url": request.video_url,
                "type": request.processing_type,
                "max_length": request.max_length
            })
            
            processing_time = time.time() - start_time
            
            # Log health status
            health_status = {
                "database": await health_monitor["database"].health_check(),
                "cache": await health_monitor["cache"].health_check(),
                "models": await health_monitor["models"].health_check(),
                "services": await health_monitor["services"].health_check()
            }
            
            logger.info(
                "Video processing with health monitoring",
                request_id=request_id,
                processing_time=processing_time,
                health_status=health_status
            )
            
            return VideoProcessingResponse(
                success=True,
                request_id=request_id,
                result=result,
                processing_time=processing_time,
                model_used=model.model_name,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing failed"
            )
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check for all dependencies."""
        health_monitor = await get_health_monitor()
        
        health_status = {
            "database": await health_monitor["database"].health_check(),
            "cache": await health_monitor["cache"].health_check(),
            "models": await health_monitor["models"].health_check(),
            "services": await health_monitor["services"].health_check()
        }
        
        overall_health = all(health_status.values())
        
        return {
            "overall_health": overall_health,
            "dependencies": health_status,
            "timestamp": time.time()
        }
    
    return app

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to demonstrate all examples."""
    
    print("🔧 FastAPI Dependency Injection Examples")
    print("=" * 50)
    
    # Run all examples
    examples = [
        ("Basic Dependency Injection", example_basic_dependency_injection),
        ("Advanced Dependency Scoping", example_advanced_dependency_scoping),
        ("Error Handling and Recovery", example_error_handling_and_recovery),
        ("Performance Optimization", example_performance_optimization),
        ("Batch Processing", example_batch_processing),
        ("Security and Validation", example_security_and_validation),
        ("Testing and Mocking", example_testing_and_mocking),
        ("Health Monitoring", example_health_monitoring)
    ]
    
    for name, example_func in examples:
        print(f"\n{name}")
        print("-" * len(name))
        app = example_func()
        print(f"✅ {name} example created successfully")
        print(f"   Routes: {len(app.routes)}")
        print(f"   Title: {app.title}")
    
    print("\n🎉 All examples completed successfully!")
    print("\n📚 Next Steps:")
    print("1. Review each example implementation")
    print("2. Test the examples with FastAPI TestClient")
    print("3. Adapt patterns to your specific use case")
    print("4. Add monitoring and logging")
    print("5. Implement security measures")

if __name__ == "__main__":
    main() 