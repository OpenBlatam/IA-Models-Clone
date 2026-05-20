from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum
import asyncio
import time
import logging
import uuid
from pathlib import Path as PathLib
from datetime import datetime
import json
    import re
    from html import escape
    import psutil
    import uvicorn
from typing import Any, List, Dict, Optional
"""
FastAPI Functional Implementation for notebooklm_ai
- Uses functional components (plain functions)
- Pydantic models for input validation and response schemas
- Declarative route definitions with clear return type annotations
- def for synchronous operations and async def for asynchronous ones
"""


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS - Input Validation and Response Schemas
# ============================================================================

class PipelineType(str, Enum):
    """Enumeration of available diffusion pipeline types."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"

class ModelType(str, Enum):
    """Enumeration of available model types."""
    STABLE_DIFFUSION_V1_5 = "stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stable-diffusion-xl"
    CONTROLNET_CANNY = "controlnet-canny"
    CONTROLNET_DEPTH = "controlnet-depth"

class DiffusionRequest(BaseModel):
    """Request model for single image generation."""
    model_config = ConfigDict(extra="forbid")
    
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Text prompt for image generation"
    )
    negative_prompt: Optional[str] = Field(
        None, 
        max_length=1000, 
        description="Negative prompt to avoid certain elements"
    )
    pipeline_type: PipelineType = Field(
        default=PipelineType.TEXT_TO_IMAGE,
        description="Type of diffusion pipeline to use"
    )
    model_type: ModelType = Field(
        default=ModelType.STABLE_DIFFUSION_V1_5,
        description="Specific model to use"
    )
    num_inference_steps: int = Field(
        default=50, 
        ge=1, 
        le=100, 
        description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=7.5, 
        ge=1.0, 
        le=20.0, 
        description="How closely to follow the prompt"
    )
    width: int = Field(
        default=512, 
        ge=64, 
        le=2048, 
        description="Image width"
    )
    height: int = Field(
        default=512, 
        ge=64, 
        le=2048, 
        description="Image height"
    )
    seed: Optional[int] = Field(
        None, 
        ge=0, 
        le=2**32-1, 
        description="Random seed for reproducible results"
    )
    batch_size: int = Field(
        default=1, 
        ge=1, 
        le=4, 
        description="Number of images to generate"
    )
    
    @validator('width', 'height')
    def validate_dimensions(cls, v: int) -> int:
        """Validate that dimensions are divisible by 8."""
        if v % 8 != 0:
            raise ValueError('Width and height must be divisible by 8')
        return v
    
    @validator('prompt')
    def validate_prompt(cls, v: str) -> str:
        """Sanitize and validate prompt."""
        sanitized = v.strip()
        if not sanitized:
            raise ValueError('Prompt cannot be empty')
        return sanitized[:1000]  # Ensure max length

class BatchDiffusionRequest(BaseModel):
    """Request model for batch image generation."""
    model_config = ConfigDict(extra="forbid")
    
    requests: List[DiffusionRequest] = Field(
        ..., 
        min_items=1, 
        max_items=10, 
        description="List of diffusion requests"
    )
    
    @validator('requests')
    def validate_batch_size(cls, v: List[DiffusionRequest]) -> List[DiffusionRequest]:
        """Validate batch size and total images."""
        total_images = sum(req.batch_size for req in v)
        if total_images > 20:
            raise ValueError('Total images in batch cannot exceed 20')
        return v

class TrainingRequest(BaseModel):
    """Request model for model training."""
    model_config = ConfigDict(extra="forbid")
    
    model_name: str = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="Name of the model to train"
    )
    dataset_path: str = Field(
        ..., 
        min_length=1, 
        description="Path to training dataset"
    )
    epochs: int = Field(
        default=100, 
        ge=1, 
        le=1000, 
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=1e-4, 
        ge=1e-6, 
        le=1e-2, 
        description="Learning rate"
    )
    batch_size: int = Field(
        default=1, 
        ge=1, 
        le=16, 
        description="Training batch size"
    )
    save_steps: int = Field(
        default=500, 
        ge=100, 
        le=10000, 
        description="Steps between model saves"
    )

class ImageToImageRequest(BaseModel):
    """Request model for image-to-image transformation."""
    model_config = ConfigDict(extra="forbid")
    
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Text prompt for transformation"
    )
    strength: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0, 
        description="Transformation strength"
    )
    guidance_scale: float = Field(
        default=7.5, 
        ge=1.0, 
        le=20.0, 
        description="Guidance scale"
    )
    num_inference_steps: int = Field(
        default=50, 
        ge=1, 
        le=100, 
        description="Number of inference steps"
    )

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class DiffusionResponse(BaseModel):
    """Response model for single image generation."""
    model_config = ConfigDict(extra="forbid")
    
    image_url: str = Field(..., description="URL to generated image")
    image_id: str = Field(..., description="Unique image identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Generation metadata"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    seed: Optional[int] = Field(None, description="Random seed used")
    model_used: str = Field(..., description="Model used for generation")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BatchDiffusionResponse(BaseModel):
    """Response model for batch image generation."""
    model_config = ConfigDict(extra="forbid")
    
    images: List[DiffusionResponse] = Field(..., description="List of generated images")
    total_processing_time: float = Field(..., description="Total processing time")
    batch_id: str = Field(..., description="Unique batch identifier")
    successful_generations: int = Field(..., description="Number of successful generations")
    failed_generations: int = Field(..., description="Number of failed generations")

class TrainingResponse(BaseModel):
    """Response model for training job."""
    model_config = ConfigDict(extra="forbid")
    
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    model_name: str = Field(..., description="Model being trained")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """Response model for errors."""
    model_config = ConfigDict(extra="forbid")
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier")

class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    gpu_available: bool = Field(..., description="GPU availability")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage in MB")

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    description: str = Field(..., description="Model description")
    parameters: int = Field(..., description="Number of parameters")
    size_mb: float = Field(..., description="Model size in MB")
    loaded: bool = Field(..., description="Whether model is loaded")

# ============================================================================
# FUNCTIONAL SERVICES - Pure Functions
# ============================================================================

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file - synchronous function."""
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    max_size = 10 * 1024 * 1024  # 10MB
    
    if file.content_type not in allowed_types:
        raise ValueError(f"File type {file.content_type} not allowed")
    
    if file.size and file.size > max_size:
        raise ValueError(f"File too large. Maximum size: {max_size} bytes")
    
    return True

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user input prompt - synchronous function."""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', prompt)
    
    # HTML escape
    sanitized = escape(sanitized)
    
    # Limit length and trim
    sanitized = sanitized.strip()[:1000]
    
    return sanitized

def generate_image_id() -> str:
    """Generate unique image ID - synchronous function."""
    return str(uuid.uuid4())

def calculate_processing_time(start_time: float) -> float:
    """Calculate processing time - synchronous function."""
    return time.time() - start_time

def validate_model_parameters(request: DiffusionRequest) -> bool:
    """Validate model parameters - synchronous function."""
    if request.width * request.height > 1024 * 1024:
        raise ValueError("Image dimensions too large")
    
    if request.num_inference_steps * request.batch_size > 1000:
        raise ValueError("Total inference steps too high")
    
    return True

# ============================================================================
# ASYNC SERVICE FUNCTIONS
# ============================================================================

async def load_diffusion_model(model_type: ModelType) -> Dict[str, Any]:
    """Load diffusion model - asynchronous function."""
    logger.info(f"Loading model: {model_type}")
    
    # Simulate model loading
    await asyncio.sleep(0.1)
    
    return {
        "model_type": model_type,
        "loaded": True,
        "parameters": 890000000,  # 890M parameters
        "size_mb": 2048.0
    }

async def generate_single_image(request: DiffusionRequest) -> DiffusionResponse:
    """Generate single image - asynchronous function."""
    start_time = time.time()
    
    # Validate request
    validate_model_parameters(request)
    
    # Sanitize inputs
    request.prompt = sanitize_prompt(request.prompt)
    if request.negative_prompt:
        request.negative_prompt = sanitize_prompt(request.negative_prompt)
    
    # Load model if needed
    await load_diffusion_model(request.model_type)
    
    # Simulate image generation
    await asyncio.sleep(2.0)  # Simulate processing time
    
    # Generate image data (placeholder)
    image_id = generate_image_id()
    image_url = f"/generated/{image_id}.png"
    
    processing_time = calculate_processing_time(start_time)
    
    return DiffusionResponse(
        image_url=image_url,
        image_id=image_id,
        metadata={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "pipeline_type": request.pipeline_type,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "batch_size": request.batch_size
        },
        processing_time=processing_time,
        seed=request.seed,
        model_used=str(request.model_type)
    )

async def generate_batch_images(requests: List[DiffusionRequest]) -> BatchDiffusionResponse:
    """Generate multiple images in batch - asynchronous function."""
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    # Process requests concurrently
    tasks = [generate_single_image(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results
    images = []
    successful = 0
    failed = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error in batch request {i}: {result}")
            failed += 1
            # Create error response for failed requests
            images.append(DiffusionResponse(
                image_url="",
                image_id=f"error_{i}",
                metadata={"error": str(result)},
                processing_time=0,
                seed=None,
                model_used="error"
            ))
        else:
            images.append(result)
            successful += 1
    
    total_time = calculate_processing_time(start_time)
    
    return BatchDiffusionResponse(
        images=images,
        total_processing_time=total_time,
        batch_id=batch_id,
        successful_generations=successful,
        failed_generations=failed
    )

async def process_image_to_image(
    image_file: UploadFile, 
    request: ImageToImageRequest
) -> DiffusionResponse:
    """Process image-to-image transformation - asynchronous function."""
    start_time = time.time()
    
    # Validate image file
    validate_image_file(image_file)
    
    # Sanitize prompt
    request.prompt = sanitize_prompt(request.prompt)
    
    # Simulate image processing
    await asyncio.sleep(3.0)
    
    image_id = generate_image_id()
    image_url = f"/transformed/{image_id}.png"
    
    processing_time = calculate_processing_time(start_time)
    
    return DiffusionResponse(
        image_url=image_url,
        image_id=image_id,
        metadata={
            "prompt": request.prompt,
            "strength": request.strength,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "original_file": image_file.filename
        },
        processing_time=processing_time,
        seed=None,
        model_used="image_to_image"
    )

async def start_training_job(request: TrainingRequest) -> TrainingResponse:
    """Start training job - asynchronous function."""
    job_id = str(uuid.uuid4())
    
    # Simulate training job creation
    await asyncio.sleep(0.5)
    
    return TrainingResponse(
        job_id=job_id,
        status="started",
        estimated_duration=request.epochs * 2,  # Rough estimate
        model_name=request.model_name
    )

async def get_available_models() -> List[ModelInfoResponse]:
    """Get available models - asynchronous function."""
    models = [
        ModelInfoResponse(
            name="stable-diffusion-v1-5",
            type="text-to-image",
            description="Stable Diffusion v1.5 - High quality image generation",
            parameters=890000000,
            size_mb=2048.0,
            loaded=True
        ),
        ModelInfoResponse(
            name="stable-diffusion-xl",
            type="text-to-image",
            description="Stable Diffusion XL - Advanced image generation",
            parameters=2800000000,
            size_mb=6144.0,
            loaded=False
        ),
        ModelInfoResponse(
            name="controlnet-canny",
            type="controlnet",
            description="ControlNet with Canny edge detection",
            parameters=890000000,
            size_mb=2048.0,
            loaded=False
        )
    ]
    
    return models

async def get_health_status() -> HealthResponse:
    """Get service health status - asynchronous function."""
    
    # Simulate model loading status
    models_loaded = {
        "stable-diffusion-v1-5": True,
        "stable-diffusion-xl": False,
        "controlnet-canny": False
    }
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = {
        "total": memory.total / (1024 * 1024),
        "available": memory.available / (1024 * 1024),
        "used": memory.used / (1024 * 1024)
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=time.time() - psutil.boot_time(),
        gpu_available=True,  # Simulate GPU detection
        models_loaded=models_loaded,
        memory_usage=memory_usage
    )

# ============================================================================
# DEPENDENCY INJECTION FUNCTIONS
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> str:
    """Get current user from JWT token - async dependency."""
    # Simulate JWT validation
    await asyncio.sleep(0.01)
    
    # In production, validate JWT token here
    return "user_123"

async def get_rate_limit_info() -> Dict[str, Any]:
    """Get rate limit information - async dependency."""
    # Simulate rate limit checking
    await asyncio.sleep(0.01)
    
    return {
        "requests_remaining": 100,
        "reset_time": time.time() + 3600
    }

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

# ============================================================================
# LIFESPAN CONTEXT MANAGER - Startup and Shutdown Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup - Initialize resources
    logger.info("🚀 Starting up FastAPI application...")
    
    # Initialize AI models cache
    app.state.model_cache = {}
    app.state.model_loading_tasks = {}
    
    # Initialize performance metrics
    app.state.request_count = 0
    app.state.error_count = 0
    app.state.avg_response_time = 0.0
    
    # Initialize cache
    app.state.response_cache = {}
    app.state.cache_hits = 0
    app.state.cache_misses = 0
    
    # Load essential models asynchronously
    logger.info("📦 Loading essential diffusion models...")
    try:
        # Load models in background
        asyncio.create_task(load_essential_models(app))
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
    
    logger.info("✅ FastAPI application startup complete")
    
    yield
    
    # Shutdown - Cleanup resources
    logger.info("🛑 Shutting down FastAPI application...")
    
    # Cancel any pending model loading tasks
    for task in app.state.model_loading_tasks.values():
        if not task.done():
            task.cancel()
    
    # Clear caches
    app.state.model_cache.clear()
    app.state.response_cache.clear()
    
    # Save final metrics
    logger.info(f"📊 Final metrics - Requests: {app.state.request_count}, "
                f"Errors: {app.state.error_count}, "
                f"Avg Response Time: {app.state.avg_response_time:.3f}s")
    
    logger.info("✅ FastAPI application shutdown complete")

async def load_essential_models(app: FastAPI) -> None:
    """Load essential models asynchronously."""
    essential_models = [
        ModelType.STABLE_DIFFUSION_V1_5,
        ModelType.STABLE_DIFFUSION_XL
    ]
    
    for model_type in essential_models:
        try:
            app.state.model_loading_tasks[model_type] = asyncio.create_task(
                load_diffusion_model(model_type)
            )
            logger.info(f"🔄 Loading model: {model_type}")
        except Exception as e:
            logger.error(f"❌ Failed to start loading {model_type}: {e}")

# ============================================================================
# MIDDLEWARE - Logging, Error Monitoring, and Performance Optimization
# ============================================================================

class LoggingMiddleware:
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info(f"📥 [{request_id}] {scope['method']} {scope['path']} - "
                   f"Client: {scope.get('client', ('unknown',))[0]}")
        
        # Track request count
        if hasattr(scope['app'].state, 'request_count'):
            scope['app'].state.request_count += 1
        
        # Add request ID to scope
        scope['request_id'] = request_id
        
        # Process request
        try:
            await self.app(scope, receive, send)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update average response time
            if hasattr(scope['app'].state, 'avg_response_time'):
                current_avg = scope['app'].state.avg_response_time
                request_count = scope['app'].state.request_count
                scope['app'].state.avg_response_time = (
                    (current_avg * (request_count - 1) + response_time) / request_count
                )
            
            logger.info(f"📤 [{request_id}] Response time: {response_time:.3f}s")
            
        except Exception as e:
            # Track error count
            if hasattr(scope['app'].state, 'error_count'):
                scope['app'].state.error_count += 1
            
            logger.error(f"❌ [{request_id}] Error: {e}")
            raise

class PerformanceMiddleware:
    """Middleware for performance monitoring and optimization."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        # Add performance headers
        async def send_with_headers(message) -> Any:
            if message["type"] == "http.response.start":
                message["headers"].extend([
                    (b"X-Process-Time", str(time.time() - start_time).encode()),
                    (b"X-Request-ID", scope.get('request_id', b'unknown')),
                    (b"X-Cache-Hit", b"false")
                ])
            await send(message)
        
        await self.app(scope, receive, send_with_headers)

class CachingMiddleware:
    """Middleware for response caching."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Only cache GET requests
        if scope["method"] != "GET":
            await self.app(scope, receive, send)
            return
        
        # Create cache key
        cache_key = f"{scope['method']}:{scope['path']}:{scope.get('query_string', b'').decode()}"
        
        # Check cache
        if hasattr(scope['app'].state, 'response_cache'):
            cached_response = scope['app'].state.response_cache.get(cache_key)
            if cached_response:
                scope['app'].state.cache_hits += 1
                
                async def send_cached(message) -> Any:
                    if message["type"] == "http.response.start":
                        message["headers"].extend([
                            (b"X-Cache-Hit", b"true"),
                            (b"X-Cache-Key", cache_key.encode())
                        ])
                    await send(message)
                
                await send_cached(cached_response)
                return
        
        scope['app'].state.cache_misses += 1
        await self.app(scope, receive, send)

def create_application() -> FastAPI:
    """Create FastAPI application instance with comprehensive middleware."""
    app = FastAPI(
        title="notebooklm_ai API",
        version="1.0.0",
        description="Production-grade AI Diffusion Models API with functional components",
        openapi_url="/api/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add middleware in order (last added = first executed)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(CachingMiddleware)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

app = create_application()

# ============================================================================
# DECLARATIVE ROUTE DEFINITIONS WITH CLEAR RETURN TYPE ANNOTATIONS
# ============================================================================

@app.post(
    "/api/v1/diffusion/generate",
    response_model=DiffusionResponse,
    status_code=200,
    summary="Generate single image from text prompt",
    description="Generate an image from a text prompt using diffusion models",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def generate_image(
    request: DiffusionRequest,
    current_user: str = Depends(get_current_user),
    rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
) -> DiffusionResponse:
    """
    Generate a single image from text prompt.
    
    Args:
        request: Diffusion request parameters
        current_user: Authenticated user
        rate_limit: Rate limit information
    
    Returns:
        DiffusionResponse: Generated image information
    
    Raises:
        HTTPException: For validation or processing errors
    """
    try:
        logger.info(f"Generating image for user {current_user}")
        return await generate_single_image(request)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post(
    "/api/v1/diffusion/generate-batch",
    response_model=BatchDiffusionResponse,
    status_code=200,
    summary="Generate multiple images in batch",
    description="Generate multiple images in a single batch request",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def generate_batch_images_endpoint(
    request: BatchDiffusionRequest,
    current_user: str = Depends(get_current_user),
    rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
) -> BatchDiffusionResponse:
    """
    Generate multiple images in batch.
    
    Args:
        request: Batch diffusion request
        current_user: Authenticated user
        rate_limit: Rate limit information
    
    Returns:
        BatchDiffusionResponse: Batch generation results
    """
    try:
        logger.info(f"Generating batch for user {current_user}")
        return await generate_batch_images(request.requests)
    except ValueError as e:
        logger.error(f"Batch validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post(
    "/api/v1/diffusion/img2img",
    response_model=DiffusionResponse,
    status_code=200,
    summary="Transform image using diffusion model",
    description="Transform an input image using diffusion model",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def image_to_image_endpoint(
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Query(..., min_length=1, max_length=1000, description="Text prompt"),
    strength: float = Query(0.8, ge=0.0, le=1.0, description="Transformation strength"),
    guidance_scale: float = Query(7.5, ge=1.0, le=20.0, description="Guidance scale"),
    num_inference_steps: int = Query(50, ge=1, le=100, description="Number of inference steps"),
    current_user: str = Depends(get_current_user)
) -> DiffusionResponse:
    """
    Transform image using diffusion model.
    
    Args:
        image: Input image file
        prompt: Text prompt for transformation
        strength: Transformation strength
        guidance_scale: Guidance scale
        num_inference_steps: Number of inference steps
        current_user: Authenticated user
    
    Returns:
        DiffusionResponse: Transformed image information
    """
    try:
        request = ImageToImageRequest(
            prompt=prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        
        logger.info(f"Processing image-to-image for user {current_user}")
        return await process_image_to_image(image, request)
    except ValueError as e:
        logger.error(f"Image-to-image validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Image-to-image processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post(
    "/api/v1/training/start",
    response_model=TrainingResponse,
    status_code=202,
    summary="Start model training job",
    description="Start a new model training job in the background",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
) -> TrainingResponse:
    """
    Start model training job.
    
    Args:
        request: Training request parameters
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
    
    Returns:
        TrainingResponse: Training job information
    """
    try:
        logger.info(f"Starting training job for user {current_user}")
        
        # Add background task for actual training
        background_tasks.add_task(
            run_training_background,
            request.model_name,
            request.dataset_path,
            request.epochs
        )
        
        return await start_training_job(request)
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get(
    "/api/v1/models",
    response_model=List[ModelInfoResponse],
    status_code=200,
    summary="Get available models",
    description="Get list of available diffusion models",
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_models(
    current_user: str = Depends(get_current_user)
) -> List[ModelInfoResponse]:
    """
    Get available models.
    
    Args:
        current_user: Authenticated user
    
    Returns:
        List[ModelInfoResponse]: Available models information
    """
    try:
        logger.info(f"Getting models for user {current_user}")
        return await get_available_models()
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health check endpoint",
    description="Check API health and system status",
    responses={
        500: {"model": ErrorResponse, "description": "Service unhealthy"}
    }
)
async def health_check() -> HealthResponse:
    """
    Check API health.
    
    Returns:
        HealthResponse: Health status information
    """
    try:
        return await get_health_status()
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.get(
    "/api/v1/models/{model_name}",
    response_model=ModelInfoResponse,
    status_code=200,
    summary="Get specific model information",
    description="Get detailed information about a specific model",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_model_info(
    model_name: str = Path(..., description="Name of the model"),
    current_user: str = Depends(get_current_user)
) -> ModelInfoResponse:
    """
    Get specific model information.
    
    Args:
        model_name: Name of the model
        current_user: Authenticated user
    
    Returns:
        ModelInfoResponse: Model information
    """
    try:
        models = await get_available_models()
        for model in models:
            if model.name == model_name:
                return model
        
        raise HTTPException(status_code=404, detail="Model not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete(
    "/api/v1/images/{image_id}",
    status_code=204,
    summary="Delete generated image",
    description="Delete a generated image by ID",
    responses={
        404: {"model": ErrorResponse, "description": "Image not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def delete_image(
    image_id: str = Path(..., description="Image ID to delete"),
    current_user: str = Depends(get_current_user)
) -> None:
    """
    Delete generated image.
    
    Args:
        image_id: ID of the image to delete
        current_user: Authenticated user
    
    Returns:
        None
    """
    try:
        logger.info(f"Deleting image {image_id} for user {current_user}")
        # Simulate image deletion
        await asyncio.sleep(0.1)
        
        # In production, actually delete the image file
        return None
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# BACKGROUND TASKS - ASYNC FUNCTIONS
# ============================================================================

async def run_training_background(model_name: str, dataset_path: str, epochs: int) -> None:
    """Run training in background - async function."""
    logger.info(f"Starting background training for {model_name}")
    
    try:
        for epoch in range(epochs):
            # Simulate training epoch
            await asyncio.sleep(1.0)
            logger.info(f"Training epoch {epoch + 1}/{epochs} completed")
        
        logger.info(f"Training completed for {model_name}")
    except Exception as e:
        logger.error(f"Training error: {e}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            detail=str(exc),
            error_code="VALIDATION_ERROR",
            request_id=str(uuid.uuid4())
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle global exceptions."""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=str(uuid.uuid4())
        ).model_dump()
    )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    uvicorn.run(
        "fastapi_functional_implementation:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 