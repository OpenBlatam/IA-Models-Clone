# FastAPI-Specific Guidelines for notebooklm_ai

## 1. Project Structure & Organization

### Recommended Directory Structure
```
notebooklm_ai/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app instance
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings and configuration
│   │   ├── security.py         # Authentication & authorization
│   │   └── database.py         # Database connections
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── diffusion.py
│   │   │   │   ├── training.py
│   │   │   │   └── evaluation.py
│   │   │   └── api.py          # API router
│   │   └── deps.py             # Dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py          # Request models
│   │   ├── response.py         # Response models
│   │   └── database.py         # Database models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── diffusion_service.py
│   │   ├── training_service.py
│   │   └── evaluation_service.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── validators.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   └── test_services/
├── alembic/                    # Database migrations
├── requirements.txt
└── docker-compose.yml
```

## 2. FastAPI Application Setup

### Main Application (`app/main.py`)
```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging

from app.core.config import settings
from app.api.v1.api import api_router
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    # Initialize AI models, database connections, etc.
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    # Cleanup resources

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Production-grade AI Diffusion Models API",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    return app

app = create_application()
```

## 3. Configuration Management

### Settings (`app/core/config.py`)
```python
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Configuration
    PROJECT_NAME: str = "notebooklm_ai"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str
    
    # AI Models
    MODEL_CACHE_DIR: str = "./models"
    DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
    
    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 1
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

## 4. Request/Response Models

### Pydantic Models (`app/models/request.py`)
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
from enum import Enum

class PipelineType(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"

class DiffusionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="Negative prompt")
    pipeline_type: PipelineType = Field(default=PipelineType.TEXT_TO_IMAGE)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: Optional[int] = Field(None, ge=0, le=2**32-1)
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v % 8 != 0:
            raise ValueError('Width and height must be divisible by 8')
        return v

class BatchDiffusionRequest(BaseModel):
    requests: List[DiffusionRequest] = Field(..., min_items=1, max_items=10)
    
    @validator('requests')
    def validate_batch_size(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 requests per batch')
        return v

class TrainingRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    dataset_path: str = Field(..., min_length=1)
    epochs: int = Field(default=100, ge=1, le=1000)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=1, ge=1, le=16)
```

### Response Models (`app/models/response.py`)
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class DiffusionResponse(BaseModel):
    image_url: str = Field(..., description="URL to generated image")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = Field(..., description="Processing time in seconds")
    seed: Optional[int] = Field(None, description="Random seed used")

class BatchDiffusionResponse(BaseModel):
    images: List[DiffusionResponse] = Field(..., description="List of generated images")
    total_processing_time: float = Field(..., description="Total processing time")

class TrainingResponse(BaseModel):
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    gpu_available: bool = Field(..., description="GPU availability")
    model_loaded: bool = Field(..., description="Model loading status")
```

## 5. API Endpoints

### Diffusion Endpoints (`app/api/v1/endpoints/diffusion.py`)
```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List
import asyncio
import io
from PIL import Image

from app.models.request import DiffusionRequest, BatchDiffusionRequest
from app.models.response import DiffusionResponse, BatchDiffusionResponse, ErrorResponse
from app.services.diffusion_service import DiffusionService
from app.api.deps import get_current_user, get_diffusion_service
from app.utils.validators import validate_image_file

router = APIRouter()

@router.post(
    "/generate",
    response_model=DiffusionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate image from text prompt"
)
async def generate_image(
    request: DiffusionRequest,
    background_tasks: BackgroundTasks,
    diffusion_service: DiffusionService = Depends(get_diffusion_service),
    current_user = Depends(get_current_user)
):
    """
    Generate an image from a text prompt using diffusion models.
    
    - **prompt**: Text description of the desired image
    - **negative_prompt**: Text description of what to avoid
    - **pipeline_type**: Type of diffusion pipeline to use
    - **num_inference_steps**: Number of denoising steps
    - **guidance_scale**: How closely to follow the prompt
    - **width/height**: Image dimensions (must be divisible by 8)
    - **seed**: Random seed for reproducible results
    """
    try:
        result = await diffusion_service.generate_image(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post(
    "/generate-batch",
    response_model=BatchDiffusionResponse,
    summary="Generate multiple images in batch"
)
async def generate_batch_images(
    request: BatchDiffusionRequest,
    diffusion_service: DiffusionService = Depends(get_diffusion_service),
    current_user = Depends(get_current_user)
):
    """Generate multiple images in a single batch request."""
    try:
        result = await diffusion_service.generate_batch(request.requests)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post(
    "/img2img",
    response_model=DiffusionResponse,
    summary="Transform image using diffusion model"
)
async def image_to_image(
    image: UploadFile = File(...),
    prompt: str = None,
    strength: float = 0.8,
    diffusion_service: DiffusionService = Depends(get_diffusion_service),
    current_user = Depends(get_current_user)
):
    """Transform an input image using diffusion model."""
    # Validate image file
    if not validate_image_file(image):
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    try:
        result = await diffusion_service.image_to_image(image, prompt, strength)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/models",
    summary="Get available models"
)
async def get_available_models(
    diffusion_service: DiffusionService = Depends(get_diffusion_service)
):
    """Get list of available diffusion models."""
    try:
        models = await diffusion_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint"
)
async def health_check(
    diffusion_service: DiffusionService = Depends(get_diffusion_service)
):
    """Check API health and model status."""
    try:
        health = await diffusion_service.get_health_status()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
```

## 6. Dependencies & Services

### Dependencies (`app/api/deps.py`)
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional

from app.core.config import settings
from app.services.diffusion_service import DiffusionService
from app.services.training_service import TrainingService

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Validate JWT token and return current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception

async def get_diffusion_service() -> DiffusionService:
    """Get diffusion service instance."""
    return DiffusionService()

async def get_training_service() -> TrainingService:
    """Get training service instance."""
    return TrainingService()
```

### Service Layer (`app/services/diffusion_service.py`)
```python
from typing import List, Dict, Any, Optional
import asyncio
import time
import logging
from pathlib import Path
import aiofiles
import uuid

from app.models.request import DiffusionRequest
from app.models.response import DiffusionResponse, BatchDiffusionResponse, HealthResponse
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

class DiffusionService:
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self._load_models()
    
    def _load_models(self):
        """Load diffusion models on startup."""
        try:
            # Load your diffusion models here
            logger.info("Loading diffusion models...")
            # Implementation depends on your specific models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def generate_image(self, request: DiffusionRequest) -> DiffusionResponse:
        """Generate a single image."""
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Generate image using appropriate pipeline
            image_data = await self._generate_with_pipeline(request)
            
            # Save image and get URL
            image_url = await self._save_image(image_data, request)
            
            processing_time = time.time() - start_time
            
            return DiffusionResponse(
                image_url=image_url,
                metadata={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "pipeline_type": request.pipeline_type,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "width": request.width,
                    "height": request.height,
                    "seed": request.seed
                },
                processing_time=processing_time,
                seed=request.seed
            )
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise
    
    async def generate_batch(self, requests: List[DiffusionRequest]) -> BatchDiffusionResponse:
        """Generate multiple images in batch."""
        start_time = time.time()
        
        try:
            # Process requests concurrently
            tasks = [self.generate_image(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            images = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch request {i}: {result}")
                    # Create error response for failed requests
                    images.append(DiffusionResponse(
                        image_url="",
                        metadata={"error": str(result)},
                        processing_time=0,
                        seed=None
                    ))
                else:
                    images.append(result)
            
            total_time = time.time() - start_time
            
            return BatchDiffusionResponse(
                images=images,
                total_processing_time=total_time
            )
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            raise
    
    async def image_to_image(self, image_file, prompt: str, strength: float) -> DiffusionResponse:
        """Transform image using diffusion model."""
        start_time = time.time()
        
        try:
            # Process image file
            image_data = await self._process_uploaded_image(image_file)
            
            # Generate transformed image
            result_image = await self._transform_image(image_data, prompt, strength)
            
            # Save and return
            image_url = await self._save_image(result_image, None)
            
            processing_time = time.time() - start_time
            
            return DiffusionResponse(
                image_url=image_url,
                metadata={"prompt": prompt, "strength": strength},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in image-to-image: {e}")
            raise
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        return [
            {
                "name": "stable-diffusion-v1-5",
                "type": "text-to-image",
                "description": "Stable Diffusion v1.5"
            },
            {
                "name": "stable-diffusion-xl",
                "type": "text-to-image",
                "description": "Stable Diffusion XL"
            }
        ]
    
    async def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        import psutil
        
        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            uptime=psutil.boot_time(),
            gpu_available=settings.DEVICE == "cuda",
            model_loaded=len(self.models) > 0
        )
    
    def _validate_request(self, request: DiffusionRequest):
        """Validate diffusion request."""
        if not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if request.width * request.height > 1024 * 1024:
            raise ValueError("Image dimensions too large")
    
    async def _generate_with_pipeline(self, request: DiffusionRequest):
        """Generate image using appropriate pipeline."""
        # Implementation depends on your specific diffusion pipeline
        # This is a placeholder
        await asyncio.sleep(1)  # Simulate processing
        return b"fake_image_data"
    
    async def _save_image(self, image_data: bytes, request: Optional[DiffusionRequest]) -> str:
        """Save image and return URL."""
        filename = f"{uuid.uuid4()}.png"
        filepath = Path(settings.MODEL_CACHE_DIR) / "generated" / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(image_data)
        
        # Return URL (implement your URL generation logic)
        return f"/generated/{filename}"
    
    async def _process_uploaded_image(self, image_file):
        """Process uploaded image file."""
        # Implementation for processing uploaded images
        pass
    
    async def _transform_image(self, image_data, prompt: str, strength: float):
        """Transform image using diffusion model."""
        # Implementation for image transformation
        pass
```

## 7. Error Handling & Validation

### Custom Exception Handlers
```python
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )

async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

# Register handlers in main.py
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValueError, value_error_handler)
```

### Input Validation (`app/utils/validators.py`)
```python
import magic
from typing import List
from fastapi import UploadFile, HTTPException

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file."""
    if not file.content_type in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed. Use: {ALLOWED_IMAGE_TYPES}"
        )
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )
    
    return True

def validate_prompt(prompt: str) -> bool:
    """Validate text prompt."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if len(prompt) > 1000:
        raise ValueError("Prompt too long. Maximum 1000 characters")
    
    # Add more validation as needed
    return True
```

## 8. Logging & Monitoring

### Logging Configuration (`app/utils/logging.py`)
```python
import logging
import sys
from pathlib import Path
from loguru import logger
import json

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    """Setup structured logging with loguru."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=True
    )
    
    # Add file handler
    log_file = Path("logs/notebooklm_ai.log")
    log_file.parent.mkdir(exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        serialize=True
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

def get_logger(name: str):
    """Get logger instance."""
    return logger.bind(name=name)
```

## 9. Testing

### Test Configuration (`tests/conftest.py`)
```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.config import settings
from app.core.database import get_db

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### API Tests (`tests/test_api/test_diffusion.py`)
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

def test_generate_image_success(client: TestClient):
    """Test successful image generation."""
    with patch('app.services.diffusion_service.DiffusionService.generate_image') as mock_generate:
        mock_generate.return_value = {
            "image_url": "/generated/test.png",
            "metadata": {"prompt": "test"},
            "processing_time": 1.5,
            "seed": 123
        }
        
        response = client.post(
            "/api/v1/diffusion/generate",
            json={
                "prompt": "A beautiful landscape",
                "width": 512,
                "height": 512
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["image_url"] == "/generated/test.png"
        assert data["processing_time"] == 1.5

def test_generate_image_invalid_request(client: TestClient):
    """Test image generation with invalid request."""
    response = client.post(
        "/api/v1/diffusion/generate",
        json={
            "prompt": "",  # Empty prompt
            "width": 513,  # Not divisible by 8
            "height": 512
        }
    )
    
    assert response.status_code == 422

def test_batch_generation(client: TestClient):
    """Test batch image generation."""
    with patch('app.services.diffusion_service.DiffusionService.generate_batch') as mock_batch:
        mock_batch.return_value = {
            "images": [
                {
                    "image_url": "/generated/test1.png",
                    "metadata": {"prompt": "test1"},
                    "processing_time": 1.0,
                    "seed": 123
                },
                {
                    "image_url": "/generated/test2.png",
                    "metadata": {"prompt": "test2"},
                    "processing_time": 1.2,
                    "seed": 456
                }
            ],
            "total_processing_time": 2.2
        }
        
        response = client.post(
            "/api/v1/diffusion/generate-batch",
            json={
                "requests": [
                    {"prompt": "Test 1", "width": 512, "height": 512},
                    {"prompt": "Test 2", "width": 512, "height": 512}
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 2
        assert data["total_processing_time"] == 2.2

def test_health_check(client: TestClient):
    """Test health check endpoint."""
    with patch('app.services.diffusion_service.DiffusionService.get_health_status') as mock_health:
        mock_health.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600.0,
            "gpu_available": True,
            "model_loaded": True
        }
        
        response = client.get("/api/v1/diffusion/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["gpu_available"] == True
```

## 10. Production Deployment

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/notebooklm_ai
      - SECRET_KEY=${SECRET_KEY}
      - DEVICE=cuda
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=notebooklm_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

## 11. Performance Optimization

### Async Processing
```python
# Background tasks for long-running operations
@router.post("/train-model")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    training_service: TrainingService = Depends(get_training_service)
):
    """Start model training in background."""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        training_service.train_model,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "started"}

# Connection pooling
from databases import Database

database = Database(settings.DATABASE_URL)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
```

### Caching
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time: int = 3600):
    """Cache decorator for API responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(
                cache_key,
                expire_time,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

@cache_result(expire_time=1800)  # Cache for 30 minutes
async def get_available_models():
    """Get available models with caching."""
    # Implementation
    pass
```

## 12. Security Best Practices

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/generate")
@limiter.limit("10/minute")  # 10 requests per minute
async def generate_image(
    request: Request,
    diffusion_request: DiffusionRequest,
    diffusion_service: DiffusionService = Depends(get_diffusion_service)
):
    # Implementation
    pass
```

### Input Sanitization
```python
import re
from html import escape

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user input prompt."""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', prompt)
    
    # HTML escape
    sanitized = escape(sanitized)
    
    # Limit length
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]
    
    return sanitized.strip()

# Use in service
async def generate_image(self, request: DiffusionRequest) -> DiffusionResponse:
    # Sanitize input
    request.prompt = sanitize_prompt(request.prompt)
    if request.negative_prompt:
        request.negative_prompt = sanitize_prompt(request.negative_prompt)
    
    # Continue with generation
    # ...
```

## 13. Monitoring & Metrics

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

MODEL_LOADED = Gauge(
    'diffusion_model_loaded',
    'Number of loaded diffusion models'
)

# Middleware to collect metrics
@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

## 14. Environment Configuration

### Environment Variables (.env)
```bash
# API Configuration
PROJECT_NAME=notebooklm_ai
VERSION=1.0.0
API_V1_STR=/api/v1

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60
ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://user:password@localhost/notebooklm_ai

# AI Models
MODEL_CACHE_DIR=./models
DEVICE=cuda
CUDA_AVAILABLE=true

# Performance
MAX_WORKERS=4
BATCH_SIZE=1

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
ALLOWED_HOSTS=["localhost","yourdomain.com"]

# Redis
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/notebooklm_ai.log
```

## 15. Best Practices Summary

### Code Organization
- Use dependency injection for services
- Separate concerns (API, business logic, data access)
- Use Pydantic models for request/response validation
- Implement proper error handling and logging

### Performance
- Use async/await for I/O operations
- Implement caching for expensive operations
- Use background tasks for long-running processes
- Monitor and optimize database queries

### Security
- Validate and sanitize all inputs
- Implement rate limiting
- Use HTTPS in production
- Secure sensitive configuration

### Testing
- Write unit tests for services
- Write integration tests for API endpoints
- Use test fixtures and mocking
- Test error conditions

### Monitoring
- Implement health checks
- Collect metrics and logs
- Set up alerting
- Monitor performance and errors

### Deployment
- Use Docker for containerization
- Implement CI/CD pipelines
- Use environment-specific configurations
- Set up proper logging and monitoring

This comprehensive FastAPI setup provides a production-ready foundation for your notebooklm_ai project with proper structure, security, performance optimization, and monitoring capabilities. 