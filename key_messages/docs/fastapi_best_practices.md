# FastAPI Best Practices - Official Documentation Alignment

## Data Models (Pydantic)

### Basic Model Structure
```python
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class ModelType(str, Enum):
    GPT2 = "gpt2"
    BERT = "bert"
    CUSTOM = "custom"

class TrainingRequest(BaseModel):
    model_type: ModelType = Field(..., description="Type of model to train")
    model_config: Dict[str, Any] = Field(..., description="Model configuration")
    training_data: List[str] = Field(..., min_items=1, description="Training data")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('training_data')
    def validate_training_data(cls, v):
        if not v or len(v) < 10:
            raise ValueError("At least 10 training samples required")
        return v

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    model_config: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PredictionRequest(BaseModel):
    text: str = Field(..., max_length=1000, description="Input text for prediction")
    model_id: str = Field(..., description="Model ID to use for prediction")
    max_length: int = Field(default=50, ge=1, le=500, description="Maximum output length")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    processing_time: float
    tokens_generated: int
```

### Advanced Model Patterns
```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import Union, Literal

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model_id: str
    batch_size: int = Field(default=32, ge=1, le=128)
    
    @validator('texts')
    def validate_texts(cls, v):
        if any(len(text) > 1000 for text in v):
            raise ValueError("All texts must be 1000 characters or less")
        return v

class ModelInfo(BaseModel):
    id: str
    name: str
    type: ModelType
    version: str
    created_at: datetime
    status: Literal["active", "inactive", "training"]
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "model_123",
                "name": "GPT-2 Key Messages",
                "type": "gpt2",
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z",
                "status": "active",
                "metrics": {"accuracy": 0.95, "loss": 0.12}
            }
        }

class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
```

## Path Operations

### RESTful Endpoints
```python
from fastapi import APIRouter, HTTPException, Depends, status, Query, Path
from typing import List, Optional

router = APIRouter()

@router.post("/train", response_model=TrainingResponse, status_code=status.HTTP_201_CREATED)
async def start_training(
    request: TrainingRequest,
    config_valid: bool = Depends(validate_training_config)
) -> TrainingResponse:
    if not request.model_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model configuration required"
        )
    
    if len(request.training_data) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 10 training samples required"
        )
    
    job_id = await create_training_job(request)
    return TrainingResponse(
        job_id=job_id,
        status="started",
        model_config=request.model_config
    )

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    skip: int = Query(default=0, ge=0, description="Number of models to skip"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of models to return"),
    model_type: Optional[ModelType] = Query(None, description="Filter by model type"),
    status_filter: Optional[str] = Query(None, description="Filter by status")
) -> List[ModelInfo]:
    models = await get_models(skip=skip, limit=limit, model_type=model_type, status=status_filter)
    return models

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str = Path(..., description="Model ID to retrieve")
) -> ModelInfo:
    model = await load_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return model

@router.post("/predict", response_model=PredictionResponse)
async def predict_messages(
    request: PredictionRequest
) -> PredictionResponse:
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input cannot be empty"
        )
    
    start_time = time.time()
    prediction = await generate_prediction(request.text, request.model_id, request.max_length, request.temperature)
    processing_time = time.time() - start_time
    
    return PredictionResponse(
        prediction=prediction["text"],
        confidence=prediction["confidence"],
        model_used=request.model_id,
        processing_time=processing_time,
        tokens_generated=prediction["tokens"]
    )

@router.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict(
    request: BatchRequest
) -> List[PredictionResponse]:
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum 100 texts per batch allowed"
        )
    
    predictions = await process_batch_predictions(
        request.texts, 
        request.model_id, 
        request.batch_size
    )
    return predictions

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str = Path(..., description="Model ID to delete")
):
    success = await delete_model_by_id(model_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
```

### Dependency Injection
```python
from fastapi import Depends, HTTPException, status
from typing import Optional

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    user = await authenticate_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user

async def validate_model_exists(model_id: str) -> Model:
    model = await load_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return model

async def get_training_config() -> Dict[str, Any]:
    config = get_config()
    return config.get("training", {})

@router.post("/predict/{model_id}")
async def predict_with_model(
    model_id: str,
    request: PredictionRequest,
    model: Model = Depends(validate_model_exists),
    user: User = Depends(get_current_user)
) -> PredictionResponse:
    # Model validation already done by dependency
    prediction = await model.predict(request.text)
    return PredictionResponse(
        prediction=prediction["text"],
        confidence=prediction["confidence"],
        model_used=model_id,
        processing_time=prediction["time"],
        tokens_generated=prediction["tokens"]
    )
```

## Middleware

### Custom Middleware Implementation
```python
import time
import logging
import json
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Any
from contextlib import asynccontextmanager

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Log request
        self.logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            self.logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": f"{process_time:.3f}s"
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": f"{process_time:.3f}s"
                }
            )
            raise

class PerformanceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics_collector):
        super().__init__(app)
        self.metrics = metrics_collector
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Track request metrics
        self.metrics.increment_counter("requests_total", {
            "method": request.method,
            "path": request.url.path
        })
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Track success metrics
            self.metrics.record_histogram("request_duration", process_time, {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code
            })
            self.metrics.increment_counter("requests_success", {
                "status_code": response.status_code
            })
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Track error metrics
            self.metrics.increment_counter("requests_error", {
                "error": type(e).__name__,
                "method": request.method,
                "path": request.url.path
            })
            self.metrics.record_histogram("request_duration", process_time, {
                "method": request.method,
                "path": request.url.path,
                "error": type(e).__name__
            })
            raise

class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, error_tracker):
        super().__init__(app)
        self.error_tracker = error_tracker
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions (expected errors)
            raise
        except Exception as e:
            # Track unexpected errors
            self.error_tracker.capture_exception(
                exception=e,
                context={
                    "url": str(request.url),
                    "method": request.method,
                    "headers": dict(request.headers),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            )
            raise

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not await self.rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return await call_next(request)
```

### Application Setup with Middleware
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Key Messages API")
    await initialize_services()
    yield
    # Shutdown
    logger.info("Shutting down Key Messages API")
    await cleanup_services()

app = FastAPI(
    title="Key Messages API",
    description="ML pipeline for key message generation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add custom middleware
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(PerformanceMiddleware, metrics_collector=get_metrics_collector())
app.add_middleware(ErrorMonitoringMiddleware, error_tracker=get_error_tracker())
app.add_middleware(RateLimitingMiddleware, rate_limiter=get_rate_limiter())

# Add built-in middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(router, prefix="/api/v1")

# Health check
@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Global exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

## Error Handling

### Custom Exception Classes
```python
from fastapi import HTTPException, status

class ModelNotFoundError(HTTPException):
    def __init__(self, model_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

class TrainingError(HTTPException):
    def __init__(self, message: str, error_code: str = "TRAINING_FAILED"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )

class ValidationError(HTTPException):
    def __init__(self, field: str, message: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error in field '{field}': {message}"
        )

class RateLimitError(HTTPException):
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
```

### Route-Level Error Handling
```python
@router.post("/train")
async def train_model(request: TrainingRequest) -> TrainingResponse:
    try:
        # Early validation
        if not request.training_data:
            raise ValidationError("training_data", "Training data is required")
        
        if len(request.training_data) < 10:
            raise ValidationError("training_data", "At least 10 training samples required")
        
        # Happy path
        result = await perform_training(request)
        return TrainingResponse(**result)
        
    except ValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Log and raise generic error
        logger.error(f"Training failed: {str(e)}")
        raise TrainingError(f"Training failed: {str(e)}")
```

## Testing

### Test Structure
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from ..main import app

client = TestClient(app)

class TestTrainingEndpoints:
    def test_start_training_success(self):
        """Test successful training request."""
        response = client.post(
            "/api/v1/train",
            json={
                "model_type": "gpt2",
                "model_config": {"hidden_size": 768},
                "training_data": ["sample text"] * 10
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"
    
    def test_start_training_validation_error(self):
        """Test validation error handling."""
        response = client.post(
            "/api/v1/train",
            json={
                "model_type": "invalid_type",
                "model_config": {},
                "training_data": []
            }
        )
        assert response.status_code == 422
    
    def test_start_training_insufficient_data(self):
        """Test insufficient training data error."""
        response = client.post(
            "/api/v1/train",
            json={
                "model_type": "gpt2",
                "model_config": {"hidden_size": 768},
                "training_data": ["sample text"] * 5  # Less than 10
            }
        )
        assert response.status_code == 400
        assert "At least 10 training samples required" in response.json()["detail"]

class TestPredictionEndpoints:
    @patch('app.services.generate_prediction')
    async def test_predict_success(self, mock_generate):
        """Test successful prediction."""
        mock_generate.return_value = {
            "text": "Generated prediction",
            "confidence": 0.95,
            "tokens": 50
        }
        
        response = client.post(
            "/api/v1/predict",
            json={
                "text": "Input text",
                "model_id": "model_123",
                "max_length": 50,
                "temperature": 0.7
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Generated prediction"
        assert data["confidence"] == 0.95
    
    def test_predict_empty_text(self):
        """Test empty text validation."""
        response = client.post(
            "/api/v1/predict",
            json={
                "text": "",
                "model_id": "model_123"
            }
        )
        assert response.status_code == 400
        assert "Text input cannot be empty" in response.json()["detail"]

class TestModelEndpoints:
    def test_get_model_not_found(self):
        """Test 404 error handling."""
        response = client.get("/api/v1/models/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_list_models_pagination(self):
        """Test pagination parameters."""
        response = client.get("/api/v1/models?skip=0&limit=5")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
```

## Configuration

### Environment Configuration
```python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "Key Messages API"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 3600
    
    # External APIs
    external_api_url: str
    external_api_timeout: int = 30
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

This implementation follows all FastAPI documentation best practices for Data Models, Path Operations, and Middleware, providing a production-ready foundation for building scalable and maintainable APIs. 