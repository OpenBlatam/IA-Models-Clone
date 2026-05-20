# FastAPI-Specific Guidelines

## Project Structure

```
key_messages/
├── api/
│   ├── __init__.py
│   ├── router.py          # Main router with all sub-routes
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   └── evaluation.py
│   └── dependencies.py    # Shared dependencies
├── utils/
│   ├── __init__.py
│   ├── validation.py
│   └── error_handling.py
├── middleware/
│   ├── __init__.py
│   ├── logging.py
│   ├── monitoring.py
│   └── performance.py
├── static/
│   └── models/           # Saved model files
├── types/
│   ├── __init__.py
│   ├── models.py         # Pydantic models
│   └── schemas.py        # Request/response schemas
└── main.py
```

## Middleware for Logging, Monitoring, and Performance

### Logging Middleware (`middleware/logging.py`)
```python
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        self.logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        self.logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        return response
```

### Performance Monitoring Middleware (`middleware/monitoring.py`)
```python
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Any

class PerformanceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics_collector):
        super().__init__(app)
        self.metrics = metrics_collector
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Track request metrics
        self.metrics.increment_counter("requests_total", {"method": request.method})
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Track success metrics
            self.metrics.record_histogram("request_duration", process_time)
            self.metrics.increment_counter("requests_success", {"status": response.status_code})
            
            return response
            
        except Exception as e:
            # Track error metrics
            self.metrics.increment_counter("requests_error", {"error": type(e).__name__})
            raise
```

### Error Monitoring Middleware (`middleware/error_handling.py`)
```python
import traceback
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

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
                    "headers": dict(request.headers)
                }
            )
            raise
```

## HTTPException Patterns for Expected Errors

### Specific HTTP Status Codes
```python
from fastapi import HTTPException, status

# 400 Bad Request - Invalid input
@router.post("/train")
async def start_training(request: TrainingRequest) -> TrainingResponse:
    if not request.training_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training data is required"
        )
    
    if len(request.training_data) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 10 training samples required"
        )
    
    # Happy path
    return await perform_training(request)

# 404 Not Found - Resource doesn't exist
@router.get("/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    model = await load_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    return model

# 409 Conflict - Resource already exists
@router.post("/models")
async def create_model(request: CreateModelRequest) -> ModelInfo:
    if await model_exists(request.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {request.name} already exists"
        )
    
    return await create_new_model(request)

# 422 Unprocessable Entity - Validation errors
@router.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction = await generate_prediction(request.text)
        return PredictionResponse(prediction=prediction)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )

# 429 Too Many Requests - Rate limiting
@router.post("/batch-predict")
async def batch_predict(request: BatchRequest) -> BatchResponse:
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum 100 texts per batch allowed"
        )
    
    return await process_batch(request.texts)

# 503 Service Unavailable - Service temporarily unavailable
@router.post("/train")
async def start_training(request: TrainingRequest) -> TrainingResponse:
    if not await is_training_service_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training service is temporarily unavailable"
        )
    
    return await start_training_job(request)
```

### Custom Error Responses
```python
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str
    timestamp: str

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

# Usage in routes
@router.get("/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    model = await load_model(model_id)
    if not model:
        raise ModelNotFoundError(model_id)
    
    return model

@router.post("/train")
async def train_model(request: TrainingRequest) -> TrainingResponse:
    try:
        return await perform_training(request)
    except Exception as e:
        raise TrainingError(f"Training failed: {str(e)}")
```

## Performance Optimization

### Async Functions for I/O-bound Tasks
```python
import asyncio
import aiofiles
import aiohttp
from typing import List, Dict, Any

# Database operations
async def load_training_data(data_id: str) -> List[str]:
    """Async database query."""
    async with get_db_connection() as conn:
        result = await conn.fetch("SELECT text FROM training_data WHERE id = $1", data_id)
        return [row['text'] for row in result]

# File operations
async def save_model_checkpoint(model_state: Dict[str, Any], path: str) -> bool:
    """Async file writing."""
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(model_state))
    return True

# HTTP requests
async def fetch_external_data(url: str) -> Dict[str, Any]:
    """Async HTTP request."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Concurrent processing
async def process_batch_predictions(texts: List[str]) -> List[str]:
    """Concurrent batch processing."""
    tasks = [generate_prediction(text) for text in texts]
    return await asyncio.gather(*tasks)
```

### Caching Strategies
```python
from functools import lru_cache
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
import redis.asyncio as redis

# In-memory caching for expensive computations
@lru_cache(maxsize=128)
def calculate_expensive_metrics(data_hash: str) -> Dict[str, float]:
    """Cache expensive metric calculations."""
    return perform_expensive_calculation(data_hash)

# Redis-based caching for API responses
@router.get("/models/{model_id}")
@cache(expire=300)  # 5 minutes
async def get_model_info(model_id: str) -> ModelInfo:
    return await load_model_info(model_id)

# Custom caching with TTL
async def get_cached_prediction(text: str, model_id: str) -> str:
    cache_key = f"prediction:{model_id}:{hash(text)}"
    
    # Try to get from cache
    cached = await redis_client.get(cache_key)
    if cached:
        return cached.decode()
    
    # Generate prediction
    prediction = await generate_prediction(text, model_id)
    
    # Cache for 1 hour
    await redis_client.setex(cache_key, 3600, prediction)
    return prediction
```

### Lazy Loading Patterns
```python
from typing import Dict, Optional
import asyncio

class LazyModelLoader:
    def __init__(self):
        self._models: Dict[str, Optional[Model]] = {}
        self._loading_locks: Dict[str, asyncio.Lock] = {}
    
    async def get_model(self, model_id: str) -> Model:
        """Lazy load model with thread-safe loading."""
        if model_id not in self._models:
            if model_id not in self._loading_locks:
                self._loading_locks[model_id] = asyncio.Lock()
            
            async with self._loading_locks[model_id]:
                # Double-check after acquiring lock
                if model_id not in self._models:
                    self._models[model_id] = await self._load_model(model_id)
        
        return self._models[model_id]
    
    async def _load_model(self, model_id: str) -> Model:
        """Load model from disk."""
        return await load_model_from_disk(model_id)
    
    async def preload_models(self, model_ids: List[str]) -> None:
        """Preload multiple models concurrently."""
        tasks = [self.get_model(model_id) for model_id in model_ids]
        await asyncio.gather(*tasks)

# Dependency injection for lazy loading
async def get_model_loader() -> LazyModelLoader:
    return LazyModelLoader()

@router.post("/predict/{model_id}")
async def predict_with_lazy_loading(
    model_id: str,
    request: PredictionRequest,
    model_loader: LazyModelLoader = Depends(get_model_loader)
) -> PredictionResponse:
    model = await model_loader.get_model(model_id)
    prediction = await model.predict(request.text)
    return PredictionResponse(prediction=prediction)
```

## Main Application with Middleware

### Main App (`main.py`)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.router import router
from .middleware.logging import LoggingMiddleware
from .middleware.monitoring import PerformanceMiddleware
from .middleware.error_handling import ErrorMonitoringMiddleware
from .config import get_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    lifespan=lifespan
)

# Add middleware
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(PerformanceMiddleware, metrics_collector=get_metrics_collector())
app.add_middleware(ErrorMonitoringMiddleware, error_tracker=get_error_tracker())

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## Router Organization

### Main Router (`api/router.py`)
```python
from fastapi import APIRouter
from .routes import training, inference, evaluation

router = APIRouter(prefix="/key-messages", tags=["key-messages"])

router.include_router(training.router, prefix="/training")
router.include_router(inference.router, prefix="/inference")
router.include_router(evaluation.router, prefix="/evaluation")
```

### Sub-Routes (`api/routes/training.py`)
```python
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
from ...types.models import TrainingRequest, TrainingResponse
from ...utils.validation import validate_training_config

router = APIRouter()

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    config_valid: bool = Depends(validate_training_config)
) -> TrainingResponse:
    # Early validation with specific HTTP exceptions
    if not request.model_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model configuration required"
        )
    
    if not request.training_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training data required"
        )
    
    if len(request.training_data) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 10 training samples required"
        )
    
    # Happy path
    return TrainingResponse(
        job_id="train_123",
        status="started",
        model_config=request.model_config
    )
```

## Type Hints and Pydantic Models

### Request/Response Models (`types/models.py`)
```python
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum

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
    created_at: str
```

## Error Handling Patterns

### Early Returns with Validation
```python
@router.post("/predict")
async def predict_messages(
    request: PredictionRequest
) -> PredictionResponse:
    # Early validation with specific HTTP exceptions
    if not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input required"
        )
    
    if len(request.text) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too long (max 1000 characters)"
        )
    
    # Happy path
    prediction = await generate_prediction(request.text)
    return PredictionResponse(prediction=prediction)
```

### Dependency Injection for Validation
```python
# utils/validation.py
from fastapi import HTTPException, Depends, status
from typing import Dict, Any

async def validate_model_exists(model_id: str) -> bool:
    if not await model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return True

# routes/inference.py
@router.post("/predict/{model_id}")
async def predict_with_model(
    model_id: str,
    request: PredictionRequest,
    model_valid: bool = Depends(validate_model_exists)
) -> PredictionResponse:
    # Happy path - validation already done
    prediction = await predict(model_id, request.text)
    return PredictionResponse(prediction=prediction)
```

## Functional Programming Style

### Pure Functions
```python
def calculate_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Pure function for metric calculation."""
    accuracy = sum(p == t for p, t in zip(predictions, targets)) / len(predictions)
    return {"accuracy": accuracy}

def validate_config(config: Dict[str, Any]) -> bool:
    """Pure function for configuration validation."""
    required_keys = ["model_type", "batch_size", "learning_rate"]
    return all(key in config for key in required_keys)
```

### Async Functions for I/O Operations
```python
async def load_training_data(data_path: str) -> List[str]:
    """Async function for data loading."""
    async with aiofiles.open(data_path, 'r') as f:
        content = await f.read()
    return content.split('\n')

async def save_model_checkpoint(model_state: Dict[str, Any], path: str) -> bool:
    """Async function for model saving."""
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(model_state))
    return True
```

## RORO Pattern Implementation

### Request Objects
```python
class TrainingRequest(BaseModel):
    model_config: Dict[str, Any]
    training_data: List[str]
    hyperparameters: Dict[str, Any]
    experiment_name: str

class InferenceRequest(BaseModel):
    text: str
    model_id: str
    max_length: int = 50
    temperature: float = 0.7
```

### Response Objects
```python
class TrainingResponse(BaseModel):
    job_id: str
    status: str
    metrics: Dict[str, float]
    model_path: str

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str
    processing_time: float
```

## Named Exports

### Router Exports (`api/routes/__init__.py`)
```python
from .training import router as training_router
from .inference import router as inference_router
from .evaluation import router as evaluation_router

__all__ = ["training_router", "inference_router", "evaluation_router"]
```

### Utility Exports (`utils/__init__.py`)
```python
from .validation import validate_config, validate_model_exists
from .error_handling import handle_training_error, handle_inference_error

__all__ = [
    "validate_config",
    "validate_model_exists", 
    "handle_training_error",
    "handle_inference_error"
]
```

## Conditional Statements

### Avoid Unnecessary Braces
```python
# Good - no braces needed
if model_exists(model_id):
    return await load_model(model_id)

# Good - early return pattern
if not training_data:
    raise HTTPException(status_code=400, detail="No training data")

# Happy path
return await train_model(training_data)
```

### Single-Line Conditionals
```python
# Good - single line
if not config: return default_config

# Good - early return
if error_condition: raise HTTPException(status_code=500, detail="Error")

# Happy path
return process_data(data)
```

## Error Handling Best Practices

### Global Error Handler
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### Route-Level Error Handling
```python
@router.post("/train")
async def train_model(request: TrainingRequest) -> TrainingResponse:
    try:
        # Early validation
        if not request.training_data:
            raise HTTPException(status_code=400, detail="Training data required")
        
        # Happy path
        result = await perform_training(request)
        return TrainingResponse(**result)
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
```

## Configuration Integration

### Config Dependencies
```python
from ..config import get_config

async def get_training_config() -> Dict[str, Any]:
    config = get_config()
    return config.get("training", {})

@router.post("/train")
async def train_model(
    request: TrainingRequest,
    config: Dict[str, Any] = Depends(get_training_config)
) -> TrainingResponse:
    # Use config in training
    batch_size = config.get("batch_size", 32)
    # Happy path
    return await train_with_config(request, config)
```

## Testing Patterns

### Test Structure
```python
import pytest
from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_training_endpoint_success():
    """Test successful training request."""
    response = client.post(
        "/key-messages/training/start",
        json={
            "model_type": "gpt2",
            "model_config": {"hidden_size": 768},
            "training_data": ["sample text"] * 10
        }
    )
    assert response.status_code == 200
    assert "job_id" in response.json()

def test_training_endpoint_validation_error():
    """Test validation error handling."""
    response = client.post(
        "/key-messages/training/start",
        json={"model_type": "invalid"}
    )
    assert response.status_code == 422

def test_model_not_found_error():
    """Test 404 error handling."""
    response = client.get("/key-messages/models/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
```

## Best Practices Summary

1. **Use Pydantic models** for all request/response validation
2. **Implement early returns** for error conditions
3. **Use dependency injection** for shared validation logic
4. **Prefer async functions** for I/O operations
5. **Use named exports** for clean module organization
6. **Avoid unnecessary else statements** after returns
7. **Implement comprehensive error handling** at multiple levels
8. **Use type hints** for all function signatures
9. **Follow RORO pattern** for request/response objects
10. **Implement proper validation** with descriptive error messages
11. **Use specific HTTP status codes** for different error types
12. **Implement middleware** for logging, monitoring, and performance
13. **Use caching strategies** for performance optimization
14. **Implement lazy loading** for resource-intensive operations
15. **Use lifespan context managers** instead of startup/shutdown events 