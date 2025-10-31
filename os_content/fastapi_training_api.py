from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.json import pydantic_encoder
import redis.asyncio as redis
import aiofiles
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
import uvicorn
from contextlib import asynccontextmanager
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
FastAPI Training API for OS Content
Comprehensive API with async operations, middleware, error handling,
Pydantic validation, and performance optimizations.
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR VALIDATION
# ============================================================================

class TrainingConfigModel(BaseModel):
    """Pydantic model for training configuration validation."""
    model_config = ConfigDict(extra="forbid")
    
    # Model parameters
    model_name: str = Field(default="transformer", description="Model architecture name")
    hidden_size: int = Field(default=768, gt=0, description="Hidden layer size")
    num_layers: int = Field(default=12, gt=0, description="Number of transformer layers")
    num_heads: int = Field(default=12, gt=0, description="Number of attention heads")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")
    
    # Training parameters
    batch_size: int = Field(default=16, gt=0, description="Training batch size")
    learning_rate: float = Field(default=2e-5, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    num_epochs: int = Field(default=10, gt=0, description="Number of training epochs")
    gradient_accumulation_steps: int = Field(default=1, gt=0, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    
    # Data parameters
    train_split: float = Field(default=0.8, gt=0, lt=1, description="Training data split")
    val_split: float = Field(default=0.1, gt=0, lt=1, description="Validation data split")
    test_split: float = Field(default=0.1, gt=0, lt=1, description="Test data split")
    max_sequence_length: int = Field(default=512, gt=0, description="Maximum sequence length")
    
    # Optimization parameters
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    use_multi_gpu: bool = Field(default=False, description="Use multiple GPUs")
    num_workers: int = Field(default=4, ge=0, description="Number of data loader workers")
    
    # Early stopping and scheduling
    early_stopping_patience: int = Field(default=5, gt=0, description="Early stopping patience")
    early_stopping_min_delta: float = Field(default=0.001, ge=0, description="Early stopping minimum delta")
    warmup_steps: int = Field(default=100, ge=0, description="Learning rate warmup steps")
    lr_scheduler_type: str = Field(default="cosine", regex="^(cosine|linear|step|plateau)$", description="Learning rate scheduler type")
    
    # Logging and checkpointing
    log_interval: int = Field(default=100, gt=0, description="Logging interval")
    save_interval: int = Field(default=1000, gt=0, description="Checkpoint save interval")
    eval_interval: int = Field(default=500, gt=0, description="Evaluation interval")
    checkpoint_dir: str = Field(default="checkpoints", description="Checkpoint directory")
    
    @validator('train_split', 'val_split', 'test_split')
    def validate_splits(cls, v, values) -> bool:
        """Validate that data splits sum to 1.0."""
        if 'train_split' in values and 'val_split' in values:
            total = values['train_split'] + values['val_split'] + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Data splits must sum to 1.0, got {total}")
        return v

class TrainingRequestModel(BaseModel):
    """Pydantic model for training request validation."""
    model_config = ConfigDict(extra="forbid")
    
    config: TrainingConfigModel = Field(..., description="Training configuration")
    dataset_path: str = Field(..., description="Path to training dataset")
    model_path: Optional[str] = Field(None, description="Path to pre-trained model")
    experiment_name: str = Field(..., description="Experiment name for tracking")
    
    @validator('dataset_path')
    def validate_dataset_path(cls, v) -> bool:
        """Validate dataset path exists."""
        if not Path(v).exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return v

class TrainingResponseModel(BaseModel):
    """Pydantic model for training response."""
    model_config = ConfigDict(extra="forbid")
    
    training_id: str = Field(..., description="Unique training ID")
    status: str = Field(..., description="Training status")
    experiment_name: str = Field(..., description="Experiment name")
    created_at: datetime = Field(..., description="Training creation timestamp")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    message: str = Field(..., description="Response message")

class MetricsModel(BaseModel):
    """Pydantic model for training metrics."""
    model_config = ConfigDict(extra="forbid")
    
    epoch: int = Field(..., description="Current epoch")
    train_loss: float = Field(..., description="Training loss")
    val_loss: float = Field(..., description="Validation loss")
    train_acc: float = Field(..., description="Training accuracy")
    val_acc: float = Field(..., description="Validation accuracy")
    learning_rate: float = Field(..., description="Current learning rate")
    timestamp: datetime = Field(..., description="Metrics timestamp")

class ErrorResponseModel(BaseModel):
    """Pydantic model for error responses."""
    model_config = ConfigDict(extra="forbid")
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# ============================================================================
# DATABASE MODELS
# ============================================================================

Base = declarative_base()

class TrainingSession(Base):
    """Database model for training sessions."""
    __tablename__ = "training_sessions"
    
    id = Column(String, primary_key=True, index=True)
    experiment_name = Column(String, index=True)
    config = Column(Text)  # JSON string
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)  # JSON string

class TrainingMetrics(Base):
    """Database model for training metrics."""
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(String, index=True)
    epoch = Column(Integer)
    train_loss = Column(Float)
    val_loss = Column(Float)
    train_acc = Column(Float)
    val_acc = Column(Float)
    learning_rate = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class TrainingError(Exception):
    """Base exception for training-related errors."""
    def __init__(self, message: str, error_code: str = "TRAINING_ERROR"):
        
    """__init__ function."""
self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DatasetError(TrainingError):
    """Exception for dataset-related errors."""
    def __init__(self, message: str):
        
    """__init__ function."""
super().__init__(message, "DATASET_ERROR")

class ModelError(TrainingError):
    """Exception for model-related errors."""
    def __init__(self, message: str):
        
    """__init__ function."""
super().__init__(message, "MODEL_ERROR")

class ConfigurationError(TrainingError):
    """Exception for configuration-related errors."""
    def __init__(self, message: str):
        
    """__init__ function."""
super().__init__(message, "CONFIGURATION_ERROR")

# ============================================================================
# MIDDLEWARE AND ERROR HANDLING
# ============================================================================

async def error_handler_middleware(request: Request, call_next):
    """Middleware for handling unexpected errors and logging."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log successful requests
        logger.info(
            f"Request {request.method} {request.url.path} completed in {process_time:.4f}s "
            f"with status {response.status_code}"
        )
        
        return response
        
    except TrainingError as e:
        # Handle expected training errors
        process_time = time.time() - start_time
        error_response = ErrorResponseModel(
            error=str(e),
            error_code=e.error_code,
            timestamp=datetime.utcnow(),
            details={"process_time": process_time}
        )
        
        logger.error(f"Training error: {e.message} (Code: {e.error_code})")
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(error_response)
        )
        
    except Exception as e:
        # Handle unexpected errors
        process_time = time.time() - start_time
        error_response = ErrorResponseModel(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.utcnow(),
            details={"process_time": process_time}
        )
        
        logger.error(f"Unexpected error in {request.method} {request.url.path}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(error_response)
        )

# ============================================================================
# CACHE AND DATABASE UTILITIES
# ============================================================================

class CacheManager:
    """Async cache manager using Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis_client = None
    
    async def connect(self) -> Any:
        """Connect to Redis."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url)
    
    async def disconnect(self) -> Any:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        await self.connect()
        value = await self.redis_client.get(key)
        if value:
            return pickle.loads(value)
        return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration."""
        await self.connect()
        serialized_value = pickle.dumps(value)
        await self.redis_client.setex(key, expire, serialized_value)
    
    async def delete(self, key: str):
        """Delete value from cache."""
        await self.connect()
        await self.redis_client.delete(key)

class DatabaseManager:
    """Async database manager."""
    
    def __init__(self, database_url: str):
        
    """__init__ function."""
self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def connect(self) -> Any:
        """Connect to database."""
        if self.engine is None:
            self.engine = create_async_engine(self.database_url)
            self.session_factory = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
    
    async def disconnect(self) -> Any:
        """Disconnect from database."""
        if self.engine:
            await self.engine.dispose()
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        await self.connect()
        return self.session_factory()

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting FastAPI Training API...")
    
    # Initialize cache manager
    app.state.cache_manager = CacheManager()
    await app.state.cache_manager.connect()
    
    # Initialize database manager
    app.state.db_manager = DatabaseManager("sqlite+aiosqlite:///./training.db")
    await app.state.db_manager.connect()
    
    # Initialize training state
    app.state.active_trainings: Dict[str, Any] = {}
    app.state.training_lock = asyncio.Lock()
    
    logger.info("FastAPI Training API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI Training API...")
    
    # Disconnect cache
    await app.state.cache_manager.disconnect()
    
    # Disconnect database
    await app.state.db_manager.disconnect()
    
    logger.info("FastAPI Training API shutdown complete")

# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_cache_manager(request: Request) -> CacheManager:
    """Dependency for cache manager."""
    return request.app.state.cache_manager

async def get_db_session(request: Request) -> AsyncSession:
    """Dependency for database session."""
    return await request.app.state.db_manager.get_session()

async def get_training_lock(request: Request) -> asyncio.Lock:
    """Dependency for training lock."""
    return request.app.state.training_lock

# ============================================================================
# ASYNC UTILITY FUNCTIONS
# ============================================================================

async def load_dataset_async(dataset_path: str) -> Dataset:
    """Async function to load dataset."""
    # Simulate async dataset loading
    await asyncio.sleep(0.1)
    
    if not Path(dataset_path).exists():
        raise DatasetError(f"Dataset not found: {dataset_path}")
    
    # Create dummy dataset for demonstration
    class DummyDataset(Dataset):
        def __init__(self, num_samples=1000, input_size=784, num_classes=10) -> Any:
            self.data = torch.randn(num_samples, input_size)
            self.targets = torch.randint(0, num_classes, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    return DummyDataset()

async def save_training_session_async(
    session: AsyncSession,
    training_id: str,
    experiment_name: str,
    config: TrainingConfigModel
) -> None:
    """Async function to save training session to database."""
    training_session = TrainingSession(
        id=training_id,
        experiment_name=experiment_name,
        config=config.model_dump_json(),
        status="pending"
    )
    
    session.add(training_session)
    await session.commit()

async def update_training_status_async(
    session: AsyncSession,
    training_id: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """Async function to update training status."""
    # In a real implementation, this would update the database
    await asyncio.sleep(0.01)  # Simulate database operation

async def save_metrics_async(
    session: AsyncSession,
    training_id: str,
    metrics: MetricsModel
) -> None:
    """Async function to save training metrics."""
    # In a real implementation, this would save to database
    await asyncio.sleep(0.01)  # Simulate database operation

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def train_model_background(
    training_id: str,
    config: TrainingConfigModel,
    dataset_path: str,
    cache_manager: CacheManager,
    db_session: AsyncSession
) -> None:
    """Background task for model training."""
    try:
        # Update status to running
        await update_training_status_async(db_session, training_id, "running")
        
        # Load dataset
        dataset = await load_dataset_async(dataset_path)
        
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self, input_size=784, num_classes=10) -> Any:
                super().__init__()
                self.fc1 = nn.Linear(input_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, num_classes)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
            
            def forward(self, x) -> Any:
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0.0
            
            # Simulate training
            for batch_idx in range(10):  # Simplified training loop
                await asyncio.sleep(0.1)  # Simulate training time
                
                # Simulate loss calculation
                loss = torch.randn(1).item()
                total_loss += loss
                
                # Save metrics periodically
                if batch_idx % 5 == 0:
                    metrics = MetricsModel(
                        epoch=epoch,
                        train_loss=loss,
                        val_loss=loss * 1.1,
                        train_acc=0.8 + epoch * 0.02,
                        val_acc=0.75 + epoch * 0.015,
                        learning_rate=config.learning_rate,
                        timestamp=datetime.utcnow()
                    )
                    
                    await save_metrics_async(db_session, training_id, metrics)
                    
                    # Cache latest metrics
                    await cache_manager.set(f"metrics:{training_id}", metrics.model_dump())
            
            # Log epoch completion
            logger.info(f"Training {training_id} - Epoch {epoch+1}/{config.num_epochs} completed")
        
        # Update status to completed
        await update_training_status_async(db_session, training_id, "completed")
        
        # Cache final results
        final_metrics = {
            "status": "completed",
            "final_loss": total_loss / (config.num_epochs * 10),
            "completed_at": datetime.utcnow().isoformat()
        }
        await cache_manager.set(f"training:{training_id}", final_metrics, expire=86400)
        
    except Exception as e:
        logger.error(f"Training {training_id} failed: {str(e)}")
        await update_training_status_async(db_session, training_id, "failed", str(e))

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="OS Content Training API",
    description="Comprehensive API for training deep learning models",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom error handling middleware
app.middleware("http")(error_handler_middleware)

# ============================================================================
# API ROUTES
# ============================================================================

@app.post("/train", response_model=TrainingResponseModel)
async def start_training(
    request: TrainingRequestModel,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    db_session: AsyncSession = Depends(get_db_session),
    training_lock: asyncio.Lock = Depends(get_training_lock)
) -> TrainingResponseModel:
    """Start a new training session."""
    async with training_lock:
        # Generate training ID
        training_id = f"training_{int(time.time())}_{hash(request.experiment_name) % 10000}"
        
        # Validate configuration
        try:
            # Additional validation could go here
            pass
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {str(e)}")
        
        # Save training session to database
        await save_training_session_async(
            db_session, training_id, request.experiment_name, request.config
        )
        
        # Cache training info
        training_info = {
            "experiment_name": request.experiment_name,
            "config": request.config.model_dump(),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        await cache_manager.set(f"training:{training_id}", training_info, expire=86400)
        
        # Start background training task
        background_tasks.add_task(
            train_model_background,
            training_id,
            request.config,
            request.dataset_path,
            cache_manager,
            db_session
        )
        
        # Calculate estimated duration
        estimated_duration = request.config.num_epochs * 10 * 0.1  # Simplified calculation
        
        return TrainingResponseModel(
            training_id=training_id,
            status="pending",
            experiment_name=request.experiment_name,
            created_at=datetime.utcnow(),
            estimated_duration=int(estimated_duration),
            message="Training started successfully"
        )

@app.get("/training/{training_id}/status")
async def get_training_status(
    training_id: str,
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """Get training status and latest metrics."""
    # Try to get from cache first
    cached_data = await cache_manager.get(f"training:{training_id}")
    if cached_data:
        return cached_data
    
    # If not in cache, return not found
    raise HTTPException(status_code=404, detail="Training session not found")

@app.get("/training/{training_id}/metrics")
async def get_training_metrics(
    training_id: str,
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """Get latest training metrics."""
    # Try to get metrics from cache
    metrics = await cache_manager.get(f"metrics:{training_id}")
    if metrics:
        return metrics
    
    raise HTTPException(status_code=404, detail="Training metrics not found")

@app.get("/training/{training_id}/metrics/stream")
async def stream_training_metrics(
    training_id: str,
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Stream training metrics in real-time."""
    async def generate():
        
    """generate function."""
while True:
            metrics = await cache_manager.get(f"metrics:{training_id}")
            if metrics:
                yield f"data: {json.dumps(metrics)}\n\n"
            
            # Check if training is complete
            training_data = await cache_manager.get(f"training:{training_id}")
            if training_data and training_data.get("status") in ["completed", "failed"]:
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.delete("/training/{training_id}")
async def cancel_training(
    training_id: str,
    cache_manager: CacheManager = Depends(get_cache_manager),
    db_session: AsyncSession = Depends(get_db_session)
) -> Dict[str, str]:
    """Cancel a training session."""
    # Check if training exists
    training_data = await cache_manager.get(f"training:{training_id}")
    if not training_data:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Update status to cancelled
    await update_training_status_async(db_session, training_id, "cancelled")
    
    # Update cache
    training_data["status"] = "cancelled"
    await cache_manager.set(f"training:{training_id}", training_data, expire=3600)
    
    return {"message": "Training cancelled successfully"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    return {
        "active_trainings": len(app.state.active_trainings),
        "cache_connected": app.state.cache_manager.redis_client is not None,
        "db_connected": app.state.db_manager.engine is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_training_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 