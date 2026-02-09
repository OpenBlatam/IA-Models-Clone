#!/usr/bin/env python3
"""
Model Serving and API Endpoints for Frontier Model Training
Provides REST API, GraphQL, and WebSocket endpoints for model inference.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from starlette.websockets import WebSocketState
import redis
import celery
from celery import Celery
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

console = Console()

class ModelStatus(Enum):
    """Model serving status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"

class RequestPriority(Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ModelConfig:
    """Configuration for model serving."""
    model_name: str
    model_path: str
    model_type: str = "transformer"
    max_batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "auto"
    precision: str = "fp16"
    enable_cache: bool = True
    cache_size: int = 1000
    enable_batching: bool = True
    batch_timeout: float = 0.1
    max_concurrent_requests: int = 100
    enable_metrics: bool = True
    enable_logging: bool = True

@dataclass
class InferenceRequest:
    """Inference request structure."""
    request_id: str
    input_data: Any
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: float = 30.0
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

@dataclass
class InferenceResponse:
    """Inference response structure."""
    request_id: str
    output_data: Any
    processing_time: float
    model_name: str
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class ModelCache:
    """LRU cache for model predictions."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

class RequestQueue:
    """Priority queue for inference requests."""
    
    def __init__(self):
        self.queues = {
            RequestPriority.CRITICAL: queue.PriorityQueue(),
            RequestPriority.HIGH: queue.PriorityQueue(),
            RequestPriority.NORMAL: queue.PriorityQueue(),
            RequestPriority.LOW: queue.PriorityQueue()
        }
        self.lock = threading.Lock()
    
    def put(self, request: InferenceRequest) -> None:
        """Add request to appropriate queue."""
        priority = request.priority
        timestamp = time.time()
        self.queues[priority].put((timestamp, request))
    
    def get(self, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        """Get next request from queues in priority order."""
        for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH, 
                        RequestPriority.NORMAL, RequestPriority.LOW]:
            try:
                timestamp, request = self.queues[priority].get(timeout=timeout)
                return request
            except queue.Empty:
                continue
        return None
    
    def size(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self.queues.values())

class ModelServer:
    """Main model server with inference capabilities."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.model = None
        self.tokenizer = None
        self.device = None
        self.status = ModelStatus.LOADING
        
        # Infrastructure
        self.cache = ModelCache(config.cache_size) if config.enable_cache else None
        self.request_queue = RequestQueue()
        self.metrics = {}
        self.active_connections = set()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        self.batch_thread = None
        self.batch_queue = queue.Queue()
        self.running = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            console.print(f"[blue]Loading model: {self.config.model_name}[/blue]")
            
            # Determine device
            if self.config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.config.device)
            
            # Load model (simplified - in practice, you'd load your actual model)
            if self.config.model_type == "transformer":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                
                if self.device.type == "cpu":
                    self.model = self.model.to(self.device)
            
            self.status = ModelStatus.READY
            console.print(f"[green]Model loaded successfully on {self.device}[/green]")
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, input_data: Any, request_id: str = None) -> InferenceResponse:
        """Single prediction."""
        if self.status != ModelStatus.READY:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        request_id = request_id or secrets.token_hex(8)
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = None
            if self.cache:
                cache_key = self._generate_cache_key(input_data)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return InferenceResponse(
                        request_id=request_id,
                        output_data=cached_result,
                        processing_time=time.time() - start_time,
                        model_name=self.config.model_name,
                        metadata={"cached": True}
                    )
            
            # Perform inference
            output_data = self._inference(input_data)
            
            # Cache result
            if self.cache and cache_key:
                self.cache.put(cache_key, output_data)
            
            # Update metrics
            self._update_metrics(time.time() - start_time)
            
            return InferenceResponse(
                request_id=request_id,
                output_data=output_data,
                processing_time=time.time() - start_time,
                model_name=self.config.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return InferenceResponse(
                request_id=request_id,
                output_data=None,
                processing_time=time.time() - start_time,
                model_name=self.config.model_name,
                status="error",
                error_message=str(e)
            )
    
    def predict_batch(self, input_data_list: List[Any]) -> List[InferenceResponse]:
        """Batch prediction."""
        if self.status != ModelStatus.READY:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        start_time = time.time()
        responses = []
        
        try:
            # Process batch
            batch_outputs = self._batch_inference(input_data_list)
            
            # Create responses
            for i, output_data in enumerate(batch_outputs):
                response = InferenceResponse(
                    request_id=secrets.token_hex(8),
                    output_data=output_data,
                    processing_time=time.time() - start_time,
                    model_name=self.config.model_name
                )
                responses.append(response)
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch inference error: {e}")
            # Return error responses for all inputs
            for i in range(len(input_data_list)):
                response = InferenceResponse(
                    request_id=secrets.token_hex(8),
                    output_data=None,
                    processing_time=time.time() - start_time,
                    model_name=self.config.model_name,
                    status="error",
                    error_message=str(e)
                )
                responses.append(response)
            
            return responses
    
    def _inference(self, input_data: Any) -> Any:
        """Perform single inference."""
        if self.config.model_type == "transformer":
            # Tokenize input
            if isinstance(input_data, str):
                inputs = self.tokenizer(input_data, return_tensors="pt", 
                                      max_length=self.config.max_sequence_length,
                                      truncation=True, padding=True)
            else:
                inputs = input_data
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_sequence_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return output_text
        
        else:
            # Generic inference
            return {"prediction": "sample_output"}
    
    def _batch_inference(self, input_data_list: List[Any]) -> List[Any]:
        """Perform batch inference."""
        if self.config.model_type == "transformer":
            # Tokenize batch
            inputs = self.tokenizer(
                input_data_list,
                return_tensors="pt",
                max_length=self.config.max_sequence_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_sequence_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs
            output_texts = []
            for output in outputs:
                output_text = self.tokenizer.decode(output, skip_special_tokens=True)
                output_texts.append(output_text)
            
            return output_texts
        
        else:
            # Generic batch inference
            return [{"prediction": f"sample_output_{i}"} for i in range(len(input_data_list))]
    
    def _generate_cache_key(self, input_data: Any) -> str:
        """Generate cache key for input data."""
        if isinstance(input_data, str):
            return hashlib.md5(input_data.encode()).hexdigest()
        else:
            return hashlib.md5(str(input_data).encode()).hexdigest()
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics."""
        if not self.config.enable_metrics:
            return
        
        current_time = time.time()
        
        # Update metrics
        if "total_requests" not in self.metrics:
            self.metrics = {
                "total_requests": 0,
                "total_processing_time": 0.0,
                "avg_processing_time": 0.0,
                "requests_per_second": 0.0,
                "last_update": current_time
            }
        
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += processing_time
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_requests"]
        )
        
        # Calculate requests per second
        time_diff = current_time - self.metrics["last_update"]
        if time_diff > 0:
            self.metrics["requests_per_second"] = 1.0 / time_diff
        
        self.metrics["last_update"] = current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.metrics.copy()
        
        # Add system metrics
        metrics["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_used": 0.0,
            "gpu_memory_total": 0.0,
            "gpu_utilization": 0.0
        }
        
        if torch.cuda.is_available():
            metrics["system"]["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**2
            metrics["system"]["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            metrics["system"]["gpu_utilization"] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        # Add cache metrics
        if self.cache:
            metrics["cache"] = {
                "size": len(self.cache.cache),
                "max_size": self.cache.max_size,
                "hit_rate": 0.0  # Would need to track hits/misses
            }
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": self.status.value,
            "model_name": self.config.model_name,
            "device": str(self.device),
            "uptime": time.time() - getattr(self, 'start_time', time.time()),
            "active_connections": len(self.active_connections),
            "queue_size": self.request_queue.size()
        }

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Prediction request model."""
    input_data: Union[str, List[str], Dict[str, Any]]
    request_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)

class PredictionResponse(BaseModel):
    """Prediction response model."""
    request_id: str
    output_data: Any
    processing_time: float
    model_name: str
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    input_data_list: List[Union[str, Dict[str, Any]]]
    batch_size: Optional[int] = Field(default=None, ge=1, le=100)

class MetricsResponse(BaseModel):
    """Metrics response model."""
    metrics: Dict[str, Any]
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_name: str
    device: str
    uptime: float
    active_connections: int
    queue_size: int

# FastAPI application
app = FastAPI(
    title="Frontier Model API",
    description="Advanced model serving API with real-time inference",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Global model server
model_server: Optional[ModelServer] = None

@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup."""
    global model_server
    
    # Load configuration
    config = ModelConfig(
        model_name="frontier-model",
        model_path="./models/frontier-model",
        model_type="transformer",
        max_batch_size=32,
        max_sequence_length=512,
        device="auto",
        precision="fp16",
        enable_cache=True,
        cache_size=1000,
        enable_batching=True,
        batch_timeout=0.1,
        max_concurrent_requests=100,
        enable_metrics=True,
        enable_logging=True
    )
    
    model_server = ModelServer(config)
    model_server.start_time = time.time()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global model_server
    if model_server:
        model_server.running = False

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    try:
        response = model_server.predict(
            input_data=request.input_data,
            request_id=request.request_id
        )
        
        return PredictionResponse(**asdict(response))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    try:
        responses = model_server.predict_batch(request.input_data_list)
        
        return [PredictionResponse(**asdict(response)) for response in responses]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model metrics."""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    metrics = model_server.get_metrics()
    
    return MetricsResponse(
        metrics=metrics,
        timestamp=datetime.now()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    health = model_server.health_check()
    
    return HealthResponse(**health)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time inference."""
    await websocket.accept()
    
    if model_server:
        model_server.active_connections.add(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Process request
            if model_server:
                response = model_server.predict(
                    input_data=request_data.get("input_data"),
                    request_id=request_data.get("request_id")
                )
                
                # Send response
                await websocket.send_text(json.dumps(asdict(response)))
            else:
                await websocket.send_text(json.dumps({
                    "error": "Model server not available"
                }))
    
    except WebSocketDisconnect:
        if model_server:
            model_server.active_connections.discard(websocket)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Frontier Model API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics",
            "health": "/health",
            "websocket": "/ws",
            "docs": "/docs"
        }
    }

def create_celery_app():
    """Create Celery app for background tasks."""
    celery_app = Celery(
        "frontier_model",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/0"
    )
    
    @celery_app.task
    def background_inference(input_data, request_id=None):
        """Background inference task."""
        if model_server:
            response = model_server.predict(input_data, request_id)
            return asdict(response)
        return None
    
    return celery_app

def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Frontier Model API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--model-path", type=str, default="./models/frontier-model", 
                       help="Path to model")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--precision", type=str, default="fp16", 
                       choices=["fp32", "fp16", "bf16"], help="Precision to use")
    
    args = parser.parse_args()
    
    console.print(f"[bold blue]Starting Frontier Model API Server[/bold blue]")
    console.print(f"Host: {args.host}")
    console.print(f"Port: {args.port}")
    console.print(f"Workers: {args.workers}")
    console.print(f"Model Path: {args.model_path}")
    console.print(f"Device: {args.device}")
    console.print(f"Precision: {args.precision}")
    
    # Run server
    uvicorn.run(
        "model_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )

if __name__ == "__main__":
    main()
