#!/usr/bin/env python3
"""
Production API - Production-ready REST API for bulk optimization system
Provides HTTP endpoints for bulk optimization operations with authentication, rate limiting, and monitoring
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import traceback

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import bulk components
from bulk_optimizer import (
    create_bulk_optimizer, BulkOptimizer, BulkOptimizerConfig,
    OperationType
)
from bulk_optimization_core import (
    BulkOptimizationResult, create_bulk_optimization_core
)
from bulk_data_processor import (
    BulkDataset, create_bulk_data_processor
)
from production_config import ProductionConfigManager, Environment
from production_logging import create_production_logger
from production_monitoring import create_production_monitor

# API Models
class OptimizationRequest(BaseModel):
    """Optimization request model."""
    models: List[Dict[str, Any]] = Field(..., description="List of models to optimize")
    strategy: str = Field("auto", description="Optimization strategy")
    config: Optional[Dict[str, Any]] = Field(None, description="Optimization configuration")
    priority: int = Field(1, description="Request priority (1-10)")

class OptimizationResponse(BaseModel):
    """Optimization response model."""
    operation_id: str = Field(..., description="Operation ID")
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    estimated_time: Optional[float] = Field(None, description="Estimated completion time")

class OperationStatusResponse(BaseModel):
    """Operation status response model."""
    operation_id: str = Field(..., description="Operation ID")
    status: str = Field(..., description="Operation status")
    progress: float = Field(..., description="Progress percentage")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Operation results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    metrics: Dict[str, Any] = Field(..., description="System metrics")

class MetricsResponse(BaseModel):
    """Metrics response model."""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    system_metrics: Dict[str, Any] = Field(..., description="System metrics")
    application_metrics: Dict[str, Any] = Field(..., description="Application metrics")
    optimization_metrics: Dict[str, Any] = Field(..., description="Optimization metrics")

# Global variables
app = FastAPI(
    title="Bulk Optimization API",
    description="Production-ready API for bulk optimization operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global instances
config_manager: Optional[ProductionConfigManager] = None
bulk_optimizer: Optional[BulkOptimizer] = None
logger: Optional[logging.Logger] = None
monitor: Optional[Any] = None
start_time: float = time.time()

# Security
security = HTTPBearer()

# Rate limiting
rate_limit_storage = {}
rate_limit_requests = 100
rate_limit_window = 60

# Operation storage
operations = {}

def get_config_manager() -> ProductionConfigManager:
    """Get configuration manager."""
    global config_manager
    if config_manager is None:
        config_manager = ProductionConfigManager()
    return config_manager

def get_bulk_optimizer() -> BulkOptimizer:
    """Get bulk optimizer instance."""
    global bulk_optimizer
    if bulk_optimizer is None:
        config = get_config_manager().get_config()
        optimizer_config = BulkOptimizerConfig(
            max_models_per_batch=config.performance.max_workers,
            enable_parallel_optimization=True,
            optimization_strategies=['memory', 'computational', 'hybrid'],
            enable_optimization_core=True,
            enable_data_processor=True,
            enable_operation_manager=True,
            max_concurrent_operations=config.max_concurrent_operations,
            enable_performance_monitoring=True
        )
        bulk_optimizer = create_bulk_optimizer(optimizer_config.__dict__)
    return bulk_optimizer

def get_logger() -> logging.Logger:
    """Get logger instance."""
    global logger
    if logger is None:
        logger = create_production_logger("bulk_optimization_api")
    return logger

def get_monitor():
    """Get monitor instance."""
    global monitor
    if monitor is None:
        monitor = create_production_monitor()
        monitor.start()
    return monitor

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    if client_ip in rate_limit_storage:
        rate_limit_storage[client_ip] = [
            timestamp for timestamp in rate_limit_storage[client_ip]
            if current_time - timestamp < rate_limit_window
        ]
    else:
        rate_limit_storage[client_ip] = []
    
    # Check rate limit
    if len(rate_limit_storage[client_ip]) >= rate_limit_requests:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": rate_limit_window}
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    
    response = await call_next(request)
    return response

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    # In production, implement proper JWT verification
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Bulk Optimization API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        monitor = get_monitor()
        health_status = monitor.get_health_status()
        metrics_summary = monitor.get_metrics_summary()
        
        return HealthResponse(
            status=health_status['overall_status'],
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            uptime=time.time() - start_time,
            metrics=metrics_summary
        )
    except Exception as e:
        get_logger().error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    try:
        monitor = get_monitor()
        metrics_summary = monitor.get_metrics_summary()
        
        return MetricsResponse(
            timestamp=datetime.now(timezone.utc),
            system_metrics=metrics_summary.get('latest_metrics', {}),
            application_metrics={},
            optimization_metrics={}
        )
    except Exception as e:
        get_logger().error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_models(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Optimize models in bulk."""
    try:
        operation_id = str(uuid.uuid4())
        
        # Log operation start
        logger = get_logger()
        logger.log_operation_start(operation_id, "bulk_optimization")
        
        # Store operation
        operations[operation_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "request": request.dict()
        }
        
        # Start optimization in background
        background_tasks.add_task(
            run_optimization_task,
            operation_id,
            request
        )
        
        return OptimizationResponse(
            operation_id=operation_id,
            status="pending",
            message="Optimization started",
            estimated_time=estimate_optimization_time(request)
        )
        
    except Exception as e:
        get_logger().error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to start optimization")

@app.get("/operations/{operation_id}", response_model=OperationStatusResponse)
async def get_operation_status(
    operation_id: str,
    token: str = Depends(verify_token)
):
    """Get operation status."""
    if operation_id not in operations:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    operation = operations[operation_id]
    
    return OperationStatusResponse(
        operation_id=operation_id,
        status=operation["status"],
        progress=operation["progress"],
        results=operation.get("results"),
        error_message=operation.get("error_message"),
        created_at=operation["created_at"],
        updated_at=operation["updated_at"]
    )

@app.get("/operations", response_model=List[OperationStatusResponse])
async def list_operations(
    status: Optional[str] = None,
    limit: int = 100,
    token: str = Depends(verify_token)
):
    """List operations."""
    try:
        filtered_operations = []
        
        for op_id, operation in operations.items():
            if status is None or operation["status"] == status:
                filtered_operations.append(OperationStatusResponse(
                    operation_id=op_id,
                    status=operation["status"],
                    progress=operation["progress"],
                    results=operation.get("results"),
                    error_message=operation.get("error_message"),
                    created_at=operation["created_at"],
                    updated_at=operation["updated_at"]
                ))
        
        # Sort by creation time (newest first)
        filtered_operations.sort(key=lambda x: x.created_at, reverse=True)
        
        return filtered_operations[:limit]
        
    except Exception as e:
        get_logger().error(f"Failed to list operations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list operations")

@app.delete("/operations/{operation_id}")
async def cancel_operation(
    operation_id: str,
    token: str = Depends(verify_token)
):
    """Cancel an operation."""
    if operation_id not in operations:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    operation = operations[operation_id]
    
    if operation["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Operation cannot be cancelled")
    
    operation["status"] = "cancelled"
    operation["updated_at"] = datetime.now(timezone.utc)
    
    return {"message": "Operation cancelled"}

@app.get("/alerts", response_model=Dict[str, Any])
async def get_alerts(token: str = Depends(verify_token)):
    """Get active alerts."""
    try:
        monitor = get_monitor()
        alerts_summary = monitor.get_alerts_summary()
        return alerts_summary
    except Exception as e:
        get_logger().error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")

# Background tasks
async def run_optimization_task(operation_id: str, request: OptimizationRequest):
    """Run optimization task in background."""
    try:
        logger = get_logger()
        logger.info(f"Starting optimization task {operation_id}")
        
        # Update operation status
        operations[operation_id]["status"] = "running"
        operations[operation_id]["progress"] = 10.0
        operations[operation_id]["updated_at"] = datetime.now(timezone.utc)
        
        # Create models from request
        models = []
        for model_data in request.models:
            # In production, you would load actual models here
            # For now, create dummy models
            import torch
            import torch.nn as nn
            
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return self.linear(x)
            
            model_name = model_data.get("name", f"model_{len(models)}")
            model = DummyModel()
            models.append((model_name, model))
        
        # Run optimization
        optimizer = get_bulk_optimizer()
        results = optimizer.optimize_models_bulk(models, request.strategy)
        
        # Update operation status
        operations[operation_id]["status"] = "completed"
        operations[operation_id]["progress"] = 100.0
        operations[operation_id]["results"] = [asdict(result) for result in results]
        operations[operation_id]["updated_at"] = datetime.now(timezone.utc)
        
        logger.log_operation_end(operation_id, True, time.time())
        logger.info(f"Optimization task {operation_id} completed")
        
    except Exception as e:
        logger = get_logger()
        logger.log_error(e, {"operation_id": operation_id})
        
        # Update operation status
        operations[operation_id]["status"] = "failed"
        operations[operation_id]["error_message"] = str(e)
        operations[operation_id]["updated_at"] = datetime.now(timezone.utc)
        
        logger.log_operation_end(operation_id, False, time.time())

def estimate_optimization_time(request: OptimizationRequest) -> float:
    """Estimate optimization time."""
    # Simple estimation based on number of models
    base_time = 10.0  # seconds
    per_model_time = 5.0  # seconds per model
    return base_time + (len(request.models) * per_model_time)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger = get_logger()
    logger.log_error(exc, {"endpoint": str(request.url)})
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logger = get_logger()
    logger.info("Bulk Optimization API starting up")
    
    # Initialize monitor
    monitor = get_monitor()
    logger.info("Production monitoring started")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logger = get_logger()
    logger.info("Bulk Optimization API shutting down")
    
    # Stop monitor
    if monitor:
        monitor.stop()
        logger.info("Production monitoring stopped")

# Main function
def create_app() -> FastAPI:
    """Create FastAPI application."""
    return app

def run_production_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    reload: bool = False
):
    """Run production server."""
    uvicorn.run(
        "production_api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Bulk Optimization API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable reload")
    
    args = parser.parse_args()
    
    print("🚀 Starting Bulk Optimization API")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    
    run_production_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )

