#!/usr/bin/env python3
"""
Enhanced Production API - Advanced FastAPI with enterprise features
Enhanced with advanced authentication, rate limiting, caching, and performance optimizations
"""

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import logging
import traceback
from functools import wraps
import redis
import aioredis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, validator
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import jwt
from passlib.context import CryptContext
import httpx
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Import enhanced components
from enhanced_production_config import (
    EnhancedProductionConfigManager, EnhancedProductionConfig, Environment
)
from production_logging import create_production_logger
from production_monitoring import create_production_monitor

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Active HTTP connections')
OPTIMIZATION_OPERATIONS = Counter('optimization_operations_total', 'Total optimization operations', ['status'])
OPTIMIZATION_DURATION = Histogram('optimization_duration_seconds', 'Optimization duration')

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Enhanced API Models
class UserCredentials(BaseModel):
    """User credentials model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    email: Optional[str] = Field(None, regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class EnhancedOptimizationRequest(BaseModel):
    """Enhanced optimization request model."""
    models: List[Dict[str, Any]] = Field(..., description="List of models to optimize")
    strategy: str = Field("auto", description="Optimization strategy")
    config: Optional[Dict[str, Any]] = Field(None, description="Optimization configuration")
    priority: int = Field(1, ge=1, le=10, description="Request priority (1-10)")
    timeout: Optional[int] = Field(None, ge=60, le=3600, description="Operation timeout in seconds")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('models')
    def validate_models(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one model is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 models allowed per request')
        return v

class OptimizationProgress(BaseModel):
    """Optimization progress model."""
    operation_id: str
    status: str
    progress: float = Field(..., ge=0, le=100)
    current_stage: str
    models_processed: int
    total_models: int
    estimated_completion: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    """System status model."""
    status: str
    version: str
    uptime: float
    timestamp: datetime
    health_checks: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    active_operations: int
    queue_size: int

class CacheManager:
    """Advanced caching manager."""
    
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis_pool = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            self.redis_pool = aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            await self.redis_pool.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_pool = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis_pool:
            return None
        
        try:
            value = await self.redis_pool.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.redis_pool:
            return False
        
        try:
            ttl = ttl or self.ttl
            await self.redis_pool.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.redis_pool:
            return False
        
        try:
            await self.redis_pool.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        if not self.redis_pool:
            return 0
        
        try:
            keys = await self.redis_pool.keys(pattern)
            if keys:
                return await self.redis_pool.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"Cache clear pattern error: {e}")
            return 0

class AuthenticationManager:
    """Advanced authentication manager."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        self.user_sessions = {}  # In production, use Redis or database
    
    def create_access_token(self, username: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        to_encode = {
            "sub": username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, username: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode = {
            "sub": username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return pwd_context.verify(plain_password, hashed_password)

class EnhancedProductionAPI:
    """Enhanced production API with advanced features."""
    
    def __init__(self, config: EnhancedProductionConfig):
        self.config = config
        self.app = FastAPI(
            title="Enhanced Bulk Optimization API",
            description="Advanced production API for bulk optimization operations",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Initialize components
        self.cache_manager = CacheManager(
            f"redis://{config.redis.host}:{config.redis.port}",
            config.cache.cache_ttl
        )
        self.auth_manager = AuthenticationManager(
            config.security.jwt_secret,
            config.security.jwt_algorithm
        )
        self.logger = create_production_logger("enhanced_api")
        self.monitor = create_production_monitor()
        
        # Operation storage
        self.operations = {}
        self.operation_queue = asyncio.Queue(maxsize=config.operation_queue_size)
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_metrics()
    
    def _setup_middleware(self):
        """Setup middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted hosts
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.app.add_middleware(SlowAPIMiddleware)
        
        # Custom middleware for metrics and logging
        self.app.middleware("http")(self._metrics_middleware)
        self.app.middleware("http")(self._logging_middleware)
    
    def _setup_routes(self):
        """Setup API routes."""
        # Authentication routes
        self.app.post("/auth/login", response_model=TokenResponse)(self.login)
        self.app.post("/auth/refresh", response_model=TokenResponse)(self.refresh_token)
        self.app.post("/auth/logout")(self.logout)
        
        # Core API routes
        self.app.get("/", response_model=Dict[str, str])(self.root)
        self.app.get("/health", response_model=SystemStatus)(self.health_check)
        self.app.get("/metrics")(self.get_metrics)
        self.app.get("/status", response_model=SystemStatus)(self.get_status)
        
        # Optimization routes
        self.app.post("/optimize", response_model=Dict[str, str])(self.optimize_models)
        self.app.get("/operations/{operation_id}", response_model=OptimizationProgress)(self.get_operation_status)
        self.app.get("/operations", response_model=List[OptimizationProgress])(self.list_operations)
        self.app.delete("/operations/{operation_id}")(self.cancel_operation)
        self.app.get("/operations/{operation_id}/results")(self.get_operation_results)
        
        # Admin routes
        self.app.get("/admin/stats")(self.get_admin_stats)
        self.app.post("/admin/cache/clear")(self.clear_cache)
        self.app.get("/admin/health")(self.get_detailed_health)
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        @self.app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    async def _metrics_middleware(self, request: Request, call_next):
        """Metrics middleware."""
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
    
    async def _logging_middleware(self, request: Request, call_next):
        """Logging middleware."""
        start_time = time.time()
        
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Log response
        self.logger.info(
            f"Response: {response.status_code} in {duration:.3f}s",
            extra={
                "status_code": response.status_code,
                "duration": duration,
                "method": request.method,
                "path": request.url.path
            }
        )
        
        return response
    
    # Authentication endpoints
    async def login(self, credentials: UserCredentials):
        """User login."""
        # In production, validate against database
        if credentials.username == "admin" and credentials.password == "admin123":
            access_token = self.auth_manager.create_access_token(credentials.username)
            refresh_token = self.auth_manager.create_refresh_token(credentials.username)
            
            return TokenResponse(
                access_token=access_token,
                expires_in=3600,
                refresh_token=refresh_token
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    async def refresh_token(self, refresh_token: str):
        """Refresh access token."""
        payload = self.auth_manager.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        username = payload.get("sub")
        access_token = self.auth_manager.create_access_token(username)
        
        return TokenResponse(
            access_token=access_token,
            expires_in=3600
        )
    
    async def logout(self, token: str = Depends(HTTPBearer())):
        """User logout."""
        # In production, add token to blacklist
        return {"message": "Logged out successfully"}
    
    # Core endpoints
    async def root(self):
        """Root endpoint."""
        return {
            "message": "Enhanced Bulk Optimization API",
            "version": "2.0.0",
            "status": "running",
            "features": [
                "Advanced Authentication",
                "Rate Limiting",
                "Caching",
                "Real-time Monitoring",
                "Background Processing"
            ]
        }
    
    async def health_check(self):
        """Health check endpoint."""
        health_status = self.monitor.get_health_status()
        metrics_summary = self.monitor.get_metrics_summary()
        
        return SystemStatus(
            status=health_status['overall_status'],
            version="2.0.0",
            uptime=time.time() - getattr(self, 'start_time', time.time()),
            timestamp=datetime.now(timezone.utc),
            health_checks=health_status,
            performance_metrics=metrics_summary,
            active_operations=len([op for op in self.operations.values() if op['status'] == 'running']),
            queue_size=self.operation_queue.qsize()
        )
    
    async def get_metrics(self):
        """Get Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    async def get_status(self):
        """Get system status."""
        return await self.health_check()
    
    # Optimization endpoints
    @limiter.limit("10/minute")
    async def optimize_models(self, request: Request, optimization_request: EnhancedOptimizationRequest):
        """Start bulk optimization."""
        operation_id = str(uuid.uuid4())
        
        # Store operation
        self.operations[operation_id] = {
            "id": operation_id,
            "status": "pending",
            "progress": 0.0,
            "current_stage": "initializing",
            "models_processed": 0,
            "total_models": len(optimization_request.models),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "request": optimization_request.dict(),
            "results": None,
            "error": None
        }
        
        # Add to queue
        await self.operation_queue.put(operation_id)
        
        # Start background processing
        asyncio.create_task(self._process_optimization(operation_id))
        
        return {
            "operation_id": operation_id,
            "status": "pending",
            "message": "Optimization started",
            "estimated_time": self._estimate_optimization_time(optimization_request)
        }
    
    async def get_operation_status(self, operation_id: str):
        """Get operation status."""
        if operation_id not in self.operations:
            raise HTTPException(status_code=404, detail="Operation not found")
        
        operation = self.operations[operation_id]
        
        return OptimizationProgress(
            operation_id=operation_id,
            status=operation['status'],
            progress=operation['progress'],
            current_stage=operation['current_stage'],
            models_processed=operation['models_processed'],
            total_models=operation['total_models'],
            estimated_completion=operation.get('estimated_completion'),
            performance_metrics=operation.get('performance_metrics')
        )
    
    async def list_operations(self, status: Optional[str] = None, limit: int = 100):
        """List operations."""
        operations = list(self.operations.values())
        
        if status:
            operations = [op for op in operations if op['status'] == status]
        
        # Sort by creation time
        operations.sort(key=lambda x: x['created_at'], reverse=True)
        
        return operations[:limit]
    
    async def cancel_operation(self, operation_id: str):
        """Cancel operation."""
        if operation_id not in self.operations:
            raise HTTPException(status_code=404, detail="Operation not found")
        
        operation = self.operations[operation_id]
        if operation['status'] in ['completed', 'failed', 'cancelled']:
            raise HTTPException(status_code=400, detail="Operation cannot be cancelled")
        
        operation['status'] = 'cancelled'
        operation['updated_at'] = datetime.now(timezone.utc)
        
        return {"message": "Operation cancelled"}
    
    async def get_operation_results(self, operation_id: str):
        """Get operation results."""
        if operation_id not in self.operations:
            raise HTTPException(status_code=404, detail="Operation not found")
        
        operation = self.operations[operation_id]
        if operation['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Operation not completed")
        
        return operation['results']
    
    # Admin endpoints
    async def get_admin_stats(self):
        """Get admin statistics."""
        return {
            "total_operations": len(self.operations),
            "active_operations": len([op for op in self.operations.values() if op['status'] == 'running']),
            "completed_operations": len([op for op in self.operations.values() if op['status'] == 'completed']),
            "failed_operations": len([op for op in self.operations.values() if op['status'] == 'failed']),
            "queue_size": self.operation_queue.qsize(),
            "cache_stats": await self._get_cache_stats(),
            "system_metrics": self.monitor.get_metrics_summary()
        }
    
    async def clear_cache(self):
        """Clear cache."""
        if self.cache_manager.redis_pool:
            await self.cache_manager.clear_pattern("*")
            return {"message": "Cache cleared"}
        else:
            return {"message": "Cache not available"}
    
    async def get_detailed_health(self):
        """Get detailed health information."""
        return {
            "health": self.monitor.get_health_status(),
            "metrics": self.monitor.get_metrics_summary(),
            "alerts": self.monitor.get_alerts_summary(),
            "operations": {
                "total": len(self.operations),
                "by_status": {
                    status: len([op for op in self.operations.values() if op['status'] == status])
                    for status in ['pending', 'running', 'completed', 'failed', 'cancelled']
                }
            }
        }
    
    # Background processing
    async def _process_optimization(self, operation_id: str):
        """Process optimization in background."""
        try:
            operation = self.operations[operation_id]
            operation['status'] = 'running'
            operation['current_stage'] = 'optimizing'
            
            # Simulate optimization process
            total_models = operation['total_models']
            for i in range(total_models):
                await asyncio.sleep(1)  # Simulate processing time
                
                operation['models_processed'] = i + 1
                operation['progress'] = (i + 1) / total_models * 100
                operation['updated_at'] = datetime.now(timezone.utc)
            
            # Mark as completed
            operation['status'] = 'completed'
            operation['current_stage'] = 'completed'
            operation['progress'] = 100.0
            operation['results'] = {
                "models_optimized": total_models,
                "optimization_time": total_models,
                "success_rate": 1.0
            }
            operation['updated_at'] = datetime.now(timezone.utc)
            
            OPTIMIZATION_OPERATIONS.labels(status='completed').inc()
            
        except Exception as e:
            operation = self.operations[operation_id]
            operation['status'] = 'failed'
            operation['error'] = str(e)
            operation['updated_at'] = datetime.now(timezone.utc)
            
            OPTIMIZATION_OPERATIONS.labels(status='failed').inc()
            
            self.logger.error(f"Optimization failed for {operation_id}: {e}")
    
    def _estimate_optimization_time(self, request: EnhancedOptimizationRequest) -> float:
        """Estimate optimization time."""
        base_time = 10.0
        per_model_time = 5.0
        return base_time + (len(request.models) * per_model_time)
    
    async def _get_cache_stats(self):
        """Get cache statistics."""
        if not self.cache_manager.redis_pool:
            return {"status": "not_available"}
        
        try:
            info = await self.cache_manager.redis_pool.info()
            return {
                "status": "available",
                "memory_used": info.get('used_memory_human'),
                "connected_clients": info.get('connected_clients'),
                "total_commands_processed": info.get('total_commands_processed')
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

def create_enhanced_app(config: EnhancedProductionConfig) -> FastAPI:
    """Create enhanced FastAPI application."""
    api = EnhancedProductionAPI(config)
    return api.app

def run_enhanced_server(config: EnhancedProductionConfig):
    """Run enhanced production server."""
    app = create_enhanced_app(config)
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.value.lower(),
        access_log=True
    )

if __name__ == "__main__":
    # Example usage
    from enhanced_production_config import create_enhanced_production_config
    
    config_manager = create_enhanced_production_config()
    config = config_manager.get_config()
    
    print("🚀 Starting Enhanced Production API")
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"Workers: {config.workers}")
    print(f"Features: Authentication, Rate Limiting, Caching, Monitoring")
    
    run_enhanced_server(config)

