"""
Advanced User Service - Microservices Example
Demonstrates: Service discovery, circuit breakers, observability, serverless optimization
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

# Import our microservices framework components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.core.service_registry import ServiceRegistry, ServiceInstance, ServiceType, ServiceStatus
from shared.core.circuit_breaker import CircuitBreakerConfig, HTTPCircuitBreaker
from shared.monitoring.observability import (
    ObservabilityManager, TracingConfig, MetricsConfig, 
    LoggingConfig, HealthCheckConfig, trace_function
)
from shared.serverless.serverless_adapter import (
    ServerlessConfig, ServerlessPlatform, optimize_for_serverless
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pydantic Models
class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    full_name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)

class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    email: Optional[str] = Field(None, regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')

class UserResponse(BaseModel):
    """User response model"""
    id: str
    username: str
    email: str
    full_name: str
    age: Optional[int]
    created_at: str
    updated_at: str
    status: str = "active"

class UserListResponse(BaseModel):
    """User list response model"""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int

# Global instances
service_registry: Optional[ServiceRegistry] = None
observability_manager: Optional[ObservabilityManager] = None
circuit_breaker: Optional[HTTPCircuitBreaker] = None
users_db: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global service_registry, observability_manager, circuit_breaker
    
    logger.info("Starting User Service...")
    
    try:
        # Initialize observability
        observability_manager = ObservabilityManager(
            tracing_config=TracingConfig(
                service_name="user-service",
                service_version="1.0.0",
                enabled=True
            ),
            metrics_config=MetricsConfig(
                enabled=True,
                prometheus_port=8001
            ),
            logging_config=LoggingConfig(
                enabled=True,
                level=structlog.stdlib.INFO
            ),
            health_config=HealthCheckConfig(
                enabled=True,
                endpoint="/health"
            )
        )
        
        await observability_manager.initialize()
        observability_manager.instrument_fastapi(app)
        
        # Initialize service registry
        service_registry = ServiceRegistry("redis://localhost:6379")
        await service_registry.start()
        
        # Register this service
        service_instance = ServiceInstance(
            service_id=str(uuid.uuid4()),
            service_name="user-service",
            service_type=ServiceType.API,
            host="localhost",
            port=8001,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            health_check_url="http://localhost:8001/health",
            metadata={
                "description": "User management service",
                "version": "1.0.0",
                "environment": "development"
            },
            last_heartbeat=time.time(),
            registered_at=time.time()
        )
        
        await service_registry.register_service(service_instance)
        
        # Initialize circuit breaker for external services
        circuit_breaker = HTTPCircuitBreaker(
            "user-service",
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                timeout=30.0
            )
        )
        
        logger.info("User Service started successfully")
        yield
        
    except Exception as e:
        logger.error("Failed to start User Service", error=str(e))
        raise
    finally:
        logger.info("Shutting down User Service...")
        
        if service_registry:
            await service_registry.stop()

# Create FastAPI app
app = FastAPI(
    title="User Service",
    description="Advanced microservice for user management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_observability() -> ObservabilityManager:
    if not observability_manager:
        raise HTTPException(status_code=503, detail="Observability not available")
    return observability_manager

async def get_circuit_breaker() -> HTTPCircuitBreaker:
    if not circuit_breaker:
        raise HTTPException(status_code=503, detail="Circuit breaker not available")
    return circuit_breaker

# Health check endpoints
@app.get("/health")
async def health_check(obs: ObservabilityManager = Depends(get_observability)):
    """Health check endpoint"""
    return await obs.get_health_status()

@app.get("/health/ready")
async def readiness_check(obs: ObservabilityManager = Depends(get_observability)):
    """Readiness check endpoint"""
    return await obs.get_readiness()

@app.get("/health/live")
async def liveness_check(obs: ObservabilityManager = Depends(get_observability)):
    """Liveness check endpoint"""
    return await obs.get_liveness()

@app.get("/metrics")
async def metrics_endpoint(obs: ObservabilityManager = Depends(get_observability)):
    """Prometheus metrics endpoint"""
    return Response(
        content=obs.get_metrics(),
        media_type="text/plain"
    )

# User endpoints
@app.post("/users", response_model=UserResponse)
@trace_function("create_user")
async def create_user(
    user_data: UserCreate,
    request: Request,
    obs: ObservabilityManager = Depends(get_observability)
):
    """Create a new user"""
    try:
        # Check if username already exists
        for user in users_db.values():
            if user["username"] == user_data.username:
                raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        for user in users_db.values():
            if user["email"] == user_data.email:
                raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user
        user_id = str(uuid.uuid4())
        now = time.time()
        
        user = {
            "id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "age": user_data.age,
            "created_at": now,
            "updated_at": now,
            "status": "active"
        }
        
        users_db[user_id] = user
        
        # Log user creation
        obs.structured_logger.info(
            "User created successfully",
            user_id=user_id,
            username=user_data.username,
            email=user_data.email
        )
        
        return UserResponse(**user)
        
    except HTTPException:
        raise
    except Exception as e:
        obs.structured_logger.error("Failed to create user", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users/{user_id}", response_model=UserResponse)
@trace_function("get_user")
async def get_user(
    user_id: str,
    obs: ObservabilityManager = Depends(get_observability)
):
    """Get user by ID"""
    try:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = users_db[user_id]
        return UserResponse(**user)
        
    except HTTPException:
        raise
    except Exception as e:
        obs.structured_logger.error("Failed to get user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users", response_model=UserListResponse)
@trace_function("list_users")
async def list_users(
    page: int = 1,
    page_size: int = 10,
    obs: ObservabilityManager = Depends(get_observability)
):
    """List users with pagination"""
    try:
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 10
        
        # Get users with pagination
        all_users = list(users_db.values())
        total = len(all_users)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        users_page = all_users[start_idx:end_idx]
        
        return UserListResponse(
            users=[UserResponse(**user) for user in users_page],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        obs.structured_logger.error("Failed to list users", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/users/{user_id}", response_model=UserResponse)
@trace_function("update_user")
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    obs: ObservabilityManager = Depends(get_observability)
):
    """Update user"""
    try:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = users_db[user_id]
        
        # Update fields
        if user_data.full_name is not None:
            user["full_name"] = user_data.full_name
        if user_data.age is not None:
            user["age"] = user_data.age
        if user_data.email is not None:
            # Check if email already exists
            for uid, u in users_db.items():
                if uid != user_id and u["email"] == user_data.email:
                    raise HTTPException(status_code=400, detail="Email already exists")
            user["email"] = user_data.email
        
        user["updated_at"] = time.time()
        
        obs.structured_logger.info("User updated successfully", user_id=user_id)
        
        return UserResponse(**user)
        
    except HTTPException:
        raise
    except Exception as e:
        obs.structured_logger.error("Failed to update user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/users/{user_id}")
@trace_function("delete_user")
async def delete_user(
    user_id: str,
    obs: ObservabilityManager = Depends(get_observability)
):
    """Delete user"""
    try:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        del users_db[user_id]
        
        obs.structured_logger.info("User deleted successfully", user_id=user_id)
        
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        obs.structured_logger.error("Failed to delete user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# External service integration example
@app.get("/users/{user_id}/profile")
@trace_function("get_user_profile")
async def get_user_profile(
    user_id: str,
    cb: HTTPCircuitBreaker = Depends(get_circuit_breaker),
    obs: ObservabilityManager = Depends(get_observability)
):
    """Get user profile with external service integration"""
    try:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = users_db[user_id]
        
        # Simulate external service call with circuit breaker
        async def call_profile_service():
            # This would be a real external service call
            await asyncio.sleep(0.1)  # Simulate network delay
            return {
                "user_id": user_id,
                "profile_data": {
                    "bio": f"Bio for {user['full_name']}",
                    "avatar_url": f"https://example.com/avatars/{user_id}.jpg",
                    "social_links": []
                }
            }
        
        try:
            profile_data = await cb.call(call_profile_service)
            
            return {
                "user": UserResponse(**user),
                "profile": profile_data["profile_data"]
            }
            
        except Exception as e:
            obs.structured_logger.warning(
                "External service call failed, returning basic profile",
                user_id=user_id,
                error=str(e)
            )
            
            # Fallback response
            return {
                "user": UserResponse(**user),
                "profile": {
                    "bio": "Profile service unavailable",
                    "avatar_url": None,
                    "social_links": []
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        obs.structured_logger.error("Failed to get user profile", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Service discovery endpoint
@app.get("/service/info")
async def service_info():
    """Get service information"""
    return {
        "service_name": "user-service",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": [
            "/users",
            "/users/{user_id}",
            "/health",
            "/metrics"
        ],
        "dependencies": [
            "redis",
            "external-profile-service"
        ]
    }

# Background task example
@app.post("/users/{user_id}/send-welcome-email")
async def send_welcome_email(
    user_id: str,
    obs: ObservabilityManager = Depends(get_observability)
):
    """Send welcome email (background task)"""
    try:
        if user_id not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = users_db[user_id]
        
        # Simulate email sending
        await asyncio.sleep(1)
        
        obs.structured_logger.info(
            "Welcome email sent",
            user_id=user_id,
            email=user["email"]
        )
        
        return {"message": "Welcome email sent successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        obs.structured_logger.error("Failed to send welcome email", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    
    # Create serverless adapter for deployment flexibility
    serverless_config = ServerlessConfig(
        platform=ServerlessPlatform.AWS_LAMBDA,
        cold_start_timeout=10.0,
        enable_compression=True,
        enable_caching=True
    )
    
    # Optimize for serverless if needed
    # serverless_adapter = optimize_for_serverless(app, ServerlessPlatform.AWS_LAMBDA)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )






























