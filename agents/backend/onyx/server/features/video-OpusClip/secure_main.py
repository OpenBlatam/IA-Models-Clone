#!/usr/bin/env python3
"""
Secure Video-OpusClip Main Application
Integrates comprehensive security features with FastAPI best practices
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Import security modules
from security_implementation import (
    SecurityConfig, InputValidator, PasswordManager, DataEncryption,
    JWTManager, RateLimiter, IntrusionDetector, SecurityLogger,
    IncidentResponse, SecurityMiddleware, SecurityLevel, IncidentType,
    UserLogin, UserRegistration, SecureVideoRequest, get_current_user
)

# Import our custom modules
from fastapi_best_practices import (
    VideoResponse, BatchVideoResponse, ErrorResponse, HealthResponse
)
from async_flows import AsyncFlowManager, AsyncFlowConfig
from async_database import (
    DatabaseConfig, DatabaseType, AsyncDatabaseOperations,
    AsyncVideoDatabase, AsyncTransactionManager
)
from async_external_apis import (
    APIConfig, APIType, AsyncYouTubeAPI, AsyncOpenAIAPI, 
    AsyncStabilityAIAPI, AsyncElevenLabsAPI
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secure_video_opusclip.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SecureSettings(BaseSettings):
    """Secure application settings"""
    app_name: str = "Secure Video-OpusClip"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Security settings
    secret_key: str = "your-secret-key-change-this-in-production"
    encryption_key: str = "your-encryption-key-change-this-in-production"
    salt: str = "your-salt-change-this-in-production"
    
    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5432/video_opusclip"
    database_type: str = "postgresql"
    
    # API settings
    youtube_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    stability_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    
    # Security settings
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Async flow settings
    max_concurrent_tasks: int = 100
    max_concurrent_connections: int = 50
    timeout: float = 30.0
    retry_attempts: int = 3
    
    class Config:
        env_file = ".env"

# Global instances
settings = SecureSettings()
security_config = SecurityConfig()
password_manager = PasswordManager(security_config.salt)
data_encryption = DataEncryption(security_config.encryption_key)
jwt_manager = JWTManager(security_config.secret_key)
security_logger = SecurityLogger()
incident_response = IncidentResponse()
intrusion_detector = IntrusionDetector(
    max_failed_attempts=settings.max_login_attempts,
    lockout_duration=settings.lockout_duration_minutes * 60
)
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window
)

# Application instances
flow_manager: Optional[AsyncFlowManager] = None
db_ops: Optional[AsyncDatabaseOperations] = None
video_db: Optional[AsyncVideoDatabase] = None
tx_manager: Optional[AsyncTransactionManager] = None
youtube_api: Optional[AsyncYouTubeAPI] = None
openai_api: Optional[AsyncOpenAIAPI] = None
stability_api: Optional[AsyncStabilityAIAPI] = None
elevenlabs_api: Optional[AsyncElevenLabsAPI] = None

# Mock user database (replace with real database)
users_db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with security"""
    global flow_manager, db_ops, video_db, tx_manager
    global youtube_api, openai_api, stability_api, elevenlabs_api
    
    logger.info("Starting Secure Video-OpusClip application...")
    
    try:
        # Initialize async flow manager
        flow_config = AsyncFlowConfig(
            max_concurrent_tasks=settings.max_concurrent_tasks,
            max_concurrent_connections=settings.max_concurrent_connections,
            timeout=settings.timeout,
            retry_attempts=settings.retry_attempts,
            enable_metrics=True,
            enable_circuit_breaker=True
        )
        flow_manager = AsyncFlowManager(flow_config)
        await flow_manager.start()
        logger.info("Async flow manager started")
        
        # Initialize database connections
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="video_opusclip",
            username="postgres",
            password="password"
        )
        
        if settings.database_type == "postgresql":
            db_ops = AsyncDatabaseOperations(DatabaseType.POSTGRESQL, db_config)
            video_db = AsyncVideoDatabase(db_ops)
            tx_manager = AsyncTransactionManager(db_ops)
            logger.info("Database connections initialized")
        
        # Initialize external APIs
        if settings.youtube_api_key:
            youtube_config = APIConfig(
                base_url="https://www.googleapis.com/youtube/v3",
                api_key=settings.youtube_api_key,
                rate_limit_per_minute=100
            )
            youtube_api = AsyncYouTubeAPI(youtube_config)
            logger.info("YouTube API initialized")
        
        if settings.openai_api_key:
            openai_config = APIConfig(
                base_url="https://api.openai.com/v1",
                api_key=settings.openai_api_key,
                rate_limit_per_minute=60
            )
            openai_api = AsyncOpenAIAPI(openai_config)
            logger.info("OpenAI API initialized")
        
        if settings.stability_api_key:
            stability_config = APIConfig(
                base_url="https://api.stability.ai/v1",
                api_key=settings.stability_api_key,
                rate_limit_per_minute=30
            )
            stability_api = AsyncStabilityAIAPI(stability_config)
            logger.info("Stability AI API initialized")
        
        if settings.elevenlabs_api_key:
            elevenlabs_config = APIConfig(
                base_url="https://api.elevenlabs.io/v1",
                api_key=settings.elevenlabs_api_key,
                rate_limit_per_minute=50
            )
            elevenlabs_api = AsyncElevenLabsAPI(elevenlabs_config)
            logger.info("ElevenLabs API initialized")
        
        logger.info("Secure Video-OpusClip application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        # Log security incident
        incident = SecurityIncident(
            id=secrets.token_urlsafe(16),
            type=IncidentType.SYSTEM_COMPROMISE,
            severity=SecurityLevel.CRITICAL,
            description=f"Application startup failed: {e}",
            timestamp=datetime.utcnow(),
            source_ip="system",
            details={"error": str(e)}
        )
        incident_response.create_incident(incident)
        raise
    finally:
        logger.info("Shutting down Secure Video-OpusClip application...")
        
        # Cleanup resources
        if flow_manager:
            await flow_manager.shutdown()
        
        if db_ops:
            await db_ops.close()
        
        logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Secure AI-driven video processing system for short-form video platforms",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for all requests"""
    client_ip = request.client.host
    
    # Check if IP is blocked
    if intrusion_detector.is_ip_blocked(client_ip):
        security_logger.log_security_event("IP_BLOCKED", {"ip": client_ip})
        raise HTTPException(status_code=429, detail="IP temporarily blocked")
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        security_logger.log_security_event("RATE_LIMIT_EXCEEDED", {"ip": client_ip})
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Detect suspicious activity
    request_body = await request.body()
    if request_body:
        suspicious_patterns = intrusion_detector.detect_suspicious_activity(
            request_body.decode()
        )
        if suspicious_patterns:
            incident = SecurityIncident(
                id=secrets.token_urlsafe(16),
                type=IncidentType.SUSPICIOUS_ACTIVITY,
                severity=SecurityLevel.MEDIUM,
                description=f"Suspicious patterns detected: {suspicious_patterns}",
                timestamp=datetime.utcnow(),
                source_ip=client_ip,
                affected_resource=str(request.url),
                details={"patterns": suspicious_patterns}
            )
            incident_response.create_incident(incident)
    
    # Process request
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response

# Setup CORS with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])

# Setup error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    security_logger.log_security_event("UNHANDLED_EXCEPTION", {
        "error": str(exc),
        "ip": request.client.host,
        "url": str(request.url)
    })
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="Internal server error",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP exception: {exc}")
    security_logger.log_security_event("HTTP_EXCEPTION", {
        "status_code": exc.status_code,
        "detail": exc.detail,
        "ip": request.client.host,
        "url": str(request.url)
    })
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )

# Dependency injection
async def get_flow_manager() -> AsyncFlowManager:
    if not flow_manager:
        raise HTTPException(status_code=503, detail="Flow manager not available")
    return flow_manager

async def get_video_db() -> AsyncVideoDatabase:
    if not video_db:
        raise HTTPException(status_code=503, detail="Database not available")
    return video_db

async def get_tx_manager() -> AsyncTransactionManager:
    if not tx_manager:
        raise HTTPException(status_code=503, detail="Transaction manager not available")
    return tx_manager

async def get_youtube_api() -> Optional[AsyncYouTubeAPI]:
    return youtube_api

async def get_openai_api() -> Optional[AsyncOpenAIAPI]:
    return openai_api

async def get_stability_api() -> Optional[AsyncStabilityAIAPI]:
    return stability_api

async def get_elevenlabs_api() -> Optional[AsyncElevenLabsAPI]:
    return elevenlabs_api

# Authentication endpoints
@app.post("/auth/register", response_model=Dict[str, Any])
async def register_user(user_data: UserRegistration, request: Request):
    """Register a new user with security validation"""
    client_ip = request.client.host
    
    # Check if email already exists
    if user_data.email in users_db:
        security_logger.log_access("unknown", "/auth/register", "register", False, client_ip)
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = password_manager.hash_password(user_data.password)
    
    # Store user (in production, use database)
    users_db[user_data.email] = {
        "email": user_data.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True
    }
    
    security_logger.log_access(user_data.email, "/auth/register", "register", True, client_ip)
    
    return {
        "success": True,
        "message": "User registered successfully",
        "data": {"email": user_data.email}
    }

@app.post("/auth/login", response_model=Dict[str, Any])
async def login_user(user_data: UserLogin, request: Request):
    """Login user with security checks"""
    client_ip = request.client.host
    
    # Check if IP is blocked
    if intrusion_detector.is_ip_blocked(client_ip):
        raise HTTPException(status_code=429, detail="IP temporarily blocked")
    
    # Check if user exists
    if user_data.email not in users_db:
        intrusion_detector.check_login_attempt(client_ip, False)
        security_logger.log_access("unknown", "/auth/login", "login", False, client_ip)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = users_db[user_data.email]
    
    # Verify password
    if not password_manager.verify_password(user_data.password, user["hashed_password"]):
        intrusion_detector.check_login_attempt(client_ip, False)
        security_logger.log_access(user_data.email, "/auth/login", "login", False, client_ip)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Successful login
    intrusion_detector.check_login_attempt(client_ip, True)
    security_logger.log_access(user_data.email, "/auth/login", "login", True, client_ip)
    
    # Create tokens
    access_token = jwt_manager.create_access_token(
        data={"sub": user_data.email},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    refresh_token = jwt_manager.create_refresh_token(
        data={"sub": user_data.email}
    )
    
    return {
        "success": True,
        "message": "Login successful",
        "data": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60
        }
    }

@app.post("/auth/refresh", response_model=Dict[str, Any])
async def refresh_token(refresh_token: str, request: Request):
    """Refresh access token"""
    client_ip = request.client.host
    
    try:
        payload = jwt_manager.verify_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        email = payload.get("sub")
        if email not in users_db:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Create new access token
        new_access_token = jwt_manager.create_access_token(
            data={"sub": email},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        security_logger.log_access(email, "/auth/refresh", "refresh", True, client_ip)
        
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "data": {
                "access_token": new_access_token,
                "token_type": "bearer",
                "expires_in": settings.access_token_expire_minutes * 60
            }
        }
    except HTTPException:
        security_logger.log_access("unknown", "/auth/refresh", "refresh", False, client_ip)
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        success=True,
        message="Secure Video-OpusClip is running",
        version=settings.app_version,
        status="healthy"
    )

# Secure video processing endpoints
@app.post("/videos", response_model=VideoResponse)
async def create_video(
    video_data: SecureVideoRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    flow_mgr: AsyncFlowManager = Depends(get_flow_manager),
    video_db_instance: AsyncVideoDatabase = Depends(get_video_db),
    youtube_api_instance: Optional[AsyncYouTubeAPI] = Depends(get_youtube_api),
    openai_api_instance: Optional[AsyncOpenAIAPI] = Depends(get_openai_api)
):
    """Create a new video processing job with security validation"""
    client_ip = request.client.host
    user_email = current_user.get("sub")
    
    try:
        # Create video record
        video_record = {
            "title": video_data.title,
            "description": video_data.description,
            "url": video_data.url,
            "duration": video_data.duration,
            "resolution": video_data.resolution,
            "status": "pending",
            "priority": video_data.priority,
            "tags": video_data.tags,
            "user_id": user_email,
            "created_at": datetime.utcnow().isoformat()
        }
        
        video_id = await video_db_instance.create_video_record(video_record)
        
        # Add background task for processing
        background_tasks.add_task(
            process_video_background,
            video_id,
            video_data,
            user_email,
            flow_mgr,
            video_db_instance,
            youtube_api_instance,
            openai_api_instance
        )
        
        security_logger.log_access(user_email, "/videos", "create", True, client_ip)
        
        return VideoResponse(
            success=True,
            message="Video processing job created",
            data={
                "id": video_id,
                "title": video_data.title,
                "status": "pending",
                "priority": video_data.priority
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        security_logger.log_access(user_email, "/videos", "create", False, client_ip)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    video_db_instance: AsyncVideoDatabase = Depends(get_video_db)
):
    """Get video by ID with access control"""
    client_ip = request.client.host
    user_email = current_user.get("sub")
    
    try:
        video = await video_db_instance.get_video_by_id(video_id)
        if not video:
            security_logger.log_access(user_email, f"/videos/{video_id}", "read", False, client_ip)
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Check if user owns the video (basic access control)
        if video.get("user_id") != user_email:
            security_logger.log_access(user_email, f"/videos/{video_id}", "read", False, client_ip)
            raise HTTPException(status_code=403, detail="Access denied")
        
        security_logger.log_access(user_email, f"/videos/{video_id}", "read", True, client_ip)
        
        return VideoResponse(
            success=True,
            message="Video retrieved successfully",
            data=video
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video: {e}")
        security_logger.log_access(user_email, f"/videos/{video_id}", "read", False, client_ip)
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/videos/{video_id}", response_model=VideoResponse)
async def update_video(
    video_id: str,
    video_data: SecureVideoRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    video_db_instance: AsyncVideoDatabase = Depends(get_video_db)
):
    """Update video with access control"""
    client_ip = request.client.host
    user_email = current_user.get("sub")
    
    try:
        # Check if video exists and user owns it
        video = await video_db_instance.get_video_by_id(video_id)
        if not video:
            security_logger.log_access(user_email, f"/videos/{video_id}", "update", False, client_ip)
            raise HTTPException(status_code=404, detail="Video not found")
        
        if video.get("user_id") != user_email:
            security_logger.log_access(user_email, f"/videos/{video_id}", "update", False, client_ip)
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update video
        success = await video_db_instance.update_video_status(video_id, "updated")
        if not success:
            raise HTTPException(status_code=404, detail="Video not found")
        
        security_logger.log_access(user_email, f"/videos/{video_id}", "update", True, client_ip)
        
        return VideoResponse(
            success=True,
            message="Video updated successfully",
            data={"id": video_id, "status": "updated"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update video: {e}")
        security_logger.log_access(user_email, f"/videos/{video_id}", "update", False, client_ip)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/videos/{video_id}", response_model=VideoResponse)
async def delete_video(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    video_db_instance: AsyncVideoDatabase = Depends(get_video_db)
):
    """Delete video with access control"""
    client_ip = request.client.host
    user_email = current_user.get("sub")
    
    try:
        # Check if video exists and user owns it
        video = await video_db_instance.get_video_by_id(video_id)
        if not video:
            security_logger.log_access(user_email, f"/videos/{video_id}", "delete", False, client_ip)
            raise HTTPException(status_code=404, detail="Video not found")
        
        if video.get("user_id") != user_email:
            security_logger.log_access(user_email, f"/videos/{video_id}", "delete", False, client_ip)
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete video (implement in database)
        security_logger.log_access(user_email, f"/videos/{video_id}", "delete", True, client_ip)
        
        return VideoResponse(
            success=True,
            message="Video deleted successfully",
            data={"id": video_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete video: {e}")
        security_logger.log_access(user_email, f"/videos/{video_id}", "delete", False, client_ip)
        raise HTTPException(status_code=500, detail=str(e))

# Background processing function
async def process_video_background(
    video_id: str,
    video_data: SecureVideoRequest,
    user_email: str,
    flow_mgr: AsyncFlowManager,
    video_db_instance: AsyncVideoDatabase,
    youtube_api_instance: Optional[AsyncYouTubeAPI],
    openai_api_instance: Optional[AsyncOpenAIAPI]
):
    """Background video processing task with security logging"""
    try:
        logger.info(f"Starting background processing for video {video_id} by user {user_email}")
        
        # Update status to processing
        await video_db_instance.update_video_status(video_id, "processing")
        
        # Create processing job
        job_data = {
            "video_id": video_id,
            "type": "video_processing",
            "status": "pending",
            "priority": video_data.priority,
            "user_id": user_email,
            "parameters": {
                "title": video_data.title,
                "description": video_data.description,
                "url": video_data.url,
                "duration": video_data.duration,
                "resolution": video_data.resolution,
                "tags": video_data.tags
            }
        }
        
        job_id = await video_db_instance.create_processing_job(job_data)
        
        # Process video using async flow
        async def video_processing_workflow():
            # Step 1: Download video
            logger.info(f"Downloading video {video_id}")
            await asyncio.sleep(2)
            
            # Step 2: Extract metadata
            logger.info(f"Extracting metadata for video {video_id}")
            await asyncio.sleep(1)
            
            # Step 3: Generate captions (if OpenAI API available)
            if openai_api_instance:
                logger.info(f"Generating captions for video {video_id}")
                try:
                    caption_prompt = f"Generate engaging captions for a video titled '{video_data.title}' with description: {video_data.description}"
                    captions = await openai_api_instance.generate_captions(
                        audio_text=caption_prompt,
                        style="casual",
                        language="en"
                    )
                    logger.info(f"Generated captions for video {video_id}")
                except Exception as e:
                    logger.warning(f"Failed to generate captions for video {video_id}: {e}")
            
            # Step 4: Create clips
            logger.info(f"Creating clips for video {video_id}")
            await asyncio.sleep(3)
            
            # Step 5: Update status to completed
            await video_db_instance.update_video_status(video_id, "completed")
            await video_db_instance.update_job_status(job_id, "completed")
            
            logger.info(f"Completed processing for video {video_id}")
        
        # Execute workflow
        await flow_mgr.execute_workflow(video_processing_workflow)
        
    except Exception as e:
        logger.error(f"Background processing failed for video {video_id}: {e}")
        await video_db_instance.update_video_status(video_id, "failed")
        if 'job_id' in locals():
            await video_db_instance.update_job_status(job_id, "failed")
        
        # Log security incident
        incident = SecurityIncident(
            id=secrets.token_urlsafe(16),
            type=IncidentType.SYSTEM_COMPROMISE,
            severity=SecurityLevel.MEDIUM,
            description=f"Video processing failed for video {video_id}",
            timestamp=datetime.utcnow(),
            source_ip="system",
            user_id=user_email,
            affected_resource=f"video:{video_id}",
            details={"error": str(e)}
        )
        incident_response.create_incident(incident)

# Security monitoring endpoints
@app.get("/security/metrics")
async def get_security_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get security metrics (admin only)"""
    user_email = current_user.get("sub")
    
    # Basic admin check (implement proper role-based access control)
    if user_email != "admin@example.com":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "success": True,
        "message": "Security metrics retrieved",
        "data": {
            "blocked_ips": len(intrusion_detector.blocked_ips),
            "failed_attempts": len(intrusion_detector.failed_attempts),
            "active_incidents": len([i for i in incident_response.incidents if i.status == "open"]),
            "total_incidents": len(incident_response.incidents)
        }
    }

@app.get("/security/incidents")
async def get_security_incidents(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get security incidents (admin only)"""
    user_email = current_user.get("sub")
    
    if user_email != "admin@example.com":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "success": True,
        "message": "Security incidents retrieved",
        "data": {
            "incidents": [
                {
                    "id": incident.id,
                    "type": incident.type.value,
                    "severity": incident.severity.value,
                    "description": incident.description,
                    "timestamp": incident.timestamp.isoformat(),
                    "status": incident.status
                }
                for incident in incident_response.incidents
            ]
        }
    }

# Main function
def main():
    """Main application entry point"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    uvicorn.run(
        "secure_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 