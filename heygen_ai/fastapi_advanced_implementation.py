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

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
import uuid
from fastapi import (
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from pydantic.types import UUID4
import jwt
from passlib.context import CryptContext
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, update, delete, func
from error_logging_implementation import BaseAppException, ErrorLogger, ErrorHandler
from custom_error_factories_implementation import ValidationErrorFactory
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Advanced FastAPI Implementation
==============================

This module demonstrates:
- Advanced FastAPI features and patterns
- Middleware and dependency injection
- Authentication and authorization
- Database integration with SQLAlchemy
- Background tasks and WebSockets
- API documentation and testing
- Error handling and validation
- Performance optimization
"""


    FastAPI, APIRouter, HTTPException, status, Request, Response, Depends,
    BackgroundTasks, WebSocket, WebSocketDisconnect, Query, Path as PathParam,
    Header, Cookie, Form, File, UploadFile, Body
)


# Import from existing implementations


# ============================================================================
# Configuration and Setup
# ============================================================================

class Settings:
    """Application settings"""
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "HeyGen AI API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Advanced FastAPI implementation for HeyGen AI features"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./heygen_ai.db"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "uploads"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/fastapi.log"


settings = Settings()

# ============================================================================
# Database Models
# ============================================================================

class Base(DeclarativeBase):
    """Base class for database models"""
    pass


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id: Mapped[UUID4] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    username: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column()
    full_name: Mapped[Optional[str]] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    is_superuser: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)


class Project(Base):
    """Project model"""
    __tablename__ = "projects"
    
    id: Mapped[UUID4] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column()
    description: Mapped[Optional[str]] = mapped_column()
    user_id: Mapped[UUID4] = mapped_column(sa.ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(default="active")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)


class MLModel(Base):
    """ML Model model"""
    __tablename__ = "ml_models"
    
    id: Mapped[UUID4] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column()
    model_type: Mapped[str] = mapped_column()
    version: Mapped[str] = mapped_column()
    file_path: Mapped[str] = mapped_column()
    accuracy: Mapped[Optional[float]] = mapped_column()
    user_id: Mapped[UUID4] = mapped_column(sa.ForeignKey("users.id"))
    project_id: Mapped[Optional[UUID4]] = mapped_column(sa.ForeignKey("projects.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# Pydantic Models
# ============================================================================

class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8, description="Password")
    
    @validator('password')
    def validate_password(cls, v: str) -> str:
        """Validate password strength"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[EmailStr] = Field(None, description="User email address")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    is_active: Optional[bool] = Field(None, description="User active status")


class UserInDB(UserBase):
    """User in database model"""
    id: UUID4
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserResponse(UserInDB):
    """User response model"""
    pass


class Token(BaseModel):
    """Token model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""
    user_id: Optional[UUID4] = None


class ProjectBase(BaseModel):
    """Base project model"""
    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(None, max_length=500, description="Project description")


class ProjectCreate(ProjectBase):
    """Project creation model"""
    pass


class ProjectUpdate(BaseModel):
    """Project update model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(None, max_length=500, description="Project description")
    status: Optional[str] = Field(None, description="Project status")


class ProjectInDB(ProjectBase):
    """Project in database model"""
    id: UUID4
    user_id: UUID4
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ProjectResponse(ProjectInDB):
    """Project response model"""
    pass


class MLModelBase(BaseModel):
    """Base ML model model"""
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy")


class MLModelCreate(MLModelBase):
    """ML model creation model"""
    project_id: Optional[UUID4] = Field(None, description="Associated project ID")


class MLModelUpdate(BaseModel):
    """ML model update model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Model name")
    model_type: Optional[str] = Field(None, description="Model type")
    version: Optional[str] = Field(None, description="Model version")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy")


class MLModelInDB(MLModelBase):
    """ML model in database model"""
    id: UUID4
    file_path: str
    user_id: UUID4
    project_id: Optional[UUID4]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class MLModelResponse(MLModelInDB):
    """ML model response model"""
    pass


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")


class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    uptime: float


# ============================================================================
# Database Setup
# ============================================================================

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ============================================================================
# Security and Authentication
# ============================================================================

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handling
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id: UUID4 = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    """Get current superuser"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# ============================================================================
# Middleware and Dependencies
# ============================================================================

class RequestMiddleware:
    """Custom request middleware"""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            # Add request ID
            request_id = str(uuid.uuid4())
            scope["request_id"] = request_id
            
            # Add start time
            scope["start_time"] = time.time()
            
            # Log request
            logging.info(f"Request started: {request_id} - {scope['method']} {scope['path']}")
        
        await self.app(scope, receive, send)


class ResponseMiddleware:
    """Custom response middleware"""
    
    def __init__(self, app: FastAPI):
        
    """__init__ function."""
self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            request_id = scope.get("request_id")
            start_time = scope.get("start_time")
            
            async def send_wrapper(message) -> Any:
                if message["type"] == "http.response.start":
                    # Add custom headers
                    headers = dict(message.get("headers", []))
                    headers[b"x-request-id"] = request_id.encode()
                    if start_time:
                        duration = time.time() - start_time
                        headers[b"x-response-time"] = f"{duration:.3f}".encode()
                    
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Rate limiting
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self) -> Any:
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [req_time for req_time in self.requests[client_id] if req_time > minute_ago]
        else:
            self.requests[client_id] = []
        
        # Check rate limit
        if len(self.requests[client_id]) >= settings.RATE_LIMIT_PER_MINUTE:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter()


async def check_rate_limit(request: Request):
    """Check rate limit for request"""
    client_id = request.client.host
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


# ============================================================================
# Error Handling
# ============================================================================

# Initialize error handling
error_logger = ErrorLogger()
error_handler = ErrorHandler(error_logger)
validation_factory = ValidationErrorFactory()


class FastAPIExceptionHandler:
    """Custom exception handler for FastAPI"""
    
    @staticmethod
    async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
        """Handle validation errors"""
        error = validation_factory.invalid_format("request", "valid format", str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_handler.handle_error(error)
        )
    
    @staticmethod
    async async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content=error_handler.handle_error(exc)
        )
    
    @staticmethod
    async def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions"""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_handler.handle_error(exc)
        )


# ============================================================================
# Background Tasks
# ============================================================================

async def process_ml_model_task(model_id: UUID4, db: AsyncSession):
    """Background task to process ML model"""
    try:
        # Simulate ML model processing
        await asyncio.sleep(5)
        
        # Update model status
        await db.execute(
            update(MLModel)
            .where(MLModel.id == model_id)
            .values(accuracy=0.95)
        )
        await db.commit()
        
        logging.info(f"ML model {model_id} processed successfully")
    except Exception as e:
        logging.error(f"Error processing ML model {model_id}: {e}")


async def send_notification_task(user_id: UUID4, message: str):
    """Background task to send notification"""
    try:
        # Simulate notification sending
        await asyncio.sleep(2)
        logging.info(f"Notification sent to user {user_id}: {message}")
    except Exception as e:
        logging.error(f"Error sending notification to user {user_id}: {e}")


# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self) -> Any:
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Connect WebSocket"""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket"""
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send personal message"""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)


manager = ConnectionManager()


# ============================================================================
# API Routes
# ============================================================================

# Create routers
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
users_router = APIRouter(prefix="/users", tags=["Users"])
projects_router = APIRouter(prefix="/projects", tags=["Projects"])
models_router = APIRouter(prefix="/models", tags=["ML Models"])
websocket_router = APIRouter(prefix="/ws", tags=["WebSocket"])


# Authentication routes
@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Register new user"""
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    result = await db.execute(select(User).where(User.username == user.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    # Send welcome notification
    background_tasks.add_task(send_notification_task, db_user.id, "Welcome to HeyGen AI!")
    
    return db_user


@auth_router.post("/login", response_model=Token)
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Login user"""
    # Get user by email
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@auth_router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token"""
    payload = verify_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user"
        )
    
    # Create new tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


# User routes
@users_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


@users_router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user"""
    update_data = user_update.dict(exclude_unset=True)
    
    # Check if email is being updated and is unique
    if "email" in update_data:
        result = await db.execute(
            select(User).where(User.email == update_data["email"], User.id != current_user.id)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being updated and is unique
    if "username" in update_data:
        result = await db.execute(
            select(User).where(User.username == update_data["username"], User.id != current_user.id)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Update user
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user


@users_router.get("/", response_model=PaginatedResponse)
async def get_users(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
):
    """Get all users (admin only)"""
    offset = (pagination.page - 1) * pagination.size
    
    # Get total count
    result = await db.execute(select(func.count(User.id)))
    total = result.scalar()
    
    # Get users
    result = await db.execute(
        select(User)
        .offset(offset)
        .limit(pagination.size)
    )
    users = result.scalars().all()
    
    pages = (total + pagination.size - 1) // pagination.size
    
    return PaginatedResponse(
        items=users,
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=pages
    )


# Project routes
@projects_router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new project"""
    db_project = Project(
        **project.dict(),
        user_id=current_user.id
    )
    
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    
    return db_project


@projects_router.get("/", response_model=PaginatedResponse)
async def get_projects(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's projects"""
    offset = (pagination.page - 1) * pagination.size
    
    # Get total count
    result = await db.execute(
        select(func.count(Project.id))
        .where(Project.user_id == current_user.id)
    )
    total = result.scalar()
    
    # Get projects
    result = await db.execute(
        select(Project)
        .where(Project.user_id == current_user.id)
        .offset(offset)
        .limit(pagination.size)
    )
    projects = result.scalars().all()
    
    pages = (total + pagination.size - 1) // pagination.size
    
    return PaginatedResponse(
        items=projects,
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=pages
    )


@projects_router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID4 = PathParam(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific project"""
    result = await db.execute(
        select(Project)
        .where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    return project


@projects_router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_update: ProjectUpdate,
    project_id: UUID4 = PathParam(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update project"""
    result = await db.execute(
        select(Project)
        .where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    update_data = project_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    await db.commit()
    await db.refresh(project)
    
    return project


@projects_router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID4 = PathParam(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete project"""
    result = await db.execute(
        select(Project)
        .where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    await db.delete(project)
    await db.commit()


# ML Model routes
@models_router.post("/", response_model=MLModelResponse, status_code=status.HTTP_201_CREATED)
async def create_ml_model(
    model: MLModelCreate,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create new ML model"""
    # Validate file
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large"
        )
    
    # Save file
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        buffer.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    # Create model record
    db_model = MLModel(
        **model.dict(),
        file_path=str(file_path),
        user_id=current_user.id
    )
    
    db.add(db_model)
    await db.commit()
    await db.refresh(db_model)
    
    # Add background task to process model
    background_tasks.add_task(process_ml_model_task, db_model.id, db)
    
    return db_model


@models_router.get("/", response_model=PaginatedResponse)
async def get_ml_models(
    pagination: PaginationParams = Depends(),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's ML models"""
    offset = (pagination.page - 1) * pagination.size
    
    # Build query
    query = select(MLModel).where(MLModel.user_id == current_user.id)
    if model_type:
        query = query.where(MLModel.model_type == model_type)
    
    # Get total count
    count_query = select(func.count(MLModel.id)).where(MLModel.user_id == current_user.id)
    if model_type:
        count_query = count_query.where(MLModel.model_type == model_type)
    
    result = await db.execute(count_query)
    total = result.scalar()
    
    # Get models
    result = await db.execute(
        query.offset(offset).limit(pagination.size)
    )
    models = result.scalars().all()
    
    pages = (total + pagination.size - 1) // pagination.size
    
    return PaginatedResponse(
        items=models,
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=pages
    )


@models_router.get("/{model_id}", response_model=MLModelResponse)
async def get_ml_model(
    model_id: UUID4 = PathParam(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific ML model"""
    result = await db.execute(
        select(MLModel)
        .where(MLModel.id == model_id, MLModel.user_id == current_user.id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    return model


@models_router.get("/{model_id}/download")
async def download_ml_model(
    model_id: UUID4 = PathParam(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Download ML model file"""
    result = await db.execute(
        select(MLModel)
        .where(MLModel.id == model_id, MLModel.user_id == current_user.id)
    )
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    file_path = Path(model.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found"
        )
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream"
    )


# WebSocket routes
@websocket_router.websocket("/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo message back
            await manager.send_personal_message(f"Message text was: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# Main Application
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application"""
    # Create app
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # Add custom middleware
    app = RequestMiddleware(app)
    app = ResponseMiddleware(app)
    
    # Include routers
    app.include_router(auth_router, prefix=settings.API_V1_STR)
    app.include_router(users_router, prefix=settings.API_V1_STR)
    app.include_router(projects_router, prefix=settings.API_V1_STR)
    app.include_router(models_router, prefix=settings.API_V1_STR)
    app.include_router(websocket_router, prefix=settings.API_V1_STR)
    
    # Add exception handlers
    app.add_exception_handler(HTTPException, FastAPIExceptionHandler.handle_http_exception)
    app.add_exception_handler(Exception, FastAPIExceptionHandler.handle_general_exception)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Custom OpenAPI schema
    def custom_openapi():
        
    """custom_openapi function."""
if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=settings.PROJECT_NAME,
            version=settings.VERSION,
            description=settings.DESCRIPTION,
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app


# Create app instance
app = create_app()


# ============================================================================
# Event Handlers
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    # Initialize database
    await init_db()
    
    # Create upload directory
    Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logging.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logging.info("Application shutting down")


# ============================================================================
# Health Check and Utility Routes
# ============================================================================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        uptime=time.time()  # Simplified uptime
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to HeyGen AI API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/api/v1/stream")
async def stream_response():
    """Stream response example"""
    async def generate():
        
    """generate function."""
for i in range(10):
            yield f"data: {json.dumps({'message': f'Stream message {i}', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    uvicorn.run(
        "fastapi_advanced_implementation:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL
    ) 