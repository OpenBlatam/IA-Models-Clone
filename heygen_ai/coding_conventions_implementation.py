from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, EmailStr, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
    import re
    import html
    import uuid
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Coding Conventions Implementation
================================

This module demonstrates:
- Python naming conventions (PEP 8)
- FastAPI best practices and conventions
- Code organization and structure
- Documentation standards
- Type hints and annotations
- Error handling conventions
- Testing conventions
- Import organization
- Code formatting standards
"""




# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Use UPPER_CASE for constants
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30
API_VERSION = "v1"
SUPPORTED_LANGUAGES = ["en", "es", "fr"]

# Configuration classes use PascalCase
class DatabaseConfig:
    """Database configuration settings."""
    
    HOST: str = "localhost"
    PORT: int = 5432
    DATABASE: str = "myapp"
    USERNAME: str = "user"
    PASSWORD: str = "password"
    
    @classmethod
    def get_connection_string(cls) -> str:
        """Get database connection string."""
        return f"postgresql://{cls.USERNAME}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"


class APIConfig:
    """API configuration settings."""
    
    TITLE: str = "My FastAPI Application"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A well-structured FastAPI application"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]


# ============================================================================
# ENUMS
# ============================================================================

class UserStatus(str, Enum):
    """User status enumeration."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HTTPMethod(str, Enum):
    """HTTP methods enumeration."""
    
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class UserCredentials:
    """User credentials data class."""
    
    username: str
    password: str
    email: str
    
    def is_valid(self) -> bool:
        """Check if credentials are valid."""
        return bool(self.username and self.password and self.email)


@dataclass
class APIResponse:
    """Standard API response structure."""
    
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self) -> Any:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BaseAPIModel(BaseModel):
    """Base model for all API models."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"  # Reject extra fields
    )


class UserCreate(BaseAPIModel):
    """Model for creating a new user."""
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    
    @validator('username')
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower()
    
    @validator('password')
    def validate_password_strength(cls, v: str) -> str:
        """Validate password strength."""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v


class UserResponse(BaseAPIModel):
    """Model for user response data."""
    
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    status: UserStatus = Field(..., description="User status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class UserUpdate(BaseAPIModel):
    """Model for updating user data."""
    
    email: Optional[EmailStr] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    status: Optional[UserStatus] = Field(None, description="User status")


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseService(ABC):
    """Abstract base class for service layer."""
    
    def __init__(self, logger: logging.Logger):
        
    """__init__ function."""
self.logger = logger
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> Any:
        """Create a new resource."""
        pass
    
    @abstractmethod
    async def get_by_id(self, resource_id: int) -> Optional[Any]:
        """Get resource by ID."""
        pass
    
    @abstractmethod
    async def update(self, resource_id: int, data: Dict[str, Any]) -> Optional[Any]:
        """Update resource."""
        pass
    
    @abstractmethod
    async def delete(self, resource_id: int) -> bool:
        """Delete resource."""
        pass


class BaseRepository(ABC):
    """Abstract base class for data access layer."""
    
    def __init__(self, session: AsyncSession):
        
    """__init__ function."""
self.session = session
    
    @abstractmethod
    async def find_by_id(self, resource_id: int) -> Optional[Any]:
        """Find resource by ID."""
        pass
    
    @abstractmethod
    async def find_all(self, skip: int = 0, limit: int = 100) -> List[Any]:
        """Find all resources with pagination."""
        pass


# ============================================================================
# SERVICE LAYER
# ============================================================================

class UserService(BaseService):
    """User service implementation."""
    
    def __init__(self, logger: logging.Logger, repository: 'UserRepository'):
        
    """__init__ function."""
super().__init__(logger)
        self.repository = repository
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        try:
            # Validate user data
            if not user_data.is_valid():
                raise ValueError("Invalid user data")
            
            # Check if user already exists
            existing_user = await self.repository.find_by_email(user_data.email)
            if existing_user:
                raise ValueError("User with this email already exists")
            
            # Create user
            user_dict = user_data.model_dump()
            user = await self.repository.create(user_dict)
            
            self.logger.info(f"User created successfully: {user.id}")
            return UserResponse.model_validate(user)
            
        except Exception as e:
            self.logger.error(f"Failed to create user: {str(e)}")
            raise
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID."""
        try:
            user = await self.repository.find_by_id(user_id)
            if not user:
                return None
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            self.logger.error(f"Failed to get user {user_id}: {str(e)}")
            raise
    
    async def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[UserResponse]:
        """Update user data."""
        try:
            # Check if user exists
            existing_user = await self.repository.find_by_id(user_id)
            if not existing_user:
                return None
            
            # Update user
            update_data = user_data.model_dump(exclude_unset=True)
            updated_user = await self.repository.update(user_id, update_data)
            
            self.logger.info(f"User updated successfully: {user_id}")
            return UserResponse.model_validate(updated_user)
            
        except Exception as e:
            self.logger.error(f"Failed to update user {user_id}: {str(e)}")
            raise
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        try:
            success = await self.repository.delete(user_id)
            if success:
                self.logger.info(f"User deleted successfully: {user_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {str(e)}")
            raise


# ============================================================================
# REPOSITORY LAYER
# ============================================================================

class UserRepository(BaseRepository):
    """User repository implementation."""
    
    async def find_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Find user by ID."""
        try:
            # Simulate database query
            await asyncio.sleep(0.1)
            
            # Mock user data
            if user_id == 1:
                return {
                    "id": 1,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "full_name": "John Doe",
                    "status": UserStatus.ACTIVE,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            return None
            
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")
    
    async def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find user by email."""
        try:
            # Simulate database query
            await asyncio.sleep(0.1)
            
            # Mock check
            if email == "john@example.com":
                return {
                    "id": 1,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "full_name": "John Doe",
                    "status": UserStatus.ACTIVE,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            return None
            
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")
    
    async def find_all(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Find all users with pagination."""
        try:
            # Simulate database query
            await asyncio.sleep(0.1)
            
            # Mock users data
            users = []
            for i in range(skip, skip + limit):
                users.append({
                    "id": i + 1,
                    "username": f"user_{i + 1}",
                    "email": f"user_{i + 1}@example.com",
                    "full_name": f"User {i + 1}",
                    "status": UserStatus.ACTIVE,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                })
            
            return users
            
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")
    
    async def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user."""
        try:
            # Simulate database insert
            await asyncio.sleep(0.1)
            
            # Mock created user
            return {
                "id": 999,
                "username": user_data["username"],
                "email": user_data["email"],
                "full_name": user_data.get("full_name"),
                "status": UserStatus.ACTIVE,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")
    
    async def update(self, user_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user data."""
        try:
            # Simulate database update
            await asyncio.sleep(0.1)
            
            # Get existing user
            existing_user = await self.find_by_id(user_id)
            if not existing_user:
                raise ValueError("User not found")
            
            # Update fields
            existing_user.update(update_data)
            existing_user["updated_at"] = datetime.utcnow()
            
            return existing_user
            
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")
    
    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        try:
            # Simulate database delete
            await asyncio.sleep(0.1)
            
            # Check if user exists
            existing_user = await self.find_by_id(user_id)
            if not existing_user:
                return False
            
            return True
            
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")


# ============================================================================
# DEPENDENCIES
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_user_repository() -> UserRepository:
    """Get user repository instance."""
    # In a real application, this would get the database session
    # and create the repository with proper dependency injection
    return UserRepository(session=None)


def get_user_service() -> UserService:
    """Get user service instance."""
    logger = get_logger("user_service")
    repository = get_user_repository()
    return UserService(logger, repository)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class AppException(Exception):
    """Base application exception."""
    
    def __init__(self, message: str, error_code: str = None):
        
    """__init__ function."""
super().__init__(message)
        self.message = message
        self.error_code = error_code


class ValidationException(AppException):
    """Validation error exception."""
    
    def __init__(self, message: str, field: str = None):
        
    """__init__ function."""
super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class NotFoundException(AppException):
    """Resource not found exception."""
    
    def __init__(self, resource_type: str, resource_id: int):
        
    """__init__ function."""
message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(message, "NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id


class DatabaseException(AppException):
    """Database operation exception."""
    
    def __init__(self, message: str, operation: str = None):
        
    """__init__ function."""
super().__init__(message, "DATABASE_ERROR")
        self.operation = operation


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_email_format(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_input(input_string: str) -> str:
    """Sanitize user input."""
    return html.escape(input_string.strip())


def generate_unique_id() -> str:
    """Generate unique identifier."""
    return str(uuid.uuid4())


def format_datetime(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return dt.isoformat()


def parse_datetime(date_string: str) -> datetime:
    """Parse datetime from ISO string."""
    return datetime.fromisoformat(date_string)


# ============================================================================
# API ROUTERS
# ============================================================================

# Create router with proper prefix and tags
user_router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
    responses={
        404: {"description": "User not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)


@user_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    Create a new user.
    
    Args:
        user_data: User creation data
        user_service: User service dependency
        
    Returns:
        UserResponse: Created user data
        
    Raises:
        HTTPException: If user creation fails
    """
    try:
        user = await user_service.create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@user_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    Get user by ID.
    
    Args:
        user_id: User ID
        user_service: User service dependency
        
    Returns:
        UserResponse: User data
        
    Raises:
        HTTPException: If user not found
    """
    try:
        user = await user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@user_router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    Update user data.
    
    Args:
        user_id: User ID
        user_data: User update data
        user_service: User service dependency
        
    Returns:
        UserResponse: Updated user data
        
    Raises:
        HTTPException: If user not found or update fails
    """
    try:
        user = await user_service.update_user(user_id, user_data)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@user_router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
) -> None:
    """
    Delete user.
    
    Args:
        user_id: User ID
        user_service: User service dependency
        
    Raises:
        HTTPException: If user not found or deletion fails
    """
    try:
        success = await user_service.delete_user(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=APIConfig.TITLE,
        version=APIConfig.VERSION,
        description=APIConfig.DESCRIPTION,
        debug=APIConfig.DEBUG
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APIConfig.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(user_router)
    
    return app


# Create application instance
app = create_app()


# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger = get_logger("app")
    logger.info("Application starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger = get_logger("app")
    logger.info("Application shutting down...")


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dict containing health status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": format_datetime(datetime.utcnow()),
        "version": APIConfig.VERSION
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def demonstrate_coding_conventions():
    """Demonstrate coding conventions in action."""
    
    print("\n=== Coding Conventions Demonstrations ===")
    
    # 1. Constants and configuration
    print("\n1. Constants and Configuration:")
    print(f"Max retry attempts: {MAX_RETRY_ATTEMPTS}")
    print(f"Database connection: {DatabaseConfig.get_connection_string()}")
    
    # 2. Enums
    print("\n2. Enums:")
    print(f"User status: {UserStatus.ACTIVE}")
    print(f"Error severity: {ErrorSeverity.HIGH}")
    
    # 3. Data classes
    print("\n3. Data Classes:")
    credentials = UserCredentials("john_doe", "password123", "john@example.com")
    print(f"Credentials valid: {credentials.is_valid()}")
    
    # 4. Pydantic models
    print("\n4. Pydantic Models:")
    user_create = UserCreate(
        username="jane_doe",
        email="jane@example.com",
        password="Password123",
        full_name="Jane Doe"
    )
    print(f"User model: {user_create.model_dump()}")
    
    # 5. Service layer
    print("\n5. Service Layer:")
    logger = get_logger("demo")
    repository = UserRepository(session=None)
    service = UserService(logger, repository)
    
    # 6. Error handling
    print("\n6. Error Handling:")
    try:
        raise ValidationException("Invalid email format", "email")
    except ValidationException as e:
        print(f"Validation error: {e.message} (field: {e.field})")
    
    # 7. Utility functions
    print("\n7. Utility Functions:")
    print(f"Email valid: {validate_email_format('test@example.com')}")
    print(f"Sanitized input: {sanitize_input('<script>alert(\"xss\")</script>')}")
    print(f"Unique ID: {generate_unique_id()}")


if __name__ == "__main__":
    
    # Run the application
    uvicorn.run(
        "coding_conventions_implementation:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 