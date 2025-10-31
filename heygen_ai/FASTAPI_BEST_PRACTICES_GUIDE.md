# FastAPI Best Practices Guide

A comprehensive guide to implementing FastAPI best practices for Data Models, Path Operations, and Middleware based on official FastAPI documentation.

## 🎯 Overview

This guide covers FastAPI best practices for:
- **Data Models**: Pydantic models with validation, serialization, and documentation
- **Path Operations**: RESTful API design with proper HTTP methods and status codes
- **Middleware**: Request/response processing, authentication, and error handling
- **Validation**: Input validation, error handling, and response formatting
- **Documentation**: Automatic API documentation generation
- **Performance**: Optimization techniques and best practices

## 📋 Table of Contents

1. [Data Models Best Practices](#data-models-best-practices)
2. [Path Operations Best Practices](#path-operations-best-practices)
3. [Middleware Best Practices](#middleware-best-practices)
4. [Validation Best Practices](#validation-best-practices)
5. [Error Handling Best Practices](#error-handling-best-practices)
6. [Documentation Best Practices](#documentation-best-practices)
7. [Performance Best Practices](#performance-best-practices)
8. [Security Best Practices](#security-best-practices)
9. [Testing Best Practices](#testing-best-practices)
10. [Deployment Best Practices](#deployment-best-practices)

## 📊 Data Models Best Practices

### **1. Pydantic Model Structure**

```python
from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum

# ✅ Good: Proper model structure following FastAPI best practices
class UserBase(BaseModel):
    """Base user model following FastAPI best practices."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's full name",
        example="John Doe"
    )
    
    email: EmailStr = Field(
        ...,
        description="User's email address",
        example="john.doe@example.com"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com"
            }
        }

# ❌ Bad: Missing validation and documentation
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### **2. Request/Response Models**

```python
# ✅ Good: Separate request and response models
class UserCreate(UserBase):
    """User creation model following FastAPI best practices."""
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User's password (min 8 chars)",
        example="SecurePass123!"
    )
    
    confirm_password: str = Field(
        ...,
        description="Password confirmation",
        example="SecurePass123!"
    )
    
    @validator('password')
    def validate_password(cls, v):
        """Password validation following FastAPI best practices."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        """Password confirmation validation."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserResponse(UserBase):
    """User response model following FastAPI best practices."""
    
    id: int = Field(
        ...,
        description="User's unique identifier",
        example=1
    )
    
    created_at: datetime = Field(
        ...,
        description="User account creation timestamp",
        example="2024-01-01T00:00:00Z"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
```

### **3. Enum Usage**

```python
# ✅ Good: Use enums for constrained values
class UserStatusEnum(str, Enum):
    """User status enumeration following FastAPI best practices."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

class UserRoleEnum(str, Enum):
    """User role enumeration following FastAPI best practices."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

class UserResponse(UserBase):
    status: UserStatusEnum = Field(
        ...,
        description="User's account status",
        example=UserStatusEnum.ACTIVE
    )
    
    role: UserRoleEnum = Field(
        ...,
        description="User's role in the system",
        example=UserRoleEnum.USER
    )
```

### **4. Nested Models**

```python
# ✅ Good: Nested models for complex data
class Address(BaseModel):
    """Address model following FastAPI best practices."""
    
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")
    zip_code: str = Field(..., description="ZIP code")
    country: str = Field(..., description="Country")

class UserProfile(BaseModel):
    """User profile model following FastAPI best practices."""
    
    user: UserResponse = Field(..., description="User information")
    address: Optional[Address] = Field(None, description="User address")
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences"
    )
```

### **5. Model Inheritance**

```python
# ✅ Good: Model inheritance for code reuse
class TimestampedModel(BaseModel):
    """Base model with timestamp fields following FastAPI best practices."""
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

class IdentifiedModel(TimestampedModel):
    """Base model with ID field following FastAPI best practices."""
    
    id: Optional[int] = None

class VideoResponse(IdentifiedModel):
    """Video response model following FastAPI best practices."""
    
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
```

## 🔗 Path Operations Best Practices

### **1. HTTP Methods Usage**

```python
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body

# ✅ Good: Proper HTTP method usage
@app.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of users to return")
):
    """Get list of users following FastAPI best practices."""
    pass

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    """Create a new user following FastAPI best practices."""
    pass

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int = Path(..., gt=0, description="User ID")):
    """Get user by ID following FastAPI best practices."""
    pass

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int = Path(..., gt=0),
    user: UserUpdate = Body(...)
):
    """Update user following FastAPI best practices."""
    pass

@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int = Path(..., gt=0)):
    """Delete user following FastAPI best practices."""
    pass

# ❌ Bad: Wrong HTTP method usage
@app.get("/users/create")  # Should use POST
@app.post("/users/{user_id}")  # Should use PUT for updates
```

### **2. Path Parameters**

```python
# ✅ Good: Path parameters with validation
@app.get("/users/{user_id}/videos/{video_id}")
async def get_user_video(
    user_id: int = Path(..., gt=0, description="User ID"),
    video_id: int = Path(..., gt=0, description="Video ID")
):
    """Get specific video for a user following FastAPI best practices."""
    pass

# ✅ Good: Path parameters with regex validation
@app.get("/users/{username}")
async def get_user_by_username(
    username: str = Path(..., regex=r'^[a-zA-Z0-9_-]+$', description="Username")
):
    """Get user by username following FastAPI best practices."""
    pass
```

### **3. Query Parameters**

```python
# ✅ Good: Query parameters with validation and documentation
@app.get("/users")
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of users to return"),
    search: Optional[str] = Query(None, min_length=1, description="Search term"),
    status: Optional[UserStatusEnum] = Query(None, description="Filter by status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    """Get filtered and paginated users following FastAPI best practices."""
    pass
```

### **4. Request Body**

```python
# ✅ Good: Request body with proper models
@app.post("/users")
async def create_user(
    user: UserCreate = Body(..., description="User data to create")
):
    """Create user following FastAPI best practices."""
    pass

# ✅ Good: Multiple request bodies
@app.post("/users")
async def create_user(
    user: UserCreate = Body(..., description="User data"),
    settings: UserSettings = Body(..., description="User settings")
):
    """Create user with settings following FastAPI best practices."""
    pass

# ✅ Good: Form data
@app.post("/users/upload")
async def upload_user_avatar(
    file: UploadFile = File(..., description="Avatar image"),
    user_id: int = Form(..., description="User ID")
):
    """Upload user avatar following FastAPI best practices."""
    pass
```

### **5. Response Models**

```python
# ✅ Good: Proper response models
@app.get("/users", response_model=List[UserResponse])
async def get_users():
    """Get users with proper response model."""
    pass

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get user with proper response model."""
    pass

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    """Create user with proper response model and status code."""
    pass

# ✅ Good: Different response models for different status codes
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    response_model=UserResponse,
    responses={
        200: {"description": "User found", "model": UserResponse},
        404: {"description": "User not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
):
    """Get user with multiple response models."""
    pass
```

### **6. Dependencies**

```python
# ✅ Good: Dependency injection
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token following FastAPI best practices."""
    pass

async def get_db():
    """Get database session following FastAPI best practices."""
    pass

@app.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user info following FastAPI best practices."""
    pass
```

## 🔄 Middleware Best Practices

### **1. Middleware Structure**

```python
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import json

# ✅ Good: Custom middleware following FastAPI best practices
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware following FastAPI best practices."""
    
    def __init__(self, app, include_body: bool = False):
        super().__init__(app)
        self.include_body = include_body
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request following FastAPI best practices."""
        start_time = time.time()
        
        # Pre-processing
        await self.pre_process(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Post-processing
            await self.post_process(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Error handling
            return await self.handle_error(request, e, start_time)
    
    async def pre_process(self, request: Request):
        """Pre-processing hook following FastAPI best practices."""
        # Log incoming request
        print(f"Request: {request.method} {request.url}")
    
    async def post_process(self, request: Request, response: Response, start_time: float):
        """Post-processing hook following FastAPI best practices."""
        duration = time.time() - start_time
        print(f"Response: {response.status_code} in {duration:.2f}s")
    
    async def handle_error(self, request: Request, error: Exception, start_time: float) -> Response:
        """Error handling hook following FastAPI best practices."""
        duration = time.time() - start_time
        print(f"Error: {error} in {duration:.2f}s")
        
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json"
        )
```

### **2. Built-in Middleware**

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# ✅ Good: CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com", "https://api.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ✅ Good: Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ✅ Good: Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)
```

### **3. Authentication Middleware**

```python
# ✅ Good: Authentication middleware following FastAPI best practices
class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware following FastAPI best practices."""
    
    def __init__(self, app, exclude_paths: List[str] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/v1/users/login"
        ]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Authenticate request following FastAPI best practices."""
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Extract and validate token
        token = self._extract_token(request)
        if not token:
            return Response(
                content=json.dumps({"error": "Missing authentication token"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Validate token
        user = await self._validate_token(token)
        if not user:
            return Response(
                content=json.dumps({"error": "Invalid authentication token"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Store user in request state
        request.state.user = user
        
        return await call_next(request)
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication."""
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract token from request headers."""
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        return None
    
    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token following FastAPI best practices."""
        # Implement your token validation logic here
        try:
            # Decode and validate JWT token
            # user = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            # return user
            return {"user_id": "1", "username": "test_user"}
        except Exception:
            return None
```

### **4. Rate Limiting Middleware**

```python
# ✅ Good: Rate limiting middleware following FastAPI best practices
class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware following FastAPI best practices."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.rate_limits: Dict[str, List[float]] = {}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting following FastAPI best practices."""
        client_id = self._get_client_id(request)
        
        if not self._check_rate_limit(client_id):
            return Response(
                content=json.dumps({"error": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        self._add_request(client_id)
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        if hasattr(request.state, "user"):
            return f"user_{request.state.user.get('user_id', 'unknown')}"
        
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip_{forwarded_for.split(',')[0].strip()}"
        
        if request.client:
            return f"ip_{request.client.host}"
        
        return "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        # Remove old requests outside the window
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if req_time > window_start
        ]
        
        return len(self.rate_limits[client_id]) < self.requests_per_minute
    
    def _add_request(self, client_id: str):
        """Add current request to rate limit tracking."""
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        self.rate_limits[client_id].append(time.time())
```

## ✅ Validation Best Practices

### **1. Field Validation**

```python
# ✅ Good: Field validation following FastAPI best practices
class UserCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's full name"
    )
    
    email: EmailStr = Field(
        ...,
        description="User's email address"
    )
    
    age: int = Field(
        ...,
        ge=0,
        le=150,
        description="User's age"
    )
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        regex=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$",
        description="Password with complexity requirements"
    )
```

### **2. Custom Validators**

```python
# ✅ Good: Custom validators following FastAPI best practices
from pydantic import validator

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str
    
    @validator('email')
    def validate_email_domain(cls, v):
        """Validate email domain following FastAPI best practices."""
        allowed_domains = ['example.com', 'gmail.com', 'yahoo.com']
        domain = v.split('@')[1]
        if domain not in allowed_domains:
            raise ValueError(f'Email domain {domain} not allowed')
        return v
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength following FastAPI best practices."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        """Validate password confirmation following FastAPI best practices."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
```

### **3. Root Validators**

```python
# ✅ Good: Root validators following FastAPI best practices
from pydantic import root_validator

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str
    age: int
    birth_date: Optional[date] = None
    
    @root_validator
    def validate_user_data(cls, values):
        """Validate user data following FastAPI best practices."""
        # Check if age matches birth date
        if 'age' in values and 'birth_date' in values and values['birth_date']:
            calculated_age = (date.today() - values['birth_date']).days // 365
            if abs(calculated_age - values['age']) > 1:
                raise ValueError('Age does not match birth date')
        
        # Check if email is not too old
        if 'email' in values:
            email_domain = values['email'].split('@')[1]
            if email_domain in ['aol.com', 'hotmail.com']:
                raise ValueError('Please use a modern email provider')
        
        return values
```

## ⚠️ Error Handling Best Practices

### **1. HTTP Exception Handling**

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# ✅ Good: HTTP exception handling following FastAPI best practices
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions following FastAPI best practices."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors following FastAPI best practices."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "error_details": exc.errors(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
```

### **2. Custom Error Responses**

```python
# ✅ Good: Custom error response models following FastAPI best practices
class ErrorResponse(BaseModel):
    """Error response model following FastAPI best practices."""
    
    success: bool = False
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

# ✅ Good: Route-level error handling
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    responses={
        200: {"description": "User found", "model": UserResponse},
        404: {"description": "User not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
):
    """Get user with proper error handling following FastAPI best practices."""
    try:
        user = await get_user_from_db(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

## 📚 Documentation Best Practices

### **1. Model Documentation**

```python
# ✅ Good: Model documentation following FastAPI best practices
class UserResponse(BaseModel):
    """User response model following FastAPI best practices.
    
    This model represents a user in the system with all necessary information
    for API responses. It includes user identification, profile information,
    and metadata.
    """
    
    id: int = Field(
        ...,
        description="User's unique identifier in the system",
        example=1
    )
    
    name: str = Field(
        ...,
        description="User's full name as displayed in the system",
        example="John Doe"
    )
    
    email: EmailStr = Field(
        ...,
        description="User's email address for communication",
        example="john.doe@example.com"
    )
    
    class Config:
        """Pydantic configuration following FastAPI best practices."""
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john.doe@example.com",
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
```

### **2. Route Documentation**

```python
# ✅ Good: Route documentation following FastAPI best practices
@app.get(
    "/users",
    response_model=List[UserResponse],
    summary="Get all users",
    description="Retrieve a paginated list of all users in the system. "
                "Supports filtering by status, search terms, and pagination.",
    response_description="List of users with pagination information",
    tags=["users"],
    responses={
        200: {
            "description": "Successfully retrieved users",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "name": "John Doe",
                            "email": "john.doe@example.com"
                        }
                    ]
                }
            }
        },
        400: {"description": "Bad request - invalid parameters"},
        401: {"description": "Unauthorized - authentication required"},
        500: {"description": "Internal server error"}
    }
)
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of users to return"),
    search: Optional[str] = Query(None, description="Search term to filter users by name or email"),
    status: Optional[UserStatusEnum] = Query(None, description="Filter users by status")
):
    """Get paginated list of users with optional filtering.
    
    This endpoint retrieves a list of users with support for:
    - Pagination using skip and limit parameters
    - Search functionality for name and email
    - Status filtering
    - Proper error handling and validation
    
    Args:
        skip: Number of users to skip for pagination
        limit: Maximum number of users to return (1-1000)
        search: Optional search term for filtering
        status: Optional status filter
        
    Returns:
        List of user objects with pagination information
        
    Raises:
        HTTPException: 400 for invalid parameters, 401 for unauthorized access
    """
    pass
```

### **3. API Documentation Configuration**

```python
# ✅ Good: FastAPI app configuration following best practices
app = FastAPI(
    title="HeyGen AI API",
    description="""
    Advanced AI-powered video generation and processing API.
    
    ## Features
    
    * **Video Generation**: Create AI-generated videos from text prompts
    * **Video Processing**: Upload and process existing videos
    * **User Management**: Complete user authentication and profile management
    * **Analytics**: Comprehensive video and user analytics
    
    ## Authentication
    
    This API uses Bearer token authentication. Include your token in the Authorization header:
    
    ```
    Authorization: Bearer your-token-here
    ```
    """,
    version="1.0.0",
    contact={
        "name": "HeyGen AI Support",
        "email": "support@heygen.ai",
        "url": "https://heygen.ai/support"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
```

## ⚡ Performance Best Practices

### **1. Response Caching**

```python
# ✅ Good: Response caching following FastAPI best practices
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@app.get("/users", response_model=List[UserResponse])
@cache(expire=300)  # Cache for 5 minutes
async def get_users():
    """Get users with caching following FastAPI best practices."""
    pass

@app.get("/users/{user_id}", response_model=UserResponse)
@cache(expire=600)  # Cache for 10 minutes
async def get_user(user_id: int):
    """Get user with caching following FastAPI best practices."""
    pass
```

### **2. Database Optimization**

```python
# ✅ Good: Database optimization following FastAPI best practices
from sqlalchemy.orm import Session
from sqlalchemy import select

@app.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get users with database optimization following FastAPI best practices."""
    # Use proper pagination
    query = select(User).offset(skip).limit(limit)
    
    # Use async database operations
    result = await db.execute(query)
    users = result.scalars().all()
    
    return users
```

### **3. Background Tasks**

```python
# ✅ Good: Background tasks following FastAPI best practices
from fastapi import BackgroundTasks

@app.post("/videos", response_model=VideoResponse)
async def create_video(
    video: VideoCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Create video with background processing following FastAPI best practices."""
    # Create video record immediately
    video_record = await create_video_record(video, current_user)
    
    # Add background task for processing
    background_tasks.add_task(process_video, video_record.id)
    
    return video_record

async def process_video(video_id: int):
    """Process video in background following FastAPI best practices."""
    # Long-running video processing
    pass
```

## 🔒 Security Best Practices

### **1. Input Validation**

```python
# ✅ Good: Input validation following FastAPI best practices
from pydantic import validator, root_validator

class UserCreate(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        regex=r'^[a-zA-Z0-9_-]+$',  # Only alphanumeric, underscore, hyphen
        description="Username (alphanumeric, underscore, hyphen only)"
    )
    
    email: EmailStr = Field(..., description="Valid email address")
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        regex=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
        description="Strong password with complexity requirements"
    )
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username following FastAPI best practices."""
        # Check for reserved usernames
        reserved_usernames = ['admin', 'root', 'system', 'api']
        if v.lower() in reserved_usernames:
            raise ValueError('Username is reserved')
        return v
```

### **2. SQL Injection Prevention**

```python
# ✅ Good: SQL injection prevention following FastAPI best practices
from sqlalchemy.orm import Session
from sqlalchemy import select

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user with SQL injection prevention following FastAPI best practices."""
    # Use parameterized queries
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

# ❌ Bad: SQL injection vulnerability
@app.get("/users/{user_id}")
async def get_user(user_id: str, db: Session = Depends(get_db)):
    # Vulnerable to SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result.fetchone()
```

### **3. CORS Configuration**

```python
# ✅ Good: CORS configuration following FastAPI best practices
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
        "https://api.example.com",
        "http://localhost:3000"  # Development only
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "Accept"
    ],
    expose_headers=["X-Total-Count", "X-Page-Count"],
    max_age=3600  # Cache preflight requests for 1 hour
)
```

## 🧪 Testing Best Practices

### **1. Model Testing**

```python
# ✅ Good: Model testing following FastAPI best practices
import pytest
from pydantic import ValidationError

def test_user_create_valid():
    """Test valid user creation following FastAPI best practices."""
    user_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "password": "SecurePass123!",
        "confirm_password": "SecurePass123!"
    }
    
    user = UserCreate(**user_data)
    assert user.name == "John Doe"
    assert user.email == "john.doe@example.com"

def test_user_create_invalid_email():
    """Test invalid email validation following FastAPI best practices."""
    user_data = {
        "name": "John Doe",
        "email": "invalid-email",
        "password": "SecurePass123!",
        "confirm_password": "SecurePass123!"
    }
    
    with pytest.raises(ValidationError):
        UserCreate(**user_data)

def test_user_create_password_mismatch():
    """Test password confirmation validation following FastAPI best practices."""
    user_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "password": "SecurePass123!",
        "confirm_password": "DifferentPass123!"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        UserCreate(**user_data)
    
    assert "Passwords do not match" in str(exc_info.value)
```

### **2. API Testing**

```python
# ✅ Good: API testing following FastAPI best practices
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)

def test_get_users():
    """Test get users endpoint following FastAPI best practices."""
    response = client.get("/users")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 0

def test_create_user():
    """Test create user endpoint following FastAPI best practices."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "SecurePass123!",
        "confirm_password": "SecurePass123!"
    }
    
    response = client.post("/users", json=user_data)
    assert response.status_code == 201
    
    data = response.json()
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "id" in data

def test_get_user_not_found():
    """Test get user not found following FastAPI best practices."""
    response = client.get("/users/999999")
    assert response.status_code == 404
    
    data = response.json()
    assert data["success"] is False
    assert "not found" in data["message"].lower()
```

## 🚀 Deployment Best Practices

### **1. Environment Configuration**

```python
# ✅ Good: Environment configuration following FastAPI best practices
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings following FastAPI best practices."""
    
    # Database
    database_url: str = Field(..., description="Database connection URL")
    database_pool_size: int = Field(default=20, description="Database pool size")
    
    # Security
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=60, description="Requests per minute")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### **2. Production Configuration**

```python
# ✅ Good: Production configuration following FastAPI best practices
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="HeyGen AI API",
    description="Production API for HeyGen AI",
    version="1.0.0",
    docs_url=None,  # Disable docs in production
    redoc_url=None,  # Disable redoc in production
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
```

## 📊 Summary

### **Key Best Practices Implemented**

1. **Data Models**: Comprehensive Pydantic models with validation, documentation, and proper inheritance
2. **Path Operations**: RESTful API design with proper HTTP methods, status codes, and error handling
3. **Middleware**: Request/response processing, authentication, rate limiting, and error handling
4. **Validation**: Input validation, custom validators, and error handling
5. **Documentation**: Automatic API documentation generation with examples
6. **Performance**: Caching, database optimization, and background tasks
7. **Security**: Input validation, SQL injection prevention, and CORS configuration
8. **Testing**: Comprehensive model and API testing
9. **Deployment**: Environment configuration and production settings

### **Implementation Checklist**

- [ ] **Data Models**: Implement Pydantic models with validation and documentation
- [ ] **Path Operations**: Design RESTful API with proper HTTP methods
- [ ] **Middleware**: Add request logging, authentication, and rate limiting
- [ ] **Validation**: Implement input validation and custom validators
- [ ] **Error Handling**: Add comprehensive error handling and responses
- [ ] **Documentation**: Configure automatic API documentation
- [ ] **Performance**: Implement caching and optimization
- [ ] **Security**: Add security measures and input validation
- [ ] **Testing**: Create comprehensive tests
- [ ] **Deployment**: Configure production settings

### **Next Steps**

1. **Integration**: Integrate with existing HeyGen AI services
2. **Customization**: Customize models and middleware for specific needs
3. **Testing**: Implement comprehensive testing suite
4. **Documentation**: Generate detailed API documentation
5. **Monitoring**: Set up production monitoring and logging
6. **Optimization**: Optimize performance and security

This comprehensive FastAPI best practices guide ensures your HeyGen AI API follows industry standards for data models, path operations, and middleware implementation. 