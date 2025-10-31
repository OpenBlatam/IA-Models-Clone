# Pydantic Schemas Guide

A comprehensive guide for using Pydantic BaseModel schemas for consistent input/output validation and response schemas in the HeyGen AI FastAPI application.

## 🎯 Overview

This guide covers:
- **Base Schemas**: Foundation schemas for consistent API responses
- **User Schemas**: User management, authentication, and profile operations
- **Video Schemas**: Video creation, management, and processing
- **API Response Schemas**: Standardized response formatting and error handling
- **Validation Patterns**: Best practices for data validation
- **Integration Examples**: How to use schemas in FastAPI endpoints

## 📋 Table of Contents

1. [Base Schemas](#base-schemas)
2. [User Schemas](#user-schemas)
3. [Video Schemas](#video-schemas)
4. [API Response Schemas](#api-response-schemas)
5. [Validation Patterns](#validation-patterns)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [Error Handling](#error-handling)

## 🏗️ Base Schemas

### Overview

Base schemas provide the foundation for consistent API responses and common patterns across all endpoints.

### Key Components

#### **BaseResponse**
```python
from api.schemas.base_schemas import BaseResponse, ResponseStatus

class BaseResponse(BaseModel):
    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS)
    message: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(default=None)
```

#### **Generic Data Response**
```python
from api.schemas.base_schemas import DataResponse, PaginatedDataResponse

# Single data response
class UserResponse(DataResponse[UserData]):
    data: UserData = Field(description="User data")

# Paginated data response
class UserListResponse(PaginatedDataResponse[UserData]):
    data: List[UserData] = Field(description="List of users")
```

#### **Common Field Models**
```python
from api.schemas.base_schemas import IDField, TimestampFields, StatusFields

class UserBase(BaseModel):
    id: str = Field(description="User ID")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    is_active: bool = Field(default=True, description="Active status")
```

### Usage Examples

```python
# Success response
response = SuccessResponse(
    message="User created successfully",
    request_id="req_123"
)

# Error response
error_response = ErrorResponse(
    status=ResponseStatus.ERROR,
    message="Validation failed",
    error_code="VALIDATION_ERROR",
    details=[{"field": "email", "message": "Invalid email format"}]
)

# Paginated response
paginated_response = PaginatedResponse(
    page=1,
    per_page=20,
    total=100,
    total_pages=5,
    has_next=True,
    has_prev=False
)
```

## 👤 User Schemas

### Overview

User schemas handle user management, authentication, and profile operations with comprehensive validation.

### Key Components

#### **User Creation**
```python
from api.schemas.user_schemas import UserCreateRequest, UserRole

class UserCreateRequest(BaseRequest):
    email: EmailStr = Field(description="User email address")
    password: str = Field(min_length=8, max_length=128)
    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    role: UserRole = Field(default=UserRole.USER)
    profile: Optional[UserProfile] = Field(default=None)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v
```

#### **User Profile**
```python
class UserProfile(BaseModel):
    bio: Optional[str] = Field(default=None, max_length=500)
    avatar_url: Optional[str] = Field(default=None)
    website: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None, max_length=100)
    company: Optional[str] = Field(default=None, max_length=100)
    job_title: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None)
    date_of_birth: Optional[date] = Field(default=None)
    
    @validator('website')
    def validate_website(cls, v):
        """Validate website URL."""
        if v and not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        """Validate phone number."""
        if v:
            digits_only = re.sub(r'\D', '', v)
            if len(digits_only) < 10:
                raise ValueError('Phone number must have at least 10 digits')
        return v
```

#### **Authentication**
```python
class LoginRequest(BaseRequest):
    email: EmailStr = Field(description="User email address")
    password: str = Field(description="User password")
    remember_me: bool = Field(default=False)
    device_info: Optional[Dict[str, Any]] = Field(default=None)

class OAuthLoginRequest(BaseRequest):
    provider: AuthProvider = Field(description="OAuth provider")
    code: str = Field(description="Authorization code")
    redirect_uri: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None)
```

#### **User Response**
```python
class UserResponse(BaseModel):
    id: str = Field(description="User ID")
    email: EmailStr = Field(description="User email address")
    first_name: str = Field(description="User first name")
    last_name: str = Field(description="User last name")
    role: UserRole = Field(description="User role")
    status: UserStatus = Field(description="User status")
    profile: Optional[UserProfile] = Field(default=None)
    created_at: datetime = Field(description="Account creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    last_login_at: Optional[datetime] = Field(default=None)
    email_verified_at: Optional[datetime] = Field(default=None)
    
    @computed_field
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def is_verified(self) -> bool:
        """Check if user email is verified."""
        return self.email_verified_at is not None
```

### Usage Examples

```python
# Create user
user_data = UserCreateRequest(
    email="john.doe@example.com",
    password="SecurePass123!",
    first_name="John",
    last_name="Doe",
    role=UserRole.USER,
    profile=UserProfile(
        bio="Software developer",
        company="Tech Corp",
        job_title="Senior Developer"
    )
)

# Login
login_data = LoginRequest(
    email="john.doe@example.com",
    password="SecurePass123!",
    remember_me=True
)

# OAuth login
oauth_data = OAuthLoginRequest(
    provider=AuthProvider.GOOGLE,
    code="authorization_code_here",
    redirect_uri="https://app.example.com/callback"
)
```

## 🎥 Video Schemas

### Overview

Video schemas handle video creation, management, processing, and analytics with comprehensive validation.

### Key Components

#### **Video Creation**
```python
from api.schemas.video_schemas import VideoCreateRequest, VideoQuality, VideoFormat

class VideoCreateRequest(BaseRequest):
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    script: VideoScript = Field(description="Video script")
    template: VideoTemplate = Field(description="Video template")
    settings: Optional[VideoSettings] = Field(default=None)
    quality: VideoQuality = Field(default=VideoQuality.HIGH)
    format: VideoFormat = Field(default=VideoFormat.MP4)
    aspect_ratio: VideoAspectRatio = Field(default=VideoAspectRatio.LANDSCAPE)
```

#### **Video Script**
```python
class VideoScript(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    language: str = Field(default="en", min_length=2, max_length=5)
    voice_id: Optional[str] = Field(default=None)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    tone: Optional[str] = Field(default=None)
    
    @validator('content')
    def validate_content(cls, v):
        """Validate script content."""
        if not v or not v.strip():
            raise ValueError('Script content cannot be empty')
        return v.strip()
    
    @validator('speed')
    def validate_speed(cls, v):
        """Validate speech speed."""
        if v < 0.5 or v > 2.0:
            raise ValueError('Speed must be between 0.5 and 2.0')
        return round(v, 2)
```

#### **Video Settings**
```python
class VideoSettings(BaseModel):
    resolution: Optional[str] = Field(default=None)
    frame_rate: Optional[int] = Field(default=None, ge=1, le=120)
    bitrate: Optional[int] = Field(default=None, ge=1000)
    audio_enabled: bool = Field(default=True)
    background_music: Optional[str] = Field(default=None)
    watermark: Optional[str] = Field(default=None)
    subtitles: bool = Field(default=False)
    subtitles_language: Optional[str] = Field(default=None)
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution format."""
        if v and not re.match(r'^\d+x\d+$', v):
            raise ValueError('Resolution must be in format WIDTHxHEIGHT (e.g., 1920x1080)')
        return v
```

#### **Video Response**
```python
class VideoResponse(BaseModel):
    id: str = Field(description="Video ID")
    title: str = Field(description="Video title")
    description: Optional[str] = Field(default=None)
    status: VideoStatus = Field(description="Video status")
    quality: VideoQuality = Field(description="Video quality")
    format: VideoFormat = Field(description="Video format")
    script: VideoScript = Field(description="Video script")
    template: VideoTemplate = Field(description="Video template")
    file_info: Optional[VideoFileInfo] = Field(default=None)
    processing_info: Optional[VideoProcessingInfo] = Field(default=None)
    user_id: str = Field(description="User ID who created the video")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    
    @computed_field
    @property
    def is_ready(self) -> bool:
        """Check if video is ready for viewing."""
        return (
            self.status == VideoStatus.COMPLETED and
            self.file_info is not None
        )
    
    @computed_field
    @property
    def is_processing(self) -> bool:
        """Check if video is being processed."""
        return self.status == VideoStatus.PROCESSING
```

### Usage Examples

```python
# Create video
video_data = VideoCreateRequest(
    title="Product Demo Video",
    description="A comprehensive demo of our new product features",
    script=VideoScript(
        content="Welcome to our product demo. Today we'll show you...",
        language="en",
        voice_id="voice_123",
        speed=1.0,
        tone="professional"
    ),
    template=VideoTemplate(
        id="template_456",
        name="Professional Template",
        category="business"
    ),
    settings=VideoSettings(
        resolution="1920x1080",
        frame_rate=30,
        bitrate=5000,
        audio_enabled=True,
        subtitles=True,
        subtitles_language="en"
    ),
    quality=VideoQuality.HIGH,
    format=VideoFormat.MP4
)

# Search videos
search_data = VideoSearchRequest(
    query="product demo",
    status=VideoStatus.COMPLETED,
    quality=VideoQuality.HIGH,
    page=1,
    per_page=20
)
```

## 📡 API Response Schemas

### Overview

API response schemas provide standardized response formatting and comprehensive error handling.

### Key Components

#### **Standard API Response**
```python
from api.schemas.api_response_schemas import StandardAPIResponse, SuccessAPIResponse

class StandardAPIResponse(BaseModel):
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    version: str = Field(default="1.0.0", description="API version")
```

#### **Error Handling**
```python
from api.schemas.api_response_schemas import APIError, DetailedErrorResponse

class APIError(BaseModel):
    code: APIErrorCode = Field(description="Error code")
    message: str = Field(description="Error message")
    details: Optional[List[Dict[str, Any]]] = Field(default=None)
    field_errors: Optional[List[ValidationError]] = Field(default=None)
    suggestions: Optional[List[str]] = Field(default=None)
    documentation_url: Optional[str] = Field(default=None)
    retry_after: Optional[int] = Field(default=None)
```

#### **Factory Functions**
```python
from api.schemas.api_response_schemas import (
    create_success_response,
    create_error_response,
    create_validation_error_response,
    create_not_found_response
)

# Success response
response = create_success_response(
    data=user_data,
    message="User created successfully",
    request_id="req_123"
)

# Error response
error_response = create_error_response(
    code=APIErrorCode.VALIDATION_ERROR,
    message="Validation failed",
    details=[{"field": "email", "message": "Invalid email format"}],
    suggestions=["Check the field_errors array for specific validation issues"],
    request_id="req_123"
)

# Validation error response
validation_response = create_validation_error_response(
    field_errors=[
        ValidationError(field="email", message="Invalid email format"),
        ValidationError(field="password", message="Password too weak")
    ],
    request_id="req_123"
)

# Not found response
not_found_response = create_not_found_response(
    resource_type="User",
    resource_id="user_123",
    request_id="req_123"
)
```

### Usage Examples

```python
# Success response with data
@router.post("/users")
async def create_user(user_data: UserCreateRequest):
    try:
        user = await user_service.create_user(user_data)
        return create_success_response(
            data=UserResponse.from_orm(user),
            message="User created successfully",
            request_id=request.state.request_id
        )
    except ValidationError as e:
        return create_validation_error_response(
            field_errors=[ValidationError(field=field, message=msg) for field, msg in e.errors()],
            request_id=request.state.request_id
        )

# Error response
@router.get("/users/{user_id}")
async def get_user(user_id: str):
    try:
        user = await user_service.get_user(user_id)
        if not user:
            return create_not_found_response(
                resource_type="User",
                resource_id=user_id,
                request_id=request.state.request_id
            )
        return create_success_response(
            data=UserResponse.from_orm(user),
            request_id=request.state.request_id
        )
    except Exception as e:
        return create_error_response(
            code=APIErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve user",
            request_id=request.state.request_id
        )
```

## ✅ Validation Patterns

### Overview

Validation patterns ensure data integrity and provide clear error messages.

### Common Validation Patterns

#### **Email Validation**
```python
from pydantic import EmailStr, validator

class UserCreateRequest(BaseRequest):
    email: EmailStr = Field(description="User email address")
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if not v or not v.strip():
            raise ValueError('Email cannot be empty')
        return v.lower().strip()
```

#### **Password Validation**
```python
@validator('password')
def validate_password(cls, v):
    """Validate password strength."""
    if len(v) < 8:
        raise ValueError('Password must be at least 8 characters long')
    
    if not re.search(r'[A-Z]', v):
        raise ValueError('Password must contain at least one uppercase letter')
    
    if not re.search(r'[a-z]', v):
        raise ValueError('Password must contain at least one lowercase letter')
    
    if not re.search(r'\d', v):
        raise ValueError('Password must contain at least one digit')
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
        raise ValueError('Password must contain at least one special character')
    
    return v
```

#### **Cross-Field Validation**
```python
@root_validator
def validate_passwords(cls, values):
    """Validate password update."""
    new_password = values.get('new_password')
    confirm_password = values.get('confirm_password')
    
    if new_password != confirm_password:
        raise ValueError('New password and confirmation do not match')
    
    return values
```

#### **Custom Validation**
```python
@validator('phone')
def validate_phone(cls, v):
    """Validate phone number."""
    if v:
        digits_only = re.sub(r'\D', '', v)
        if len(digits_only) < 10:
            raise ValueError('Phone number must have at least 10 digits')
    return v

@validator('website')
def validate_website(cls, v):
    """Validate website URL."""
    if v and not v.startswith(('http://', 'https://')):
        v = 'https://' + v
    return v
```

### Validation Error Handling

```python
from fastapi import HTTPException
from pydantic import ValidationError

@router.post("/users")
async def create_user(user_data: UserCreateRequest):
    try:
        user = await user_service.create_user(user_data)
        return create_success_response(
            data=UserResponse.from_orm(user),
            message="User created successfully"
        )
    except ValidationError as e:
        field_errors = []
        for error in e.errors():
            field_errors.append(ValidationError(
                field=error['loc'][0],
                message=error['msg'],
                value=error.get('input')
            ))
        
        return create_validation_error_response(
            field_errors=field_errors,
            message="Validation failed"
        )
```

## 🔗 Integration Examples

### FastAPI Endpoint Integration

#### **User Management Endpoints**
```python
from fastapi import APIRouter, Depends, HTTPException
from api.schemas.user_schemas import (
    UserCreateRequest, UserUpdateRequest, UserResponse, UserListResponse
)
from api.schemas.api_response_schemas import (
    create_success_response, create_error_response, create_not_found_response
)

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=SuccessAPIResponse[UserResponse])
async def create_user(
    user_data: UserCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new user."""
    try:
        user = await user_service.create_user(user_data)
        return create_success_response(
            data=UserResponse.from_orm(user),
            message="User created successfully"
        )
    except ValidationError as e:
        return create_validation_error_response(
            field_errors=[ValidationError(field=field, message=msg) for field, msg in e.errors()]
        )

@router.get("/", response_model=PaginatedAPIResponse[UserResponse])
async def list_users(
    search: UserSearchRequest = Depends(),
    current_user: User = Depends(get_current_user)
):
    """List users with pagination and search."""
    try:
        users, pagination = await user_service.list_users(search)
        return PaginatedAPIResponse(
            success=True,
            data=[UserResponse.from_orm(user) for user in users],
            pagination=pagination
        )
    except Exception as e:
        return create_error_response(
            code=APIErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve users"
        )

@router.get("/{user_id}", response_model=SuccessAPIResponse[UserResponse])
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get user by ID."""
    try:
        user = await user_service.get_user(user_id)
        if not user:
            return create_not_found_response("User", user_id)
        
        return create_success_response(
            data=UserResponse.from_orm(user)
        )
    except Exception as e:
        return create_error_response(
            code=APIErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve user"
        )

@router.put("/{user_id}", response_model=SuccessAPIResponse[UserResponse])
async def update_user(
    user_id: str,
    user_data: UserUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update user."""
    try:
        user = await user_service.update_user(user_id, user_data)
        if not user:
            return create_not_found_response("User", user_id)
        
        return create_success_response(
            data=UserResponse.from_orm(user),
            message="User updated successfully"
        )
    except ValidationError as e:
        return create_validation_error_response(
            field_errors=[ValidationError(field=field, message=msg) for field, msg in e.errors()]
        )

@router.delete("/{user_id}", response_model=SuccessAPIResponse[bool])
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete user."""
    try:
        success = await user_service.delete_user(user_id)
        if not success:
            return create_not_found_response("User", user_id)
        
        return create_success_response(
            data=True,
            message="User deleted successfully"
        )
    except Exception as e:
        return create_error_response(
            code=APIErrorCode.INTERNAL_ERROR,
            message="Failed to delete user"
        )
```

#### **Video Management Endpoints**
```python
from api.schemas.video_schemas import (
    VideoCreateRequest, VideoUpdateRequest, VideoResponse, VideoListResponse
)

@router.post("/", response_model=SuccessAPIResponse[VideoResponse])
async def create_video(
    video_data: VideoCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new video."""
    try:
        video = await video_service.create_video(video_data, current_user.id)
        return create_success_response(
            data=VideoResponse.from_orm(video),
            message="Video creation started"
        )
    except ValidationError as e:
        return create_validation_error_response(
            field_errors=[ValidationError(field=field, message=msg) for field, msg in e.errors()]
        )

@router.get("/", response_model=PaginatedAPIResponse[VideoResponse])
async def list_videos(
    search: VideoSearchRequest = Depends(),
    current_user: User = Depends(get_current_user)
):
    """List videos with pagination and search."""
    try:
        videos, pagination = await video_service.list_videos(search, current_user.id)
        return PaginatedAPIResponse(
            success=True,
            data=[VideoResponse.from_orm(video) for video in videos],
            pagination=pagination
        )
    except Exception as e:
        return create_error_response(
            code=APIErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve videos"
        )

@router.get("/{video_id}", response_model=SuccessAPIResponse[VideoResponse])
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get video by ID."""
    try:
        video = await video_service.get_video(video_id, current_user.id)
        if not video:
            return create_not_found_response("Video", video_id)
        
        return create_success_response(
            data=VideoResponse.from_orm(video)
        )
    except Exception as e:
        return create_error_response(
            code=APIErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve video"
        )
```

### Database Integration

#### **SQLAlchemy Model Integration**
```python
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True  # For SQLAlchemy models
```

#### **Repository Pattern Integration**
```python
class UserRepository:
    def __init__(self, db: Session):
        self.db = db
    
    async def create_user(self, user_data: UserCreateRequest) -> User:
        """Create a new user."""
        user = User(
            id=str(uuid.uuid4()),
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return await self.db.query(User).filter(User.id == user_id).first()
    
    async def list_users(self, search: UserSearchRequest) -> Tuple[List[User], PaginationInfo]:
        """List users with pagination."""
        query = self.db.query(User)
        
        if search.query:
            query = query.filter(
                or_(
                    User.first_name.ilike(f"%{search.query}%"),
                    User.last_name.ilike(f"%{search.query}%"),
                    User.email.ilike(f"%{search.query}%")
                )
            )
        
        if search.role:
            query = query.filter(User.role == search.role)
        
        if search.status:
            query = query.filter(User.status == search.status)
        
        total = await query.count()
        users = await query.offset(search.offset).limit(search.per_page).all()
        
        pagination = PaginationInfo(
            page=search.page,
            per_page=search.per_page,
            total=total,
            total_pages=(total + search.per_page - 1) // search.per_page,
            has_next=search.page * search.per_page < total,
            has_prev=search.page > 1
        )
        
        return users, pagination
```

## 🏆 Best Practices

### 1. Schema Design

#### **Consistent Naming**
```python
# ✅ Good: Consistent naming conventions
class UserCreateRequest(BaseRequest):
    email: EmailStr
    password: str

class UserUpdateRequest(BaseRequest):
    email: Optional[EmailStr] = None
    password: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str

# ❌ Bad: Inconsistent naming
class CreateUser(BaseRequest):
    email: EmailStr

class UpdateUserData(BaseRequest):
    email: Optional[EmailStr] = None

class UserData(BaseModel):
    id: str
    email: str
```

#### **Field Validation**
```python
# ✅ Good: Comprehensive validation
class VideoCreateRequest(BaseRequest):
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(max_length=1000)
    quality: VideoQuality = Field(default=VideoQuality.HIGH)
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Video title cannot be empty')
        return v.strip()

# ❌ Bad: Minimal validation
class VideoCreateRequest(BaseRequest):
    title: str
    description: Optional[str] = None
    quality: VideoQuality = VideoQuality.HIGH
```

#### **Computed Fields**
```python
# ✅ Good: Use computed fields for derived data
class UserResponse(BaseModel):
    first_name: str
    last_name: str
    
    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def is_verified(self) -> bool:
        return self.email_verified_at is not None

# ❌ Bad: Store derived data
class UserResponse(BaseModel):
    first_name: str
    last_name: str
    full_name: str  # Redundant
    is_verified: bool  # Redundant
```

### 2. Error Handling

#### **Structured Error Responses**
```python
# ✅ Good: Structured error responses
@router.post("/users")
async def create_user(user_data: UserCreateRequest):
    try:
        user = await user_service.create_user(user_data)
        return create_success_response(
            data=UserResponse.from_orm(user),
            message="User created successfully"
        )
    except ValidationError as e:
        return create_validation_error_response(
            field_errors=[ValidationError(field=field, message=msg) for field, msg in e.errors()]
        )
    except DuplicateEmailError:
        return create_error_response(
            code=APIErrorCode.RESOURCE_ALREADY_EXISTS,
            message="User with this email already exists"
        )

# ❌ Bad: Generic error responses
@router.post("/users")
async def create_user(user_data: UserCreateRequest):
    try:
        user = await user_service.create_user(user_data)
        return {"success": True, "data": user}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### **Validation Error Details**
```python
# ✅ Good: Detailed validation errors
class ValidationError(BaseModel):
    field: str = Field(description="Field name with error")
    message: str = Field(description="Error message")
    value: Optional[Any] = Field(default=None, description="Invalid value")
    code: Optional[str] = Field(default=None, description="Error code")

# ❌ Bad: Generic error messages
class ValidationError(BaseModel):
    message: str
```

### 3. Performance Optimization

#### **Lazy Loading**
```python
# ✅ Good: Lazy loading for optional fields
class UserResponse(BaseModel):
    id: str
    email: str
    profile: Optional[UserProfile] = None
    
    @computed_field
    @property
    def profile_complete(self) -> bool:
        return self.profile is not None and self.profile.bio is not None

# ❌ Bad: Always load all data
class UserResponse(BaseModel):
    id: str
    email: str
    profile: UserProfile  # Always required
```

#### **Pagination**
```python
# ✅ Good: Efficient pagination
class PaginatedResponse(BaseModel):
    data: List[Any]
    pagination: PaginationInfo
    
    @computed_field
    @property
    def has_more(self) -> bool:
        return self.pagination.has_next

# ❌ Bad: Load all data
class PaginatedResponse(BaseModel):
    data: List[Any]
    total: int
    # No pagination info
```

### 4. Security

#### **Input Sanitization**
```python
# ✅ Good: Sanitize inputs
class UserCreateRequest(BaseRequest):
    email: EmailStr
    
    @validator('email')
    def validate_email(cls, v):
        if not v or not v.strip():
            raise ValueError('Email cannot be empty')
        return v.lower().strip()

# ❌ Bad: No sanitization
class UserCreateRequest(BaseRequest):
    email: str  # No validation
```

#### **Sensitive Data Handling**
```python
# ✅ Good: Exclude sensitive data from responses
class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    # No password field
    
    class Config:
        exclude = {"password", "hashed_password"}

# ❌ Bad: Include sensitive data
class UserResponse(BaseModel):
    id: str
    email: str
    password: str  # Sensitive data exposed
```

## 🚀 Advanced Patterns

### 1. Generic Schemas

```python
from typing import Generic, TypeVar, List

T = TypeVar('T')

class DataResponse(GenericModel, Generic[T]):
    success: bool = True
    data: T
    message: Optional[str] = None

class PaginatedDataResponse(GenericModel, Generic[T]):
    success: bool = True
    data: List[T]
    pagination: PaginationInfo
    message: Optional[str] = None

# Usage
UserResponse = DataResponse[UserData]
UserListResponse = PaginatedDataResponse[UserData]
```

### 2. Conditional Fields

```python
from pydantic import root_validator

class VideoResponse(BaseModel):
    id: str
    status: VideoStatus
    file_info: Optional[VideoFileInfo] = None
    
    @root_validator
    def validate_file_info(cls, values):
        status = values.get('status')
        file_info = values.get('file_info')
        
        if status == VideoStatus.COMPLETED and not file_info:
            raise ValueError('Completed videos must have file information')
        
        return values
```

### 3. Custom Validators

```python
from pydantic import validator
import re

class PhoneNumber(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError('string required')
        
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', v)
        
        if len(digits_only) < 10:
            raise ValueError('Phone number must have at least 10 digits')
        
        return cls(digits_only)

class UserCreateRequest(BaseRequest):
    phone: Optional[PhoneNumber] = None
```

### 4. Schema Inheritance

```python
class BaseUser(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str

class UserCreateRequest(BaseUser):
    password: str
    confirm_password: str
    
    @root_validator
    def validate_passwords(cls, values):
        if values.get('password') != values.get('confirm_password'):
            raise ValueError('Passwords do not match')
        return values

class UserUpdateRequest(BaseUser):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
```

## 📚 Additional Resources

- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [FastAPI Request Body](https://fastapi.tiangolo.com/tutorial/body/)
- [FastAPI Response Model](https://fastapi.tiangolo.com/tutorial/response-model/)
- [Pydantic Validation](https://pydantic-docs.helpmanual.io/usage/validators/)

## 🚀 Next Steps

1. **Implement the schema system** in your FastAPI application
2. **Create custom validators** for your specific business logic
3. **Set up automated testing** for schema validation
4. **Document your schemas** with examples and descriptions
5. **Monitor validation errors** in production
6. **Optimize schema performance** based on usage patterns
7. **Create schema migration tools** for API versioning

This comprehensive Pydantic schema system provides your HeyGen AI API with:
- **Consistent data validation** across all endpoints
- **Structured error responses** with detailed information
- **Type safety** and IDE support
- **Performance optimization** through efficient validation
- **Security** through input sanitization and validation
- **Maintainability** through clear schema organization

The system is designed to scale with your application while maintaining consistency and reliability across all API endpoints. 