from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import (
from pydantic.types import UUID4
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
FastAPI Data Models - Best Practices

This module implements FastAPI data models following official documentation
best practices for Pydantic models, field validation, and response schemas.
"""

    BaseModel, 
    Field, 
    ConfigDict, 
    EmailStr, 
    HttpUrl, 
    validator,
    root_validator,
    computed_field
)

# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, Enum):
    """User roles enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class ProductCategory(str, Enum):
    """Product categories enumeration."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HOME = "home"
    BOOKS = "books"
    SPORTS = "sports"
    BEAUTY = "beauty"
    FOOD = "food"
    OTHER = "other"

class DescriptionTone(str, Enum):
    """Description tone enumeration."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CREATIVE = "creative"
    TECHNICAL = "technical"

class DescriptionLength(str, Enum):
    """Description length enumeration."""
    SHORT = "short"      # 50-100 words
    MEDIUM = "medium"    # 100-200 words
    LONG = "long"        # 200-300 words
    EXTENDED = "extended" # 300+ words

class Language(str, Enum):
    """Supported languages enumeration."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"

class Status(str, Enum):
    """Status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DELETED = "deleted"

class Priority(str, Enum):
    """Priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# =============================================================================
# BASE MODELS
# =============================================================================

class BaseModelConfig:
    """Base configuration for all models."""
    model_config = ConfigDict(
        # Use alias_generator for consistent field naming
        alias_generator=lambda string: string.replace("_", "-"),
        # Allow population by field name
        populate_by_name=True,
        # Validate assignment
        validate_assignment=True,
        # Extra fields behavior
        extra="forbid",
        # JSON encoders
        json_encoders={
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            UUID4: lambda v: str(v)
        }
    )

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )

    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v) -> Any:
        """Set updated_at to current time if not provided."""
        return v or datetime.utcnow()

class IDMixin(BaseModel):
    """Mixin for ID fields."""
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier"
    )

# =============================================================================
# USER MODELS
# =============================================================================

class UserBase(BaseModel, BaseModelConfig):
    """Base user model."""
    email: EmailStr = Field(
        description="User email address",
        examples=["user@example.com"]
    )
    username: str = Field(
        min_length=3,
        max_length=50,
        description="Username",
        examples=["john_doe"]
    )
    is_active: bool = Field(
        default=True,
        description="Whether the user is active"
    )
    role: UserRole = Field(
        default=UserRole.USER,
        description="User role"
    )

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(
        min_length=8,
        description="User password",
        examples=["secure_password_123"]
    )

    @validator('password')
    def validate_password_strength(cls, v) -> bool:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel, BaseModelConfig):
    """User update model."""
    email: Optional[EmailStr] = Field(
        None,
        description="User email address"
    )
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        description="Username"
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether the user is active"
    )
    role: Optional[UserRole] = Field(
        None,
        description="User role"
    )

class User(UserBase, IDMixin, TimestampMixin):
    """Complete user model."""
    is_admin: bool = Field(
        default=False,
        description="Whether the user has admin privileges"
    )
    last_login: Optional[datetime] = Field(
        None,
        description="Last login timestamp"
    )

    @computed_field
    @property
    def display_name(self) -> str:
        """Computed display name."""
        return f"{self.username} ({self.email})"

# =============================================================================
# PRODUCT DESCRIPTION MODELS
# =============================================================================

class ProductDescriptionBase(BaseModel, BaseModelConfig):
    """Base product description model."""
    product_name: str = Field(
        min_length=1,
        max_length=200,
        description="Product name",
        examples=["iPhone 15 Pro"]
    )
    category: ProductCategory = Field(
        description="Product category"
    )
    features: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="Product features",
        examples=[["5G connectivity", "A17 Pro chip", "48MP camera"]]
    )
    target_audience: Optional[str] = Field(
        None,
        max_length=500,
        description="Target audience description"
    )
    tone: DescriptionTone = Field(
        default=DescriptionTone.PROFESSIONAL,
        description="Description tone"
    )
    length: DescriptionLength = Field(
        default=DescriptionLength.MEDIUM,
        description="Description length"
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Description language"
    )

class ProductDescriptionCreate(ProductDescriptionBase):
    """Product description creation model."""
    user_id: Optional[UUID4] = Field(
        None,
        description="User ID who created the description"
    )

    @validator('features')
    def validate_features(cls, v) -> bool:
        """Validate features list."""
        if len(v) > 20:
            raise ValueError('Maximum 20 features allowed')
        return [feature.strip() for feature in v if feature.strip()]

class ProductDescriptionUpdate(BaseModel, BaseModelConfig):
    """Product description update model."""
    product_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Product name"
    )
    category: Optional[ProductCategory] = Field(
        None,
        description="Product category"
    )
    features: Optional[List[str]] = Field(
        None,
        max_items=20,
        description="Product features"
    )
    target_audience: Optional[str] = Field(
        None,
        max_length=500,
        description="Target audience description"
    )
    tone: Optional[DescriptionTone] = Field(
        None,
        description="Description tone"
    )
    length: Optional[DescriptionLength] = Field(
        None,
        description="Description length"
    )
    language: Optional[Language] = Field(
        None,
        description="Description language"
    )

class ProductDescription(ProductDescriptionBase, IDMixin, TimestampMixin):
    """Complete product description model."""
    user_id: UUID4 = Field(
        description="User ID who created the description"
    )
    generated_description: str = Field(
        description="Generated product description"
    )
    status: Status = Field(
        default=Status.ACTIVE,
        description="Description status"
    )
    version: int = Field(
        default=1,
        description="Description version"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @computed_field
    @property
    def word_count(self) -> int:
        """Computed word count."""
        return len(self.generated_description.split())

    @computed_field
    @property
    def character_count(self) -> int:
        """Computed character count."""
        return len(self.generated_description)

# =============================================================================
# GENERATION OPTIONS MODELS
# =============================================================================

class GenerationOptions(BaseModel, BaseModelConfig):
    """Generation options model."""
    use_ai_model: str = Field(
        default="gpt-4",
        description="AI model to use for generation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature (creativity)"
    )
    max_tokens: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum tokens for generation"
    )
    include_keywords: bool = Field(
        default=True,
        description="Include SEO keywords"
    )
    include_call_to_action: bool = Field(
        default=False,
        description="Include call to action"
    )
    custom_prompt: Optional[str] = Field(
        None,
        max_length=1000,
        description="Custom generation prompt"
    )

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ProductDescriptionRequest(BaseModel, BaseModelConfig):
    """Product description generation request."""
    product_name: str = Field(
        min_length=1,
        max_length=200,
        description="Product name"
    )
    category: ProductCategory = Field(
        description="Product category"
    )
    features: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="Product features"
    )
    target_audience: Optional[str] = Field(
        None,
        max_length=500,
        description="Target audience"
    )
    tone: DescriptionTone = Field(
        default=DescriptionTone.PROFESSIONAL,
        description="Description tone"
    )
    length: DescriptionLength = Field(
        default=DescriptionLength.MEDIUM,
        description="Description length"
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Description language"
    )
    options: Optional[GenerationOptions] = Field(
        None,
        description="Generation options"
    )

class ProductDescriptionResponse(BaseModel, BaseModelConfig):
    """Product description generation response."""
    status: str = Field(
        description="Response status"
    )
    message: str = Field(
        description="Response message"
    )
    data: ProductDescription = Field(
        description="Generated product description"
    )
    cached: bool = Field(
        default=False,
        description="Whether response was served from cache"
    )
    generation_time: Optional[float] = Field(
        None,
        description="Generation time in seconds"
    )

# =============================================================================
# BATCH OPERATION MODELS
# =============================================================================

class BatchProduct(BaseModel, BaseModelConfig):
    """Single product in batch operation."""
    product_name: str = Field(
        min_length=1,
        max_length=200,
        description="Product name"
    )
    category: ProductCategory = Field(
        description="Product category"
    )
    features: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="Product features"
    )
    target_audience: Optional[str] = Field(
        None,
        max_length=500,
        description="Target audience"
    )

class BatchGenerationRequest(BaseModel, BaseModelConfig):
    """Batch generation request."""
    products: List[BatchProduct] = Field(
        min_items=1,
        max_items=100,
        description="Products to generate descriptions for"
    )
    options: Optional[GenerationOptions] = Field(
        None,
        description="Generation options for all products"
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Batch processing priority"
    )

class BatchResult(BaseModel, BaseModelConfig):
    """Single batch result."""
    product_name: str = Field(
        description="Product name"
    )
    success: bool = Field(
        description="Whether generation was successful"
    )
    description_id: Optional[UUID4] = Field(
        None,
        description="Generated description ID"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    generation_time: Optional[float] = Field(
        None,
        description="Generation time in seconds"
    )

class BatchGenerationResponse(BaseModel, BaseModelConfig):
    """Batch generation response."""
    status: str = Field(
        description="Response status"
    )
    message: str = Field(
        description="Response message"
    )
    data: Dict[str, Any] = Field(
        description="Batch results"
    )
    summary: Dict[str, Any] = Field(
        description="Batch summary statistics"
    )

# =============================================================================
# PAGINATION MODELS
# =============================================================================

class PaginationParams(BaseModel, BaseModelConfig):
    """Pagination parameters."""
    page: int = Field(
        default=1,
        ge=1,
        description="Page number"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page"
    )
    sort_by: Optional[str] = Field(
        None,
        description="Sort field"
    )
    sort_order: Optional[str] = Field(
        None,
        regex="^(asc|desc)$",
        description="Sort order (asc or desc)"
    )

class PaginationInfo(BaseModel, BaseModelConfig):
    """Pagination information."""
    page: int = Field(
        description="Current page"
    )
    limit: int = Field(
        description="Items per page"
    )
    total: int = Field(
        description="Total items"
    )
    pages: int = Field(
        description="Total pages"
    )
    has_next: bool = Field(
        description="Whether there's a next page"
    )
    has_prev: bool = Field(
        description="Whether there's a previous page"
    )

class PaginatedResponse(BaseModel, BaseModelConfig):
    """Paginated response wrapper."""
    status: str = Field(
        description="Response status"
    )
    message: str = Field(
        description="Response message"
    )
    data: List[Any] = Field(
        description="Response data"
    )
    pagination: PaginationInfo = Field(
        description="Pagination information"
    )

# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorDetail(BaseModel, BaseModelConfig):
    """Error detail model."""
    field: Optional[str] = Field(
        None,
        description="Field that caused the error"
    )
    message: str = Field(
        description="Error message"
    )
    code: Optional[str] = Field(
        None,
        description="Error code"
    )

class ErrorResponse(BaseModel, BaseModelConfig):
    """Error response model."""
    status: str = Field(
        default="error",
        description="Response status"
    )
    message: str = Field(
        description="Error message"
    )
    error_code: int = Field(
        description="HTTP error code"
    )
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

# =============================================================================
# HEALTH AND STATUS MODELS
# =============================================================================

class HealthStatus(BaseModel, BaseModelConfig):
    """Health status model."""
    status: str = Field(
        description="Health status"
    )
    timestamp: datetime = Field(
        description="Health check timestamp"
    )
    version: str = Field(
        description="API version"
    )
    uptime: float = Field(
        description="Service uptime in seconds"
    )

class ComponentHealth(BaseModel, BaseModelConfig):
    """Component health model."""
    name: str = Field(
        description="Component name"
    )
    status: str = Field(
        description="Component status"
    )
    response_time: Optional[float] = Field(
        None,
        description="Component response time"
    )
    error: Optional[str] = Field(
        None,
        description="Component error message"
    )

class SystemHealth(BaseModel, BaseModelConfig):
    """System health model."""
    overall_status: str = Field(
        description="Overall system status"
    )
    components: List[ComponentHealth] = Field(
        description="Component health status"
    )
    timestamp: datetime = Field(
        description="Health check timestamp"
    )

# =============================================================================
# METRICS MODELS
# =============================================================================

class PerformanceMetrics(BaseModel, BaseModelConfig):
    """Performance metrics model."""
    request_count: int = Field(
        description="Total request count"
    )
    average_response_time: float = Field(
        description="Average response time in seconds"
    )
    error_rate: float = Field(
        description="Error rate percentage"
    )
    active_connections: int = Field(
        description="Active connections"
    )
    memory_usage: float = Field(
        description="Memory usage percentage"
    )
    cpu_usage: float = Field(
        description="CPU usage percentage"
    )

class CacheMetrics(BaseModel, BaseModelConfig):
    """Cache metrics model."""
    hit_rate: float = Field(
        description="Cache hit rate percentage"
    )
    total_requests: int = Field(
        description="Total cache requests"
    )
    cache_size: int = Field(
        description="Current cache size"
    )
    evicted_items: int = Field(
        description="Number of evicted items"
    )

# =============================================================================
# ADMIN MODELS
# =============================================================================

class AdminDashboard(BaseModel, BaseModelConfig):
    """Admin dashboard model."""
    total_users: int = Field(
        description="Total number of users"
    )
    total_descriptions: int = Field(
        description="Total number of descriptions"
    )
    active_sessions: int = Field(
        description="Active user sessions"
    )
    system_health: SystemHealth = Field(
        description="System health status"
    )
    performance_metrics: PerformanceMetrics = Field(
        description="Performance metrics"
    )
    recent_errors: List[ErrorResponse] = Field(
        description="Recent system errors"
    )

class SystemConfig(BaseModel, BaseModelConfig):
    """System configuration model."""
    database_url: Optional[str] = Field(
        None,
        description="Database connection URL"
    )
    cache_url: Optional[str] = Field(
        None,
        description="Cache connection URL"
    )
    max_connections: int = Field(
        default=100,
        description="Maximum database connections"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    rate_limit: int = Field(
        default=1000,
        description="Rate limit per hour"
    )
    debug_mode: bool = Field(
        default=False,
        description="Debug mode enabled"
    )

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_product_name(name: str) -> str:
    """Validate product name."""
    if not name.strip():
        raise ValueError("Product name cannot be empty")
    if len(name) > 200:
        raise ValueError("Product name too long")
    return name.strip()

def validate_features(features: List[str]) -> List[str]:
    """Validate product features."""
    if len(features) > 20:
        raise ValueError("Maximum 20 features allowed")
    return [feature.strip() for feature in features if feature.strip()]

# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Export all models for easy import
__all__ = [
    # Enums
    "UserRole", "ProductCategory", "DescriptionTone", "DescriptionLength",
    "Language", "Status", "Priority",
    
    # Base models
    "BaseModelConfig", "TimestampMixin", "IDMixin",
    
    # User models
    "UserBase", "UserCreate", "UserUpdate", "User",
    
    # Product description models
    "ProductDescriptionBase", "ProductDescriptionCreate", 
    "ProductDescriptionUpdate", "ProductDescription",
    
    # Generation models
    "GenerationOptions", "ProductDescriptionRequest", "ProductDescriptionResponse",
    
    # Batch models
    "BatchProduct", "BatchGenerationRequest", "BatchResult", "BatchGenerationResponse",
    
    # Pagination models
    "PaginationParams", "PaginationInfo", "PaginatedResponse",
    
    # Error models
    "ErrorDetail", "ErrorResponse",
    
    # Health models
    "HealthStatus", "ComponentHealth", "SystemHealth",
    
    # Metrics models
    "PerformanceMetrics", "CacheMetrics",
    
    # Admin models
    "AdminDashboard", "SystemConfig",
    
    # Validation utilities
    "validate_product_name", "validate_features"
] 