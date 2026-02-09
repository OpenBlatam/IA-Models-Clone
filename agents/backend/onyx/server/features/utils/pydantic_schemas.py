from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic, Callable, ClassVar
from datetime import datetime, date
from enum import Enum
import uuid
import re
from functools import wraps
import time
from pydantic import (
import orjson
import structlog
from .optimized_base_model import OptimizedBaseModel
from .http_response_models import SuccessResponse, ErrorResponse, ListResponse, PaginationInfo
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Pydantic Schema System - Comprehensive Input/Output Validation
Production-ready Pydantic schema system with consistent validation, response models, and Onyx integration.
"""


    BaseModel, ConfigDict, Field, computed_field, field_validator, 
    model_validator, ValidationError, EmailStr, HttpUrl, IPvAnyAddress,
    validator, root_validator
)


logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)
InputT = TypeVar('InputT', bound='BaseInputModel')
OutputT = TypeVar('OutputT', bound='BaseOutputModel')

class ValidationLevel(str, Enum):
    """Validation levels for different environments."""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"

class SchemaConfig:
    """Global schema configuration."""
    validation_level: ValidationLevel = ValidationLevel.NORMAL
    enable_performance_monitoring: bool = True
    enable_audit_logging: bool = True
    max_field_length: int = 10000
    max_nested_depth: int = 10
    allowed_file_types: List[str] = ["jpg", "jpeg", "png", "gif", "pdf", "doc", "docx"]
    max_file_size_mb: int = 50

class BaseInputModel(OptimizedBaseModel):
    """
    Base model for all input validation.
    
    Features:
    - Strict validation with detailed error messages
    - Performance monitoring
    - Audit logging
    - Sanitization and normalization
    - Custom validation rules
    """
    
    model_config = ConfigDict(
        # Inherit from OptimizedBaseModel
        **OptimizedBaseModel.model_config,
        
        # Input-specific settings
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        str_min_length=1,
        
        # Error handling
        error_msg_templates={
            "value_error.missing": "Field '{field_name}' is required",
            "value_error.any_str.min_length": "Field '{field_name}' must be at least {limit_value} characters",
            "value_error.any_str.max_length": "Field '{field_name}' must be at most {limit_value} characters",
            "value_error.email": "Field '{field_name}' must be a valid email address",
            "value_error.url": "Field '{field_name}' must be a valid URL",
        }
    )
    
    # Performance tracking
    _validation_start_time: ClassVar[Optional[float]] = None
    
    def __init__(self, **data: Any) -> None:
        """Initialize with performance tracking."""
        if SchemaConfig.enable_performance_monitoring:
            self._validation_start_time = time.perf_counter()
        
        try:
            super().__init__(**data)
            self._log_validation_success()
        except ValidationError as e:
            self._log_validation_error(e)
            raise
    
    def _log_validation_success(self) -> None:
        """Log successful validation."""
        if SchemaConfig.enable_performance_monitoring and self._validation_start_time:
            duration = time.perf_counter() - self._validation_start_time
            logger.debug(
                "Input validation successful",
                model=self.__class__.__name__,
                duration_ms=duration * 1000,
                field_count=len(self.model_fields)
            )
    
    def _log_validation_error(self, error: ValidationError) -> None:
        """Log validation error."""
        if SchemaConfig.enable_performance_monitoring and self._validation_start_time:
            duration = time.perf_counter() - self._validation_start_time
            logger.warning(
                "Input validation failed",
                model=self.__class__.__name__,
                duration_ms=duration * 1000,
                error_count=len(error.errors()),
                errors=[str(e) for e in error.errors()]
            )
    
    @model_validator(mode='before')
    @classmethod
    def validate_input_data(cls, data: Any) -> bool:
        """Pre-validation hook for input sanitization."""
        if isinstance(data, dict):
            # Sanitize string fields
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = value.strip()
            
            # Check for required fields based on validation level
            if SchemaConfig.validation_level == ValidationLevel.STRICT:
                required_fields = cls._get_required_fields()
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        return data
    
    @classmethod
    def _get_required_fields(cls) -> List[str]:
        """Get list of required fields."""
        required_fields = []
        for field_name, field_info in cls.model_fields.items():
            if field_info.is_required():
                required_fields.append(field_name)
        return required_fields
    
    def sanitize(self) -> Dict[str, Any]:
        """Sanitize input data for safe processing."""
        data = self.model_dump()
        
        # Remove sensitive fields
        sensitive_fields = ['password', 'token', 'secret', 'key']
        for field in sensitive_fields:
            if field in data:
                data[field] = '[REDACTED]'
        
        return data
    
    def validate_business_rules(self) -> List[str]:
        """Validate business-specific rules."""
        errors = []
        
        # Check field length limits
        for field_name, value in self.model_dump().items():
            if isinstance(value, str) and len(value) > SchemaConfig.max_field_length:
                errors.append(f"Field '{field_name}' exceeds maximum length of {SchemaConfig.max_field_length}")
        
        return errors

class BaseOutputModel(OptimizedBaseModel):
    """
    Base model for all output responses.
    
    Features:
    - Consistent response formatting
    - Performance monitoring
    - Data transformation
    - Caching support
    - Serialization optimization
    """
    
    model_config = ConfigDict(
        # Inherit from OptimizedBaseModel
        **OptimizedBaseModel.model_config,
        
        # Output-specific settings
        extra="ignore",
        validate_assignment=False,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            uuid.UUID: str,
        }
    )
    
    # Response metadata
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    
    @computed_field
    @property
    def response_age_seconds(self) -> float:
        """Get response age in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()
    
    def to_response(self, status_code: int = 200, message: str = "Success") -> SuccessResponse:
        """Convert to standardized response format."""
        return SuccessResponse(
            success=True,
            data=self.model_dump(),
            message=message,
            status_code=status_code,
            timestamp=self.timestamp,
            response_id=self.response_id
        )
    
    def to_error_response(self, error_message: str, status_code: int = 400) -> ErrorResponse:
        """Convert to error response format."""
        return ErrorResponse(
            success=False,
            error=error_message,
            status_code=status_code,
            timestamp=self.timestamp,
            response_id=self.response_id
        )

class PaginatedOutputModel(BaseOutputModel, Generic[T]):
    """
    Base model for paginated responses.
    """
    
    items: List[T] = Field(default_factory=list)
    pagination: PaginationInfo = Field(default_factory=PaginationInfo)
    total_count: int = Field(default=0)
    
    @computed_field
    @property
    def item_count(self) -> int:
        """Get number of items in response."""
        return len(self.items)
    
    @computed_field
    @property
    def has_more(self) -> bool:
        """Check if there are more items available."""
        return self.pagination.page < self.pagination.total_pages
    
    def to_list_response(self, message: str = "Items retrieved successfully") -> ListResponse:
        """Convert to list response format."""
        return ListResponse(
            success=True,
            data=self.items,
            pagination=self.pagination,
            total_count=self.total_count,
            message=message,
            timestamp=self.timestamp,
            response_id=self.response_id
        )

class ValidationSchema(BaseModel):
    """
    Schema for defining validation rules.
    """
    
    field_name: str
    field_type: str
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    custom_validator: Optional[str] = None
    
    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Validate regex pattern."""
        if v is not None:
            try:
                re.compile(v)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {v}")
        return v

class SchemaRegistry:
    """
    Registry for managing schemas and their validation rules.
    """
    
    def __init__(self) -> Any:
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._validation_rules: Dict[str, List[ValidationSchema]] = {}
        self._custom_validators: Dict[str, Callable] = {}
    
    def register_schema(self, name: str, schema_class: Type[BaseModel]) -> None:
        """Register a schema class."""
        self._schemas[name] = schema_class
        logger.info(f"Registered schema: {name}")
    
    def get_schema(self, name: str) -> Optional[Type[BaseModel]]:
        """Get a schema class by name."""
        return self._schemas.get(name)
    
    def register_validation_rules(self, schema_name: str, rules: List[ValidationSchema]) -> None:
        """Register validation rules for a schema."""
        self._validation_rules[schema_name] = rules
        logger.info(f"Registered validation rules for schema: {schema_name}")
    
    def register_custom_validator(self, name: str, validator_func: Callable) -> None:
        """Register a custom validator function."""
        self._custom_validators[name] = validator_func
        logger.info(f"Registered custom validator: {name}")
    
    def validate_data(self, schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against registered schema and rules."""
        schema_class = self.get_schema(schema_name)
        if not schema_class:
            raise ValueError(f"Schema not found: {schema_name}")
        
        # Apply custom validation rules
        rules = self._validation_rules.get(schema_name, [])
        for rule in rules:
            if rule.custom_validator and rule.custom_validator in self._custom_validators:
                validator_func = self._custom_validators[rule.custom_validator]
                data = validator_func(data, rule)
        
        # Validate with Pydantic
        return schema_class(**data).model_dump()

# Global schema registry
schema_registry = SchemaRegistry()

class SchemaFactory:
    """
    Factory for creating schemas dynamically.
    """
    
    @staticmethod
    def create_input_schema(
        name: str,
        fields: Dict[str, Any],
        base_class: Type[BaseInputModel] = BaseInputModel,
        **kwargs
    ) -> Type[BaseInputModel]:
        """Create an input schema dynamically."""
        
        # Add common fields
        fields.update({
            'request_id': (str, Field(default_factory=lambda: str(uuid.uuid4()))),
            'timestamp': (datetime, Field(default_factory=datetime.utcnow)),
            'source': (str, Field(default="api")),
        })
        
        # Create the schema class
        schema_class = type(name, (base_class,), fields)
        
        # Register the schema
        schema_registry.register_schema(name, schema_class)
        
        return schema_class
    
    @staticmethod
    def create_output_schema(
        name: str,
        fields: Dict[str, Any],
        base_class: Type[BaseOutputModel] = BaseOutputModel,
        **kwargs
    ) -> Type[BaseOutputModel]:
        """Create an output schema dynamically."""
        
        # Add common fields
        fields.update({
            'response_id': (str, Field(default_factory=lambda: str(uuid.uuid4()))),
            'timestamp': (datetime, Field(default_factory=datetime.utcnow)),
            'version': (str, Field(default="1.0.0")),
        })
        
        # Create the schema class
        schema_class = type(name, (base_class,), fields)
        
        # Register the schema
        schema_registry.register_schema(name, schema_class)
        
        return schema_class

def validate_input(func: Callable) -> Callable:
    """Decorator for input validation."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Extract input model from function signature
        if args and isinstance(args[0], BaseInputModel):
            input_model = args[0]
            
            # Validate business rules
            business_errors = input_model.validate_business_rules()
            if business_errors:
                raise ValidationError(f"Business validation failed: {', '.join(business_errors)}")
            
            # Log input for audit
            if SchemaConfig.enable_audit_logging:
                sanitized_data = input_model.sanitize()
                logger.info(
                    "Input validation completed",
                    model=input_model.__class__.__name__,
                    request_id=getattr(input_model, 'request_id', None),
                    sanitized_data=sanitized_data
                )
        
        return func(*args, **kwargs)
    
    return wrapper

def validate_output(func: Callable) -> Callable:
    """Decorator for output validation."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        result = func(*args, **kwargs)
        
        # Validate output if it's a BaseOutputModel
        if isinstance(result, BaseOutputModel):
            # Log output for audit
            if SchemaConfig.enable_audit_logging:
                logger.info(
                    "Output validation completed",
                    model=result.__class__.__name__,
                    response_id=result.response_id,
                    item_count=getattr(result, 'item_count', None)
                )
        
        return result
    
    return wrapper

# Common field validators
def validate_email(email: str) -> str:
    """Validate and normalize email address."""
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        raise ValueError("Invalid email format")
    return email.lower().strip()

def validate_phone(phone: str) -> str:
    """Validate and normalize phone number."""
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Validate length (assuming 10-15 digits for international numbers)
    if len(digits_only) < 10 or len(digits_only) > 15:
        raise ValueError("Phone number must be 10-15 digits")
    
    return digits_only

def validate_url(url: str) -> str:
    """Validate and normalize URL."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Basic URL validation
    if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", url):
        raise ValueError("Invalid URL format")
    
    return url

def validate_file_type(filename: str) -> str:
    """Validate file type."""
    if '.' not in filename:
        raise ValueError("File must have an extension")
    
    extension = filename.split('.')[-1].lower()
    if extension not in SchemaConfig.allowed_file_types:
        raise ValueError(f"File type '{extension}' not allowed. Allowed types: {SchemaConfig.allowed_file_types}")
    
    return filename

# Example schemas
class UserCreateInput(BaseInputModel):
    """Input schema for user creation."""
    
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    website: Optional[str] = Field(None, max_length=255)
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_email(v)
    
    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_phone(v)
        return v
    
    @field_validator("website")
    @classmethod
    def validate_website(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_url(v)
        return v

class UserOutput(BaseOutputModel):
    """Output schema for user data."""
    
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")
    website: Optional[str] = Field(None, description="Website URL")
    is_active: bool = Field(default=True, description="Account status")
    created_at: datetime = Field(..., description="Account creation date")
    updated_at: datetime = Field(..., description="Last update date")
    
    @computed_field
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Get display name (username or full name)."""
        return self.username or self.full_name

class UserListOutput(PaginatedOutputModel[UserOutput]):
    """Output schema for paginated user list."""
    
    items: List[UserOutput] = Field(default_factory=list, description="List of users")
    total_count: int = Field(default=0, description="Total number of users")
    pagination: PaginationInfo = Field(default_factory=PaginationInfo, description="Pagination information")

class BlogPostCreateInput(BaseInputModel):
    """Input schema for blog post creation."""
    
    title: str = Field(..., min_length=1, max_length=200, description="Post title")
    content: str = Field(..., min_length=10, max_length=10000, description="Post content")
    excerpt: Optional[str] = Field(None, max_length=500, description="Post excerpt")
    tags: List[str] = Field(default_factory=list, max_length=10, description="Post tags")
    category: str = Field(..., min_length=1, max_length=50, description="Post category")
    is_published: bool = Field(default=False, description="Publication status")
    featured_image: Optional[str] = Field(None, description="Featured image URL")
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and clean title."""
        v = v.strip()
        if not v:
            raise ValueError("Title cannot be empty")
        return v
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and clean tags."""
        # Remove duplicates and empty tags
        cleaned_tags = list(set(tag.strip().lower() for tag in v if tag.strip()))
        
        # Limit number of tags
        if len(cleaned_tags) > 10:
            raise ValueError("Maximum 10 tags allowed")
        
        return cleaned_tags
    
    @field_validator("featured_image")
    @classmethod
    def validate_featured_image(cls, v: Optional[str]) -> Optional[str]:
        """Validate featured image URL."""
        if v is not None:
            return validate_url(v)
        return v

class BlogPostOutput(BaseOutputModel):
    """Output schema for blog post data."""
    
    id: str = Field(..., description="Post ID")
    title: str = Field(..., description="Post title")
    content: str = Field(..., description="Post content")
    excerpt: Optional[str] = Field(None, description="Post excerpt")
    tags: List[str] = Field(default_factory=list, description="Post tags")
    category: str = Field(..., description="Post category")
    is_published: bool = Field(..., description="Publication status")
    featured_image: Optional[str] = Field(None, description="Featured image URL")
    author_id: str = Field(..., description="Author ID")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    view_count: int = Field(default=0, description="View count")
    like_count: int = Field(default=0, description="Like count")
    
    @computed_field
    @property
    def reading_time_minutes(self) -> int:
        """Estimate reading time in minutes."""
        words_per_minute = 200
        word_count = len(self.content.split())
        return max(1, word_count // words_per_minute)
    
    @computed_field
    @property
    def is_popular(self) -> bool:
        """Check if post is popular."""
        return self.view_count > 1000 or self.like_count > 100

# Register example schemas
schema_registry.register_schema("UserCreateInput", UserCreateInput)
schema_registry.register_schema("UserOutput", UserOutput)
schema_registry.register_schema("UserListOutput", UserListOutput)
schema_registry.register_schema("BlogPostCreateInput", BlogPostCreateInput)
schema_registry.register_schema("BlogPostOutput", BlogPostOutput)

# Register custom validators
schema_registry.register_custom_validator("validate_email", validate_email)
schema_registry.register_custom_validator("validate_phone", validate_phone)
schema_registry.register_custom_validator("validate_url", validate_url)
schema_registry.register_custom_validator("validate_file_type", validate_file_type) 