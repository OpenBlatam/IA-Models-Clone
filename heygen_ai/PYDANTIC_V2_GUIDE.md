# Pydantic v2 Implementation Guide

## Overview

This guide provides comprehensive information about the enhanced Pydantic v2 implementation for the HeyGen AI backend. The implementation includes advanced features, performance optimizations, and enterprise-grade capabilities.

## 🚀 Key Features

### Enhanced Models (`api/models_v2.py`)
- **Computed Fields**: Automatic calculation of derived values
- **Advanced Validators**: Custom validation with comprehensive error handling
- **Performance Optimizations**: Caching and batch processing
- **Structured Error Handling**: Consistent error codes and messages
- **Async Support**: Async validators and serializers
- **Type Safety**: Full type hints and validation

### Advanced Validators (`api/validators_v2.py`)
- **Content Safety**: Script content validation with inappropriate content detection
- **Format Validation**: Email, URL, ID format validation
- **Business Logic**: Voice, avatar, language validation
- **Cross-field Validation**: Script-duration matching, batch limits
- **Performance Caching**: Validation result caching
- **Async Validators**: Database checks and quota validation

### Enhanced Serializers (`api/serializers_v2.py`)
- **Custom Serialization**: DateTime, Decimal, File, URL serialization
- **Performance Monitoring**: Serialization metrics and caching
- **Async Serializers**: Async file and URL serialization
- **Batch Processing**: Efficient batch serialization
- **Smart Serialization**: Automatic method selection

## 📊 Model Architecture

### Base Models

```python
class BaseHeyGenModel(BaseModel):
    """Base model with enhanced Pydantic v2 configuration."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        validate_default=True,
        extra='forbid',  # Reject extra fields
        frozen=False,  # Allow mutation for now
        use_enum_values=True,
        
        # JSON configuration
        json_encoders={
            datetime: serialize_datetime,
            Decimal: serialize_decimal,
        },
        
        # Schema generation
        json_schema_extra={
            "examples": [],
            "additionalProperties": False
        },
        
        # Validation
        str_strip_whitespace=True,
        str_min_length=1,
        
        # Error handling
        error_msg_templates={
            'value_error.missing': 'This field is required',
            'value_error.any_str.min_length': 'Minimum length is {limit_value}',
            'value_error.any_str.max_length': 'Maximum length is {limit_value}',
        }
    )
```

### Timestamped Models

```python
class TimestampedModel(BaseHeyGenModel):
    """Base model with automatic timestamps."""
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    
    @model_validator(mode='after')
    def update_timestamp(self) -> 'TimestampedModel':
        """Update timestamp on model changes."""
        self.updated_at = datetime.now(timezone.utc)
        return self
```

### Identified Models

```python
class IdentifiedModel(BaseHeyGenModel):
    """Base model with automatic ID generation."""
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )
    
    @computed_field
    @property
    def short_id(self) -> str:
        """Get short version of ID."""
        return self.id[:8]
```

## 🔧 Advanced Request Models

### Enhanced Video Request

```python
class CreateVideoRequest(BaseHeyGenModel):
    """Enhanced request model for creating a video."""
    
    script: Annotated[str, PlainValidator(validate_script_content)] = Field(
        ..., 
        min_length=10, 
        max_length=10000,
        description="Script text for the video"
    )
    avatar_id: Annotated[str, PlainValidator(validate_avatar_id)] = Field(
        ..., 
        description="ID of the avatar to use"
    )
    voice_id: Annotated[str, PlainValidator(validate_voice_id)] = Field(
        ..., 
        description="ID of the voice to use"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Language code"
    )
    style: VideoStyle = Field(
        default=VideoStyle.PROFESSIONAL,
        description="Video style"
    )
    resolution: Resolution = Field(
        default=Resolution.FULL_HD_1080P,
        description="Video resolution"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP4,
        description="Output video format"
    )
    quality: QualityLevel = Field(
        default=QualityLevel.HIGH,
        description="Video quality level"
    )
    duration: Annotated[Optional[int], PlainValidator(validate_duration)] = Field(
        None, 
        ge=5, 
        le=3600,
        description="Video duration in seconds (5-3600)"
    )
    background: Optional[HttpUrl] = Field(
        None, 
        description="Background image/video URL"
    )
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom video settings"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @computed_field
    @property
    def estimated_file_size(self) -> int:
        """Estimate file size based on duration and quality."""
        if not self.duration:
            return 0
        
        bitrate = QualityLevel.get_bitrate(self.quality, self.resolution)
        return int((bitrate * self.duration) / 8)  # Convert bits to bytes
    
    @computed_field
    @property
    def estimated_processing_time(self) -> int:
        """Estimate processing time in seconds."""
        base_time = 30  # Base processing time
        duration_factor = (self.duration or 60) / 60  # Factor based on duration
        quality_factor = {"low": 0.5, "medium": 1.0, "high": 1.5, "ultra": 2.0}[self.quality]
        
        return int(base_time * duration_factor * quality_factor)
    
    @computed_field
    @property
    def word_count(self) -> int:
        """Calculate word count from script."""
        return len(self.script.split())
    
    @computed_field
    @property
    def reading_time(self) -> float:
        """Estimate reading time in minutes."""
        words_per_minute = 150
        return self.word_count / words_per_minute
    
    @model_validator(mode='after')
    def validate_script_duration_match(self) -> 'CreateVideoRequest':
        """Validate that script length matches duration if specified."""
        if self.duration and self.reading_time > self.duration / 60:
            raise ValueError(
                f"Script is too long for {self.duration} seconds. "
                f"Estimated reading time: {self.reading_time:.1f} minutes"
            )
        return self
```

## 🛡️ Advanced Validators

### Content Safety Validation

```python
def validate_script_content_v2(v: str) -> str:
    """Enhanced script content validation with comprehensive checks."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_SCRIPT_CONTENT,
            "Script must be a string"
        )
    
    # Strip whitespace
    v = v.strip()
    
    # Check for empty content
    if not v:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_TOO_SHORT,
            "Script cannot be empty"
        )
    
    # Check minimum length
    if len(v) < 10:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_TOO_SHORT,
            f"Script must be at least 10 characters long (current: {len(v)})"
        )
    
    # Check maximum length
    if len(v) > 10000:
        raise PydanticCustomError(
            ValidationErrorCode.SCRIPT_TOO_LONG,
            f"Script cannot exceed 10,000 characters (current: {len(v)})"
        )
    
    # Check for inappropriate content
    inappropriate_patterns = [
        r'\b(spam|scam|phishing|malware|virus)\b',
        r'\b(hack|crack|exploit|vulnerability)\b',
        r'\b(drugs|weapons|illegal)\b',
        r'\b(hate|discrimination|racism)\b'
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, v.lower()):
            raise PydanticCustomError(
                ValidationErrorCode.INAPPROPRIATE_CONTENT,
                "Script contains inappropriate content"
            )
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\b(click here|buy now|limited time|act fast)\b',
        r'\b(free money|get rich|earn money)\b',
        r'\b(password|credit card|social security)\b'
    ]
    
    suspicious_count = sum(1 for pattern in suspicious_patterns if re.search(pattern, v.lower()))
    if suspicious_count > 2:
        raise PydanticCustomError(
            ValidationErrorCode.SUSPICIOUS_CONTENT,
            "Script contains suspicious patterns"
        )
    
    # Check for balanced content (not too repetitive)
    words = v.lower().split()
    if len(words) > 10:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        max_freq = max(word_freq.values())
        if max_freq > len(words) * 0.3:  # More than 30% repetition
            raise PydanticCustomError(
                ValidationErrorCode.INAPPROPRIATE_CONTENT,
                "Script contains too much repetition"
            )
    
    return v
```

### Email Validation

```python
def validate_email_v2(v: str) -> str:
    """Enhanced email validation."""
    if not isinstance(v, str):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Email must be a string"
        )
    
    v = v.strip().lower()
    
    # Basic email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, v):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Invalid email format"
        )
    
    # Check for disposable email domains
    disposable_domains = {
        'tempmail.org', '10minutemail.com', 'guerrillamail.com',
        'mailinator.com', 'throwaway.email', 'temp-mail.org'
    }
    
    domain = v.split('@')[1]
    if domain in disposable_domains:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Disposable email addresses are not allowed"
        )
    
    # Check for suspicious patterns
    if re.search(r'(admin|root|test|demo|example)', v):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_EMAIL_FORMAT,
            "Email contains suspicious patterns"
        )
    
    return v
```

### Async Validators

```python
async def validate_voice_exists_async(voice_id: str) -> str:
    """Async validator to check if voice exists in database."""
    # This would typically check against a database
    # For now, we'll simulate the check
    
    # Simulate database check
    await asyncio.sleep(0.01)  # Simulate database query
    
    # Mock validation - in real implementation, check database
    if voice_id.startswith('invalid_'):
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_VOICE_ID,
            f"Voice ID '{voice_id}' does not exist"
        )
    
    return voice_id


async def validate_user_quota_async(user_id: str, operation: str) -> None:
    """Async validator to check user quota."""
    # Simulate quota check
    await asyncio.sleep(0.01)
    
    # Mock quota validation
    if user_id.startswith('quota_exceeded_'):
        raise PydanticCustomError(
            "QUOTA_EXCEEDED",
            f"User quota exceeded for operation: {operation}"
        )
```

## 🔄 Enhanced Serializers

### DateTime Serializers

```python
def serialize_datetime_iso(v: datetime) -> str:
    """Serialize datetime to ISO format with timezone handling."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.isoformat()


def serialize_datetime_unix(v: datetime) -> int:
    """Serialize datetime to Unix timestamp."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return int(v.timestamp())


def serialize_datetime_readable(v: datetime) -> str:
    """Serialize datetime to human-readable format."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.strftime("%Y-%m-%d %H:%M:%S UTC")
```

### File Serializers

```python
def serialize_file_base64(v: Path) -> str:
    """Serialize file to base64 string."""
    try:
        if not v.exists():
            raise PydanticCustomError(
                SerializationErrorCode.FILE_NOT_FOUND,
                f"File not found: {v}"
            )
        
        # Check file size (10MB limit)
        file_size = v.stat().st_size
        if file_size > 10 * 1024 * 1024:
            raise PydanticCustomError(
                SerializationErrorCode.FILE_TOO_LARGE,
                f"File too large: {file_size} bytes"
            )
        
        with open(v, 'rb') as f:
            content = f.read()
            return base64.b64encode(content).decode('utf-8')
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize file: {e}"
        )


def serialize_file_hash(v: Path) -> str:
    """Serialize file to SHA256 hash."""
    try:
        if not v.exists():
            raise PydanticCustomError(
                SerializationErrorCode.FILE_NOT_FOUND,
                f"File not found: {v}"
            )
        
        with open(v, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to hash file: {e}"
        )
```

### Performance Optimized Serializers

```python
class SerializationCache:
    """Cache for serialization results to improve performance."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached serialization result."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cached serialization result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


def serialize_with_cache(serializer_func: Callable[[Any], Any], value: Any) -> Any:
    """Serialize with caching for performance."""
    # Create cache key
    cache_key = f"{serializer_func.__name__}:{hash(str(value))}"
    
    # Check cache
    cached_result = serialization_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Perform serialization
    result = serializer_func(value)
    
    # Cache result
    serialization_cache.set(cache_key, result)
    
    return result
```

## 📊 Enhanced Response Models

### Video Response with Computed Fields

```python
class VideoResponse(TimestampedModel, IdentifiedModel):
    """Enhanced response model for video generation."""
    
    video_id: str = Field(..., description="Unique video ID")
    status: VideoStatus = Field(..., description="Video generation status")
    output_url: Optional[HttpUrl] = Field(None, description="URL to generated video")
    duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")
    file_size: Annotated[Optional[int], PlainValidator(validate_file_size)] = Field(
        None, 
        description="Video file size in bytes"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Video metadata"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Generation progress percentage")
    processing_time: Optional[float] = Field(None, ge=0, description="Total processing time in seconds")
    
    # Original request data
    script: str = Field(..., description="Original script")
    avatar_id: str = Field(..., description="Avatar ID used")
    voice_id: str = Field(..., description="Voice ID used")
    language: LanguageCode = Field(..., description="Language used")
    style: VideoStyle = Field(..., description="Style used")
    resolution: Resolution = Field(..., description="Resolution used")
    output_format: OutputFormat = Field(..., description="Output format used")
    quality: QualityLevel = Field(..., description="Quality level used")
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if video generation is completed."""
        return self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Check if video generation failed."""
        return self.status == VideoStatus.FAILED
    
    @computed_field
    @property
    def file_size_mb(self) -> Optional[float]:
        """Get file size in MB."""
        if self.file_size is None:
            return None
        return round(self.file_size / (1024 * 1024), 2)
    
    @computed_field
    @property
    def duration_formatted(self) -> Optional[str]:
        """Get formatted duration string."""
        if self.duration is None:
            return None
        
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @computed_field
    @property
    def status_description(self) -> str:
        """Get human-readable status description."""
        return VideoStatus.get_description(self.status)
    
    @computed_field
    @property
    def language_name(self) -> str:
        """Get language name."""
        return LanguageCode.get_language_name(self.language)
    
    @computed_field
    @property
    def style_description(self) -> str:
        """Get style description."""
        return VideoStyle.get_style_description(self.style)
    
    @computed_field
    @property
    def resolution_dimensions(self) -> tuple[int, int]:
        """Get resolution dimensions."""
        return Resolution.get_dimensions(self.resolution)
    
    @computed_field
    @property
    def mime_type(self) -> str:
        """Get MIME type for output format."""
        return OutputFormat.get_mime_type(self.output_format)
```

## 🚀 Performance Optimizations

### Validation Caching

```python
class ValidationCache:
    """Cache for validation results to improve performance."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached validation result."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cached validation result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


def validate_with_cache(validator_func: Callable[[Any], Any], value: Any) -> Any:
    """Validate with caching for performance."""
    # Create cache key
    cache_key = f"{validator_func.__name__}:{hash(str(value))}"
    
    # Check cache
    cached_result = validation_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Perform validation
    result = validator_func(value)
    
    # Cache result
    validation_cache.set(cache_key, result)
    
    return result
```

### Batch Validation

```python
def validate_batch_scripts(scripts: List[str]) -> List[str]:
    """Validate a batch of scripts efficiently."""
    validated_scripts = []
    errors = []
    
    for i, script in enumerate(scripts):
        try:
            validated_script = validate_script_content_v2(script)
            validated_scripts.append(validated_script)
        except PydanticCustomError as e:
            errors.append(f"Script {i + 1}: {e.message}")
    
    if errors:
        raise PydanticCustomError(
            ValidationErrorCode.INVALID_SCRIPT_CONTENT,
            f"Batch validation failed: {'; '.join(errors)}"
        )
    
    return validated_scripts
```

## 📝 Error Handling

### Custom Error Codes

```python
class ValidationErrorCode:
    """Validation error codes for consistent error handling."""
    
    # Content validation
    INVALID_SCRIPT_CONTENT = "INVALID_SCRIPT_CONTENT"
    SCRIPT_TOO_SHORT = "SCRIPT_TOO_SHORT"
    SCRIPT_TOO_LONG = "SCRIPT_TOO_LONG"
    INAPPROPRIATE_CONTENT = "INAPPROPRIATE_CONTENT"
    
    # Format validation
    INVALID_EMAIL_FORMAT = "INVALID_EMAIL_FORMAT"
    INVALID_URL_FORMAT = "INVALID_URL_FORMAT"
    INVALID_PHONE_FORMAT = "INVALID_PHONE_FORMAT"
    INVALID_ID_FORMAT = "INVALID_ID_FORMAT"
    
    # Range validation
    VALUE_TOO_SMALL = "VALUE_TOO_SMALL"
    VALUE_TOO_LARGE = "VALUE_TOO_LARGE"
    DURATION_OUT_OF_RANGE = "DURATION_OUT_OF_RANGE"
    FILE_SIZE_TOO_LARGE = "FILE_SIZE_TOO_LARGE"
    
    # Business logic validation
    INVALID_VOICE_ID = "INVALID_VOICE_ID"
    INVALID_AVATAR_ID = "INVALID_AVATAR_ID"
    INVALID_LANGUAGE_CODE = "INVALID_LANGUAGE_CODE"
    INVALID_RESOLUTION = "INVALID_RESOLUTION"
    INVALID_QUALITY_LEVEL = "INVALID_QUALITY_LEVEL"
    
    # Cross-field validation
    SCRIPT_DURATION_MISMATCH = "SCRIPT_DURATION_MISMATCH"
    BATCH_SIZE_EXCEEDED = "BATCH_SIZE_EXCEEDED"
    TOTAL_SIZE_EXCEEDED = "TOTAL_SIZE_EXCEEDED"
    
    # Security validation
    SUSPICIOUS_CONTENT = "SUSPICIOUS_CONTENT"
    MALICIOUS_URL = "MALICIOUS_URL"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
```

### Error Extraction Utilities

```python
def extract_validation_errors(validation_error: ValidationError) -> List[Dict[str, Any]]:
    """Extract validation errors in a structured format."""
    errors = []
    
    for error in validation_error.errors():
        error_info = {
            'field': '.'.join(str(loc) for loc in error['loc']),
            'message': error['msg'],
            'type': error['type'],
            'input': error.get('input'),
            'ctx': error.get('ctx', {})
        }
        errors.append(error_info)
    
    return errors


def format_validation_error(validation_error: ValidationError) -> str:
    """Format validation error as a user-friendly message."""
    errors = extract_validation_errors(validation_error)
    
    if len(errors) == 1:
        error = errors[0]
        return f"{error['field']}: {error['message']}"
    else:
        error_messages = [f"{error['field']}: {error['message']}" for error in errors]
        return f"Validation failed: {'; '.join(error_messages)}"
```

## 🔧 Usage Examples

### Basic Model Usage

```python
from api.models_v2 import CreateVideoRequest, VideoResponse

# Create a video request
video_request = CreateVideoRequest(
    script="Hello world! This is a test video.",
    avatar_id="avatar_123",
    voice_id="voice_456",
    language="en",
    style="professional",
    duration=30
)

# Access computed fields
print(f"Word count: {video_request.word_count}")
print(f"Reading time: {video_request.reading_time:.1f} minutes")
print(f"Estimated file size: {video_request.estimated_file_size} bytes")
print(f"Estimated processing time: {video_request.estimated_processing_time} seconds")

# Validate the model
try:
    validated_request = video_request.model_validate(video_request.model_dump())
    print("Validation successful!")
except ValidationError as e:
    print(f"Validation failed: {format_validation_error(e)}")
```

### Advanced Validation

```python
from api.validators_v2 import validate_script_content_v2, validate_email_v2

# Validate script content
try:
    validated_script = validate_script_content_v2("This is a valid script content.")
    print("Script validation successful!")
except PydanticCustomError as e:
    print(f"Script validation failed: {e.message}")

# Validate email
try:
    validated_email = validate_email_v2("user@example.com")
    print("Email validation successful!")
except PydanticCustomError as e:
    print(f"Email validation failed: {e.message}")
```

### Async Validation

```python
import asyncio
from api.validators_v2 import validate_voice_exists_async, validate_user_quota_async

async def validate_video_request_async(request_data: dict):
    """Async validation of video request."""
    try:
        # Validate voice exists
        voice_id = await validate_voice_exists_async(request_data['voice_id'])
        
        # Validate user quota
        await validate_user_quota_async(request_data['user_id'], 'video_generation')
        
        print("Async validation successful!")
        return True
    except PydanticCustomError as e:
        print(f"Async validation failed: {e.message}")
        return False

# Run async validation
asyncio.run(validate_video_request_async({
    'voice_id': 'voice_123',
    'user_id': 'user_456'
}))
```

### Custom Serialization

```python
from api.serializers_v2 import serialize_datetime_iso, serialize_file_base64
from pathlib import Path

# Serialize datetime
from datetime import datetime
dt = datetime.now()
iso_string = serialize_datetime_iso(dt)
print(f"ISO datetime: {iso_string}")

# Serialize file
file_path = Path("example.txt")
try:
    base64_content = serialize_file_base64(file_path)
    print(f"File base64: {base64_content[:50]}...")
except PydanticCustomError as e:
    print(f"File serialization failed: {e.message}")
```

## 🔄 Migration Guide

### From Pydantic v1 to v2

#### 1. Update Dependencies

```bash
# Update to Pydantic v2
pip install "pydantic>=2.0.0"

# Install additional dependencies
pip install pydantic-settings
```

#### 2. Update Model Configuration

```python
# Pydantic v1
class MyModel(BaseModel):
    class Config:
        validate_assignment = True
        extra = "forbid"

# Pydantic v2
class MyModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )
```

#### 3. Update Validators

```python
# Pydantic v1
class MyModel(BaseModel):
    field: str
    
    @validator('field')
    def validate_field(cls, v):
        return v.upper()

# Pydantic v2
class MyModel(BaseModel):
    field: str
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        return v.upper()
```

#### 4. Update Serialization

```python
# Pydantic v1
model.dict()
model.json()

# Pydantic v2
model.model_dump()
model.model_dump_json()
```

#### 5. Update Error Handling

```python
# Pydantic v1
from pydantic import ValidationError

# Pydantic v2
from pydantic import ValidationError
# Error handling remains the same
```

## 📊 Performance Monitoring

### Serialization Metrics

```python
from api.serializers_v2 import serialization_metrics, monitor_serialization

@monitor_serialization
def my_serialization_function(data):
    # Your serialization logic here
    return serialized_data

# Get metrics
stats = serialization_metrics.get_stats()
print(f"Total serializations: {stats['total_serializations']}")
print(f"Average time: {stats['avg_serialization_time']:.4f} seconds")
print(f"Error count: {stats['error_count']}")
```

### Validation Metrics

```python
from api.validators_v2 import validation_cache

# Get cache statistics
cache_stats = {
    'size': len(validation_cache.cache),
    'max_size': validation_cache.max_size,
    'hit_rate': 'N/A'  # Would need to implement hit tracking
}

print(f"Validation cache size: {cache_stats['size']}/{cache_stats['max_size']}")
```

## 🎯 Best Practices

### 1. Use Computed Fields

```python
@computed_field
@property
def derived_value(self) -> str:
    """Compute derived value from other fields."""
    return f"{self.field1}_{self.field2}"
```

### 2. Implement Custom Validators

```python
@field_validator('field_name')
@classmethod
def validate_field_name(cls, v: str) -> str:
    """Custom field validation."""
    if not v.strip():
        raise ValueError("Field cannot be empty")
    return v.strip()
```

### 3. Use Model Validators for Cross-field Validation

```python
@model_validator(mode='after')
def validate_cross_fields(self) -> 'MyModel':
    """Validate relationships between fields."""
    if self.field1 and self.field2:
        if self.field1 > self.field2:
            raise ValueError("field1 cannot be greater than field2")
    return self
```

### 4. Implement Async Validators for Database Checks

```python
async def validate_exists_async(value: str) -> str:
    """Async validator for database existence checks."""
    # Database check logic here
    return value
```

### 5. Use Caching for Performance

```python
def validate_with_cache(validator_func, value):
    """Use caching for expensive validations."""
    return validate_with_cache(validator_func, value)
```

### 6. Implement Comprehensive Error Handling

```python
try:
    model = MyModel.model_validate(data)
except ValidationError as e:
    errors = extract_validation_errors(e)
    # Handle errors appropriately
```

## 🔧 Configuration

### Environment Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Pydantic configuration
    validate_assignment: bool = True
    extra: str = "forbid"
    use_enum_values: bool = True
    
    # Validation settings
    max_script_length: int = 10000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    
    class Config:
        env_file = ".env"
```

## 🚀 Conclusion

This enhanced Pydantic v2 implementation provides:

1. **Advanced Models**: Computed fields, automatic timestamps, and ID generation
2. **Comprehensive Validation**: Content safety, format validation, and business logic
3. **Performance Optimizations**: Caching, batch processing, and async support
4. **Enhanced Serialization**: Custom serializers with performance monitoring
5. **Error Handling**: Structured error codes and user-friendly messages
6. **Type Safety**: Full type hints and validation
7. **Monitoring**: Performance metrics and caching statistics

The implementation follows Pydantic v2 best practices and provides a solid foundation for building scalable, maintainable, and performant data validation and serialization systems. 