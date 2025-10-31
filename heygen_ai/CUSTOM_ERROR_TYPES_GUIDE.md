# Custom Error Types and Error Factories Guide

## Overview

This guide documents the comprehensive custom error types and error factories implemented in the HeyGen AI FastAPI backend. The system provides domain-specific error types and factory methods for consistent, type-safe error handling across the application.

## Key Features

### 1. Domain-Specific Error Types
- **Video Processing Errors**: Specific errors for video generation, rendering, and processing
- **Voice Synthesis Errors**: Errors related to text-to-speech and voice processing
- **Template Processing Errors**: Errors for template rendering and processing
- **File Processing Errors**: Errors for file upload, download, and processing operations
- **Business Logic Errors**: Errors for quota limits, content moderation, and feature availability

### 2. Error Factory Pattern
- **Consistent Error Creation**: Standardized error creation with automatic logging
- **Domain-Specific Factories**: Specialized factory methods for different error types
- **Convenience Methods**: Pre-built error creation for common scenarios
- **Context Preservation**: Automatic context and metadata preservation

### 3. Type Safety and Consistency
- **Strongly Typed Errors**: Each error type has specific parameters and validation
- **Consistent Error Structure**: All errors follow the same base structure
- **Automatic Logging**: All errors are automatically logged with full context
- **User-Friendly Messages**: Automatic generation of user-friendly error messages

## Error Categories

### Core Error Categories
```python
class ErrorCategory(Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERIALIZATION = "serialization"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    VIDEO_PROCESSING = "video_processing"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_EXHAUSTED = "retry_exhausted"
    CONCURRENCY = "concurrency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
```

### Domain-Specific Error Categories
```python
# Video Processing
VIDEO_GENERATION = "video_generation"
VIDEO_RENDERING = "video_rendering"
VIDEO_UPLOAD = "video_upload"
VIDEO_DOWNLOAD = "video_download"

# Content Processing
TEMPLATE_PROCESSING = "template_processing"
VOICE_SYNTHESIS = "voice_synthesis"
AUDIO_PROCESSING = "audio_processing"
FILE_PROCESSING = "file_processing"

# Business Logic
API_INTEGRATION = "api_integration"
PAYMENT_PROCESSING = "payment_processing"
USER_MANAGEMENT = "user_management"
CONTENT_MODERATION = "content_moderation"
QUOTA_EXCEEDED = "quota_exceeded"
FEATURE_UNAVAILABLE = "feature_unavailable"
MAINTENANCE_MODE = "maintenance_mode"
DEPRECATED_FEATURE = "deprecated_feature"
```

## Custom Error Types

### Video Processing Errors

#### VideoGenerationError
```python
class VideoGenerationError(HeyGenBaseError):
    """Video generation specific error"""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        generation_stage: Optional[str] = None,
        template_id: Optional[str] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create video generation error
error = error_factory.video_generation_error(
    message="Template rendering failed",
    video_id="vid123",
    generation_stage="template_rendering",
    template_id="tpl456",
    context={"user_id": "user123"}
)
```

#### VideoRenderingError
```python
class VideoRenderingError(HeyGenBaseError):
    """Video rendering specific error"""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        rendering_stage: Optional[str] = None,
        render_settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create video rendering error
error = error_factory.video_rendering_error(
    message="Rendering pipeline failed",
    video_id="vid123",
    rendering_stage="final_rendering",
    render_settings={"quality": "high", "format": "mp4"},
    context={"user_id": "user123"}
)
```

### Voice Synthesis Errors

#### VoiceSynthesisError
```python
class VoiceSynthesisError(HeyGenBaseError):
    """Voice synthesis specific error"""
    
    def __init__(
        self,
        message: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create voice synthesis error
error = error_factory.voice_synthesis_error(
    message="Voice synthesis failed",
    voice_id="voice_en_001",
    language="en-US",
    text_length=1500,
    context={"user_id": "user123"}
)
```

### Template Processing Errors

#### TemplateProcessingError
```python
class TemplateProcessingError(HeyGenBaseError):
    """Template processing specific error"""
    
    def __init__(
        self,
        message: str,
        template_id: Optional[str] = None,
        template_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create template processing error
error = error_factory.template_processing_error(
    message="Template validation failed",
    template_id="tpl456",
    template_type="business_presentation",
    processing_stage="validation",
    context={"user_id": "user123"}
)
```

### File Processing Errors

#### FileProcessingError
```python
class FileProcessingError(HeyGenBaseError):
    """File processing specific error"""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create file processing error
error = error_factory.file_processing_error(
    message="File upload failed",
    file_path="/uploads/video.mp4",
    file_type="video/mp4",
    file_size=1024000,
    operation="upload",
    context={"user_id": "user123"}
)
```

### Business Logic Errors

#### QuotaExceededError
```python
class QuotaExceededError(HeyGenBaseError):
    """Quota exceeded error"""
    
    def __init__(
        self,
        message: str,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        limit: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create quota exceeded error
error = error_factory.quota_exceeded_error(
    message="Video generation quota exceeded",
    quota_type="video_generation",
    current_usage=15,
    limit=10,
    reset_time=datetime.utcnow() + timedelta(hours=1),
    context={"user_id": "user123"}
)
```

#### ContentModerationError
```python
class ContentModerationError(HeyGenBaseError):
    """Content moderation error"""
    
    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        moderation_result: Optional[str] = None,
        flagged_content: Optional[str] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create content moderation error
error = error_factory.content_moderation_error(
    message="Content violation detected",
    content_type="script",
    moderation_result="violation",
    flagged_content="inappropriate language",
    context={"user_id": "user123"}
)
```

#### FeatureUnavailableError
```python
class FeatureUnavailableError(HeyGenBaseError):
    """Feature unavailable error"""
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        reason: Optional[str] = None,
        available_alternatives: Optional[List[str]] = None,
        **kwargs
    ):
```

**Usage:**
```python
# Create feature unavailable error
error = error_factory.feature_unavailable_error(
    message="Feature temporarily unavailable",
    feature_name="advanced_rendering",
    reason="maintenance",
    available_alternatives=["basic_rendering", "standard_rendering"],
    context={"user_id": "user123"}
)
```

## Error Factory Methods

### Domain-Specific Factory Methods

#### Video Processing Factories
```python
# Video generation error
error_factory.video_generation_error(
    message="Generation failed",
    video_id="vid123",
    generation_stage="initialization",
    template_id="tpl456"
)

# Video rendering error
error_factory.video_rendering_error(
    message="Rendering failed",
    video_id="vid123",
    rendering_stage="final_rendering",
    render_settings={"quality": "high"}
)
```

#### Voice Synthesis Factories
```python
# Voice synthesis error
error_factory.voice_synthesis_error(
    message="Synthesis failed",
    voice_id="voice_en_001",
    language="en-US",
    text_length=1500
)
```

#### Template Processing Factories
```python
# Template processing error
error_factory.template_processing_error(
    message="Processing failed",
    template_id="tpl456",
    template_type="business_presentation",
    processing_stage="validation"
)
```

#### File Processing Factories
```python
# File processing error
error_factory.file_processing_error(
    message="Upload failed",
    file_path="/uploads/video.mp4",
    file_type="video/mp4",
    file_size=1024000,
    operation="upload"
)
```

### Convenience Factory Methods

#### Common Error Scenarios
```python
# Invalid video ID
error_factory.invalid_video_id_error("invalid_video_id")

# Video not found
error_factory.video_not_found_error("vid123")

# Template not found
error_factory.template_not_found_error("tpl456")

# Voice not found
error_factory.voice_not_found_error("voice_en_001")

# User quota exceeded
error_factory.user_quota_exceeded_error(
    user_id="user123",
    quota_type="video_generation",
    current_usage=15,
    limit=10
)

# Content violation
error_factory.content_violation_error(
    content_type="script",
    violation_reason="inappropriate language"
)

# Video generation timeout
error_factory.video_generation_timeout_error(
    video_id="vid123",
    timeout_duration=300.0
)

# Voice synthesis failed
error_factory.voice_synthesis_failed_error(
    voice_id="voice_en_001",
    language="en-US",
    text_length=1500
)

# Template processing failed
error_factory.template_processing_failed_error(
    template_id="tpl456",
    template_type="business_presentation",
    processing_stage="rendering"
)

# File upload failed
error_factory.file_upload_failed_error(
    file_path="/uploads/video.mp4",
    file_type="video/mp4",
    file_size=1024000
)

# Feature deprecated
error_factory.feature_deprecated_error(
    feature_name="old_rendering_engine",
    deprecation_date=datetime.utcnow(),
    replacement_feature="new_rendering_engine"
)
```

## User-Friendly Message Generation

### Domain-Specific Messages
```python
# Video generation messages
UserFriendlyMessageGenerator.get_video_generation_message("template_rendering")
# Returns: "Video generation failed at template_rendering. Please try again."

# Video rendering messages
UserFriendlyMessageGenerator.get_video_rendering_message("final_rendering")
# Returns: "Video rendering failed at final_rendering. Please try again."

# Voice synthesis messages
UserFriendlyMessageGenerator.get_voice_synthesis_message("en-US")
# Returns: "Voice synthesis failed for en-US. Please try a different voice or language."

# Template processing messages
UserFriendlyMessageGenerator.get_template_processing_message("business_presentation")
# Returns: "Template processing failed for business_presentation. Please try a different template."

# File processing messages
UserFriendlyMessageGenerator.get_file_processing_message("upload")
# Returns: "File upload failed. Please try again."

# Quota exceeded messages
UserFriendlyMessageGenerator.get_quota_exceeded_message("video_generation", reset_time)
# Returns: "Your video_generation quota has been exceeded. It will reset at 14:30."

# Content moderation messages
UserFriendlyMessageGenerator.get_content_moderation_message("script")
# Returns: "Your script content was flagged. Please review and modify it."

# Feature unavailable messages
UserFriendlyMessageGenerator.get_feature_unavailable_message("advanced_rendering", "maintenance")
# Returns: "The advanced_rendering feature is currently unavailable: maintenance."

# Maintenance mode messages
UserFriendlyMessageGenerator.get_maintenance_mode_message(30)
# Returns: "We're currently performing maintenance. Please try again in 30 minutes."

# Deprecated feature messages
UserFriendlyMessageGenerator.get_deprecated_feature_message("old_engine", "new_engine")
# Returns: "The old_engine feature has been deprecated. Please use new_engine instead."
```

## Error Handling in Routes

### Using Custom Error Types in Video Routes
```python
@router.post("/generate", response_model=VideoGenerationResponse)
@handle_errors(
    category=ErrorCategory.VIDEO_GENERATION,
    operation="generate_video",
    retry_on_failure=True,
    max_retries=2,
    circuit_breaker=video_processing_circuit_breaker
)
async def generate_video_roro(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> VideoGenerationResponse:
    """Generate video using RORO pattern with comprehensive early validation"""
    
    # EARLY VALIDATION - Request context
    _validate_request_context(request, user_id)
    
    # EARLY VALIDATION - User permissions
    _validate_user_permissions(user_id, "video_generation")
    
    # EARLY VALIDATION - Rate limits
    await _validate_rate_limits(user_id, "video_generation")
    
    # EARLY VALIDATION - Input data types
    expected_types = {
        "script": str,
        "voice_id": str,
        "language": str,
        "quality": str,
        "duration": (int, type(None)),
        "custom_settings": (dict, type(None))
    }
    _validate_input_data_types(request_data, expected_types)
    
    # EARLY VALIDATION - RORO request format
    is_valid: bool
    roro_request: VideoGenerationRequest
    
    try:
        is_valid, roro_request = validate_roro_request(request_data, VideoGenerationRequest)
    except Exception as e:
        raise error_factory.validation_error(
            message="Invalid request format",
            context={"operation": "roro_validation", "user_id": user_id}
        )
    
    if not is_valid:
        raise error_factory.validation_error(
            message="Request validation failed",
            context={"operation": "request_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Video generation request
    try:
        validate_video_generation_request(roro_request.dict())
    except ValidationError as e:
        raise error_factory.validation_error(
            message=str(e),
            context={"operation": "video_generation_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Video ID format
    if not validate_video_id_format(roro_request.video_id):
        raise error_factory.invalid_video_id_error(
            roro_request.video_id,
            context={"operation": "video_id_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Script content
    is_valid, errors = validate_script_content(roro_request.script)
    if not is_valid:
        raise error_factory.validation_error(
            message="Script validation failed",
            validation_errors=errors,
            context={"operation": "script_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Voice ID
    is_valid, errors = validate_voice_id(roro_request.voice_id)
    if not is_valid:
        raise error_factory.validation_error(
            message="Voice ID validation failed",
            validation_errors=errors,
            context={"operation": "voice_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Language code
    is_valid, errors = validate_language_code(roro_request.language)
    if not is_valid:
        raise error_factory.validation_error(
            message="Language validation failed",
            validation_errors=errors,
            context={"operation": "language_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Quality settings
    is_valid, errors = validate_quality_settings(roro_request.quality, roro_request.duration)
    if not is_valid:
        raise error_factory.validation_error(
            message="Quality settings validation failed",
            validation_errors=errors,
            context={"operation": "quality_validation", "user_id": user_id}
        )
    
    # EARLY VALIDATION - Business logic constraints
    try:
        is_valid, errors = await validate_business_logic_constraints(
            user_id,
            "video_generation",
            {"quality": roro_request.quality, "duration": roro_request.duration}
        )
    except Exception as e:
        raise error_factory.validation_error(
            message="Business logic validation failed",
            context={"operation": "business_logic_validation", "user_id": user_id}
        )
    
    if not is_valid:
        raise error_factory.validation_error(
            message="Business logic constraints violated",
            validation_errors=errors,
            context={"operation": "business_logic_validation", "user_id": user_id}
        )
    
    # HAPPY PATH - Generate video ID
    video_id = generate_video_id(user_id)
    
    # HAPPY PATH - Calculate estimated duration
    estimated_duration = calculate_estimated_duration(roro_request.script, roro_request.quality)
    
    # HAPPY PATH - Prepare video data
    video_data = prepare_video_data_for_creation(roro_request, video_id, user_id, estimated_duration)
    
    # HAPPY PATH - Create video record
    try:
        await create_video_record(session, video_data)
    except Exception as e:
        raise error_factory.database_error(
            message="Failed to create video record",
            operation="create_video_record",
            context={"video_id": video_id, "user_id": user_id}
        )
    
    # HAPPY PATH - Add background task
    background_tasks.add_task(
        process_video_background,
        video_id,
        roro_request.dict(),
        user_id
    )
    
    # HAPPY PATH - Create response data
    response_data = create_generation_response_data(video_id, roro_request, estimated_duration)
    
    # HAPPY PATH - Return success response
    return create_success_response(response_data)
```

## Error Response Examples

### Video Generation Error Response
```json
{
  "error": {
    "id": "abc12345",
    "code": "VIDEO_GENERATION_ERROR",
    "message": "Video generation failed at template_rendering. Please try again.",
    "category": "video_generation",
    "severity": "high",
    "timestamp": "2024-01-01T12:00:00Z",
    "details": {
      "video_id": "vid123",
      "generation_stage": "template_rendering",
      "template_id": "tpl456"
    },
    "retry_after": null
  }
}
```

### Quota Exceeded Error Response
```json
{
  "error": {
    "id": "def67890",
    "code": "QUOTA_EXCEEDED_ERROR",
    "message": "Your video_generation quota has been exceeded. It will reset at 14:30.",
    "category": "quota_exceeded",
    "severity": "medium",
    "timestamp": "2024-01-01T12:00:00Z",
    "details": {
      "quota_type": "video_generation",
      "current_usage": 15,
      "limit": 10,
      "reset_time": "2024-01-01T14:30:00Z"
    },
    "retry_after": 9000
  }
}
```

### Content Moderation Error Response
```json
{
  "error": {
    "id": "ghi13579",
    "code": "CONTENT_MODERATION_ERROR",
    "message": "Your script content was flagged. Please review and modify it.",
    "category": "content_moderation",
    "severity": "high",
    "timestamp": "2024-01-01T12:00:00Z",
    "details": {
      "content_type": "script",
      "moderation_result": "violation",
      "flagged_content": "inappropriate language"
    },
    "retry_after": null
  }
}
```

## Best Practices

### 1. Error Type Selection
- **Use specific error types**: Choose the most specific error type for the situation
- **Avoid generic errors**: Don't use generic errors when specific ones exist
- **Consistent categorization**: Use consistent error categories across the application

### 2. Error Factory Usage
- **Use factory methods**: Always use error factory methods instead of direct instantiation
- **Include context**: Always include relevant context information
- **Use convenience methods**: Use convenience methods for common error scenarios

### 3. Error Handling
- **Early validation**: Validate inputs early and fail fast
- **Proper error propagation**: Let errors bubble up to the appropriate handler
- **User-friendly messages**: Ensure all errors have user-friendly messages

### 4. Logging and Monitoring
- **Automatic logging**: All errors are automatically logged with full context
- **Error tracking**: Use error IDs for tracking and debugging
- **Monitoring**: Monitor error patterns and frequencies

## Integration with Monitoring

### Error Metrics
```python
# Track error rates by category
error_counter = Counter('heygen_errors_total', 'Total errors', ['category', 'severity'])

# Track specific error types
video_generation_errors = Counter('video_generation_errors_total', 'Video generation errors')
quota_exceeded_errors = Counter('quota_exceeded_errors_total', 'Quota exceeded errors')
content_moderation_errors = Counter('content_moderation_errors_total', 'Content moderation errors')
```

### Error Alerting
```python
# Alert on high-severity errors
if error.severity == ErrorSeverity.CRITICAL:
    send_alert(f"Critical error: {error.error_code} - {error.message}")

# Alert on quota exceeded errors
if isinstance(error, QuotaExceededError):
    send_quota_alert(f"User {user_id} exceeded {error.details['quota_type']} quota")
```

## Conclusion

The custom error types and error factories provide:

1. **Type Safety**: Strongly typed errors with specific parameters
2. **Consistency**: Standardized error creation and handling
3. **Domain Specificity**: Errors tailored to specific business domains
4. **User Experience**: Automatic generation of user-friendly messages
5. **Developer Experience**: Easy-to-use factory methods and convenience functions
6. **Monitoring**: Comprehensive error tracking and alerting capabilities

This system ensures consistent, maintainable, and user-friendly error handling across the HeyGen AI application. 