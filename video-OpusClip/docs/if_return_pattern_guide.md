# If-Return Pattern Guide

## Overview

This guide documents the **if-return pattern** - a coding practice that avoids unnecessary else statements by using early returns for error conditions and edge cases. This pattern improves code readability, reduces nesting, and makes functions easier to understand and maintain.

## Table of Contents

1. [Principles](#principles)
2. [Patterns](#patterns)
3. [Implementation Examples](#implementation-examples)
4. [Benefits](#benefits)
5. [Best Practices](#best-practices)
6. [Migration Guide](#migration-guide)
7. [Testing Strategy](#testing-strategy)
8. [Code Quality Metrics](#code-quality-metrics)

## Principles

### 1. Fail Fast, Return Early

- **Early validation**: Check error conditions first
- **Immediate returns**: Return or raise exceptions as soon as errors are detected
- **No unnecessary nesting**: Avoid deeply nested if-else structures

### 2. Clear Flow Control

- **Linear execution**: Main logic flows in a straight line
- **Error handling first**: All error conditions handled at the beginning
- **Happy path last**: Success case is the final, unindented code

### 3. Reduced Complexity

- **Lower cyclomatic complexity**: Fewer decision points
- **Easier testing**: Each path is clearly defined
- **Better maintainability**: Changes don't affect deeply nested code

## Patterns

### Basic If-Return Pattern

```python
def function_with_if_return(param1, param2):
    """Function that uses if-return pattern."""
    # ERROR HANDLING: Validate input parameters
    if not param1:
        raise ValidationError("param1 is required")
    
    if not isinstance(param2, str):
        raise ValidationError("param2 must be a string")
    
    if len(param2) > 1000:
        raise ValidationError("param2 too long")
    
    # HAPPY PATH: Main logic
    result = process_data(param1, param2)
    return result
```

### Validation Function Pattern

```python
def validate_field(value, field_name):
    """Validate field using if-return pattern."""
    # ERROR HANDLING: Check for None/empty
    if not value:
        raise ValidationError(f"{field_name} is required")
    
    # ERROR HANDLING: Check data type
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    
    # ERROR HANDLING: Check length
    if len(value) > 100:
        raise ValidationError(f"{field_name} too long")
    
    # HAPPY PATH: Field is valid (implicit return)
```

### Error Analysis Pattern

```python
def analyze_error_pattern(error: Exception) -> str:
    """Analyze error patterns using if-return pattern."""
    error_str = str(error).lower()
    
    if "memory" in error_str or "out of memory" in error_str:
        return "memory_related"
    
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout_related"
    
    if "connection" in error_str or "network" in error_str:
        return "network_related"
    
    if "permission" in error_str or "access" in error_str:
        return "permission_related"
    
    return "unknown_pattern"
```

## Implementation Examples

### 1. Before (Unnecessary Else)

```python
def validate_url(url):
    if not url:
        raise ValidationError("URL required")
    else:
        if len(url) > 2048:
            raise ValidationError("URL too long")
        else:
            if is_valid_url(url):
                return True
            else:
                raise ValidationError("Invalid URL")
```

### 2. After (If-Return Pattern)

```python
def validate_url(url):
    # ERROR HANDLING: Empty or None URL
    if not url:
        raise ValidationError("URL required")
    
    # ERROR HANDLING: Too long
    if len(url) > 2048:
        raise ValidationError("URL too long")
    
    # ERROR HANDLING: Invalid format
    if not is_valid_url(url):
        raise ValidationError("Invalid URL")
    
    # HAPPY PATH: URL is valid (implicit return)
```

### 3. Before (Nested If-Else)

```python
def process_data(data):
    if data is not None:
        if isinstance(data, dict):
            if 'required_field' in data:
                if data['required_field']:
                    result = process_valid_data(data)
                    return result
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None
```

### 4. After (If-Return Pattern)

```python
def process_data(data):
    # ERROR HANDLING: Check for None
    if data is None:
        return None
    
    # ERROR HANDLING: Check data type
    if not isinstance(data, dict):
        return None
    
    # ERROR HANDLING: Check required field
    if 'required_field' not in data:
        return None
    
    # ERROR HANDLING: Check field value
    if not data['required_field']:
        return None
    
    # HAPPY PATH: Process valid data
    result = process_valid_data(data)
    return result
```

### 5. Before (Complex Conditional)

```python
def analyze_error(error):
    if "memory" in str(error):
        return "memory_related"
    elif "timeout" in str(error):
        return "timeout_related"
    elif "connection" in str(error):
        return "network_related"
    elif "permission" in str(error):
        return "permission_related"
    else:
        return "unknown_pattern"
```

### 6. After (If-Return Pattern)

```python
def analyze_error(error):
    error_str = str(error).lower()
    
    if "memory" in error_str:
        return "memory_related"
    
    if "timeout" in error_str:
        return "timeout_related"
    
    if "connection" in error_str:
        return "network_related"
    
    if "permission" in error_str:
        return "permission_related"
    
    return "unknown_pattern"
```

## Benefits

### 1. Improved Readability

- **Clear flow**: Easy to follow the execution path
- **Reduced nesting**: No deeply nested conditions
- **Linear structure**: Code reads from top to bottom

### 2. Better Maintainability

- **Isolated changes**: Modifying error handling doesn't affect main logic
- **Easier debugging**: Error conditions are clearly separated
- **Reduced complexity**: Lower cyclomatic complexity

### 3. Enhanced Performance

- **Fail fast**: Errors are caught early, avoiding unnecessary processing
- **Optimized flow**: Main logic is the most common path
- **Reduced overhead**: No unnecessary else statement evaluation

### 4. Superior Testing

- **Clear test cases**: Each path is clearly defined
- **Easier mocking**: Error conditions can be easily isolated
- **Better coverage**: Test structure mirrors code structure

## Best Practices

### 1. Error Handling Order

1. **Input validation**: Check for None, empty, wrong types
2. **Business rules**: Check length, format, constraints
3. **System state**: Check resources, availability
4. **Security**: Check for malicious input
5. **Dependencies**: Check external services

### 2. Comment Structure

```python
# ERROR HANDLING: [Description of what's being checked]
if condition:
    raise ErrorType("Error message")

# HAPPY PATH: [Description of main logic]
main_logic()
```

### 3. Return Types

```python
# For validation functions
def validate_field(value):
    if not value:
        raise ValidationError("Value required")
    # Implicit return (None) for success

# For processing functions
def process_data(data):
    if not data:
        return None  # Early return for error
    # Process data and return result
    return result
```

### 4. Exception Handling

```python
# ERROR HANDLING: Use specific exceptions
if not value:
    raise ValidationError("Value required")

# HAPPY PATH: Handle exceptions in main logic
try:
    result = process_data(value)
    return result
except SpecificError as e:
    logger.error("Specific error", error=str(e))
    raise
```

### 5. Logging Strategy

```python
# ERROR HANDLING: Log warnings for recoverable errors
logger.warning("Validation failed", field=field_name)

# HAPPY PATH: Log info for successful operations
logger.info("Operation completed successfully", result=result)
```

## Migration Guide

### Step 1: Identify Unnecessary Else Statements

Find functions with unnecessary else statements:

```python
# Look for patterns like this
def mixed_function(param):
    if not param:
        return error_result
    else:
        result = process(param)
        if result:
            return result
        else:
            return None
```

### Step 2: Extract Error Conditions

Move all error conditions to the beginning:

```python
def refactored_function(param):
    # ERROR HANDLING: All error checks first
    if not param:
        return error_result
    
    if not is_valid(param):
        return None
    
    # HAPPY PATH: Main logic last
    result = process(param)
    return result
```

### Step 3: Add Comments

Add clear section comments:

```python
def final_function(param):
    # ERROR HANDLING: Input validation
    if not param:
        raise ValidationError("Parameter required")
    
    # ERROR HANDLING: Business rules
    if not is_valid(param):
        raise ValidationError("Invalid parameter")
    
    # HAPPY PATH: Process and return
    result = process_valid_data(param)
    return result
```

### Step 4: Update Tests

Update tests to reflect new structure:

```python
def test_function_structure():
    """Test that function follows if-return pattern."""
    source = inspect.getsource(function)
    assert "# ERROR HANDLING:" in source
    assert "# HAPPY PATH:" in source
    
    # Verify no unnecessary else statements
    lines = source.split('\n')
    else_count = sum(1 for line in lines if 'else:' in line)
    try_except_count = sum(1 for line in lines if line.strip().startswith(('try:', 'except')))
    
    # Else statements should only be in try-except blocks
    assert else_count <= try_except_count
```

## Testing Strategy

### 1. Structure Tests

```python
def test_if_return_pattern():
    """Test that functions use if-return pattern."""
    functions = [func1, func2, func3]
    
    for func in functions:
        source = inspect.getsource(func)
        lines = source.split('\n')
        
        # Count else statements
        else_count = sum(1 for line in lines if 'else:' in line)
        
        # Count try-except blocks (where else might be necessary)
        try_except_count = sum(1 for line in lines 
                              if line.strip().startswith(('try:', 'except')))
        
        # Else statements should only be in try-except blocks
        assert else_count <= try_except_count
```

### 2. Functionality Tests

```python
def test_if_return_works():
    """Test that if-return pattern actually works."""
    # Test that validation functions raise exceptions immediately
    with pytest.raises(ValidationError):
        validate_field("")
    
    # Test that valid inputs don't raise exceptions
    validate_field("valid_value")
```

### 3. Performance Tests

```python
def test_if_return_performance():
    """Test that if-return pattern is fast."""
    import time
    
    # Test that validation functions fail fast
    start = time.time()
    for _ in range(1000):
        try:
            validate_field("")
        except ValidationError:
            pass
    validation_time = time.time() - start
    
    # Should be fast
    assert validation_time < 0.1
```

### 4. Integration Tests

```python
def test_api_endpoint_if_return():
    """Test that API endpoints follow if-return pattern."""
    with patch('system.validate_health'):
        response = client.post("/api/endpoint", json=valid_data)
        assert response.status_code == 200
```

## Code Quality Metrics

### 1. Structure Metrics

- **Else statement count**: Should be minimal (only in try-except blocks)
- **Nesting depth**: Should be reduced
- **Cyclomatic complexity**: Should be lower

### 2. Readability Metrics

- **Linear flow**: Code should read from top to bottom
- **Clear separation**: Error handling and happy path should be distinct
- **Comment coverage**: All sections should be clearly commented

### 3. Performance Metrics

- **Early return speed**: Error conditions should be fast
- **Main logic speed**: Happy path should be optimized
- **Memory usage**: Should be consistent between error and success paths

### 4. Maintainability Metrics

- **Change isolation**: Modifying error handling shouldn't affect happy path
- **Test coverage**: Both error and happy paths should be well tested
- **Documentation**: Structure should be self-documenting

## Examples from Video-OpusClip

### Error Analysis Function

```python
def _analyze_error_pattern(self, error: Exception) -> str:
    """Analyze error patterns to help with debugging."""
    error_str = str(error).lower()
    
    if "memory" in error_str or "out of memory" in error_str:
        return "memory_related"
    
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout_related"
    
    if "connection" in error_str or "network" in error_str:
        return "network_related"
    
    if "permission" in error_str or "access" in error_str:
        return "permission_related"
    
    return "unknown_pattern"
```

### Validation Functions

```python
def validate_youtube_url(url: str, field_name: str = "youtube_url") -> None:
    """Validate YouTube URL and raise ValidationError if invalid."""
    # ERROR HANDLING: Empty or None URL
    if not url:
        raise create_validation_error("YouTube URL is required", field_name, url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Wrong data type
    if not isinstance(url, str):
        raise create_validation_error("YouTube URL must be a string", field_name, url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Extremely long URL (potential DoS)
    if len(url) > 2048:
        raise create_validation_error("YouTube URL too long (max 2048 characters)", field_name, url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Malicious patterns
    malicious_patterns = ["javascript:", "data:", "vbscript:", "file://", "ftp://"]
    url_lower = url.lower()
    for pattern in malicious_patterns:
        if pattern in url_lower:
            raise create_validation_error(f"Malicious URL pattern detected: {pattern}", field_name, url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Standard validation
    if not is_valid_youtube_url(url):
        raise create_validation_error(f"Invalid YouTube URL format: {url}", field_name, url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # HAPPY PATH: URL is valid (implicit return)
```

### API Endpoints

```python
@app.post("/api/v1/video/process", response_model=VideoClipResponse)
@handle_processing_errors
async def process_video(request: VideoClipRequest, processor: VideoProcessor = Depends(get_video_processor), req: Request = None):
    """Process a single video clip with enhanced error handling and system monitoring."""
    # ERROR HANDLING: Extract request ID first
    request_id = getattr(req.state, 'request_id', None) if req else None
    
    # ERROR HANDLING: Validate request object - early return
    if not request:
        raise ValidationError("Request object is required", "request", None, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Check for None/empty YouTube URL - early return
    if not request.youtube_url or not request.youtube_url.strip():
        raise ValidationError("YouTube URL is required and cannot be empty", "youtube_url", request.youtube_url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: System health validation (critical) - early return
    try:
        validate_system_health()
    except CriticalSystemError as e:
        logger.critical("System health validation failed during video processing", error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: Security validation (high priority) - early return
    malicious_patterns = ["javascript:", "data:", "vbscript:", "file://", "ftp://", "eval(", "exec(", "system("]
    if any(pattern in request.youtube_url.lower() for pattern in malicious_patterns):
        logger.warning("Malicious input detected in YouTube URL", url=request.youtube_url, request_id=request_id)
        raise SecurityError("Malicious input detected in YouTube URL", "malicious_input", {"url": request.youtube_url})
    
    # ERROR HANDLING: Processor validation - early return
    if not processor:
        raise ConfigurationError("Video processor is not available", "video_processor", ErrorCode.MISSING_CONFIG)
    
    # Start processing timer after all validations
    start_time = time.perf_counter()
    
    # HAPPY PATH: Process video and return response
    try:
        # Processing with monitoring
        response = processor.process_video(request)
        
        processing_time = time.perf_counter() - start_time
        
        logger.info("Video processing completed", youtube_url=request.youtube_url, processing_time=processing_time, success=response.success, request_id=getattr(req.state, 'request_id', None) if req else None)
        
        return response
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error("Video processing failed", error=str(e), processing_time=processing_time, request_id=getattr(req.state, 'request_id', None) if req else None)
        raise ProcessingError(f"Video processing failed: {str(e)}", "process_video")
```

### Gradio Demo Functions

```python
def generate_image(prompt: str, guidance_scale: float = 7.5, num_inference_steps: int = 30) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Generate image with comprehensive error handling and validation."""
    # ERROR HANDLING: Check for None/empty prompt first
    if prompt is None:
        logger.warning("Prompt is None")
        return None, "Prompt cannot be None"
    
    if not prompt or not prompt.strip():
        logger.warning("Prompt is empty")
        return None, "Prompt cannot be empty"
    
    # ERROR HANDLING: Validate data types first
    if not isinstance(prompt, str):
        logger.warning(f"Prompt must be a string, got {type(prompt)}")
        return None, f"Prompt must be a string, got {type(prompt)}"
    
    # ERROR HANDLING: Check if pipeline is available first
    if pipe is None:
        logger.error("Image generation pipeline not available")
        return None, "Image generation pipeline not available. Please check system configuration."
    
    # HAPPY PATH: Generate image and return result
    try:
        # Generate image
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            result = pipe(prompt=prompt.strip(), guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
            
            # ERROR HANDLING: Check result immediately
            if not result or not hasattr(result, 'images'):
                logger.error("Invalid result from pipeline")
                return None, "Invalid result from image generation pipeline"
            
            image = result.images[0]
            
            # ERROR HANDLING: Validate generated image immediately
            if image is None:
                logger.error("Generated image is None")
                return None, "Generated image is None"
        
        logger.info(f"Image generated successfully for prompt: {prompt[:50]}...")
        return image, None
        
    except Exception as e:
        logger.error(f"Unexpected error during image generation: {e}")
        return None, f"Unexpected error: {str(e)}"
```

## Conclusion

The if-return pattern is a powerful technique that improves code readability, maintainability, and performance. By avoiding unnecessary else statements and using early returns for error conditions, developers can create code that is easier to understand, debug, and maintain.

The key principles are:
1. **Fail fast**: Handle all error conditions first
2. **Return early**: Use early returns for error conditions
3. **Clear separation**: Use comments to clearly mark sections
4. **Linear flow**: Keep the main logic in a straight line

This approach leads to more robust, maintainable, and user-friendly code that follows the principle of "optimize for the common case" while ensuring that error conditions are handled comprehensively and clearly. 