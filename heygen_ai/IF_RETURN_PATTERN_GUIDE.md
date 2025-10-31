# 🚀 If-Return Pattern Implementation Guide

## Overview

This guide documents the **If-Return Pattern** implementation in the HeyGen AI FastAPI backend. This pattern avoids unnecessary else statements by using early returns, making the code cleaner, more readable, and reducing nesting complexity.

## 🎯 Key Principles

### 1. **Early Returns**
- Return early when conditions are met
- Avoid unnecessary else statements
- Reduce nesting and improve readability
- Make the main logic path more obvious

### 2. **Guard Clauses**
- Use guard clauses to handle edge cases first
- Return early for invalid conditions
- Keep the main logic path clean and unindented
- Improve code flow and maintainability

### 3. **Positive Conditions**
- Use positive conditions when possible
- Return early for success cases
- Avoid negative logic that requires else clauses
- Make the happy path more prominent

## 🔍 Implementation Patterns

### 1. **Resource Usage Validation**

#### Before (With Else Statement)
```python
def _check_resource_usage(resource_type: str, current_usage: float, limit: float) -> None:
    if current_usage > limit:
        raise error_factory.resource_exhaustion_error(
            message=f"{resource_type} usage exceeded limit",
            resource_type=resource_type,
            current_usage=current_usage,
            limit=limit
        )
    else:
        # Continue with normal processing
        pass
```

#### After (If-Return Pattern)
```python
def _check_resource_usage(resource_type: str, current_usage: float, limit: float) -> None:
    if current_usage <= limit:
        return
    
    raise error_factory.resource_exhaustion_error(
        message=f"{resource_type} usage exceeded limit",
        resource_type=resource_type,
        current_usage=current_usage,
        limit=limit
    )
```

### 2. **Type Validation**

#### Before (With Else Statement)
```python
def _validate_input_types(data: Dict[str, Any], expected_types: Dict[str, type]) -> None:
    for field, expected_type in expected_types.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                raise error_factory.validation_error(
                    message=f"Field '{field}' must be of type {expected_type.__name__}",
                    field=field,
                    value=data[field],
                    validation_errors=[f"Expected {expected_type.__name__}, got {type(data[field]).__name__}"]
                )
            else:
                # Field type is valid, continue
                continue
        else:
            # Field not present, skip validation
            continue
```

#### After (If-Return Pattern)
```python
def _validate_input_types(data: Dict[str, Any], expected_types: Dict[str, type]) -> None:
    for field, expected_type in expected_types.items():
        if field not in data:
            continue
        
        if isinstance(data[field], expected_type):
            continue
        
        raise error_factory.validation_error(
            message=f"Field '{field}' must be of type {expected_type.__name__}",
            field=field,
            value=data[field],
            validation_errors=[f"Expected {expected_type.__name__}, got {type(data[field]).__name__}"]
        )
```

### 3. **Required Fields Validation**

#### Before (With Else Statement)
```python
def _validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        else:
            if data[field] is None:
                missing_fields.append(field)
            else:
                if isinstance(data[field], str) and not data[field].strip():
                    missing_fields.append(field)
                else:
                    # Field is valid
                    pass
    
    if missing_fields:
        raise error_factory.validation_error(
            message=f"Missing required fields: {', '.join(missing_fields)}",
            validation_errors=[f"Required fields: {', '.join(missing_fields)}"],
            details={"missing_fields": missing_fields}
        )
    else:
        # All fields are present and valid
        pass
```

#### After (If-Return Pattern)
```python
def _validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
            continue
        
        if data[field] is None:
            missing_fields.append(field)
            continue
        
        if isinstance(data[field], str) and not data[field].strip():
            missing_fields.append(field)
            continue
    
    if not missing_fields:
        return
    
    raise error_factory.validation_error(
        message=f"Missing required fields: {', '.join(missing_fields)}",
        validation_errors=[f"Required fields: {', '.join(missing_fields)}"],
        details={"missing_fields": missing_fields}
    )
```

### 4. **String Safety Validation**

#### Before (With Else Statement)
```python
def _validate_string_safety(value: str, field_name: str) -> None:
    if isinstance(value, str):
        # Check for potential SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|\bOR\b|\bAND\b)",
            r"(\b(TRUE|FALSE|NULL)\b)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise error_factory.validation_error(
                    message=f"Field '{field_name}' contains potentially unsafe content",
                    field=field_name,
                    value="[REDACTED]",
                    validation_errors=["Content contains potentially unsafe patterns"]
                )
            else:
                # Pattern not found, continue checking
                continue
        
        # Check for script injection patterns
        script_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise error_factory.validation_error(
                    message=f"Field '{field_name}' contains potentially unsafe script content",
                    field=field_name,
                    value="[REDACTED]",
                    validation_errors=["Content contains potentially unsafe script patterns"]
                )
            else:
                # Pattern not found, continue checking
                continue
    else:
        # Not a string, skip validation
        pass
```

#### After (If-Return Pattern)
```python
def _validate_string_safety(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        return
    
    # Check for potential SQL injection patterns
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|\bOR\b|\bAND\b)",
        r"(\b(TRUE|FALSE|NULL)\b)",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise error_factory.validation_error(
                message=f"Field '{field_name}' contains potentially unsafe content",
                field=field_name,
                value="[REDACTED]",
                validation_errors=["Content contains potentially unsafe patterns"]
            )
    
    # Check for script injection patterns
    script_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise error_factory.validation_error(
                message=f"Field '{field_name}' contains potentially unsafe script content",
                field=field_name,
                value="[REDACTED]",
                validation_errors=["Content contains potentially unsafe script patterns"]
            )
```

## 🛡️ Enhanced Validation Functions

### 1. **Quality Settings Validation**
```python
def validate_quality_settings(quality: str, duration: Optional[int] = None) -> Tuple[bool, List[str]]:
    """Validate quality settings for video generation with early error handling"""
    errors = []
    
    # Early validation - check if quality is provided
    if not quality:
        errors.append("Quality level is required")
        return False, errors
    
    # Early validation - check if quality is string
    if not isinstance(quality, str):
        errors.append("Quality level must be a string")
        return False, errors
    
    # Valid quality levels
    valid_qualities = [QualityLevel.LOW.value, QualityLevel.MEDIUM.value, QualityLevel.HIGH.value]
    
    if quality not in valid_qualities:
        errors.append(f"Quality level '{quality}' is not valid. Valid options: {', '.join(valid_qualities)}")
        return False, errors
    
    # Check duration constraints based on quality
    if duration is None:
        return len(errors) == 0, errors
    
    if not isinstance(duration, int):
        errors.append("Duration must be an integer")
        return False, errors
    
    if quality == QualityLevel.LOW.value and duration > 120:
        errors.append("Low quality videos cannot exceed 120 seconds")
    
    if quality == QualityLevel.MEDIUM.value and duration > 300:
        errors.append("Medium quality videos cannot exceed 300 seconds")
    
    if quality == QualityLevel.HIGH.value and duration > 600:
        errors.append("High quality videos cannot exceed 600 seconds")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors
```

### 2. **Video Duration Validation**
```python
def validate_video_duration(duration: Optional[int]) -> Tuple[bool, List[str]]:
    """Validate video duration with early error handling"""
    errors = []
    
    if duration is None:
        return True, errors
    
    # Early validation - check if duration is integer
    if not isinstance(duration, int):
        errors.append("Duration must be an integer")
        return False, errors
    
    # Early validation - check if duration is positive
    if duration <= 0:
        errors.append("Duration must be positive")
        return False, errors
    
    if duration < 5:
        errors.append("Video duration must be at least 5 seconds")
    
    if duration > 600:
        errors.append("Video duration cannot exceed 600 seconds (10 minutes)")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors
```

### 3. **User ID Validation**
```python
def validate_user_id(user_id: str) -> Tuple[bool, List[str]]:
    """Validate user ID format with early error handling"""
    errors = []
    
    # Early validation - check if user_id is provided
    if not user_id:
        errors.append("User ID is required")
        return False, errors
    
    # Early validation - check if user_id is string
    if not isinstance(user_id, str):
        errors.append("User ID must be a string")
        return False, errors
    
    # Early validation - check length
    if len(user_id) < 3 or len(user_id) > 50:
        errors.append("User ID must be between 3 and 50 characters")
        return False, errors
    
    # User ID format validation (adjust based on your user ID format)
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        errors.append("User ID must contain only letters, numbers, underscores, and hyphens")
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors
```

## 🔄 Route Handler Patterns

### 1. **Request Context Validation**
```python
def _validate_request_context(request: Request, user_id: str) -> None:
    """Validate request context at the beginning of functions"""
    # Early validation - check if request has required headers
    if not request.headers.get("user-agent"):
        raise error_factory.validation_error(
            message="User-Agent header is required",
            field="user-agent",
            context={"operation": "request_validation", "user_id": user_id}
        )
    
    # Early validation - check request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
        raise error_factory.validation_error(
            message="Request payload too large",
            field="content-length",
            value=content_length,
            context={"operation": "request_validation", "user_id": user_id}
        )
```

### 2. **User Permissions Validation**
```python
def _validate_user_permissions(user_id: str, operation: str) -> None:
    """Validate user permissions at the beginning of functions"""
    # Early validation - check if user_id is valid
    if not user_id or not isinstance(user_id, str):
        raise error_factory.validation_error(
            message="Invalid user ID",
            field="user_id",
            value=user_id,
            context={"operation": operation}
        )
    
    # Early validation - check user_id format
    if len(user_id) < 3 or len(user_id) > 50:
        raise error_factory.validation_error(
            message="User ID length invalid",
            field="user_id",
            value=user_id,
            context={"operation": operation}
        )
```

### 3. **Rate Limit Validation**
```python
async def _validate_rate_limits(user_id: str, operation: str) -> None:
    """Validate rate limits at the beginning of functions"""
    # Early validation - check rate limits
    # In production, this would check against Redis or database
    # For now, we'll simulate rate limit checking
    current_time = time.time()
    rate_limit_key = f"rate_limit:{user_id}:{operation}"
    
    # Simulate rate limit check
    if operation == "video_generation":
        # Allow max 10 video generations per hour
        hourly_limit = 10
        # In production: check actual usage from cache/database
        pass
    
    if operation == "api_request":
        # Allow max 1000 API requests per minute
        minute_limit = 1000
        # In production: check actual usage from cache/database
        pass
```

## 📊 Pure Functions with If-Return Pattern

### 1. **Progress Calculation**
```python
def calculate_progress_from_status(status_result: Dict[str, Any]) -> float:
    """Calculate progress from status result with early validation"""
    # Early validation - check if status_result is dict
    if not isinstance(status_result, dict):
        raise error_factory.validation_error(
            message="Status result must be a dictionary",
            context={"operation": "calculate_progress"}
        )
    
    if status_result["is_completed"]:
        return 100.0
    
    if status_result["is_failed"]:
        return 0.0
    
    # Happy path - estimate progress based on processing time
    processing_time: float = status_result.get("processing_time", 0.0)
    estimated_duration: float = 60.0  # Default 60 seconds
    return calculate_progress(processing_time, estimated_duration)
```

### 2. **File Size Retrieval**
```python
def get_file_size_from_status(status_result: Dict[str, Any]) -> Optional[int]:
    """Get file size from status result with early validation"""
    # Early validation - check if status_result is dict
    if not isinstance(status_result, dict):
        raise error_factory.validation_error(
            message="Status result must be a dictionary",
            context={"operation": "get_file_size"}
        )
    
    if not status_result["has_output_file"]:
        return None
    
    # Happy path - get file size
    try:
        import os
        return os.path.getsize(status_result["file_path"])
    except Exception:
        return None
```

### 3. **Response Data Creation**
```python
def create_generation_response_data(
    video_id: str,
    request: VideoGenerationRequest,
    estimated_duration: int
) -> Dict[str, Any]:
    """Create generation response data with early validation"""
    # Early validation - check if all required parameters are provided
    if not video_id:
        raise error_factory.validation_error(
            message="Video ID is required",
            context={"operation": "create_generation_response"}
        )
    
    if not isinstance(estimated_duration, int) or estimated_duration <= 0:
        raise error_factory.validation_error(
            message="Estimated duration must be a positive integer",
            field="estimated_duration",
            value=estimated_duration,
            context={"operation": "create_generation_response"}
        )
    
    # Happy path - return response data
    return {
        "video_id": video_id,
        "status": VideoStatus.PROCESSING.value,
        "processing_time": 0.0,
        "estimated_completion": time.time() + estimated_duration,
        "metadata": {
            "script_length": len(request.script),
            "quality": request.quality,
            "voice_id": request.voice_id,
            "language": request.language
        }
    }
```

## 🚀 Benefits of If-Return Pattern

### 1. **Improved Readability**
- **Reduced Nesting**: Eliminates unnecessary else statements and reduces nesting
- **Clear Flow**: Makes the main logic path more obvious
- **Better Scanning**: Easier to scan and understand code structure
- **Simplified Logic**: Each condition is handled independently

### 2. **Enhanced Maintainability**
- **Modular Conditions**: Each condition can be modified independently
- **Easy Extension**: Adding new conditions is straightforward
- **Clear Dependencies**: The order of operations is obvious
- **Consistent Pattern**: Uniform approach across all functions

### 3. **Better Performance**
- **Early Exit**: Functions return as soon as conditions are met
- **Reduced Computation**: Avoid unnecessary processing for invalid inputs
- **Better Resource Usage**: Fail fast to conserve system resources
- **Improved Response Times**: Faster error responses to clients

### 4. **Reduced Complexity**
- **Lower Cognitive Load**: Fewer nested conditions to follow
- **Simpler Debugging**: Each condition is isolated and explicit
- **Better Testing**: Easier to test individual conditions
- **Clearer Intent**: Code intent is more obvious

## 📈 Code Quality Metrics

### Before (With Else Statements)
```python
# Cyclomatic Complexity: 8
# Nesting Depth: 4 levels
# Lines of Code: 35
# Cognitive Complexity: High
# Error Handling: Mixed with main logic
```

### After (If-Return Pattern)
```python
# Cyclomatic Complexity: 4
# Nesting Depth: 1 level
# Lines of Code: 25
# Cognitive Complexity: Low
# Error Handling: Separated and systematic
```

## 🔧 Best Practices

### 1. **Use Positive Conditions**
```python
# Good - Positive condition with early return
if current_usage <= limit:
    return

# Avoid - Negative condition requiring else
if current_usage > limit:
    raise error_factory.resource_exhaustion_error(...)
else:
    # Continue with logic
    pass
```

### 2. **Group Related Validations**
```python
# Good - Group related validations together
def validate_user_input(user_id: str, username: str, email: str):
    # User ID validations
    if not user_id:
        return False, ["User ID required"]
    
    if not isinstance(user_id, str):
        return False, ["User ID must be string"]
    
    # Username validations
    if not username:
        return False, ["Username required"]
    
    # Email validations
    if not email:
        return False, ["Email required"]
    
    return True, []
```

### 3. **Use Descriptive Variable Names**
```python
# Good - Clear variable names
missing_fields = []
if field not in data:
    missing_fields.append(field)
    continue

# Avoid - Unclear variable names
missing = []
if f not in d:
    missing.append(f)
    continue
```

### 4. **Provide Context in Errors**
```python
# Good - Rich error context
raise error_factory.validation_error(
    message="Invalid user ID",
    field="user_id",
    value=user_id,
    context={"operation": "user_validation"}
)

# Avoid - Minimal error information
raise error_factory.validation_error(
    message="Invalid input"
)
```

### 5. **Use Continue for Loops**
```python
# Good - Use continue for loop iterations
for field in required_fields:
    if field not in data:
        missing_fields.append(field)
        continue
    
    if data[field] is None:
        missing_fields.append(field)
        continue
    
    # Process valid field
    process_field(data[field])

# Avoid - Nested if-else in loops
for field in required_fields:
    if field in data:
        if data[field] is not None:
            process_field(data[field])
        else:
            missing_fields.append(field)
    else:
        missing_fields.append(field)
```

## 📖 Common Patterns

### 1. **Validation Functions**
```python
def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    
    # Early validation - check if data is provided
    if not data:
        errors.append("Data is required")
        return False, errors
    
    # Early validation - check if data is dict
    if not isinstance(data, dict):
        errors.append("Data must be a dictionary")
        return False, errors
    
    # Validate required fields
    required_fields = ["id", "name", "email"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Required field '{field}' is missing")
            continue
        
        if data[field] is None:
            errors.append(f"Field '{field}' cannot be null")
            continue
    
    # Happy path - return success if no errors found
    return len(errors) == 0, errors
```

### 2. **Processing Functions**
```python
def process_data(data: List[str]) -> List[str]:
    # Early validation - check if data is provided
    if not data:
        return []
    
    # Early validation - check if data is list
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    
    # Process data
    processed = []
    for item in data:
        if not item:
            continue
        
        processed_item = item.upper().strip()
        if processed_item:
            processed.append(processed_item)
    
    # Happy path - return processed data
    return processed
```

### 3. **Route Handlers**
```python
@router.post("/process")
async def process_request(request_data: Dict[str, Any], user_id: str):
    # EARLY VALIDATION - Request context
    if not request_data:
        raise ValidationError("Request data required")
    
    # EARLY VALIDATION - User permissions
    if not user_id:
        raise ValidationError("User ID required")
    
    # EARLY VALIDATION - Business rules
    if not is_user_authorized(user_id):
        raise AuthorizationError("User not authorized")
    
    # Process request
    result = await process_user_request(request_data, user_id)
    
    # Happy path - return success response
    return create_success_response(result)
```

## 📖 Conclusion

The **If-Return Pattern** provides significant benefits for code quality and maintainability:

- **Improved Readability**: Eliminates unnecessary else statements and reduces nesting
- **Enhanced Maintainability**: Modular conditions that can be modified independently
- **Better Performance**: Early exits and reduced computation
- **Reduced Complexity**: Lower cognitive load and clearer intent
- **Easier Testing**: Isolated conditions that can be tested independently

This pattern is particularly valuable in validation-heavy applications like the HeyGen AI API, where input validation and error handling are critical for security and reliability. The implementation demonstrates how to transform complex nested logic into clean, readable, and maintainable code using early returns instead of unnecessary else statements. 