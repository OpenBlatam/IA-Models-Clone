# Gradio Error Handling and Input Validation Guide

## Overview

This guide covers the comprehensive error handling and input validation system for Gradio applications in the AI Video system. The system provides robust error management, user-friendly error messages, and proper input validation with error categorization.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Error Categories](#error-categories)
4. [Input Validation](#input-validation)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Integration Guide](#integration-guide)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

## Architecture

The Gradio error handling system follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Interface                         │
├─────────────────────────────────────────────────────────────┤
│                Error Display Components                     │
├─────────────────────────────────────────────────────────────┤
│                GradioErrorHandler                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Error Categorization │ User Messages │ Error Storage │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│              GradioInputValidator                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Text Validation │ Numeric Validation│ File Validation│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Decorators                              │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ @gradio_error_handler │ @gradio_input_validator │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. GradioErrorHandler

The main error handling class that provides:

- **Error Categorization**: Automatically categorizes errors based on content
- **User-Friendly Messages**: Converts technical errors to user-readable messages
- **Error Storage**: Maintains error history and statistics
- **Retry Logic**: Determines if errors can be retried

```python
from .gradio_error_handling import GradioErrorHandler

error_handler = GradioErrorHandler()

# Categorize an error
category, severity = error_handler.categorize_error(exception)

# Get user-friendly message
user_message = error_handler.get_user_friendly_message(exception, category)

# Format for Gradio display
title, description, details = error_handler.format_error_for_gradio(exception)
```

### 2. GradioInputValidator

Handles input validation with configurable rules:

```python
from .gradio_error_handling import GradioInputValidator, InputValidationRule

validator = GradioInputValidator(error_handler)

# Define custom validation rules
custom_rules = {
    "prompt_text": InputValidationRule(
        field_name="prompt_text",
        required=True,
        min_length=20,
        max_length=1000,
        custom_validator=validate_prompt_content
    )
}

# Validate inputs
validator.validate_text_input(text, "prompt_text")
validator.validate_numeric_input(value, "duration")
validator.validate_file_input(file_path, "uploaded_file")
```

### 3. Error Categories

The system categorizes errors into the following types:

| Category | Description | Severity | Retryable |
|----------|-------------|----------|-----------|
| `INPUT_VALIDATION` | Invalid user input | Medium | No |
| `FILE_PROCESSING` | File format/size issues | Medium | Yes |
| `MODEL_LOADING` | AI model loading failures | High | Yes |
| `GPU_MEMORY` | GPU memory issues | High | Yes |
| `NETWORK` | Network connectivity issues | Medium | Yes |
| `CONFIGURATION` | System configuration errors | Critical | No |
| `PERMISSION` | Access permission issues | High | No |
| `TIMEOUT` | Operation timeouts | Medium | Yes |
| `UNKNOWN` | Unclassified errors | Medium | Yes |

### 4. Error Severity Levels

- **LOW**: Minor issues, non-critical
- **MEDIUM**: Moderate issues, may affect functionality
- **HIGH**: Serious issues, affects core functionality
- **CRITICAL**: System-breaking issues

## Input Validation

### Validation Rules

The system supports comprehensive validation rules:

```python
InputValidationRule(
    field_name="example_field",
    required=True,
    min_length=10,
    max_length=1000,
    min_value=1,
    max_value=100,
    allowed_values=["option1", "option2"],
    file_types=["mp4", "avi"],
    max_file_size=100 * 1024 * 1024,  # 100MB
    custom_validator=custom_validation_function
)
```

### Built-in Validators

1. **Text Validation**:
   - Length constraints
   - Required field checking
   - Custom content validation

2. **Numeric Validation**:
   - Range validation
   - Allowed values
   - Type checking

3. **File Validation**:
   - File existence
   - File type checking
   - File size limits

### Custom Validators

```python
def validate_prompt_content(text: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Custom validator for prompt content."""
    # Check for inappropriate content
    inappropriate_words = ["inappropriate", "offensive", "harmful"]
    text_lower = text.lower()
    
    for word in inappropriate_words:
        if word in text_lower:
            raise ValidationError(f"Prompt contains inappropriate content: {word}")
    
    # Check for minimum meaningful content
    if len(text.split()) < 3:
        raise ValidationError("Prompt must contain at least 3 words")
```

## Usage Examples

### 1. Basic Error Handling

```python
from .gradio_error_handling import gradio_error_handler

@gradio_error_handler(show_technical=False, log_errors=True)
def generate_video(prompt: str, quality: str, duration: int):
    # Your video generation logic here
    if not prompt:
        raise ValidationError("Prompt is required")
    
    # Process video generation
    return {"video_path": "path/to/video.mp4", "status": "success"}
```

### 2. Input Validation

```python
from .gradio_error_handling import gradio_input_validator, InputValidationRule

@gradio_input_validator({
    "prompt": InputValidationRule(
        field_name="prompt",
        required=True,
        min_length=10,
        max_length=500
    ),
    "quality": InputValidationRule(
        field_name="quality",
        required=True,
        allowed_values=["low", "medium", "high", "ultra"]
    )
})
def process_video_request(prompt: str, quality: str):
    # Input is already validated
    return {"status": "processing", "prompt": prompt, "quality": quality}
```

### 3. Complete Gradio Application

```python
import gradio as gr
from .gradio_error_handling import (
    GradioErrorHandler, GradioInputValidator,
    create_gradio_error_components, update_error_display
)

class AIVideoGradioApp:
    def __init__(self):
        self.error_handler = GradioErrorHandler()
        self.validator = GradioInputValidator(self.error_handler)
    
    def create_interface(self):
        with gr.Blocks() as interface:
            # Create error display components
            error_title, error_description, error_details = create_gradio_error_components()
            
            # Input components
            prompt = gr.Textbox(label="Video Prompt")
            quality = gr.Dropdown(choices=["low", "medium", "high"], value="medium")
            generate_btn = gr.Button("Generate Video")
            
            # Output components
            output_video = gr.Video(label="Generated Video")
            success_message = gr.HTML(visible=False)
            
            # Event handler
            def generate_video(prompt_text, quality_setting):
                try:
                    # Validate inputs
                    self.validator.validate_text_input(prompt_text, "prompt")
                    self.validator.validate_numeric_input(quality_setting, "quality")
                    
                    # Generate video (your logic here)
                    result = {"video_path": "path/to/video.mp4", "status": "success"}
                    
                    return result, "✅ Video generated successfully!", False, False, False
                    
                except Exception as e:
                    # Handle error
                    title, description, details = self.error_handler.format_error_for_gradio(e)
                    return None, "", True, True, True
            
            generate_btn.click(
                fn=generate_video,
                inputs=[prompt, quality],
                outputs=[output_video, success_message, error_title, error_description, error_details]
            )
        
        return interface
```

### 4. Error Display Components

```python
def create_error_display():
    """Create error display components for Gradio."""
    error_title = gr.HTML(
        value="",
        elem_classes=["error-title"],
        visible=False
    )
    
    error_description = gr.HTML(
        value="",
        elem_classes=["error-description"],
        visible=False
    )
    
    error_details = gr.HTML(
        value="",
        elem_classes=["error-details"],
        visible=False
    )
    
    return error_title, error_description, error_details

def update_error_display(result, error_components):
    """Update error display based on result."""
    if result.get("error", False):
        # Show error components
        return (
            gr.HTML(value=result["title"], visible=True),
            gr.HTML(value=result["description"], visible=True),
            gr.HTML(value=result["details"], visible=True),
            False  # Hide success message
        )
    else:
        # Hide error components
        return (
            gr.HTML(value="", visible=False),
            gr.HTML(value="", visible=False),
            gr.HTML(value="", visible=False),
            True  # Show success message
        )
```

## Best Practices

### 1. Error Handling

- **Use decorators** for consistent error handling across functions
- **Categorize errors** properly for better user experience
- **Provide helpful suggestions** for error resolution
- **Log errors** for debugging and monitoring
- **Avoid showing technical details** to end users

### 2. Input Validation

- **Validate early** in the function execution
- **Use specific validation rules** for each field
- **Provide clear error messages** for validation failures
- **Implement custom validators** for business logic
- **Handle optional fields** appropriately

### 3. User Experience

- **Show user-friendly messages** instead of technical errors
- **Provide actionable suggestions** for error resolution
- **Use appropriate icons** and styling for error display
- **Maintain consistent error formatting** across the application
- **Allow retry for appropriate errors**

### 4. Performance

- **Cache validation rules** where possible
- **Limit error history size** to prevent memory issues
- **Use async validation** for heavy operations
- **Batch validation** for multiple inputs when possible

## Integration Guide

### 1. With Existing Gradio Apps

```python
# Add to existing Gradio app
from .gradio_error_handling import gradio_error_handler, gradio_input_validator

# Wrap existing functions
@gradio_error_handler()
@gradio_input_validator()
def your_existing_function(input1, input2):
    # Your existing logic
    pass
```

### 2. With AI Video System

```python
from .main import get_system
from .gradio_error_handling import GradioErrorHandler

class IntegratedGradioApp:
    def __init__(self):
        self.error_handler = GradioErrorHandler()
        self.system = None
    
    async def initialize(self):
        self.system = await get_system()
    
    @gradio_error_handler()
    async def generate_video(self, prompt: str, quality: str):
        # Validate inputs
        if not prompt or len(prompt) < 10:
            raise ValidationError("Prompt must be at least 10 characters")
        
        # Use AI Video system
        request = VideoRequest(
            input_text=prompt,
            quality=quality,
            duration=30,
            output_format="mp4"
        )
        
        response = await self.system.generate_video(request)
        return response
```

### 3. With Monitoring Systems

```python
# Integrate with monitoring
def log_error_for_monitoring(error_info: GradioErrorInfo):
    """Log error for external monitoring systems."""
    monitoring_data = {
        "error_id": error_info.error_id,
        "category": error_info.category.value,
        "severity": error_info.severity.value,
        "timestamp": error_info.timestamp.isoformat(),
        "user_message": error_info.user_message,
        "technical_message": error_info.technical_message
    }
    
    # Send to monitoring system
    monitoring_client.send_event("gradio_error", monitoring_data)
```

## Testing

### 1. Unit Tests

```python
import pytest
from .gradio_error_handling import GradioErrorHandler, GradioInputValidator

def test_error_categorization():
    error_handler = GradioErrorHandler()
    
    # Test GPU memory error
    gpu_error = Exception("CUDA out of memory")
    category, severity = error_handler.categorize_error(gpu_error)
    
    assert category == ErrorCategory.GPU_MEMORY
    assert severity == ErrorSeverity.HIGH

def test_input_validation():
    validator = GradioInputValidator(GradioErrorHandler())
    
    # Test valid input
    validator.validate_text_input("Valid input text", "input_text")
    
    # Test invalid input
    with pytest.raises(ValidationError):
        validator.validate_text_input("Short", "input_text")
```

### 2. Integration Tests

```python
async def test_complete_flow():
    """Test complete error handling flow."""
    app = AIVideoGradioApp()
    await app.initialize()
    
    # Test successful generation
    result = await app.generate_video(
        prompt="Valid video prompt",
        quality="medium",
        duration=30
    )
    
    assert result["success"] is True
    
    # Test error handling
    result = await app.generate_video(
        prompt="Short",
        quality="invalid",
        duration=3
    )
    
    assert result["error"] is True
    assert "Invalid Input" in result["title"]
```

### 3. Performance Tests

```python
def test_error_handler_performance():
    """Test error handler performance under load."""
    error_handler = GradioErrorHandler()
    
    # Generate many errors
    for i in range(1000):
        error = ValidationError(f"Test error {i}")
        error_info = error_handler.create_error_info(
            error=error,
            category=ErrorCategory.INPUT_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            user_message=f"User message {i}",
            technical_message=f"Technical message {i}"
        )
        error_handler._store_error(error_info)
    
    # Verify performance
    assert len(error_handler.error_history) <= 1000  # Max history size
    stats = error_handler.get_error_statistics()
    assert stats["total_errors"] == 1000
```

## Troubleshooting

### Common Issues

1. **Error not categorized correctly**:
   - Check error message content
   - Add custom categorization logic
   - Review error category keywords

2. **Validation rules not working**:
   - Verify rule configuration
   - Check field name matching
   - Ensure custom validators are callable

3. **Error display not showing**:
   - Check Gradio component visibility
   - Verify error formatting
   - Ensure proper event binding

4. **Performance issues**:
   - Monitor error history size
   - Check validation rule complexity
   - Review async operations

### Debug Mode

Enable debug mode for detailed error information:

```python
@gradio_error_handler(show_technical=True, log_errors=True)
def debug_function():
    # Function with detailed error reporting
    pass
```

### Error Statistics

Monitor error patterns using statistics:

```python
error_handler = GradioErrorHandler()
stats = error_handler.get_error_statistics()

print(f"Total errors: {stats['total_errors']}")
print(f"Error counts: {stats['error_counts']}")
print(f"Recent errors: {stats['recent_errors']}")
```

## Conclusion

The Gradio error handling and input validation system provides a robust foundation for building user-friendly AI applications. By following the patterns and best practices outlined in this guide, you can create applications that handle errors gracefully and provide excellent user experience.

For additional support or questions, refer to the test suite and example applications in the codebase. 