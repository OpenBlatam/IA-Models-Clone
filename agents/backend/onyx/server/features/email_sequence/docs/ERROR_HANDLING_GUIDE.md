# Error Handling and Validation System Guide

## Overview

The Email Sequence AI System includes a comprehensive error handling and validation system designed to provide robust, user-friendly error management for data loading, model inference, and other error-prone operations. This system ensures graceful degradation and clear error reporting throughout the application.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Error Types and Handling](#error-types-and-handling)
4. [Input Validation](#input-validation)
5. [Data Loading Error Handling](#data-loading-error-handling)
6. [Model Inference Error Handling](#model-inference-error-handling)
7. [Gradio Integration](#gradio-integration)
8. [Best Practices](#best-practices)
9. [Examples and Usage](#examples-and-usage)
10. [Testing and Debugging](#testing-and-debugging)

## System Architecture

The error handling system is built with a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Interface Layer                   │
├─────────────────────────────────────────────────────────────┤
│                 GradioErrorHandler                          │
├─────────────────────────────────────────────────────────────┤
│              ModelInferenceErrorHandler                     │
├─────────────────────────────────────────────────────────────┤
│                DataLoaderErrorHandler                       │
├─────────────────────────────────────────────────────────────┤
│                   InputValidator                            │
├─────────────────────────────────────────────────────────────┤
│                    ErrorHandler                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Comprehensive Error Logging**: All errors are logged with context, timestamps, and stack traces
- **Graceful Degradation**: System continues operation even when errors occur
- **User-Friendly Messages**: Clear, actionable error messages for end users
- **Debug Mode Support**: Detailed error information for developers
- **Input Validation**: Robust validation for all user inputs
- **Safe Operations**: Wrapper functions for error-prone operations

## Core Components

### 1. ErrorHandler

The central error handling component that manages error logging and provides safe execution wrappers.

```python
from core.error_handling import ErrorHandler

# Initialize with debug mode
error_handler = ErrorHandler(debug_mode=True)

# Log errors with context
error_handler.log_error(error, "Data loading", "load_csv_file")

# Safe execution
result, error = error_handler.safe_execute(function, *args, context="Operation context")

# Get error summary
summary = error_handler.get_error_summary()
```

**Key Methods:**
- `log_error()`: Log errors with context and operation information
- `safe_execute()`: Execute functions with error handling
- `safe_async_execute()`: Execute async functions with error handling
- `get_error_summary()`: Get summary of recent errors

### 2. InputValidator

Validates user inputs and configuration parameters.

```python
from core.error_handling import InputValidator

validator = InputValidator()

# Validate model type
is_valid, error = validator.validate_model_type("GPT-3.5")

# Validate sequence length
is_valid, error = validator.validate_sequence_length(5)

# Validate creativity level
is_valid, error = validator.validate_creativity_level(0.7)

# Validate subscriber data
is_valid, error = validator.validate_subscriber_data(subscriber_dict)
```

**Validation Rules:**
- **Model Type**: Must be one of ["GPT-3.5", "GPT-4", "Claude", "Custom", "Custom Model"]
- **Sequence Length**: Must be integer between 1 and 10
- **Creativity Level**: Must be float between 0.1 and 1.0
- **Subscriber Data**: Must contain required fields (id, email, name, company)

### 3. DataLoaderErrorHandler

Handles errors in data loading and file operations.

```python
from core.error_handling import DataLoaderErrorHandler

data_handler = DataLoaderErrorHandler(error_handler)

# Safe CSV loading
df, error = data_handler.safe_load_csv("data.csv")

# Safe JSON loading
data, error = data_handler.safe_load_json("config.json")

# Safe data saving
success, error = data_handler.safe_save_data(data, "output.json", "json")
```

**Supported Operations:**
- CSV file loading with pandas
- JSON file loading and saving
- File existence validation
- Empty file detection
- Format validation

### 4. ModelInferenceErrorHandler

Handles errors in model loading and inference operations.

```python
from core.error_handling import ModelInferenceErrorHandler

model_handler = ModelInferenceErrorHandler(error_handler)

# Safe model loading
model, error = model_handler.safe_model_load("model.pth", "pytorch")

# Safe inference
outputs, error = model_handler.safe_model_inference(model, inputs)

# Safe batch inference
results, errors = model_handler.safe_batch_inference(model, batch_inputs)
```

**Error Handling:**
- GPU out of memory errors
- Model file not found
- Runtime errors during inference
- Batch processing errors

### 5. GradioErrorHandler

Provides error handling specifically for Gradio applications.

```python
from core.error_handling import GradioErrorHandler

gradio_handler = GradioErrorHandler(error_handler, debug_mode=True)

# Safe Gradio function execution
result = gradio_handler.safe_gradio_function(function, *args, **kwargs)

# Validate Gradio inputs
is_valid, errors = gradio_handler.validate_gradio_inputs(inputs)

# Format errors for Gradio display
formatted_error = gradio_handler._format_gradio_error("Error Type", "Error message")
```

## Error Types and Handling

### Custom Exception Classes

```python
class EmailSequenceError(Exception):
    """Base exception for email sequence system"""
    pass

class ValidationError(EmailSequenceError):
    """Raised when input validation fails"""
    pass

class ModelError(EmailSequenceError):
    """Raised when model operations fail"""
    pass

class DataError(EmailSequenceError):
    """Raised when data loading or processing fails"""
    pass

class ConfigurationError(EmailSequenceError):
    """Raised when configuration is invalid"""
    pass
```

### Error Handling Patterns

#### 1. Try-Except with Specific Exceptions

```python
try:
    result = risky_operation()
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Validation failed: {e}")
    return {"error": "Invalid input", "details": str(e)}
except ModelError as e:
    # Handle model errors
    logger.error(f"Model operation failed: {e}")
    return {"error": "Model error", "details": str(e)}
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
    return {"error": "Unexpected error", "details": str(e)}
```

#### 2. Safe Execution Wrappers

```python
# Using safe_execute
result, error = error_handler.safe_execute(
    risky_function, 
    arg1, arg2, 
    context="Function context"
)

if error:
    # Handle error
    print(f"Error occurred: {error}")
else:
    # Use result
    process_result(result)
```

#### 3. Decorator Pattern

```python
@handle_data_operation
def data_processing_function(data):
    # Function implementation
    return processed_data

@handle_model_operation
def model_inference_function(inputs):
    # Model inference
    return outputs

@handle_async_operation
async def async_processing_function(data):
    # Async processing
    return result
```

## Input Validation

### Validation Methods

```python
# Model type validation
is_valid, error = validator.validate_model_type("GPT-3.5")

# Sequence length validation
is_valid, error = validator.validate_sequence_length(5)

# Creativity level validation
is_valid, error = validator.validate_creativity_level(0.7)

# Subscriber data validation
is_valid, error = validator.validate_subscriber_data({
    "id": "sub_123",
    "email": "user@example.com",
    "name": "John Doe",
    "company": "Example Corp"
})

# Training configuration validation
is_valid, error = validator.validate_training_config({
    "max_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
})
```

### Validation Rules

| Field | Type | Validation Rules |
|-------|------|------------------|
| model_type | string | Must be in ["GPT-3.5", "GPT-4", "Claude", "Custom", "Custom Model"] |
| sequence_length | integer | Must be between 1 and 10 |
| creativity_level | float | Must be between 0.1 and 1.0 |
| subscriber.email | string | Must contain "@" and "." |
| training.max_epochs | integer | Must be at least 1 |
| training.batch_size | integer | Must be at least 1 |
| training.learning_rate | float | Must be positive |

## Data Loading Error Handling

### CSV Loading

```python
# Safe CSV loading with error handling
df, error = data_handler.safe_load_csv("data.csv")

if error:
    if "File not found" in error:
        # Handle missing file
        print("Data file not found. Please check the file path.")
    elif "empty" in error.lower():
        # Handle empty file
        print("Data file is empty. Please provide valid data.")
    elif "parsing" in error.lower():
        # Handle parsing errors
        print("Data file format is invalid. Please check the CSV format.")
    else:
        # Handle other errors
        print(f"Error loading data: {error}")
else:
    # Process the data
    process_dataframe(df)
```

### JSON Loading

```python
# Safe JSON loading with error handling
config, error = data_handler.safe_load_json("config.json")

if error:
    if "File not found" in error:
        print("Configuration file not found.")
    elif "Invalid JSON format" in error:
        print("Configuration file has invalid JSON format.")
    else:
        print(f"Error loading configuration: {error}")
else:
    # Use configuration
    apply_configuration(config)
```

### Data Saving

```python
# Safe data saving
success, error = data_handler.safe_save_data(
    data, 
    "output.json", 
    "json"
)

if not success:
    if "Unsupported file type" in error:
        print("Unsupported file format for saving.")
    else:
        print(f"Error saving data: {error}")
else:
    print("Data saved successfully.")
```

## Model Inference Error Handling

### Model Loading

```python
# Safe model loading
model, error = model_handler.safe_model_load("model.pth", "pytorch")

if error:
    if "Model file not found" in error:
        print("Model file not found. Please check the model path.")
    elif "GPU out of memory" in error:
        print("GPU memory insufficient. Try using CPU or reducing batch size.")
    elif "Unsupported model type" in error:
        print("Unsupported model format. Please use PyTorch models.")
    else:
        print(f"Error loading model: {error}")
else:
    # Use the model
    model.eval()
```

### Model Inference

```python
# Safe model inference
outputs, error = model_handler.safe_model_inference(model, inputs)

if error:
    if "GPU out of memory" in error:
        print("GPU memory insufficient during inference.")
    elif "Runtime error" in error:
        print("Runtime error during model inference.")
    else:
        print(f"Error during inference: {error}")
else:
    # Process outputs
    process_outputs(outputs)
```

### Batch Inference

```python
# Safe batch inference
results, errors = model_handler.safe_batch_inference(
    model, 
    batch_inputs, 
    batch_size=32
)

# Check for errors in batch
for i, error in enumerate(errors):
    if error:
        print(f"Error in batch item {i}: {error}")
    else:
        # Process successful result
        process_result(results[i])
```

## Gradio Integration

### Error Handling in Gradio Functions

```python
def safe_gradio_function(inputs):
    """Gradio function with comprehensive error handling"""
    
    try:
        # Validate inputs
        is_valid, errors = gradio_handler.validate_gradio_inputs(inputs)
        if not is_valid:
            return {
                "error": True,
                "message": "Input validation failed",
                "details": errors
            }
        
        # Process inputs
        result = process_inputs(inputs)
        
        return {
            "error": False,
            "result": result
        }
        
    except ValidationError as e:
        return gradio_handler._format_gradio_error("Validation Error", str(e))
    except ModelError as e:
        return gradio_handler._format_gradio_error("Model Error", str(e))
    except Exception as e:
        return gradio_handler._format_gradio_error("Unexpected Error", str(e))
```

### Input Validation in Gradio

```python
def validate_and_process(model_type, sequence_length, creativity_level):
    """Validate and process Gradio inputs"""
    
    # Validate each input
    validation_errors = []
    
    is_valid, error = validator.validate_model_type(model_type)
    if not is_valid:
        validation_errors.append(f"Model type: {error}")
    
    is_valid, error = validator.validate_sequence_length(sequence_length)
    if not is_valid:
        validation_errors.append(f"Sequence length: {error}")
    
    is_valid, error = validator.validate_creativity_level(creativity_level)
    if not is_valid:
        validation_errors.append(f"Creativity level: {error}")
    
    if validation_errors:
        return {
            "error": True,
            "message": "Validation failed",
            "details": validation_errors
        }
    
    # Process valid inputs
    return process_valid_inputs(model_type, sequence_length, creativity_level)
```

## Best Practices

### 1. Always Use Safe Execution

```python
# Good: Use safe execution
result, error = error_handler.safe_execute(risky_function, *args)

# Bad: Direct execution without error handling
result = risky_function(*args)  # May crash the application
```

### 2. Provide Context for Errors

```python
# Good: Provide context
error_handler.log_error(error, "Data loading from CSV", "load_subscriber_data")

# Bad: No context
error_handler.log_error(error)  # Hard to debug
```

### 3. Use Specific Exception Types

```python
# Good: Use specific exceptions
raise ValidationError("Invalid model type")

# Bad: Use generic exceptions
raise Exception("Something went wrong")
```

### 4. Validate Inputs Early

```python
# Good: Validate inputs first
def process_data(data, config):
    # Validate inputs
    is_valid, error = validator.validate_data(data)
    if not is_valid:
        raise ValidationError(error)
    
    # Process valid data
    return process_valid_data(data, config)

# Bad: Process without validation
def process_data(data, config):
    # Process without validation (may fail later)
    return process_data_internally(data, config)
```

### 5. Provide User-Friendly Error Messages

```python
# Good: User-friendly message
return {
    "error": True,
    "message": "Please select a valid model type from the dropdown menu.",
    "details": "Invalid model type: 'Custom Model'"
}

# Bad: Technical error message
return {
    "error": True,
    "message": "ValueError: 'Custom Model' not in ['GPT-3.5', 'GPT-4', 'Claude', 'Custom']"
}
```

### 6. Log Errors for Debugging

```python
# Good: Log errors with context
try:
    result = risky_operation()
except Exception as e:
    error_handler.log_error(e, "Sequence generation", "generate_email_sequence")
    raise

# Bad: Silent error handling
try:
    result = risky_operation()
except Exception as e:
    pass  # Error is lost
```

## Examples and Usage

### Complete Example: Email Sequence Generation

```python
def generate_email_sequence_with_error_handling(
    model_type: str,
    sequence_length: int,
    creativity_level: float,
    target_audience: str,
    industry_focus: str
):
    """Generate email sequence with comprehensive error handling"""
    
    # Initialize error handling
    error_handler = ErrorHandler(debug_mode=True)
    validator = InputValidator()
    
    try:
        # Step 1: Validate inputs
        validation_errors = []
        
        is_valid, error = validator.validate_model_type(model_type)
        if not is_valid:
            validation_errors.append(f"Model type: {error}")
        
        is_valid, error = validator.validate_sequence_length(sequence_length)
        if not is_valid:
            validation_errors.append(f"Sequence length: {error}")
        
        is_valid, error = validator.validate_creativity_level(creativity_level)
        if not is_valid:
            validation_errors.append(f"Creativity level: {error}")
        
        if validation_errors:
            return {
                "error": True,
                "message": "Input validation failed",
                "details": validation_errors
            }
        
        # Step 2: Load configuration
        data_handler = DataLoaderErrorHandler(error_handler)
        config, error = data_handler.safe_load_json("config.json")
        
        if error:
            return {
                "error": True,
                "message": "Failed to load configuration",
                "details": error
            }
        
        # Step 3: Generate sequence
        sequence_generator = EmailSequenceGenerator(config)
        sequence, error = error_handler.safe_execute(
            sequence_generator.generate_sequence,
            target_audience=target_audience,
            industry_focus=industry_focus,
            context="Sequence generation"
        )
        
        if error:
            return {
                "error": True,
                "message": "Failed to generate sequence",
                "details": error
            }
        
        # Step 4: Save results
        success, error = data_handler.safe_save_data(
            sequence.to_dict(),
            "generated_sequence.json",
            "json"
        )
        
        if not success:
            error_handler.log_error(
                Exception(error),
                "Saving generated sequence",
                "save_sequence"
            )
            # Continue despite save error
        
        return {
            "error": False,
            "sequence": sequence.to_dict(),
            "message": "Sequence generated successfully"
        }
        
    except Exception as e:
        error_handler.log_error(e, "Email sequence generation", "generate_sequence")
        return {
            "error": True,
            "message": "An unexpected error occurred",
            "details": str(e) if error_handler.debug_mode else "Please try again"
        }
```

### Gradio Integration Example

```python
import gradio as gr
from core.error_handling import GradioErrorHandler, ErrorHandler

# Initialize error handling
error_handler = ErrorHandler(debug_mode=True)
gradio_handler = GradioErrorHandler(error_handler, debug_mode=True)

def safe_gradio_sequence_generation(
    model_type: str,
    sequence_length: int,
    creativity_level: float,
    target_audience: str
):
    """Safe Gradio function for sequence generation"""
    
    return gradio_handler.safe_gradio_function(
        generate_email_sequence_with_error_handling,
        model_type=model_type,
        sequence_length=sequence_length,
        creativity_level=creativity_level,
        target_audience=target_audience,
        industry_focus="Technology"
    )

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Email Sequence Generator")
    
    with gr.Row():
        model_type = gr.Dropdown(
            choices=["GPT-3.5", "GPT-4", "Claude", "Custom"],
            value="GPT-3.5",
            label="AI Model"
        )
        
        sequence_length = gr.Slider(
            minimum=1,
            maximum=10,
            value=3,
            step=1,
            label="Sequence Length"
        )
    
    creativity_level = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.7,
        step=0.1,
        label="Creativity Level"
    )
    
    target_audience = gr.Textbox(
        value="Tech professionals",
        label="Target Audience"
    )
    
    generate_btn = gr.Button("Generate Sequence")
    
    output = gr.JSON(label="Result")
    
    generate_btn.click(
        fn=safe_gradio_sequence_generation,
        inputs=[model_type, sequence_length, creativity_level, target_audience],
        outputs=[output]
    )

app.launch()
```

## Testing and Debugging

### Running Error Handling Tests

```bash
# Run all error handling tests
python tests/test_error_handling.py

# Run specific test class
python -m unittest tests.test_error_handling.TestErrorHandler

# Run specific test method
python -m unittest tests.test_error_handling.TestErrorHandler.test_error_logging
```

### Debug Mode

```python
# Enable debug mode for detailed error information
error_handler = ErrorHandler(debug_mode=True)

# Debug mode provides:
# - Full stack traces
# - Detailed error context
# - Verbose logging
```

### Error Monitoring

```python
# Get error summary
summary = error_handler.get_error_summary()

print(f"Total errors: {summary['total_errors']}")
print(f"Recent errors: {len(summary['recent_errors'])}")
print(f"Error types: {summary['error_type_distribution']}")

# Monitor specific error types
if "ValidationError" in summary['error_type_distribution']:
    print("Validation errors detected - check input validation")
```

### Performance Monitoring

```python
# Monitor error rates
import time

start_time = time.time()
# ... perform operations ...
end_time = time.time()

error_rate = len(error_handler.error_log) / (end_time - start_time)
print(f"Error rate: {error_rate:.2f} errors per second")
```

## Conclusion

The error handling and validation system provides a robust foundation for the Email Sequence AI System, ensuring:

- **Reliability**: Graceful handling of errors without system crashes
- **User Experience**: Clear, actionable error messages
- **Debugging**: Comprehensive error logging and monitoring
- **Maintainability**: Structured error handling patterns
- **Scalability**: Modular design for easy extension

By following the patterns and best practices outlined in this guide, developers can build reliable, user-friendly applications that handle errors gracefully and provide excellent debugging capabilities. 