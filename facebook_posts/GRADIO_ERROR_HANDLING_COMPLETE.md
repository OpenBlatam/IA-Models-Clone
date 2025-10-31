# Gradio Error Handling and Input Validation - Complete Documentation

## Overview

The Gradio Error Handling and Input Validation system provides comprehensive error handling, input validation, and user-friendly error messages for Gradio applications. This system ensures robust and reliable deep learning demos with proper error management.

## Architecture

### Core Components

1. **InputValidator**: Comprehensive input validation system
2. **ErrorHandler**: Centralized error handling and logging
3. **GradioErrorHandler**: Gradio-specific error handling wrapper
4. **RobustGradioInterface**: Complete robust interface system
5. **ValidationRule**: Configurable validation rules
6. **ErrorConfig**: Error handling configuration

### Key Features

- **Comprehensive Validation**: Multiple validation types (required, type, range, format, size, content)
- **User-Friendly Errors**: Clear, actionable error messages
- **Robust Error Handling**: Graceful degradation and recovery
- **Logging System**: Detailed error logging for debugging
- **Timeout Protection**: Prevents hanging operations
- **Retry Logic**: Automatic retry with exponential backoff

## Error Types

### ErrorType Enum

```python
class ErrorType(Enum):
    INPUT_VALIDATION = "input_validation"
    MODEL_ERROR = "model_error"
    PROCESSING_ERROR = "processing_error"
    VISUALIZATION_ERROR = "visualization_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
```

### Validation Types

```python
class ValidationType(Enum):
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    SIZE_CHECK = "size_check"
    CONTENT_CHECK = "content_check"
```

## Input Validation System

### ValidationRule Configuration

```python
@dataclass
class ValidationRule:
    validation_type: ValidationType
    field_name: str
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_types: Optional[List[type]] = None
    allowed_formats: Optional[List[str]] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    custom_validator: Optional[Callable] = None
    error_message: Optional[str] = None
```

### InputValidator Class

```python
class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self, validation_rules: List[ValidationRule]):
        self.validation_rules = validation_rules
        self.logger = self._setup_logging()
    
    def validate_input(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all inputs according to rules."""
        errors = []
        
        for rule in self.validation_rules:
            field_value = inputs.get(rule.field_name)
            
            # Check if required field is present
            if rule.required and field_value is None:
                errors.append(f"Field '{rule.field_name}' is required")
                continue
            
            # Skip validation if field is None and not required
            if field_value is None:
                continue
            
            # Type check
            if rule.allowed_types:
                if not any(isinstance(field_value, t) for t in rule.allowed_types):
                    errors.append(f"Field '{rule.field_name}' must be one of {rule.allowed_types}")
                    continue
            
            # Range check for numeric values
            if rule.min_value is not None or rule.max_value is not None:
                if isinstance(field_value, (int, float)):
                    if rule.min_value is not None and field_value < rule.min_value:
                        errors.append(f"Field '{rule.field_name}' must be >= {rule.min_value}")
                    if rule.max_value is not None and field_value > rule.max_value:
                        errors.append(f"Field '{rule.field_name}' must be <= {rule.max_value}")
                else:
                    errors.append(f"Field '{rule.field_name}' must be numeric for range validation")
            
            # Size check
            if rule.min_size is not None or rule.max_size is not None:
                if hasattr(field_value, '__len__'):
                    size = len(field_value)
                    if rule.min_size is not None and size < rule.min_size:
                        errors.append(f"Field '{rule.field_name}' must have at least {rule.min_size} elements")
                    if rule.max_size is not None and size > rule.max_size:
                        errors.append(f"Field '{rule.field_name}' must have at most {rule.max_size} elements")
                else:
                    errors.append(f"Field '{rule.field_name}' must be a sequence for size validation")
            
            # Format check
            if rule.allowed_formats:
                if isinstance(field_value, str):
                    if not any(field_value.endswith(fmt) for fmt in rule.allowed_formats):
                        errors.append(f"Field '{rule.field_name}' must have one of these formats: {rule.allowed_formats}")
                else:
                    errors.append(f"Field '{rule.field_name}' must be a string for format validation")
            
            # Custom validator
            if rule.custom_validator:
                try:
                    if not rule.custom_validator(field_value):
                        error_msg = rule.error_message or f"Field '{rule.field_name}' failed custom validation"
                        errors.append(error_msg)
                except Exception as e:
                    errors.append(f"Custom validation failed for '{rule.field_name}': {str(e)}")
        
        return len(errors) == 0, errors
```

### Specialized Validation Methods

#### Image Validation

```python
def validate_image(self, image: Image.Image, min_size: Tuple[int, int] = (28, 28), 
                  max_size: Tuple[int, int] = (1024, 1024)) -> Tuple[bool, List[str]]:
    """Validate image input."""
    errors = []
    
    if image is None:
        errors.append("Image is required")
        return False, errors
    
    # Check image size
    width, height = image.size
    min_width, min_height = min_size
    max_width, max_height = max_size
    
    if width < min_width or height < min_height:
        errors.append(f"Image must be at least {min_width}x{min_height} pixels")
    
    if width > max_width or height > max_height:
        errors.append(f"Image must be at most {max_width}x{max_height} pixels")
    
    # Check image mode
    if image.mode not in ['RGB', 'L', 'RGBA']:
        errors.append("Image must be RGB, grayscale, or RGBA")
    
    return len(errors) == 0, errors
```

#### Numeric List Validation

```python
def validate_numeric_list(self, value: str, expected_length: int = 10, 
                        min_val: float = -1000, max_val: float = 1000) -> Tuple[bool, List[str]]:
    """Validate comma-separated numeric list."""
    errors = []
    
    if not value or not value.strip():
        errors.append("Input cannot be empty")
        return False, errors
    
    try:
        # Split and convert to numbers
        numbers = [float(x.strip()) for x in value.split(',')]
        
        # Check length
        if len(numbers) != expected_length:
            errors.append(f"Expected {expected_length} values, got {len(numbers)}")
        
        # Check range
        for i, num in enumerate(numbers):
            if num < min_val or num > max_val:
                errors.append(f"Value {i+1} ({num}) must be between {min_val} and {max_val}")
        
    except ValueError as e:
        errors.append(f"Invalid numeric format: {str(e)}")
    
    return len(errors) == 0, errors
```

#### Text Input Validation

```python
def validate_text_input(self, text: str, min_length: int = 1, max_length: int = 1000,
                       allowed_chars: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Validate text input."""
    errors = []
    
    if text is None:
        errors.append("Text input is required")
        return False, errors
    
    if len(text) < min_length:
        errors.append(f"Text must be at least {min_length} characters long")
    
    if len(text) > max_length:
        errors.append(f"Text must be at most {max_length} characters long")
    
    if allowed_chars:
        invalid_chars = [char for char in text if char not in allowed_chars]
        if invalid_chars:
            errors.append(f"Text contains invalid characters: {invalid_chars}")
    
    return len(errors) == 0, errors
```

## Error Handling System

### ErrorConfig Configuration

```python
@dataclass
class ErrorConfig:
    """Error handling configuration."""
    show_traceback: bool = False
    log_errors: bool = True
    return_partial_results: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    graceful_degradation: bool = True
```

### ErrorHandler Class

```python
class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, config: ErrorConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.error_counts = {}
    
    def handle_error(self, error: Exception, error_type: ErrorType, 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle errors with appropriate responses."""
        error_id = f"{error_type.value}_{int(time.time())}"
        
        # Log error
        if self.config.log_errors:
            self.logger.error(f"Error {error_id}: {str(error)}")
            if self.config.show_traceback:
                self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update error counts
        self.error_counts[error_type.value] = self.error_counts.get(error_type.value, 0) + 1
        
        # Create user-friendly error message
        error_message = self._create_user_friendly_message(error, error_type)
        
        # Create response
        response = {
            'success': False,
            'error_id': error_id,
            'error_type': error_type.value,
            'message': error_message,
            'timestamp': time.time()
        }
        
        if context:
            response['context'] = context
        
        return response
```

### User-Friendly Error Messages

```python
def _create_user_friendly_message(self, error: Exception, error_type: ErrorType) -> str:
    """Create user-friendly error message."""
    if error_type == ErrorType.INPUT_VALIDATION:
        return f"Input validation error: {str(error)}"
    
    elif error_type == ErrorType.MODEL_ERROR:
        return "Model processing error. Please try again with different inputs."
    
    elif error_type == ErrorType.PROCESSING_ERROR:
        return "Data processing error. Please check your input format."
    
    elif error_type == ErrorType.VISUALIZATION_ERROR:
        return "Visualization error. Please try again."
    
    elif error_type == ErrorType.SYSTEM_ERROR:
        return "System error. Please try again later."
    
    elif error_type == ErrorType.NETWORK_ERROR:
        return "Network error. Please check your connection and try again."
    
    elif error_type == ErrorType.MEMORY_ERROR:
        return "Memory error. Please try with smaller inputs."
    
    elif error_type == ErrorType.TIMEOUT_ERROR:
        return "Request timed out. Please try again."
    
    else:
        return f"An error occurred: {str(error)}"
```

### Retry Logic with Exponential Backoff

```python
def retry_with_backoff(self, func: Callable, *args, max_retries: int = None, 
                      **kwargs) -> Any:
    """Retry function with exponential backoff."""
    if max_retries is None:
        max_retries = self.config.max_retries
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    raise Exception("Max retries exceeded")
```

## Gradio Integration

### GradioErrorHandler Class

```python
class GradioErrorHandler:
    """Gradio-specific error handling wrapper."""
    
    def __init__(self, validator: InputValidator, error_handler: ErrorHandler):
        self.validator = validator
        self.error_handler = error_handler
    
    def safe_function_wrapper(self, func: Callable, *args, **kwargs) -> Callable:
        """Wrap function with error handling."""
        def wrapper(*func_args, **func_kwargs):
            try:
                # Validate inputs
                inputs = self._extract_inputs(func_args, func_kwargs)
                is_valid, validation_errors = self.validator.validate_input(inputs)
                
                if not is_valid:
                    error_response = self.error_handler.handle_error(
                        ValueError(f"Validation errors: {validation_errors}"),
                        ErrorType.INPUT_VALIDATION,
                        {'validation_errors': validation_errors}
                    )
                    return self._format_error_response(error_response)
                
                # Execute function with timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Function execution timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.error_handler.config.timeout_seconds)
                
                try:
                    result = func(*func_args, **func_kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                except TimeoutError:
                    error_response = self.error_handler.handle_error(
                        TimeoutError("Function execution timed out"),
                        ErrorType.TIMEOUT_ERROR
                    )
                    return self._format_error_response(error_response)
                
            except Exception as e:
                error_type = self._classify_error(e)
                error_response = self.error_handler.handle_error(e, error_type)
                return self._format_error_response(error_response)
        
        return wrapper
```

### Error Classification

```python
def _classify_error(self, error: Exception) -> ErrorType:
    """Classify error type."""
    if isinstance(error, (ValueError, TypeError)):
        return ErrorType.INPUT_VALIDATION
    elif isinstance(error, (RuntimeError, OSError)):
        return ErrorType.SYSTEM_ERROR
    elif isinstance(error, MemoryError):
        return ErrorType.MEMORY_ERROR
    elif isinstance(error, TimeoutError):
        return ErrorType.TIMEOUT_ERROR
    elif isinstance(error, ConnectionError):
        return ErrorType.NETWORK_ERROR
    else:
        return ErrorType.PROCESSING_ERROR
```

## Robust Interface Examples

### Robust Classification Demo

```python
def create_robust_classification_demo(self):
    """Create robust classification demo with error handling."""
    
    @self.gradio_error_handler.safe_function_wrapper
    def classify_digit_with_validation(image, confidence_threshold=0.5):
        """Classify digit with comprehensive validation."""
        # Additional validation
        if image is None:
            raise ValueError("Please upload an image")
        
        # Validate image
        is_valid, errors = self.validator.validate_image(image)
        if not is_valid:
            raise ValueError(f"Image validation failed: {errors}")
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        # Process image
        try:
            image = image.convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image) / 255.0
            image_tensor = torch.FloatTensor(image_array).flatten().unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")
        
        # Model inference
        try:
            with torch.no_grad():
                # Simple model for demo
                model = nn.Sequential(
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10),
                    nn.Softmax(dim=1)
                )
                
                output = model(image_tensor)
                probabilities = output.squeeze().numpy()
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
                
                if confidence < confidence_threshold:
                    return f"Low confidence prediction: {prediction} ({confidence:.3f})", None
                
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        # Create visualization
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax1.imshow(image_array, cmap='gray')
            ax1.set_title(f'Predicted: {prediction} (Confidence: {confidence:.3f})')
            ax1.axis('off')
            
            # Probability distribution
            ax2.bar(range(10), probabilities)
            ax2.set_title('Class Probabilities')
            ax2.set_xlabel('Digit')
            ax2.set_ylabel('Probability')
            ax2.set_xticks(range(10))
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return f"Prediction: {prediction} (Confidence: {confidence:.3f})", buf.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"Visualization failed: {str(e)}")
```

### Robust Regression Demo

```python
def create_robust_regression_demo(self):
    """Create robust regression demo with error handling."""
    
    @self.gradio_error_handler.safe_function_wrapper
    def predict_value_with_validation(features, model_type="linear"):
        """Predict value with comprehensive validation."""
        # Validate features
        is_valid, errors = self.validator.validate_numeric_list(features)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {errors}")
        
        # Validate model type
        if model_type not in ["linear", "polynomial", "neural"]:
            raise ValueError("Model type must be 'linear', 'polynomial', or 'neural'")
        
        try:
            # Parse features
            feature_values = [float(x.strip()) for x in features.split(',')]
            input_tensor = torch.FloatTensor(feature_values).unsqueeze(0)
            
            # Model inference
            with torch.no_grad():
                if model_type == "linear":
                    model = nn.Sequential(
                        nn.Linear(10, 1)
                    )
                elif model_type == "polynomial":
                    model = nn.Sequential(
                        nn.Linear(10, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
                else:  # neural
                    model = nn.Sequential(
                        nn.Linear(10, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
                
                prediction = model(input_tensor)
                predicted_value = prediction.item()
                
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        # Create visualization
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Feature importance
            feature_names = [f'F{i+1}' for i in range(10)]
            ax1.bar(feature_names, feature_values)
            ax1.set_title('Input Features')
            ax1.set_xlabel('Feature')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
            
            # Prediction
            ax2.bar(['Predicted'], [predicted_value], color='green')
            ax2.set_title('Predicted Value')
            ax2.set_ylabel('Value')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return f"Predicted Value: {predicted_value:.4f}", buf.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"Visualization failed: {str(e)}")
```

## Usage Examples

### Basic Usage

```python
# Create validation rules
validation_rules = [
    ValidationRule(
        validation_type=ValidationType.REQUIRED,
        field_name="image",
        required=True
    ),
    ValidationRule(
        validation_type=ValidationType.RANGE_CHECK,
        field_name="confidence_threshold",
        min_value=0.0,
        max_value=1.0
    )
]

# Create error handler
error_config = ErrorConfig(
    show_traceback=False,
    log_errors=True,
    return_partial_results=True,
    timeout_seconds=30,
    max_retries=3,
    graceful_degradation=True
)

# Create components
validator = InputValidator(validation_rules)
error_handler = ErrorHandler(error_config)
gradio_error_handler = GradioErrorHandler(validator, error_handler)

# Wrap function
@gradio_error_handler.safe_function_wrapper
def my_function(image, confidence_threshold):
    # Function implementation
    pass
```

### Advanced Usage

```python
# Custom validation function
def validate_custom_format(value):
    """Custom validation for specific format."""
    if not isinstance(value, str):
        return False
    return value.startswith("data:")

# Custom validation rule
custom_rule = ValidationRule(
    validation_type=ValidationType.CONTENT_CHECK,
    field_name="custom_input",
    custom_validator=validate_custom_format,
    error_message="Input must start with 'data:'"
)

# Error handling with context
try:
    result = process_data(input_data)
except Exception as e:
    error_response = error_handler.handle_error(
        e, 
        ErrorType.PROCESSING_ERROR,
        context={'input_size': len(input_data), 'operation': 'process_data'}
    )
```

## Best Practices

### Error Handling Best Practices

1. **User-Friendly Messages**: Always provide clear, actionable error messages
2. **Graceful Degradation**: Return partial results when possible
3. **Proper Logging**: Log errors for debugging while showing user-friendly messages
4. **Timeout Protection**: Prevent hanging operations
5. **Retry Logic**: Implement retry with exponential backoff for transient errors

### Input Validation Best Practices

1. **Early Validation**: Validate inputs as early as possible
2. **Comprehensive Rules**: Cover all possible input scenarios
3. **Custom Validators**: Use custom validators for complex validation logic
4. **Clear Error Messages**: Provide specific error messages for each validation failure
5. **Performance**: Keep validation efficient for real-time applications

### Gradio Integration Best Practices

1. **Safe Wrapping**: Always wrap functions with error handling
2. **Timeout Protection**: Use timeouts to prevent hanging
3. **User Feedback**: Provide immediate feedback for validation errors
4. **Graceful Errors**: Handle errors gracefully without crashing the interface
5. **Error Recovery**: Allow users to correct errors and retry

## Configuration Options

### ErrorConfig Options

```python
error_config = ErrorConfig(
    show_traceback=False,        # Show full traceback in logs
    log_errors=True,             # Enable error logging
    return_partial_results=True, # Return partial results on error
    timeout_seconds=30,          # Function timeout
    max_retries=3,              # Max retry attempts
    graceful_degradation=True    # Enable graceful degradation
)
```

### ValidationRule Options

```python
validation_rule = ValidationRule(
    validation_type=ValidationType.RANGE_CHECK,
    field_name="confidence",
    required=True,
    min_value=0.0,
    max_value=1.0,
    allowed_types=[float, int],
    custom_validator=None,
    error_message="Confidence must be between 0.0 and 1.0"
)
```

## Monitoring and Debugging

### Error Logging

```python
# Error logs are automatically created
# Check gradio_errors.log for detailed error information

# Monitor error counts
error_counts = error_handler.error_counts
print(f"Error counts: {error_counts}")
```

### Performance Monitoring

```python
# Monitor validation performance
import time

start_time = time.time()
is_valid, errors = validator.validate_input(inputs)
validation_time = time.time() - start_time

print(f"Validation took {validation_time:.3f} seconds")
```

## Conclusion

The Gradio Error Handling and Input Validation system provides a comprehensive solution for creating robust, user-friendly deep learning applications. The system ensures:

- **Reliability**: Comprehensive error handling prevents crashes
- **User Experience**: Clear error messages guide users to correct issues
- **Maintainability**: Structured error handling and logging
- **Performance**: Efficient validation and timeout protection
- **Flexibility**: Configurable validation rules and error handling

This system is essential for production-ready deep learning applications that need to handle various user inputs and provide reliable, user-friendly experiences. 