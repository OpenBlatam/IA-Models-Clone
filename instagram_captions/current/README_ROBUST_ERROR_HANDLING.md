# 🤖 Robust Error Handling and Input Validation System

## 📋 Overview

The Robust Error Handling and Input Validation System provides comprehensive error handling, input validation, and user-friendly error recovery for Gradio applications. This system ensures reliable operation of AI/ML applications with detailed error tracking, retry mechanisms, and helpful user feedback.

## 🎯 Key Features

### 🔒 **Advanced Input Validation**
- **Comprehensive Validation Rules**: Text, numeric, model, file, URL, email, JSON, and custom validation
- **Pattern Matching**: Built-in regex patterns for common input types
- **Range Validation**: Min/max value and length constraints
- **Custom Validators**: Extensible validation framework
- **Real-time Feedback**: Immediate validation results with helpful messages

### 🛡️ **Robust Error Handling**
- **Retry Mechanisms**: Automatic retry with configurable attempts and delays
- **Error Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL classification
- **Error Recovery Suggestions**: Context-aware recovery tips
- **Detailed Error Logging**: Comprehensive error tracking and history
- **Error Monitoring**: Real-time error statistics and monitoring

### 🧹 **Input Sanitization**
- **Text Cleaning**: Remove excessive whitespace and dangerous characters
- **XSS Prevention**: Basic protection against script injection
- **Data Normalization**: Consistent input formatting
- **Duplicate Removal**: Clean label lists and other data

### 📊 **Error Monitoring**
- **Error Statistics**: Total errors, error counts by type
- **Recent Error History**: Last 10 errors with details
- **Critical Error Tracking**: Monitor high-severity issues
- **Performance Metrics**: Error rates and recovery success

## 🏗️ System Architecture

### Core Components

#### 1. **ValidationRule** (Dataclass)
```python
@dataclass
class ValidationRule:
    validation_type: ValidationType
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    required: bool = True
    custom_validator: Optional[Callable] = None
    error_message: Optional[str] = None
```

#### 2. **ErrorInfo** (Dataclass)
```python
@dataclass
class ErrorInfo:
    error_type: str
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    retry_count: int
    max_retries: int
    recovery_suggestion: Optional[str]
```

#### 3. **RobustInputValidator**
- Comprehensive validation engine
- Built-in validation patterns
- Extensible validation framework
- Real-time validation feedback

#### 4. **RobustErrorHandler**
- Retry mechanism with exponential backoff
- Error severity classification
- Recovery suggestion generation
- Error history tracking

#### 5. **InputSanitizer**
- Text sanitization and cleaning
- XSS prevention
- Data normalization
- Duplicate removal

## 🚀 Quick Start

### Basic Usage

```python
from robust_error_handling_system import RobustGradioInterface

# Create robust interface
robust_interface = RobustGradioInterface()

# Create Gradio interface
interface = robust_interface.create_robust_interface()

# Launch the app
interface.launch()
```

### Custom Validation Rules

```python
from robust_error_handling_system import ValidationRule, ValidationType

# Add custom validation rule
validator = RobustInputValidator()

# Text validation with custom pattern
validator.add_validation_rule("username", ValidationRule(
    validation_type=ValidationType.TEXT,
    min_length=3,
    max_length=20,
    pattern=r'^[a-zA-Z0-9_]+$',
    error_message="❌ Username must be 3-20 characters, alphanumeric and underscore only"
))

# Numeric validation with range
validator.add_validation_rule("age", ValidationRule(
    validation_type=ValidationType.NUMERIC,
    min_value=0,
    max_value=150,
    error_message="❌ Age must be between 0 and 150"
))

# Custom validation function
def validate_email_domain(email: str) -> bool:
    allowed_domains = ['gmail.com', 'yahoo.com', 'outlook.com']
    domain = email.split('@')[-1] if '@' in email else ''
    return domain in allowed_domains

validator.add_validation_rule("email", ValidationRule(
    validation_type=ValidationType.CUSTOM,
    custom_validator=validate_email_domain,
    error_message="❌ Email must be from allowed domains"
))
```

### Error Handling Decorator

```python
from robust_error_handling_system import RobustErrorHandler

error_handler = RobustErrorHandler(max_retries=3, retry_delay=1.0)

@error_handler.handle_exception
def risky_function(input_data):
    # This function will automatically retry on failure
    result = process_data(input_data)
    return result
```

## 📝 Validation Types

### 1. **Text Validation**
```python
ValidationRule(
    validation_type=ValidationType.TEXT,
    min_length=10,
    max_length=1000,
    pattern='email',  # Built-in patterns: email, url, phone, date, time
    required=True
)
```

### 2. **Numeric Validation**
```python
ValidationRule(
    validation_type=ValidationType.NUMERIC,
    min_value=0.0,
    max_value=1.0,
    required=True
)
```

### 3. **Model Validation**
```python
ValidationRule(
    validation_type=ValidationType.MODEL,
    required=True
)
```

### 4. **File Validation**
```python
ValidationRule(
    validation_type=ValidationType.FILE,
    max_value=10*1024*1024,  # 10MB max file size
    required=True
)
```

### 5. **JSON Validation**
```python
ValidationRule(
    validation_type=ValidationType.JSON,
    required=True
)
```

### 6. **Custom Validation**
```python
def custom_validator(value):
    # Your custom validation logic
    return True

ValidationRule(
    validation_type=ValidationType.CUSTOM,
    custom_validator=custom_validator,
    error_message="❌ Custom validation failed"
)
```

## 🛠️ Built-in Validation Patterns

### Text Patterns
- **email**: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- **url**: `^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$`
- **phone**: `^\+?[\d\s\-\(\)]{10,}$`
- **date**: `^\d{4}-\d{2}-\d{2}$`
- **time**: `^\d{2}:\d{2}(:\d{2})?$`

### Model Validation
Supported models with descriptions:
- **gpt2**: Fast text generation
- **gpt2-medium**: Balanced performance
- **gpt2-large**: High quality generation
- **gpt2-xl**: Best quality (slower)
- **bert-base-uncased**: Good for classification
- **bert-base-cased**: Case-sensitive analysis
- **roberta-base**: Robust performance
- **distilbert-base-uncased**: Fast and efficient
- **t5-small**: Versatile text generation
- **t5-base**: Balanced T5 model

## 🔧 Error Severity Levels

### ErrorSeverity Enum
- **LOW**: Minor issues (ValueError, TypeError)
- **MEDIUM**: Standard errors (default)
- **HIGH**: Serious issues (MemoryError, OSError)
- **CRITICAL**: System-critical (KeyboardInterrupt, SystemExit)

### Automatic Severity Detection
```python
# The system automatically determines severity based on error type
if isinstance(error, (ValueError, TypeError)):
    severity = ErrorSeverity.LOW
elif isinstance(error, (MemoryError, OSError)):
    severity = ErrorSeverity.HIGH
elif isinstance(error, (KeyboardInterrupt, SystemExit)):
    severity = ErrorSeverity.CRITICAL
```

## 🔄 Retry Mechanism

### Configuration
```python
error_handler = RobustErrorHandler(
    max_retries=3,      # Maximum retry attempts
    retry_delay=1.0     # Delay between retries (seconds)
)
```

### Retry Behavior
1. **First Attempt**: Execute function normally
2. **On Failure**: Log error and wait for retry_delay
3. **Retry Attempts**: Repeat up to max_retries times
4. **Final Failure**: Return formatted error response

### Example Retry Flow
```
Attempt 1: Function execution → FAIL
Wait 1.0s → Attempt 2: Function execution → FAIL
Wait 1.0s → Attempt 3: Function execution → FAIL
Wait 1.0s → Attempt 4: Function execution → FAIL
Return error response
```

## 💡 Recovery Suggestions

### Context-Aware Suggestions
The system provides specific recovery suggestions based on error type:

- **ValueError**: "Please check your input values and ensure they are within valid ranges."
- **TypeError**: "Please ensure you're using the correct data types for your inputs."
- **MemoryError**: "Try reducing batch size or model size. Close other applications to free memory."
- **OSError**: "Check file permissions and ensure the file path is correct."
- **ConnectionError**: "Check your internet connection and try again."
- **TimeoutError**: "The operation took too long. Try with smaller inputs or check your system resources."
- **CUDAOutOfMemoryError**: "GPU memory is full. Try reducing batch size or use CPU instead."

## 📊 Error Monitoring

### Error Summary
```python
error_summary = robust_interface.get_error_summary()

# Returns:
{
    "total_errors": 15,
    "error_counts": {
        "ValueError": 5,
        "TypeError": 3,
        "MemoryError": 2,
        "OSError": 1
    },
    "recent_errors": [...],  # Last 10 errors with details
    "critical_errors": [...]  # Errors with CRITICAL severity
}
```

### Error History
Each error is logged with:
- Error type and severity
- Timestamp
- Retry count
- Recovery suggestion
- Full traceback
- Function arguments

## 🧹 Input Sanitization

### Text Sanitization
```python
# Remove excessive whitespace and dangerous characters
sanitized_text = InputSanitizer.sanitize_text("  <script>alert('xss')</script>  ")
# Result: "alert('xss')"
```

### Numeric Sanitization
```python
# Convert to float with fallback
sanitized_number = InputSanitizer.sanitize_numeric("123.45")
# Result: 123.45

sanitized_number = InputSanitizer.sanitize_numeric("invalid")
# Result: 0.0
```

### Model Name Sanitization
```python
# Remove dangerous characters from model names
sanitized_model = InputSanitizer.sanitize_model_name("gpt2<script>")
# Result: "gpt2"
```

### Label Sanitization
```python
# Clean and deduplicate labels
sanitized_labels = InputSanitizer.sanitize_labels("positive, negative, positive, neutral")
# Result: ["positive", "negative", "neutral"]
```

## 🎨 Gradio Integration

### Interface Components
The robust system provides:

1. **Text Generation Tab**: With comprehensive validation
2. **Sentiment Analysis Tab**: With input sanitization
3. **Text Classification Tab**: With label validation
4. **Error Monitoring Tab**: Real-time error statistics

### Event Handlers
All event handlers are wrapped with error handling:
```python
generate_btn.click(
    fn=self.robust_text_generation,
    inputs=[prompt_input, max_length_input, temperature_input, model_name_input],
    outputs=[output_text, None, status_output]
)
```

### Status Feedback
Real-time status updates with emojis:
- ✅ Success operations
- ❌ Validation failures
- ⚠️ Error occurrences
- 🔄 Processing states

## 🔧 Configuration Options

### Error Handler Configuration
```python
error_handler = RobustErrorHandler(
    max_retries=5,           # More retries for critical operations
    retry_delay=2.0          # Longer delay between retries
)
```

### Validation Rule Configuration
```python
# Strict validation
strict_rule = ValidationRule(
    validation_type=ValidationType.TEXT,
    min_length=50,
    max_length=500,
    required=True,
    error_message="❌ Text must be exactly 50-500 characters"
)

# Lenient validation
lenient_rule = ValidationRule(
    validation_type=ValidationType.TEXT,
    min_length=1,
    max_length=10000,
    required=False
)
```

## 🚀 Advanced Usage

### Custom Error Handler
```python
class CustomErrorHandler(RobustErrorHandler):
    def _get_recovery_suggestion(self, error_type: str, error: Exception) -> str:
        # Add custom recovery suggestions
        custom_suggestions = {
            "CustomError": "This is a custom error. Please contact support.",
            "NetworkError": "Check your network connection and firewall settings."
        }
        
        # Use custom suggestions or fall back to default
        return custom_suggestions.get(error_type, super()._get_recovery_suggestion(error_type, error))
```

### Custom Validator
```python
def validate_complex_input(value: str) -> bool:
    """Complex validation logic"""
    # Check multiple conditions
    if not value:
        return False
    
    # Check format
    if not re.match(r'^[A-Z]{2}\d{4}$', value):
        return False
    
    # Check business rules
    if value.startswith('XX'):
        return False
    
    return True

# Add to validator
validator.add_validation_rule("complex_field", ValidationRule(
    validation_type=ValidationType.CUSTOM,
    custom_validator=validate_complex_input,
    error_message="❌ Invalid format. Must be 2 letters + 4 digits, not starting with XX"
))
```

### Error Monitoring Dashboard
```python
def create_error_dashboard():
    """Create a comprehensive error monitoring dashboard"""
    error_summary = robust_interface.get_error_summary()
    
    dashboard_html = f"""
    <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <h3>📊 Error Monitoring Dashboard</h3>
        <p><strong>Total Errors:</strong> {error_summary['total_errors']}</p>
        <p><strong>Critical Errors:</strong> {len(error_summary['critical_errors'])}</p>
        
        <h4>Error Breakdown:</h4>
        <ul>
        """
    
    for error_type, count in error_summary['error_counts'].items():
        dashboard_html += f"<li>{error_type}: {count}</li>"
    
    dashboard_html += """
        </ul>
    </div>
    """
    
    return dashboard_html
```

## 🧪 Testing

### Validation Testing
```python
def test_validation():
    validator = RobustInputValidator()
    
    # Test text validation
    is_valid, message = validator.validate_input("prompt", "Hello world")
    assert is_valid == True
    
    is_valid, message = validator.validate_input("prompt", "")
    assert is_valid == False
    
    # Test numeric validation
    is_valid, message = validator.validate_input("max_length", 50)
    assert is_valid == True
    
    is_valid, message = validator.validate_input("max_length", 2000)
    assert is_valid == False
```

### Error Handling Testing
```python
def test_error_handling():
    error_handler = RobustErrorHandler(max_retries=2)
    
    @error_handler.handle_exception
    def failing_function():
        raise ValueError("Test error")
    
    # Should retry and eventually return error response
    result = failing_function()
    assert "Test error" in result[0]
```

## 📈 Performance Considerations

### Validation Performance
- **Cached Patterns**: Regex patterns are compiled once
- **Early Exit**: Validation stops on first failure
- **Efficient Checks**: Optimized validation order

### Error Handling Performance
- **Minimal Overhead**: Decorator pattern for clean integration
- **Configurable Retries**: Balance between reliability and performance
- **Memory Efficient**: Error history with configurable limits

### Monitoring Performance
- **Lazy Loading**: Error summaries generated on demand
- **Limited History**: Recent errors only to prevent memory bloat
- **Efficient Logging**: Structured logging for easy parsing

## 🔒 Security Features

### Input Sanitization
- **XSS Prevention**: Remove dangerous script tags
- **SQL Injection Prevention**: Sanitize database inputs
- **Path Traversal Prevention**: Validate file paths

### Error Information
- **Sensitive Data Filtering**: Don't log passwords or tokens
- **Stack Trace Sanitization**: Remove internal paths
- **User-Friendly Messages**: Don't expose system details

## 🚀 Deployment

### Production Configuration
```python
# Production-ready configuration
robust_interface = RobustGradioInterface()

# Configure for production
error_handler = RobustErrorHandler(
    max_retries=3,
    retry_delay=1.0
)

# Add production logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_nlp.log'),
        logging.StreamHandler()
    ]
)
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7861

CMD ["python", "robust_error_handling_system.py"]
```

## 📚 Best Practices

### 1. **Validation Rules**
- Define clear, specific validation rules
- Use meaningful error messages
- Test validation with edge cases
- Keep validation rules maintainable

### 2. **Error Handling**
- Use appropriate retry counts for different operations
- Provide helpful recovery suggestions
- Monitor error patterns in production
- Regularly review error logs

### 3. **Input Sanitization**
- Always sanitize user inputs
- Use appropriate sanitization methods
- Test sanitization with malicious inputs
- Keep sanitization rules updated

### 4. **Monitoring**
- Set up error alerts for critical errors
- Monitor error rates and trends
- Use error data for system improvements
- Regular error log analysis

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning Error Prediction**: Predict errors before they occur
- **Advanced Recovery Strategies**: Automatic error recovery
- **Performance Optimization**: Faster validation and error handling
- **Integration APIs**: Easy integration with other systems
- **Visual Error Analytics**: Interactive error visualization
- **A/B Testing Support**: Error handling experimentation

### Extension Points
- **Custom Error Types**: Define application-specific errors
- **Validation Plugins**: Modular validation system
- **Error Reporting**: Integration with error reporting services
- **Performance Profiling**: Detailed performance analysis

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd robust-error-handling

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 robust_error_handling_system.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation
- This README file
- Inline code documentation
- Example usage in the codebase

### Issues
- Report bugs via GitHub Issues
- Request features via GitHub Issues
- Ask questions via GitHub Discussions

### Community
- Join our Discord server
- Follow us on Twitter
- Subscribe to our newsletter

---

**🤖 Robust Error Handling System** - Making AI applications more reliable, one error at a time!




