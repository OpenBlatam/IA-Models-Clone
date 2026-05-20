"""
Robust Error Handling and Input Validation System for Gradio Apps
Advanced error handling with retry mechanisms, detailed logging, and user-friendly recovery
"""

import gradio as gr
import torch
import numpy as np
import logging
import traceback
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Validation types"""
    TEXT = "text"
    NUMERIC = "numeric"
    MODEL = "model"
    FILE = "file"
    URL = "url"
    EMAIL = "email"
    JSON = "json"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Validation rule configuration"""
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


@dataclass
class ErrorInfo:
    """Error information structure"""
    error_type: str
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    recovery_suggestion: Optional[str] = None


class RobustInputValidator:
    """Advanced input validation with comprehensive rules"""
    
    def __init__(self):
        self.validation_rules = {}
        self.custom_validators = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        # Text validation patterns
        self.text_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$',
            'phone': r'^\+?[\d\s\-\(\)]{10,}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'time': r'^\d{2}:\d{2}(:\d{2})?$'
        }
        
        # Model validation
        self.valid_models = {
            "gpt2": {"type": "text-generation", "max_length": 1024, "description": "Fast text generation"},
            "gpt2-medium": {"type": "text-generation", "max_length": 1024, "description": "Balanced performance"},
            "gpt2-large": {"type": "text-generation", "max_length": 1024, "description": "High quality generation"},
            "gpt2-xl": {"type": "text-generation", "max_length": 1024, "description": "Best quality (slower)"},
            "bert-base-uncased": {"type": "classification", "max_length": 512, "description": "Good for classification"},
            "bert-base-cased": {"type": "classification", "max_length": 512, "description": "Case-sensitive analysis"},
            "roberta-base": {"type": "classification", "max_length": 512, "description": "Robust performance"},
            "distilbert-base-uncased": {"type": "classification", "max_length": 512, "description": "Fast and efficient"},
            "t5-small": {"type": "text-generation", "max_length": 512, "description": "Versatile text generation"},
            "t5-base": {"type": "text-generation", "max_length": 512, "description": "Balanced T5 model"}
        }
    
    def add_validation_rule(self, field_name: str, rule: ValidationRule):
        """Add a custom validation rule"""
        self.validation_rules[field_name] = rule
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """Add a custom validator function"""
        self.custom_validators[name] = validator_func
    
    def validate_text(self, text: str, rule: ValidationRule) -> Tuple[bool, str]:
        """Validate text input with comprehensive checks"""
        if not text and rule.required:
            return False, "❌ This field is required"
        
        if not text:
            return True, "✅ Valid input"
        
        text = text.strip()
        
        # Length validation
        if rule.min_length and len(text) < rule.min_length:
            return False, f"❌ Text must be at least {rule.min_length} characters long"
        
        if rule.max_length and len(text) > rule.max_length:
            return False, f"❌ Text must be no more than {rule.max_length} characters"
        
        # Pattern validation
        if rule.pattern:
            if rule.pattern in self.text_patterns:
                pattern = self.text_patterns[rule.pattern]
            else:
                pattern = rule.pattern
            
            if not re.match(pattern, text):
                return False, f"❌ Invalid format for {rule.pattern}"
        
        # Custom validation
        if rule.custom_validator:
            try:
                result = rule.custom_validator(text)
                if not result:
                    return False, rule.error_message or "❌ Custom validation failed"
            except Exception as e:
                return False, f"❌ Validation error: {str(e)}"
        
        return True, "✅ Valid input"
    
    def validate_numeric(self, value: Union[int, float, str], rule: ValidationRule) -> Tuple[bool, str]:
        """Validate numeric input"""
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            return False, "❌ Please enter a valid number"
        
        # Range validation
        if rule.min_value is not None and num_val < rule.min_value:
            return False, f"❌ Value must be at least {rule.min_value}"
        
        if rule.max_value is not None and num_val > rule.max_value:
            return False, f"❌ Value must be no more than {rule.max_value}"
        
        return True, "✅ Valid input"
    
    def validate_model(self, model_name: str, rule: ValidationRule) -> Tuple[bool, str]:
        """Validate model name"""
        if model_name not in self.valid_models:
            valid_models = ", ".join(self.valid_models.keys())
            return False, f"❌ Invalid model. Choose from: {valid_models}"
        
        model_info = self.valid_models[model_name]
        return True, f"✅ {model_info['description']}"
    
    def validate_json(self, json_str: str, rule: ValidationRule) -> Tuple[bool, str]:
        """Validate JSON input"""
        try:
            json.loads(json_str)
            return True, "✅ Valid JSON"
        except json.JSONDecodeError as e:
            return False, f"❌ Invalid JSON format: {str(e)}"
    
    def validate_file(self, file_path: str, rule: ValidationRule) -> Tuple[bool, str]:
        """Validate file input"""
        import os
        
        if not os.path.exists(file_path):
            return False, "❌ File does not exist"
        
        if not os.path.isfile(file_path):
            return False, "❌ Path is not a file"
        
        # File size validation
        file_size = os.path.getsize(file_path)
        if rule.max_value and file_size > rule.max_value:
            return False, f"❌ File size exceeds maximum ({rule.max_value} bytes)"
        
        return True, "✅ Valid file"
    
    def validate_input(self, field_name: str, value: Any) -> Tuple[bool, str]:
        """Main validation method"""
        if field_name not in self.validation_rules:
            return True, "✅ No validation rule specified"
        
        rule = self.validation_rules[field_name]
        
        try:
            if rule.validation_type == ValidationType.TEXT:
                return self.validate_text(str(value), rule)
            elif rule.validation_type == ValidationType.NUMERIC:
                return self.validate_numeric(value, rule)
            elif rule.validation_type == ValidationType.MODEL:
                return self.validate_model(str(value), rule)
            elif rule.validation_type == ValidationType.JSON:
                return self.validate_json(str(value), rule)
            elif rule.validation_type == ValidationType.FILE:
                return self.validate_file(str(value), rule)
            elif rule.validation_type == ValidationType.CUSTOM:
                if rule.custom_validator:
                    result = rule.custom_validator(value)
                    return (True, "✅ Valid input") if result else (False, rule.error_message or "❌ Custom validation failed")
                else:
                    return True, "✅ No custom validator specified"
            else:
                return True, "✅ Unknown validation type"
        
        except Exception as e:
            logger.error(f"Validation error for {field_name}: {e}")
            return False, f"❌ Validation error: {str(e)}"


class RobustErrorHandler:
    """Advanced error handling with retry mechanisms and recovery"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_history = []
        self.error_counts = {}
    
    def log_error(self, error_info: ErrorInfo):
        """Log error information"""
        self.error_history.append(error_info)
        self.error_counts[error_info.error_type] = self.error_counts.get(error_info.error_type, 0) + 1
        
        logger.error(f"Error [{error_info.severity.value}]: {error_info.message}")
        logger.error(f"Details: {error_info.details}")
        logger.error(f"Retry count: {error_info.retry_count}/{error_info.max_retries}")
        
        if error_info.recovery_suggestion:
            logger.info(f"Recovery suggestion: {error_info.recovery_suggestion}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "critical_errors": [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        }
    
    def handle_exception(self, func: Callable) -> Callable:
        """Decorator for robust exception handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    error_info = self._create_error_info(e, attempt)
                    self.log_error(error_info)
                    
                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error("Max retries reached. Giving up.")
            
            # If we get here, all retries failed
            return self._format_error_response(last_error)
        
        return wrapper
    
    def _create_error_info(self, error: Exception, attempt: int) -> ErrorInfo:
        """Create error information from exception"""
        error_type = type(error).__name__
        
        # Determine severity based on error type
        severity = ErrorSeverity.MEDIUM
        if isinstance(error, (ValueError, TypeError)):
            severity = ErrorSeverity.LOW
        elif isinstance(error, (MemoryError, OSError)):
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            severity = ErrorSeverity.CRITICAL
        
        # Create recovery suggestion
        recovery_suggestion = self._get_recovery_suggestion(error_type, error)
        
        return ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=str(error),
            details={
                "traceback": traceback.format_exc(),
                "args": str(args) if 'args' in locals() else "N/A",
                "kwargs": str(kwargs) if 'kwargs' in locals() else "N/A"
            },
            retry_count=attempt,
            max_retries=self.max_retries,
            recovery_suggestion=recovery_suggestion
        )
    
    def _get_recovery_suggestion(self, error_type: str, error: Exception) -> str:
        """Get recovery suggestion based on error type"""
        suggestions = {
            "ValueError": "Please check your input values and ensure they are within valid ranges.",
            "TypeError": "Please ensure you're using the correct data types for your inputs.",
            "MemoryError": "Try reducing batch size or model size. Close other applications to free memory.",
            "OSError": "Check file permissions and ensure the file path is correct.",
            "ConnectionError": "Check your internet connection and try again.",
            "TimeoutError": "The operation took too long. Try with smaller inputs or check your system resources.",
            "CUDAOutOfMemoryError": "GPU memory is full. Try reducing batch size or use CPU instead.",
            "ModuleNotFoundError": "Required module is not installed. Please install missing dependencies.",
            "ImportError": "There's an issue with module imports. Check your installation.",
            "FileNotFoundError": "The specified file was not found. Check the file path.",
            "PermissionError": "You don't have permission to access this resource.",
            "KeyboardInterrupt": "Operation was cancelled by user.",
            "SystemExit": "Application is shutting down."
        }
        
        return suggestions.get(error_type, "Please try again or contact support if the problem persists.")
    
    def _format_error_response(self, error: Exception) -> Tuple[str, Optional[Any], str]:
        """Format error response for Gradio"""
        error_message = f"⚠️ An error occurred: {str(error)}"
        
        # Add helpful context based on error type
        if isinstance(error, ValueError):
            error_message += "\n💡 Tip: Check your input values and ensure they are valid."
        elif isinstance(error, TypeError):
            error_message += "\n💡 Tip: Make sure you're using the correct data types."
        elif isinstance(error, MemoryError):
            error_message += "\n💡 Tip: Try reducing batch size or close other applications."
        elif isinstance(error, OSError):
            error_message += "\n💡 Tip: Check file permissions and paths."
        
        return error_message, None, "❌ Error occurred"


class InputSanitizer:
    """Input sanitization and cleaning"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove potentially dangerous characters (basic XSS prevention)
        dangerous_chars = ['<script>', '</script>', 'javascript:', 'onload=', 'onerror=']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text
    
    @staticmethod
    def sanitize_numeric(value: Union[int, float, str]) -> float:
        """Sanitize numeric input"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def sanitize_model_name(model_name: str) -> str:
        """Sanitize model name"""
        if not model_name:
            return "gpt2"
        
        # Only allow alphanumeric, hyphens, and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', model_name)
        return sanitized if sanitized else "gpt2"
    
    @staticmethod
    def sanitize_labels(labels: str) -> List[str]:
        """Sanitize label input"""
        if not labels:
            return []
        
        # Split by comma and clean each label
        label_list = [label.strip() for label in labels.split(',') if label.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in label_list:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        return unique_labels


class RobustGradioInterface:
    """Robust Gradio interface with advanced error handling and validation"""
    
    def __init__(self):
        self.validator = RobustInputValidator()
        self.error_handler = RobustErrorHandler()
        self.sanitizer = InputSanitizer()
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """Setup comprehensive validation rules"""
        # Text generation validation
        self.validator.add_validation_rule("prompt", ValidationRule(
            validation_type=ValidationType.TEXT,
            min_length=1,
            max_length=1000,
            required=True,
            error_message="❌ Please enter a prompt"
        ))
        
        self.validator.add_validation_rule("max_length", ValidationRule(
            validation_type=ValidationType.NUMERIC,
            min_value=10,
            max_value=1000,
            required=True,
            error_message="❌ Max length must be between 10 and 1000"
        ))
        
        self.validator.add_validation_rule("temperature", ValidationRule(
            validation_type=ValidationType.NUMERIC,
            min_value=0.1,
            max_value=2.0,
            required=True,
            error_message="❌ Temperature must be between 0.1 and 2.0"
        ))
        
        self.validator.add_validation_rule("model_name", ValidationRule(
            validation_type=ValidationType.MODEL,
            required=True,
            error_message="❌ Please select a valid model"
        ))
        
        # Sentiment analysis validation
        self.validator.add_validation_rule("sentiment_text", ValidationRule(
            validation_type=ValidationType.TEXT,
            min_length=10,
            max_length=2000,
            required=True,
            error_message="❌ Text must be between 10 and 2000 characters"
        ))
        
        # Classification validation
        self.validator.add_validation_rule("classification_text", ValidationRule(
            validation_type=ValidationType.TEXT,
            min_length=10,
            max_length=2000,
            required=True,
            error_message="❌ Text must be between 10 and 2000 characters"
        ))
        
        self.validator.add_validation_rule("labels", ValidationRule(
            validation_type=ValidationType.TEXT,
            min_length=3,
            max_length=500,
            required=True,
            error_message="❌ Please provide at least 2 labels separated by commas"
        ))
        
        # Training validation
        self.validator.add_validation_rule("training_data", ValidationRule(
            validation_type=ValidationType.TEXT,
            min_length=50,
            max_length=10000,
            required=True,
            error_message="❌ Training data must be between 50 and 10000 characters"
        ))
        
        self.validator.add_validation_rule("epochs", ValidationRule(
            validation_type=ValidationType.NUMERIC,
            min_value=1,
            max_value=100,
            required=True,
            error_message="❌ Epochs must be between 1 and 100"
        ))
        
        self.validator.add_validation_rule("learning_rate", ValidationRule(
            validation_type=ValidationType.NUMERIC,
            min_value=1e-6,
            max_value=1e-2,
            required=True,
            error_message="❌ Learning rate must be between 1e-6 and 1e-2"
        ))
        
        self.validator.add_validation_rule("batch_size", ValidationRule(
            validation_type=ValidationType.NUMERIC,
            min_value=1,
            max_value=64,
            required=True,
            error_message="❌ Batch size must be between 1 and 64"
        ))
    
    @RobustErrorHandler.handle_exception
    def robust_text_generation(self, prompt: str, max_length: int, temperature: float, model_name: str) -> Tuple[str, Optional[Any], str]:
        """Robust text generation with comprehensive validation"""
        # Sanitize inputs
        prompt = self.sanitizer.sanitize_text(prompt)
        max_length = int(self.sanitizer.sanitize_numeric(max_length))
        temperature = self.sanitizer.sanitize_numeric(temperature)
        model_name = self.sanitizer.sanitize_model_name(model_name)
        
        # Validate inputs
        validations = [
            self.validator.validate_input("prompt", prompt),
            self.validator.validate_input("max_length", max_length),
            self.validator.validate_input("temperature", temperature),
            self.validator.validate_input("model_name", model_name)
        ]
        
        # Check all validations
        for is_valid, message in validations:
            if not is_valid:
                return message, None, "❌ Validation Failed"
        
        # Perform text generation (placeholder for actual implementation)
        try:
            # Simulate text generation
            generated_text = f"Generated text based on: '{prompt[:50]}...' with length {max_length} and temperature {temperature} using {model_name}"
            
            return generated_text, None, "✅ Generation Complete"
        
        except Exception as e:
            raise e
    
    @RobustErrorHandler.handle_exception
    def robust_sentiment_analysis(self, text: str) -> Tuple[str, Optional[Any], str]:
        """Robust sentiment analysis with comprehensive validation"""
        # Sanitize input
        text = self.sanitizer.sanitize_text(text)
        
        # Validate input
        is_valid, message = self.validator.validate_input("sentiment_text", text)
        if not is_valid:
            return message, None, "❌ Validation Failed"
        
        # Perform sentiment analysis (placeholder for actual implementation)
        try:
            # Simulate sentiment analysis
            sentiment_result = {
                "label": "positive",
                "score": 0.85,
                "positive": 0.85,
                "neutral": 0.10,
                "negative": 0.05
            }
            
            result_text = f"🎭 **Sentiment Analysis Results**\n\n"
            result_text += f"**Overall Sentiment:** {sentiment_result['label']}\n"
            result_text += f"**Confidence:** {sentiment_result['score']:.3f}\n\n"
            result_text += f"**Analyzed Text:**\n{text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            result_text += f"📊 **Detailed Breakdown:**\n"
            result_text += f"• Positive: {sentiment_result['positive']:.3f}\n"
            result_text += f"• Neutral: {sentiment_result['neutral']:.3f}\n"
            result_text += f"• Negative: {sentiment_result['negative']:.3f}"
            
            return result_text, None, "✅ Analysis Complete"
        
        except Exception as e:
            raise e
    
    @RobustErrorHandler.handle_exception
    def robust_text_classification(self, text: str, labels: str) -> Tuple[str, Optional[Any], str]:
        """Robust text classification with comprehensive validation"""
        # Sanitize inputs
        text = self.sanitizer.sanitize_text(text)
        label_list = self.sanitizer.sanitize_labels(labels)
        
        # Validate inputs
        is_valid_text, message_text = self.validator.validate_input("classification_text", text)
        if not is_valid_text:
            return message_text, None, "❌ Validation Failed"
        
        is_valid_labels, message_labels = self.validator.validate_input("labels", labels)
        if not is_valid_labels:
            return message_labels, None, "❌ Validation Failed"
        
        if len(label_list) < 2:
            return "❌ At least 2 labels required", None, "❌ Insufficient Labels"
        
        # Perform classification (placeholder for actual implementation)
        try:
            # Simulate classification
            classification_result = {
                "label": label_list[0],
                "score": 0.92,
                "scores": {label: np.random.random() for label in label_list}
            }
            
            result_text = f"🏷️ **Text Classification Results**\n\n"
            result_text += f"**Best Match:** {classification_result['label']}\n"
            result_text += f"**Confidence:** {classification_result['score']:.3f}\n\n"
            result_text += f"**Analyzed Text:**\n{text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            result_text += f"📊 **All Classification Scores:**\n"
            for label, score in classification_result['scores'].items():
                emoji = "🥇" if label == classification_result['label'] else "📊"
                result_text += f"{emoji} {label}: {score:.3f}\n"
            
            return result_text, None, "✅ Classification Complete"
        
        except Exception as e:
            raise e
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        return self.error_handler.get_error_summary()
    
    def create_robust_interface(self) -> gr.Blocks:
        """Create robust Gradio interface with error handling"""
        
        with gr.Blocks(title="🤖 Robust NLP Platform", theme="default") as interface:
            
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
                <h1>🤖 Robust NLP Platform</h1>
                <p>Advanced error handling and input validation for reliable AI operations</p>
            </div>
            """)
            
            with gr.Tabs():
                
                # Text Generation Tab
                with gr.Tab("📝 Text Generation"):
                    gr.Markdown("### 🎨 Create Text with Robust Error Handling")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="🎯 Input Prompt",
                                placeholder="Enter your creative prompt here...",
                                lines=4
                            )
                            
                            with gr.Row():
                                max_length_input = gr.Slider(
                                    minimum=10, maximum=1000, value=100,
                                    step=10, label="📏 Max Length"
                                )
                                temperature_input = gr.Slider(
                                    minimum=0.1, maximum=2.0, value=0.7,
                                    step=0.1, label="🌡️ Temperature"
                                )
                            
                            model_name_input = gr.Dropdown(
                                choices=list(self.validator.valid_models.keys()),
                                value="gpt2", 
                                label="🤖 AI Model"
                            )
                            
                            generate_btn = gr.Button("🚀 Generate Text", variant="primary")
                        
                        with gr.Column(scale=2):
                            output_text = gr.Textbox(
                                label="✨ Generated Text",
                                lines=12,
                                interactive=False
                            )
                            status_output = gr.HTML(label="📈 Status")
                
                # Sentiment Analysis Tab
                with gr.Tab("🎭 Sentiment Analysis"):
                    gr.Markdown("### 🎭 Analyze Sentiment with Validation")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            sentiment_text_input = gr.Textbox(
                                label="📝 Text for Analysis",
                                placeholder="Enter text to analyze its emotional tone...",
                                lines=6
                            )
                            sentiment_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
                        
                        with gr.Column(scale=2):
                            sentiment_output = gr.Markdown(label="📊 Analysis Results")
                            sentiment_status = gr.HTML(label="📈 Status")
                
                # Text Classification Tab
                with gr.Tab("🏷️ Text Classification"):
                    gr.Markdown("### 🏷️ Classify Text with Robust Validation")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            classification_text_input = gr.Textbox(
                                label="📝 Text to Classify",
                                placeholder="Enter text to classify...",
                                lines=4
                            )
                            classification_labels_input = gr.Textbox(
                                label="🏷️ Labels (comma-separated)",
                                placeholder="positive, negative, neutral, urgent, casual",
                                lines=1
                            )
                            classification_btn = gr.Button("🏷️ Classify Text", variant="primary")
                        
                        with gr.Column(scale=2):
                            classification_output = gr.Markdown(label="📊 Classification Results")
                            classification_status = gr.HTML(label="📈 Status")
                
                # Error Monitoring Tab
                with gr.Tab("📊 Error Monitoring"):
                    gr.Markdown("### 📊 Monitor System Errors and Performance")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            error_summary_btn = gr.Button("🔄 Refresh Error Summary", variant="primary")
                        
                        with gr.Column(scale=2):
                            error_summary_output = gr.JSON(label="📊 Error Summary")
                            error_status = gr.HTML(label="📈 Status")
            
            # Event handlers
            generate_btn.click(
                fn=self.robust_text_generation,
                inputs=[prompt_input, max_length_input, temperature_input, model_name_input],
                outputs=[output_text, None, status_output]
            )
            
            sentiment_btn.click(
                fn=self.robust_sentiment_analysis,
                inputs=[sentiment_text_input],
                outputs=[sentiment_output, None, sentiment_status]
            )
            
            classification_btn.click(
                fn=self.robust_text_classification,
                inputs=[classification_text_input, classification_labels_input],
                outputs=[classification_output, None, classification_status]
            )
            
            error_summary_btn.click(
                fn=self.get_error_summary,
                inputs=[],
                outputs=[error_summary_output]
            )
        
        return interface


def create_robust_gradio_app():
    """Create and launch the robust Gradio app"""
    robust_interface = RobustGradioInterface()
    interface = robust_interface.create_robust_interface()
    return interface


if __name__ == "__main__":
    # Create and launch the robust app
    app = create_robust_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )




