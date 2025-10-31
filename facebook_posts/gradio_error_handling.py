from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import json
import time
import traceback
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
    from deep_learning_framework import DeepLearningFramework, FrameworkConfig, TaskType
    from evaluation_metrics import EvaluationManager, MetricConfig, MetricType
    from gradient_clipping_nan_handling import NumericalStabilityManager
    from early_stopping_scheduling import TrainingManager
    from efficient_data_loading import EfficientDataLoader
    from data_splitting_validation import DataSplitter
    from training_evaluation import TrainingManager as TrainingEvalManager
    from diffusion_models import DiffusionModel, DiffusionConfig
    from advanced_transformers import AdvancedTransformerModel
    from llm_training import AdvancedLLMTrainer
    from model_finetuning import ModelFineTuner
    from custom_modules import AdvancedNeuralNetwork
    from weight_initialization import AdvancedWeightInitializer
    from normalization_techniques import AdvancedLayerNorm
    from loss_functions import AdvancedCrossEntropyLoss
    from optimization_algorithms import AdvancedAdamW
    from attention_mechanisms import MultiHeadAttention
    from tokenization_sequence import AdvancedTokenizer
    from framework_utils import MetricsTracker, ModelAnalyzer, PerformanceMonitor
    from deep_learning_integration import DeepLearningIntegration, IntegrationConfig, IntegrationType, ComponentType
                import signal
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Gradio Error Handling and Input Validation
Comprehensive error handling and input validation for Gradio applications.
"""

warnings.filterwarnings('ignore')

# Import our custom modules
try:
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


class ErrorType(Enum):
    """Types of errors that can occur."""
    INPUT_VALIDATION = "input_validation"
    MODEL_ERROR = "model_error"
    PROCESSING_ERROR = "processing_error"
    VISUALIZATION_ERROR = "visualization_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"


class ValidationType(Enum):
    """Types of input validation."""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    SIZE_CHECK = "size_check"
    CONTENT_CHECK = "content_check"


@dataclass
class ValidationRule:
    """Validation rule configuration."""
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


@dataclass
class ErrorConfig:
    """Error handling configuration."""
    show_traceback: bool = False
    log_errors: bool = True
    return_partial_results: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    graceful_degradation: bool = True


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self, validation_rules: List[ValidationRule]):
        
    """__init__ function."""
self.validation_rules = validation_rules
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
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


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, config: ErrorConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        self.error_counts = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for error handling."""
        if self.config.log_errors:
            logging.basicConfig(
                level=logging.ERROR,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('gradio_errors.log'),
                    logging.StreamHandler()
                ]
            )
        return logging.getLogger(__name__)
    
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


class GradioErrorHandler:
    """Gradio-specific error handling wrapper."""
    
    def __init__(self, validator: InputValidator, error_handler: ErrorHandler):
        
    """__init__ function."""
self.validator = validator
        self.error_handler = error_handler
    
    def safe_function_wrapper(self, func: Callable, *args, **kwargs) -> Callable:
        """Wrap function with error handling."""
        def wrapper(*func_args, **func_kwargs) -> Any:
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
                
                def timeout_handler(signum, frame) -> Any:
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
    
    def _extract_inputs(self, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        """Extract inputs for validation."""
        inputs = {}
        
        # Add positional arguments
        for i, arg in enumerate(args):
            inputs[f"arg_{i}"] = arg
        
        # Add keyword arguments
        inputs.update(kwargs)
        
        return inputs
    
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
    
    def _format_error_response(self, error_response: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        """Format error response for Gradio."""
        error_message = error_response['message']
        
        if self.error_handler.config.return_partial_results:
            return error_message, None
        else:
            return error_message, None


class RobustGradioInterface:
    """Robust Gradio interface with comprehensive error handling."""
    
    def __init__(self) -> Any:
        # Setup validation rules
        self.validation_rules = [
            ValidationRule(
                validation_type=ValidationType.REQUIRED,
                field_name="image",
                required=True
            ),
            ValidationRule(
                validation_type=ValidationType.TYPE_CHECK,
                field_name="text_input",
                allowed_types=[str]
            ),
            ValidationRule(
                validation_type=ValidationType.RANGE_CHECK,
                field_name="numeric_input",
                min_value=0.0,
                max_value=100.0
            )
        ]
        
        # Setup error handling
        error_config = ErrorConfig(
            show_traceback=False,
            log_errors=True,
            return_partial_results=True,
            timeout_seconds=30,
            max_retries=3,
            graceful_degradation=True
        )
        
        self.validator = InputValidator(self.validation_rules)
        self.error_handler = ErrorHandler(error_config)
        self.gradio_error_handler = GradioErrorHandler(self.validator, self.error_handler)
    
    def create_robust_classification_demo(self) -> Any:
        """Create robust classification demo with error handling."""
        
        @self.gradio_error_handler.safe_function_wrapper
        def classify_digit_with_validation(image, confidence_threshold=0.5) -> Any:
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
        
        # Create interface with error handling
        demo = gr.Interface(
            fn=classify_digit_with_validation,
            inputs=[
                gr.Image(type="pil", label="Upload Handwritten Digit"),
                gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, 
                         label="Confidence Threshold")
            ],
            outputs=[
                gr.Textbox(label="Prediction"),
                gr.Image(label="Visualization")
            ],
            title="Robust Handwritten Digit Classification",
            description="Upload a handwritten digit image to classify it (0-9) with error handling",
            examples=[
                ["examples/digit_0.png", 0.5],
                ["examples/digit_1.png", 0.7],
                ["examples/digit_2.png", 0.6]
            ],
            allow_flagging="never"
        )
        
        return demo
    
    def create_robust_regression_demo(self) -> Any:
        """Create robust regression demo with error handling."""
        
        @self.gradio_error_handler.safe_function_wrapper
        def predict_value_with_validation(features, model_type="linear") -> Any:
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
        
        # Create interface
        demo = gr.Interface(
            fn=predict_value_with_validation,
            inputs=[
                gr.Textbox(
                    label="Input Features",
                    placeholder="Enter 10 comma-separated values (e.g., 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0)"
                ),
                gr.Dropdown(
                    choices=["linear", "polynomial", "neural"],
                    value="linear",
                    label="Model Type"
                )
            ],
            outputs=[
                gr.Textbox(label="Prediction"),
                gr.Image(label="Visualization")
            ],
            title="Robust Regression Prediction",
            description="Enter 10 feature values to predict a continuous output with error handling",
            examples=[
                ["1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0", "linear"],
                ["0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0", "polynomial"],
                ["-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5", "neural"]
            ],
            allow_flagging="never"
        )
        
        return demo
    
    def create_robust_evaluation_demo(self) -> Any:
        """Create robust evaluation demo with error handling."""
        
        @self.gradio_error_handler.safe_function_wrapper
        def evaluate_predictions_with_validation(y_true, y_pred, task_type="classification") -> Any:
            """Evaluate predictions with comprehensive validation."""
            # Validate inputs
            if not y_true or not y_pred:
                raise ValueError("Both true and predicted values are required")
            
            # Validate task type
            if task_type not in ["classification", "regression"]:
                raise ValueError("Task type must be 'classification' or 'regression'")
            
            try:
                # Parse inputs
                true_values = [float(x.strip()) for x in y_true.split(',')]
                pred_values = [float(x.strip()) for x in y_pred.split(',')]
                
                if len(true_values) != len(pred_values):
                    raise ValueError("Number of true and predicted values must match")
                
                if len(true_values) == 0:
                    raise ValueError("At least one value is required")
                
                # Convert to numpy arrays
                y_true_array = np.array(true_values)
                y_pred_array = np.array(pred_values)
                
                # Calculate metrics
                if task_type == "classification":
                    y_true_classes = y_true_array.astype(int)
                    y_pred_classes = y_pred_array.astype(int)
                    
                    accuracy = np.mean(y_true_classes == y_pred_classes)
                    
                    metrics = {}
                    for class_id in np.unique(y_true_classes):
                        tp = np.sum((y_true_classes == class_id) & (y_pred_classes == class_id))
                        fp = np.sum((y_true_classes != class_id) & (y_pred_classes == class_id))
                        fn = np.sum((y_true_classes == class_id) & (y_pred_classes != class_id))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        metrics[f'Class_{class_id}'] = {
                            'Precision': precision,
                            'Recall': recall,
                            'F1': f1
                        }
                    
                    metrics['Overall'] = {'Accuracy': accuracy}
                    
                else:  # regression
                    mse = np.mean((y_true_array - y_pred_array) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_true_array - y_pred_array))
                    r2 = 1 - np.sum((y_true_array - y_pred_array) ** 2) / np.sum((y_true_array - np.mean(y_true_array)) ** 2)
                    
                    metrics = {
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R²': r2
                    }
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot
                ax1.scatter(y_true_array, y_pred_array, alpha=0.6)
                ax1.plot([y_true_array.min(), y_true_array.max()], 
                        [y_true_array.min(), y_true_array.max()], 'r--', lw=2)
                ax1.set_xlabel('True Values')
                ax1.set_ylabel('Predicted Values')
                ax1.set_title('True vs Predicted Values')
                ax1.grid(True)
                
                # Metrics bar plot
                if task_type == "classification":
                    metric_names = list(metrics['Overall'].keys())
                    metric_values = list(metrics['Overall'].values())
                else:
                    metric_names = list(metrics.keys())
                    metric_values = list(metrics.values())
                
                ax2.bar(metric_names, metric_values, color='skyblue', edgecolor='black')
                ax2.set_title('Evaluation Metrics')
                ax2.set_ylabel('Score')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                # Format results
                results_text = f"Task Type: {task_type}\n\n"
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        results_text += f"{metric_name}:\n"
                        for sub_metric, value in metric_value.items():
                            results_text += f"  {sub_metric}: {value:.4f}\n"
                    else:
                        results_text += f"{metric_name}: {metric_value:.4f}\n"
                
                return results_text, buf.getvalue()
                
            except Exception as e:
                raise RuntimeError(f"Evaluation failed: {str(e)}")
        
        # Create interface
        demo = gr.Interface(
            fn=evaluate_predictions_with_validation,
            inputs=[
                gr.Textbox(label="True Values", placeholder="Enter comma-separated true values"),
                gr.Textbox(label="Predicted Values", placeholder="Enter comma-separated predicted values"),
                gr.Dropdown(choices=["classification", "regression"], value="classification", label="Task Type")
            ],
            outputs=[
                gr.Textbox(label="Evaluation Results"),
                gr.Image(label="Visualization")
            ],
            title="Robust Model Evaluation",
            description="Evaluate model predictions with various metrics and error handling",
            examples=[
                ["0,1,0,1,0,1,0,1,0,1", "0,1,0,1,0,1,0,1,0,1", "classification"],
                ["1.0,2.0,3.0,4.0,5.0", "1.1,1.9,3.1,3.9,5.1", "regression"]
            ],
            allow_flagging="never"
        )
        
        return demo
    
    def create_all_robust_demos(self) -> Any:
        """Create all robust demos with error handling."""
        demos = [
            ("Robust Classification", self.create_robust_classification_demo()),
            ("Robust Regression", self.create_robust_regression_demo()),
            ("Robust Evaluation", self.create_robust_evaluation_demo())
        ]
        
        # Create tabbed interface
        with gr.Blocks(title="Robust Deep Learning Demos") as demo:
            gr.Markdown("# Robust Deep Learning Interactive Demos")
            gr.Markdown("Explore deep learning capabilities with comprehensive error handling and input validation")
            
            with gr.Tabs():
                for name, demo_interface in demos:
                    with gr.TabItem(name):
                        demo_interface.render()
        
        return demo


def launch_robust_demos():
    """Launch robust demos with error handling."""
    print("Launching Robust Deep Learning Interactive Demos...")
    
    # Create robust demo manager
    robust_manager = RobustGradioInterface()
    
    # Create combined demo
    combined_demo = robust_manager.create_all_robust_demos()
    
    # Launch the demo
    combined_demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )


if __name__ == "__main__":
    # Launch the robust demos
    launch_robust_demos() 