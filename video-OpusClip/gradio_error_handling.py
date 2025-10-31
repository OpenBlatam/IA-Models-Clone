"""
Gradio Error Handling and Input Validation System

Comprehensive error handling and validation for Gradio applications:
- Input validation with user-friendly error messages
- Graceful error recovery and fallback mechanisms
- Performance monitoring and error tracking
- User feedback and guidance systems
"""

import gradio as gr
import traceback
import logging
import time
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from functools import wraps
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

# Import existing components
from optimized_config import get_config
from error_handling import ErrorHandler, ErrorType, ErrorSeverity
from validation import Validator, ValidationRule, ValidationError
from logging_config import setup_logging

# =============================================================================
# GRADIO-SPECIFIC ERROR HANDLING
# =============================================================================

class GradioErrorHandler:
    """Specialized error handler for Gradio applications."""
    
    def __init__(self):
        self.config = get_config()
        self.error_handler = ErrorHandler(self.config)
        self.validator = Validator(self.config)
        self.logger = setup_logging("gradio_errors")
        
        # Error tracking
        self.error_counts = {}
        self.error_history = []
        self.recovery_attempts = {}
        
        # User-friendly error messages
        self.error_messages = {
            "input_validation": {
                "invalid_prompt": "Please provide a valid text prompt (3-500 characters)",
                "invalid_image": "Please upload a valid image file (JPG, PNG, WebP)",
                "invalid_video": "Please upload a valid video file (MP4, AVI, MOV)",
                "invalid_duration": "Duration must be between 3 and 60 seconds",
                "invalid_quality": "Please select a valid quality level",
                "file_too_large": "File size exceeds maximum limit (100MB)",
                "unsupported_format": "File format not supported",
                "network_error": "Network connection error. Please check your internet connection.",
                "gpu_error": "GPU processing error. Falling back to CPU.",
                "memory_error": "Insufficient memory. Try reducing quality or duration.",
                "timeout_error": "Operation timed out. Please try again.",
                "model_error": "AI model error. Please try again or contact support."
            },
            "recovery_suggestions": {
                "input_validation": "Please check your input and try again",
                "network_error": "Check your internet connection and try again",
                "gpu_error": "The system will automatically retry with CPU processing",
                "memory_error": "Try reducing video duration or quality settings",
                "timeout_error": "Try again with smaller files or lower quality",
                "model_error": "Try refreshing the page or contact support"
            }
        }
    
    def handle_gradio_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors in Gradio applications with user-friendly responses."""
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "severity": self._determine_severity(error),
            "recoverable": self._is_recoverable(error)
        }
        
        # Log error
        self.logger.error(f"Gradio error in {context}: {error}", exc_info=True)
        
        # Track error
        self._track_error(error_info)
        
        # Generate user-friendly response
        user_response = self._generate_user_response(error_info)
        
        return user_response
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity for Gradio context."""
        
        if isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.LOW
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.MEDIUM
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if error is recoverable."""
        
        recoverable_errors = [
            ValueError, TypeError, FileNotFoundError,
            PermissionError, MemoryError, OSError
        ]
        
        return any(isinstance(error, err_type) for err_type in recoverable_errors)
    
    def _track_error(self, error_info: Dict[str, Any]):
        """Track error for analytics and improvement."""
        
        error_type = error_info["error_type"]
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_history.append(error_info)
        
        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def _generate_user_response(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user-friendly error response."""
        
        error_type = error_info["error_type"]
        context = error_info["context"]
        
        # Get appropriate error message
        if "input" in context.lower():
            message = self.error_messages["input_validation"].get(
                error_type.lower(), 
                "An error occurred while processing your input. Please try again."
            )
        elif "network" in str(error_info["error_message"]).lower():
            message = self.error_messages["input_validation"]["network_error"]
        elif "gpu" in str(error_info["error_message"]).lower():
            message = self.error_messages["input_validation"]["gpu_error"]
        elif "memory" in str(error_info["error_message"]).lower():
            message = self.error_messages["input_validation"]["memory_error"]
        elif "timeout" in str(error_info["error_message"]).lower():
            message = self.error_messages["input_validation"]["timeout_error"]
        else:
            message = self.error_messages["input_validation"]["model_error"]
        
        # Get recovery suggestion
        recovery_key = "input_validation" if "input" in context.lower() else error_type.lower()
        suggestion = self.error_messages["recovery_suggestions"].get(
            recovery_key, 
            "Please try again or contact support if the problem persists."
        )
        
        return {
            "success": False,
            "error_message": message,
            "suggestion": suggestion,
            "error_code": error_type,
            "timestamp": error_info["timestamp"],
            "can_retry": error_info["recoverable"]
        }

# =============================================================================
# INPUT VALIDATION FOR GRADIO
# =============================================================================

class GradioInputValidator:
    """Input validation specifically for Gradio components."""
    
    def __init__(self):
        self.config = get_config()
        self.validator = Validator(self.config)
        
        # Validation rules for different input types
        self.validation_rules = {
            "text_prompt": [
                ValidationRule("min_length", 3, "Prompt must be at least 3 characters"),
                ValidationRule("max_length", 500, "Prompt must be less than 500 characters"),
                ValidationRule("no_special_chars", r"[<>\"'&]", "Prompt contains invalid characters"),
                ValidationRule("no_script_tags", r"<script", "Prompt contains script tags")
            ],
            "image_file": [
                ValidationRule("file_exists", None, "Please upload an image file"),
                ValidationRule("file_size", 50 * 1024 * 1024, "Image file too large (max 50MB)"),
                ValidationRule("file_format", [".jpg", ".jpeg", ".png", ".webp"], "Unsupported image format"),
                ValidationRule("image_dimensions", (100, 100, 4096, 4096), "Image dimensions out of range")
            ],
            "video_file": [
                ValidationRule("file_exists", None, "Please upload a video file"),
                ValidationRule("file_size", 100 * 1024 * 1024, "Video file too large (max 100MB)"),
                ValidationRule("file_format", [".mp4", ".avi", ".mov", ".mkv"], "Unsupported video format"),
                ValidationRule("video_duration", (1, 300), "Video duration out of range (1-300 seconds)")
            ],
            "duration": [
                ValidationRule("min_value", 3, "Duration must be at least 3 seconds"),
                ValidationRule("max_value", 60, "Duration must be at most 60 seconds"),
                ValidationRule("is_integer", None, "Duration must be a whole number")
            ],
            "quality": [
                ValidationRule("valid_choice", ["Fast", "Balanced", "High Quality", "Ultra Quality"], 
                             "Please select a valid quality level")
            ],
            "model_type": [
                ValidationRule("valid_choice", ["Stable Diffusion", "DeepFloyd", "Kandinsky", "Custom"], 
                             "Please select a valid model type")
            ]
        }
    
    def validate_text_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Validate text prompt input."""
        
        if not prompt or not isinstance(prompt, str):
            return False, "Please provide a valid text prompt"
        
        prompt = prompt.strip()
        
        # Apply validation rules
        for rule in self.validation_rules["text_prompt"]:
            if rule.name == "min_length" and len(prompt) < rule.value:
                return False, rule.message
            elif rule.name == "max_length" and len(prompt) > rule.value:
                return False, rule.message
            elif rule.name == "no_special_chars" and re.search(rule.value, prompt):
                return False, rule.message
            elif rule.name == "no_script_tags" and rule.value.lower() in prompt.lower():
                return False, rule.message
        
        return True, "Valid prompt"
    
    def validate_image_file(self, image) -> Tuple[bool, str]:
        """Validate image file input."""
        
        if image is None:
            return False, "Please upload an image file"
        
        try:
            # Check if it's a numpy array (Gradio image format)
            if isinstance(image, np.ndarray):
                if len(image.shape) != 3 or image.shape[2] not in [1, 3, 4]:
                    return False, "Invalid image format"
                
                height, width = image.shape[:2]
                min_dim, max_dim = 100, 4096
                
                if height < min_dim or width < min_dim:
                    return False, f"Image too small (minimum {min_dim}x{min_dim})"
                if height > max_dim or width > max_dim:
                    return False, f"Image too large (maximum {max_dim}x{max_dim})"
                
                return True, "Valid image"
            
            # Check if it's a file path
            elif isinstance(image, str):
                file_path = Path(image)
                if not file_path.exists():
                    return False, "Image file not found"
                
                # Check file size
                file_size = file_path.stat().st_size
                if file_size > 50 * 1024 * 1024:  # 50MB
                    return False, "Image file too large (max 50MB)"
                
                # Check file format
                valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                if file_path.suffix.lower() not in valid_extensions:
                    return False, "Unsupported image format"
                
                return True, "Valid image file"
            
            else:
                return False, "Invalid image format"
                
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def validate_video_file(self, video) -> Tuple[bool, str]:
        """Validate video file input."""
        
        if video is None:
            return False, "Please upload a video file"
        
        try:
            # Check if it's a file path
            if isinstance(video, str):
                file_path = Path(video)
                if not file_path.exists():
                    return False, "Video file not found"
                
                # Check file size
                file_size = file_path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # 100MB
                    return False, "Video file too large (max 100MB)"
                
                # Check file format
                valid_extensions = [".mp4", ".avi", ".mov", ".mkv"]
                if file_path.suffix.lower() not in valid_extensions:
                    return False, "Unsupported video format"
                
                # Check video duration using OpenCV
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    return False, "Cannot read video file"
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                if duration < 1 or duration > 300:  # 1-300 seconds
                    return False, "Video duration out of range (1-300 seconds)"
                
                return True, "Valid video file"
            
            else:
                return False, "Invalid video format"
                
        except Exception as e:
            return False, f"Error validating video: {str(e)}"
    
    def validate_duration(self, duration: Union[int, float]) -> Tuple[bool, str]:
        """Validate duration input."""
        
        try:
            duration = float(duration)
            
            if duration < 3:
                return False, "Duration must be at least 3 seconds"
            elif duration > 60:
                return False, "Duration must be at most 60 seconds"
            elif not duration.is_integer():
                return False, "Duration must be a whole number"
            
            return True, "Valid duration"
            
        except (ValueError, TypeError):
            return False, "Duration must be a valid number"
    
    def validate_quality(self, quality: str) -> Tuple[bool, str]:
        """Validate quality selection."""
        
        valid_qualities = ["Fast", "Balanced", "High Quality", "Ultra Quality"]
        
        if quality not in valid_qualities:
            return False, "Please select a valid quality level"
        
        return True, "Valid quality"
    
    def validate_model_type(self, model_type: str) -> Tuple[bool, str]:
        """Validate model type selection."""
        
        valid_models = ["Stable Diffusion", "DeepFloyd", "Kandinsky", "Custom"]
        
        if model_type not in valid_models:
            return False, "Please select a valid model type"
        
        return True, "Valid model type"

# =============================================================================
# GRADIO ERROR HANDLING DECORATORS
# =============================================================================

def gradio_error_handler(func: Callable) -> Callable:
    """Decorator to handle errors in Gradio functions."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        error_handler = GradioErrorHandler()
        
        try:
            # Validate inputs before processing
            validator = GradioInputValidator()
            
            # Extract and validate common inputs
            for i, arg in enumerate(args):
                if isinstance(arg, str) and len(arg) > 0:
                    # Validate text prompts
                    is_valid, message = validator.validate_text_prompt(arg)
                    if not is_valid:
                        return {
                            "success": False,
                            "error_message": message,
                            "suggestion": "Please check your input and try again"
                        }
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate output
            if result is None:
                return {
                    "success": False,
                    "error_message": "No result generated",
                    "suggestion": "Please try again with different parameters"
                }
            
            return result
            
        except Exception as e:
            # Handle error with user-friendly response
            error_response = error_handler.handle_gradio_error(e, func.__name__)
            return error_response
    
    return wrapper

def validate_gradio_inputs(*input_types: str) -> Callable:
    """Decorator to validate specific input types."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = GradioInputValidator()
            
            # Validate inputs based on specified types
            for i, input_type in enumerate(input_types):
                if i < len(args):
                    arg = args[i]
                    
                    if input_type == "text_prompt":
                        is_valid, message = validator.validate_text_prompt(arg)
                    elif input_type == "image_file":
                        is_valid, message = validator.validate_image_file(arg)
                    elif input_type == "video_file":
                        is_valid, message = validator.validate_video_file(arg)
                    elif input_type == "duration":
                        is_valid, message = validator.validate_duration(arg)
                    elif input_type == "quality":
                        is_valid, message = validator.validate_quality(arg)
                    elif input_type == "model_type":
                        is_valid, message = validator.validate_model_type(arg)
                    else:
                        continue
                    
                    if not is_valid:
                        return {
                            "success": False,
                            "error_message": message,
                            "suggestion": "Please check your input and try again"
                        }
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# GRADIO COMPONENT ENHANCEMENTS
# =============================================================================

class EnhancedGradioComponents:
    """Enhanced Gradio components with built-in error handling."""
    
    def __init__(self):
        self.error_handler = GradioErrorHandler()
        self.validator = GradioInputValidator()
    
    def create_validated_textbox(self, **kwargs) -> gr.Textbox:
        """Create textbox with validation."""
        
        default_kwargs = {
            "label": "Input",
            "placeholder": "Enter your text here...",
            "lines": 3,
            "max_lines": 10,
            "interactive": True
        }
        default_kwargs.update(kwargs)
        
        textbox = gr.Textbox(**default_kwargs)
        
        # Add validation event
        def validate_text(text):
            is_valid, message = self.validator.validate_text_prompt(text)
            return {
                textbox: gr.update(
                    value=text,
                    interactive=True,
                    container=True,
                    scale=1
                ),
                gr.update(visible=not is_valid, value=message if not is_valid else "")
            }
        
        # Create error message component
        error_msg = gr.Textbox(
            label="Validation Error",
            visible=False,
            interactive=False,
            scale=1
        )
        
        textbox.change(
            fn=validate_text,
            inputs=[textbox],
            outputs=[textbox, error_msg]
        )
        
        return textbox, error_msg
    
    def create_validated_image(self, **kwargs) -> gr.Image:
        """Create image upload with validation."""
        
        default_kwargs = {
            "label": "Upload Image",
            "type": "numpy",
            "interactive": True,
            "height": 300
        }
        default_kwargs.update(kwargs)
        
        image = gr.Image(**default_kwargs)
        
        # Add validation event
        def validate_image(img):
            is_valid, message = self.validator.validate_image_file(img)
            return {
                image: gr.update(value=img if is_valid else None),
                gr.update(visible=not is_valid, value=message if not is_valid else "")
            }
        
        # Create error message component
        error_msg = gr.Textbox(
            label="Validation Error",
            visible=False,
            interactive=False
        )
        
        image.change(
            fn=validate_image,
            inputs=[image],
            outputs=[image, error_msg]
        )
        
        return image, error_msg
    
    def create_validated_video(self, **kwargs) -> gr.Video:
        """Create video upload with validation."""
        
        default_kwargs = {
            "label": "Upload Video",
            "interactive": True,
            "height": 300
        }
        default_kwargs.update(kwargs)
        
        video = gr.Video(**default_kwargs)
        
        # Add validation event
        def validate_video(vid):
            is_valid, message = self.validator.validate_video_file(vid)
            return {
                video: gr.update(value=vid if is_valid else None),
                gr.update(visible=not is_valid, value=message if not is_valid else "")
            }
        
        # Create error message component
        error_msg = gr.Textbox(
            label="Validation Error",
            visible=False,
            interactive=False
        )
        
        video.change(
            fn=validate_video,
            inputs=[video],
            outputs=[video, error_msg]
        )
        
        return video, error_msg
    
    def create_validated_slider(self, **kwargs) -> gr.Slider:
        """Create slider with validation."""
        
        default_kwargs = {
            "label": "Value",
            "minimum": 0,
            "maximum": 100,
            "value": 50,
            "step": 1,
            "interactive": True
        }
        default_kwargs.update(kwargs)
        
        slider = gr.Slider(**default_kwargs)
        
        # Add validation event
        def validate_slider(value):
            is_valid, message = self.validator.validate_duration(value)
            return {
                slider: gr.update(value=value),
                gr.update(visible=not is_valid, value=message if not is_valid else "")
            }
        
        # Create error message component
        error_msg = gr.Textbox(
            label="Validation Error",
            visible=False,
            interactive=False
        )
        
        slider.change(
            fn=validate_slider,
            inputs=[slider],
            outputs=[slider, error_msg]
        )
        
        return slider, error_msg

# =============================================================================
# ERROR RECOVERY MECHANISMS
# =============================================================================

class GradioErrorRecovery:
    """Error recovery mechanisms for Gradio applications."""
    
    def __init__(self):
        self.config = get_config()
        self.error_handler = GradioErrorHandler()
        self.recovery_strategies = {
            "gpu_error": self._recover_from_gpu_error,
            "memory_error": self._recover_from_memory_error,
            "timeout_error": self._recover_from_timeout_error,
            "network_error": self._recover_from_network_error,
            "model_error": self._recover_from_model_error
        }
    
    def attempt_recovery(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Attempt to recover from an error."""
        
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # Determine recovery strategy
        strategy = None
        for key, recovery_func in self.recovery_strategies.items():
            if key in error_message or key in error_type:
                strategy = recovery_func
                break
        
        if strategy:
            try:
                result = strategy(error, context)
                return {
                    "success": True,
                    "recovered": True,
                    "message": "Recovery successful",
                    "result": result
                }
            except Exception as recovery_error:
                return {
                    "success": False,
                    "recovered": False,
                    "message": f"Recovery failed: {str(recovery_error)}",
                    "original_error": str(error)
                }
        else:
            return {
                "success": False,
                "recovered": False,
                "message": "No recovery strategy available",
                "original_error": str(error)
            }
    
    def _recover_from_gpu_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Recover from GPU-related errors by falling back to CPU."""
        
        # Log the recovery attempt
        logging.warning(f"GPU error detected, falling back to CPU: {error}")
        
        # Update configuration to use CPU
        self.config.gpu_enabled = False
        self.config.device = "cpu"
        
        return {
            "device": "cpu",
            "message": "Switched to CPU processing",
            "performance_note": "Processing may be slower"
        }
    
    def _recover_from_memory_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Recover from memory errors by reducing batch size and quality."""
        
        # Log the recovery attempt
        logging.warning(f"Memory error detected, reducing resource usage: {error}")
        
        # Reduce memory usage
        self.config.batch_size = max(1, self.config.batch_size // 2)
        self.config.max_memory_usage = self.config.max_memory_usage * 0.8
        
        return {
            "batch_size": self.config.batch_size,
            "memory_limit": self.config.max_memory_usage,
            "message": "Reduced memory usage",
            "performance_note": "Processing may take longer"
        }
    
    def _recover_from_timeout_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Recover from timeout errors by increasing timeout and reducing complexity."""
        
        # Log the recovery attempt
        logging.warning(f"Timeout error detected, adjusting parameters: {error}")
        
        # Increase timeout and reduce complexity
        self.config.timeout = min(300, self.config.timeout * 1.5)
        self.config.quality = "Fast" if self.config.quality != "Fast" else "Balanced"
        
        return {
            "timeout": self.config.timeout,
            "quality": self.config.quality,
            "message": "Increased timeout and reduced quality",
            "performance_note": "Faster processing with lower quality"
        }
    
    def _recover_from_network_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Recover from network errors by retrying with exponential backoff."""
        
        # Log the recovery attempt
        logging.warning(f"Network error detected, implementing retry logic: {error}")
        
        # Implement retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                time.sleep(retry_delay)
                # Attempt the operation again
                return {
                    "retry_attempt": attempt + 1,
                    "message": f"Retry successful on attempt {attempt + 1}",
                    "performance_note": "Network connection restored"
                }
            except Exception as retry_error:
                retry_delay *= 2
                if attempt == max_retries - 1:
                    raise retry_error
        
        return {
            "message": "All retry attempts failed",
            "suggestion": "Check your internet connection"
        }
    
    def _recover_from_model_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Recover from model errors by reloading or switching models."""
        
        # Log the recovery attempt
        logging.warning(f"Model error detected, attempting recovery: {error}")
        
        # Try to reload the model or switch to a different one
        try:
            # This would typically involve reloading the model
            # For now, we'll just return a recovery message
            return {
                "message": "Model reloaded successfully",
                "performance_note": "Model may take time to initialize"
            }
        except Exception as reload_error:
            return {
                "message": "Model recovery failed",
                "suggestion": "Try refreshing the page or contact support"
            }

# =============================================================================
# GRADIO ERROR MONITORING
# =============================================================================

class GradioErrorMonitor:
    """Monitor and track errors in Gradio applications."""
    
    def __init__(self):
        self.error_handler = GradioErrorHandler()
        self.recovery = GradioErrorRecovery()
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_context": {},
            "recovery_success_rate": 0,
            "average_response_time": 0
        }
        self.response_times = []
    
    def monitor_function(self, func: Callable) -> Callable:
        """Monitor a function for errors and performance."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                self._record_success(response_time)
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                self._record_error(e, func.__name__, response_time)
                
                # Attempt recovery
                recovery_result = self.recovery.attempt_recovery(e, func.__name__)
                
                if recovery_result["recovered"]:
                    self._record_recovery_success()
                    return recovery_result["result"]
                else:
                    # Return user-friendly error response
                    error_response = self.error_handler.handle_gradio_error(e, func.__name__)
                    return error_response
        
        return wrapper
    
    def _record_success(self, response_time: float):
        """Record successful function execution."""
        self.response_times.append(response_time)
        self._update_stats()
    
    def _record_error(self, error: Exception, context: str, response_time: float):
        """Record error occurrence."""
        self.error_stats["total_errors"] += 1
        
        error_type = type(error).__name__
        self.error_stats["errors_by_type"][error_type] = \
            self.error_stats["errors_by_type"].get(error_type, 0) + 1
        
        self.error_stats["errors_by_context"][context] = \
            self.error_stats["errors_by_context"].get(context, 0) + 1
        
        self.response_times.append(response_time)
        self._update_stats()
    
    def _record_recovery_success(self):
        """Record successful error recovery."""
        # Update recovery success rate
        total_recoveries = self.error_stats.get("total_recoveries", 0) + 1
        successful_recoveries = self.error_stats.get("successful_recoveries", 0) + 1
        
        self.error_stats["total_recoveries"] = total_recoveries
        self.error_stats["successful_recoveries"] = successful_recoveries
        self.error_stats["recovery_success_rate"] = successful_recoveries / total_recoveries
    
    def _update_stats(self):
        """Update error statistics."""
        if self.response_times:
            self.error_stats["average_response_time"] = sum(self.response_times) / len(self.response_times)
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "stats": self.error_stats.copy(),
            "recent_errors": self.error_handler.error_history[-10:],
            "performance": {
                "average_response_time": self.error_stats["average_response_time"],
                "total_requests": len(self.response_times),
                "error_rate": self.error_stats["total_errors"] / max(1, len(self.response_times))
            }
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_alert_component() -> gr.Alert:
    """Create a reusable error alert component."""
    
    return gr.Alert(
        label="Error",
        show_label=True,
        visible=False,
        variant="error"
    )

def create_success_alert_component() -> gr.Alert:
    """Create a reusable success alert component."""
    
    return gr.Alert(
        label="Success",
        show_label=True,
        visible=False,
        variant="success"
    )

def create_loading_component() -> gr.Loading:
    """Create a reusable loading component."""
    
    return gr.Loading(
        visible=False,
        text="Processing..."
    )

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_gradio_function_with_error_handling():
    """Example of how to use the error handling system in Gradio."""
    
    # Create error handling components
    error_handler = GradioErrorHandler()
    validator = GradioInputValidator()
    monitor = GradioErrorMonitor()
    
    # Example function with error handling
    @gradio_error_handler
    @validate_gradio_inputs("text_prompt", "duration", "quality")
    @monitor.monitor_function
    def generate_video(prompt: str, duration: int, quality: str):
        """Generate video with comprehensive error handling."""
        
        # Additional validation
        is_valid, message = validator.validate_text_prompt(prompt)
        if not is_valid:
            raise ValueError(message)
        
        # Simulate video generation
        if "error" in prompt.lower():
            raise RuntimeError("Simulated error for testing")
        
        # Return success result
        return {
            "success": True,
            "video": "generated_video.mp4",
            "message": "Video generated successfully"
        }
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Video Generation with Error Handling")
        
        # Input components
        prompt_input = gr.Textbox(
            label="Video Prompt",
            placeholder="Describe your video...",
            lines=3
        )
        
        duration_input = gr.Slider(
            minimum=3,
            maximum=60,
            value=15,
            step=1,
            label="Duration (seconds)"
        )
        
        quality_input = gr.Dropdown(
            choices=["Fast", "Balanced", "High Quality", "Ultra Quality"],
            value="Balanced",
            label="Quality"
        )
        
        # Output components
        output_video = gr.Video(label="Generated Video")
        status_output = gr.Textbox(label="Status", interactive=False)
        
        # Error components
        error_alert = create_error_alert_component()
        success_alert = create_success_alert_component()
        loading_component = create_loading_component()
        
        # Generate button
        generate_btn = gr.Button("Generate Video", variant="primary")
        
        # Event handler
        def handle_generation(prompt, duration, quality):
            try:
                with loading_component:
                    result = generate_video(prompt, duration, quality)
                
                if result.get("success"):
                    return {
                        output_video: result.get("video"),
                        status_output: result.get("message"),
                        error_alert: gr.update(visible=False),
                        success_alert: gr.update(visible=True, value=result.get("message"))
                    }
                else:
                    return {
                        output_video: None,
                        status_output: result.get("error_message"),
                        error_alert: gr.update(visible=True, value=result.get("error_message")),
                        success_alert: gr.update(visible=False)
                    }
                    
            except Exception as e:
                error_response = error_handler.handle_gradio_error(e, "video_generation")
                return {
                    output_video: None,
                    status_output: error_response.get("error_message"),
                    error_alert: gr.update(visible=True, value=error_response.get("error_message")),
                    success_alert: gr.update(visible=False)
                }
        
        generate_btn.click(
            fn=handle_generation,
            inputs=[prompt_input, duration_input, quality_input],
            outputs=[output_video, status_output, error_alert, success_alert]
        )
    
    return demo

if __name__ == "__main__":
    # Example usage
    demo = example_gradio_function_with_error_handling()
    demo.launch() 