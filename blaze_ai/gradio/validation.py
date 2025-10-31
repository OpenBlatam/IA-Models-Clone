"""
Input validation and error handling for Gradio interfaces.
"""
from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger(__name__)

class GradioValidator:
    """Comprehensive input validation for Gradio interfaces."""
    
    @staticmethod
    def validate_text_input(text: str, field_name: str = "text", min_length: int = 1, max_length: int = 10000) -> Tuple[bool, str]:
        """Validate text input."""
        if not text or not isinstance(text, str):
            return False, f"{field_name} must be a non-empty string."
        
        text = text.strip()
        if len(text) < min_length:
            return False, f"{field_name} must be at least {min_length} characters long."
        
        if len(text) > max_length:
            return False, f"{field_name} must be no more than {max_length} characters long."
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"{field_name} contains potentially harmful content."
        
        return True, ""
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], field_name: str, min_val: float, max_val: float) -> Tuple[bool, str]:
        """Validate numeric value within range."""
        if not isinstance(value, (int, float)):
            return False, f"{field_name} must be a number."
        
        if value < min_val or value > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}."
        
        return True, ""
    
    @staticmethod
    def validate_integer_range(value: int, field_name: str, min_val: int, max_val: int) -> Tuple[bool, str]:
        """Validate integer value within range."""
        if not isinstance(value, int):
            return False, f"{field_name} must be an integer."
        
        if value < min_val or value > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}."
        
        return True, ""
    
    @staticmethod
    def validate_boolean(value: Any, field_name: str) -> Tuple[bool, str]:
        """Validate boolean value."""
        if not isinstance(value, bool):
            return False, f"{field_name} must be a boolean value."
        
        return True, ""
    
    @staticmethod
    def validate_file_path(file_path: str, field_name: str = "file", required_extensions: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Validate file path."""
        if not file_path or not isinstance(file_path, str):
            return False, f"{field_name} path must be a non-empty string."
        
        path = Path(file_path)
        if not path.exists():
            return False, f"{field_name} file does not exist: {file_path}"
        
        if required_extensions and path.suffix.lower() not in [ext.lower() for ext in required_extensions]:
            return False, f"{field_name} must have one of these extensions: {', '.join(required_extensions)}"
        
        return True, ""
    
    @staticmethod
    def validate_json_data(json_str: str, field_name: str = "JSON data") -> Tuple[bool, str]:
        """Validate JSON string."""
        if not json_str or not isinstance(json_str, str):
            return False, f"{field_name} must be a non-empty string."
        
        try:
            json.loads(json_str)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"{field_name} is not valid JSON: {str(e)}"
    
    @staticmethod
    def validate_image_input(image: Any, field_name: str = "image") -> Tuple[bool, str]:
        """Validate image input."""
        if image is None:
            return False, f"{field_name} is required."
        
        if not isinstance(image, Image.Image):
            return False, f"{field_name} must be a PIL Image."
        
        # Check image dimensions
        width, height = image.size
        if width < 64 or height < 64:
            return False, f"{field_name} dimensions must be at least 64x64 pixels."
        
        if width > 2048 or height > 2048:
            return False, f"{field_name} dimensions must be no more than 2048x2048 pixels."
        
        return True, ""

class TextGenerationValidator:
    """Validator for text generation parameters."""
    
    @staticmethod
    def validate_text_generation_params(
        prompt: str,
        max_length: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        num_return_sequences: int
    ) -> Tuple[bool, str]:
        """Validate text generation parameters."""
        
        # Validate prompt
        is_valid, error_msg = GradioValidator.validate_text_input(prompt, "Prompt", 1, 5000)
        if not is_valid:
            return False, error_msg
        
        # Validate max_length
        is_valid, error_msg = GradioValidator.validate_integer_range(max_length, "Max Length", 1, 2000)
        if not is_valid:
            return False, error_msg
        
        # Validate temperature
        is_valid, error_msg = GradioValidator.validate_numeric_range(temperature, "Temperature", 0.1, 2.0)
        if not is_valid:
            return False, error_msg
        
        # Validate top_p
        is_valid, error_msg = GradioValidator.validate_numeric_range(top_p, "Top-p", 0.1, 1.0)
        if not is_valid:
            return False, error_msg
        
        # Validate do_sample
        is_valid, error_msg = GradioValidator.validate_boolean(do_sample, "Do Sample")
        if not is_valid:
            return False, error_msg
        
        # Validate num_return_sequences
        is_valid, error_msg = GradioValidator.validate_integer_range(num_return_sequences, "Number of Sequences", 1, 10)
        if not is_valid:
            return False, error_msg
        
        return True, ""

class ImageGenerationValidator:
    """Validator for image generation parameters."""
    
    @staticmethod
    def validate_image_generation_params(
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: Optional[int],
        num_images: int
    ) -> Tuple[bool, str]:
        """Validate image generation parameters."""
        
        # Validate prompt
        is_valid, error_msg = GradioValidator.validate_text_input(prompt, "Image Prompt", 1, 1000)
        if not is_valid:
            return False, error_msg
        
        # Validate negative prompt (optional)
        if negative_prompt:
            is_valid, error_msg = GradioValidator.validate_text_input(negative_prompt, "Negative Prompt", 0, 1000)
            if not is_valid:
                return False, error_msg
        
        # Validate num_steps
        is_valid, error_msg = GradioValidator.validate_integer_range(num_steps, "Number of Steps", 10, 100)
        if not is_valid:
            return False, error_msg
        
        # Validate guidance_scale
        is_valid, error_msg = GradioValidator.validate_numeric_range(guidance_scale, "Guidance Scale", 1.0, 20.0)
        if not is_valid:
            return False, error_msg
        
        # Validate width
        is_valid, error_msg = GradioValidator.validate_integer_range(width, "Width", 256, 1024)
        if not is_valid:
            return False, error_msg
        
        # Validate height
        is_valid, error_msg = GradioValidator.validate_integer_range(height, "Height", 256, 1024)
        if not is_valid:
            return False, error_msg
        
        # Validate seed (optional)
        if seed is not None:
            is_valid, error_msg = GradioValidator.validate_integer_range(seed, "Seed", -2**31, 2**31-1)
            if not is_valid:
                return False, error_msg
        
        # Validate num_images
        is_valid, error_msg = GradioValidator.validate_integer_range(num_images, "Number of Images", 1, 4)
        if not is_valid:
            return False, error_msg
        
        return True, ""

class TrainingValidator:
    """Validator for training parameters."""
    
    @staticmethod
    def validate_training_params(
        training_data: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        model_type: str
    ) -> Tuple[bool, str]:
        """Validate training parameters."""
        
        # Validate training_data (JSON)
        is_valid, error_msg = GradioValidator.validate_json_data(training_data, "Training Data")
        if not is_valid:
            return False, error_msg
        
        # Validate epochs
        is_valid, error_msg = GradioValidator.validate_integer_range(epochs, "Epochs", 1, 1000)
        if not is_valid:
            return False, error_msg
        
        # Validate batch_size
        is_valid, error_msg = GradioValidator.validate_integer_range(batch_size, "Batch Size", 1, 128)
        if not is_valid:
            return False, error_msg
        
        # Validate learning_rate
        is_valid, error_msg = GradioValidator.validate_numeric_range(learning_rate, "Learning Rate", 1e-6, 1.0)
        if not is_valid:
            return False, error_msg
        
        # Validate model_type
        valid_model_types = ["llm", "diffusion", "classifier", "regressor"]
        if model_type not in valid_model_types:
            return False, f"Model type must be one of: {', '.join(valid_model_types)}"
        
        return True, ""

class GradioErrorHandler:
    """Error handler for Gradio interfaces."""
    
    @staticmethod
    def handle_validation_error(error_msg: str) -> Tuple[str, Dict]:
        """Handle validation errors with user-friendly messages."""
        return f"❌ Validation Error: {error_msg}", {"error": error_msg, "type": "validation"}
    
    @staticmethod
    def handle_model_error(error: Exception, operation: str) -> Tuple[str, Dict]:
        """Handle model-related errors."""
        error_msg = f"Model error during {operation}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        return f"🤖 Model Error: {str(error)}", {"error": str(error), "type": "model", "operation": operation}
    
    @staticmethod
    def handle_system_error(error: Exception, operation: str) -> Tuple[str, Dict]:
        """Handle system-related errors."""
        error_msg = f"System error during {operation}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        return f"💻 System Error: {str(error)}", {"error": str(error), "type": "system", "operation": operation}
    
    @staticmethod
    def handle_timeout_error(operation: str, timeout_seconds: int) -> Tuple[str, Dict]:
        """Handle timeout errors."""
        error_msg = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        logger.warning(error_msg)
        return f"⏰ Timeout: {operation} took too long", {"error": error_msg, "type": "timeout", "operation": operation}
    
    @staticmethod
    def handle_memory_error(operation: str) -> Tuple[str, Dict]:
        """Handle memory-related errors."""
        error_msg = f"Insufficient memory for {operation}"
        logger.error(error_msg)
        return f"💾 Memory Error: Not enough memory for {operation}", {"error": error_msg, "type": "memory", "operation": operation}
    
    @staticmethod
    def handle_gpu_error(error: Exception, operation: str) -> Tuple[str, Dict]:
        """Handle GPU-related errors."""
        error_msg = f"GPU error during {operation}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        return f"🎮 GPU Error: {str(error)}", {"error": str(error), "type": "gpu", "operation": operation}

class SafeGradioExecutor:
    """Safe executor for Gradio operations with comprehensive error handling."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(__name__)
    
    async def execute_with_timeout(self, operation: str, func, *args, **kwargs):
        """Execute function with timeout and error handling."""
        import asyncio
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_safe(operation, func, *args, **kwargs),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            return GradioErrorHandler.handle_timeout_error(operation, self.timeout_seconds)
        except Exception as e:
            return self._handle_generic_error(e, operation)
    
    async def _execute_safe(self, operation: str, func, *args, **kwargs):
        """Execute function with comprehensive error handling."""
        try:
            # Check GPU memory before operation
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = gpu_memory - allocated_memory
                
                if free_memory < 1e9:  # Less than 1GB free
                    return GradioErrorHandler.handle_memory_error(operation)
            
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            return GradioErrorHandler.handle_memory_error(operation)
        except torch.cuda.CudaError as e:
            return GradioErrorHandler.handle_gpu_error(e, operation)
        except Exception as e:
            return self._handle_generic_error(e, operation)
    
    def _handle_generic_error(self, error: Exception, operation: str):
        """Handle generic errors."""
        error_type = type(error).__name__
        
        if "model" in error_type.lower() or "model" in str(error).lower():
            return GradioErrorHandler.handle_model_error(error, operation)
        elif "memory" in error_type.lower() or "memory" in str(error).lower():
            return GradioErrorHandler.handle_memory_error(operation)
        elif "gpu" in error_type.lower() or "cuda" in str(error).lower():
            return GradioErrorHandler.handle_gpu_error(error, operation)
        else:
            return GradioErrorHandler.handle_system_error(error, operation)

# Factory functions for easy access
def get_gradio_validator() -> GradioValidator:
    """Get a Gradio validator instance."""
    return GradioValidator()

def get_text_generation_validator() -> TextGenerationValidator:
    """Get a text generation validator instance."""
    return TextGenerationValidator()

def get_image_generation_validator() -> ImageGenerationValidator:
    """Get an image generation validator instance."""
    return ImageGenerationValidator()

def get_training_validator() -> TrainingValidator:
    """Get a training validator instance."""
    return TrainingValidator()

def get_gradio_error_handler() -> GradioErrorHandler:
    """Get a Gradio error handler instance."""
    return GradioErrorHandler()

def get_safe_gradio_executor(timeout_seconds: int = 300) -> SafeGradioExecutor:
    """Get a safe Gradio executor instance."""
    return SafeGradioExecutor(timeout_seconds)

__all__ = [
    "GradioValidator",
    "TextGenerationValidator", 
    "ImageGenerationValidator",
    "TrainingValidator",
    "GradioErrorHandler",
    "SafeGradioExecutor",
    "get_gradio_validator",
    "get_text_generation_validator",
    "get_image_generation_validator", 
    "get_training_validator",
    "get_gradio_error_handler",
    "get_safe_gradio_executor"
]
