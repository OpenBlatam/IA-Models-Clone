from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
import sys
import logging
import traceback
import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
import numpy as np
import torch
import gradio as gr
from PIL import Image
import json
import re
from functools import wraps
from datetime import datetime
from production_code import MultiGPUTrainer, TrainingConfiguration
                import io
            from PIL import ImageDraw, ImageFont
            from PIL import ImageDraw
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Error Handling and Input Validation for Gradio Apps
==================================================

This module provides comprehensive error handling and input validation
for Gradio applications with:
- Input validation decorators
- Error handling utilities
- User-friendly error messages
- Graceful degradation
- Logging and monitoring
- Recovery mechanisms
"""


# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioErrorHandler:
    """Comprehensive error handling and validation for Gradio apps"""
    
    def __init__(self) -> Any:
        self.error_log = []
        self.validation_rules = {}
        self.recovery_strategies = {}
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
        logger.info("Gradio Error Handler initialized")
    
    def _initialize_validation_rules(self) -> Any:
        """Initialize validation rules for different input types"""
        self.validation_rules = {
            'text': {
                'min_length': 1,
                'max_length': 10000,
                'allowed_chars': re.compile(r'^[a-zA-Z0-9\s\.,!?;:\'\"()-_+=@#$%^&*()\[\]{}|\\/<>~`]+$'),
                'forbidden_words': ['script', 'javascript', 'eval', 'exec']
            },
            'image': {
                'max_size_mb': 50,
                'allowed_formats': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
                'max_dimensions': (4096, 4096),
                'min_dimensions': (1, 1)
            },
            'audio': {
                'max_duration_seconds': 300,  # 5 minutes
                'max_size_mb': 100,
                'allowed_formats': ['.wav', '.mp3', '.flac', '.ogg'],
                'sample_rate_range': (8000, 48000)
            },
            'number': {
                'min_value': -1e6,
                'max_value': 1e6,
                'precision': 6
            },
            'file': {
                'max_size_mb': 100,
                'allowed_extensions': ['.txt', '.json', '.csv', '.xml']
            }
        }
    
    def log_error(self, error: Exception, context: str = "", user_input: Any = None):
        """Log error with context and user input"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'user_input': str(user_input)[:200] if user_input else None,
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_entry)
        logger.error(f"Gradio Error in {context}: {error}")
        
        # Keep only last 1000 errors
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-1000:]
    
    def validate_text_input(self, text: str, field_name: str = "text") -> Tuple[bool, str]:
        """Validate text input"""
        try:
            if not text or not isinstance(text, str):
                return False, f"{field_name} must be a non-empty string"
            
            text = text.strip()
            if len(text) < self.validation_rules['text']['min_length']:
                return False, f"{field_name} must be at least {self.validation_rules['text']['min_length']} characters"
            
            if len(text) > self.validation_rules['text']['max_length']:
                return False, f"{field_name} must be less than {self.validation_rules['text']['max_length']} characters"
            
            # Check for forbidden content
            for word in self.validation_rules['text']['forbidden_words']:
                if word.lower() in text.lower():
                    return False, f"{field_name} contains forbidden content"
            
            # Check for suspicious patterns
            if re.search(r'<script|javascript:|eval\(|exec\(', text, re.IGNORECASE):
                return False, f"{field_name} contains potentially harmful content"
            
            return True, "Valid"
            
        except Exception as e:
            self.log_error(e, f"text_validation_{field_name}", text)
            return False, f"Validation error: {str(e)}"
    
    def validate_image_input(self, image: Union[Image.Image, np.ndarray, str], field_name: str = "image") -> Tuple[bool, str]:
        """Validate image input"""
        try:
            if image is None:
                return False, f"{field_name} is required"
            
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                if not os.path.exists(image):
                    return False, f"{field_name} file not found"
                image = Image.open(image)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Check dimensions
            width, height = image.size
            max_w, max_h = self.validation_rules['image']['max_dimensions']
            min_w, min_h = self.validation_rules['image']['min_dimensions']
            
            if width > max_w or height > max_h:
                return False, f"{field_name} dimensions must be less than {max_w}x{max_h}"
            
            if width < min_w or height < min_h:
                return False, f"{field_name} dimensions must be at least {min_w}x{min_h}"
            
            # Check file size (approximate)
            try:
                # Save to bytes to check size
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
                
                if size_mb > self.validation_rules['image']['max_size_mb']:
                    return False, f"{field_name} size must be less than {self.validation_rules['image']['max_size_mb']}MB"
            except:
                pass  # Skip size check if it fails
            
            return True, "Valid"
            
        except Exception as e:
            self.log_error(e, f"image_validation_{field_name}", str(image))
            return False, f"Validation error: {str(e)}"
    
    def validate_audio_input(self, audio: Tuple[np.ndarray, int], field_name: str = "audio") -> Tuple[bool, str]:
        """Validate audio input"""
        try:
            if audio is None:
                return False, f"{field_name} is required"
            
            if not isinstance(audio, tuple) or len(audio) != 2:
                return False, f"{field_name} must be a tuple of (audio_data, sample_rate)"
            
            audio_data, sample_rate = audio
            
            if not isinstance(audio_data, np.ndarray):
                return False, f"{field_name} audio data must be a numpy array"
            
            if not isinstance(sample_rate, int):
                return False, f"{field_name} sample rate must be an integer"
            
            # Check sample rate
            min_sr, max_sr = self.validation_rules['audio']['sample_rate_range']
            if sample_rate < min_sr or sample_rate > max_sr:
                return False, f"{field_name} sample rate must be between {min_sr} and {max_sr} Hz"
            
            # Check duration
            duration = len(audio_data) / sample_rate
            max_duration = self.validation_rules['audio']['max_duration_seconds']
            if duration > max_duration:
                return False, f"{field_name} duration must be less than {max_duration} seconds"
            
            # Check for NaN or infinite values
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                return False, f"{field_name} contains invalid audio data"
            
            return True, "Valid"
            
        except Exception as e:
            self.log_error(e, f"audio_validation_{field_name}", str(audio))
            return False, f"Validation error: {str(e)}"
    
    def validate_number_input(self, number: Union[int, float], field_name: str = "number") -> Tuple[bool, str]:
        """Validate number input"""
        try:
            if number is None:
                return False, f"{field_name} is required"
            
            if not isinstance(number, (int, float)):
                return False, f"{field_name} must be a number"
            
            min_val = self.validation_rules['number']['min_value']
            max_val = self.validation_rules['number']['max_value']
            
            if number < min_val or number > max_val:
                return False, f"{field_name} must be between {min_val} and {max_val}"
            
            # Check for NaN or infinite values
            if np.isnan(number) or np.isinf(number):
                return False, f"{field_name} must be a finite number"
            
            return True, "Valid"
            
        except Exception as e:
            self.log_error(e, f"number_validation_{field_name}", number)
            return False, f"Validation error: {str(e)}"
    
    def validate_file_input(self, file_path: str, field_name: str = "file") -> Tuple[bool, str]:
        """Validate file input"""
        try:
            if not file_path or not isinstance(file_path, str):
                return False, f"{field_name} must be a valid file path"
            
            if not os.path.exists(file_path):
                return False, f"{field_name} file not found"
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            max_size = self.validation_rules['file']['max_size_mb']
            
            if file_size_mb > max_size:
                return False, f"{field_name} size must be less than {max_size}MB"
            
            # Check file extension
            _, ext = os.path.splitext(file_path)
            allowed_extensions = self.validation_rules['file']['allowed_extensions']
            
            if ext.lower() not in allowed_extensions:
                return False, f"{field_name} must have one of these extensions: {', '.join(allowed_extensions)}"
            
            return True, "Valid"
            
        except Exception as e:
            self.log_error(e, f"file_validation_{field_name}", file_path)
            return False, f"Validation error: {str(e)}"
    
    def create_user_friendly_error_message(self, error: Exception, context: str = "") -> str:
        """Create user-friendly error messages"""
        error_type = type(error).__name__
        
        # Common error patterns
        if "CUDA" in str(error) or "GPU" in str(error):
            return f"⚠️ **Hardware Issue**: GPU processing error. Please try again or contact support if the problem persists."
        
        elif "memory" in str(error).lower():
            return f"⚠️ **Memory Issue**: Not enough memory available. Please try with smaller inputs or restart the application."
        
        elif "timeout" in str(error).lower():
            return f"⏱️ **Timeout**: The operation took too long. Please try again with simpler inputs."
        
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            return f"🌐 **Network Issue**: Connection problem. Please check your internet connection and try again."
        
        elif "permission" in str(error).lower():
            return f"🔒 **Permission Issue**: Access denied. Please check file permissions or contact administrator."
        
        elif "file not found" in str(error).lower():
            return f"📁 **File Not Found**: The requested file could not be found. Please check the file path."
        
        elif "validation" in str(error).lower():
            return f"✅ **Input Validation**: {str(error)}"
        
        else:
            return f"❌ **Unexpected Error**: An unexpected error occurred. Please try again or contact support if the problem persists.\n\n**Error Details**: {str(error)}"
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """Safely execute a function with error handling"""
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful execution
            logger.info(f"Function {func.__name__} executed successfully in {execution_time:.2f}s")
            
            return result, "Success"
            
        except Exception as e:
            self.log_error(e, f"function_execution_{func.__name__}", {"args": args, "kwargs": kwargs})
            error_message = self.create_user_friendly_error_message(e, func.__name__)
            return None, error_message
    
    def retry_on_error(self, max_retries: int = 3, delay: float = 1.0):
        """Decorator to retry functions on error"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                
                # All retries failed
                self.log_error(last_error, f"retry_failed_{func.__name__}", {"args": args, "kwargs": kwargs})
                raise last_error
            
            return wrapper
        return decorator
    
    def validate_inputs(self, **validations) -> Callable:
        """Decorator to validate function inputs"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Validate inputs based on validation rules
                for param_name, validation_type in validations.items():
                    if param_name in kwargs:
                        value = kwargs[param_name]
                        
                        if validation_type == 'text':
                            is_valid, message = self.validate_text_input(value, param_name)
                        elif validation_type == 'image':
                            is_valid, message = self.validate_image_input(value, param_name)
                        elif validation_type == 'audio':
                            is_valid, message = self.validate_audio_input(value, param_name)
                        elif validation_type == 'number':
                            is_valid, message = self.validate_number_input(value, param_name)
                        elif validation_type == 'file':
                            is_valid, message = self.validate_file_input(value, param_name)
                        else:
                            is_valid, message = True, "Valid"
                        
                        if not is_valid:
                            raise ValueError(f"Input validation failed for {param_name}: {message}")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        if not self.error_log:
            return {"total_errors": 0, "recent_errors": []}
        
        # Count error types
        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Get recent errors (last 10)
        recent_errors = self.error_log[-10:]
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "recent_errors": recent_errors,
            "last_error_time": self.error_log[-1]['timestamp'] if self.error_log else None
        }


class ErrorHandledGradioInterface:
    """Gradio interface with comprehensive error handling"""
    
    def __init__(self) -> Any:
        self.error_handler = GradioErrorHandler()
        self.config = TrainingConfiguration(
            enable_gradio_demo=True,
            gradio_port=7865,
            gradio_share=False
        )
        
        logger.info("Error-Handled Gradio Interface initialized")
    
    @GradioErrorHandler().validate_inputs(
        prompt='text',
        max_length='number',
        temperature='number'
    )
    def generate_text_with_validation(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> Tuple[str, str]:
        """Generate text with comprehensive validation and error handling"""
        
        def text_generation_logic():
            
    """text_generation_logic function."""
# Simulate text generation
            if not prompt or len(prompt.strip()) < 3:
                raise ValueError("Prompt must be at least 3 characters long")
            
            if max_length < 10 or max_length > 1000:
                raise ValueError("Max length must be between 10 and 1000")
            
            if temperature < 0.1 or temperature > 2.0:
                raise ValueError("Temperature must be between 0.1 and 2.0")
            
            # Simulate processing delay
            time.sleep(0.5)
            
            # Generate text
            words = prompt.split()
            if len(words) < 3:
                words.extend(["artificial", "intelligence", "technology"])
            
            generated_text = " ".join(words[-3:]) + " " + " ".join([
                "is", "revolutionizing", "the", "way", "we", "think", "about",
                "machine", "learning", "and", "deep", "neural", "networks."
            ])
            
            if temperature > 0.5:
                generated_text += " " + " ".join([
                    "The", "future", "looks", "promising", "for", "AI", "applications."
                ])
            
            return generated_text[:max_length]
        
        result, status = self.error_handler.safe_execute(text_generation_logic)
        
        if result is None:
            return "", status
        else:
            return result, "✅ Text generated successfully!"
    
    @GradioErrorHandler().validate_inputs(
        prompt='text',
        style='text',
        size='text'
    )
    def generate_image_with_validation(self, prompt: str, style: str = "realistic", size: str = "medium") -> Tuple[Image.Image, str]:
        """Generate image with comprehensive validation and error handling"""
        
        def image_generation_logic():
            
    """image_generation_logic function."""
# Validate inputs
            if not prompt or len(prompt.strip()) < 5:
                raise ValueError("Image prompt must be at least 5 characters long")
            
            valid_styles = ['realistic', 'artistic', 'minimal', 'abstract']
            if style not in valid_styles:
                raise ValueError(f"Style must be one of: {', '.join(valid_styles)}")
            
            valid_sizes = ['small', 'medium', 'large']
            if size not in valid_sizes:
                raise ValueError(f"Size must be one of: {', '.join(valid_sizes)}")
            
            # Simulate processing delay
            time.sleep(1.0)
            
            # Create sample image
            dimensions = {
                'small': (256, 256),
                'medium': (512, 512),
                'large': (1024, 1024)
            }
            
            width, height = dimensions[size]
            img = Image.new('RGB', (width, height), color='white')
            
            # Add visual elements based on style
            
            draw = ImageDraw.Draw(img)
            
            if style == 'realistic':
                draw.rectangle([0, 0, width, height//2], fill='lightblue')
                draw.rectangle([0, height//2, width, height], fill='green')
                draw.ellipse([width-100, 50, width-50, 100], fill='yellow')
            elif style == 'artistic':
                for i in range(0, width, 50):
                    draw.line([(i, 0), (i, height)], fill='purple', width=2)
            elif style == 'minimal':
                draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], 
                             outline='black', width=3)
            elif style == 'abstract':
                for i in range(10):
                    x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                    x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                    color = tuple(np.random.randint(0, 255, 3))
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=5)
            
            # Add text overlay
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), f"Style: {style}", fill='black', font=font)
            draw.text((10, height-30), f"Size: {size}", fill='black', font=font)
            
            return img
        
        result, status = self.error_handler.safe_execute(image_generation_logic)
        
        if result is None:
            # Return a placeholder error image
            error_img = Image.new('RGB', (512, 512), color='red')
            draw = ImageDraw.Draw(error_img)
            draw.text((50, 250), "Error generating image", fill='white')
            return error_img, status
        else:
            return result, "✅ Image generated successfully!"
    
    @GradioErrorHandler().validate_inputs(
        audio_input='audio',
        effect='text',
        intensity='number'
    )
    def process_audio_with_validation(self, audio_input, effect: str = "noise_reduction", intensity: float = 0.5) -> Tuple[Tuple[np.ndarray, int], str]:
        """Process audio with comprehensive validation and error handling"""
        
        def audio_processing_logic():
            
    """audio_processing_logic function."""
# Validate inputs
            if audio_input is None:
                raise ValueError("Audio input is required")
            
            valid_effects = ['noise_reduction', 'equalizer', 'reverb', 'pitch_shift']
            if effect not in valid_effects:
                raise ValueError(f"Effect must be one of: {', '.join(valid_effects)}")
            
            if intensity < 0.0 or intensity > 1.0:
                raise ValueError("Intensity must be between 0.0 and 1.0")
            
            # Simulate processing delay
            time.sleep(0.8)
            
            # Process audio
            audio_data, sample_rate = audio_input
            
            if effect == "noise_reduction":
                processed_audio = audio_data * (1 - intensity * 0.3)
            elif effect == "equalizer":
                processed_audio = audio_data * np.random.uniform(0.5, 1.5, len(audio_data))
            elif effect == "reverb":
                delay = int(sample_rate * 0.1 * intensity)
                processed_audio = audio_data.copy()
                processed_audio[delay:] += audio_data[:-delay] * intensity * 0.3
            elif effect == "pitch_shift":
                shift_factor = 1 + (intensity - 0.5) * 0.4
                processed_audio = np.interp(
                    np.arange(len(audio_data)),
                    np.arange(len(audio_data)) * shift_factor,
                    audio_data
                )
            else:
                processed_audio = audio_data
            
            return (processed_audio, sample_rate)
        
        result, status = self.error_handler.safe_execute(audio_processing_logic)
        
        if result is None:
            return None, status
        else:
            return result, "✅ Audio processed successfully!"
    
    def create_error_handled_interface(self) -> gr.Interface:
        """Create Gradio interface with comprehensive error handling"""
        
        def get_error_summary():
            """Get error summary for monitoring"""
            summary = self.error_handler.get_error_summary()
            return json.dumps(summary, indent=2)
        
        # Create interface
        with gr.Blocks(
            title="Error-Handled AI Interface",
            theme=gr.themes.Soft(),
            css="""
            .error-message {
                background-color: #ffebee;
                border: 1px solid #f44336;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                color: #c62828;
            }
            .success-message {
                background-color: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                color: #2e7d32;
            }
            .validation-error {
                background-color: #fff3e0;
                border: 1px solid #ff9800;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                color: #e65100;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🛡️ Error-Handled AI Interface")
            gr.Markdown("Robust AI capabilities with comprehensive error handling and input validation")
            
            with gr.Tabs():
                with gr.TabItem("📝 Text Generation"):
                    gr.Markdown("### Generate text with validation and error handling")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_prompt = gr.Textbox(
                                label="Text Prompt",
                                placeholder="Enter your text prompt...",
                                lines=3
                            )
                            
                            text_length = gr.Slider(
                                minimum=10, maximum=1000, value=100, step=10,
                                label="Max Length"
                            )
                            
                            text_temp = gr.Slider(
                                minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                                label="Temperature"
                            )
                            
                            text_btn = gr.Button("🚀 Generate Text", variant="primary")
                        
                        with gr.Column(scale=1):
                            text_output = gr.Textbox(
                                label="Generated Text",
                                lines=8,
                                interactive=False
                            )
                            
                            text_status = gr.Markdown(label="Status")
                
                with gr.TabItem("🎨 Image Generation"):
                    gr.Markdown("### Generate images with validation and error handling")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_prompt = gr.Textbox(
                                label="Image Description",
                                placeholder="Describe the image you want to create...",
                                lines=3
                            )
                            
                            image_style = gr.Dropdown(
                                choices=['realistic', 'artistic', 'minimal', 'abstract'],
                                value='realistic',
                                label="Style"
                            )
                            
                            image_size = gr.Dropdown(
                                choices=['small', 'medium', 'large'],
                                value='medium',
                                label="Size"
                            )
                            
                            image_btn = gr.Button("🎨 Generate Image", variant="primary")
                        
                        with gr.Column(scale=1):
                            image_output = gr.Image(
                                label="Generated Image",
                                type="pil"
                            )
                            
                            image_status = gr.Markdown(label="Status")
                
                with gr.TabItem("🎵 Audio Processing"):
                    gr.Markdown("### Process audio with validation and error handling")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            audio_input = gr.Audio(
                                label="Input Audio",
                                type="numpy"
                            )
                            
                            audio_effect = gr.Dropdown(
                                choices=['noise_reduction', 'equalizer', 'reverb', 'pitch_shift'],
                                value='noise_reduction',
                                label="Effect"
                            )
                            
                            audio_intensity = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                                label="Intensity"
                            )
                            
                            audio_btn = gr.Button("🎵 Process Audio", variant="primary")
                        
                        with gr.Column(scale=1):
                            audio_output = gr.Audio(
                                label="Processed Audio",
                                type="numpy"
                            )
                            
                            audio_status = gr.Markdown(label="Status")
                
                with gr.TabItem("📊 Error Monitoring"):
                    gr.Markdown("### Monitor errors and system status")
                    
                    with gr.Row():
                        with gr.Column():
                            error_summary_btn = gr.Button("📊 Get Error Summary")
                            error_summary_output = gr.JSON(label="Error Summary")
                        
                        with gr.Column():
                            gr.Markdown("### System Status")
                            gr.Markdown("""
                            **Error Handling Features:**
                            - ✅ Input validation
                            - ✅ Error logging
                            - ✅ User-friendly messages
                            - ✅ Graceful degradation
                            - ✅ Recovery mechanisms
                            
                            **Validation Rules:**
                            - Text: Length, content filtering
                            - Images: Size, format, dimensions
                            - Audio: Duration, sample rate, quality
                            - Numbers: Range, precision
                            """)
            
            # Event handlers
            text_btn.click(
                fn=self.generate_text_with_validation,
                inputs=[text_prompt, text_length, text_temp],
                outputs=[text_output, text_status]
            )
            
            image_btn.click(
                fn=self.generate_image_with_validation,
                inputs=[image_prompt, image_style, image_size],
                outputs=[image_output, image_status]
            )
            
            audio_btn.click(
                fn=self.process_audio_with_validation,
                inputs=[audio_input, audio_effect, audio_intensity],
                outputs=[audio_output, audio_status]
            )
            
            error_summary_btn.click(
                fn=get_error_summary,
                inputs=[],
                outputs=[error_summary_output]
            )
        
        return interface
    
    def launch_interface(self, port: int = 7865, share: bool = False):
        """Launch the error-handled interface"""
        print("🛡️ Launching Error-Handled Gradio Interface...")
        
        interface = self.create_error_handled_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the error-handled interface"""
    print("🛡️ Starting Error-Handled Gradio Interface...")
    
    interface = ErrorHandledGradioInterface()
    interface.launch_interface(port=7865, share=False)


match __name__:
    case "__main__":
    main() 