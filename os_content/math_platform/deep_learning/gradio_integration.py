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
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
import traceback
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import base64
import io
import re
import hashlib
from functools import wraps
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Gradio Integration for Deep Learning Models
Production-ready implementation of interactive demos with proper error handling and input validation.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GradioConfig:
    """Configuration for Gradio applications."""
    # App parameters
    app_title: str = "Deep Learning Model Demo"
    app_description: str = "Interactive demo for deep learning models"
    app_theme: str = "default"  # default, soft, glass, monochrome
    
    # Server parameters
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    debug: bool = False
    
    # Model parameters
    model_path: str = ""
    model_type: str = "transformer"  # transformer, diffusion, classification, generation
    device: str = "cuda"
    dtype: str = "float16"
    
    # Interface parameters
    max_input_length: int = 512
    max_output_length: int = 1024
    max_batch_size: int = 4
    timeout: int = 300
    
    # Security parameters
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    
    # UI parameters
    show_examples: bool = True
    show_advanced_options: bool = True
    enable_download: bool = True
    enable_upload: bool = True
    
    # Logging parameters
    log_predictions: bool = True
    log_errors: bool = True
    save_predictions: bool = False
    predictions_dir: str = "./predictions"


class InputValidator:
    """Input validation utilities for Gradio apps."""
    
    def __init__(self, config: GradioConfig):
        
    """__init__ function."""
self.config = config
        self.rate_limit_cache = {}
    
    def validate_text_input(self, text: str, max_length: Optional[int] = None) -> Tuple[bool, str]:
        """Validate text input."""
        if not text or not text.strip():
            return False, "Text input cannot be empty"
        
        if max_length is None:
            max_length = self.config.max_input_length
        
        if len(text) > max_length:
            return False, f"Text input too long. Maximum length: {max_length} characters"
        
        # Check for potentially harmful content
        if self._contains_harmful_content(text):
            return False, "Input contains potentially harmful content"
        
        return True, "Valid input"
    
    def validate_image_input(self, image: Image.Image) -> Tuple[bool, str]:
        """Validate image input."""
        if image is None:
            return False, "Image input cannot be empty"
        
        # Check image dimensions
        width, height = image.size
        if width < 64 or height < 64:
            return False, "Image too small. Minimum size: 64x64 pixels"
        
        if width > 2048 or height > 2048:
            return False, "Image too large. Maximum size: 2048x2048 pixels"
        
        # Check file size (approximate)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        if len(img_byte_arr) > 10 * 1024 * 1024:  # 10MB
            return False, "Image file too large. Maximum size: 10MB"
        
        return True, "Valid image"
    
    def validate_audio_input(self, audio: Tuple[int, np.ndarray]) -> Tuple[bool, str]:
        """Validate audio input."""
        if audio is None:
            return False, "Audio input cannot be empty"
        
        sample_rate, audio_data = audio
        
        if sample_rate < 8000 or sample_rate > 48000:
            return False, "Invalid sample rate. Must be between 8000 and 48000 Hz"
        
        if len(audio_data) < 1000:  # Minimum 1 second at 1kHz
            return False, "Audio too short. Minimum duration: 1 second"
        
        if len(audio_data) > 60 * sample_rate:  # Maximum 60 seconds
            return False, "Audio too long. Maximum duration: 60 seconds"
        
        return True, "Valid audio"
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, str]:
        """Check rate limiting for user."""
        if not self.config.enable_rate_limiting:
            return True, "Rate limiting disabled"
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self.rate_limit_cache = {
            k: v for k, v in self.rate_limit_cache.items() 
            if v > minute_ago
        }
        
        # Check user requests
        user_requests = [
            timestamp for timestamp in self.rate_limit_cache.values()
            if timestamp > minute_ago
        ]
        
        if len(user_requests) >= self.config.max_requests_per_minute:
            return False, f"Rate limit exceeded. Maximum {self.config.max_requests_per_minute} requests per minute"
        
        # Add current request
        self.rate_limit_cache[user_id] = current_time
        
        return True, "Rate limit OK"
    
    def _contains_harmful_content(self, text: str) -> bool:
        """Check for potentially harmful content."""
        harmful_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'<iframe',
            r'<object',
            r'<embed'
        ]
        
        text_lower = text.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def sanitize_output(self, output: str) -> str:
        """Sanitize output text."""
        if not self.config.enable_output_sanitization:
            return output
        
        # Remove potentially harmful HTML/JS
        output = re.sub(r'<script.*?>.*?</script>', '', output, flags=re.IGNORECASE)
        output = re.sub(r'<iframe.*?>.*?</iframe>', '', output, flags=re.IGNORECASE)
        output = re.sub(r'<object.*?>.*?</object>', '', output, flags=re.IGNORECASE)
        output = re.sub(r'<embed.*?>', '', output, flags=re.IGNORECASE)
        
        return output


class ErrorHandler:
    """Error handling utilities for Gradio apps."""
    
    def __init__(self, config: GradioConfig):
        
    """__init__ function."""
self.config = config
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: str = "") -> Tuple[str, Optional[Image.Image]]:
        """Handle errors gracefully."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log error
        if self.config.log_errors:
            logger.error(f"Error in {context}: {error_type}: {error_message}")
            logger.error(traceback.format_exc())
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create user-friendly error message
        if "CUDA" in error_type or "GPU" in error_message:
            user_message = "GPU memory error. Please try with a smaller input or restart the application."
        elif "timeout" in error_message.lower():
            user_message = "Request timed out. Please try again with a smaller input."
        elif "memory" in error_message.lower():
            user_message = "Memory error. Please try with a smaller input."
        else:
            user_message = f"An error occurred: {error_message}"
        
        # Create error visualization
        error_image = self._create_error_image(error_type, user_message)
        
        return user_message, error_image
    
    def _create_error_image(self, error_type: str, message: str) -> Image.Image:
        """Create error visualization."""
        # Create a simple error image
        width, height = 400, 200
        image = Image.new('RGB', (width, height), color='#ffebee')
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to use a font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw error icon
        draw.ellipse([50, 50, 150, 150], outline='#f44336', width=3)
        draw.line([75, 75, 125, 125], fill='#f44336', width=3)
        draw.line([75, 125, 125, 75], fill='#f44336', width=3)
        
        # Draw text
        if font:
            draw.text((170, 60), f"Error: {error_type}", fill='#d32f2f', font=font)
            draw.text((170, 90), message[:50], fill='#d32f2f', font=font)
        else:
            draw.text((170, 60), f"Error: {error_type}", fill='#d32f2f')
            draw.text((170, 90), message[:50], fill='#d32f2f')
        
        return image
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()


class BaseGradioApp(ABC):
    """Base class for Gradio applications."""
    
    def __init__(self, config: GradioConfig):
        
    """__init__ function."""
self.config = config
        self.validator = InputValidator(config)
        self.error_handler = ErrorHandler(config)
        self.model = None
        self.app = None
        
        # Initialize components
        self._load_model()
        self._create_interface()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model."""
        pass
    
    @abstractmethod
    def _create_interface(self) -> Any:
        """Create the Gradio interface."""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Make prediction with error handling."""
        pass
    
    def _validate_inputs(self, *args, **kwargs) -> Tuple[bool, str]:
        """Validate all inputs."""
        # Generate user ID for rate limiting
        user_id = self._generate_user_id(args, kwargs)
        
        # Check rate limiting
        rate_ok, rate_message = self.validator.check_rate_limit(user_id)
        if not rate_ok:
            return False, rate_message
        
        # Validate specific inputs based on model type
        if self.config.model_type == "transformer":
            return self._validate_transformer_inputs(*args, **kwargs)
        elif self.config.model_type == "diffusion":
            return self._validate_diffusion_inputs(*args, **kwargs)
        elif self.config.model_type == "classification":
            return self._validate_classification_inputs(*args, **kwargs)
        else:
            return True, "Input validation passed"
    
    def _validate_transformer_inputs(self, *args, **kwargs) -> Tuple[bool, str]:
        """Validate transformer model inputs."""
        if len(args) > 0:
            text = args[0]
            is_valid, message = self.validator.validate_text_input(text)
            if not is_valid:
                return False, message
        
        return True, "Valid transformer inputs"
    
    def _validate_diffusion_inputs(self, *args, **kwargs) -> Tuple[bool, str]:
        """Validate diffusion model inputs."""
        if len(args) > 0:
            prompt = args[0]
            is_valid, message = self.validator.validate_text_input(prompt)
            if not is_valid:
                return False, message
        
        return True, "Valid diffusion inputs"
    
    def _validate_classification_inputs(self, *args, **kwargs) -> Tuple[bool, str]:
        """Validate classification model inputs."""
        if len(args) > 0:
            image = args[0]
            is_valid, message = self.validator.validate_image_input(image)
            if not is_valid:
                return False, message
        
        return True, "Valid classification inputs"
    
    def _generate_user_id(self, args: tuple, kwargs: dict) -> str:
        """Generate user ID for rate limiting."""
        # Create a hash of the inputs to identify unique users
        input_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(input_str.encode()).hexdigest()[:8]
    
    def _log_prediction(self, inputs: tuple, outputs: Any, processing_time: float):
        """Log prediction for analytics."""
        if not self.config.log_predictions:
            return
        
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.config.model_type,
            'inputs': str(inputs),
            'outputs': str(outputs),
            'processing_time': processing_time,
            'success': True
        }
        
        logger.info(f"Prediction logged: {prediction_log}")
        
        # Save to file if enabled
        if self.config.save_predictions:
            self._save_prediction_to_file(prediction_log)
    
    def _save_prediction_to_file(self, prediction_log: dict):
        """Save prediction to file."""
        os.makedirs(self.config.predictions_dir, exist_ok=True)
        
        filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.config.predictions_dir, filename)
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(prediction_log, f, indent=2)
    
    def launch(self, **kwargs) -> Any:
        """Launch the Gradio app."""
        if self.app is None:
            raise ValueError("Interface not created. Call _create_interface() first.")
        
        launch_kwargs = {
            'server_name': self.config.server_name,
            'server_port': self.config.server_port,
            'share': self.config.share,
            'debug': self.config.debug,
            'show_error': self.config.debug,
            'quiet': not self.config.debug
        }
        launch_kwargs.update(kwargs)
        
        logger.info(f"Launching Gradio app on {self.config.server_name}:{self.config.server_port}")
        return self.app.launch(**launch_kwargs)


class TransformerGradioApp(BaseGradioApp):
    """Gradio app for transformer models."""
    
    def _load_model(self) -> Any:
        """Load transformer model."""
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            logger.warning("Model path not provided or does not exist. Using dummy model.")
            self.model = self._create_dummy_model()
        else:
            try:
                self.model = torch.load(self.config.model_path, map_location=self.config.device)
                self.model.eval()
                logger.info("Transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = self._create_dummy_model()
    
    def _create_dummy_model(self) -> nn.Module:
        """Create a dummy model for demonstration."""
        class DummyTransformer(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(128, 8, batch_first=True),
                    num_layers=6
                )
                self.classifier = nn.Linear(128, 2)
            
            def forward(self, input_ids, attention_mask=None) -> Any:
                x = self.embedding(input_ids)
                x = self.transformer(x)
                x = x.mean(dim=1)
                return self.classifier(x)
        
        return DummyTransformer()
    
    def _create_interface(self) -> Any:
        """Create transformer interface."""
        with gr.Blocks(title=self.config.app_title, theme=self.config.app_theme) as interface:
            gr.Markdown(f"# {self.config.app_title}")
            gr.Markdown(self.config.app_description)
            
            with gr.Row():
                with gr.Column():
                    # Input section
                    gr.Markdown("## Input")
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter your text here...",
                        max_lines=5,
                        lines=3
                    )
                    
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=10,
                            maximum=512,
                            value=100,
                            step=10,
                            label="Max Output Length"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature"
                        )
                    
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    # Output section
                    gr.Markdown("## Output")
                    text_output = gr.Textbox(
                        label="Generated Text",
                        lines=5,
                        interactive=False
                    )
                    
                    confidence_output = gr.Label(label="Confidence")
                    processing_time = gr.Textbox(label="Processing Time", interactive=False)
            
            # Examples
            if self.config.show_examples:
                gr.Markdown("## Examples")
                examples = [
                    ["Hello, how are you today?"],
                    ["The weather is beautiful today."],
                    ["I love machine learning and AI."]
                ]
                gr.Examples(examples=examples, inputs=text_input)
            
            # Advanced options
            if self.config.show_advanced_options:
                with gr.Accordion("Advanced Options", open=False):
                    gr.Markdown("### Model Parameters")
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top-p (nucleus sampling)"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k"
                    )
            
            # Event handler
            generate_btn.click(
                fn=self.predict,
                inputs=[text_input, max_length, temperature, top_p, top_k],
                outputs=[text_output, confidence_output, processing_time]
            )
        
        self.app = interface
    
    def predict(self, text: str, max_length: int, temperature: float, 
                top_p: float, top_k: int) -> Tuple[str, Dict[str, float], str]:
        """Make prediction with error handling."""
        start_time = time.time()
        
        try:
            # Validate inputs
            is_valid, message = self._validate_inputs(text)
            if not is_valid:
                return f"Error: {message}", {"error": 1.0}, "0.0s"
            
            # Process input
            if not text.strip():
                return "Please enter some text.", {"empty": 1.0}, "0.0s"
            
            # Dummy prediction (replace with actual model inference)
            processed_text = f"Processed: {text[:50]}..."
            confidence = {"positive": 0.8, "negative": 0.2}
            
            processing_time = time.time() - start_time
            
            # Log prediction
            self._log_prediction(
                inputs=(text, max_length, temperature, top_p, top_k),
                outputs=processed_text,
                processing_time=processing_time
            )
            
            return (
                processed_text,
                confidence,
                f"{processing_time:.2f}s"
            )
        
        except Exception as e:
            error_message, error_image = self.error_handler.handle_error(e, "transformer_predict")
            processing_time = time.time() - start_time
            
            return (
                error_message,
                {"error": 1.0},
                f"{processing_time:.2f}s"
            )


class DiffusionGradioApp(BaseGradioApp):
    """Gradio app for diffusion models."""
    
    def _load_model(self) -> Any:
        """Load diffusion model."""
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            logger.warning("Model path not provided or does not exist. Using dummy model.")
            self.model = self._create_dummy_diffusion_model()
        else:
            try:
                self.model = torch.load(self.config.model_path, map_location=self.config.device)
                self.model.eval()
                logger.info("Diffusion model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = self._create_dummy_diffusion_model()
    
    def _create_dummy_diffusion_model(self) -> nn.Module:
        """Create a dummy diffusion model for demonstration."""
        class DummyDiffusion(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 3, 3, padding=1)
            
            def forward(self, x, timesteps) -> Any:
                x = torch.relu(self.conv1(x))
                x = self.conv2(x)
                return x
        
        return DummyDiffusion()
    
    def _create_interface(self) -> Any:
        """Create diffusion interface."""
        with gr.Blocks(title=self.config.app_title, theme=self.config.app_theme) as interface:
            gr.Markdown(f"# {self.config.app_title}")
            gr.Markdown(self.config.app_description)
            
            with gr.Row():
                with gr.Column():
                    # Input section
                    gr.Markdown("## Input")
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="A beautiful landscape with mountains...",
                        max_lines=3,
                        lines=2
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="blurry, low quality, distorted...",
                        max_lines=2,
                        lines=1
                    )
                    
                    with gr.Row():
                        num_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Number of Steps"
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    
                    with gr.Row():
                        width = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Width"
                        )
                        height = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Height"
                        )
                    
                    generate_btn = gr.Button("Generate Image", variant="primary")
                
                with gr.Column():
                    # Output section
                    gr.Markdown("## Generated Image")
                    image_output = gr.Image(
                        label="Generated Image",
                        type="pil"
                    )
                    
                    processing_time = gr.Textbox(label="Processing Time", interactive=False)
                    seed_output = gr.Number(label="Seed", interactive=False)
            
            # Examples
            if self.config.show_examples:
                gr.Markdown("## Examples")
                examples = [
                    ["A beautiful sunset over the ocean", "blurry, low quality"],
                    ["A majestic mountain landscape", "distorted, ugly"],
                    ["A cute cat sitting in a garden", "scary, dark"]
                ]
                gr.Examples(examples=examples, inputs=[prompt_input, negative_prompt])
            
            # Advanced options
            if self.config.show_advanced_options:
                with gr.Accordion("Advanced Options", open=False):
                    gr.Markdown("### Generation Parameters")
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)",
                        precision=0
                    )
                    eta = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        label="Eta (DDIM)"
                    )
            
            # Event handler
            generate_btn.click(
                fn=self.predict,
                inputs=[prompt_input, negative_prompt, num_steps, guidance_scale, 
                       width, height, seed, eta],
                outputs=[image_output, processing_time, seed_output]
            )
        
        self.app = interface
    
    def predict(self, prompt: str, negative_prompt: str, num_steps: int, 
                guidance_scale: float, width: int, height: int, seed: int, 
                eta: float) -> Tuple[Image.Image, str, int]:
        """Make prediction with error handling."""
        start_time = time.time()
        
        try:
            # Validate inputs
            is_valid, message = self._validate_inputs(prompt)
            if not is_valid:
                return self._create_error_image(message), "0.0s", -1
            
            # Process input
            if not prompt.strip():
                return self._create_error_image("Please enter a prompt."), "0.0s", -1
            
            # Generate random seed if not provided
            if seed == -1:
                seed = np.random.randint(0, 2**32)
            
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Dummy image generation (replace with actual diffusion model)
            image = self._generate_dummy_image(prompt, width, height)
            
            processing_time = time.time() - start_time
            
            # Log prediction
            self._log_prediction(
                inputs=(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, eta),
                outputs="image_generated",
                processing_time=processing_time
            )
            
            return (
                image,
                f"{processing_time:.2f}s",
                seed
            )
        
        except Exception as e:
            error_message, error_image = self.error_handler.handle_error(e, "diffusion_predict")
            processing_time = time.time() - start_time
            
            return (
                error_image,
                f"{processing_time:.2f}s",
                -1
            )
    
    def _generate_dummy_image(self, prompt: str, width: int, height: int) -> Image.Image:
        """Generate a dummy image for demonstration."""
        # Create a simple gradient image based on the prompt
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Create gradient based on prompt length
        prompt_length = len(prompt)
        r = (prompt_length * 7) % 256
        g = (prompt_length * 13) % 256
        b = (prompt_length * 19) % 256
        
        for y in range(height):
            for x in range(width):
                # Create gradient effect
                r_val = int(r * (1 - y / height))
                g_val = int(g * (x / width))
                b_val = int(b * ((x + y) / (width + height)))
                
                draw.point((x, y), fill=(r_val, g_val, b_val))
        
        # Add text overlay
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"Generated: {prompt[:30]}..."
        if font:
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((10, 10), text, fill=(255, 255, 255))
        
        return image
    
    def _create_error_image(self, message: str) -> Image.Image:
        """Create error image."""
        image = Image.new('RGB', (512, 512), color='#ffebee')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        if font:
            draw.text((50, 250), message, fill='#d32f2f', font=font)
        else:
            draw.text((50, 250), message, fill='#d32f2f')
        
        return image


class ClassificationGradioApp(BaseGradioApp):
    """Gradio app for classification models."""
    
    def _load_model(self) -> Any:
        """Load classification model."""
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            logger.warning("Model path not provided or does not exist. Using dummy model.")
            self.model = self._create_dummy_classification_model()
        else:
            try:
                self.model = torch.load(self.config.model_path, map_location=self.config.device)
                self.model.eval()
                logger.info("Classification model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = self._create_dummy_classification_model()
    
    def _create_dummy_classification_model(self) -> nn.Module:
        """Create a dummy classification model for demonstration."""
        class DummyClassifier(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 10)
            
            def forward(self, x) -> Any:
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        return DummyClassifier()
    
    def _create_interface(self) -> Any:
        """Create classification interface."""
        with gr.Blocks(title=self.config.app_title, theme=self.config.app_theme) as interface:
            gr.Markdown(f"# {self.config.app_title}")
            gr.Markdown(self.config.app_description)
            
            with gr.Row():
                with gr.Column():
                    # Input section
                    gr.Markdown("## Input")
                    image_input = gr.Image(
                        label="Input Image",
                        type="pil"
                    )
                    
                    with gr.Row():
                        confidence_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Confidence Threshold"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-k Predictions"
                        )
                    
                    classify_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column():
                    # Output section
                    gr.Markdown("## Predictions")
                    predictions_output = gr.Label(label="Classification Results")
                    
                    processing_time = gr.Textbox(label="Processing Time", interactive=False)
                    
                    # Visualization
                    gr.Markdown("## Visualization")
                    plot_output = gr.Plot(label="Confidence Scores")
            
            # Examples
            if self.config.show_examples:
                gr.Markdown("## Examples")
                # Add example images here
                pass
            
            # Event handler
            classify_btn.click(
                fn=self.predict,
                inputs=[image_input, confidence_threshold, top_k],
                outputs=[predictions_output, processing_time, plot_output]
            )
        
        self.app = interface
    
    def predict(self, image: Image.Image, confidence_threshold: float, 
                top_k: int) -> Tuple[Dict[str, float], str, plt.Figure]:
        """Make prediction with error handling."""
        start_time = time.time()
        
        try:
            # Validate inputs
            is_valid, message = self._validate_inputs(image)
            if not is_valid:
                return {"error": 1.0}, "0.0s", self._create_error_plot(message)
            
            # Process input
            if image is None:
                return {"error": 1.0}, "0.0s", self._create_error_plot("Please upload an image.")
            
            # Dummy classification (replace with actual model inference)
            class_names = ["cat", "dog", "bird", "car", "tree", "house", "person", "book", "phone", "laptop"]
            scores = np.random.softmax(np.random.randn(10))
            
            # Filter by confidence threshold
            predictions = {}
            for i, (class_name, score) in enumerate(zip(class_names, scores)):
                if score >= confidence_threshold:
                    predictions[class_name] = float(score)
            
            # Get top-k predictions
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
            predictions_dict = dict(sorted_predictions)
            
            processing_time = time.time() - start_time
            
            # Create visualization
            plot = self._create_confidence_plot(predictions_dict)
            
            # Log prediction
            self._log_prediction(
                inputs=(image.size, confidence_threshold, top_k),
                outputs=predictions_dict,
                processing_time=processing_time
            )
            
            return (
                predictions_dict,
                f"{processing_time:.2f}s",
                plot
            )
        
        except Exception as e:
            error_message, error_image = self.error_handler.handle_error(e, "classification_predict")
            processing_time = time.time() - start_time
            
            return (
                {"error": 1.0},
                f"{processing_time:.2f}s",
                self._create_error_plot(error_message)
            )
    
    def _create_confidence_plot(self, predictions: Dict[str, float]) -> plt.Figure:
        """Create confidence score visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        classes = list(predictions.keys())
        scores = list(predictions.values())
        
        bars = ax.barh(classes, scores, color='skyblue')
        ax.set_xlabel('Confidence Score')
        ax.set_title('Classification Confidence Scores')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center')
        
        plt.tight_layout()
        return fig
    
    def _create_error_plot(self, message: str) -> plt.Figure:
        """Create error visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {message}", ha='center', va='center', 
               transform=ax.transAxes, fontsize=14, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def create_gradio_app(config: GradioConfig) -> BaseGradioApp:
    """Create Gradio app based on model type."""
    if config.model_type == "transformer":
        return TransformerGradioApp(config)
    elif config.model_type == "diffusion":
        return DiffusionGradioApp(config)
    elif config.model_type == "classification":
        return ClassificationGradioApp(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# Example usage
if __name__ == "__main__":
    # Create configuration for transformer app
    transformer_config = GradioConfig(
        app_title="Transformer Text Generation",
        app_description="Generate text using transformer models",
        model_type="transformer",
        server_port=7860,
        show_examples=True,
        show_advanced_options=True
    )
    
    # Create and launch transformer app
    transformer_app = create_gradio_app(transformer_config)
    transformer_app.launch()
    
    # Create configuration for diffusion app
    diffusion_config = GradioConfig(
        app_title="Diffusion Image Generation",
        app_description="Generate images using diffusion models",
        model_type="diffusion",
        server_port=7861,
        show_examples=True,
        show_advanced_options=True
    )
    
    # Create and launch diffusion app
    diffusion_app = create_gradio_app(diffusion_config)
    diffusion_app.launch()
    
    # Create configuration for classification app
    classification_config = GradioConfig(
        app_title="Image Classification",
        app_description="Classify images using deep learning models",
        model_type="classification",
        server_port=7862,
        show_examples=True,
        show_advanced_options=True
    )
    
    # Create and launch classification app
    classification_app = create_gradio_app(classification_config)
    classification_app.launch() 