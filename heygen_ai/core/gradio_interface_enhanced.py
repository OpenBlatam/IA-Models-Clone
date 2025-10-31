"""
Enhanced Gradio Interface for HeyGen AI.

This module provides a comprehensive Gradio interface for interacting with
transformer models, diffusion models, and other AI components. Includes
proper error handling, input validation, and user experience features.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import warnings

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gradio as gr
from gradio import Blocks, Interface, Tab, Row, Column, Group
from gradio.components import (
    Textbox, Image, Slider, Dropdown, Checkbox, Button, 
    Number, Radio, File, Video, Audio, Gallery, Plot
)

# Import our enhanced modules
from .transformer_models_enhanced import TransformerManager, TransformerConfig
from .diffusion_models_enhanced import DiffusionPipelineManager, DiffusionConfig
from .model_training_enhanced import ModelTrainer, TrainingConfig

logger = logging.getLogger(__name__)


class GradioInterfaceManager:
    """Manages the Gradio interface for HeyGen AI components."""
    
    def __init__(self):
        """Initialize the Gradio interface manager."""
        self.transformer_manager = None
        self.diffusion_manager = None
        self.current_model = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize interface components
        self._setup_interface()
    
    def _setup_interface(self):
        """Set up the main Gradio interface."""
        with gr.Blocks(
            title="HeyGen AI - Enhanced Interface",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            .main-header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            """
        ) as self.interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🚀 HeyGen AI - Enhanced Interface</h1>
                <p>Advanced AI models for text generation, image creation, and more</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                # Text Generation Tab
                with gr.Tab("📝 Text Generation", id=0):
                    self._create_text_generation_tab()
                
                # Image Generation Tab
                with gr.Tab("🎨 Image Generation", id=1):
                    self._create_image_generation_tab()
                
                # Model Training Tab
                with gr.Tab("🏋️ Model Training", id=2):
                    self._create_model_training_tab()
                
                # Settings Tab
                with gr.Tab("⚙️ Settings", id=3):
                    self._create_settings_tab()
    
    def _create_text_generation_tab(self):
        """Create the text generation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                    value="gpt2",
                    label="Select Model",
                    info="Choose a pre-trained transformer model"
                )
                
                # Input prompt
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=4,
                    max_lines=10
                )
                
                # Generation parameters
                with gr.Group("Generation Parameters"):
                    max_length = gr.Slider(
                        minimum=10, maximum=500, value=100, step=10,
                        label="Maximum Length",
                        info="Maximum number of tokens to generate"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature",
                        info="Controls randomness in generation (lower = more focused)"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                        label="Top-p (Nucleus Sampling)",
                        info="Controls diversity by considering only top-p probability mass"
                    )
                    
                    do_sample = gr.Checkbox(
                        label="Use Sampling", value=True,
                        info="Enable sampling instead of greedy decoding"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🚀 Generate Text",
                    variant="primary",
                    size="lg"
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column(scale=2):
                # Output
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                # Generation info
                generation_info = gr.JSON(
                    label="Generation Information",
                    visible=False
                )
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_text,
            inputs=[model_dropdown, prompt_input, max_length, temperature, top_p, do_sample],
            outputs=[output_text, generation_info]
        )
        
        clear_btn.click(
            fn=self._clear_text_generation,
            inputs=[],
            outputs=[prompt_input, output_text, generation_info]
        )
    
    def _create_image_generation_tab(self):
        """Create the image generation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=["stable_diffusion", "stable_diffusion_xl", "text_to_video"],
                    value="stable_diffusion",
                    label="Select Model",
                    info="Choose a diffusion model type"
                )
                
                # Input prompt
                prompt_input = gr.Textbox(
                    label="Image Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    max_lines=5
                )
                
                # Negative prompt
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Describe what you don't want in the image...",
                    lines=2,
                    max_lines=3
                )
                
                # Generation parameters
                with gr.Group("Image Parameters"):
                    width = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Width",
                        info="Image width in pixels"
                    )
                    
                    height = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Height",
                        info="Image height in pixels"
                    )
                    
                    num_images = gr.Slider(
                        minimum=1, maximum=4, value=1, step=1,
                        label="Number of Images",
                        info="How many images to generate"
                    )
                    
                    num_steps = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="Inference Steps",
                        info="More steps = higher quality but slower"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt (higher = more faithful)"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    size="lg"
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column(scale=2):
                # Output gallery
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
                
                # Generation info
                generation_info = gr.JSON(
                    label="Generation Information",
                    visible=False
                )
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_image,
            inputs=[model_dropdown, prompt_input, negative_prompt, width, height, 
                   num_images, num_steps, guidance_scale],
            outputs=[output_gallery, generation_info]
        )
        
        clear_btn.click(
            fn=self._clear_image_generation,
            inputs=[],
            outputs=[prompt_input, negative_prompt, output_gallery, generation_info]
        )
    
    def _create_model_training_tab(self):
        """Create the model training tab."""
        with gr.Row():
            with gr.Column(scale=2):
                # Training configuration
                with gr.Group("Training Configuration"):
                    model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="Enter model name or path",
                        value="gpt2"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1, maximum=32, value=8, step=1,
                        label="Batch Size",
                        info="Training batch size"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=1e-6, maximum=1e-3, value=5e-5, step=1e-6,
                        label="Learning Rate",
                        info="Training learning rate"
                    )
                    
                    num_epochs = gr.Slider(
                        minimum=1, maximum=100, value=10, step=1,
                        label="Number of Epochs",
                        info="Total training epochs"
                    )
                    
                    use_fp16 = gr.Checkbox(
                        label="Use FP16 (Mixed Precision)",
                        value=True,
                        info="Enable mixed precision training for speed"
                    )
                
                # Data upload
                with gr.Group("Training Data"):
                    data_file = gr.File(
                        label="Upload Training Data",
                        file_types=[".txt", ".json", ".csv"],
                        file_count="single"
                    )
                    
                    data_preview = gr.Textbox(
                        label="Data Preview",
                        lines=5,
                        interactive=False
                    )
                
                # Training controls
                with gr.Group("Training Controls"):
                    start_training_btn = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        size="lg"
                    )
                    
                    stop_training_btn = gr.Button(
                        "⏹️ Stop Training",
                        variant="stop",
                        size="lg"
                    )
                    
                    resume_training_btn = gr.Button(
                        "▶️ Resume Training",
                        variant="secondary"
                    )
            
            with gr.Column(scale=2):
                # Training progress
                training_progress = gr.Plot(
                    label="Training Progress",
                    show_label=True
                )
                
                # Training logs
                training_logs = gr.Textbox(
                    label="Training Logs",
                    lines=15,
                    max_lines=30,
                    interactive=False
                )
                
                # Model download
                model_download = gr.File(
                    label="Trained Model",
                    visible=False
                )
        
        # Event handlers
        data_file.change(
            fn=self._preview_data,
            inputs=[data_file],
            outputs=[data_preview]
        )
        
        start_training_btn.click(
            fn=self._start_training,
            inputs=[model_name, batch_size, learning_rate, num_epochs, use_fp16, data_file],
            outputs=[training_progress, training_logs, model_download]
        )
    
    def _create_settings_tab(self):
        """Create the settings tab."""
        with gr.Row():
            with gr.Column(scale=1):
                # Model settings
                with gr.Group("Model Settings"):
                    device_selection = gr.Radio(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="Device Selection",
                        info="Choose computation device"
                    )
                    
                    model_cache_dir = gr.Textbox(
                        label="Model Cache Directory",
                        placeholder="/path/to/model/cache",
                        value=os.path.expanduser("~/.cache/huggingface")
                    )
                    
                    enable_logging = gr.Checkbox(
                        label="Enable Detailed Logging",
                        value=True
                    )
                
                # Performance settings
                with gr.Group("Performance Settings"):
                    enable_attention_slicing = gr.Checkbox(
                        label="Enable Attention Slicing",
                        value=True,
                        info="Reduce memory usage for large models"
                    )
                    
                    enable_vae_slicing = gr.Checkbox(
                        label="Enable VAE Slicing",
                        value=True,
                        info="Reduce memory usage for image generation"
                    )
                    
                    enable_xformers = gr.Checkbox(
                        label="Enable xFormers",
                        value=True,
                        info="Use xFormers for memory efficient attention"
                    )
            
            with gr.Column(scale=1):
                # System information
                with gr.Group("System Information"):
                    system_info = gr.JSON(
                        label="System Details",
                        value=self._get_system_info()
                    )
                
                # Model information
                with gr.Group("Model Information"):
                    model_info = gr.JSON(
                        label="Loaded Models",
                        value=self._get_model_info()
                    )
                
                # Actions
                with gr.Group("Actions"):
                    refresh_btn = gr.Button("🔄 Refresh Info", variant="secondary")
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
        
        # Event handlers
        refresh_btn.click(
            fn=self._refresh_info,
            inputs=[],
            outputs=[system_info, model_info]
        )
        
        clear_cache_btn.click(
            fn=self._clear_cache,
            inputs=[],
            outputs=[system_info, model_info]
        )
    
    def _generate_text(self, model_name: str, prompt: str, max_length: int,
                       temperature: float, top_p: float, do_sample: bool) -> Tuple[str, Dict]:
        """Generate text using the selected model."""
        try:
            if not prompt.strip():
                return "", {"error": "Please provide a prompt"}
            
            # Initialize transformer manager if needed
            if self.transformer_manager is None or self.current_model != model_name:
                config = TransformerConfig(model_name=model_name)
                self.transformer_manager = TransformerManager(config)
                self.transformer_manager.load_pretrained_model()
                self.current_model = model_name
            
            # Generate text
            start_time = time.time()
            generated_text = self.transformer_manager.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            generation_time = time.time() - start_time
            
            # Prepare generation info
            info = {
                "model": model_name,
                "prompt_length": len(prompt),
                "generated_length": len(generated_text),
                "generation_time": f"{generation_time:.2f}s",
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample
                }
            }
            
            return generated_text, info
            
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            logger.error(error_msg)
            return "", {"error": error_msg}
    
    def _generate_image(self, model_type: str, prompt: str, negative_prompt: str,
                       width: int, height: int, num_images: int, num_steps: int,
                       guidance_scale: float) -> Tuple[List, Dict]:
        """Generate images using the selected diffusion model."""
        try:
            if not prompt.strip():
                return [], {"error": "Please provide a prompt"}
            
            # Initialize diffusion manager if needed
            if self.diffusion_manager is None:
                config = DiffusionConfig(
                    model_type=model_type,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale
                )
                self.diffusion_manager = DiffusionPipelineManager(config)
            
            # Generate images
            start_time = time.time()
            images = self.diffusion_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=num_images
            )
            generation_time = time.time() - start_time
            
            # Convert PIL images to format suitable for Gradio
            image_list = []
            for img in images:
                image_list.append(img)
            
            # Prepare generation info
            info = {
                "model_type": model_type,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "dimensions": f"{width}x{height}",
                "num_images": num_images,
                "generation_time": f"{generation_time:.2f}s",
                "parameters": {
                    "num_steps": num_steps,
                    "guidance_scale": guidance_scale
                }
            }
            
            return image_list, info
            
        except Exception as e:
            error_msg = f"Error generating images: {str(e)}"
            logger.error(error_msg)
            return [], {"error": error_msg}
    
    def _start_training(self, model_name: str, batch_size: int, learning_rate: float,
                        num_epochs: int, use_fp16: bool, data_file) -> Tuple[Any, str, Any]:
        """Start model training."""
        try:
            if not data_file:
                return None, "Please upload training data", None
            
            # This is a placeholder for actual training implementation
            # In practice, you'd implement the full training pipeline here
            training_log = f"""
            Starting training with:
            - Model: {model_name}
            - Batch size: {batch_size}
            - Learning rate: {learning_rate}
            - Epochs: {num_epochs}
            - FP16: {use_fp16}
            - Data file: {data_file.name}
            
            Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return None, training_log, None
            
        except Exception as e:
            error_msg = f"Error starting training: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, None
    
    def _preview_data(self, data_file) -> str:
        """Preview uploaded training data."""
        try:
            if not data_file:
                return "No file uploaded"
            
            # Read and preview the first few lines
            with open(data_file.name, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]
                preview = ''.join(lines)
            
            return f"Data preview (first 10 lines):\n\n{preview}"
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "python_version": f"{torch.version.python}",
            "pytorch_version": f"{torch.__version__}",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU"
        }
    
    def _get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            "transformer_model": self.current_model if self.transformer_manager else None,
            "diffusion_model": "Initialized" if self.diffusion_manager else None
        }
        return info
    
    def _refresh_info(self) -> Tuple[Dict, Dict]:
        """Refresh system and model information."""
        return self._get_system_info(), self._get_model_info()
    
    def _clear_cache(self) -> Tuple[Dict, Dict]:
        """Clear model cache."""
        try:
            # This is a placeholder for cache clearing logic
            logger.info("Cache cleared")
            return self._get_system_info(), self._get_model_info()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return self._get_system_info(), self._get_model_info()
    
    def _clear_text_generation(self) -> Tuple[str, str, Dict]:
        """Clear text generation inputs and outputs."""
        return "", "", {}
    
    def _clear_image_generation(self) -> Tuple[str, str, List, Dict]:
        """Clear image generation inputs and outputs."""
        return "", "", [], {}
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        return self.interface.launch(**kwargs)


def create_demo_interface():
    """Create and return a demo interface."""
    manager = GradioInterfaceManager()
    return manager


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_demo_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
