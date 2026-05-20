from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
from diffusion_pipelines import (
from diffusion_training_evaluation import (
from gradient_optimization import (
from advanced_data_loading import (
            import cv2
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Gradio Interactive Demo for Diffusion Models
===========================================

Comprehensive Gradio interface for diffusion model inference and visualization:
- Multiple pipeline types (Stable Diffusion, SDXL, Img2Img, Inpaint, ControlNet)
- Real-time image generation and editing
- Advanced parameter controls
- Batch processing and comparison
- Training monitoring and visualization
- Model performance analysis

Features: Interactive UI, real-time generation, parameter tuning,
batch processing, and comprehensive visualization tools.
"""


# Import our diffusion components
    DiffusionPipelineManager, PipelineConfig, GenerationRequest
)
    DiffusionTrainer, TrainingConfig, DiffusionEvaluator, EvaluationConfig
)
    TrainingStabilityManager, GradientConfig
)
    AdvancedDataLoader, DataConfig
)

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for the Gradio demo."""
    # Model settings
    default_model: str = "runwayml/stable-diffusion-v1-5"
    xl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "lllyasviel/control_v11p_sd15_canny"
    
    # Generation settings
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    default_height: int = 512
    default_width: int = 512
    
    # UI settings
    max_batch_size: int = 4
    max_prompt_length: int = 500
    enable_advanced_controls: bool = True
    enable_training_demo: bool = True
    
    # Performance settings
    use_half_precision: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True


class DiffusionDemo:
    """Main demo class for diffusion model inference."""
    
    def __init__(self, config: DemoConfig):
        
    """__init__ function."""
self.config = config
        self.pipeline_manager = None
        self.current_pipeline = None
        self.current_pipeline_key = None
        
        # Initialize pipeline manager
        self._initialize_pipelines()
        
        # Demo state
        self.generation_history = []
        self.training_stats = {}
        
    def _initialize_pipelines(self) -> Any:
        """Initialize pipeline manager and load models."""
        try:
            pipeline_config = PipelineConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_attention_slicing=self.config.enable_attention_slicing,
                enable_vae_slicing=self.config.enable_vae_slicing,
                enable_xformers_memory_efficient_attention=True
            )
            
            self.pipeline_manager = DiffusionPipelineManager(pipeline_config)
            logger.info("Pipeline manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline manager: {e}")
            self.pipeline_manager = None
    
    async def load_pipeline(self, pipeline_type: str, model_name: Optional[str] = None) -> str:
        """Load a specific pipeline type."""
        if not self.pipeline_manager:
            return "Pipeline manager not available"
        
        try:
            if pipeline_type == "stable_diffusion":
                pipeline_key = await self.pipeline_manager.load_stable_diffusion_pipeline(model_name)
            elif pipeline_type == "stable_diffusion_xl":
                pipeline_key = await self.pipeline_manager.load_stable_diffusion_xl_pipeline()
            elif pipeline_type == "img2img":
                pipeline_key = await self.pipeline_manager.load_img2img_pipeline(model_name)
            elif pipeline_type == "inpaint":
                pipeline_key = await self.pipeline_manager.load_inpaint_pipeline(model_name)
            elif pipeline_type == "controlnet":
                pipeline_key = await self.pipeline_manager.load_controlnet_pipeline(
                    model_name or self.config.default_model,
                    self.config.controlnet_model
                )
            else:
                return f"Unknown pipeline type: {pipeline_type}"
            
            self.current_pipeline_key = pipeline_key
            self.current_pipeline = self.pipeline_manager.pipelines[pipeline_key]
            
            return f"Successfully loaded {pipeline_type} pipeline"
            
        except Exception as e:
            return f"Failed to load pipeline: {str(e)}"
    
    async def generate_image(self, prompt: str, negative_prompt: str = "",
                           num_steps: int = 30, guidance_scale: float = 7.5,
                           height: int = 512, width: int = 512,
                           seed: Optional[int] = None) -> Tuple[Image.Image, str]:
        """Generate a single image."""
        if not self.current_pipeline:
            return None, "No pipeline loaded"
        
        try:
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=seed
            )
            
            # Generate image
            start_time = time.time()
            images = await self.pipeline_manager.generate_image(
                self.current_pipeline_key, request
            )
            generation_time = time.time() - start_time
            
            if not images:
                return None, "No images generated"
            
            # Save to history
            self.generation_history.append({
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'parameters': {
                    'steps': num_steps,
                    'guidance_scale': guidance_scale,
                    'height': height,
                    'width': width,
                    'seed': seed
                },
                'generation_time': generation_time,
                'timestamp': time.time()
            })
            
            return images[0], f"Generated in {generation_time:.2f}s"
            
        except Exception as e:
            return None, f"Generation failed: {str(e)}"
    
    async def generate_batch(self, prompts: List[str], negative_prompts: List[str],
                           num_steps: int = 30, guidance_scale: float = 7.5,
                           height: int = 512, width: int = 512,
                           seed: Optional[int] = None) -> Tuple[List[Image.Image], str]:
        """Generate multiple images in batch."""
        if not self.current_pipeline:
            return [], "No pipeline loaded"
        
        try:
            # Create batch requests
            requests = []
            for prompt, negative_prompt in zip(prompts, negative_prompts):
                request = GenerationRequest(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    seed=seed
                )
                requests.append(request)
            
            # Generate batch
            start_time = time.time()
            batch_results = await self.pipeline_manager.batch_generate(
                self.current_pipeline_key, requests
            )
            generation_time = time.time() - start_time
            
            # Flatten results
            images = []
            for batch in batch_results:
                images.extend(batch)
            
            return images, f"Generated {len(images)} images in {generation_time:.2f}s"
            
        except Exception as e:
            return [], f"Batch generation failed: {str(e)}"
    
    async def img2img_generation(self, image: Image.Image, prompt: str,
                               negative_prompt: str = "", strength: float = 0.8,
                               num_steps: int = 30, guidance_scale: float = 7.5) -> Tuple[Image.Image, str]:
        """Generate image-to-image transformation."""
        if not self.current_pipeline:
            return None, "No img2img pipeline loaded"
        
        try:
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            start_time = time.time()
            images = await self.pipeline_manager.generate_image(
                self.current_pipeline_key, request
            )
            generation_time = time.time() - start_time
            
            if not images:
                return None, "No images generated"
            
            return images[0], f"Transformed in {generation_time:.2f}s"
            
        except Exception as e:
            return None, f"Img2Img generation failed: {str(e)}"
    
    async def inpaint_generation(self, image: Image.Image, mask: Image.Image,
                               prompt: str, negative_prompt: str = "",
                               num_steps: int = 30, guidance_scale: float = 7.5) -> Tuple[Image.Image, str]:
        """Generate inpainting."""
        if not self.current_pipeline:
            return None, "No inpaint pipeline loaded"
        
        try:
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            start_time = time.time()
            images = await self.pipeline_manager.generate_image(
                self.current_pipeline_key, request
            )
            generation_time = time.time() - start_time
            
            if not images:
                return None, "No images generated"
            
            return images[0], f"Inpainted in {generation_time:.2f}s"
            
        except Exception as e:
            return None, f"Inpainting failed: {str(e)}"
    
    def create_control_image(self, image: Image.Image, control_type: str = "canny") -> Image.Image:
        """Create control image for ControlNet."""
        if control_type == "canny":
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply Canny edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Convert back to PIL
            control_img = Image.fromarray(edges)
            return control_img
        
        elif control_type == "depth":
            # Simple depth approximation (placeholder)
            img_array = np.array(image)
            gray = np.mean(img_array, axis=2)
            depth = (255 - gray).astype(np.uint8)
            return Image.fromarray(depth)
        
        else:
            return image
    
    async def controlnet_generation(self, control_image: Image.Image, prompt: str,
                                  negative_prompt: str = "", control_type: str = "canny",
                                  control_scale: float = 1.0, num_steps: int = 30,
                                  guidance_scale: float = 7.5) -> Tuple[Image.Image, str]:
        """Generate with ControlNet."""
        if not self.current_pipeline:
            return None, "No ControlNet pipeline loaded"
        
        try:
            # Create control image
            processed_control = self.create_control_image(control_image, control_type)
            
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=processed_control,
                controlnet_conditioning_scale=control_scale,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            start_time = time.time()
            images = await self.pipeline_manager.generate_image(
                self.current_pipeline_key, request
            )
            generation_time = time.time() - start_time
            
            if not images:
                return None, "No images generated"
            
            return images[0], f"Controlled generation in {generation_time:.2f}s"
            
        except Exception as e:
            return None, f"ControlNet generation failed: {str(e)}"
    
    def create_parameter_comparison(self, base_image: Image.Image, 
                                  parameter_name: str, values: List[float]) -> List[Image.Image]:
        """Create parameter comparison visualization."""
        # This would generate images with different parameter values
        # For demo purposes, we'll create a simple visualization
        images = []
        
        for i, value in enumerate(values):
            # Create a simple visualization showing parameter effect
            img = base_image.copy()
            draw = ImageDraw.Draw(img)
            
            # Add parameter info
            text = f"{parameter_name}: {value}"
            draw.text((10, 10), text, fill=(255, 255, 255))
            
            # Add border with color based on value
            color = int(255 * (i / len(values)))
            draw.rectangle([0, 0, img.width-1, img.height-1], outline=(color, 255-color, 0), width=3)
            
            images.append(img)
        
        return images
    
    def create_training_visualization(self, training_stats: Dict[str, Any]) -> Image.Image:
        """Create training visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        if 'losses' in training_stats:
            axes[0, 0].plot(training_stats['losses'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
        
        # Gradient norm plot
        if 'grad_norms' in training_stats:
            axes[0, 1].plot(training_stats['grad_norms'])
            axes[0, 1].set_title('Gradient Norm')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Norm')
        
        # Learning rate plot
        if 'learning_rates' in training_stats:
            axes[1, 0].plot(training_stats['learning_rates'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('LR')
        
        # Stability score plot
        if 'stability_scores' in training_stats:
            axes[1, 1].plot(training_stats['stability_scores'])
            axes[1, 1].set_title('Stability Score')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return Image.fromarray(img_array)


def create_gradio_interface():
    """Create the Gradio interface."""
    config = DemoConfig()
    demo = DiffusionDemo(config)
    
    # Define the interface
    with gr.Blocks(title="Advanced Diffusion Model Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Advanced Diffusion Model Demo")
        gr.Markdown("Interactive demo for diffusion model inference and visualization")
        
        with gr.Tabs():
            # Tab 1: Basic Generation
            with gr.TabItem("🖼️ Image Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Pipeline Selection")
                        pipeline_type = gr.Dropdown(
                            choices=["stable_diffusion", "stable_diffusion_xl", "img2img", "inpaint", "controlnet"],
                            value="stable_diffusion",
                            label="Pipeline Type"
                        )
                        model_name = gr.Textbox(
                            value=config.default_model,
                            label="Model Name (optional)"
                        )
                        load_btn = gr.Button("Load Pipeline", variant="primary")
                        load_status = gr.Textbox(label="Load Status", interactive=False)
                        
                        gr.Markdown("### Generation Parameters")
                        prompt = gr.Textbox(
                            value="A beautiful landscape with mountains and lake, digital art",
                            label="Prompt",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            value="blurry, low quality, distorted",
                            label="Negative Prompt",
                            lines=2
                        )
                        
                        with gr.Row():
                            num_steps = gr.Slider(1, 100, config.default_steps, label="Steps")
                            guidance_scale = gr.Slider(1.0, 20.0, config.default_guidance_scale, label="Guidance Scale")
                        
                        with gr.Row():
                            height = gr.Slider(256, 1024, config.default_height, step=64, label="Height")
                            width = gr.Slider(256, 1024, config.default_width, step=64, label="Width")
                        
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        generate_btn = gr.Button("Generate Image", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Image")
                        output_image = gr.Image(label="Generated Image")
                        generation_status = gr.Textbox(label="Generation Status", interactive=False)
                        
                        gr.Markdown("### Generation History")
                        history_gallery = gr.Gallery(
                            label="Recent Generations",
                            show_label=True,
                            elem_id="gallery"
                        )
            
            # Tab 2: Batch Generation
            with gr.TabItem("📦 Batch Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Parameters")
                        batch_prompts = gr.Textbox(
                            value="A beautiful sunset\nA majestic mountain\nA serene lake\nA cozy cottage",
                            label="Prompts (one per line)",
                            lines=6
                        )
                        batch_negative_prompt = gr.Textbox(
                            value="blurry, low quality",
                            label="Negative Prompt (applied to all)",
                            lines=2
                        )
                        
                        with gr.Row():
                            batch_steps = gr.Slider(1, 100, config.default_steps, label="Steps")
                            batch_guidance = gr.Slider(1.0, 20.0, config.default_guidance_scale, label="Guidance Scale")
                        
                        batch_seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        batch_generate_btn = gr.Button("Generate Batch", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Results")
                        batch_output = gr.Gallery(label="Batch Results")
                        batch_status = gr.Textbox(label="Batch Status", interactive=False)
            
            # Tab 3: Image-to-Image
            with gr.TabItem("🔄 Image-to-Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Image")
                        input_image = gr.Image(label="Input Image", type="pil")
                        
                        gr.Markdown("### Transformation Parameters")
                        img2img_prompt = gr.Textbox(
                            value="Turn this into a painting",
                            label="Prompt",
                            lines=2
                        )
                        img2img_negative = gr.Textbox(
                            value="blurry, low quality",
                            label="Negative Prompt",
                            lines=2
                        )
                        strength = gr.Slider(0.1, 1.0, 0.8, label="Strength")
                        
                        with gr.Row():
                            img2img_steps = gr.Slider(1, 100, config.default_steps, label="Steps")
                            img2img_guidance = gr.Slider(1.0, 20.0, config.default_guidance_scale, label="Guidance Scale")
                        
                        img2img_btn = gr.Button("Transform Image", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Transformed Image")
                        img2img_output = gr.Image(label="Transformed Image")
                        img2img_status = gr.Textbox(label="Transformation Status", interactive=False)
            
            # Tab 4: Inpainting
            with gr.TabItem("🎨 Inpainting"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Image and Mask")
                        inpaint_image = gr.Image(label="Input Image", type="pil")
                        inpaint_mask = gr.Image(label="Mask (white = inpaint area)", type="pil")
                        
                        gr.Markdown("### Inpainting Parameters")
                        inpaint_prompt = gr.Textbox(
                            value="A beautiful flower",
                            label="Prompt",
                            lines=2
                        )
                        inpaint_negative = gr.Textbox(
                            value="blurry, low quality",
                            label="Negative Prompt",
                            lines=2
                        )
                        
                        with gr.Row():
                            inpaint_steps = gr.Slider(1, 100, config.default_steps, label="Steps")
                            inpaint_guidance = gr.Slider(1.0, 20.0, config.default_guidance_scale, label="Guidance Scale")
                        
                        inpaint_btn = gr.Button("Inpaint", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Inpainted Result")
                        inpaint_output = gr.Image(label="Inpainted Image")
                        inpaint_status = gr.Textbox(label="Inpainting Status", interactive=False)
            
            # Tab 5: ControlNet
            with gr.TabItem("🎛️ ControlNet"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Control Image")
                        control_image = gr.Image(label="Control Image", type="pil")
                        
                        gr.Markdown("### Control Parameters")
                        control_prompt = gr.Textbox(
                            value="A beautiful landscape",
                            label="Prompt",
                            lines=2
                        )
                        control_negative = gr.Textbox(
                            value="blurry, low quality",
                            label="Negative Prompt",
                            lines=2
                        )
                        
                        control_type = gr.Dropdown(
                            choices=["canny", "depth"],
                            value="canny",
                            label="Control Type"
                        )
                        control_scale = gr.Slider(0.1, 2.0, 1.0, label="Control Scale")
                        
                        with gr.Row():
                            control_steps = gr.Slider(1, 100, config.default_steps, label="Steps")
                            control_guidance = gr.Slider(1.0, 20.0, config.default_guidance_scale, label="Guidance Scale")
                        
                        control_btn = gr.Button("Generate with Control", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Controlled Generation")
                        control_output = gr.Image(label="Controlled Image")
                        control_status = gr.Textbox(label="Control Status", interactive=False)
            
            # Tab 6: Training Demo
            with gr.TabItem("🏋️ Training Demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Configuration")
                        training_config = gr.JSON(
                            value={
                                "learning_rate": 1e-4,
                                "num_epochs": 10,
                                "batch_size": 1,
                                "gradient_clipping": True,
                                "early_stopping": True
                            },
                            label="Training Config"
                        )
                        
                        start_training_btn = gr.Button("Start Training Demo", variant="primary")
                        stop_training_btn = gr.Button("Stop Training", variant="secondary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Progress")
                        training_progress = gr.Plot(label="Training Progress")
                        training_status = gr.Textbox(label="Training Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=demo.load_pipeline,
            inputs=[pipeline_type, model_name],
            outputs=[load_status]
        )
        
        generate_btn.click(
            fn=demo.generate_image,
            inputs=[prompt, negative_prompt, num_steps, guidance_scale, height, width, seed],
            outputs=[output_image, generation_status]
        )
        
        batch_generate_btn.click(
            fn=lambda prompts, neg_prompt, steps, guidance, seed: demo.generate_batch(
                prompts.split('\n'), [neg_prompt] * len(prompts.split('\n')), steps, guidance, 512, 512, seed
            ),
            inputs=[batch_prompts, batch_negative_prompt, batch_steps, batch_guidance, batch_seed],
            outputs=[batch_output, batch_status]
        )
        
        img2img_btn.click(
            fn=demo.img2img_generation,
            inputs=[input_image, img2img_prompt, img2img_negative, strength, img2img_steps, img2img_guidance],
            outputs=[img2img_output, img2img_status]
        )
        
        inpaint_btn.click(
            fn=demo.inpaint_generation,
            inputs=[inpaint_image, inpaint_mask, inpaint_prompt, inpaint_negative, inpaint_steps, inpaint_guidance],
            outputs=[inpaint_output, inpaint_status]
        )
        
        control_btn.click(
            fn=demo.controlnet_generation,
            inputs=[control_image, control_prompt, control_negative, control_type, control_scale, control_steps, control_guidance],
            outputs=[control_output, control_status]
        )
        
        # Update history gallery
        def update_history():
            
    """update_history function."""
if demo.generation_history:
                # Return recent images (placeholder - would need to store actual images)
                return [Image.new('RGB', (512, 512), (100, 100, 100)) for _ in range(min(5, len(demo.generation_history)))]
            return []
        
        generate_btn.click(
            fn=update_history,
            outputs=[history_gallery]
        )
    
    return interface


def main():
    """Launch the Gradio demo."""
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )


match __name__:
    case "__main__":
    main() 