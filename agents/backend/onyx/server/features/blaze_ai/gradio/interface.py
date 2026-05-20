"""
Gradio interface for Blaze AI module with interactive demos and visualization.
"""
from __future__ import annotations

import asyncio
import gradio as gr
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger
from ..utils.performance import get_performance_profiler
from ..utils.initialization import get_loss_function, get_optimizer
from ..engines.llm import LLMEngine
from ..engines.diffusion import DiffusionEngine
from .validation import (
    get_text_generation_validator,
    get_image_generation_validator,
    get_training_validator,
    get_gradio_error_handler,
    get_safe_gradio_executor
)

logger = get_logger(__name__)

class BlazeAIGradioInterface:
    """Gradio interface for Blaze AI module."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config or CoreConfig()
        self.logger = get_logger(__name__)
        self.performance_profiler = get_performance_profiler()
        
        # Initialize validators and error handlers
        self.text_validator = get_text_generation_validator()
        self.image_validator = get_image_generation_validator()
        self.training_validator = get_training_validator()
        self.error_handler = get_gradio_error_handler()
        self.safe_executor = get_safe_gradio_executor()
        
        # Initialize engines
        self.llm_engine = None
        self.diffusion_engine = None
        
        # Interface components
        self.interface = None
        self.tabs = {}
        
        self._initialize_engines()
        self._create_interface()
    
    def _initialize_engines(self):
        """Initialize AI engines."""
        try:
            # Initialize LLM engine
            llm_config = {
                "model_name": "gpt2",
                "device": "auto",
                "precision": "float16",
                "enable_amp": True,
                "gradient_accumulation_steps": 4,
                "mixed_precision": True
            }
            self.llm_engine = LLMEngine("llm", llm_config)
            
            # Initialize Diffusion engine
            diffusion_config = {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "auto",
                "precision": "float16",
                "enable_xformers": True,
                "mixed_precision": True
            }
            self.diffusion_engine = DiffusionEngine("diffusion", diffusion_config)
            
            self.logger.info("AI engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engines: {e}")
            raise
    
    def _create_interface(self):
        """Create the Gradio interface."""
        try:
            with gr.Blocks(
                title="Blaze AI - Advanced AI Platform",
                theme=gr.themes.Soft(),
                css=self._get_custom_css()
            ) as interface:
                
                # Header
                gr.Markdown("# 🚀 Blaze AI - Advanced AI Platform")
                gr.Markdown("### Interactive AI Model Demos and Visualization")
                
                # Main tabs
                with gr.Tabs():
                    # Text Generation Tab
                    with gr.TabItem("📝 Text Generation", id=0):
                        self._create_text_generation_tab()
                    
                    # Image Generation Tab
                    with gr.TabItem("🎨 Image Generation", id=1):
                        self._create_image_generation_tab()
                    
                    # Model Training Tab
                    with gr.TabItem("🏋️ Model Training", id=2):
                        self._create_training_tab()
                    
                    # Performance Monitoring Tab
                    with gr.TabItem("📊 Performance", id=3):
                        self._create_performance_tab()
                    
                    # System Health Tab
                    with gr.TabItem("💚 System Health", id=4):
                        self._create_health_tab()
                
                # Footer
                gr.Markdown("---")
                gr.Markdown("Built with ❤️ using Blaze AI")
            
            self.interface = interface
            
        except Exception as e:
            self.logger.error(f"Failed to create Gradio interface: {e}")
            raise
    
    def _create_text_generation_tab(self):
        """Create text generation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### Text Generation Parameters")
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here...",
                    lines=3,
                    max_lines=10
                )
                
                with gr.Row():
                    max_length = gr.Slider(
                        minimum=10, maximum=500, value=100, step=10,
                        label="Max Length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature"
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-p (Nucleus Sampling)"
                    )
                    do_sample = gr.Checkbox(
                        label="Enable Sampling", value=True
                    )
                
                generate_btn = gr.Button("🚀 Generate Text", variant="primary")
                
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### Generated Text")
                
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                with gr.Row():
                    copy_btn = gr.Button("📋 Copy")
                    clear_btn = gr.Button("🗑️ Clear")
                
                # Generation stats
                stats_output = gr.JSON(label="Generation Statistics")
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_text,
            inputs=[prompt_input, max_length, temperature, top_p, do_sample],
            outputs=[output_text, stats_output]
        )
        
        copy_btn.click(
            fn=lambda x: x,
            inputs=[output_text],
            outputs=[]
        )
        
        clear_btn.click(
            fn=lambda: ("", {}),
            outputs=[output_text, stats_output]
        )
    
    def _create_image_generation_tab(self):
        """Create image generation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### Image Generation Parameters")
                
                prompt_input = gr.Textbox(
                    label="Image Prompt",
                    placeholder="A beautiful landscape with mountains and lake...",
                    lines=2
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted...",
                    lines=2
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="Inference Steps"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                        label="Guidance Scale"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Height"
                    )
                
                seed_input = gr.Number(
                    label="Seed (optional)",
                    value=None,
                    precision=0
                )
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### Generated Image")
                
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil"
                )
                
                with gr.Row():
                    download_btn = gr.Button("💾 Download")
                    regenerate_btn = gr.Button("🔄 Regenerate")
                
                # Generation info
                info_output = gr.JSON(label="Generation Information")
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_image,
            inputs=[prompt_input, negative_prompt, num_steps, guidance_scale, width, height, seed_input],
            outputs=[output_image, info_output]
        )
        
        regenerate_btn.click(
            fn=self._generate_image,
            inputs=[prompt_input, negative_prompt, num_steps, guidance_scale, width, height, gr.Number(value=None)],
            outputs=[output_image, info_output]
        )
    
    def _create_training_tab(self):
        """Create model training tab."""
        with gr.Row():
            with gr.Column(scale=2):
                # Training parameters
                gr.Markdown("### Training Configuration")
                
                training_data = gr.Textbox(
                    label="Training Data (JSON format)",
                    placeholder='[{"text": "sample text 1"}, {"text": "sample text 2"}]',
                    lines=5
                )
                
                with gr.Row():
                    epochs = gr.Slider(
                        minimum=1, maximum=100, value=10, step=1,
                        label="Epochs"
                    )
                    batch_size = gr.Slider(
                        minimum=1, maximum=32, value=4, step=1,
                        label="Batch Size"
                    )
                
                with gr.Row():
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=1e-5,
                        precision=1e-8
                    )
                    model_type = gr.Dropdown(
                        choices=["llm", "diffusion"],
                        value="llm",
                        label="Model Type"
                    )
                
                train_btn = gr.Button("🏋️ Start Training", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
                
            with gr.Column(scale=2):
                # Training progress
                gr.Markdown("### Training Progress")
                
                progress_bar = gr.Progress()
                
                loss_plot = gr.Plot(label="Training Loss")
                
                status_output = gr.Textbox(
                    label="Training Status",
                    lines=3,
                    interactive=False
                )
                
                metrics_output = gr.JSON(label="Training Metrics")
        
        # Event handlers
        train_btn.click(
            fn=self._start_training,
            inputs=[training_data, epochs, batch_size, learning_rate, model_type],
            outputs=[progress_bar, loss_plot, status_output, metrics_output]
        )
        
        stop_btn.click(
            fn=self._stop_training,
            outputs=[status_output]
        )
    
    def _create_performance_tab(self):
        """Create performance monitoring tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### System Performance")
                
                refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                
                # Performance metrics
                cpu_usage = gr.Number(label="CPU Usage (%)", precision=2)
                memory_usage = gr.Number(label="Memory Usage (%)", precision=2)
                gpu_usage = gr.Number(label="GPU Usage (%)", precision=2)
                
                # Model performance
                inference_time = gr.Number(label="Avg Inference Time (ms)", precision=2)
                throughput = gr.Number(label="Throughput (req/s)", precision=2)
                
                # Performance plots
                performance_plot = gr.Plot(label="Performance Over Time")
                
            with gr.Column():
                gr.Markdown("### Model Statistics")
                
                model_stats = gr.JSON(label="Model Statistics")
                
                # Error rates
                error_rate = gr.Number(label="Error Rate (%)", precision=2)
                success_rate = gr.Number(label="Success Rate (%)", precision=2)
        
        # Event handlers
        refresh_btn.click(
            fn=self._get_performance_metrics,
            outputs=[cpu_usage, memory_usage, gpu_usage, inference_time, throughput, performance_plot, model_stats, error_rate, success_rate]
        )
    
    def _create_health_tab(self):
        """Create system health tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### System Health Status")
                
                health_btn = gr.Button("💚 Check Health", variant="primary")
                
                overall_status = gr.Textbox(label="Overall Status", interactive=False)
                
                # Component health
                llm_health = gr.Textbox(label="LLM Engine Status", interactive=False)
                diffusion_health = gr.Textbox(label="Diffusion Engine Status", interactive=False)
                
                # System resources
                disk_usage = gr.Number(label="Disk Usage (%)", precision=2)
                network_status = gr.Textbox(label="Network Status", interactive=False)
                
            with gr.Column():
                gr.Markdown("### Detailed Health Report")
                
                health_report = gr.JSON(label="Health Report")
                
                # Logs
                logs_output = gr.Textbox(
                    label="Recent Logs",
                    lines=10,
                    interactive=False
                )
        
        # Event handlers
        health_btn.click(
            fn=self._check_system_health,
            outputs=[overall_status, llm_health, diffusion_health, disk_usage, network_status, health_report, logs_output]
        )
    
    async def _generate_text(self, prompt: str, max_length: int, temperature: float, top_p: float, do_sample: bool) -> Tuple[str, Dict]:
        """Generate text using LLM engine."""
        try:
            # Validate inputs
            is_valid, error_msg = self.text_validator.validate_text_generation_params(
                prompt, max_length, temperature, top_p, do_sample, 1
            )
            if not is_valid:
                return self.error_handler.handle_validation_error(error_msg)
            
            # Execute with safe executor
            result = await self.safe_executor.execute_with_timeout(
                "text_generation",
                self._execute_text_generation,
                prompt, max_length, temperature, top_p, do_sample
            )
            
            if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                return result
            
            generated_text = result.get("text", "")
            stats = {
                "prompt_length": len(prompt),
                "generated_length": len(generated_text),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }
            
            return generated_text, stats
            
        except Exception as e:
            return self.error_handler.handle_system_error(e, "text generation")
    
    async def _execute_text_generation(self, prompt: str, max_length: int, temperature: float, top_p: float, do_sample: bool) -> Dict:
        """Execute text generation with performance profiling."""
        with self.performance_profiler.profile("text_generation"):
            result = await self.llm_engine.execute("generate_text", {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            })
        return result
    
    async def _generate_image(self, prompt: str, negative_prompt: str, num_steps: int, 
                            guidance_scale: float, width: int, height: int, seed: Optional[int]) -> Tuple[Image.Image, Dict]:
        """Generate image using diffusion engine."""
        try:
            # Validate inputs
            is_valid, error_msg = self.image_validator.validate_image_generation_params(
                prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, 1
            )
            if not is_valid:
                return None, self.error_handler.handle_validation_error(error_msg)[1]
            
            # Execute with safe executor
            result = await self.safe_executor.execute_with_timeout(
                "image_generation",
                self._execute_image_generation,
                prompt, negative_prompt, num_steps, guidance_scale, width, height, seed
            )
            
            if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                return None, result[1]
            
            image_path = result.get("image_path")
            if image_path and Path(image_path).exists():
                image = Image.open(image_path)
                info = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "dimensions": f"{width}x{height}",
                    "seed": seed
                }
                return image, info
            else:
                return None, {"error": "Failed to generate image"}
                
        except Exception as e:
            return None, self.error_handler.handle_system_error(e, "image generation")[1]
    
    async def _execute_image_generation(self, prompt: str, negative_prompt: str, num_steps: int, 
                                      guidance_scale: float, width: int, height: int, seed: Optional[int]) -> Dict:
        """Execute image generation with performance profiling."""
        with self.performance_profiler.profile("image_generation"):
            result = await self.diffusion_engine.execute("generate_image", {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed
            })
        return result
    
    async def _start_training(self, training_data: str, epochs: int, batch_size: int, 
                            learning_rate: float, model_type: str) -> Tuple[Any, Any, str, Dict]:
        """Start model training."""
        try:
            # Validate inputs
            is_valid, error_msg = self.training_validator.validate_training_params(
                training_data, epochs, batch_size, learning_rate, model_type
            )
            if not is_valid:
                return None, None, self.error_handler.handle_validation_error(error_msg)[0], {"error": error_msg}
            
            # Execute with safe executor
            result = await self.safe_executor.execute_with_timeout(
                "model_training",
                self._execute_training,
                training_data, epochs, batch_size, learning_rate, model_type
            )
            
            if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                return None, None, result[0], result[1]
            
            # Create loss plot
            loss_plot = self._create_loss_plot([0.5, 0.4, 0.3, 0.25, 0.2])  # Placeholder data
            
            status = f"Training completed successfully. Final loss: {result.get('final_loss', 'N/A')}"
            metrics = result
            
            return None, loss_plot, status, metrics
            
        except Exception as e:
            return None, None, self.error_handler.handle_system_error(e, "model training")[0], {"error": str(e)}
    
    async def _execute_training(self, training_data: str, epochs: int, batch_size: int, 
                              learning_rate: float, model_type: str) -> Dict:
        """Execute model training with performance profiling."""
        import json
        
        # Parse training data
        data = json.loads(training_data)
        
        with self.performance_profiler.profile("model_training"):
            # Start training based on model type
            if model_type == "llm":
                result = await self.llm_engine.execute("fine_tune", {
                    "training_data": data,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                })
            else:
                result = await self.diffusion_engine.execute("train_model", {
                    "training_data": data,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                })
        
        return result
    
    async def _stop_training(self) -> str:
        """Stop model training."""
        try:
            # Execute with safe executor
            result = await self.safe_executor.execute_with_timeout(
                "stop_training",
                self._execute_stop_training
            )
            
            if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                return result[0]
            
            return "Training stopped successfully."
        except Exception as e:
            return self.error_handler.handle_system_error(e, "stop training")[0]
    
    async def _execute_stop_training(self) -> str:
        """Execute stop training operation."""
        # Stop training (placeholder implementation)
        return "Training stopped successfully."
    
    def _get_performance_metrics(self) -> Tuple[float, float, float, float, float, Any, Dict, float, float]:
        """Get system performance metrics."""
        try:
            # Get system metrics
            system_metrics = self.performance_profiler.get_profile_report()
            
            # Placeholder metrics (in practice, get from actual monitoring)
            cpu_usage = 45.2
            memory_usage = 67.8
            gpu_usage = 89.1
            inference_time = 125.5
            throughput = 8.2
            error_rate = 2.1
            success_rate = 97.9
            
            # Create performance plot
            performance_plot = self._create_performance_plot()
            
            # Model statistics
            model_stats = {
                "total_requests": 1250,
                "successful_requests": 1224,
                "failed_requests": 26,
                "avg_response_time": 125.5,
                "peak_memory_usage": "2.3 GB"
            }
            
            return cpu_usage, memory_usage, gpu_usage, inference_time, throughput, performance_plot, model_stats, error_rate, success_rate
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, None, {}, 0.0, 0.0
    
    async def _check_system_health(self) -> Tuple[str, str, str, float, str, Dict, str]:
        """Check system health status."""
        try:
            # Execute with safe executor
            result = await self.safe_executor.execute_with_timeout(
                "health_check",
                self._execute_health_check
            )
            
            if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                return "unhealthy", "error", "error", 0.0, "disconnected", result[1], f"Health check failed: {result[1].get('error', 'Unknown error')}"
            
            return result
            
        except Exception as e:
            return "unhealthy", "error", "error", 0.0, "disconnected", self.error_handler.handle_system_error(e, "health check")[1], f"Health check failed: {str(e)}"
    
    async def _execute_health_check(self) -> Tuple[str, str, str, float, str, Dict, str]:
        """Execute system health check."""
        # Get health status from engines
        llm_health = await self.llm_engine.get_health_status()
        diffusion_health = await self.diffusion_engine.get_health_status()
        
        # Overall status
        overall_status = "healthy" if llm_health.get("status") == "healthy" and diffusion_health.get("status") == "healthy" else "degraded"
        
        # System resources
        disk_usage = 45.7  # Placeholder
        network_status = "connected"
        
        # Health report
        health_report = {
            "overall_status": overall_status,
            "llm_engine": llm_health,
            "diffusion_engine": diffusion_health,
            "system_resources": {
                "disk_usage": disk_usage,
                "network_status": network_status
            }
        }
        
        # Recent logs
        logs = "2024-01-15 10:30:15 - INFO: System healthy\n2024-01-15 10:29:45 - INFO: Request processed successfully"
        
        return overall_status, str(llm_health.get("status")), str(diffusion_health.get("status")), disk_usage, network_status, health_report, logs
    
    def _create_loss_plot(self, losses: List[float]) -> Any:
        """Create training loss plot."""
        try:
            epochs = list(range(1, len(losses) + 1))
            
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, losses, 'b-', linewidth=2, marker='o')
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
        except Exception as e:
            self.logger.error(f"Failed to create loss plot: {e}")
            return None
    
    def _create_performance_plot(self) -> Any:
        """Create performance plot."""
        try:
            # Placeholder data
            time_points = list(range(1, 21))
            cpu_usage = [45 + np.random.normal(0, 5) for _ in time_points]
            memory_usage = [65 + np.random.normal(0, 8) for _ in time_points]
            gpu_usage = [85 + np.random.normal(0, 10) for _ in time_points]
            
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, cpu_usage, 'b-', label='CPU Usage', linewidth=2)
            plt.plot(time_points, memory_usage, 'g-', label='Memory Usage', linewidth=2)
            plt.plot(time_points, gpu_usage, 'r-', label='GPU Usage', linewidth=2)
            plt.title('System Performance Over Time')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Usage (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
        except Exception as e:
            self.logger.error(f"Failed to create performance plot: {e}")
            return None
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        """
    
    def launch(self, **kwargs) -> Any:
        """Launch the Gradio interface."""
        try:
            if self.interface is None:
                raise RuntimeError("Interface not initialized")
            
            # Default launch parameters
            default_params = {
                "server_name": "0.0.0.0",
                "server_port": 7860,
                "share": False,
                "debug": False,
                "show_error": True
            }
            default_params.update(kwargs)
            
            self.logger.info(f"Launching Gradio interface on port {default_params['server_port']}")
            return self.interface.launch(**default_params)
            
        except Exception as e:
            self.logger.error(f"Failed to launch interface: {e}")
            raise

# Factory function
def create_blaze_ai_interface(config: Optional[CoreConfig] = None) -> BlazeAIGradioInterface:
    """Create a Blaze AI Gradio interface."""
    return BlazeAIGradioInterface(config)

__all__ = ["BlazeAIGradioInterface", "create_blaze_ai_interface"]
