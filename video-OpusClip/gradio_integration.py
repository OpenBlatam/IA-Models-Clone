"""
Enhanced Gradio Integration for Video-OpusClip

Comprehensive Gradio interface that integrates all optimized components:
- Configuration management
- Caching system
- Video processing pipeline
- Performance monitoring
- Evaluation metrics
- Error handling
- Real-time analytics
"""

import gradio as gr
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import json

# Import optimized components
from optimized_config import OptimizedConfig, get_config
from optimized_cache import OptimizedCache
from optimized_video_processor import OptimizedVideoProcessor
from optimized_api import OptimizedAPI
from performance_monitor import PerformanceMonitor
from optimized_evaluation import OptimizedEvaluator
from optimized_libraries import (
    NeuralNetworkModels, DiffusionPipelines, 
    TextProcessor, VideoProcessingUtils
)
from optimized_gradient_handling import GradientHandler
from optimized_data_loader import OptimizedDataLoader
from optimized_training import OptimizedTrainer

# Import error handling
from error_factories import (
    error_factory, context_manager,
    create_validation_error, create_processing_error, create_inference_error,
    create_resource_error, create_error_context
)
from logging_config import EnhancedLogger

# =============================================================================
# GLOBAL COMPONENTS
# =============================================================================

# Initialize components
config = get_config()
cache = OptimizedCache(config)
performance_monitor = PerformanceMonitor(config)
evaluator = OptimizedEvaluator(config)
gradient_handler = GradientHandler(config)
logger = EnhancedLogger("gradio_integration")

# Initialize models
models = NeuralNetworkModels(config)
diffusion_pipelines = DiffusionPipelines(config)
text_processor = TextProcessor(config)
video_utils = VideoProcessingUtils(config)

# Initialize processors
video_processor = OptimizedVideoProcessor(config, cache, performance_monitor)
api = OptimizedAPI(config, cache, performance_monitor)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=config.env.MAX_WORKERS)

# =============================================================================
# GRADIO INTERFACE COMPONENTS
# =============================================================================

class GradioInterface:
    """Enhanced Gradio interface with all optimized features."""
    
    def __init__(self):
        self.config = config
        self.cache = cache
        self.performance_monitor = performance_monitor
        self.evaluator = evaluator
        self.video_processor = video_processor
        self.api = api
        
        # State management
        self.current_session = None
        self.processing_queue = []
        self.results_cache = {}
        
        # Initialize interface
        self._create_interface()
    
    def _create_interface(self):
        """Create the main Gradio interface."""
        
        # Create tabs for different functionalities
        with gr.Blocks(
            title="Video-OpusClip AI Studio",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
            }
            """
        ) as self.interface:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🎬 Video-OpusClip AI Studio</h1>
                <p>Advanced AI-powered video processing and viral content generation</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Video Processing Tab
                with gr.TabItem("🎥 Video Processing"):
                    self._create_video_processing_tab()
                
                # AI Generation Tab
                with gr.TabItem("🤖 AI Generation"):
                    self._create_ai_generation_tab()
                
                # Viral Analysis Tab
                with gr.TabItem("📈 Viral Analysis"):
                    self._create_viral_analysis_tab()
                
                # Training Tab
                with gr.TabItem("🏋️ Training"):
                    self._create_training_tab()
                
                # Performance Tab
                with gr.TabItem("⚡ Performance"):
                    self._create_performance_tab()
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings"):
                    self._create_settings_tab()
    
    def _create_video_processing_tab(self):
        """Create video processing interface."""
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📁 Input")
                
                video_input = gr.Video(
                    label="Upload Video",
                    source="upload",
                    type="numpy"
                )
                
                url_input = gr.Textbox(
                    label="Video URL",
                    placeholder="https://example.com/video.mp4"
                )
                
                processing_options = gr.Group(label="Processing Options")
                
                with processing_options:
                    target_duration = gr.Slider(
                        minimum=5, maximum=300, value=30, step=5,
                        label="Target Duration (seconds)"
                    )
                    
                    quality_preset = gr.Radio(
                        choices=["fast", "balanced", "quality"],
                        value="balanced",
                        label="Quality Preset"
                    )
                    
                    enable_audio = gr.Checkbox(
                        value=True,
                        label="Process Audio"
                    )
                    
                    enable_subtitles = gr.Checkbox(
                        value=True,
                        label="Generate Subtitles"
                    )
                
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📤 Output")
                
                processed_video = gr.Video(
                    label="Processed Video"
                )
                
                processing_status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                processing_metrics = gr.JSON(
                    label="Processing Metrics"
                )
        
        # Event handlers
        process_btn.click(
            fn=self._process_video,
            inputs=[
                video_input, url_input, target_duration,
                quality_preset, enable_audio, enable_subtitles
            ],
            outputs=[processed_video, processing_status, processing_metrics]
        )
    
    def _create_ai_generation_tab(self):
        """Create AI generation interface."""
        
        with gr.Row():
            with gr.Column(scale=1):
                # Text-to-Video
                gr.Markdown("### 🎬 Text-to-Video")
                
                text_prompt = gr.Textbox(
                    label="Video Prompt",
                    placeholder="A cat playing with a ball in a sunny garden...",
                    lines=3
                )
                
                video_duration = gr.Slider(
                    minimum=3, maximum=30, value=10, step=1,
                    label="Duration (seconds)"
                )
                
                generation_params = gr.Group(label="Generation Parameters")
                
                with generation_params:
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.1,
                        label="Guidance Scale"
                    )
                    
                    num_inference_steps = gr.Slider(
                        minimum=10, maximum=100, value=30, step=1,
                        label="Inference Steps"
                    )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)"
                    )
                
                generate_video_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                # Image-to-Video
                gr.Markdown("### 🖼️ Image-to-Video")
                
                image_input = gr.Image(
                    label="Input Image",
                    type="pil"
                )
                
                motion_strength = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                    label="Motion Strength"
                )
                
                generate_from_image_btn = gr.Button(
                    "🎬 Generate from Image",
                    variant="primary"
                )
        
        with gr.Row():
            # Output section
            gr.Markdown("### 📤 Generated Content")
            
            generated_video = gr.Video(
                label="Generated Video"
            )
            
            generation_status = gr.Textbox(
                label="Generation Status",
                interactive=False
            )
            
            generation_metrics = gr.JSON(
                label="Generation Metrics"
            )
        
        # Event handlers
        generate_video_btn.click(
            fn=self._generate_video_from_text,
            inputs=[text_prompt, video_duration, guidance_scale, num_inference_steps, seed],
            outputs=[generated_video, generation_status, generation_metrics]
        )
        
        generate_from_image_btn.click(
            fn=self._generate_video_from_image,
            inputs=[image_input, motion_strength],
            outputs=[generated_video, generation_status, generation_metrics]
        )
    
    def _create_viral_analysis_tab(self):
        """Create viral analysis interface."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Content Analysis")
                
                content_input = gr.Textbox(
                    label="Content Text/Description",
                    placeholder="Describe your content...",
                    lines=4
                )
                
                content_type = gr.Radio(
                    choices=["video", "image", "text"],
                    value="video",
                    label="Content Type"
                )
                
                target_platform = gr.Dropdown(
                    choices=["tiktok", "youtube", "instagram", "twitter"],
                    value="tiktok",
                    label="Target Platform"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Viral Potential",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Analysis Results")
                
                viral_score = gr.Number(
                    label="Viral Score",
                    interactive=False
                )
                
                engagement_prediction = gr.Number(
                    label="Predicted Engagement",
                    interactive=False
                )
                
                recommendations = gr.Textbox(
                    label="Optimization Recommendations",
                    lines=5,
                    interactive=False
                )
                
                analysis_metrics = gr.JSON(
                    label="Detailed Metrics"
                )
        
        # Event handlers
        analyze_btn.click(
            fn=self._analyze_viral_potential,
            inputs=[content_input, content_type, target_platform],
            outputs=[viral_score, engagement_prediction, recommendations, analysis_metrics]
        )
    
    def _create_training_tab(self):
        """Create training interface."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🏋️ Model Training")
                
                training_data = gr.File(
                    label="Training Data",
                    file_types=[".json", ".csv", ".txt"]
                )
                
                model_type = gr.Dropdown(
                    choices=["caption_generator", "viral_predictor", "quality_assessor"],
                    value="caption_generator",
                    label="Model Type"
                )
                
                training_params = gr.Group(label="Training Parameters")
                
                with training_params:
                    epochs = gr.Slider(
                        minimum=1, maximum=100, value=10, step=1,
                        label="Epochs"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=1e-6, maximum=1e-2, value=1e-4, step=1e-5,
                        label="Learning Rate"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1, maximum=64, value=16, step=1,
                        label="Batch Size"
                    )
                
                train_btn = gr.Button(
                    "🚀 Start Training",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Training Progress")
                
                training_progress = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Progress (%)",
                    interactive=False
                )
                
                training_loss = gr.LinePlot(
                    label="Training Loss",
                    x="epoch",
                    y="loss"
                )
                
                validation_metrics = gr.JSON(
                    label="Validation Metrics"
                )
        
        # Event handlers
        train_btn.click(
            fn=self._start_training,
            inputs=[training_data, model_type, epochs, learning_rate, batch_size],
            outputs=[training_progress, training_loss, validation_metrics]
        )
    
    def _create_performance_tab(self):
        """Create performance monitoring interface."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ System Performance")
                
                performance_metrics = gr.JSON(
                    label="Real-time Metrics"
                )
                
                refresh_btn = gr.Button(
                    "🔄 Refresh Metrics",
                    variant="secondary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Performance Charts")
                
                cpu_usage = gr.LinePlot(
                    label="CPU Usage",
                    x="time",
                    y="usage"
                )
                
                memory_usage = gr.LinePlot(
                    label="Memory Usage",
                    x="time",
                    y="usage"
                )
                
                gpu_usage = gr.LinePlot(
                    label="GPU Usage",
                    x="time",
                    y="usage"
                )
        
        # Auto-refresh metrics
        refresh_btn.click(
            fn=self._get_performance_metrics,
            outputs=[performance_metrics, cpu_usage, memory_usage, gpu_usage]
        )
    
    def _create_settings_tab(self):
        """Create settings interface."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                # Performance settings
                max_workers = gr.Slider(
                    minimum=1, maximum=32, value=config.env.MAX_WORKERS,
                    step=1,
                    label="Max Workers"
                )
                
                batch_size = gr.Slider(
                    minimum=1, maximum=100, value=config.env.BATCH_SIZE,
                    step=1,
                    label="Batch Size"
                )
                
                enable_gpu = gr.Checkbox(
                    value=config.env.USE_GPU,
                    label="Enable GPU"
                )
                
                enable_caching = gr.Checkbox(
                    value=config.env.ENABLE_CACHING,
                    label="Enable Caching"
                )
                
                # Save settings
                save_btn = gr.Button(
                    "💾 Save Settings",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Advanced Settings")
                
                log_level = gr.Dropdown(
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    value=config.env.LOG_LEVEL,
                    label="Log Level"
                )
                
                cache_size = gr.Slider(
                    minimum=100, maximum=10000, value=config.env.CACHE_SIZE,
                    step=100,
                    label="Cache Size"
                )
                
                timeout = gr.Slider(
                    minimum=10, maximum=300, value=config.env.TIMEOUT,
                    step=10,
                    label="Timeout (seconds)"
                )
                
                # Reset to defaults
                reset_btn = gr.Button(
                    "🔄 Reset to Defaults",
                    variant="secondary"
                )
        
        # Event handlers
        save_btn.click(
            fn=self._save_settings,
            inputs=[max_workers, batch_size, enable_gpu, enable_caching, log_level, cache_size, timeout],
            outputs=[]
        )
        
        reset_btn.click(
            fn=self._reset_settings,
            outputs=[max_workers, batch_size, enable_gpu, enable_caching, log_level, cache_size, timeout]
        )

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

    def _process_video(
        self,
        video_input: Optional[np.ndarray],
        url_input: str,
        target_duration: float,
        quality_preset: str,
        enable_audio: bool,
        enable_subtitles: bool
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Process video with optimized pipeline."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("video_processing", "gradio", "process_video")
                context_manager.start_timing()
            
            # Validate inputs
            if video_input is None and not url_input:
                return None, "❌ No video provided", {}
            
            # Process video
            if video_input is not None:
                # Process uploaded video
                result = self.video_processor.process_video(
                    video_data=video_input,
                    target_duration=target_duration,
                    quality_preset=quality_preset,
                    enable_audio=enable_audio,
                    enable_subtitles=enable_subtitles
                )
            else:
                # Process video from URL
                result = self.video_processor.process_video_from_url(
                    url=url_input,
                    target_duration=target_duration,
                    quality_preset=quality_preset,
                    enable_audio=enable_audio,
                    enable_subtitles=enable_subtitles
                )
            
            # Get metrics
            metrics = self.performance_monitor.get_metrics()
            
            return result["processed_video"], "✅ Processing completed", metrics
            
        except Exception as e:
            logger.error("Video processing failed", error=e)
            return None, f"❌ Error: {str(e)}", {}

    def _generate_video_from_text(
        self,
        prompt: str,
        duration: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Generate video from text prompt."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("video_generation", "gradio", "generate_from_text")
                context_manager.start_timing()
            
            # Validate inputs
            if not prompt or not prompt.strip():
                return None, "❌ Please provide a prompt", {}
            
            # Generate video
            result = diffusion_pipelines.generate_video_from_text(
                prompt=prompt,
                duration=duration,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed if seed > 0 else None
            )
            
            # Get metrics
            metrics = self.performance_monitor.get_metrics()
            
            return result["video"], "✅ Video generated successfully", metrics
            
        except Exception as e:
            logger.error("Video generation failed", error=e)
            return None, f"❌ Error: {str(e)}", {}

    def _generate_video_from_image(
        self,
        image: Optional[np.ndarray],
        motion_strength: float
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Generate video from image."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("video_generation", "gradio", "generate_from_image")
                context_manager.start_timing()
            
            # Validate inputs
            if image is None:
                return None, "❌ Please provide an image", {}
            
            # Generate video
            result = diffusion_pipelines.generate_video_from_image(
                image=image,
                motion_strength=motion_strength
            )
            
            # Get metrics
            metrics = self.performance_monitor.get_metrics()
            
            return result["video"], "✅ Video generated successfully", metrics
            
        except Exception as e:
            logger.error("Video generation failed", error=e)
            return None, f"❌ Error: {str(e)}", {}

    def _analyze_viral_potential(
        self,
        content: str,
        content_type: str,
        platform: str
    ) -> Tuple[float, float, str, Dict]:
        """Analyze viral potential of content."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("viral_analysis", "gradio", "analyze_potential")
                context_manager.start_timing()
            
            # Validate inputs
            if not content or not content.strip():
                return 0.0, 0.0, "❌ Please provide content to analyze", {}
            
            # Analyze content
            analysis = self.api.analyze_viral_potential(
                content=content,
                content_type=content_type,
                platform=platform
            )
            
            # Extract results
            viral_score = analysis.get("viral_score", 0.0)
            engagement_prediction = analysis.get("engagement_prediction", 0.0)
            recommendations = analysis.get("recommendations", "No recommendations available")
            
            return viral_score, engagement_prediction, recommendations, analysis
            
        except Exception as e:
            logger.error("Viral analysis failed", error=e)
            return 0.0, 0.0, f"❌ Error: {str(e)}", {}

    def _start_training(
        self,
        training_data: Optional[str],
        model_type: str,
        epochs: int,
        learning_rate: float,
        batch_size: int
    ) -> Tuple[float, Dict, Dict]:
        """Start model training."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("model_training", "gradio", "start_training")
                context_manager.start_timing()
            
            # Validate inputs
            if not training_data:
                return 0.0, {}, {"error": "No training data provided"}
            
            # Initialize trainer
            trainer = OptimizedTrainer(
                config=self.config,
                model_type=model_type,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            # Start training (this would be async in production)
            # For demo purposes, we'll simulate training
            progress = 0.0
            loss_data = {"epoch": [], "loss": []}
            validation_metrics = {}
            
            return progress, loss_data, validation_metrics
            
        except Exception as e:
            logger.error("Training failed", error=e)
            return 0.0, {}, {"error": str(e)}

    def _get_performance_metrics(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """Get real-time performance metrics."""
        
        try:
            # Get current metrics
            metrics = self.performance_monitor.get_metrics()
            
            # Format for charts
            cpu_data = {
                "time": [time.time()],
                "usage": [metrics.get("cpu_usage", 0.0)]
            }
            
            memory_data = {
                "time": [time.time()],
                "usage": [metrics.get("memory_usage", 0.0)]
            }
            
            gpu_data = {
                "time": [time.time()],
                "usage": [metrics.get("gpu_usage", 0.0)]
            }
            
            return metrics, cpu_data, memory_data, gpu_data
            
        except Exception as e:
            logger.error("Failed to get performance metrics", error=e)
            return {}, {}, {}, {}

    def _save_settings(
        self,
        max_workers: int,
        batch_size: int,
        enable_gpu: bool,
        enable_caching: bool,
        log_level: str,
        cache_size: int,
        timeout: float
    ):
        """Save configuration settings."""
        
        try:
            # Update configuration
            self.config.env.MAX_WORKERS = max_workers
            self.config.env.BATCH_SIZE = batch_size
            self.config.env.USE_GPU = enable_gpu
            self.config.env.ENABLE_CACHING = enable_caching
            self.config.env.LOG_LEVEL = log_level
            self.config.env.CACHE_SIZE = cache_size
            self.config.env.TIMEOUT = timeout
            
            # Save to file
            self.config.save_to_file("gradio_config.yaml")
            
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error("Failed to save settings", error=e)

    def _reset_settings(self) -> Tuple[int, int, bool, bool, str, int, float]:
        """Reset settings to defaults."""
        
        try:
            # Reset to default configuration
            default_config = OptimizedConfig()
            
            return (
                default_config.env.MAX_WORKERS,
                default_config.env.BATCH_SIZE,
                default_config.env.USE_GPU,
                default_config.env.ENABLE_CACHING,
                default_config.env.LOG_LEVEL,
                default_config.env.CACHE_SIZE,
                default_config.env.TIMEOUT
            )
            
        except Exception as e:
            logger.error("Failed to reset settings", error=e)
            return (4, 10, True, True, "INFO", 1000, 30.0)

# =============================================================================
# MAIN INTERFACE
# =============================================================================

def create_gradio_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""
    interface = GradioInterface()
    return interface.interface

def launch_gradio(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the Gradio interface."""
    
    # Create interface
    interface = create_gradio_interface()
    
    # Launch with optimized settings
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    # Launch the interface
    launch_gradio(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    ) 