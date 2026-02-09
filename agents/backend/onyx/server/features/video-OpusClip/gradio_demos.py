"""
Interactive Gradio Demos for Video-OpusClip

Comprehensive demos showcasing model inference and visualization capabilities:
- Text-to-Video Generation
- Image-to-Video Generation  
- Viral Analysis & Prediction
- Performance Monitoring & Visualization
- Training Progress & Metrics
- Real-time System Analytics
"""

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import asyncio
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64

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
logger = EnhancedLogger("gradio_demos")

# Initialize models
models = NeuralNetworkModels(config)
diffusion_pipelines = DiffusionPipelines(config)
text_processor = TextProcessor(config)
video_utils = VideoProcessingUtils(config)

# Initialize processors
video_processor = OptimizedVideoProcessor(config, cache, performance_monitor)
api = OptimizedAPI(config, cache, performance_monitor)

# =============================================================================
# DEMO 1: TEXT-TO-VIDEO GENERATION
# =============================================================================

def create_text_to_video_demo():
    """Create interactive text-to-video generation demo."""
    
    def generate_video_from_text(
        prompt: str,
        duration: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        model_preset: str,
        quality_preset: str
    ) -> Tuple[Optional[np.ndarray], str, Dict, Dict]:
        """Generate video from text with detailed metrics."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("text_to_video", "gradio_demo", "generate_video")
                context_manager.start_timing()
            
            # Validate inputs
            if not prompt or not prompt.strip():
                return None, "❌ Please provide a prompt", {}, {}
            
            # Generate video
            start_time = time.time()
            
            result = diffusion_pipelines.generate_video_from_text(
                prompt=prompt.strip(),
                duration=duration,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed if seed > 0 else None,
                model_preset=model_preset,
                quality_preset=quality_preset
            )
            
            generation_time = time.time() - start_time
            
            # Get performance metrics
            metrics = performance_monitor.get_metrics()
            
            # Create detailed metrics
            detailed_metrics = {
                "generation_time": generation_time,
                "prompt_length": len(prompt),
                "model_preset": model_preset,
                "quality_preset": quality_preset,
                "parameters": {
                    "duration": duration,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed
                },
                "performance": metrics
            }
            
            # Create visualization
            viz_data = create_generation_visualization(detailed_metrics)
            
            return result["video"], "✅ Video generated successfully", detailed_metrics, viz_data
            
        except Exception as e:
            logger.error("Video generation failed", error=e)
            return None, f"❌ Error: {str(e)}", {}, {}
    
    def create_generation_visualization(metrics: Dict) -> Dict:
        """Create visualization data for generation metrics."""
        
        # Create parameter comparison chart
        param_data = {
            "Parameter": ["Duration", "Guidance Scale", "Inference Steps"],
            "Value": [
                metrics["parameters"]["duration"],
                metrics["parameters"]["guidance_scale"],
                metrics["parameters"]["num_inference_steps"]
            ],
            "Optimal Range": ["3-30s", "1-20", "10-100"]
        }
        
        # Create performance timeline
        timeline_data = {
            "Time": [0, metrics["generation_time"] * 0.25, metrics["generation_time"] * 0.5, 
                    metrics["generation_time"] * 0.75, metrics["generation_time"]],
            "Progress": [0, 25, 50, 75, 100],
            "Stage": ["Start", "Initialization", "Processing", "Finalization", "Complete"]
        }
        
        return {
            "parameters": param_data,
            "timeline": timeline_data,
            "generation_time": metrics["generation_time"]
        }
    
    # Create interface
    with gr.Blocks(title="Text-to-Video Generation Demo") as demo:
        gr.Markdown("# 🎬 Text-to-Video Generation Demo")
        gr.Markdown("Generate videos from text prompts with advanced AI models")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Input Parameters")
                
                prompt_input = gr.Textbox(
                    label="Video Prompt",
                    placeholder="A majestic dragon flying through a mystical forest at sunset...",
                    lines=3
                )
                
                with gr.Row():
                    duration = gr.Slider(
                        minimum=3, maximum=30, value=10, step=1,
                        label="Duration (seconds)"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.1,
                        label="Guidance Scale"
                    )
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=10, maximum=100, value=30, step=1,
                        label="Inference Steps"
                    )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)"
                    )
                
                with gr.Row():
                    model_preset = gr.Dropdown(
                        choices=["stable-diffusion", "deepfloyd", "kandinsky"],
                        value="stable-diffusion",
                        label="Model Preset"
                    )
                    
                    quality_preset = gr.Dropdown(
                        choices=["fast", "balanced", "quality"],
                        value="balanced",
                        label="Quality Preset"
                    )
                
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📤 Generated Video")
                
                video_output = gr.Video(
                    label="Generated Video"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        with gr.Row():
            # Metrics section
            gr.Markdown("### 📊 Generation Metrics")
            
            metrics_output = gr.JSON(
                label="Detailed Metrics"
            )
            
            # Visualization section
            gr.Markdown("### 📈 Performance Visualization")
            
            param_chart = gr.Plot(
                label="Parameter Comparison"
            )
            
            timeline_chart = gr.Plot(
                label="Generation Timeline"
            )
        
        # Event handlers
        generate_btn.click(
            fn=generate_video_from_text,
            inputs=[
                prompt_input, duration, guidance_scale, num_inference_steps,
                seed, model_preset, quality_preset
            ],
            outputs=[video_output, status_output, metrics_output, param_chart]
        )
    
    return demo

# =============================================================================
# DEMO 2: IMAGE-TO-VIDEO GENERATION
# =============================================================================

def create_image_to_video_demo():
    """Create interactive image-to-video generation demo."""
    
    def generate_video_from_image(
        image: Optional[np.ndarray],
        motion_strength: float,
        motion_direction: str,
        duration: int,
        interpolation_frames: int,
        style_transfer: bool,
        enhance_quality: bool
    ) -> Tuple[Optional[np.ndarray], str, Dict, Dict]:
        """Generate video from image with motion effects."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("image_to_video", "gradio_demo", "generate_video")
                context_manager.start_timing()
            
            # Validate inputs
            if image is None:
                return None, "❌ Please provide an image", {}, {}
            
            # Generate video
            start_time = time.time()
            
            result = diffusion_pipelines.generate_video_from_image(
                image=image,
                motion_strength=motion_strength,
                motion_direction=motion_direction,
                duration=duration,
                interpolation_frames=interpolation_frames,
                style_transfer=style_transfer,
                enhance_quality=enhance_quality
            )
            
            generation_time = time.time() - start_time
            
            # Get performance metrics
            metrics = performance_monitor.get_metrics()
            
            # Create detailed metrics
            detailed_metrics = {
                "generation_time": generation_time,
                "image_size": image.shape[:2] if image is not None else None,
                "motion_parameters": {
                    "motion_strength": motion_strength,
                    "motion_direction": motion_direction,
                    "duration": duration,
                    "interpolation_frames": interpolation_frames
                },
                "enhancement": {
                    "style_transfer": style_transfer,
                    "enhance_quality": enhance_quality
                },
                "performance": metrics
            }
            
            # Create visualization
            viz_data = create_motion_visualization(detailed_metrics)
            
            return result["video"], "✅ Video generated successfully", detailed_metrics, viz_data
            
        except Exception as e:
            logger.error("Image-to-video generation failed", error=e)
            return None, f"❌ Error: {str(e)}", {}, {}
    
    def create_motion_visualization(metrics: Dict) -> Dict:
        """Create visualization data for motion effects."""
        
        # Create motion strength visualization
        motion_data = {
            "Frame": list(range(metrics["motion_parameters"]["interpolation_frames"])),
            "Motion": [
                metrics["motion_parameters"]["motion_strength"] * (i / metrics["motion_parameters"]["interpolation_frames"])
                for i in range(metrics["motion_parameters"]["interpolation_frames"])
            ]
        }
        
        # Create direction mapping
        direction_data = {
            "Direction": ["Horizontal", "Vertical", "Diagonal", "Circular"],
            "Intensity": [0.8, 0.6, 0.9, 0.7]
        }
        
        return {
            "motion": motion_data,
            "direction": direction_data,
            "generation_time": metrics["generation_time"]
        }
    
    # Create interface
    with gr.Blocks(title="Image-to-Video Generation Demo") as demo:
        gr.Markdown("# 🖼️ Image-to-Video Generation Demo")
        gr.Markdown("Transform static images into dynamic videos with motion effects")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 🖼️ Input Image")
                
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy"
                )
                
                gr.Markdown("### ⚙️ Motion Parameters")
                
                with gr.Row():
                    motion_strength = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="Motion Strength"
                    )
                    
                    motion_direction = gr.Dropdown(
                        choices=["horizontal", "vertical", "diagonal", "circular"],
                        value="horizontal",
                        label="Motion Direction"
                    )
                
                with gr.Row():
                    duration = gr.Slider(
                        minimum=3, maximum=30, value=10, step=1,
                        label="Duration (seconds)"
                    )
                    
                    interpolation_frames = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5,
                        label="Interpolation Frames"
                    )
                
                with gr.Row():
                    style_transfer = gr.Checkbox(
                        value=False,
                        label="Style Transfer"
                    )
                    
                    enhance_quality = gr.Checkbox(
                        value=True,
                        label="Enhance Quality"
                    )
                
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📤 Generated Video")
                
                video_output = gr.Video(
                    label="Generated Video"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        with gr.Row():
            # Metrics section
            gr.Markdown("### 📊 Generation Metrics")
            
            metrics_output = gr.JSON(
                label="Detailed Metrics"
            )
            
            # Visualization section
            gr.Markdown("### 📈 Motion Visualization")
            
            motion_chart = gr.Plot(
                label="Motion Strength Over Time"
            )
            
            direction_chart = gr.Plot(
                label="Direction Intensity"
            )
        
        # Event handlers
        generate_btn.click(
            fn=generate_video_from_image,
            inputs=[
                image_input, motion_strength, motion_direction, duration,
                interpolation_frames, style_transfer, enhance_quality
            ],
            outputs=[video_output, status_output, metrics_output, motion_chart]
        )
    
    return demo

# =============================================================================
# DEMO 3: VIRAL ANALYSIS & PREDICTION
# =============================================================================

def create_viral_analysis_demo():
    """Create interactive viral analysis and prediction demo."""
    
    def analyze_viral_potential(
        content: str,
        content_type: str,
        platform: str,
        target_audience: str,
        content_category: str,
        include_hashtags: bool,
        analyze_sentiment: bool
    ) -> Tuple[float, float, str, Dict, Dict, Dict]:
        """Analyze viral potential with comprehensive metrics."""
        
        try:
            # Set operation context
            if context_manager:
                context_manager.set_operation_context("viral_analysis", "gradio_demo", "analyze_potential")
                context_manager.start_timing()
            
            # Validate inputs
            if not content or not content.strip():
                return 0.0, 0.0, "❌ Please provide content to analyze", {}, {}, {}
            
            # Analyze content
            start_time = time.time()
            
            analysis = api.analyze_viral_potential(
                content=content.strip(),
                content_type=content_type,
                platform=platform,
                target_audience=target_audience,
                content_category=content_category,
                include_hashtags=include_hashtags,
                analyze_sentiment=analyze_sentiment
            )
            
            analysis_time = time.time() - start_time
            
            # Extract results
            viral_score = analysis.get("viral_score", 0.0)
            engagement_prediction = analysis.get("engagement_prediction", 0.0)
            recommendations = analysis.get("recommendations", "No recommendations available")
            
            # Create detailed metrics
            detailed_metrics = {
                "analysis_time": analysis_time,
                "content_length": len(content),
                "analysis_parameters": {
                    "content_type": content_type,
                    "platform": platform,
                    "target_audience": target_audience,
                    "content_category": content_category,
                    "include_hashtags": include_hashtags,
                    "analyze_sentiment": analyze_sentiment
                },
                "results": analysis
            }
            
            # Create visualizations
            score_viz = create_score_visualization(viral_score, engagement_prediction)
            platform_viz = create_platform_comparison(platform, viral_score)
            
            return viral_score, engagement_prediction, recommendations, detailed_metrics, score_viz, platform_viz
            
        except Exception as e:
            logger.error("Viral analysis failed", error=e)
            return 0.0, 0.0, f"❌ Error: {str(e)}", {}, {}, {}
    
    def create_score_visualization(viral_score: float, engagement: float) -> Dict:
        """Create visualization for viral scores."""
        
        # Create radar chart data
        categories = ["Viral Potential", "Engagement", "Shareability", "Timing", "Relevance"]
        scores = [viral_score, engagement, viral_score * 0.9, viral_score * 0.8, viral_score * 0.95]
        
        # Create bar chart data
        bar_data = {
            "Metric": ["Viral Score", "Engagement Prediction"],
            "Score": [viral_score, engagement],
            "Color": ["#FF6B6B", "#4ECDC4"]
        }
        
        return {
            "radar": {"categories": categories, "scores": scores},
            "bar": bar_data
        }
    
    def create_platform_comparison(platform: str, score: float) -> Dict:
        """Create platform comparison visualization."""
        
        platforms = ["tiktok", "youtube", "instagram", "twitter"]
        base_scores = [0.8, 0.6, 0.7, 0.5]  # Platform-specific base scores
        
        # Adjust score based on platform
        platform_index = platforms.index(platform.lower()) if platform.lower() in platforms else 0
        adjusted_score = score * base_scores[platform_index]
        
        comparison_data = {
            "Platform": platforms,
            "Base Score": base_scores,
            "Adjusted Score": [adjusted_score if i == platform_index else base_scores[i] * 0.7 for i in range(len(platforms))]
        }
        
        return comparison_data
    
    # Create interface
    with gr.Blocks(title="Viral Analysis Demo") as demo:
        gr.Markdown("# 📈 Viral Analysis & Prediction Demo")
        gr.Markdown("Analyze content viral potential across different platforms")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Content Analysis")
                
                content_input = gr.Textbox(
                    label="Content Description",
                    placeholder="Describe your content in detail...",
                    lines=4
                )
                
                with gr.Row():
                    content_type = gr.Dropdown(
                        choices=["video", "image", "text", "story"],
                        value="video",
                        label="Content Type"
                    )
                    
                    platform = gr.Dropdown(
                        choices=["tiktok", "youtube", "instagram", "twitter", "facebook"],
                        value="tiktok",
                        label="Target Platform"
                    )
                
                with gr.Row():
                    target_audience = gr.Dropdown(
                        choices=["gen-z", "millennials", "gen-x", "boomers", "all"],
                        value="gen-z",
                        label="Target Audience"
                    )
                    
                    content_category = gr.Dropdown(
                        choices=["entertainment", "education", "news", "lifestyle", "gaming", "music"],
                        value="entertainment",
                        label="Content Category"
                    )
                
                with gr.Row():
                    include_hashtags = gr.Checkbox(
                        value=True,
                        label="Include Hashtag Analysis"
                    )
                    
                    analyze_sentiment = gr.Checkbox(
                        value=True,
                        label="Analyze Sentiment"
                    )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Viral Potential",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Results section
                gr.Markdown("### 📊 Analysis Results")
                
                with gr.Row():
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
        
        with gr.Row():
            # Metrics section
            gr.Markdown("### 📈 Detailed Metrics")
            
            metrics_output = gr.JSON(
                label="Analysis Metrics"
            )
            
            # Visualization section
            gr.Markdown("### 📊 Score Visualization")
            
            score_chart = gr.Plot(
                label="Viral Score Breakdown"
            )
            
            platform_chart = gr.Plot(
                label="Platform Comparison"
            )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_viral_potential,
            inputs=[
                content_input, content_type, platform, target_audience,
                content_category, include_hashtags, analyze_sentiment
            ],
            outputs=[
                viral_score, engagement_prediction, recommendations,
                metrics_output, score_chart, platform_chart
            ]
        )
    
    return demo

# =============================================================================
# DEMO 4: PERFORMANCE MONITORING & VISUALIZATION
# =============================================================================

def create_performance_monitoring_demo():
    """Create interactive performance monitoring and visualization demo."""
    
    def get_performance_metrics(refresh_interval: int = 5) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Get real-time performance metrics with visualization."""
        
        try:
            # Get current metrics
            metrics = performance_monitor.get_metrics()
            
            # Create time series data
            current_time = time.time()
            
            # Create CPU usage chart
            cpu_data = {
                "Time": [current_time - refresh_interval, current_time],
                "Usage": [metrics.get("cpu_usage", 0) * 0.8, metrics.get("cpu_usage", 0)],
                "Load": [metrics.get("cpu_load", 0) * 0.8, metrics.get("cpu_load", 0)]
            }
            
            # Create memory usage chart
            memory_data = {
                "Time": [current_time - refresh_interval, current_time],
                "Usage": [metrics.get("memory_usage", 0) * 0.8, metrics.get("memory_usage", 0)],
                "Available": [100 - metrics.get("memory_usage", 0) * 0.8, 100 - metrics.get("memory_usage", 0)]
            }
            
            # Create GPU usage chart
            gpu_data = {
                "Time": [current_time - refresh_interval, current_time],
                "Usage": [metrics.get("gpu_usage", 0) * 0.8, metrics.get("gpu_usage", 0)],
                "Memory": [metrics.get("gpu_memory_usage", 0) * 0.8, metrics.get("gpu_memory_usage", 0)]
            }
            
            # Create system overview
            system_overview = {
                "System": {
                    "CPU Cores": metrics.get("cpu_cores", 0),
                    "Total Memory": f"{metrics.get('total_memory_gb', 0):.1f} GB",
                    "GPU Model": metrics.get("gpu_model", "N/A"),
                    "GPU Memory": f"{metrics.get('gpu_memory_gb', 0):.1f} GB"
                },
                "Performance": {
                    "CPU Usage": f"{metrics.get('cpu_usage', 0):.1f}%",
                    "Memory Usage": f"{metrics.get('memory_usage', 0):.1f}%",
                    "GPU Usage": f"{metrics.get('gpu_usage', 0):.1f}%",
                    "Disk Usage": f"{metrics.get('disk_usage', 0):.1f}%"
                },
                "Network": {
                    "Upload Speed": f"{metrics.get('upload_speed_mbps', 0):.1f} Mbps",
                    "Download Speed": f"{metrics.get('download_speed_mbps', 0):.1f} Mbps",
                    "Latency": f"{metrics.get('latency_ms', 0):.1f} ms"
                }
            }
            
            return metrics, cpu_data, memory_data, gpu_data, system_overview
            
        except Exception as e:
            logger.error("Failed to get performance metrics", error=e)
            return {}, {}, {}, {}, {}
    
    def start_performance_monitoring(monitoring_duration: int = 60) -> str:
        """Start continuous performance monitoring."""
        
        try:
            # Start monitoring in background
            def monitor_loop():
                start_time = time.time()
                while time.time() - start_time < monitoring_duration:
                    get_performance_metrics()
                    time.sleep(5)
            
            # Start monitoring thread
            import threading
            monitor_thread = threading.Thread(target=monitor_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            return f"✅ Performance monitoring started for {monitoring_duration} seconds"
            
        except Exception as e:
            logger.error("Failed to start performance monitoring", error=e)
            return f"❌ Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Performance Monitoring Demo") as demo:
        gr.Markdown("# ⚡ Performance Monitoring & Visualization Demo")
        gr.Markdown("Real-time system performance monitoring and analytics")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Control section
                gr.Markdown("### 🎛️ Monitoring Controls")
                
                refresh_interval = gr.Slider(
                    minimum=1, maximum=30, value=5, step=1,
                    label="Refresh Interval (seconds)"
                )
                
                monitoring_duration = gr.Slider(
                    minimum=30, maximum=300, value=60, step=30,
                    label="Monitoring Duration (seconds)"
                )
                
                with gr.Row():
                    refresh_btn = gr.Button(
                        "🔄 Refresh Metrics",
                        variant="secondary"
                    )
                    
                    start_monitoring_btn = gr.Button(
                        "▶️ Start Monitoring",
                        variant="primary"
                    )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                # System overview
                gr.Markdown("### 💻 System Overview")
                
                system_overview = gr.JSON(
                    label="System Information"
                )
        
        with gr.Row():
            # CPU monitoring
            gr.Markdown("### 🖥️ CPU Monitoring")
            
            cpu_chart = gr.Plot(
                label="CPU Usage Over Time"
            )
            
            cpu_metrics = gr.JSON(
                label="CPU Metrics"
            )
        
        with gr.Row():
            # Memory monitoring
            gr.Markdown("### 💾 Memory Monitoring")
            
            memory_chart = gr.Plot(
                label="Memory Usage Over Time"
            )
            
            memory_metrics = gr.JSON(
                label="Memory Metrics"
            )
        
        with gr.Row():
            # GPU monitoring
            gr.Markdown("### 🎮 GPU Monitoring")
            
            gpu_chart = gr.Plot(
                label="GPU Usage Over Time"
            )
            
            gpu_metrics = gr.JSON(
                label="GPU Metrics"
            )
        
        # Event handlers
        refresh_btn.click(
            fn=get_performance_metrics,
            inputs=[refresh_interval],
            outputs=[cpu_metrics, cpu_chart, memory_chart, gpu_chart, system_overview]
        )
        
        start_monitoring_btn.click(
            fn=start_performance_monitoring,
            inputs=[monitoring_duration],
            outputs=[status_output]
        )
    
    return demo

# =============================================================================
# DEMO 5: TRAINING PROGRESS & METRICS
# =============================================================================

def create_training_demo():
    """Create interactive training progress and metrics demo."""
    
    def simulate_training(
        model_type: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        dataset_size: int
    ) -> Tuple[float, Dict, Dict, Dict, Dict]:
        """Simulate training process with real-time updates."""
        
        try:
            # Initialize training metrics
            training_loss = []
            validation_loss = []
            accuracy = []
            learning_rates = []
            
            # Simulate training progress
            for epoch in range(epochs):
                # Simulate training loss (decreasing)
                train_loss = 2.0 * np.exp(-epoch / 10) + 0.1 * np.random.random()
                training_loss.append(train_loss)
                
                # Simulate validation loss
                val_loss = train_loss + 0.2 * np.random.random()
                validation_loss.append(val_loss)
                
                # Simulate accuracy (increasing)
                acc = 0.3 + 0.6 * (1 - np.exp(-epoch / 8)) + 0.05 * np.random.random()
                accuracy.append(acc)
                
                # Simulate learning rate schedule
                lr = learning_rate * (0.9 ** (epoch // 5))
                learning_rates.append(lr)
                
                # Small delay to simulate processing
                time.sleep(0.1)
            
            # Create training metrics
            training_metrics = {
                "final_loss": training_loss[-1],
                "final_accuracy": accuracy[-1],
                "best_epoch": np.argmin(validation_loss),
                "training_time": epochs * 0.1,
                "model_type": model_type,
                "parameters": {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "dataset_size": dataset_size
                }
            }
            
            # Create visualization data
            loss_data = {
                "Epoch": list(range(epochs)),
                "Training Loss": training_loss,
                "Validation Loss": validation_loss
            }
            
            accuracy_data = {
                "Epoch": list(range(epochs)),
                "Accuracy": accuracy
            }
            
            lr_data = {
                "Epoch": list(range(epochs)),
                "Learning Rate": learning_rates
            }
            
            return (
                training_metrics["final_accuracy"],
                training_metrics,
                loss_data,
                accuracy_data,
                lr_data
            )
            
        except Exception as e:
            logger.error("Training simulation failed", error=e)
            return 0.0, {}, {}, {}, {}
    
    # Create interface
    with gr.Blocks(title="Training Progress Demo") as demo:
        gr.Markdown("# 🏋️ Training Progress & Metrics Demo")
        gr.Markdown("Interactive training simulation with real-time metrics")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Training parameters
                gr.Markdown("### ⚙️ Training Parameters")
                
                model_type = gr.Dropdown(
                    choices=["caption_generator", "viral_predictor", "quality_assessor", "style_transfer"],
                    value="caption_generator",
                    label="Model Type"
                )
                
                with gr.Row():
                    epochs = gr.Slider(
                        minimum=5, maximum=50, value=20, step=1,
                        label="Epochs"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=1e-6, maximum=1e-2, value=1e-4, step=1e-5,
                        label="Learning Rate"
                    )
                
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=8, maximum=128, value=32, step=8,
                        label="Batch Size"
                    )
                    
                    dataset_size = gr.Slider(
                        minimum=1000, maximum=100000, value=10000, step=1000,
                        label="Dataset Size"
                    )
                
                train_btn = gr.Button(
                    "🚀 Start Training",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Training results
                gr.Markdown("### 📊 Training Results")
                
                final_accuracy = gr.Number(
                    label="Final Accuracy",
                    interactive=False
                )
                
                training_metrics = gr.JSON(
                    label="Training Metrics"
                )
        
        with gr.Row():
            # Loss visualization
            gr.Markdown("### 📉 Loss Curves")
            
            loss_chart = gr.Plot(
                label="Training & Validation Loss"
            )
        
        with gr.Row():
            # Accuracy visualization
            gr.Markdown("### 📈 Accuracy Progress")
            
            accuracy_chart = gr.Plot(
                label="Training Accuracy"
            )
            
            # Learning rate visualization
            gr.Markdown("### 📊 Learning Rate Schedule")
            
            lr_chart = gr.Plot(
                label="Learning Rate Over Time"
            )
        
        # Event handlers
        train_btn.click(
            fn=simulate_training,
            inputs=[model_type, epochs, learning_rate, batch_size, dataset_size],
            outputs=[final_accuracy, training_metrics, loss_chart, accuracy_chart, lr_chart]
        )
    
    return demo

# =============================================================================
# MAIN DEMO LAUNCHER
# =============================================================================

def create_main_demo():
    """Create main demo launcher with all demos."""
    
    with gr.Blocks(
        title="Video-OpusClip Interactive Demos",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
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
        .demo-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎬 Video-OpusClip Interactive Demos</h1>
            <p>Comprehensive AI-powered video processing and analysis demonstrations</p>
        </div>
        """)
        
        # Demo selection
        with gr.Tabs():
            
            # Text-to-Video Demo
            with gr.TabItem("🎬 Text-to-Video"):
                text_to_video_demo = create_text_to_video_demo()
            
            # Image-to-Video Demo
            with gr.TabItem("🖼️ Image-to-Video"):
                image_to_video_demo = create_image_to_video_demo()
            
            # Viral Analysis Demo
            with gr.TabItem("📈 Viral Analysis"):
                viral_analysis_demo = create_viral_analysis_demo()
            
            # Performance Monitoring Demo
            with gr.TabItem("⚡ Performance"):
                performance_demo = create_performance_monitoring_demo()
            
            # Training Demo
            with gr.TabItem("🏋️ Training"):
                training_demo = create_training_demo()
    
    return demo

def launch_demos(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch all interactive demos."""
    
    # Create main demo
    demo = create_main_demo()
    
    # Launch with optimized settings
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    # Launch the demos
    launch_demos(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    ) 