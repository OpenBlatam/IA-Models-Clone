"""
User-Friendly Interfaces for Video-OpusClip

Modern, intuitive interfaces that showcase model capabilities:
- Clean, responsive design
- Intuitive navigation and workflows
- Enhanced user experience
- Accessibility features
- Mobile-friendly layouts
- Interactive tutorials and guides
"""

import gradio as gr
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import json
from pathlib import Path
import asyncio
from datetime import datetime

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
from visualization_utils import DemoVisualizer

# Import error handling components
from gradio_error_handling import (
    GradioErrorHandler, GradioInputValidator, GradioErrorRecovery,
    GradioErrorMonitor, gradio_error_handler, validate_gradio_inputs,
    EnhancedGradioComponents, create_error_alert_component,
    create_success_alert_component, create_loading_component
)

# =============================================================================
# CUSTOM CSS FOR MODERN DESIGN
# =============================================================================

MODERN_CSS = """
/* Modern Design System */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #4ecdc4;
    --warning-color: #ff6b6b;
    --info-color: #45b7d1;
    --light-bg: #f8f9fa;
    --dark-bg: #343a40;
    --text-primary: #2c3e50;
    --text-secondary: #6c757d;
    --border-radius: 12px;
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    --transition: all 0.3s ease;
}

/* Global Styles */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header Design */
.app-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: var(--shadow);
}

.app-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.app-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0;
}

/* Card Design */
.feature-card {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow);
    border: 1px solid #e9ecef;
    transition: var(--transition);
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* Button Styles */
.primary-btn {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow) !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: var(--light-bg) !important;
    border: 2px solid var(--primary-color) !important;
    color: var(--primary-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
}

.secondary-btn:hover {
    background: var(--primary-color) !important;
    color: white !important;
}

/* Input Styles */
.modern-input {
    border-radius: var(--border-radius) !important;
    border: 2px solid #e9ecef !important;
    padding: 12px 16px !important;
    transition: var(--transition) !important;
}

.modern-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Slider Styles */
.modern-slider {
    border-radius: var(--border-radius) !important;
}

/* Progress Indicators */
.progress-bar {
    background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    border-radius: var(--border-radius);
    height: 8px;
    transition: width 0.3s ease;
}

/* Status Indicators */
.status-success {
    background: var(--success-color);
    color: white;
    padding: 8px 16px;
    border-radius: var(--border-radius);
    font-weight: 600;
}

.status-warning {
    background: var(--warning-color);
    color: white;
    padding: 8px 16px;
    border-radius: var(--border-radius);
    font-weight: 600;
}

.status-info {
    background: var(--info-color);
    color: white;
    padding: 8px 16px;
    border-radius: var(--border-radius);
    font-weight: 600;
}

/* Metric Cards */
.metric-card {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    border-left: 4px solid var(--primary-color);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Responsive Design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }
    
    .app-header {
        padding: 1.5rem;
    }
    
    .app-header h1 {
        font-size: 2rem;
    }
    
    .feature-card {
        padding: 1rem;
    }
}

/* Animation Classes */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.slide-in {
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Loading States */
.loading-spinner {
    border: 3px solid #f3f3f3;
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Tooltip Styles */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: var(--dark-bg);
    color: white;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
"""

# =============================================================================
# MAIN USER-FRIENDLY INTERFACE
# =============================================================================

class UserFriendlyInterface:
    """Main user-friendly interface with modern design and intuitive UX."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = OptimizedCache(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.evaluator = OptimizedEvaluator(self.config)
        self.video_processor = OptimizedVideoProcessor(self.config, self.cache, self.performance_monitor)
        self.api = OptimizedAPI(self.config, self.cache, self.performance_monitor)
        self.visualizer = DemoVisualizer()
        
        # Initialize models
        self.models = NeuralNetworkModels(self.config)
        self.diffusion_pipelines = DiffusionPipelines(self.config)
        self.text_processor = TextProcessor(self.config)
        self.video_utils = VideoProcessingUtils(self.config)
        
        # Initialize error handling components
        self.error_handler = GradioErrorHandler()
        self.input_validator = GradioInputValidator()
        self.error_recovery = GradioErrorRecovery()
        self.error_monitor = GradioErrorMonitor()
        self.enhanced_components = EnhancedGradioComponents()
        
        # User session data
        self.user_session = {
            "current_project": None,
            "recent_generations": [],
            "favorite_prompts": [],
            "performance_history": []
        }
    
    def create_main_interface(self):
        """Create the main user-friendly interface."""
        
        with gr.Blocks(
            title="Video-OpusClip AI Studio",
            theme=gr.themes.Soft(),
            css=MODERN_CSS
        ) as interface:
            
            # Modern Header
            gr.HTML("""
            <div class="app-header fade-in">
                <h1>🎬 Video-OpusClip AI Studio</h1>
                <p>Transform your ideas into viral videos with advanced AI technology</p>
                <div style="margin-top: 1rem;">
                    <span class="status-success">🚀 Ready to Create</span>
                </div>
            </div>
            """)
            
            # Main Navigation
            with gr.Tabs(selected=0) as main_tabs:
                
                # Quick Start Tab
                with gr.TabItem("🚀 Quick Start", id=0):
                    self._create_quick_start_tab()
                
                # AI Generation Tab
                with gr.TabItem("🤖 AI Generation", id=1):
                    self._create_ai_generation_tab()
                
                # Video Processing Tab
                with gr.TabItem("🎥 Video Processing", id=2):
                    self._create_video_processing_tab()
                
                # Viral Analysis Tab
                with gr.TabItem("📈 Viral Analysis", id=3):
                    self._create_viral_analysis_tab()
                
                # Performance Tab
                with gr.TabItem("⚡ Performance", id=4):
                    self._create_performance_tab()
                
                # Projects Tab
                with gr.TabItem("📁 Projects", id=5):
                    self._create_projects_tab()
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings", id=6):
                    self._create_settings_tab()
        
        return interface
    
    def _create_quick_start_tab(self):
        """Create quick start tab with guided workflows."""
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ### 🎯 Quick Start Guide
                Choose your workflow to get started quickly:
                """)
                
                # Workflow Selection
                with gr.Group():
                    workflow_choice = gr.Radio(
                        choices=[
                            "🎬 Generate video from text",
                            "🖼️ Transform image to video", 
                            "📈 Analyze content viral potential",
                            "🎥 Process existing video"
                        ],
                        value="🎬 Generate video from text",
                        label="Select Workflow",
                        elem_classes=["modern-input"]
                    )
                
                # Dynamic workflow content
                with gr.Group(visible=True) as text_to_video_workflow:
                    gr.Markdown("### 🎬 Text-to-Video Generation")
                    
                    # Enhanced components with validation
                    quick_prompt, prompt_error = self.enhanced_components.create_validated_textbox(
                        label="Describe your video",
                        placeholder="A majestic dragon flying through a mystical forest at sunset...",
                        lines=3,
                        elem_classes=["modern-input"]
                    )
                    
                    with gr.Row():
                        quick_duration, duration_error = self.enhanced_components.create_validated_slider(
                            minimum=3, maximum=30, value=10, step=1,
                            label="Duration (seconds)",
                            elem_classes=["modern-slider"]
                        )
                        
                        quick_quality = gr.Dropdown(
                            choices=["Fast", "Balanced", "High Quality"],
                            value="Balanced",
                            label="Quality",
                            elem_classes=["modern-input"]
                        )
                    
                    quick_generate_btn = gr.Button(
                        "🎬 Generate Video",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    
                    # Error and success alerts
                    quick_error_alert = create_error_alert_component()
                    quick_success_alert = create_success_alert_component()
                    quick_loading = create_loading_component()
                
                with gr.Group(visible=False) as image_to_video_workflow:
                    gr.Markdown("### 🖼️ Image-to-Video Transformation")
                    
                    quick_image = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        elem_classes=["modern-input"]
                    )
                    
                    quick_motion = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="Motion Strength",
                        elem_classes=["modern-slider"]
                    )
                    
                    quick_transform_btn = gr.Button(
                        "🎬 Transform to Video",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                
                with gr.Group(visible=False) as viral_analysis_workflow:
                    gr.Markdown("### 📈 Viral Analysis")
                    
                    quick_content = gr.Textbox(
                        label="Content Description",
                        placeholder="Describe your content...",
                        lines=3,
                        elem_classes=["modern-input"]
                    )
                    
                    quick_platform = gr.Dropdown(
                        choices=["TikTok", "YouTube", "Instagram", "Twitter"],
                        value="TikTok",
                        label="Target Platform",
                        elem_classes=["modern-input"]
                    )
                    
                    quick_analyze_btn = gr.Button(
                        "🔍 Analyze Viral Potential",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                
                with gr.Group(visible=False) as video_processing_workflow:
                    gr.Markdown("### 🎥 Video Processing")
                    
                    quick_video = gr.Video(
                        label="Upload Video",
                        elem_classes=["modern-input"]
                    )
                    
                    quick_process_btn = gr.Button(
                        "🎥 Process Video",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
            
            with gr.Column(scale=1):
                # Results Panel
                gr.Markdown("### 📤 Results")
                
                quick_output = gr.Video(
                    label="Generated Content",
                    elem_classes=["feature-card"]
                )
                
                quick_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["modern-input"]
                )
                
                # Quick Metrics
                with gr.Group():
                    gr.Markdown("### 📊 Quick Metrics")
                    
                    quick_metrics = gr.JSON(
                        label="Generation Metrics",
                        elem_classes=["feature-card"]
                    )
        
        # Event handlers for workflow switching
        workflow_choice.change(
            fn=self._switch_workflow,
            inputs=[workflow_choice],
            outputs=[text_to_video_workflow, image_to_video_workflow, viral_analysis_workflow, video_processing_workflow]
        )
        
        # Event handlers for quick actions
        def handle_quick_generation(prompt, duration, quality):
            """Handle quick generation with error handling."""
            try:
                with quick_loading:
                    result = self._quick_text_to_video(prompt, duration, quality)
                
                if isinstance(result, tuple) and len(result) == 3:
                    video, status, metrics = result
                    
                    if video is not None and "error" not in str(metrics):
                        return {
                            quick_output: video,
                            quick_status: status,
                            quick_metrics: metrics,
                            quick_error_alert: gr.update(visible=False),
                            quick_success_alert: gr.update(visible=True, value=status)
                        }
                    else:
                        return {
                            quick_output: None,
                            quick_status: status,
                            quick_metrics: metrics,
                            quick_error_alert: gr.update(visible=True, value=status),
                            quick_success_alert: gr.update(visible=False)
                        }
                else:
                    return {
                        quick_output: None,
                        quick_status: "Unexpected result format",
                        quick_metrics: {"status": "error"},
                        quick_error_alert: gr.update(visible=True, value="Unexpected result format"),
                        quick_success_alert: gr.update(visible=False)
                    }
                    
            except Exception as e:
                error_response = self.error_handler.handle_gradio_error(e, "quick_generation")
                return {
                    quick_output: None,
                    quick_status: error_response["error_message"],
                    quick_metrics: {"status": "error", "error": error_response},
                    quick_error_alert: gr.update(visible=True, value=error_response["error_message"]),
                    quick_success_alert: gr.update(visible=False)
                }
        
        quick_generate_btn.click(
            fn=handle_quick_generation,
            inputs=[quick_prompt, quick_duration, quick_quality],
            outputs=[quick_output, quick_status, quick_metrics, quick_error_alert, quick_success_alert]
        )
        
        quick_transform_btn.click(
            fn=self._quick_image_to_video,
            inputs=[quick_image, quick_motion],
            outputs=[quick_output, quick_status, quick_metrics]
        )
        
        quick_analyze_btn.click(
            fn=self._quick_viral_analysis,
            inputs=[quick_content, quick_platform],
            outputs=[quick_status, quick_metrics]
        )
        
        quick_process_btn.click(
            fn=self._quick_video_processing,
            inputs=[quick_video],
            outputs=[quick_output, quick_status, quick_metrics]
        )
    
    def _create_ai_generation_tab(self):
        """Create advanced AI generation tab."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 AI Generation Studio")
                
                # Model Selection
                with gr.Group():
                    gr.Markdown("#### 🎯 Model Configuration")
                    
                    model_type = gr.Dropdown(
                        choices=["Stable Diffusion", "DeepFloyd", "Kandinsky", "Custom"],
                        value="Stable Diffusion",
                        label="AI Model",
                        elem_classes=["modern-input"]
                    )
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=1.0, maximum=20.0, value=7.5, step=0.1,
                            label="Guidance Scale",
                            elem_classes=["modern-slider"]
                        )
                        
                        num_steps = gr.Slider(
                            minimum=10, maximum=100, value=30, step=1,
                            label="Inference Steps",
                            elem_classes=["modern-slider"]
                        )
                    
                    seed_input = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)",
                        elem_classes=["modern-input"]
                    )
                
                # Advanced Settings
                with gr.Group():
                    gr.Markdown("#### ⚙️ Advanced Settings")
                    
                    with gr.Row():
                        enable_audio = gr.Checkbox(
                            value=True,
                            label="Generate Audio",
                            elem_classes=["modern-input"]
                        )
                        
                        enable_subtitles = gr.Checkbox(
                            value=True,
                            label="Generate Subtitles",
                            elem_classes=["modern-input"]
                        )
                    
                    style_preset = gr.Dropdown(
                        choices=["Cinematic", "Artistic", "Realistic", "Anime", "Custom"],
                        value="Cinematic",
                        label="Style Preset",
                        elem_classes=["modern-input"]
                    )
            
            with gr.Column(scale=1):
                # Input Section
                gr.Markdown("### 📝 Content Input")
                
                prompt_input = gr.Textbox(
                    label="Video Prompt",
                    placeholder="Describe your video in detail...",
                    lines=4,
                    elem_classes=["modern-input"]
                )
                
                with gr.Row():
                    duration = gr.Slider(
                        minimum=3, maximum=60, value=15, step=1,
                        label="Duration (seconds)",
                        elem_classes=["modern-slider"]
                    )
                    
                    quality = gr.Dropdown(
                        choices=["Fast", "Balanced", "High Quality", "Ultra Quality"],
                        value="Balanced",
                        label="Quality Level",
                        elem_classes=["modern-input"]
                    )
                
                # Generation Controls
                with gr.Row():
                    generate_btn = gr.Button(
                        "🎬 Generate Video",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    
                    stop_btn = gr.Button(
                        "⏹️ Stop Generation",
                        variant="secondary",
                        size="lg",
                        elem_classes=["secondary-btn"]
                    )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Output Section
                gr.Markdown("### 📤 Generated Content")
                
                output_video = gr.Video(
                    label="Generated Video",
                    elem_classes=["feature-card"]
                )
                
                generation_status = gr.Textbox(
                    label="Generation Status",
                    interactive=False,
                    elem_classes=["modern-input"]
                )
            
            with gr.Column(scale=1):
                # Metrics Section
                gr.Markdown("### 📊 Generation Metrics")
                
                generation_metrics = gr.JSON(
                    label="Detailed Metrics",
                    elem_classes=["feature-card"]
                )
                
                # Progress Bar
                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Generation Progress",
                    interactive=False,
                    elem_classes=["modern-slider"]
                )
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_advanced_video,
            inputs=[
                prompt_input, duration, quality, model_type,
                guidance_scale, num_steps, seed_input,
                enable_audio, enable_subtitles, style_preset
            ],
            outputs=[output_video, generation_status, generation_metrics, progress_bar]
        )
    
    def _create_video_processing_tab(self):
        """Create video processing tab."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Video Processing Studio")
                
                # Input Section
                with gr.Group():
                    gr.Markdown("#### 📁 Input")
                    
                    video_input = gr.Video(
                        label="Upload Video",
                        elem_classes=["modern-input"]
                    )
                    
                    url_input = gr.Textbox(
                        label="Or Video URL",
                        placeholder="https://example.com/video.mp4",
                        elem_classes=["modern-input"]
                    )
                
                # Processing Options
                with gr.Group():
                    gr.Markdown("#### ⚙️ Processing Options")
                    
                    with gr.Row():
                        target_duration = gr.Slider(
                            minimum=5, maximum=300, value=30, step=5,
                            label="Target Duration (seconds)",
                            elem_classes=["modern-slider"]
                        )
                        
                        target_resolution = gr.Dropdown(
                            choices=["720p", "1080p", "1440p", "4K"],
                            value="1080p",
                            label="Target Resolution",
                            elem_classes=["modern-input"]
                        )
                    
                    with gr.Row():
                        enable_enhancement = gr.Checkbox(
                            value=True,
                            label="AI Enhancement",
                            elem_classes=["modern-input"]
                        )
                        
                        enable_stabilization = gr.Checkbox(
                            value=False,
                            label="Video Stabilization",
                            elem_classes=["modern-input"]
                        )
                    
                    processing_preset = gr.Dropdown(
                        choices=["Fast", "Balanced", "Quality", "Custom"],
                        value="Balanced",
                        label="Processing Preset",
                        elem_classes=["modern-input"]
                    )
            
            with gr.Column(scale=1):
                # Output Section
                gr.Markdown("### 📤 Processed Video")
                
                processed_video = gr.Video(
                    label="Processed Video",
                    elem_classes=["feature-card"]
                )
                
                processing_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    elem_classes=["modern-input"]
                )
                
                # Processing Metrics
                processing_metrics = gr.JSON(
                    label="Processing Metrics",
                    elem_classes=["feature-card"]
                )
        
        with gr.Row():
            # Processing Controls
            with gr.Column(scale=1):
                process_btn = gr.Button(
                    "🎥 Process Video",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"]
                )
            
            with gr.Column(scale=1):
                # Quick Actions
                with gr.Group():
                    gr.Markdown("#### ⚡ Quick Actions")
                    
                    with gr.Row():
                        enhance_btn = gr.Button(
                            "✨ Enhance",
                            variant="secondary",
                            elem_classes=["secondary-btn"]
                        )
                        
                        compress_btn = gr.Button(
                            "🗜️ Compress",
                            variant="secondary",
                            elem_classes=["secondary-btn"]
                        )
                        
                        convert_btn = gr.Button(
                            "🔄 Convert",
                            variant="secondary",
                            elem_classes=["secondary-btn"]
                        )
        
        # Event handlers
        process_btn.click(
            fn=self._process_video,
            inputs=[
                video_input, url_input, target_duration, target_resolution,
                enable_enhancement, enable_stabilization, processing_preset
            ],
            outputs=[processed_video, processing_status, processing_metrics]
        )
    
    def _create_viral_analysis_tab(self):
        """Create viral analysis tab."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Viral Analysis Studio")
                
                # Content Input
                with gr.Group():
                    gr.Markdown("#### 📝 Content Analysis")
                    
                    content_input = gr.Textbox(
                        label="Content Description",
                        placeholder="Describe your content in detail...",
                        lines=4,
                        elem_classes=["modern-input"]
                    )
                    
                    with gr.Row():
                        content_type = gr.Dropdown(
                            choices=["Video", "Image", "Text", "Story"],
                            value="Video",
                            label="Content Type",
                            elem_classes=["modern-input"]
                        )
                        
                        target_platform = gr.Dropdown(
                            choices=["TikTok", "YouTube", "Instagram", "Twitter", "Facebook"],
                            value="TikTok",
                            label="Target Platform",
                            elem_classes=["modern-input"]
                        )
                    
                    with gr.Row():
                        target_audience = gr.Dropdown(
                            choices=["Gen Z", "Millennials", "Gen X", "Boomers", "All"],
                            value="Gen Z",
                            label="Target Audience",
                            elem_classes=["modern-input"]
                        )
                        
                        content_category = gr.Dropdown(
                            choices=["Entertainment", "Education", "News", "Lifestyle", "Gaming", "Music"],
                            value="Entertainment",
                            label="Content Category",
                            elem_classes=["modern-input"]
                        )
                
                # Analysis Options
                with gr.Group():
                    gr.Markdown("#### 🔍 Analysis Options")
                    
                    with gr.Row():
                        include_hashtags = gr.Checkbox(
                            value=True,
                            label="Hashtag Analysis",
                            elem_classes=["modern-input"]
                        )
                        
                        analyze_sentiment = gr.Checkbox(
                            value=True,
                            label="Sentiment Analysis",
                            elem_classes=["modern-input"]
                        )
                        
                        predict_engagement = gr.Checkbox(
                            value=True,
                            label="Engagement Prediction",
                            elem_classes=["modern-input"]
                        )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Viral Potential",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
            
            with gr.Column(scale=1):
                # Results Section
                gr.Markdown("### 📊 Analysis Results")
                
                with gr.Row():
                    viral_score = gr.Number(
                        label="Viral Score",
                        interactive=False,
                        elem_classes=["modern-input"]
                    )
                    
                    engagement_prediction = gr.Number(
                        label="Engagement Prediction",
                        interactive=False,
                        elem_classes=["modern-input"]
                    )
                
                recommendations = gr.Textbox(
                    label="Optimization Recommendations",
                    lines=5,
                    interactive=False,
                    elem_classes=["modern-input"]
                )
                
                analysis_metrics = gr.JSON(
                    label="Detailed Analysis",
                    elem_classes=["feature-card"]
                )
        
        with gr.Row():
            # Visualization Section
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Visual Analytics")
                
                viral_chart = gr.Plot(
                    label="Viral Potential Breakdown",
                    elem_classes=["feature-card"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Platform Comparison")
                
                platform_chart = gr.Plot(
                    label="Platform Performance",
                    elem_classes=["feature-card"]
                )
        
        # Event handlers
        analyze_btn.click(
            fn=self._analyze_viral_potential,
            inputs=[
                content_input, content_type, target_platform,
                target_audience, content_category,
                include_hashtags, analyze_sentiment, predict_engagement
            ],
            outputs=[
                viral_score, engagement_prediction, recommendations,
                analysis_metrics, viral_chart, platform_chart
            ]
        )
    
    def _create_performance_tab(self):
        """Create performance monitoring tab."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Performance Dashboard")
                
                # System Overview
                with gr.Group():
                    gr.Markdown("#### 💻 System Overview")
                    
                    with gr.Row():
                        cpu_usage = gr.Number(
                            label="CPU Usage (%)",
                            interactive=False,
                            elem_classes=["modern-input"]
                        )
                        
                        memory_usage = gr.Number(
                            label="Memory Usage (%)",
                            interactive=False,
                            elem_classes=["modern-input"]
                        )
                    
                    with gr.Row():
                        gpu_usage = gr.Number(
                            label="GPU Usage (%)",
                            interactive=False,
                            elem_classes=["modern-input"]
                        )
                        
                        disk_usage = gr.Number(
                            label="Disk Usage (%)",
                            interactive=False,
                            elem_classes=["modern-input"]
                        )
                
                # Performance Controls
                with gr.Group():
                    gr.Markdown("#### 🎛️ Monitoring Controls")
                    
                    with gr.Row():
                        refresh_interval = gr.Slider(
                            minimum=1, maximum=30, value=5, step=1,
                            label="Refresh Interval (seconds)",
                            elem_classes=["modern-slider"]
                        )
                        
                        monitoring_duration = gr.Slider(
                            minimum=30, maximum=300, value=60, step=30,
                            label="Monitoring Duration (seconds)",
                            elem_classes=["modern-slider"]
                        )
                    
                    with gr.Row():
                        start_monitoring_btn = gr.Button(
                            "▶️ Start Monitoring",
                            variant="primary",
                            elem_classes=["primary-btn"]
                        )
                        
                        stop_monitoring_btn = gr.Button(
                            "⏹️ Stop Monitoring",
                            variant="secondary",
                            elem_classes=["secondary-btn"]
                        )
            
            with gr.Column(scale=1):
                # Performance Charts
                gr.Markdown("### 📊 Performance Charts")
                
                cpu_chart = gr.Plot(
                    label="CPU Usage Over Time",
                    elem_classes=["feature-card"]
                )
                
                memory_chart = gr.Plot(
                    label="Memory Usage Over Time",
                    elem_classes=["feature-card"]
                )
        
        with gr.Row():
            # Detailed Metrics
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Detailed Metrics")
                
                performance_metrics = gr.JSON(
                    label="System Metrics",
                    elem_classes=["feature-card"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🚨 Performance Alerts")
                
                performance_alerts = gr.JSON(
                    label="Active Alerts",
                    elem_classes=["feature-card"]
                )
        
        # Event handlers
        start_monitoring_btn.click(
            fn=self._start_performance_monitoring,
            inputs=[refresh_interval, monitoring_duration],
            outputs=[cpu_usage, memory_usage, gpu_usage, disk_usage, performance_metrics]
        )
    
    def _create_projects_tab(self):
        """Create projects management tab."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Project Management")
                
                # Project List
                with gr.Group():
                    gr.Markdown("#### 📋 Your Projects")
                    
                    project_list = gr.Dropdown(
                        choices=["No projects yet"],
                        value="No projects yet",
                        label="Select Project",
                        elem_classes=["modern-input"]
                    )
                    
                    with gr.Row():
                        new_project_btn = gr.Button(
                            "➕ New Project",
                            variant="primary",
                            elem_classes=["primary-btn"]
                        )
                        
                        delete_project_btn = gr.Button(
                            "🗑️ Delete Project",
                            variant="secondary",
                            elem_classes=["secondary-btn"]
                        )
                
                # Project Details
                with gr.Group():
                    gr.Markdown("#### 📝 Project Details")
                    
                    project_name = gr.Textbox(
                        label="Project Name",
                        elem_classes=["modern-input"]
                    )
                    
                    project_description = gr.Textbox(
                        label="Description",
                        lines=3,
                        elem_classes=["modern-input"]
                    )
                    
                    project_type = gr.Dropdown(
                        choices=["Video Generation", "Video Processing", "Viral Analysis", "Mixed"],
                        value="Video Generation",
                        label="Project Type",
                        elem_classes=["modern-input"]
                    )
            
            with gr.Column(scale=1):
                # Project Assets
                gr.Markdown("### 🎬 Project Assets")
                
                project_assets = gr.File(
                    label="Project Files",
                    file_count="multiple",
                    elem_classes=["feature-card"]
                )
                
                # Recent Activity
                gr.Markdown("### 📅 Recent Activity")
                
                recent_activity = gr.JSON(
                    label="Activity Log",
                    elem_classes=["feature-card"]
                )
        
        with gr.Row():
            # Project Statistics
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Project Statistics")
                
                project_stats = gr.JSON(
                    label="Statistics",
                    elem_classes=["feature-card"]
                )
            
            with gr.Column(scale=1):
                # Export Options
                gr.Markdown("### 📤 Export Options")
                
                with gr.Group():
                    export_format = gr.Dropdown(
                        choices=["JSON", "CSV", "PDF", "ZIP"],
                        value="JSON",
                        label="Export Format",
                        elem_classes=["modern-input"]
                    )
                    
                    export_btn = gr.Button(
                        "📤 Export Project",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )
    
    def _create_settings_tab(self):
        """Create settings tab."""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Application Settings")
                
                # General Settings
                with gr.Group():
                    gr.Markdown("#### 🔧 General Settings")
                    
                    with gr.Row():
                        max_workers = gr.Slider(
                            minimum=1, maximum=32, value=8, step=1,
                            label="Max Workers",
                            elem_classes=["modern-slider"]
                        )
                        
                        batch_size = gr.Slider(
                            minimum=1, maximum=100, value=16, step=1,
                            label="Batch Size",
                            elem_classes=["modern-slider"]
                        )
                    
                    with gr.Row():
                        enable_gpu = gr.Checkbox(
                            value=True,
                            label="Enable GPU",
                            elem_classes=["modern-input"]
                        )
                        
                        enable_caching = gr.Checkbox(
                            value=True,
                            label="Enable Caching",
                            elem_classes=["modern-input"]
                        )
                    
                    log_level = gr.Dropdown(
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        value="INFO",
                        label="Log Level",
                        elem_classes=["modern-input"]
                    )
                
                # Performance Settings
                with gr.Group():
                    gr.Markdown("#### ⚡ Performance Settings")
                    
                    with gr.Row():
                        cache_size = gr.Slider(
                            minimum=100, maximum=10000, value=1000, step=100,
                            label="Cache Size",
                            elem_classes=["modern-slider"]
                        )
                        
                        timeout = gr.Slider(
                            minimum=10, maximum=300, value=60, step=10,
                            label="Timeout (seconds)",
                            elem_classes=["modern-slider"]
                        )
                    
                    enable_optimization = gr.Checkbox(
                        value=True,
                        label="Enable Auto-Optimization",
                        elem_classes=["modern-input"]
                    )
            
            with gr.Column(scale=1):
                # User Preferences
                gr.Markdown("### 👤 User Preferences")
                
                with gr.Group():
                    gr.Markdown("#### 🎨 Interface Preferences")
                    
                    theme_choice = gr.Dropdown(
                        choices=["Light", "Dark", "Auto"],
                        value="Light",
                        label="Theme",
                        elem_classes=["modern-input"]
                    )
                    
                    language_choice = gr.Dropdown(
                        choices=["English", "Spanish", "French", "German"],
                        value="English",
                        label="Language",
                        elem_classes=["modern-input"]
                    )
                    
                    notifications = gr.Checkbox(
                        value=True,
                        label="Enable Notifications",
                        elem_classes=["modern-input"]
                    )
                
                # Save Settings
                with gr.Group():
                    save_settings_btn = gr.Button(
                        "💾 Save Settings",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    
                    reset_settings_btn = gr.Button(
                        "🔄 Reset to Defaults",
                        variant="secondary",
                        elem_classes=["secondary-btn"]
                    )
        
        # Event handlers
        save_settings_btn.click(
            fn=self._save_settings,
            inputs=[
                max_workers, batch_size, enable_gpu, enable_caching, log_level,
                cache_size, timeout, enable_optimization,
                theme_choice, language_choice, notifications
            ],
            outputs=[]
        )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

    def _switch_workflow(self, choice):
        """Switch between different workflows."""
        workflows = [
            choice == "🎬 Generate video from text",
            choice == "🖼️ Transform image to video",
            choice == "📈 Analyze content viral potential",
            choice == "🎥 Process existing video"
        ]
        return workflows
    
    @gradio_error_handler
    @validate_gradio_inputs("text_prompt", "duration", "quality")
    @error_monitor.monitor_function
    def _quick_text_to_video(self, prompt, duration, quality):
        """Quick text-to-video generation with error handling."""
        
        # Additional validation
        is_valid, message = self.input_validator.validate_text_prompt(prompt)
        if not is_valid:
            raise ValueError(message)
        
        is_valid, message = self.input_validator.validate_duration(duration)
        if not is_valid:
            raise ValueError(message)
        
        is_valid, message = self.input_validator.validate_quality(quality)
        if not is_valid:
            raise ValueError(message)
        
        try:
            # Simulate generation
            time.sleep(2)
            
            # Create sample video (in real implementation, this would call the AI model)
            sample_video = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            metrics = {
                "generation_time": 2.5,
                "prompt_length": len(prompt),
                "quality": quality,
                "duration": duration,
                "status": "success"
            }
            
            return sample_video, "✅ Video generated successfully!", metrics
            
        except Exception as e:
            # Attempt recovery
            recovery_result = self.error_recovery.attempt_recovery(e, "quick_text_to_video")
            
            if recovery_result["recovered"]:
                # Retry with recovery settings
                return self._quick_text_to_video(prompt, duration, "Fast")  # Fallback to fast quality
            else:
                # Return error response
                error_response = self.error_handler.handle_gradio_error(e, "quick_text_to_video")
                return None, error_response["error_message"], {"status": "error", "error": error_response}
    
    def _quick_image_to_video(self, image, motion_strength):
        """Quick image-to-video transformation."""
        try:
            if image is None:
                return None, "❌ Please provide an image", {}
            
            # Simulate transformation
            time.sleep(1.5)
            
            # Create sample video
            sample_video = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            metrics = {
                "transformation_time": 1.5,
                "motion_strength": motion_strength,
                "image_size": image.shape[:2] if image is not None else None
            }
            
            return sample_video, "✅ Video transformed successfully!", metrics
        except Exception as e:
            return None, f"❌ Error: {str(e)}", {}
    
    def _quick_viral_analysis(self, content, platform):
        """Quick viral analysis."""
        try:
            # Simulate analysis
            time.sleep(1)
            
            viral_score = 0.75
            engagement = 0.68
            
            metrics = {
                "viral_score": viral_score,
                "engagement_prediction": engagement,
                "platform": platform,
                "content_length": len(content)
            }
            
            return f"✅ Analysis complete! Viral Score: {viral_score:.2f}", metrics
        except Exception as e:
            return f"❌ Error: {str(e)}", {}
    
    def _quick_video_processing(self, video):
        """Quick video processing."""
        try:
            if video is None:
                return None, "❌ Please provide a video", {}
            
            # Simulate processing
            time.sleep(2)
            
            metrics = {
                "processing_time": 2.0,
                "original_size": "10MB",
                "processed_size": "8MB",
                "compression_ratio": 0.8
            }
            
            return video, "✅ Video processed successfully!", metrics
        except Exception as e:
            return None, f"❌ Error: {str(e)}", {}

# =============================================================================
# LAUNCH FUNCTION
# =============================================================================

def launch_user_friendly_interface(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the user-friendly interface."""
    
    # Create interface
    interface = UserFriendlyInterface()
    demo = interface.create_main_interface()
    
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
    # Launch the user-friendly interface
    launch_user_friendly_interface(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    ) 