#!/usr/bin/env python3
"""
🚀 Gradio Demo Launcher for Facebook Posts Feature
==================================================

Central hub for launching all Gradio demos related to the Facebook Posts feature.
Provides easy access to gradient clipping, NaN handling, performance optimization,
and experiment tracking interfaces.
"""

import os
import sys
import json
import time
import threading
import subprocess
import socket
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import gradio as gr

# Import our centralized logging configuration
from logging_config import (
    get_logger, log_system_event, log_error_with_context
)

warnings.filterwarnings('ignore')

logger = get_logger(__name__)

# =============================================================================
# DEMO CONFIGURATIONS
# =============================================================================

DEMO_CONFIGS = {
    "gradient_clipping_nan_handling": {
        "name": "Gradient Clipping & NaN Handling",
        "description": "Interactive interface for numerical stability techniques",
        "file": "gradio_enhanced_interface.py",
        "port": 7861,
        "category": "Numerical Stability"
    },
    "performance_optimization": {
        "name": "Performance Optimization",
        "description": "Multi-GPU training, gradient accumulation, and mixed precision",
        "file": "gradio_performance_optimization.py",
        "port": 7862,
        "category": "Performance"
    },
    "real_time_training": {
        "name": "Real-Time Training Demo",
        "description": "Live training visualization and monitoring",
        "file": "gradio_realtime_training_demo.py",
        "port": 7863,
        "category": "Training"
    },
    "experiment_tracking": {
        "name": "Experiment Tracking",
        "description": "TensorBoard and Weights & Biases integration with Transformers and Diffusers support",
        "file": "gradio_experiment_tracking.py",
        "port": 7864,
        "category": "Monitoring"
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port: int) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    while port < start_port + 100:
        if check_port_available(port):
            return port
        port += 1
    return start_port

def get_demo_status(demo_name: str) -> Dict[str, Any]:
    """Get the status of a specific demo."""
    config = DEMO_CONFIGS.get(demo_name)
    if not config:
        return {"error": f"Unknown demo: {demo_name}"}
    
    try:
        # Check if demo is running
        port = config["port"]
        is_running = not check_port_available(port)
        
        # Check if demo file exists
        file_path = Path(config["file"])
        file_exists = file_path.exists()
        
        # Get file size and modification time
        file_info = {}
        if file_exists:
            stat = file_path.stat()
            file_info = {
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            }
        
        return {
            "name": config["name"],
            "description": config["description"],
            "category": config["category"],
            "port": port,
            "is_running": is_running,
            "file_exists": file_exists,
            "file_info": file_info,
            "status": "Running" if is_running else "Stopped" if file_exists else "Missing"
        }
        
    except Exception as e:
        logger.error(f"Error getting demo status for {demo_name}: {e}")
        return {"error": str(e)}

def get_all_demo_statuses() -> Dict[str, Dict[str, Any]]:
    """Get status of all demos."""
    return {name: get_demo_status(name) for name in DEMO_CONFIGS.keys()}

def launch_demo(demo_name: str) -> str:
    """Launch a specific demo."""
    config = DEMO_CONFIGS.get(demo_name)
    if not config:
        return f"❌ Unknown demo: {demo_name}"
    
    try:
        # Check if demo is already running
        if not check_port_available(config["port"]):
            return f"❌ Demo {config['name']} is already running on port {config['port']}"
        
        # Check if demo file exists
        file_path = Path(config["file"])
        if not file_path.exists():
            return f"❌ Demo file not found: {config['file']}"
        
        # Find available port
        available_port = find_available_port(config["port"])
        
        # Launch demo in subprocess
        cmd = [sys.executable, str(file_path)]
        env = os.environ.copy()
        env["GRADIO_SERVER_PORT"] = str(available_port)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=Path(__file__).parent
        )
        
        # Wait a moment to see if it starts successfully
        time.sleep(3)
        if process.poll() is None:
            # Demo started successfully
            url = f"http://localhost:{available_port}"
            logger.info(f"Demo {demo_name} launched successfully on port {available_port}")
            return f"✅ Demo {config['name']} launched successfully!\n\n" \
                   f"🌐 Access at: {url}\n" \
                   f"🔌 Port: {available_port}\n" \
                   f"📁 File: {config['file']}\n\n" \
                   f"Click the link above to open the demo in your browser."
        else:
            # Demo failed to start
            stdout, stderr = process.communicate()
            error_msg = stderr if stderr else stdout
            logger.error(f"Demo {demo_name} failed to start: {error_msg}")
            return f"❌ Failed to launch demo {config['name']}:\n{error_msg}"
            
    except Exception as e:
        logger.error(f"Error launching demo {demo_name}: {e}")
        return f"❌ Error launching demo: {e}"

def stop_demo(demo_name: str) -> str:
    """Stop a specific demo."""
    config = DEMO_CONFIGS.get(demo_name)
    if not config:
        return f"❌ Unknown demo: {demo_name}"
    
    try:
        # Check if demo is running
        if check_port_available(config["port"]):
            return f"❌ Demo {config['name']} is not running"
        
        # Find and kill process using the port
        try:
            # Use netstat to find process using the port
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if f":{config['port']}" in line and "LISTENING" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                # Kill the process
                                subprocess.run(["taskkill", "/PID", pid, "/F"], shell=True)
                                logger.info(f"Stopped demo {demo_name} (PID: {pid})")
                                return f"✅ Demo {config['name']} stopped successfully (PID: {pid})"
                            except Exception as e:
                                logger.error(f"Failed to kill process {pid}: {e}")
                                return f"❌ Failed to stop demo: {e}"
            
            return f"❌ Could not find process using port {config['port']}"
            
        except Exception as e:
            logger.error(f"Error stopping demo {demo_name}: {e}")
            return f"❌ Error stopping demo: {e}"
            
    except Exception as e:
        logger.error(f"Error stopping demo {demo_name}: {e}")
        return f"❌ Error stopping demo: {e}"

def refresh_demo_statuses() -> Dict[str, Dict[str, Any]]:
    """Refresh the status of all demos."""
    return get_all_demo_statuses()

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent
        }
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_info = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 2)
        }
        
        # Python info
        python_info = {
            "version": sys.version,
            "executable": sys.executable
        }
        
        return {
            "cpu": {"count": cpu_count, "usage_percent": cpu_percent},
            "memory": memory_info,
            "disk": disk_info,
            "python": python_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_demo_launcher_interface():
    """Create the main demo launcher interface."""
    
    with gr.Blocks(title="🚀 Gradio Demo Launcher", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🚀 Gradio Demo Launcher
        
        Central hub for launching all Gradio demos related to the Facebook Posts feature.
        Manage numerical stability, performance optimization, training visualization, and experiment tracking.
        **Now with enhanced Transformers, Language Model, and Diffusion Model support!**
        """)
        
        with gr.Tabs():
            
            # Demo Overview Tab
            with gr.Tab("📋 Demo Overview"):
                gr.Markdown("### Available Demos")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
                    system_info_btn = gr.Button("💻 System Info")
                
                # Demo status table
                demo_status_output = gr.JSON(
                    label="Demo Status",
                    value=get_all_demo_statuses()
                )
                
                # System information
                system_info_output = gr.JSON(
                    label="System Information",
                    value={}
                )
                
                # Event handlers
                refresh_btn.click(
                    fn=refresh_demo_statuses,
                    outputs=demo_status_output
                )
                
                system_info_btn.click(
                    fn=get_system_info,
                    outputs=system_info_output
                )
            
            # Demo Management Tab
            with gr.Tab("🚀 Demo Management"):
                gr.Markdown("### Launch and Control Demos")
                
                with gr.Row():
                    demo_selector = gr.Dropdown(
                        label="Select Demo",
                        choices=list(DEMO_CONFIGS.keys()),
                        value=list(DEMO_CONFIGS.keys())[0] if DEMO_CONFIGS else None
                    )
                
                with gr.Row():
                    launch_btn = gr.Button("🚀 Launch Demo", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Demo", variant="stop")
                
                with gr.Row():
                    demo_output = gr.Textbox(
                        label="Demo Output",
                        lines=8,
                        interactive=False
                    )
                
                # Event handlers
                launch_btn.click(
                    fn=launch_demo,
                    inputs=[demo_selector],
                    outputs=demo_output
                )
                
                stop_btn.click(
                    fn=stop_demo,
                    inputs=[demo_selector],
                    outputs=demo_output
                )
            
            # Enhanced Quick Actions Tab
            with gr.Tab("⚡ Quick Actions"):
                gr.Markdown("### Quick Demo Actions")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🧮 Numerical Stability")
                        launch_numerical_btn = gr.Button(
                            "Launch Gradient Clipping & NaN Handling",
                            variant="primary",
                            size="sm"
                        )
                        
                        gr.Markdown("#### 🚀 Performance")
                        launch_performance_btn = gr.Button(
                            "Launch Performance Optimization",
                            variant="primary",
                            size="sm"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📊 Training")
                        launch_training_btn = gr.Button(
                            "Launch Real-Time Training",
                            variant="primary",
                            size="sm"
                        )
                        
                        gr.Markdown("#### 🔬 Experiment Tracking")
                        launch_tracking_btn = gr.Button(
                            "Launch Experiment Tracking",
                            variant="primary",
                            size="sm"
                        )
                
                with gr.Row():
                    gr.Markdown("#### 🤖 **NEW: Transformers & Language Models**")
                    launch_transformers_btn = gr.Button(
                        "🚀 Launch Language Model Tracking",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Row():
                    gr.Markdown("#### 🎨 **NEW: Diffusion Models & Image Generation**")
                    launch_diffusion_btn = gr.Button(
                        "🎨 Launch Diffusion Model Tracking",
                        variant="primary",
                        size="lg"
                    )
                
                quick_actions_output = gr.Textbox(
                    label="Quick Actions Output",
                    lines=6,
                    interactive=False
                )
                
                # Event handlers for quick actions
                launch_numerical_btn.click(
                    fn=lambda: launch_demo("gradient_clipping_nan_handling"),
                    outputs=quick_actions_output
                )
                
                launch_performance_btn.click(
                    fn=lambda: launch_demo("performance_optimization"),
                    outputs=quick_actions_output
                )
                
                launch_training_btn.click(
                    fn=lambda: launch_demo("real_time_training"),
                    outputs=quick_actions_output
                )
                
                launch_tracking_btn.click(
                    fn=lambda: launch_demo("experiment_tracking"),
                    outputs=quick_actions_output
                )
                
                launch_transformers_btn.click(
                    fn=lambda: launch_demo("experiment_tracking"),
                    outputs=quick_actions_output
                )
                
                launch_diffusion_btn.click(
                    fn=lambda: launch_demo("experiment_tracking"),
                    outputs=quick_actions_output
                )
            
            # Enhanced Help Tab
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## 📚 Demo Launcher Guide
                
                ### 🚀 Getting Started
                1. **Overview**: Check the status of all available demos
                2. **Management**: Launch and stop specific demos
                3. **Quick Actions**: Use one-click buttons for common demos
                
                ### 🔧 Available Demos
                
                #### 🧮 Numerical Stability
                - **Gradient Clipping & NaN Handling**: Interactive interface for numerical stability techniques
                - Features: Gradient clipping methods, NaN/Inf detection, stability monitoring
                
                #### 🚀 Performance Optimization
                - **Performance Optimization**: Multi-GPU training, gradient accumulation, mixed precision
                - Features: Multi-GPU management, gradient accumulation, AMP training, profiling
                
                #### 📊 Training Visualization
                - **Real-Time Training Demo**: Live training visualization and monitoring
                - Features: Real-time plots, training metrics, interactive controls
                
                #### 🔬 Experiment Tracking
                - **Experiment Tracking**: TensorBoard and Weights & Biases integration
                - Features: Training monitoring, metric logging, visualization, checkpointing
                - **NEW: Transformers Support**: Language model tracking, attention analysis, gradient flow monitoring
                - **NEW: Diffusers Support**: Diffusion model generation, attention heatmaps, latent space analysis
                
                ### 🤖 **NEW: Transformers & Language Model Features**
                
                #### Language Model Tracking
                - **Perplexity Monitoring**: Track language model perplexity during training
                - **BLEU Score Tracking**: Monitor translation quality metrics
                - **Token Accuracy**: Track token-level prediction accuracy
                - **Attention Analysis**: Visualize attention weights and patterns
                - **Gradient Flow Analysis**: Monitor gradient flow through transformer layers
                
                #### Supported Model Types
                - **BERT**: Bidirectional Encoder Representations from Transformers
                - **GPT-2**: Generative Pre-trained Transformer 2
                - **T5**: Text-to-Text Transfer Transformer
                - **RoBERTa**: Robustly Optimized BERT Pretraining Approach
                - **DistilBERT**: Distilled version of BERT
                - **Custom Transformers**: Support for custom architectures
                
                #### Advanced Features
                - **Attention Heatmaps**: Visualize attention patterns in real-time
                - **Layer Norm Statistics**: Monitor layer normalization behavior
                - **Sequence Length Analysis**: Track performance across different sequence lengths
                - **Vocabulary Analysis**: Monitor token distribution and usage
                
                ### 🎨 **NEW: Diffusion Models & Image Generation Features**
                
                #### Diffusion Model Generation
                - **Noise Level Monitoring**: Track noise reduction during generation
                - **Denoising Steps Analysis**: Monitor step-by-step generation progress
                - **Guidance Scale Tracking**: Monitor classifier-free guidance effectiveness
                - **Image Quality Scoring**: Track generation quality metrics
                - **Generation Time Analysis**: Monitor performance and optimization
                
                #### Supported Model Types
                - **Stable Diffusion**: State-of-the-art text-to-image generation
                - **DDIM**: Denoising Diffusion Implicit Models
                - **DDPM**: Denoising Diffusion Probabilistic Models
                - **Latent Diffusion**: Efficient latent space diffusion
                - **Custom Diffusion**: Support for custom architectures
                
                #### Advanced Features
                - **Cross-Attention Heatmaps**: Visualize text-image attention patterns
                - **Latent Space Analysis**: Monitor latent representation statistics
                - **Memory Usage Tracking**: Monitor GPU memory consumption
                - **Scheduler Step Analysis**: Track diffusion scheduler behavior
                - **Noise Prediction Loss**: Monitor model training quality
                
                ### 💡 Tips
                - Use the Overview tab to check demo status before launching
                - Each demo runs on a different port to avoid conflicts
                - Use Quick Actions for one-click demo launches
                - Check System Info to monitor resource usage
                - Demos can be stopped and restarted as needed
                - **For language models**: Start with smaller models and gradually scale up
                
                ### 🔍 Troubleshooting
                - If a demo fails to launch, check the output for error messages
                - Ensure the required demo files exist in the current directory
                - Check if ports are available (demos will automatically find free ports)
                - Use the refresh button to update demo status
                - **For Transformers**: Ensure you have sufficient GPU memory for large models
                """)
        
        # Footer
        gr.Markdown("""
        ---
        **🚀 Gradio Demo Launcher** | Facebook Posts Feature
        
        Manage all your deep learning demos from one central interface.
        **Enhanced with Transformers support for modern language model research.**
        """)
    
    return interface

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to launch the demo launcher."""
    try:
        # Create the interface
        interface = create_demo_launcher_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch demo launcher: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
