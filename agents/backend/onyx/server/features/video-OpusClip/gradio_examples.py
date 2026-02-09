#!/usr/bin/env python3
"""
Gradio Examples for Video-OpusClip

Comprehensive examples demonstrating Gradio library usage
in the Video-OpusClip system for creating interactive web interfaces.
"""

import gradio as gr
import torch
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# BASIC INTERFACE EXAMPLES
# =============================================================================

def example_simple_interface():
    """Example 1: Simple text-to-text interface."""
    
    print("🎨 Example 1: Simple Interface")
    print("=" * 50)
    
    def process_text(text):
        """Simple text processing function."""
        return f"Processed: {text.upper()}"
    
    # Create simple interface
    demo = gr.Interface(
        fn=process_text,
        inputs=gr.Textbox(label="Input Text", placeholder="Enter text..."),
        outputs=gr.Textbox(label="Output Text"),
        title="Simple Text Processor",
        description="A basic Gradio interface example"
    )
    
    return demo

def example_multimodal_interface():
    """Example 2: Multimodal interface with text and image."""
    
    print("\n🎨 Example 2: Multimodal Interface")
    print("=" * 50)
    
    def process_text_and_image(text, image):
        """Process text and image inputs."""
        
        if image is not None:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Simple processing: add text overlay
            processed_img = img_array.copy()
            
            # Return both processed text and image
            return f"Text: {text.upper()}", processed_img
        else:
            return f"Text: {text.upper()}", None
    
    # Create multimodal interface
    demo = gr.Interface(
        fn=process_text_and_image,
        inputs=[
            gr.Textbox(label="Text Input", placeholder="Enter text..."),
            gr.Image(label="Image Input", type="numpy")
        ],
        outputs=[
            gr.Textbox(label="Processed Text"),
            gr.Image(label="Processed Image")
        ],
        title="Multimodal Processor",
        description="Process both text and image inputs"
    )
    
    return demo

# =============================================================================
# ADVANCED INTERFACE EXAMPLES
# =============================================================================

def example_blocks_interface():
    """Example 3: Advanced interface using Blocks."""
    
    print("\n🎨 Example 3: Blocks Interface")
    print("=" * 50)
    
    with gr.Blocks(
        title="Video-OpusClip Blocks Demo",
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
            border-left: 4px solid #28a745;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎬 Video-OpusClip Blocks Demo</h1>
            <p>Advanced interface using Gradio Blocks</p>
        </div>
        """)
        
        # Main content
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Section")
                
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter text to process...",
                    lines=3
                )
                
                image_input = gr.Image(
                    label="Image Input",
                    type="numpy"
                )
                
                process_btn = gr.Button(
                    "🔄 Process",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Output Section")
                
                text_output = gr.Textbox(
                    label="Processed Text",
                    lines=3,
                    interactive=False
                )
                
                image_output = gr.Image(
                    label="Processed Image"
                )
                
                metrics_output = gr.JSON(
                    label="Processing Metrics"
                )
        
        # Processing function
        def process_inputs(text, image):
            """Process text and image inputs."""
            
            start_time = time.time()
            
            # Process text
            processed_text = f"Processed: {text.upper()}" if text else "No text provided"
            
            # Process image
            processed_image = None
            if image is not None:
                processed_image = image.copy()
            
            # Calculate metrics
            processing_time = time.time() - start_time
            metrics = {
                "processing_time": f"{processing_time:.3f}s",
                "text_length": len(text) if text else 0,
                "image_processed": image is not None,
                "timestamp": time.strftime("%H:%M:%S")
            }
            
            return processed_text, processed_image, metrics
        
        # Connect function
        process_btn.click(
            fn=process_inputs,
            inputs=[text_input, image_input],
            outputs=[text_output, image_output, metrics_output]
        )
    
    return demo

def example_tabbed_interface():
    """Example 4: Tabbed interface with multiple features."""
    
    print("\n🎨 Example 4: Tabbed Interface")
    print("=" * 50)
    
    with gr.Blocks(
        title="Video-OpusClip Tabbed Demo",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🎬 Video-OpusClip Tabbed Demo</h1>
            <p>Multiple features organized in tabs</p>
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Text Processing Tab
            with gr.TabItem("📝 Text Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Text Analysis")
                        
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to analyze...",
                            lines=4
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze Text",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Analysis Results")
                        
                        word_count = gr.Number(
                            label="Word Count",
                            interactive=False
                        )
                        
                        char_count = gr.Number(
                            label="Character Count",
                            interactive=False
                        )
                        
                        sentiment = gr.Textbox(
                            label="Sentiment",
                            interactive=False
                        )
                
                def analyze_text(text):
                    """Analyze text input."""
                    if not text:
                        return 0, 0, "No text provided"
                    
                    words = len(text.split())
                    chars = len(text)
                    
                    # Simple sentiment analysis
                    positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
                    negative_words = ["bad", "terrible", "awful", "horrible", "disgusting"]
                    
                    text_lower = text.lower()
                    positive_count = sum(1 for word in positive_words if word in text_lower)
                    negative_count = sum(1 for word in negative_words if word in text_lower)
                    
                    if positive_count > negative_count:
                        sentiment_result = "Positive"
                    elif negative_count > positive_count:
                        sentiment_result = "Negative"
                    else:
                        sentiment_result = "Neutral"
                    
                    return words, chars, sentiment_result
                
                analyze_btn.click(
                    fn=analyze_text,
                    inputs=text_input,
                    outputs=[word_count, char_count, sentiment]
                )
            
            # Image Generation Tab
            with gr.TabItem("🎨 Image Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Generate Image")
                        
                        prompt_input = gr.Textbox(
                            label="Image Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=2
                        )
                        
                        size_slider = gr.Slider(
                            minimum=256,
                            maximum=512,
                            value=256,
                            step=64,
                            label="Image Size"
                        )
                        
                        generate_btn = gr.Button(
                            "🎨 Generate Image",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Image")
                        
                        image_output = gr.Image(
                            label="Generated Image",
                            height=300
                        )
                
                def generate_image(prompt, size):
                    """Generate a simple image (simulated)."""
                    if not prompt:
                        return None
                    
                    # Create a simple colored image based on prompt
                    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    
                    # Add some variation based on prompt length
                    if len(prompt) > 20:
                        img_array = img_array + np.random.randint(-50, 50, img_array.shape)
                        img_array = np.clip(img_array, 0, 255)
                    
                    return img_array
                
                generate_btn.click(
                    fn=generate_image,
                    inputs=[prompt_input, size_slider],
                    outputs=image_output
                )
            
            # Data Visualization Tab
            with gr.TabItem("📊 Data Visualization"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Create Visualization")
                        
                        chart_type = gr.Dropdown(
                            choices=["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"],
                            value="Bar Chart",
                            label="Chart Type"
                        )
                        
                        data_points = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Data Points"
                        )
                        
                        create_chart_btn = gr.Button(
                            "📊 Create Chart",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Chart")
                        
                        chart_output = gr.Plot(
                            label="Data Visualization"
                        )
                
                def create_chart(chart_type, data_points):
                    """Create a data visualization chart."""
                    
                    # Generate sample data
                    x = list(range(1, data_points + 1))
                    y = np.random.randint(10, 100, data_points)
                    
                    if chart_type == "Bar Chart":
                        fig = go.Figure(data=go.Bar(x=x, y=y))
                    elif chart_type == "Line Chart":
                        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
                    elif chart_type == "Scatter Plot":
                        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))
                    elif chart_type == "Pie Chart":
                        fig = go.Figure(data=go.Pie(labels=[f"Item {i}" for i in x], values=y))
                    
                    fig.update_layout(
                        title=f"{chart_type} - {data_points} Data Points",
                        xaxis_title="X Axis",
                        yaxis_title="Y Axis"
                    )
                    
                    return fig
                
                create_chart_btn.click(
                    fn=create_chart,
                    inputs=[chart_type, data_points],
                    outputs=chart_output
                )
    
    return demo

# =============================================================================
# REAL-TIME PROCESSING EXAMPLES
# =============================================================================

def example_realtime_processing():
    """Example 5: Real-time processing with live updates."""
    
    print("\n🎨 Example 5: Real-time Processing")
    print("=" * 50)
    
    with gr.Blocks(
        title="Video-OpusClip Real-time Demo",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>⚡ Video-OpusClip Real-time Demo</h1>
            <p>Live processing with real-time updates</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Live Processing")
                
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Type here for real-time processing...",
                    lines=3
                )
                
                processing_speed = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Processing Speed (seconds)"
                )
                
                start_processing_btn = gr.Button(
                    "🚀 Start Processing",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Live Results")
                
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    interactive=False,
                    label="Processing Progress"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                results_output = gr.JSON(
                    label="Processing Results"
                )
        
        def realtime_processing(text, speed, progress=gr.Progress()):
            """Real-time processing with progress updates."""
            
            if not text:
                return 0, "No text provided", {}
            
            results = []
            steps = 10
            
            for i in range(steps):
                # Update progress
                progress((i + 1) / steps, desc=f"Processing step {i + 1}/{steps}")
                
                # Simulate processing
                time.sleep(speed / steps)
                
                # Generate step result
                step_result = {
                    "step": i + 1,
                    "text_processed": text[:len(text) * (i + 1) // steps],
                    "timestamp": time.strftime("%H:%M:%S"),
                    "progress": f"{((i + 1) / steps) * 100:.0f}%"
                }
                
                results.append(step_result)
            
            # Final result
            final_result = {
                "original_text": text,
                "processed_text": text.upper(),
                "processing_steps": results,
                "total_time": speed,
                "final_status": "Completed"
            }
            
            return 100, "Processing completed successfully!", final_result
        
        start_processing_btn.click(
            fn=realtime_processing,
            inputs=[input_text, processing_speed],
            outputs=[progress_bar, status_output, results_output]
        )
    
    return demo

# =============================================================================
# BATCH PROCESSING EXAMPLES
# =============================================================================

def example_batch_processing():
    """Example 6: Batch processing interface."""
    
    print("\n🎨 Example 6: Batch Processing")
    print("=" * 50)
    
    with gr.Blocks(
        title="Video-OpusClip Batch Processing Demo",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>📦 Video-OpusClip Batch Processing Demo</h1>
            <p>Process multiple items efficiently</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Batch Input")
                
                batch_input = gr.Textbox(
                    label="Batch Items (one per line)",
                    placeholder="Enter items to process, one per line...",
                    lines=5
                )
                
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Batch Size"
                )
                
                process_batch_btn = gr.Button(
                    "📦 Process Batch",
                    variant="primary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Batch Results")
                
                batch_results = gr.JSON(
                    label="Processing Results"
                )
                
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=3,
                    interactive=False
                )
        
        def process_batch(items_text, batch_size):
            """Process items in batches."""
            
            if not items_text:
                return {}, "No items provided"
            
            # Parse items
            items = [item.strip() for item in items_text.split('\n') if item.strip()]
            
            if not items:
                return {}, "No valid items found"
            
            # Process in batches
            results = []
            total_items = len(items)
            
            for i in range(0, total_items, batch_size):
                batch = items[i:i + batch_size]
                batch_results = []
                
                for j, item in enumerate(batch):
                    # Simulate processing
                    time.sleep(0.1)
                    
                    processed_item = {
                        "original": item,
                        "processed": item.upper(),
                        "batch": i // batch_size + 1,
                        "position": j + 1,
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    
                    batch_results.append(processed_item)
                
                results.extend(batch_results)
            
            # Create summary
            summary = f"Processed {total_items} items in {(total_items + batch_size - 1) // batch_size} batches"
            
            return results, summary
        
        process_batch_btn.click(
            fn=process_batch,
            inputs=[batch_input, batch_size],
            outputs=[batch_results, summary_output]
        )
    
    return demo

# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

def example_video_opusclip_integration():
    """Example 7: Integration with Video-OpusClip components."""
    
    print("\n🎨 Example 7: Video-OpusClip Integration")
    print("=" * 50)
    
    # Import Video-OpusClip components
    try:
        from optimized_config import get_config
        config = get_config()
        print("✅ Optimized config imported")
    except ImportError:
        config = {}
        print("⚠️ Optimized config not available")
    
    try:
        from performance_monitor import PerformanceMonitor
        performance_monitor = PerformanceMonitor(config)
        print("✅ Performance monitor imported")
    except ImportError:
        performance_monitor = None
        print("⚠️ Performance monitor not available")
    
    class VideoOpusClipGradioIntegration:
        """Integration class for Video-OpusClip components."""
        
        def __init__(self):
            self.config = config
            self.performance_monitor = performance_monitor
            self.setup_components()
        
        def setup_components(self):
            """Setup integration components."""
            print("✅ Integration components setup complete")
        
        def process_video_demo(self, video_description, duration, quality):
            """Demo video processing with integration."""
            
            start_time = time.time()
            
            try:
                # Simulate video processing
                processing_time = duration * 0.1
                time.sleep(processing_time)
                
                # Get performance metrics if available
                metrics = {}
                if self.performance_monitor:
                    metrics = self.performance_monitor.get_metrics()
                
                # Create result
                result = {
                    "video_description": video_description,
                    "duration": duration,
                    "quality": quality,
                    "processing_time": time.time() - start_time,
                    "status": "Success",
                    "metrics": metrics,
                    "config": {
                        "model_type": self.config.get("model_type", "default"),
                        "device": self.config.get("device", "cpu")
                    }
                }
                
                return result
                
            except Exception as e:
                return {
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "status": "Error"
                }
        
        def create_interface(self):
            """Create the integrated interface."""
            
            with gr.Blocks(
                title="Video-OpusClip Integration Demo",
                theme=gr.themes.Soft()
            ) as demo:
                
                gr.HTML("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                    <h1>🔗 Video-OpusClip Integration Demo</h1>
                    <p>Integration with Video-OpusClip components</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Video Processing")
                        
                        video_desc = gr.Textbox(
                            label="Video Description",
                            placeholder="Describe the video to process...",
                            lines=3
                        )
                        
                        duration_slider = gr.Slider(
                            minimum=5,
                            maximum=60,
                            value=30,
                            step=5,
                            label="Duration (seconds)"
                        )
                        
                        quality_dropdown = gr.Dropdown(
                            choices=["Fast", "Balanced", "Quality"],
                            value="Balanced",
                            label="Quality Preset"
                        )
                        
                        process_btn = gr.Button(
                            "🎬 Process Video",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Processing Results")
                        
                        results_json = gr.JSON(
                            label="Processing Results"
                        )
                        
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                
                # Connect function
                process_btn.click(
                    fn=self.process_video_demo,
                    inputs=[video_desc, duration_slider, quality_dropdown],
                    outputs=results_json
                )
            
            return demo
    
    # Create and return integrated interface
    integration = VideoOpusClipGradioIntegration()
    return integration.create_interface()

# =============================================================================
# PERFORMANCE MONITORING EXAMPLES
# =============================================================================

def example_performance_monitoring():
    """Example 8: Performance monitoring interface."""
    
    print("\n🎨 Example 8: Performance Monitoring")
    print("=" * 50)
    
    with gr.Blocks(
        title="Video-OpusClip Performance Demo",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>⚡ Video-OpusClip Performance Demo</h1>
            <p>System monitoring and performance optimization</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # System Metrics Tab
            with gr.TabItem("📊 System Metrics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Real-time System Monitoring")
                        
                        refresh_btn = gr.Button(
                            "🔄 Refresh Metrics",
                            variant="primary"
                        )
                        
                        auto_refresh = gr.Checkbox(
                            label="Auto-refresh every 5 seconds",
                            value=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Current Metrics")
                        
                        metrics_output = gr.JSON(
                            label="System Metrics"
                        )
                        
                        cpu_gauge = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            interactive=False,
                            label="CPU Usage (%)"
                        )
                        
                        memory_gauge = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            interactive=False,
                            label="Memory Usage (%)"
                        )
                
                def get_system_metrics():
                    """Get system performance metrics."""
                    import psutil
                    
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    metrics = {
                        "cpu_usage": f"{cpu_percent:.1f}%",
                        "memory_usage": f"{memory.percent:.1f}%",
                        "memory_available": f"{memory.available / (1024**3):.1f} GB",
                        "memory_total": f"{memory.total / (1024**3):.1f} GB",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "system_info": {
                            "platform": psutil.sys.platform,
                            "python_version": psutil.sys.version,
                            "cpu_count": psutil.cpu_count()
                        }
                    }
                    
                    return metrics, cpu_percent, memory.percent
                
                refresh_btn.click(
                    fn=get_system_metrics,
                    inputs=[],
                    outputs=[metrics_output, cpu_gauge, memory_gauge]
                )
            
            # Processing Performance Tab
            with gr.TabItem("⚙️ Processing Performance"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Performance Test")
                        
                        test_duration = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=0.5,
                            label="Test Duration (seconds)"
                        )
                        
                        test_type = gr.Dropdown(
                            choices=["CPU Intensive", "Memory Intensive", "Mixed"],
                            value="Mixed",
                            label="Test Type"
                        )
                        
                        run_test_btn = gr.Button(
                            "🚀 Run Performance Test",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Test Results")
                        
                        test_results = gr.JSON(
                            label="Performance Test Results"
                        )
                        
                        performance_chart = gr.Plot(
                            label="Performance Over Time"
                        )
                
                def run_performance_test(duration, test_type):
                    """Run a performance test."""
                    
                    start_time = time.time()
                    results = []
                    
                    # Simulate different types of tests
                    if test_type == "CPU Intensive":
                        for i in range(int(duration * 10)):
                            # CPU intensive operation
                            _ = sum(range(10000))
                            results.append({
                                "time": i * 0.1,
                                "cpu_usage": 80 + np.random.randint(-10, 10),
                                "memory_usage": 20 + np.random.randint(-5, 5)
                            })
                            time.sleep(0.1)
                    
                    elif test_type == "Memory Intensive":
                        for i in range(int(duration * 10)):
                            # Memory intensive operation
                            _ = np.random.rand(1000, 1000)
                            results.append({
                                "time": i * 0.1,
                                "cpu_usage": 30 + np.random.randint(-5, 5),
                                "memory_usage": 70 + np.random.randint(-10, 10)
                            })
                            time.sleep(0.1)
                    
                    else:  # Mixed
                        for i in range(int(duration * 10)):
                            _ = sum(range(5000))
                            _ = np.random.rand(500, 500)
                            results.append({
                                "time": i * 0.1,
                                "cpu_usage": 50 + np.random.randint(-15, 15),
                                "memory_usage": 40 + np.random.randint(-10, 10)
                            })
                            time.sleep(0.1)
                    
                    # Create performance chart
                    times = [r["time"] for r in results]
                    cpu_usage = [r["cpu_usage"] for r in results]
                    memory_usage = [r["memory_usage"] for r in results]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=times, y=cpu_usage, mode='lines', name='CPU Usage'))
                    fig.add_trace(go.Scatter(x=times, y=memory_usage, mode='lines', name='Memory Usage'))
                    
                    fig.update_layout(
                        title=f"Performance Test: {test_type}",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Usage (%)",
                        height=400
                    )
                    
                    test_summary = {
                        "test_type": test_type,
                        "duration": duration,
                        "total_time": time.time() - start_time,
                        "avg_cpu": np.mean(cpu_usage),
                        "avg_memory": np.mean(memory_usage),
                        "max_cpu": np.max(cpu_usage),
                        "max_memory": np.max(memory_usage)
                    }
                    
                    return test_summary, fig
                
                run_test_btn.click(
                    fn=run_performance_test,
                    inputs=[test_duration, test_type],
                    outputs=[test_results, performance_chart]
                )
    
    return demo

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all Gradio examples."""
    
    print("🚀 Gradio Examples for Video-OpusClip")
    print("=" * 60)
    
    examples = {
        "1": ("Simple Interface", example_simple_interface),
        "2": ("Multimodal Interface", example_multimodal_interface),
        "3": ("Blocks Interface", example_blocks_interface),
        "4": ("Tabbed Interface", example_tabbed_interface),
        "5": ("Real-time Processing", example_realtime_processing),
        "6": ("Batch Processing", example_batch_processing),
        "7": ("Video-OpusClip Integration", example_video_opusclip_integration),
        "8": ("Performance Monitoring", example_performance_monitoring)
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\n0. Exit")
    
    while True:
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "0":
            print("👋 Exiting...")
            break
        
        if choice in examples:
            name, func = examples[choice]
            print(f"\n🎨 Running: {name}")
            print("=" * 50)
            
            try:
                demo = func()
                print(f"✅ {name} created successfully")
                print("🚀 Launching interface...")
                
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=7860 + int(choice),
                    share=False,
                    debug=False
                )
                
            except Exception as e:
                print(f"❌ Error running {name}: {e}")
        
        else:
            print("❌ Invalid choice. Please enter a number between 0-8.")

if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    print("\n🔧 Next Steps:")
    print("1. Explore the launched interfaces in your browser")
    print("2. Read the GRADIO_GUIDE.md for detailed usage")
    print("3. Run quick_start_gradio_guide.py for basic setup")
    print("4. Integrate with your Video-OpusClip workflow") 