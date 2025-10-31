#!/usr/bin/env python3
"""
Quick Start Gradio for Video-OpusClip

This script demonstrates how to quickly get started with Gradio
in the Video-OpusClip system for creating interactive web interfaces.
"""

import sys
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gradio_installation():
    """Check if Gradio is properly installed."""
    
    print("🔍 Checking Gradio Installation")
    print("=" * 50)
    
    try:
        import gradio as gr
        print(f"✅ Gradio version: {gr.__version__}")
        
        # Test basic imports
        from gradio import Interface, Blocks, Textbox, Button, Image
        print("✅ Core components imported successfully")
        
        from gradio import themes
        print("✅ Themes imported successfully")
        
        from gradio_client import Client
        print("✅ Gradio client imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Gradio import error: {e}")
        print("💡 Install with: pip install gradio[all]")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def quick_start_basic_interface():
    """Basic Gradio interface creation."""
    
    print("\n🎨 Quick Start: Basic Interface")
    print("=" * 50)
    
    try:
        import gradio as gr
        
        # Simple function
        def greet(name):
            return f"Hello, {name}! Welcome to Video-OpusClip!"
        
        # Create basic interface
        demo = gr.Interface(
            fn=greet,
            inputs=gr.Textbox(label="Your Name", placeholder="Enter your name..."),
            outputs=gr.Textbox(label="Greeting"),
            title="Video-OpusClip Basic Demo",
            description="A simple Gradio interface demonstration"
        )
        
        print("✅ Basic interface created successfully")
        print("🚀 Launching interface...")
        
        # Launch interface
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Basic interface error: {e}")
        return False

def quick_start_advanced_interface():
    """Advanced Gradio interface with multiple components."""
    
    print("\n🎨 Quick Start: Advanced Interface")
    print("=" * 50)
    
    try:
        import gradio as gr
        import numpy as np
        
        # Simulate AI processing functions
        def process_text(text_input):
            """Process text input."""
            return f"Processed: {text_input.upper()}"
        
        def generate_image(prompt, size):
            """Generate a simple image (simulated)."""
            # Create a simple colored image
            img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            return img_array
        
        def analyze_viral_potential(content, platform):
            """Analyze viral potential (simulated)."""
            import random
            score = random.uniform(0.1, 0.9)
            recommendations = [
                "Use trending hashtags",
                "Post at peak engagement times",
                "Include call-to-action elements"
            ]
            return {
                "viral_score": score,
                "recommendations": recommendations,
                "platform": platform
            }
        
        # Create advanced interface
        with gr.Blocks(
            title="Video-OpusClip Advanced Demo",
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
            """
        ) as demo:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🎬 Video-OpusClip Advanced Demo</h1>
                <p>Interactive AI video processing and generation platform</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Text Processing Tab
                with gr.TabItem("📝 Text Processing"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Process Text")
                            
                            text_input = gr.Textbox(
                                label="Input Text",
                                placeholder="Enter text to process...",
                                lines=3
                            )
                            
                            process_btn = gr.Button(
                                "🔄 Process Text",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            
                            text_output = gr.Textbox(
                                label="Processed Text",
                                lines=3,
                                interactive=False
                            )
                    
                    # Connect function
                    process_btn.click(
                        fn=process_text,
                        inputs=text_input,
                        outputs=text_output
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
                    
                    # Connect function
                    generate_btn.click(
                        fn=generate_image,
                        inputs=[prompt_input, size_slider],
                        outputs=image_output
                    )
                
                # Viral Analysis Tab
                with gr.TabItem("📈 Viral Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Analyze Content")
                            
                            content_input = gr.Textbox(
                                label="Content Description",
                                placeholder="Describe your content...",
                                lines=3
                            )
                            
                            platform_select = gr.Dropdown(
                                choices=["TikTok", "YouTube", "Instagram", "Twitter"],
                                value="TikTok",
                                label="Target Platform"
                            )
                            
                            analyze_btn = gr.Button(
                                "🔍 Analyze Viral Potential",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Analysis Results")
                            
                            viral_score = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.5,
                                interactive=False,
                                label="Viral Score"
                            )
                            
                            recommendations = gr.JSON(
                                label="Recommendations"
                            )
                    
                    # Connect function
                    analyze_btn.click(
                        fn=analyze_viral_potential,
                        inputs=[content_input, platform_select],
                        outputs=[viral_score, recommendations]
                    )
        
        print("✅ Advanced interface created successfully")
        print("🚀 Launching advanced interface...")
        
        # Launch interface
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=False
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced interface error: {e}")
        return False

def quick_start_integration_demo():
    """Demonstrate integration with Video-OpusClip components."""
    
    print("\n🔗 Quick Start: Video-OpusClip Integration")
    print("=" * 50)
    
    try:
        import gradio as gr
        
        # Import Video-OpusClip components
        try:
            from optimized_config import get_config
            print("✅ Optimized config imported")
        except ImportError:
            print("⚠️ Optimized config not available")
            get_config = lambda: {}
        
        try:
            from performance_monitor import PerformanceMonitor
            print("✅ Performance monitor imported")
        except ImportError:
            print("⚠️ Performance monitor not available")
            PerformanceMonitor = None
        
        try:
            from enhanced_error_handling import safe_load_ai_model
            print("✅ Enhanced error handling imported")
        except ImportError:
            print("⚠️ Enhanced error handling not available")
            safe_load_ai_model = None
        
        # Create integrated interface
        class VideoOpusClipGradioDemo:
            """Demo integration with Video-OpusClip."""
            
            def __init__(self):
                self.config = get_config()
                self.performance_monitor = PerformanceMonitor() if PerformanceMonitor else None
                self.setup_components()
            
            def setup_components(self):
                """Setup demo components."""
                print("✅ Demo components setup complete")
            
            def process_video_demo(self, video_description, duration, quality):
                """Demo video processing."""
                
                try:
                    # Simulate processing
                    processing_time = duration * 0.1
                    time.sleep(processing_time)
                    
                    # Get performance metrics if available
                    metrics = {}
                    if self.performance_monitor:
                        metrics = self.performance_monitor.get_metrics()
                    
                    return {
                        "processed_video": "demo_video.mp4",
                        "processing_time": processing_time,
                        "duration": duration,
                        "quality": quality,
                        "metrics": metrics,
                        "status": "Success"
                    }
                    
                except Exception as e:
                    return {
                        "processed_video": None,
                        "error": str(e),
                        "status": "Error"
                    }
            
            def create_interface(self):
                """Create the integrated interface."""
                
                with gr.Blocks(
                    title="Video-OpusClip Integration Demo",
                    theme=gr.themes.Soft()
                ) as demo:
                    
                    # Header
                    gr.HTML("""
                    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                        <h1>🎬 Video-OpusClip Integration Demo</h1>
                        <p>Demonstrating Gradio integration with Video-OpusClip components</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Video Processing Demo")
                            
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
                            gr.Markdown("### Results")
                            
                            results_json = gr.JSON(
                                label="Processing Results"
                            )
                    
                    # Connect function
                    process_btn.click(
                        fn=self.process_video_demo,
                        inputs=[video_desc, duration_slider, quality_dropdown],
                        outputs=results_json
                    )
                
                return demo
        
        # Create and launch integrated interface
        interface = VideoOpusClipGradioDemo()
        demo = interface.create_interface()
        
        print("✅ Integration demo created successfully")
        print("🚀 Launching integration demo...")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            debug=False
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Integration demo error: {e}")
        return False

def quick_start_performance_demo():
    """Demonstrate performance monitoring and optimization."""
    
    print("\n⚡ Quick Start: Performance Demo")
    print("=" * 50)
    
    try:
        import gradio as gr
        import time
        import psutil
        
        def get_system_metrics():
            """Get system performance metrics."""
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%",
                "memory_available": f"{memory.available / (1024**3):.1f} GB",
                "timestamp": time.strftime("%H:%M:%S")
            }
            
            return metrics
        
        def simulate_processing(processing_time):
            """Simulate processing with progress updates."""
            
            start_time = time.time()
            progress_steps = []
            
            for i in range(10):
                time.sleep(processing_time / 10)
                progress = (i + 1) / 10
                progress_steps.append({
                    "step": i + 1,
                    "progress": f"{progress * 100:.0f}%",
                    "time_elapsed": f"{time.time() - start_time:.1f}s"
                })
            
            return {
                "total_time": time.time() - start_time,
                "progress_steps": progress_steps,
                "status": "Completed"
            }
        
        # Create performance demo interface
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
                        
                        with gr.Column():
                            gr.Markdown("### Current Metrics")
                            
                            metrics_output = gr.JSON(
                                label="System Metrics"
                            )
                    
                    # Connect function
                    refresh_btn.click(
                        fn=get_system_metrics,
                        inputs=[],
                        outputs=metrics_output
                    )
                
                # Processing Demo Tab
                with gr.TabItem("⚙️ Processing Demo"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Simulate Processing")
                            
                            processing_time = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=0.5,
                                label="Processing Time (seconds)"
                            )
                            
                            start_btn = gr.Button(
                                "🚀 Start Processing",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Processing Results")
                            
                            results_output = gr.JSON(
                                label="Processing Results"
                            )
                    
                    # Connect function
                    start_btn.click(
                        fn=simulate_processing,
                        inputs=processing_time,
                        outputs=results_output
                    )
        
        print("✅ Performance demo created successfully")
        print("🚀 Launching performance demo...")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7863,
            share=False,
            debug=False
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Performance demo error: {e}")
        return False

def run_all_quick_starts():
    """Run all Gradio quick start demonstrations."""
    
    print("🚀 Gradio Quick Start for Video-OpusClip")
    print("=" * 60)
    
    results = {}
    
    # Check installation
    results['installation'] = check_gradio_installation()
    
    if results['installation']:
        print("\n🎯 Choose a demo to run:")
        print("1. Basic Interface (Port 7860)")
        print("2. Advanced Interface (Port 7861)")
        print("3. Integration Demo (Port 7862)")
        print("4. Performance Demo (Port 7863)")
        print("5. Run all demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "1":
            results['basic'] = quick_start_basic_interface()
        elif choice == "2":
            results['advanced'] = quick_start_advanced_interface()
        elif choice == "3":
            results['integration'] = quick_start_integration_demo()
        elif choice == "4":
            results['performance'] = quick_start_performance_demo()
        elif choice == "5":
            print("\n⚠️ Note: Running all demos will open multiple browser tabs")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                results['basic'] = quick_start_basic_interface()
                results['advanced'] = quick_start_advanced_interface()
                results['integration'] = quick_start_integration_demo()
                results['performance'] = quick_start_performance_demo()
        elif choice == "0":
            print("👋 Exiting...")
            return
        else:
            print("❌ Invalid choice")
            return
    
    # Summary
    print("\n📊 Quick Start Summary")
    print("=" * 60)
    
    if results.get('installation'):
        print("✅ Installation: Successful")
        
        if results.get('basic'):
            print("✅ Basic Interface: Launched on http://127.0.0.1:7860")
        
        if results.get('advanced'):
            print("✅ Advanced Interface: Launched on http://127.0.0.1:7861")
        
        if results.get('integration'):
            print("✅ Integration Demo: Launched on http://127.0.0.1:7862")
        
        if results.get('performance'):
            print("✅ Performance Demo: Launched on http://127.0.0.1:7863")
        
        print("\n🎉 Gradio quick starts completed successfully!")
        print("📁 Check the browser tabs for the interfaces")
        
    else:
        print("❌ Installation failed - please check your setup")
    
    return results

if __name__ == "__main__":
    # Run all quick starts
    results = run_all_quick_starts()
    
    print("\n🔧 Next Steps:")
    print("1. Explore the launched interfaces in your browser")
    print("2. Read the GRADIO_GUIDE.md for detailed usage")
    print("3. Check gradio_examples.py for more examples")
    print("4. Integrate with your Video-OpusClip workflow") 