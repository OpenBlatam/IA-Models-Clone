#!/usr/bin/env python3
"""
Demo Launcher for Video-OpusClip Interactive Demos

Specialized launcher for running interactive demos with configuration options,
dependency checking, and helpful features.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_environment():
    """Setup environment for demos."""
    
    # Set demo-specific environment variables
    os.environ.setdefault("DEMO_MODE", "true")
    os.environ.setdefault("ENABLE_VISUALIZATIONS", "true")
    os.environ.setdefault("ENABLE_REAL_TIME_MONITORING", "true")
    os.environ.setdefault("DEMO_CACHE_SIZE", "1000")
    os.environ.setdefault("DEMO_TIMEOUT", "300")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_demo_dependencies():
    """Check if demo dependencies are installed."""
    required_packages = [
        "gradio",
        "torch",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "pandas",
        "opencv-python",
        "pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing demo dependencies: {', '.join(missing_packages)}")
        print("\nTo install demo dependencies, run:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements_optimized.txt")
        return False
    
    return True

def print_demo_banner():
    """Print demo banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                🎬 Video-OpusClip Interactive Demos          ║
    ║                                                              ║
    ║              Advanced AI Model Inference & Visualization     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def list_available_demos():
    """List all available demos."""
    demos = {
        "text-to-video": {
            "name": "Text-to-Video Generation",
            "description": "Generate videos from text prompts with AI models",
            "features": ["Multiple model presets", "Quality controls", "Real-time metrics"]
        },
        "image-to-video": {
            "name": "Image-to-Video Generation",
            "description": "Transform static images into dynamic videos",
            "features": ["Motion effects", "Style transfer", "Quality enhancement"]
        },
        "viral-analysis": {
            "name": "Viral Analysis & Prediction",
            "description": "Analyze content viral potential across platforms",
            "features": ["Multi-platform analysis", "Engagement prediction", "Optimization tips"]
        },
        "performance": {
            "name": "Performance Monitoring",
            "description": "Real-time system performance visualization",
            "features": ["CPU/Memory/GPU monitoring", "Real-time charts", "System analytics"]
        },
        "training": {
            "name": "Training Progress & Metrics",
            "description": "Interactive training simulation and visualization",
            "features": ["Loss curves", "Accuracy tracking", "Learning rate schedules"]
        },
        "all": {
            "name": "All Demos",
            "description": "Complete demo suite with all features",
            "features": ["All demos in tabs", "Unified interface", "Cross-demo integration"]
        }
    }
    
    print("\n🎯 Available Demos:")
    print("-" * 50)
    
    for demo_id, demo_info in demos.items():
        print(f"\n📌 {demo_info['name']} ({demo_id})")
        print(f"   {demo_info['description']}")
        print("   Features:")
        for feature in demo_info['features']:
            print(f"   • {feature}")
    
    return demos

def launch_specific_demo(demo_name: str, config: dict):
    """Launch a specific demo."""
    
    try:
        if demo_name == "text-to-video":
            from gradio_demos import create_text_to_video_demo
            demo = create_text_to_video_demo()
            
        elif demo_name == "image-to-video":
            from gradio_demos import create_image_to_video_demo
            demo = create_image_to_video_demo()
            
        elif demo_name == "viral-analysis":
            from gradio_demos import create_viral_analysis_demo
            demo = create_viral_analysis_demo()
            
        elif demo_name == "performance":
            from gradio_demos import create_performance_monitoring_demo
            demo = create_performance_monitoring_demo()
            
        elif demo_name == "training":
            from gradio_demos import create_training_demo
            demo = create_training_demo()
            
        elif demo_name == "all":
            from gradio_demos import create_main_demo
            demo = create_main_demo()
            
        else:
            print(f"❌ Unknown demo: {demo_name}")
            return False
        
        # Launch demo
        demo.launch(
            server_name=config["host"],
            server_port=config["port"],
            share=config["share"],
            debug=config["debug"],
            show_error=True,
            quiet=False
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching demo {demo_name}: {e}")
        if config["debug"]:
            import traceback
            traceback.print_exc()
        return False

def run_demo_benchmark():
    """Run a quick benchmark to test demo performance."""
    
    print("\n⚡ Running Demo Performance Benchmark...")
    
    try:
        import time
        import numpy as np
        
        # Test basic operations
        start_time = time.time()
        
        # Test numpy operations
        test_array = np.random.rand(1000, 1000)
        result = np.linalg.eig(test_array)
        
        numpy_time = time.time() - start_time
        
        # Test matplotlib
        start_time = time.time()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(np.random.rand(100))
        plt.close()
        
        matplotlib_time = time.time() - start_time
        
        # Test plotly
        start_time = time.time()
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Scatter(y=np.random.rand(100)))
        fig.to_dict()
        
        plotly_time = time.time() - start_time
        
        print(f"✅ Benchmark completed:")
        print(f"   NumPy operations: {numpy_time:.3f}s")
        print(f"   Matplotlib plotting: {matplotlib_time:.3f}s")
        print(f"   Plotly visualization: {plotly_time:.3f}s")
        
        total_time = numpy_time + matplotlib_time + plotly_time
        print(f"   Total benchmark time: {total_time:.3f}s")
        
        if total_time < 2.0:
            print("🚀 Performance: Excellent")
        elif total_time < 5.0:
            print("⚡ Performance: Good")
        else:
            print("🐌 Performance: Slow - consider optimizing")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def show_demo_help():
    """Show demo help information."""
    help_text = """
    🎬 Video-OpusClip Interactive Demos - Help Guide
    
    Demo Types:
    • text-to-video: Generate videos from text prompts
    • image-to-video: Transform images into videos
    • viral-analysis: Analyze content viral potential
    • performance: Monitor system performance
    • training: Training simulation and metrics
    • all: Complete demo suite
    
    Usage Examples:
    • python demo_launcher.py --demo text-to-video
    • python demo_launcher.py --demo all --share --debug
    • python demo_launcher.py --list
    • python demo_launcher.py --benchmark
    
    Tips:
    • Use --share to create public links
    • Use --debug for detailed error information
    • Use --benchmark to test performance
    • Use --list to see all available demos
    
    For more information, see the demo documentation.
    """
    print(help_text)

def main():
    """Main demo launcher function."""
    
    print_demo_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Video-OpusClip Interactive Demos")
    parser.add_argument("--demo", default="all", help="Demo to launch")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--list", action="store_true", help="List available demos")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--help-demo", action="store_true", help="Show demo help")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help_demo:
        show_demo_help()
        return
    
    # List demos if requested
    if args.list:
        list_available_demos()
        return
    
    # Run benchmark if requested
    if args.benchmark:
        run_demo_benchmark()
        return
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_demo_dependencies():
        sys.exit(1)
    
    # Validate demo name
    available_demos = list_available_demos()
    if args.demo not in available_demos:
        print(f"❌ Unknown demo: {args.demo}")
        print("Use --list to see available demos")
        sys.exit(1)
    
    # Prepare configuration
    config = {
        "host": args.host,
        "port": args.port,
        "share": args.share,
        "debug": args.debug
    }
    
    print(f"\n🚀 Launching {available_demos[args.demo]['name']}...")
    print(f"📍 Host: {config['host']}")
    print(f"🔌 Port: {config['port']}")
    print(f"🌐 Public Link: {'Yes' if config['share'] else 'No'}")
    print(f"🐛 Debug Mode: {'Yes' if config['debug'] else 'No'}")
    
    # Launch demo
    try:
        success = launch_specific_demo(args.demo, config)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 