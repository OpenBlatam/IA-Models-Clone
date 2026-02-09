#!/usr/bin/env python3
"""
User-Friendly Interface Launcher for Video-OpusClip

Specialized launcher with interactive tutorials, onboarding,
and guided workflows for optimal user experience.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_interface_environment():
    """Setup environment for user-friendly interfaces."""
    
    # Set interface-specific environment variables
    os.environ.setdefault("USER_FRIENDLY_MODE", "true")
    os.environ.setdefault("ENABLE_TUTORIALS", "true")
    os.environ.setdefault("ENABLE_ONBOARDING", "true")
    os.environ.setdefault("ENABLE_GUIDED_WORKFLOWS", "true")
    os.environ.setdefault("ENABLE_ACCESSIBILITY", "true")
    os.environ.setdefault("ENABLE_MOBILE_OPTIMIZATION", "true")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_interface_dependencies():
    """Check if interface dependencies are installed."""
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
        print(f"❌ Missing interface dependencies: {', '.join(missing_packages)}")
        print("\nTo install interface dependencies, run:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements_optimized.txt")
        return False
    
    return True

def print_interface_banner():
    """Print interface banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              🎬 Video-OpusClip User-Friendly Interface      ║
    ║                                                              ║
    ║              Modern, Intuitive, and Accessible Design        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def show_interface_features():
    """Show interface features and capabilities."""
    features = {
        "quick_start": {
            "name": "🚀 Quick Start",
            "description": "Guided workflows for beginners",
            "features": ["Step-by-step tutorials", "Pre-configured templates", "Smart defaults"]
        },
        "ai_generation": {
            "name": "🤖 AI Generation Studio",
            "description": "Advanced AI video generation",
            "features": ["Multiple AI models", "Advanced parameters", "Real-time preview"]
        },
        "video_processing": {
            "name": "🎥 Video Processing Studio",
            "description": "Professional video editing tools",
            "features": ["AI enhancement", "Stabilization", "Format conversion"]
        },
        "viral_analysis": {
            "name": "📈 Viral Analysis Studio",
            "description": "Content optimization and analysis",
            "features": ["Multi-platform analysis", "Engagement prediction", "Optimization tips"]
        },
        "performance": {
            "name": "⚡ Performance Dashboard",
            "description": "Real-time system monitoring",
            "features": ["Resource tracking", "Performance alerts", "Optimization recommendations"]
        },
        "projects": {
            "name": "📁 Project Management",
            "description": "Organize and manage your work",
            "features": ["Project organization", "Asset management", "Export options"]
        },
        "settings": {
            "name": "⚙️ Settings & Preferences",
            "description": "Customize your experience",
            "features": ["Performance tuning", "Interface customization", "User preferences"]
        }
    }
    
    print("\n🎯 Interface Features:")
    print("-" * 50)
    
    for feature_id, feature_info in features.items():
        print(f"\n📌 {feature_info['name']}")
        print(f"   {feature_info['description']}")
        print("   Features:")
        for feature in feature_info['features']:
            print(f"   • {feature}")
    
    return features

def create_user_onboarding():
    """Create user onboarding experience."""
    
    onboarding_steps = [
        {
            "step": 1,
            "title": "Welcome to Video-OpusClip!",
            "description": "Let's get you started with creating amazing videos.",
            "action": "Click 'Next' to continue"
        },
        {
            "step": 2,
            "title": "Choose Your Workflow",
            "description": "Select the type of content you want to create.",
            "options": ["Text-to-Video", "Image-to-Video", "Video Processing", "Viral Analysis"]
        },
        {
            "step": 3,
            "title": "Configure Your Settings",
            "description": "Set up your preferences for optimal performance.",
            "settings": ["Quality Level", "Processing Speed", "Output Format"]
        },
        {
            "step": 4,
            "title": "Ready to Create!",
            "description": "You're all set! Start creating your first video.",
            "action": "Begin creating"
        }
    ]
    
    return onboarding_steps

def show_interactive_tutorials():
    """Show available interactive tutorials."""
    
    tutorials = {
        "getting_started": {
            "name": "🎯 Getting Started",
            "duration": "5 minutes",
            "description": "Learn the basics of Video-OpusClip",
            "topics": ["Interface navigation", "Basic video generation", "Saving your work"]
        },
        "ai_generation": {
            "name": "🤖 AI Video Generation",
            "duration": "10 minutes",
            "description": "Master AI-powered video creation",
            "topics": ["Prompt engineering", "Model selection", "Parameter tuning"]
        },
        "viral_optimization": {
            "name": "📈 Viral Content Optimization",
            "duration": "8 minutes",
            "description": "Create content that goes viral",
            "topics": ["Platform analysis", "Audience targeting", "Engagement optimization"]
        },
        "advanced_features": {
            "name": "⚡ Advanced Features",
            "duration": "15 minutes",
            "description": "Explore advanced capabilities",
            "topics": ["Batch processing", "Custom workflows", "Performance optimization"]
        }
    }
    
    print("\n📚 Interactive Tutorials:")
    print("-" * 50)
    
    for tutorial_id, tutorial_info in tutorials.items():
        print(f"\n📖 {tutorial_info['name']} ({tutorial_info['duration']})")
        print(f"   {tutorial_info['description']}")
        print("   Topics:")
        for topic in tutorial_info['topics']:
            print(f"   • {topic}")
    
    return tutorials

def launch_interface_with_tutorials(interface_type: str, config: dict):
    """Launch interface with interactive tutorials."""
    
    try:
        if interface_type == "user_friendly":
            from user_friendly_interfaces import launch_user_friendly_interface
            launch_user_friendly_interface(
                server_name=config["host"],
                server_port=config["port"],
                share=config["share"],
                debug=config["debug"]
            )
        elif interface_type == "demo":
            from demo_launcher import launch_specific_demo
            launch_specific_demo("all", config)
        elif interface_type == "simple":
            from gradio_demo import demo
            demo.launch(
                server_name=config["host"],
                server_port=config["port"],
                share=config["share"],
                debug=config["debug"]
            )
        else:
            print(f"❌ Unknown interface type: {interface_type}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching interface {interface_type}: {e}")
        if config["debug"]:
            import traceback
            traceback.print_exc()
        return False

def run_interface_benchmark():
    """Run interface performance benchmark."""
    
    print("\n⚡ Running Interface Performance Benchmark...")
    
    try:
        import time
        import numpy as np
        
        # Test interface responsiveness
        start_time = time.time()
        
        # Test UI rendering
        import gradio as gr
        test_interface = gr.Interface(
            fn=lambda x: x,
            inputs=gr.Textbox(),
            outputs=gr.Textbox()
        )
        
        ui_time = time.time() - start_time
        
        # Test visualization rendering
        start_time = time.time()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(np.random.rand(100))
        plt.close()
        
        viz_time = time.time() - start_time
        
        # Test data processing
        start_time = time.time()
        test_data = np.random.rand(1000, 1000)
        result = np.linalg.eig(test_data)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Interface benchmark completed:")
        print(f"   UI rendering: {ui_time:.3f}s")
        print(f"   Visualization: {viz_time:.3f}s")
        print(f"   Data processing: {processing_time:.3f}s")
        
        total_time = ui_time + viz_time + processing_time
        print(f"   Total benchmark time: {total_time:.3f}s")
        
        if total_time < 3.0:
            print("🚀 Performance: Excellent")
        elif total_time < 6.0:
            print("⚡ Performance: Good")
        else:
            print("🐌 Performance: Slow - consider optimizing")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def show_interface_help():
    """Show interface help information."""
    help_text = """
    🎬 Video-OpusClip User-Friendly Interface - Help Guide
    
    Interface Types:
    • user_friendly: Modern, intuitive interface with tutorials
    • demo: Interactive demo suite
    • simple: Basic interface for quick testing
    
    Features:
    • 🚀 Quick Start: Guided workflows for beginners
    • 🤖 AI Generation: Advanced video creation tools
    • 🎥 Video Processing: Professional editing capabilities
    • 📈 Viral Analysis: Content optimization tools
    • ⚡ Performance: Real-time monitoring dashboard
    • 📁 Projects: Work organization and management
    • ⚙️ Settings: Customization and preferences
    
    Usage Examples:
    • python interface_launcher.py --interface user_friendly
    • python interface_launcher.py --interface demo --share --debug
    • python interface_launcher.py --features
    • python interface_launcher.py --tutorials
    • python interface_launcher.py --benchmark
    
    Tips:
    • Use --tutorials to see available learning resources
    • Use --features to explore interface capabilities
    • Use --benchmark to test performance
    • Use --share to create public links
    • Use --debug for detailed error information
    
    For more information, see the interface documentation.
    """
    print(help_text)

def main():
    """Main interface launcher function."""
    
    print_interface_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Video-OpusClip User-Friendly Interface")
    parser.add_argument("--interface", default="user_friendly", help="Interface type to launch")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--features", action="store_true", help="Show interface features")
    parser.add_argument("--tutorials", action="store_true", help="Show available tutorials")
    parser.add_argument("--onboarding", action="store_true", help="Show onboarding steps")
    parser.add_argument("--benchmark", action="store_true", help="Run interface benchmark")
    parser.add_argument("--help-interface", action="store_true", help="Show interface help")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help_interface:
        show_interface_help()
        return
    
    # Show features if requested
    if args.features:
        show_interface_features()
        return
    
    # Show tutorials if requested
    if args.tutorials:
        show_interactive_tutorials()
        return
    
    # Show onboarding if requested
    if args.onboarding:
        onboarding_steps = create_user_onboarding()
        print("\n🎓 User Onboarding:")
        print("-" * 50)
        for step in onboarding_steps:
            print(f"\n📝 Step {step['step']}: {step['title']}")
            print(f"   {step['description']}")
            if 'action' in step:
                print(f"   Action: {step['action']}")
            if 'options' in step:
                print("   Options:")
                for option in step['options']:
                    print(f"   • {option}")
            if 'settings' in step:
                print("   Settings:")
                for setting in step['settings']:
                    print(f"   • {setting}")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        run_interface_benchmark()
        return
    
    # Setup environment
    setup_interface_environment()
    
    # Check dependencies
    if not check_interface_dependencies():
        sys.exit(1)
    
    # Validate interface type
    valid_interfaces = ["user_friendly", "demo", "simple"]
    if args.interface not in valid_interfaces:
        print(f"❌ Unknown interface type: {args.interface}")
        print(f"Valid options: {', '.join(valid_interfaces)}")
        sys.exit(1)
    
    # Prepare configuration
    config = {
        "host": args.host,
        "port": args.port,
        "share": args.share,
        "debug": args.debug
    }
    
    print(f"\n🚀 Launching {args.interface} interface...")
    print(f"📍 Host: {config['host']}")
    print(f"🔌 Port: {config['port']}")
    print(f"🌐 Public Link: {'Yes' if config['share'] else 'No'}")
    print(f"🐛 Debug Mode: {'Yes' if config['debug'] else 'No'}")
    
    # Show quick tips
    print(f"\n💡 Quick Tips:")
    print(f"   • Use --features to explore capabilities")
    print(f"   • Use --tutorials to see learning resources")
    print(f"   • Use --help-interface for detailed help")
    
    # Launch interface
    try:
        success = launch_interface_with_tutorials(args.interface, config)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interface interrupted by user")
    except Exception as e:
        print(f"❌ Interface failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 