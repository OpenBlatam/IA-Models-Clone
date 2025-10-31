#!/usr/bin/env python3
"""
Quick Start Script for Video-OpusClip Gradio Interface

This script provides an easy way to launch the Gradio interface
with interactive configuration and helpful prompts.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎬 Video-OpusClip AI Studio              ║
    ║                                                              ║
    ║              Advanced AI-powered video processing            ║
    ║              and viral content generation                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "gradio",
        "torch",
        "numpy",
        "pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements_optimized.txt")
        return False
    
    return True

def get_user_preferences():
    """Get user preferences interactively."""
    print("\n🔧 Configuration Options:")
    print("-" * 30)
    
    # Host configuration
    host = input("Host (default: 127.0.0.1): ").strip() or "127.0.0.1"
    
    # Port configuration
    while True:
        try:
            port = input("Port (default: 7860): ").strip() or "7860"
            port = int(port)
            if 1024 <= port <= 65535:
                break
            else:
                print("❌ Port must be between 1024 and 65535")
        except ValueError:
            print("❌ Invalid port number")
    
    # Share option
    share = input("Create public link? (y/N): ").strip().lower() == 'y'
    
    # Debug mode
    debug = input("Enable debug mode? (y/N): ").strip().lower() == 'y'
    
    # Interface type
    print("\n🎯 Interface Options:")
    print("1. Enhanced Interface (Full features)")
    print("2. Simple Interface (Basic features)")
    
    while True:
        choice = input("Choose interface (1/2, default: 1): ").strip() or "1"
        if choice in ["1", "2"]:
            break
        print("❌ Please choose 1 or 2")
    
    simple = choice == "2"
    
    return {
        "host": host,
        "port": port,
        "share": share,
        "debug": debug,
        "simple": simple
    }

def setup_environment():
    """Setup environment variables for optimal performance."""
    # Set default environment variables
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("ENABLE_STRUCTURED_LOGGING", "true")
    os.environ.setdefault("USE_GPU", "true")
    os.environ.setdefault("ENABLE_CACHING", "true")
    os.environ.setdefault("MAX_WORKERS", str(os.cpu_count() or 4))
    os.environ.setdefault("BATCH_SIZE", "16")
    
    print("✅ Environment configured")

def launch_interface(config):
    """Launch the Gradio interface."""
    print(f"\n🚀 Launching Video-OpusClip AI Studio...")
    print(f"📍 Host: {config['host']}")
    print(f"🔌 Port: {config['port']}")
    print(f"🌐 Public Link: {'Yes' if config['share'] else 'No'}")
    print(f"🐛 Debug Mode: {'Yes' if config['debug'] else 'No'}")
    print(f"🎯 Interface: {'Simple' if config['simple'] else 'Enhanced'}")
    
    try:
        if config['simple']:
            # Launch simple interface
            from gradio_demo import demo
            demo.launch(
                server_name=config['host'],
                server_port=config['port'],
                share=config['share'],
                debug=config['debug']
            )
        else:
            # Launch enhanced interface
            from gradio_integration import launch_gradio
            launch_gradio(
                server_name=config['host'],
                server_port=config['port'],
                share=config['share'],
                debug=config['debug']
            )
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        if config['debug']:
            import traceback
            traceback.print_exc()

def show_help():
    """Show help information."""
    help_text = """
    🎬 Video-OpusClip AI Studio - Quick Start Guide
    
    Features:
    • 🎥 Video Processing: Upload and process videos
    • 🤖 AI Generation: Generate videos from text or images
    • 📈 Viral Analysis: Analyze content viral potential
    • 🏋️ Training: Train custom AI models
    • ⚡ Performance: Monitor system performance
    • ⚙️ Settings: Configure system settings
    
    Usage:
    • Upload videos and process them with AI
    • Generate viral content from text prompts
    • Analyze content performance across platforms
    • Train custom models for your specific needs
    
    Tips:
    • Use GPU for faster AI processing
    • Enable caching for better performance
    • Monitor system resources in Performance tab
    • Adjust settings based on your hardware
    
    For more information, see GRADIO_INTEGRATION.md
    """
    print(help_text)

def main():
    """Main function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quick Start for Video-OpusClip Gradio Interface")
    parser.add_argument("--auto", action="store_true", help="Use default settings without prompts")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--simple", action="store_true", help="Use simple interface")
    parser.add_argument("--help-features", action="store_true", help="Show feature help")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help_features:
        show_help()
        return
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Get configuration
    if args.auto:
        # Use command line arguments
        config = {
            "host": args.host,
            "port": args.port,
            "share": args.share,
            "debug": args.debug,
            "simple": args.simple
        }
    else:
        # Get user preferences interactively
        config = get_user_preferences()
    
    # Launch interface
    launch_interface(config)

if __name__ == "__main__":
    main() 