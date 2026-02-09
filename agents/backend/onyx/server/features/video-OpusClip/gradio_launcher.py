#!/usr/bin/env python3
"""
Gradio Launcher for Video-OpusClip

Simple launcher script to start the Gradio interface with proper configuration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_environment():
    """Setup environment variables and logging."""
    
    # Set default environment variables
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("ENABLE_STRUCTURED_LOGGING", "true")
    os.environ.setdefault("USE_GPU", "true")
    os.environ.setdefault("ENABLE_CACHING", "true")
    os.environ.setdefault("MAX_WORKERS", str(os.cpu_count() or 4))
    
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "gradio",
        "torch",
        "diffusers",
        "transformers",
        "numpy",
        "pillow",
        "opencv-python"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main launcher function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Video-OpusClip Gradio Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--simple", action="store_true", help="Use simple interface")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🚀 Starting Video-OpusClip Gradio Interface...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🌐 Share: {args.share}")
    print(f"🐛 Debug: {args.debug}")
    
    try:
        if args.simple:
            # Use simple interface
            from gradio_demo import demo
            demo.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                debug=args.debug
            )
        else:
            # Use enhanced interface
            from gradio_integration import launch_gradio
            launch_gradio(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                debug=args.debug
            )
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 