#!/usr/bin/env python3
"""
Video-OpusClip Application Launcher
Simple script to run the main application with proper configuration
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def setup_environment():
    """Setup environment variables and configuration"""
    # Set default environment variables if not present
    os.environ.setdefault("APP_NAME", "Video-OpusClip")
    os.environ.setdefault("APP_VERSION", "1.0.0")
    os.environ.setdefault("DEBUG", "False")
    os.environ.setdefault("HOST", "0.0.0.0")
    os.environ.setdefault("PORT", "8000")
    os.environ.setdefault("RELOAD", "True")
    
    # Database settings
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:password@localhost:5432/video_opusclip")
    os.environ.setdefault("DATABASE_TYPE", "postgresql")
    
    # Async flow settings
    os.environ.setdefault("MAX_CONCURRENT_TASKS", "100")
    os.environ.setdefault("MAX_CONCURRENT_CONNECTIONS", "50")
    os.environ.setdefault("TIMEOUT", "30.0")
    os.environ.setdefault("RETRY_ATTEMPTS", "3")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_opusclip.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        import asyncio
        import aiohttp
        print("✓ Core dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements_main.txt")
        return False
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("⚠ PyTorch not available - some features may be limited")
    
    try:
        import transformers
        print("✓ Transformers available")
    except ImportError:
        print("⚠ Transformers not available - some features may be limited")
    
    try:
        import diffusers
        print("✓ Diffusers available")
    except ImportError:
        print("⚠ Diffusers not available - some features may be limited")
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Video-OpusClip Application Launcher")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n📋 Configuration:")
    print(f"  Host: {os.environ.get('HOST', '0.0.0.0')}")
    print(f"  Port: {os.environ.get('PORT', '8000')}")
    print(f"  Debug: {os.environ.get('DEBUG', 'False')}")
    print(f"  Reload: {os.environ.get('RELOAD', 'True')}")
    print(f"  Database: {os.environ.get('DATABASE_TYPE', 'postgresql')}")
    
    print("\n🔧 Starting application...")
    
    try:
        # Import and run the main application
        from main import app, settings
        
        import uvicorn
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n⏹ Application stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 