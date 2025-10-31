"""
Installation Script for Ultimate Opus Clip Improvements

This script installs and configures all improvements for the Ultimate Opus Clip system.
"""

import subprocess
import sys
import os
from pathlib import Path
import structlog

logger = structlog.get_logger("install_improvements")

def install_dependencies():
    """Install required dependencies."""
    try:
        logger.info("Installing dependencies...")
        
        # Install from requirements file
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_opus_clip.txt"
        ])
        
        # Install additional performance packages
        performance_packages = [
            "psutil",
            "pyyaml",
            "structlog",
            "aiofiles",
            "httpx"
        ]
        
        for package in performance_packages:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        logger.info("Dependencies installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Setup required directories."""
    try:
        directories = [
            "outputs",
            "cache",
            "logs",
            "temp",
            "models",
            "data"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup directories: {e}")
        return False

def configure_system():
    """Configure system settings."""
    try:
        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write("# Ultimate Opus Clip Environment Variables\n")
                f.write("DEBUG=false\n")
                f.write("LOG_LEVEL=INFO\n")
                f.write("MAX_WORKERS=4\n")
                f.write("CACHE_SIZE=1024\n")
        
        logger.info("System configuration completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure system: {e}")
        return False

def verify_installation():
    """Verify installation."""
    try:
        # Test imports
        import cv2
        import torch
        import fastapi
        import structlog
        
        logger.info("Core dependencies verified")
        
        # Test GPU availability
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU detected - using CPU only")
        
        return True
        
    except Exception as e:
        logger.error(f"Installation verification failed: {e}")
        return False

def main():
    """Main installation function."""
    logger.info("Starting Ultimate Opus Clip improvements installation...")
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up directories", setup_directories),
        ("Configuring system", configure_system),
        ("Verifying installation", verify_installation)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Step: {step_name}")
        if not step_func():
            logger.error(f"Failed at step: {step_name}")
            return False
    
    logger.info("Ultimate Opus Clip improvements installed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


