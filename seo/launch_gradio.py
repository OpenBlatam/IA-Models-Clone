#!/usr/bin/env python3
"""
Quick Launch Script for Gradio SEO Interface
Ultra-Optimized SEO Evaluation System with Gradient Clipping and NaN/Inf Handling
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'transformers', 'gradio', 'numpy', 'pandas', 
        'matplotlib', 'seaborn', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    logger.info("Installing missing dependencies...")
    
    try:
        # Install from requirements file
        requirements_file = Path(__file__).parent / "requirements_gradio.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_gradio.txt not found")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA is available - {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.warning("⚠️ CUDA is not available - using CPU")
            return False
    except ImportError:
        logger.warning("⚠️ PyTorch not installed - cannot check CUDA")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "logs",
        "runs",
        "checkpoints"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def launch_interface():
    """Launch the Gradio interface."""
    try:
        from gradio_seo_interface import create_gradio_interface
        
        logger.info("🚀 Launching Gradio SEO Interface...")
        
        # Create the interface
        demo = create_gradio_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Gradio interface: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to launch interface: {e}")
        return False

def main():
    """Main launch function."""
    print("=" * 60)
    print("🚀 Ultra-Optimized SEO Evaluation System")
    print("   With Gradient Clipping and NaN/Inf Handling")
    print("=" * 60)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_dependencies():
                logger.error("❌ Failed to install dependencies. Exiting.")
                return
        else:
            logger.error("❌ Cannot proceed without required dependencies. Exiting.")
            return
    
    # Check CUDA
    logger.info("Checking CUDA availability...")
    cuda_available = check_cuda()
    
    # Create directories
    logger.info("Creating necessary directories...")
    create_directories()
    
    # Display system information
    print("\n" + "=" * 60)
    print("📊 System Information")
    print("=" * 60)
    
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    import gradio as gr
    print(f"Gradio Version: {gr.__version__}")
    
    # Launch interface
    print("\n" + "=" * 60)
    print("🌐 Launching Web Interface...")
    print("=" * 60)
    print("📱 Local URL: http://localhost:7860")
    print("🌍 Public URL: Will be provided after launch")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        launch_interface()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
