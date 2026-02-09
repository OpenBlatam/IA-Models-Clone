#!/usr/bin/env python3
"""
Quick Launch Script for Interactive SEO Model Demos
Ultra-Optimized SEO Evaluation System with Advanced Visualization
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
        'torch', 'transformers', 'gradio', 'plotly', 'numpy', 'pandas', 
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
        requirements_file = Path(__file__).parent / "requirements_interactive_demos.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_interactive_demos.txt not found")
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
        "checkpoints",
        "demo_outputs",
        "visualizations"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def launch_interactive_demos():
    """Launch the interactive demos interface."""
    try:
        from gradio_interactive_demos import create_interactive_demos
        
        logger.info("🚀 Launching Interactive SEO Model Demos...")
        
        # Create the interface
        demo = create_interactive_demos()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import interactive demos: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to launch demos: {e}")
        return False

def main():
    """Main launch function."""
    print("=" * 70)
    print("🚀 Interactive SEO Model Demos")
    print("   Ultra-Optimized SEO Evaluation System with Advanced Visualization")
    print("=" * 70)
    
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
    print("\n" + "=" * 70)
    print("📊 System Information")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch not available")
    
    try:
        import gradio as gr
        print(f"Gradio Version: {gr.__version__}")
    except ImportError:
        print("Gradio not available")
    
    try:
        import plotly
        print(f"Plotly Version: {plotly.__version__}")
    except ImportError:
        print("Plotly not available")
    
    # Display demo features
    print("\n" + "=" * 70)
    print("🎯 Interactive Demo Features")
    print("=" * 70)
    print("🔍 Real-time SEO Analysis")
    print("   • Instant text evaluation with live metrics")
    print("   • Interactive Plotly visualizations")
    print("   • SEO performance radar charts")
    print("   • Keyword analysis heatmaps")
    print()
    print("📊 Batch SEO Analysis")
    print("   • Multiple text comparison")
    print("   • Performance ranking")
    print("   • Correlation analysis")
    print("   • Multi-metric radar charts")
    print()
    print("🏋️ Interactive Training Demo")
    print("   • Real-time training progress")
    print("   • Configurable hyperparameters")
    print("   • 3D training surface visualization")
    print("   • Live metrics monitoring")
    
    # Launch interface
    print("\n" + "=" * 70)
    print("🌐 Launching Interactive Demos...")
    print("=" * 70)
    print("📱 Local URL: http://localhost:7861")
    print("🌍 Public URL: Will be provided after launch")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        launch_interactive_demos()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
