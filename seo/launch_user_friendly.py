#!/usr/bin/env python3
"""
Quick Launch Script for User-Friendly SEO Model Interface
Enhanced UX with Intuitive Workflows and Visual Appeal
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
        requirements_file = Path(__file__).parent / "requirements_user_friendly.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_user_friendly.txt not found")
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
        "visualizations",
        "user_experience"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def launch_user_friendly_interface():
    """Launch the user-friendly interface."""
    try:
        from gradio_user_friendly_interface import create_user_friendly_interface
        
        logger.info("🚀 Launching User-Friendly SEO Model Interface...")
        
        # Create the interface
        demo = create_user_friendly_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import user-friendly interface: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to launch interface: {e}")
        return False

def main():
    """Main launch function."""
    print("=" * 80)
    print("🚀 User-Friendly SEO Model Interface")
    print("   Enhanced UX with Intuitive Workflows and Visual Appeal")
    print("=" * 80)
    
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
    print("\n" + "=" * 80)
    print("📊 System Information")
    print("=" * 80)
    
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
    
    # Display interface features
    print("\n" + "=" * 80)
    print("🎯 User-Friendly Interface Features")
    print("=" * 80)
    print("🏠 Welcome & Overview")
    print("   • Easy model initialization with clear feedback")
    print("   • Quick start guide for new users")
    print("   • Comprehensive capability overview")
    print("   • Feature highlights and use cases")
    print()
    print("🔍 Text Analysis")
    print("   • Real-time SEO analysis with timing feedback")
    print("   • User-friendly reports with actionable recommendations")
    print("   • Interactive visualizations (gauges, dashboards)")
    print("   • Analysis depth options (basic/comprehensive)")
    print()
    print("📊 Batch Analysis")
    print("   • Multiple text processing and comparison")
    print("   • Performance ranking and insights")
    print("   • Comparative visualizations")
    print("   • Detailed results table")
    print()
    print("🏋️ Training Demo")
    print("   • Configurable training parameters")
    print("   • Real-time progress tracking")
    print("   • Live performance visualization")
    print("   • Educational insights and feedback")
    print()
    print("🎨 Enhanced UX")
    print("   • Modern gradient design with professional styling")
    print("   • Responsive layout for all devices")
    print("   • Smooth animations and hover effects")
    print("   • Accessibility features and keyboard navigation")
    
    # Launch interface
    print("\n" + "=" * 80)
    print("🌐 Launching User-Friendly Interface...")
    print("=" * 80)
    print("📱 Local URL: http://localhost:7862")
    print("🌍 Public URL: Will be provided after launch")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 80)
    
    try:
        launch_user_friendly_interface()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
