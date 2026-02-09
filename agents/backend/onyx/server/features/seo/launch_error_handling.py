#!/usr/bin/env python3
"""
Quick Launch Script for Error Handling and Input Validation Demo
Comprehensive error handling with user-friendly messages and recovery strategies
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
        'torch', 'gradio', 'numpy', 'pandas'
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
        requirements_file = Path(__file__).parent / "requirements_error_handling.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_error_handling.txt not found")
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
        "error_logs",
        "validation_tests",
        "demo_outputs",
        "recovery_examples"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def launch_error_handling_demo():
    """Launch the error handling demo interface."""
    try:
        from gradio_error_handling import create_error_handling_interface
        
        logger.info("🚀 Launching Error Handling and Input Validation Demo...")
        
        # Create the interface
        demo = create_error_handling_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7863,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import error handling demo: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to launch demo: {e}")
        return False

def main():
    """Main launch function."""
    print("=" * 80)
    print("🛡️ Error Handling and Input Validation Demo")
    print("   Comprehensive error handling with user-friendly messages and recovery strategies")
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
        import numpy as np
        print(f"NumPy Version: {np.__version__}")
    except ImportError:
        print("NumPy not available")
    
    # Display demo features
    print("\n" + "=" * 80)
    print("🎯 Error Handling Demo Features")
    print("=" * 80)
    print("🔍 Input Validation")
    print("   • Text validation (length, character set, security)")
    print("   • Number validation (range, type, boundaries)")
    print("   • File validation (type, size, security)")
    print("   • URL/Email validation (format, structure)")
    print()
    print("🚨 Error Handling")
    print("   • Validation errors with clear messages")
    print("   • Model errors with troubleshooting tips")
    print("   • System errors with recovery guidance")
    print("   • Memory and network error handling")
    print()
    print("🔄 Recovery Strategies")
    print("   • Automatic retry for simple operations")
    print("   • Actionable recovery suggestions")
    print("   • Error prevention and fallback mechanisms")
    print("   • Graceful degradation strategies")
    print()
    print("📊 Error Monitoring")
    print("   • Comprehensive error logging")
    print("   • Error pattern analysis")
    print("   • Performance impact tracking")
    print("   • User experience monitoring")
    print()
    print("🛡️ Security Features")
    print("   • Input sanitization and validation")
    print("   • Malicious content blocking")
    print("   • File type restrictions")
    print("   • Script injection prevention")
    
    # Launch demo
    print("\n" + "=" * 80)
    print("🌐 Launching Error Handling Demo...")
    print("=" * 80)
    print("📱 Local URL: http://localhost:7863")
    print("🌍 Public URL: Will be provided after launch")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 80)
    
    try:
        launch_error_handling_demo()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
