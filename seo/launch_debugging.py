#!/usr/bin/env python3
"""
Quick Launch Script for Advanced Error Handling and Debugging Tools
Comprehensive debugging with real-time monitoring, error tracking, and troubleshooting
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
        'torch', 'gradio', 'numpy', 'pandas', 'psutil'
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
        requirements_file = Path(__file__).parent / "requirements_debugging.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_debugging.txt not found")
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
        "debug_logs",
        "error_tracking",
        "performance_metrics",
        "memory_snapshots",
        "exported_data"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def launch_debugging_tools():
    """Launch the debugging tools interface."""
    try:
        from gradio_debugging_tools import create_debugging_interface
        
        logger.info("🚀 Launching Advanced Error Handling and Debugging Tools...")
        
        # Create the interface
        demo = create_debugging_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7864,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import debugging tools: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to launch debugging tools: {e}")
        return False

def main():
    """Main launch function."""
    print("=" * 80)
    print("🔧 Advanced Error Handling and Debugging Tools")
    print("   Comprehensive debugging with real-time monitoring and error tracking")
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
        import psutil
        print(f"psutil Version: {psutil.__version__}")
    except ImportError:
        print("psutil not available")
    
    # Display debugging features
    print("\n" + "=" * 80)
    print("🎯 Debugging Tools Features")
    print("=" * 80)
    print("🔧 Advanced Debugging")
    print("   • Debug mode toggle (user-friendly vs detailed)")
    print("   • Function breakpoints with conditions")
    print("   • Performance profiling and metrics")
    print("   • Memory monitoring and snapshots")
    print()
    print("📊 System Monitoring")
    print("   • Real-time CPU, memory, disk monitoring")
    print("   • GPU memory and utilization tracking")
    print("   • System health status indicators")
    print("   • Background monitoring threads")
    print()
    print("🚨 Error Tracking & Analysis")
    print("   • Comprehensive error logging with context")
    print("   • Error pattern analysis and statistics")
    print("   • Recovery strategies and suggestions")
    print("   • Performance impact tracking")
    print()
    print("🖥️ Debugging Interface")
    print("   • Interactive Gradio dashboard")
    print("   • Real-time system health display")
    print("   • Error analysis and reporting")
    print("   • Data export capabilities")
    print()
    print("🧪 Testing & Validation")
    print("   • Error simulation and testing")
    print("   • Performance benchmarking")
    print("   • System health validation")
    print("   • Debug data management")
    
    # Launch debugging tools
    print("\n" + "=" * 80)
    print("🌐 Launching Debugging Tools...")
    print("=" * 80)
    print("📱 Local URL: http://localhost:7864")
    print("🌍 Public URL: Will be provided after launch")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 80)
    
    try:
        launch_debugging_tools()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
