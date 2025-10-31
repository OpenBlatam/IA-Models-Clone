#!/usr/bin/env python3
"""
Quick Launch Script for Robust Operations Module
Comprehensive try-except blocks for error-prone operations
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
        'torch', 'numpy', 'pandas'
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
        requirements_file = Path(__file__).parent / "requirements_robust_operations.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_robust_operations.txt not found")
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
        "robust_operations_logs",
        "error_reports",
        "performance_metrics",
        "exported_data"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def run_examples():
    """Run example robust operations."""
    try:
        from gradio_robust_operations import (
            RobustOperationManager, 
            example_data_loading, 
            example_model_inference, 
            example_data_processing
        )
        
        logger.info("🚀 Running Robust Operations Examples...")
        
        # Run examples
        print("\n=== Running Examples ===")
        example_data_loading()
        example_model_inference()
        example_data_processing()
        
        # Test manager
        print("\n=== Testing Manager ===")
        manager = RobustOperationManager()
        
        # Test safe operation
        def test_function(x):
            return x * 2
        
        result, error = manager.safe_operation("test_operation", test_function, 5)
        if error:
            print(f"❌ Operation failed: {error}")
        else:
            print(f"✅ Operation successful: {result}")
        
        # Get statistics
        stats = manager.get_comprehensive_statistics()
        print(f"📊 Statistics: {stats}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import robust operations: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to run examples: {e}")
        return False

def main():
    """Main launch function."""
    print("=" * 80)
    print("🛡️ Robust Operations with Comprehensive Try-Except Blocks")
    print("   Enhanced error handling for data loading, model inference, and processing")
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
        import numpy as np
        print(f"NumPy Version: {np.__version__}")
    except ImportError:
        print("NumPy not available")
    
    try:
        import pandas as pd
        print(f"Pandas Version: {pd.__version__}")
    except ImportError:
        print("Pandas not available")
    
    # Display robust operations features
    print("\n" + "=" * 80)
    print("🎯 Robust Operations Features")
    print("=" * 80)
    print("🛡️ Comprehensive Error Handling")
    print("   • Extensive try-except blocks for all operations")
    print("   • Automatic retry with exponential backoff")
    print("   • Error classification and recovery strategies")
    print("   • Graceful degradation and fallback mechanisms")
    print()
    print("📊 Robust Data Loading")
    print("   • File type detection and validation")
    print("   • Multiple encoding fallbacks")
    print("   • Batch loading with error isolation")
    print("   • Corrupted file handling")
    print()
    print("🎯 Robust Model Inference")
    print("   • Automatic device setup and fallback")
    print("   • GPU memory monitoring and cleanup")
    print("   • Input validation and preparation")
    print("   • Batch inference with error handling")
    print()
    print("🔄 Robust Data Processing")
    print("   • Thread-safe parallel processing")
    print("   • Memory error recovery")
    print("   • Function validation and monitoring")
    print("   • Individual error handling per item")
    print()
    print("🛠️ Utility Tools")
    print("   • Decorators for robust operations")
    print("   • Context managers for safe resources")
    print("   • Performance monitoring and statistics")
    print("   • Comprehensive error reporting")
    
    # Run examples
    print("\n" + "=" * 80)
    print("🧪 Running Examples...")
    print("=" * 80)
    
    try:
        if run_examples():
            print("✅ Examples completed successfully!")
        else:
            print("❌ Examples failed")
    except Exception as e:
        logger.error(f"❌ Failed to run examples: {e}")
    
    # Display usage instructions
    print("\n" + "=" * 80)
    print("📚 Usage Instructions")
    print("=" * 80)
    print("1. Import the module:")
    print("   from gradio_robust_operations import RobustOperationManager")
    print()
    print("2. Initialize manager:")
    print("   manager = RobustOperationManager()")
    print()
    print("3. Use robust operations:")
    print("   result, error = manager.safe_operation('op_name', function, *args)")
    print()
    print("4. Check statistics:")
    print("   stats = manager.get_comprehensive_statistics()")
    print()
    print("5. Export error reports:")
    print("   manager.export_error_report('error_report.json')")
    print()
    print("6. Use decorators:")
    print("   @robust_operation(max_retries=3)")
    print("   def risky_function(): pass")
    print()
    print("7. Use context managers:")
    print("   with robust_file_operation('file.txt') as f:")
    print("       content = f.read()")
    
    print("\n" + "=" * 80)
    print("🎉 Robust Operations Module Ready!")
    print("=" * 80)
    print("📁 Logs: robust_operations_logs/")
    print("📊 Reports: error_reports/")
    print("📈 Metrics: performance_metrics/")
    print("💾 Data: exported_data/")
    print("=" * 80)

if __name__ == "__main__":
    main()
