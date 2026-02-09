#!/usr/bin/env python3
"""
Quick Launch Script for Comprehensive Logging System
Advanced logging for training progress and errors
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import time

# Setup basic logging for the launch script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'numpy', 'pandas', 'psutil'
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
        requirements_file = Path(__file__).parent / "requirements_comprehensive_logging.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error("❌ requirements_comprehensive_logging.txt not found")
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
        "logs",
        "logs/training_metrics",
        "logs/error_reports",
        "logs/system_monitoring",
        "logs/performance_metrics"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def run_logging_demo():
    """Run comprehensive logging demonstration."""
    try:
        from comprehensive_logging import setup_logging, LoggingConfig
        
        logger.info("🚀 Running Comprehensive Logging Demo...")
        
        # Setup logging with comprehensive configuration
        config = LoggingConfig(
            log_level="DEBUG",
            log_dir="./logs",
            enable_console=True,
            enable_file=True,
            enable_json=True,
            log_training_metrics=True,
            log_system_metrics=True,
            log_gpu_metrics=True,
            log_memory_usage=True,
            log_performance=True,
            log_errors=True,
            log_warnings=True,
            log_debug=True,
            structured_format=True,
            include_timestamp=True,
            include_context=True,
            include_stack_trace=True,
            max_queue_size=1000,
            flush_interval=1.0,
            enable_async_logging=True,
            enable_thread_safety=True
        )
        
        comprehensive_logger = setup_logging("demo_logger", **config.__dict__)
        
        print("\n=== Comprehensive Logging Demo ===")
        
        # Demo 1: Basic logging
        print("\n1. Basic Logging Demo")
        comprehensive_logger.log_info("Starting comprehensive logging demo")
        comprehensive_logger.log_warning("This is a warning message")
        comprehensive_logger.log_debug("Debug information for developers")
        
        # Demo 2: Training metrics logging
        print("\n2. Training Metrics Logging Demo")
        for epoch in range(3):
            for step in range(5):
                # Simulate training metrics
                loss = 1.0 / (epoch * 5 + step + 1) + 0.1
                accuracy = max(0, 1 - loss + 0.05)
                
                comprehensive_logger.log_training_step(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=0.001,
                    gradient_norm=1.0 + step * 0.1,
                    memory_usage=2.5 + step * 0.1
                )
                time.sleep(0.1)  # Simulate work
            
            # Log epoch summary
            comprehensive_logger.log_epoch_summary(
                epoch=epoch,
                train_loss=0.5,
                val_loss=0.6,
                train_accuracy=0.8,
                val_accuracy=0.75
            )
        
        # Demo 3: Error tracking demo
        print("\n3. Error Tracking Demo")
        try:
            # Simulate an error
            raise ValueError("Simulated training error")
        except Exception as e:
            comprehensive_logger.log_error(
                error=e,
                context={
                    "operation": "training_simulation",
                    "epoch": 2,
                    "step": 3,
                    "batch_size": 32
                },
                severity="ERROR",
                recovery_attempted=True
            )
            
            # Simulate recovery
            comprehensive_logger.error_tracker.track_recovery_success(
                error_type="ValueError",
                recovery_method="automatic_restart"
            )
        
        # Demo 4: Performance tracking demo
        print("\n4. Performance Tracking Demo")
        
        # Using context manager
        with comprehensive_logger.performance_tracking("data_processing"):
            time.sleep(0.5)  # Simulate data processing
        
        # Manual performance logging
        start_time = time.time()
        time.sleep(0.3)  # Simulate operation
        duration = time.time() - start_time
        comprehensive_logger.log_performance(
            "model_inference",
            duration,
            batch_size=64,
            input_size="512x512"
        )
        
        # Demo 5: System monitoring demo
        print("\n5. System Monitoring Demo")
        system_metrics = comprehensive_logger.system_monitor._collect_system_metrics()
        print(f"   CPU Usage: {system_metrics.get('cpu_percent', 'N/A')}%")
        print(f"   Memory Usage: {system_metrics.get('memory_percent', 'N/A')}%")
        print(f"   GPU Count: {system_metrics.get('gpu_count', 'N/A')}")
        
        # Demo 6: Get logging summary
        print("\n6. Logging Summary Demo")
        summary = comprehensive_logger.get_logging_summary()
        
        print(f"   Training Metrics: {summary['training_metrics']['metrics_count']} entries")
        print(f"   Total Errors: {summary['error_analysis']['total_errors']}")
        print(f"   Recovery Rate: {summary['error_analysis']['recovery_success_rate']:.2%}")
        print(f"   Log Files Created:")
        for log_type, log_path in summary['log_files'].items():
            print(f"     - {log_type}: {log_path}")
        
        # Demo 7: Error analysis
        print("\n7. Error Analysis Demo")
        error_analysis = comprehensive_logger.error_tracker.get_error_analysis()
        print(f"   Most Common Errors:")
        for error_info in error_analysis['most_common_errors'][:3]:
            print(f"     - {error_info['error_type']}: {error_info['count']} occurrences")
        
        print(f"   Error Trends: {len(error_analysis['error_trends']['hourly_distribution'])} time periods")
        
        # Cleanup
        comprehensive_logger.cleanup()
        
        logger.info("✅ Comprehensive logging demo completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import comprehensive logging: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to run demo: {e}")
        return False

def show_log_files():
    """Display information about generated log files."""
    log_dir = Path(__file__).parent / "logs"
    
    if not log_dir.exists():
        print("❌ No logs directory found")
        return
    
    print("\n=== Generated Log Files ===")
    
    log_files = {
        "application.log": "Main application logs (human-readable)",
        "application.jsonl": "JSON-formatted logs (machine-readable)",
        "training_metrics.jsonl": "Training step metrics (JSONL format)",
        "training_progress.csv": "Training progress (CSV format)",
        "errors.jsonl": "Error tracking and analysis (JSONL format)",
        "error_summary.json": "Error analysis summary (JSON format)",
        "system_metrics.jsonl": "System resource metrics (JSONL format)"
    }
    
    for filename, description in log_files.items():
        file_path = log_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            print(f"✅ {filename} ({size_str}) - {description}")
        else:
            print(f"❌ {filename} - {description}")

def show_usage_examples():
    """Display usage examples for the comprehensive logging system."""
    print("\n=== Usage Examples ===")
    
    examples = [
        {
            "title": "Basic Setup",
            "code": """from comprehensive_logging import setup_logging

# Initialize with default configuration
logger = setup_logging("seo_evaluation")

# Or with custom configuration
logger = setup_logging(
    name="seo_evaluation",
    log_level="DEBUG",
    log_dir="./logs",
    enable_console=True,
    enable_file=True,
    log_training_metrics=True,
    log_system_metrics=True
)"""
        },
        {
            "title": "Training Logging",
            "code": """# Log training steps
logger.log_training_step(
    epoch=epoch,
    step=step,
    loss=loss,
    accuracy=accuracy,
    learning_rate=optimizer.param_groups[0]['lr'],
    gradient_norm=gradient_norm,
    memory_usage=torch.cuda.memory_allocated() / 1e9
)

# Log epoch summary
logger.log_epoch_summary(
    epoch=epoch,
    train_loss=avg_train_loss,
    val_loss=avg_val_loss,
    train_accuracy=avg_train_acc,
    val_accuracy=avg_val_acc
)"""
        },
        {
            "title": "Error Tracking",
            "code": """try:
    result = model.inference(input_data)
except Exception as e:
    logger.log_error(
        error=e,
        context={"operation": "model_inference"},
        severity="ERROR",
        recovery_attempted=False
    )
    
    # Track successful recovery
    logger.error_tracker.track_recovery_success(
        error_type=type(e).__name__,
        recovery_method="fallback_method"
    )"""
        },
        {
            "title": "Performance Tracking",
            "code": """# Using context manager
with logger.performance_tracking("data_loading"):
    data = load_large_dataset()

# Manual performance logging
start_time = time.time()
result = expensive_operation()
duration = time.time() - start_time
logger.log_performance("expensive_operation", duration)"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print("   Code:")
        for line in example['code'].split('\n'):
            print(f"      {line}")
        print()

def main():
    """Main launch function."""
    print("=" * 80)
    print("📊 Comprehensive Logging System for SEO Evaluation")
    print("   Advanced logging for training progress and errors")
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
    
    try:
        import psutil
        print(f"psutil Version: {psutil.__version__}")
    except ImportError:
        print("psutil not available")
    
    # Display comprehensive logging features
    print("\n" + "=" * 80)
    print("🎯 Comprehensive Logging Features")
    print("=" * 80)
    print("🚀 Core Logging Capabilities")
    print("   • Structured logging with JSON format")
    print("   • Multi-output support (console, file, JSON)")
    print("   • Async logging for high-performance applications")
    print("   • Thread-safe multi-threaded environment support")
    print()
    print("📊 Training Metrics Logging")
    print("   • Step-by-step training metrics (loss, accuracy, learning rate)")
    print("   • Epoch completion summaries")
    print("   • Progress tracking in CSV and JSONL formats")
    print("   • Performance metrics (time, memory, gradient norms)")
    print()
    print("🛡️ Error Tracking & Analysis")
    print("   • Error classification by type and severity")
    print("   • Recovery attempt tracking and success rates")
    print("   • Trend analysis and pattern identification")
    print("   • Context preservation for debugging")
    print()
    print("💻 System Monitoring")
    print("   • Real-time resource metrics (CPU, memory, disk, network)")
    print("   • GPU monitoring (CUDA memory allocation)")
    print("   • Continuous system health tracking")
    print("   • Performance profiling and operation timing")
    print()
    print("🔧 Advanced Features")
    print("   • Configurable logging for different environments")
    print("   • Automatic log rotation and backup management")
    print("   • Context managers for performance tracking")
    print("   • Easy integration with existing systems")
    
    # Run demo
    print("\n" + "=" * 80)
    print("🧪 Running Comprehensive Logging Demo...")
    print("=" * 80)
    
    try:
        if run_logging_demo():
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed")
    except Exception as e:
        logger.error(f"❌ Failed to run demo: {e}")
    
    # Show generated log files
    show_log_files()
    
    # Show usage examples
    show_usage_examples()
    
    # Display integration instructions
    print("\n" + "=" * 80)
    print("🔗 Integration Instructions")
    print("=" * 80)
    print("1. Import the logging system:")
    print("   from comprehensive_logging import setup_logging")
    print()
    print("2. Initialize with your configuration:")
    print("   logger = setup_logging('your_app_name', log_level='INFO')")
    print()
    print("3. Use in your training loops:")
    print("   logger.log_training_step(epoch, step, loss, accuracy)")
    print()
    print("4. Track errors and performance:")
    print("   logger.log_error(error, context)")
    print("   with logger.performance_tracking('operation'): pass")
    print()
    print("5. Get comprehensive summaries:")
    print("   summary = logger.get_logging_summary()")
    print()
    print("6. Cleanup when done:")
    print("   logger.cleanup()")
    
    print("\n" + "=" * 80)
    print("🎉 Comprehensive Logging System Ready!")
    print("=" * 80)
    print("📁 Logs: logs/")
    print("📊 Metrics: logs/training_metrics/")
    print("🛡️ Errors: logs/error_reports/")
    print("💻 System: logs/system_monitoring/")
    print("📈 Performance: logs/performance_metrics/")
    print("=" * 80)
    print("📚 Documentation: README_COMPREHENSIVE_LOGGING.md")
    print("🔧 Requirements: requirements_comprehensive_logging.txt")
    print("🚀 Launch: python launch_comprehensive_logging.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
