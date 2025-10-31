#!/usr/bin/env python3
"""
Quick Launch Script for PyTorch Debugging Tools
Demonstrates PyTorch debugging capabilities integrated with comprehensive logging
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
        "logs/profiler",
        "logs/debug_reports",
        "logs/gradient_analysis",
        "logs/memory_tracking"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def run_pytorch_debugging_demo():
    """Run PyTorch debugging demonstration."""
    try:
        from comprehensive_logging import setup_logging, LoggingConfig
        
        logger.info("🚀 Running PyTorch Debugging Demo...")
        
        # Setup logging with comprehensive PyTorch debugging
        config = LoggingConfig(
            log_level="DEBUG",
            log_dir="./logs",
            enable_console=True,
            enable_file=True,
            enable_json=True,
            enable_pytorch_debugging=True,
            enable_autograd_anomaly_detection=True,
            enable_gradient_debugging=True,
            enable_memory_debugging=True,
            enable_tensor_debugging=True,
            enable_profiler=True
        )
        
        comprehensive_logger = setup_logging("pytorch_debug_demo", **config.__dict__)
        
        print("\n=== PyTorch Debugging Tools Demo ===")
        
        # Demo 1: Basic PyTorch debugging setup
        print("\n1. PyTorch Debugging Setup Demo")
        print(f"   Anomaly Detection: {comprehensive_logger.pytorch_debug.anomaly_detection_enabled}")
        print(f"   Gradient Debugging: {config.enable_gradient_debugging}")
        print(f"   Memory Debugging: {config.enable_memory_debugging}")
        print(f"   Tensor Debugging: {config.enable_tensor_debugging}")
        print(f"   Profiler: {config.enable_profiler}")
        
        # Demo 2: Create and debug a PyTorch model
        print("\n2. Model Creation and Debugging Demo")
        import torch
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Debug model state
        model_debug = comprehensive_logger.debug_model_state(model)
        print(f"   Model Parameters: {model_debug['total_parameters']}")
        print(f"   Trainable Parameters: {model_debug['trainable_parameters']}")
        print(f"   Training Mode: {model_debug['model_training_mode']}")
        
        # Demo 3: Tensor debugging
        print("\n3. Tensor Debugging Demo")
        x = torch.randn(32, 10, requires_grad=True)
        y = torch.randn(32, 1)
        
        # Debug input tensor
        input_debug = comprehensive_logger.debug_tensor(x, "input_tensor")
        print(f"   Input Shape: {input_debug['input_tensor_shape']}")
        print(f"   Input Device: {input_debug['input_tensor_device']}")
        print(f"   Requires Grad: {input_debug['input_tensor_requires_grad']}")
        print(f"   Has NaN: {input_debug['input_tensor_has_nan']}")
        
        # Demo 4: Training with debugging
        print("\n4. Training with Debugging Demo")
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop with debugging
        for step in range(3):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            
            # Debug gradients
            gradient_debug = comprehensive_logger.debug_model_gradients(model, loss, step=step)
            print(f"   Step {step}: Loss={loss.item():.4f}, Grad Norm={gradient_debug['total_grad_norm']:.4f}")
            
            # Debug memory
            memory_debug = comprehensive_logger.debug_model_memory(step=step)
            if torch.cuda.is_available():
                print(f"   Step {step}: CUDA Memory={memory_debug.get('cuda_memory_allocated', 'N/A')} GB")
            
            # Optimizer step
            optimizer.step()
            
            # Log training step with debugging
            debug_info = comprehensive_logger.log_training_step_with_debug(
                epoch=0,
                step=step,
                loss=loss.item(),
                model=model,
                accuracy=0.8,
                learning_rate=optimizer.param_groups[0]['lr']
            )
            
            time.sleep(0.1)  # Simulate work
        
        # Demo 5: Context managers
        print("\n5. Context Managers Demo")
        
        # PyTorch debugging context
        with comprehensive_logger.pytorch_debugging("test_operation", enable_anomaly_detection=True):
            test_tensor = torch.randn(5, 5, requires_grad=True)
            test_loss = test_tensor.sum()
            test_loss.backward()
            print("   Test operation completed with anomaly detection")
        
        # Gradient debugging context
        with comprehensive_logger.gradient_debugging(model, "test_gradient_op"):
            test_output = model(x[:5])
            test_loss = criterion(test_output, y[:5])
            test_loss.backward()
            print("   Test gradient operation completed with debugging")
        
        # Demo 6: Profiler demo
        print("\n6. Profiler Demo")
        comprehensive_logger.start_profiler()
        
        # Run some operations to profile
        for i in range(5):
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        comprehensive_logger.stop_profiler()
        print("   Profiler completed - traces saved to ./logs/profiler/")
        
        # Demo 7: Error handling with debugging
        print("\n7. Error Handling with Debugging Demo")
        try:
            # Create problematic tensor
            problem_tensor = torch.randn(5, 5)
            problem_tensor[0, 0] = float('nan')  # Introduce NaN
            
            # This should trigger an error
            result = problem_tensor.sum()
            
        except Exception as e:
            # Log error with debugging context
            comprehensive_logger.log_error(
                error=e,
                context={
                    "operation": "problematic_tensor_operation",
                    "tensor_debug": comprehensive_logger.debug_tensor(problem_tensor, "problem_tensor"),
                    "model_state": comprehensive_logger.debug_model_state(model)
                },
                severity="ERROR"
            )
            print("   Error logged with comprehensive debugging context")
        
        # Demo 8: Get comprehensive summary
        print("\n8. Comprehensive Summary Demo")
        summary = comprehensive_logger.get_logging_summary()
        
        print(f"   Training Metrics: {summary['training_metrics']['metrics_count']} entries")
        print(f"   Total Errors: {summary['error_analysis']['total_errors']}")
        
        # PyTorch debugging summary
        pytorch_debug = summary['pytorch_debug']
        print(f"   Anomaly Detection: {pytorch_debug['anomaly_detection_enabled']}")
        print(f"   Profiler Active: {pytorch_debug['profiler_active']}")
        print(f"   Gradient History: {len(pytorch_debug['gradient_history'])} entries")
        
        # Cleanup
        comprehensive_logger.cleanup()
        
        logger.info("✅ PyTorch debugging demo completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import comprehensive logging: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to run demo: {e}")
        return False

def show_debug_files():
    """Display information about generated debug files."""
    log_dir = Path(__file__).parent / "logs"
    
    if not log_dir.exists():
        print("❌ No logs directory found")
        return
    
    print("\n=== Generated Debug Files ===")
    
    debug_files = {
        "application.log": "Main application logs with debugging info",
        "application.jsonl": "JSON-formatted logs with debugging context",
        "training_metrics.jsonl": "Training metrics with debugging data",
        "errors.jsonl": "Error logs with debugging context",
        "system_metrics.jsonl": "System metrics including GPU memory"
    }
    
    for filename, description in debug_files.items():
        file_path = log_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            print(f"✅ {filename} ({size_str}) - {description}")
        else:
            print(f"❌ {filename} - {description}")
    
    # Check profiler directory
    profiler_dir = log_dir / "profiler"
    if profiler_dir.exists():
        profiler_files = list(profiler_dir.glob("*.json"))
        if profiler_files:
            print(f"✅ Profiler traces: {len(profiler_files)} files")
        else:
            print("⚠️ Profiler directory exists but no traces found")

def show_usage_examples():
    """Display usage examples for PyTorch debugging."""
    print("\n=== PyTorch Debugging Usage Examples ===")
    
    examples = [
        {
            "title": "Basic Setup",
            "code": """from comprehensive_logging import setup_logging

# Initialize with PyTorch debugging enabled
logger = setup_logging(
    "seo_evaluation",
    enable_pytorch_debugging=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=True
)"""
        },
        {
            "title": "Gradient Debugging",
            "code": """# Debug model gradients during training
output = model(input_data)
loss = criterion(output, target)
loss.backward()

# Comprehensive gradient debugging
gradient_debug = logger.debug_model_gradients(model, loss)
print(f"Gradient norm: {gradient_debug['total_grad_norm']}")
print(f"NaN gradients: {gradient_debug['nan_gradients']}")"""
        },
        {
            "title": "Memory Debugging",
            "code": """# Debug PyTorch memory usage
memory_debug = logger.debug_model_memory()

if torch.cuda.is_available():
    print(f"CUDA Memory: {memory_debug['cuda_memory_allocated']:.2f} GB")
    print(f"Memory Fragmentation: {memory_debug['cuda_memory_fragmentation']:.2%}")"""
        },
        {
            "title": "Context Managers",
            "code": """# Use context manager for specific operations
with logger.pytorch_debugging("critical_operation", enable_anomaly_detection=True):
    # This operation will have anomaly detection enabled
    output = model(input_data)
    # Anomaly detection automatically disabled after context"""
        },
        {
            "title": "Training with Debugging",
            "code": """# Log training step with comprehensive debugging
debug_info = logger.log_training_step_with_debug(
    epoch=epoch,
    step=step,
    loss=loss.item(),
    model=model,
    accuracy=accuracy
)

# Access debugging information
gradient_debug = debug_info['gradient_debug']
memory_debug = debug_info['memory_debug']
model_debug = debug_info['model_debug']"""
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
    print("🔧 PyTorch Debugging Tools for SEO Evaluation")
    print("   Integrated debugging with comprehensive logging system")
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
    
    # Display PyTorch debugging features
    print("\n" + "=" * 80)
    print("🎯 PyTorch Debugging Features")
    print("=" * 80)
    print("🚀 Core Debugging Capabilities")
    print("   • Autograd anomaly detection (torch.autograd.set_detect_anomaly())")
    print("   • Comprehensive gradient analysis and monitoring")
    print("   • CUDA and CPU memory usage tracking")
    print("   • Detailed tensor information and validation")
    print("   • Model parameter analysis and health monitoring")
    print()
    print("🔧 Advanced Debugging Tools")
    print("   • PyTorch profiler integration for performance analysis")
    print("   • Context managers for selective debugging")
    print("   • Real-time monitoring during training")
    print("   • Comprehensive logging integration")
    print()
    print("📊 Debugging Analytics")
    print("   • Gradient history tracking over time")
    print("   • Memory usage patterns and fragmentation analysis")
    print("   • Rich debugging context for error analysis")
    print("   • Performance metrics and optimization insights")
    print()
    print("🔗 SEO System Integration")
    print("   • Seamless integration with SEO evaluation system")
    print("   • Enhanced error handling with debugging context")
    print("   • Training optimization through debugging insights")
    print("   • Performance monitoring and bottleneck identification")
    
    # Run demo
    print("\n" + "=" * 80)
    print("🧪 Running PyTorch Debugging Demo...")
    print("=" * 80)
    
    try:
        if run_pytorch_debugging_demo():
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed")
    except Exception as e:
        logger.error(f"❌ Failed to run demo: {e}")
    
    # Show generated debug files
    show_debug_files()
    
    # Show usage examples
    show_usage_examples()
    
    # Display integration instructions
    print("\n" + "=" * 80)
    print("🔗 Integration Instructions")
    print("=" * 80)
    print("1. Import the logging system:")
    print("   from comprehensive_logging import setup_logging")
    print()
    print("2. Initialize with PyTorch debugging:")
    print("   logger = setup_logging('your_app', enable_pytorch_debugging=True)")
    print()
    print("3. Use debugging methods:")
    print("   logger.debug_model_gradients(model, loss)")
    print("   logger.debug_model_memory()")
    print("   logger.debug_tensor(tensor, 'name')")
    print()
    print("4. Use context managers:")
    print("   with logger.pytorch_debugging('operation'): pass")
    print("   with logger.gradient_debugging(model, 'op'): pass")
    print()
    print("5. Get comprehensive summaries:")
    print("   summary = logger.get_logging_summary()")
    print()
    print("6. Cleanup when done:")
    print("   logger.cleanup()")
    
    print("\n" + "=" * 80)
    print("🎉 PyTorch Debugging Tools Ready!")
    print("=" * 80)
    print("📁 Logs: logs/")
    print("🔧 Profiler: logs/profiler/")
    print("📊 Debug Reports: logs/debug_reports/")
    print("📈 Gradient Analysis: logs/gradient_analysis/")
    print("💾 Memory Tracking: logs/memory_tracking/")
    print("=" * 80)
    print("📚 Documentation: README_PYTORCH_DEBUGGING.md")
    print("🧪 Tests: test_pytorch_debugging.py")
    print("🚀 Launch: python launch_pytorch_debugging.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
