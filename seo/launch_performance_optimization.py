#!/usr/bin/env python3
"""
Quick Launch Script for Performance Optimization Module
Advanced optimization techniques for maximum performance and efficiency
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'numpy', 'pandas', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    missing = check_dependencies()
    if not missing:
        logger.info("All dependencies are already installed!")
        return True
    
    logger.info(f"Installing missing packages: {', '.join(missing)}")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_performance_optimization.txt"
        ])
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def check_cuda():
    """Check CUDA availability and version."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            
            logger.info(f"✓ CUDA is available (version {cuda_version})")
            logger.info(f"✓ GPU devices: {device_count}")
            logger.info(f"✓ Primary GPU: {device_name}")
            return True
        else:
            logger.warning("⚠ CUDA is not available - will use CPU optimizations")
            return False
    except ImportError:
        logger.error("✗ PyTorch not available")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "models",
        "cache",
        "profiles"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")

def run_performance_demo():
    """Run the performance optimization demo."""
    try:
        logger.info("🚀 Starting Performance Optimization Demo...")
        
        # Import and run the demo
        from performance_optimization import (
            PerformanceConfig, PerformanceOptimizer, ModelOptimizer,
            TrainingOptimizer, CacheManager, PerformanceProfiler
        )
        import torch.nn as nn
        
        # Create configuration
        config = PerformanceConfig(
            enable_amp=True,
            enable_compile=True,
            enable_gradient_checkpointing=True,
            num_workers=2,
            enable_profiling=True,
            enable_system_optimization=True
        )
        
        logger.info("✓ Performance configuration created")
        
        # Initialize performance optimizer
        optimizer = PerformanceOptimizer(config)
        logger.info("✓ Performance optimizer initialized")
        
        # Start monitoring
        optimizer.performance_monitor.start_monitoring()
        logger.info("✓ Performance monitoring started")
        
        # Create a sample model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        logger.info("✓ Sample model created")
        
        # Apply model optimizations
        model_optimizer = ModelOptimizer(model, config)
        optimized_model = model_optimizer.get_optimized_model()
        logger.info("✓ Model optimizations applied")
        
        # Setup training optimizer
        training_optimizer = TrainingOptimizer(optimized_model, config)
        logger.info("✓ Training optimizer configured")
        
        # Setup cache manager
        cache_manager = CacheManager(config)
        logger.info("✓ Cache manager initialized")
        
        # Setup profiler
        profiler = PerformanceProfiler(config)
        logger.info("✓ Performance profiler ready")
        
        # Example training loop with optimizations
        optimizer_opt = torch.optim.Adam(optimized_model.parameters())
        logger.info("✓ Optimizer configured")
        
        logger.info("🔄 Running optimized training loop...")
        
        with training_optimizer.training_context():
            with profiler.profile_context("training_loop"):
                for epoch in range(3):
                    # Simulate training
                    dummy_input = torch.randn(16, 100)
                    dummy_target = torch.randn(16, 10)
                    
                    # Forward pass
                    output = optimized_model(dummy_input)
                    loss = nn.MSELoss()(output, dummy_target)
                    
                    # Optimized training step
                    training_optimizer.optimize_training_step(loss, optimizer_opt)
                    
                    logger.info(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
                    
                    # Cache some data
                    cache_manager.cache_data(f"epoch_{epoch}", dummy_input)
        
        logger.info("✓ Training loop completed")
        
        # Get performance summary
        performance_summary = optimizer.performance_monitor.get_performance_summary()
        cache_stats = cache_manager.get_cache_stats()
        optimization_stats = training_optimizer.get_optimization_stats()
        
        logger.info("\n📊 === Performance Summary ===")
        logger.info(f"CPU Usage: {performance_summary.get('cpu_stats', {}).get('mean', 0):.2f}%")
        logger.info(f"Memory Usage: {performance_summary.get('memory_stats', {}).get('mean', 0):.2f}%")
        logger.info(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0):.2f}")
        logger.info(f"Training Steps: {optimization_stats.get('step', 0)}")
        
        # Stop monitoring
        optimizer.performance_monitor.stop_monitoring()
        logger.info("✓ Performance monitoring stopped")
        
        logger.info("🎉 Performance Optimization Demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running performance demo: {e}")
        return False

def show_optimization_files():
    """Show generated optimization files."""
    files = [
        "performance_optimization.py",
        "requirements_performance_optimization.txt",
        "README_PERFORMANCE_OPTIMIZATION.md"
    ]
    
    logger.info("\n📁 Generated Performance Optimization Files:")
    for file in files:
        if Path(file).exists():
            logger.info(f"  ✓ {file}")
        else:
            logger.warning(f"  ✗ {file} (missing)")

def show_usage_examples():
    """Show usage examples."""
    examples = [
        "Basic Setup:",
        "  from performance_optimization import PerformanceConfig, PerformanceOptimizer",
        "  config = PerformanceConfig(enable_amp=True, enable_compile=True)",
        "  optimizer = PerformanceOptimizer(config)",
        "",
        "Model Optimization:",
        "  from performance_optimization import ModelOptimizer",
        "  model_optimizer = ModelOptimizer(model, config)",
        "  optimized_model = model_optimizer.get_optimized_model()",
        "",
        "Training Optimization:",
        "  from performance_optimization import TrainingOptimizer",
        "  training_optimizer = TrainingOptimizer(optimized_model, config)",
        "  with training_optimizer.training_context():",
        "      # Your training loop here",
        "",
        "Performance Monitoring:",
        "  optimizer.performance_monitor.start_monitoring()",
        "  summary = optimizer.performance_monitor.get_performance_summary()",
        "  optimizer.performance_monitor.stop_monitoring()"
    ]
    
    logger.info("\n💡 Usage Examples:")
    for example in examples:
        logger.info(f"  {example}")

def main():
    """Main execution function."""
    logger.info("🚀 Performance Optimization Module Launcher")
    logger.info("=" * 50)
    
    # Check dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies. Exiting.")
        return
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Show files
    show_optimization_files()
    
    # Run demo
    if run_performance_demo():
        logger.info("\n✅ Performance optimization module is ready!")
        show_usage_examples()
    else:
        logger.error("\n❌ Failed to run performance demo")
    
    logger.info("\n" + "=" * 50)
    logger.info("🎯 Performance Optimization Module Launch Complete!")

if __name__ == "__main__":
    main()
