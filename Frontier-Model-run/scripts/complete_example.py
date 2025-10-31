#!/usr/bin/env python3
"""
Complete Example: Enhanced Frontier Model Training Pipeline
This example demonstrates all the enhanced features working together.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager, Environment, FrontierConfig
from error_handler import setup_logging, error_context, ErrorType, LogLevel
from performance_monitor import create_metrics_collector, TrainingMetrics, SystemMetrics
from test_framework import TestRunner, TestGenerator
from deployment_manager import DeploymentManager, DeploymentConfig, DeploymentType

def setup_environment():
    """Setup the training environment."""
    print("🚀 Setting up Enhanced Frontier Model Training Environment")
    
    # Create necessary directories
    directories = [
        "./logs",
        "./metrics", 
        "./test_results",
        "./data",
        "./models",
        "./config",
        "./tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def demonstrate_configuration_management():
    """Demonstrate configuration management features."""
    print("\n📋 Configuration Management Demo")
    
    # Create config manager
    manager = ConfigManager()
    
    # Create default configurations for different environments
    environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
    
    for env in environments:
        config_path = f"./config/{env.value}.yaml"
        manager.create_default_config(config_path, env)
        print(f"✅ Created {env.value} configuration: {config_path}")
    
    # Load and modify configuration
    config = manager.load_config("./config/development.yaml")
    config.training.batch_size = 16
    config.training.learning_rate = 1e-4
    config.model.name = "deepseek-ai/deepseek-r1"
    
    # Validate configuration
    issues = manager.validate_config(config)
    if issues:
        print(f"⚠️  Configuration issues: {issues}")
    else:
        print("✅ Configuration validation passed")
    
    # Display configuration
    manager.display_config(config)
    
    # Save modified configuration
    manager.save_config(config, "./config/custom.yaml")
    print("✅ Saved custom configuration")

def demonstrate_error_handling_and_logging():
    """Demonstrate error handling and logging features."""
    print("\n🛡️  Error Handling and Logging Demo")
    
    # Setup logging
    logger = setup_logging(
        log_dir="./logs",
        log_level=LogLevel.INFO,
        enable_console_logging=True,
        enable_file_logging=True,
        enable_sentry=False  # Set to True with valid DSN
    )
    
    logger.info("Starting error handling demonstration")
    
    # Demonstrate error logging
    try:
        # Simulate an error
        raise ValueError("This is a test error for demonstration")
    except Exception as e:
        logger.log_error(
            e, 
            ErrorType.UNKNOWN, 
            "demo", 
            "error_handling",
            {"demo_data": "example"}
        )
    
    # Demonstrate performance monitoring
    logger.start_performance_monitoring(interval=2.0)
    time.sleep(5)  # Let it collect some metrics
    logger.stop_performance_monitoring()
    
    # Get performance summary
    summary = logger.get_performance_summary()
    print(f"📊 Performance Summary: {summary}")
    
    print("✅ Error handling and logging demonstration completed")

def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring features."""
    print("\n📊 Performance Monitoring Demo")
    
    # Create metrics collector
    collector = create_metrics_collector(
        log_dir="./metrics",
        enable_tensorboard=True,
        enable_wandb=False,  # Set to True with valid project
        enable_mlflow=False  # Set to True with valid experiment
    )
    
    # Start system monitoring
    collector.start_monitoring(interval=1.0)
    
    # Simulate training metrics
    print("📈 Simulating training metrics...")
    for step in range(50):
        metrics = TrainingMetrics(
            step=step,
            epoch=step // 10,
            training_loss=1.0 - step * 0.02,
            validation_loss=1.2 - step * 0.015,
            learning_rate=0.001 * (0.9 ** (step // 10)),
            batch_time=0.1 + step * 0.001,
            throughput=100 - step * 0.5,
            memory_usage=50 + step * 0.2,
            gpu_memory_used=1000 + step * 10,
            gpu_memory_total=8000,
            gpu_utilization=60 + step * 0.3,
            cpu_usage=40 + step * 0.1
        )
        collector.log_training_metrics(metrics)
        
        if step % 10 == 0:
            print(f"  Step {step}: Loss={metrics.training_loss:.4f}, LR={metrics.learning_rate:.6f}")
        
        time.sleep(0.1)  # Simulate processing time
    
    # Stop monitoring
    collector.stop_monitoring()
    
    # Generate performance report
    report_path = collector.generate_report()
    print(f"📋 Performance report generated: {report_path}")
    
    # Get summary
    summary = collector.get_summary()
    print(f"📊 Training Summary: {summary}")
    
    print("✅ Performance monitoring demonstration completed")

def demonstrate_testing_framework():
    """Demonstrate testing framework features."""
    print("\n🧪 Testing Framework Demo")
    
    # Generate test files
    generator = TestGenerator(output_dir="./tests")
    generator.generate_all_tests()
    print("✅ Generated test files")
    
    # Create test runner
    runner = TestRunner(
        test_dir="./tests",
        output_dir="./test_results",
        enable_coverage=True,
        enable_html_report=True,
        enable_parallel=True,
        max_workers=2
    )
    
    # Create default test suites
    runner.create_default_test_suites()
    
    # Run tests
    print("🏃 Running tests...")
    results = runner.run_all_tests()
    
    # Display results
    if results['failed'] == 0 and results['errors'] == 0:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  Some tests failed: {results['failed']} failed, {results['errors']} errors")
    
    print("✅ Testing framework demonstration completed")

def demonstrate_deployment():
    """Demonstrate deployment features."""
    print("\n🚀 Deployment Demo")
    
    # Create deployment configuration
    deployment_config = DeploymentConfig(
        name="frontier-model-demo",
        deployment_type=DeploymentType.DOCKER,
        environment=Environment.DEVELOPMENT,
        image_name="frontier-model",
        image_tag="latest",
        replicas=1,
        cpu_limit="2",
        memory_limit="4Gi",
        cpu_request="1",
        memory_request="2Gi",
        gpu_enabled=False,
        ports=[8080],
        environment_vars={
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "INFO"
        }
    )
    
    # Create deployment manager
    manager = DeploymentManager()
    
    # Note: Actual deployment would require Docker/Kubernetes setup
    print("📋 Deployment configuration created:")
    print(f"  - Name: {deployment_config.name}")
    print(f"  - Type: {deployment_config.deployment_type.value}")
    print(f"  - Environment: {deployment_config.environment.value}")
    print(f"  - Image: {deployment_config.image_name}:{deployment_config.image_tag}")
    print(f"  - Resources: {deployment_config.cpu_request}/{deployment_config.memory_request}")
    
    print("✅ Deployment demonstration completed")

def demonstrate_integrated_training():
    """Demonstrate integrated training with all features."""
    print("\n🎯 Integrated Training Demo")
    
    # Setup logging
    logger = setup_logging(
        log_dir="./logs",
        log_level=LogLevel.INFO,
        enable_console_logging=True
    )
    
    # Setup monitoring
    collector = create_metrics_collector(
        log_dir="./metrics",
        enable_tensorboard=True
    )
    
    # Load configuration
    manager = ConfigManager()
    config = manager.load_config("./config/development.yaml")
    
    # Start monitoring
    collector.start_monitoring(interval=2.0)
    
    try:
        with error_context(ErrorType.TRAINING, "demo", "integrated_training") as (logger, handler):
            logger.info("Starting integrated training demonstration")
            
            # Simulate training loop
            for step in range(20):
                # Simulate training step
                loss = 1.0 - step * 0.05
                learning_rate = config.training.learning_rate * (0.9 ** (step // 5))
                
                # Create training metrics
                metrics = TrainingMetrics(
                    step=step,
                    epoch=step // 5,
                    training_loss=loss,
                    learning_rate=learning_rate,
                    batch_time=0.1,
                    throughput=100.0,
                    memory_usage=50 + step * 0.5,
                    gpu_memory_used=1000 + step * 20,
                    gpu_memory_total=8000,
                    gpu_utilization=60 + step * 1.5,
                    cpu_usage=40 + step * 0.5
                )
                
                # Log metrics
                collector.log_training_metrics(metrics)
                
                if step % 5 == 0:
                    logger.info(f"Step {step}: Loss={loss:.4f}, LR={learning_rate:.6f}")
                
                time.sleep(0.2)  # Simulate processing time
            
            logger.info("Integrated training demonstration completed successfully")
    
    except Exception as e:
        logger.log_error(e, ErrorType.TRAINING, "demo", "integrated_training")
        raise
    
    finally:
        # Stop monitoring and generate report
        collector.stop_monitoring()
        report_path = collector.generate_report()
        logger.info(f"Performance report generated: {report_path}")
        
        # Get final summary
        summary = collector.get_summary()
        logger.info(f"Final training summary: {summary}")
    
    print("✅ Integrated training demonstration completed")

def cleanup_demo():
    """Cleanup demo files and directories."""
    print("\n🧹 Cleaning up demo files...")
    
    # Remove temporary files
    temp_dirs = ["./logs", "./metrics", "./test_results", "./tests"]
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            print(f"🗑️  Removed: {temp_dir}")
    
    print("✅ Cleanup completed")

def main():
    """Main demonstration function."""
    print("🌟 Enhanced Frontier Model Training - Complete Demo")
    print("=" * 60)
    
    try:
        # Setup environment
        setup_environment()
        
        # Demonstrate all features
        demonstrate_configuration_management()
        demonstrate_error_handling_and_logging()
        demonstrate_performance_monitoring()
        demonstrate_testing_framework()
        demonstrate_deployment()
        demonstrate_integrated_training()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Review the generated configuration files in ./config/")
        print("2. Check the performance reports in ./metrics/")
        print("3. Examine the test results in ./test_results/")
        print("4. Review the logs in ./logs/")
        print("5. Customize configurations for your specific use case")
        print("6. Set up monitoring tools (Wandb, MLflow, Sentry)")
        print("7. Configure deployment targets (Docker, Kubernetes)")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return 1
    
    finally:
        # Ask user if they want to cleanup
        response = input("\n🧹 Do you want to cleanup demo files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_demo()
        else:
            print("📁 Demo files preserved for inspection")
    
    return 0

if __name__ == "__main__":
    exit(main())
