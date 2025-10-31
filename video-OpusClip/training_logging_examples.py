#!/usr/bin/env python3
"""
Training Logging Examples for Video-OpusClip

Practical examples demonstrating comprehensive training logging:
- Training progress tracking
- Error logging and recovery
- Metrics collection and visualization
- Performance monitoring
- Checkpointing and model saving
"""

import sys
import os
import time
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_example_environment():
    """Setup environment for training examples."""
    print("🔧 Setting up training logging examples...")
    
    # Create example directories
    os.makedirs("example_training_logs", exist_ok=True)
    os.makedirs("example_models", exist_ok=True)
    
    print("✅ Environment setup complete")

# =============================================================================
# EXAMPLE 1: BASIC TRAINING LOGGING
# =============================================================================

def example_basic_training_logging():
    """Demonstrate basic training logging functionality."""
    print("\n📊 Example 1: Basic Training Logging")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig
    
    # Create configuration
    config = TrainingConfig(
        log_dir="example_training_logs/basic_training",
        enable_tensorboard=False,
        enable_wandb=False,
        save_checkpoints=True
    )
    
    # Initialize logger
    logger = TrainingLogger(config)
    
    try:
        # Start training
        logger.start_training("basic_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5
        })
        
        # Simulate training loop
        for epoch in range(5):
            logger.log_epoch_start(epoch, 5)
            
            for step in range(100):
                # Simulate training step
                loss = 1.0 / (step + 1) + random.uniform(0, 0.1)
                accuracy = 0.8 + step * 0.001 + random.uniform(-0.01, 0.01)
                
                # Log step
                logger.log_step(
                    step=step,
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=0.001
                )
                
                # Simulate occasional errors
                if step == 50:
                    try:
                        raise RuntimeError("Simulated training error")
                    except Exception as e:
                        logger.log_error(e, step, epoch, "Training step")
            
            # Log epoch end
            avg_loss = sum([1.0 / (i + 1) for i in range(100)]) / 100
            logger.log_epoch_end(epoch, {
                "loss": avg_loss,
                "accuracy": 0.85
            })
        
        # Stop training
        logger.stop_training()
        print("✅ Basic training logging completed")
        
    except Exception as e:
        logger.log_error(e, context="Basic training example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 2: ERROR RECOVERY AND HANDLING
# =============================================================================

def example_error_recovery():
    """Demonstrate error recovery during training."""
    print("\n🚨 Example 2: Error Recovery and Handling")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig, TrainingEvent
    
    config = TrainingConfig(
        log_dir="example_training_logs/error_recovery",
        error_recovery_enabled=True,
        save_checkpoints=True
    )
    
    logger = TrainingLogger(config)
    
    try:
        logger.start_training("error_recovery_model", {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 3
        })
        
        for epoch in range(3):
            logger.log_epoch_start(epoch, 3)
            
            for step in range(50):
                try:
                    # Simulate different types of errors
                    if step == 10:
                        raise RuntimeError("CUDA out of memory")
                    elif step == 25:
                        raise ValueError("Invalid input data")
                    elif step == 40:
                        raise ConnectionError("Network timeout")
                    
                    # Normal training step
                    loss = 1.0 / (step + 1)
                    logger.log_step(step, loss)
                    
                except Exception as e:
                    logger.log_error(e, step, epoch, "Training step")
                    
                    # Continue training after error
                    print(f"🔄 Continuing training after error: {e}")
            
            logger.log_epoch_end(epoch, {"loss": 0.1})
        
        logger.stop_training()
        print("✅ Error recovery example completed")
        
    except Exception as e:
        logger.log_error(e, context="Error recovery example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 3: COMPREHENSIVE METRICS LOGGING
# =============================================================================

def example_comprehensive_metrics():
    """Demonstrate comprehensive metrics logging."""
    print("\n📈 Example 3: Comprehensive Metrics Logging")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig, TrainingMetrics
    
    config = TrainingConfig(
        log_dir="example_training_logs/comprehensive_metrics",
        enable_tensorboard=True,
        performance_monitoring=True
    )
    
    logger = TrainingLogger(config)
    
    try:
        logger.start_training("comprehensive_model", {
            "learning_rate": 0.0001,
            "batch_size": 64,
            "epochs": 3
        })
        
        for epoch in range(3):
            logger.log_epoch_start(epoch, 3)
            
            for step in range(100):
                # Simulate comprehensive metrics
                loss = 1.0 / (step + 1) + random.uniform(0, 0.05)
                accuracy = 0.9 + step * 0.0005 + random.uniform(-0.005, 0.005)
                learning_rate = 0.0001 * (0.95 ** epoch)
                gradient_norm = 0.5 + random.uniform(-0.1, 0.1)
                
                # Create comprehensive metrics
                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=learning_rate,
                    gradient_norm=gradient_norm,
                    memory_usage=512 + random.uniform(-50, 50),  # MB
                    gpu_usage=80 + random.uniform(-10, 10),  # %
                    training_time=0.1 + random.uniform(0, 0.05)  # seconds
                )
                
                logger.log_metrics(metrics)
                
                # Log performance metrics
                if step % 20 == 0:
                    performance_metrics = {
                        "gpu_memory_used": 2048 + random.uniform(-100, 100),
                        "gpu_memory_allocated": 3072 + random.uniform(-200, 200),
                        "cpu_memory_used": 1024 + random.uniform(-50, 50),
                        "batch_time": 0.15 + random.uniform(0, 0.1),
                        "data_loading_time": 0.02 + random.uniform(0, 0.01)
                    }
                    logger.log_performance_metrics(performance_metrics)
                
                # Log memory usage
                if step % 25 == 0:
                    logger.log_memory_usage()
            
            # Validation
            val_metrics = {
                "loss": 0.05 + random.uniform(0, 0.02),
                "accuracy": 0.95 + random.uniform(-0.01, 0.01),
                "f1_score": 0.94 + random.uniform(-0.01, 0.01),
                "precision": 0.96 + random.uniform(-0.01, 0.01),
                "recall": 0.93 + random.uniform(-0.01, 0.01)
            }
            
            logger.log_validation(step, val_metrics)
            logger.log_epoch_end(epoch, val_metrics)
        
        logger.stop_training()
        print("✅ Comprehensive metrics logging completed")
        
    except Exception as e:
        logger.log_error(e, context="Comprehensive metrics example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 4: CHECKPOINTING AND MODEL SAVING
# =============================================================================

def example_checkpointing():
    """Demonstrate checkpointing and model saving."""
    print("\n💾 Example 4: Checkpointing and Model Saving")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig
    
    config = TrainingConfig(
        log_dir="example_training_logs/checkpointing",
        save_checkpoints=True,
        checkpoint_interval=25
    )
    
    logger = TrainingLogger(config)
    
    try:
        logger.start_training("checkpoint_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2
        })
        
        best_loss = float('inf')
        
        for epoch in range(2):
            logger.log_epoch_start(epoch, 2)
            
            for step in range(100):
                # Simulate training
                loss = 1.0 / (step + 1) + random.uniform(0, 0.1)
                logger.log_step(step, loss)
                
                # Regular checkpoints
                if step % 25 == 0:
                    model_state = {
                        "weights": f"model_weights_step_{step}",
                        "optimizer": f"optimizer_state_step_{step}",
                        "epoch": epoch,
                        "step": step
                    }
                    
                    logger.log_checkpoint(
                        model_state=model_state,
                        step=step,
                        metrics={"loss": loss}
                    )
                
                # Best model checkpoint
                if loss < best_loss:
                    best_loss = loss
                    best_model_state = {
                        "weights": f"best_model_weights",
                        "optimizer": f"best_optimizer_state",
                        "epoch": epoch,
                        "step": step
                    }
                    
                    logger.log_checkpoint(
                        model_state=best_model_state,
                        step=step,
                        metrics={"loss": loss, "best": True}
                    )
            
            # Save model at end of epoch
            epoch_model_path = f"example_models/model_epoch_{epoch}.pt"
            logger.log_model_save(
                model_path=epoch_model_path,
                metrics={"loss": best_loss, "epoch": epoch}
            )
            
            logger.log_epoch_end(epoch, {"loss": best_loss})
        
        # Save final model
        final_model_path = "example_models/final_model.pt"
        logger.log_model_save(
            model_path=final_model_path,
            metrics={"final_loss": best_loss}
        )
        
        logger.stop_training()
        print("✅ Checkpointing example completed")
        
    except Exception as e:
        logger.log_error(e, context="Checkpointing example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 5: HYPERPARAMETER LOGGING
# =============================================================================

def example_hyperparameter_logging():
    """Demonstrate hyperparameter logging and updates."""
    print("\n⚙️ Example 5: Hyperparameter Logging")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig
    
    config = TrainingConfig(
        log_dir="example_training_logs/hyperparameters"
    )
    
    logger = TrainingLogger(config)
    
    try:
        logger.start_training("hyperparameter_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 3
        })
        
        # Initial hyperparameters
        initial_hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": 0.0001,
            "momentum": 0.9
        }
        
        logger.log_hyperparameter_update(initial_hyperparams)
        
        for epoch in range(3):
            logger.log_epoch_start(epoch, 3)
            
            for step in range(50):
                loss = 1.0 / (step + 1)
                logger.log_step(step, loss)
                
                # Simulate hyperparameter updates
                if step == 20 and epoch == 0:
                    # Reduce learning rate
                    new_lr = 0.001 * 0.5
                    logger.log_hyperparameter_update({
                        "learning_rate": new_lr,
                        "reason": "Plateau detection"
                    })
                
                elif step == 30 and epoch == 1:
                    # Increase batch size
                    new_batch_size = 64
                    logger.log_hyperparameter_update({
                        "batch_size": new_batch_size,
                        "reason": "Memory optimization"
                    })
                
                elif step == 40 and epoch == 2:
                    # Change optimizer
                    logger.log_hyperparameter_update({
                        "optimizer": "SGD",
                        "momentum": 0.8,
                        "reason": "Convergence improvement"
                    })
            
            logger.log_epoch_end(epoch, {"loss": 0.1})
        
        logger.stop_training()
        print("✅ Hyperparameter logging example completed")
        
    except Exception as e:
        logger.log_error(e, context="Hyperparameter logging example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 6: TRAINING ANALYTICS AND REPORTING
# =============================================================================

def example_training_analytics():
    """Demonstrate training analytics and reporting."""
    print("\n📊 Example 6: Training Analytics and Reporting")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig
    
    config = TrainingConfig(
        log_dir="example_training_logs/analytics",
        metrics_history_size=1000
    )
    
    logger = TrainingLogger(config)
    
    try:
        logger.start_training("analytics_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2
        })
        
        # Generate training data
        for epoch in range(2):
            logger.log_epoch_start(epoch, 2)
            
            for step in range(100):
                loss = 1.0 / (step + 1) + random.uniform(0, 0.1)
                accuracy = 0.8 + step * 0.001 + random.uniform(-0.01, 0.01)
                
                logger.log_step(step, loss, accuracy=accuracy)
                
                # Simulate some errors
                if step in [25, 75]:
                    try:
                        raise RuntimeError(f"Simulated error at step {step}")
                    except Exception as e:
                        logger.log_error(e, step, epoch, "Training step")
            
            logger.log_epoch_end(epoch, {"loss": 0.1, "accuracy": 0.85})
        
        logger.stop_training()
        
        # Generate analytics
        print("\n📈 Training Analytics:")
        print("-" * 30)
        
        # Training summary
        summary = logger.get_training_summary()
        print(f"Training Status: {summary['status']}")
        print(f"Current Epoch: {summary['current_epoch']}")
        print(f"Current Step: {summary['current_step']}")
        print(f"Training Duration: {summary['training_duration']:.2f}s")
        print(f"Total Metrics: {summary['total_metrics']}")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Recent Loss: {summary['recent_loss']:.4f}")
        
        # Error analysis
        print("\n🚨 Error Analysis:")
        print("-" * 30)
        error_summary = logger.get_errors_summary()
        print(f"Total Errors: {error_summary['total_errors']}")
        print(f"Error Types: {error_summary['error_types']}")
        print(f"Recovery Success Rate: {error_summary['recovery_success_rate']:.2%}")
        
        # Metrics history
        print("\n📊 Metrics History (last 10):")
        print("-" * 30)
        recent_metrics = logger.get_metrics_history(limit=10)
        for metric in recent_metrics:
            print(f"Step {metric['step']}: Loss={metric['loss']:.4f}, Accuracy={metric.get('accuracy', 'N/A'):.4f}")
        
        # Export training data
        export_path = "example_training_logs/analytics/training_export.json"
        logger.export_training_logs(export_path)
        print(f"\n📤 Training data exported to: {export_path}")
        
        print("✅ Training analytics example completed")
        
    except Exception as e:
        logger.log_error(e, context="Training analytics example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 7: INTEGRATION WITH EXTERNAL SYSTEMS
# =============================================================================

def example_external_integration():
    """Demonstrate integration with external logging systems."""
    print("\n🔗 Example 7: External System Integration")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig
    
    # Simulate external logging system
    class ExternalLogger:
        def __init__(self):
            self.logs = []
        
        def log_metric(self, name, value, step):
            self.logs.append({"name": name, "value": value, "step": step})
        
        def log_event(self, event_type, message):
            self.logs.append({"event": event_type, "message": message})
    
    external_logger = ExternalLogger()
    
    config = TrainingConfig(
        log_dir="example_training_logs/external_integration",
        enable_tensorboard=True,
        enable_wandb=False
    )
    
    logger = TrainingLogger(config)
    
    try:
        logger.start_training("integration_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2
        })
        
        # Hook into logger for external integration
        original_log_metrics = logger.log_metrics
        
        def enhanced_log_metrics(metrics):
            # Call original method
            original_log_metrics(metrics)
            
            # Log to external system
            external_logger.log_metric("loss", metrics.loss, metrics.step)
            if metrics.accuracy:
                external_logger.log_metric("accuracy", metrics.accuracy, metrics.step)
        
        logger.log_metrics = enhanced_log_metrics
        
        # Training loop
        for epoch in range(2):
            logger.log_epoch_start(epoch, 2)
            
            for step in range(50):
                loss = 1.0 / (step + 1)
                accuracy = 0.8 + step * 0.001
                
                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    accuracy=accuracy
                )
                
                logger.log_metrics(metrics)
            
            logger.log_epoch_end(epoch, {"loss": 0.1, "accuracy": 0.85})
        
        logger.stop_training()
        
        # Show external logs
        print(f"\n📤 External System Logs ({len(external_logger.logs)} entries):")
        print("-" * 50)
        for log in external_logger.logs[-10:]:  # Show last 10
            print(f"{log}")
        
        print("✅ External integration example completed")
        
    except Exception as e:
        logger.log_error(e, context="External integration example")
    finally:
        logger.close()

# =============================================================================
# EXAMPLE 8: REAL-TIME MONITORING
# =============================================================================

def example_real_time_monitoring():
    """Demonstrate real-time training monitoring."""
    print("\n⏱️ Example 8: Real-Time Monitoring")
    print("=" * 50)
    
    from training_logger import TrainingLogger, TrainingConfig
    import threading
    import time
    
    config = TrainingConfig(
        log_dir="example_training_logs/real_time",
        performance_monitoring=True
    )
    
    logger = TrainingLogger(config)
    
    # Monitoring thread
    def monitor_training():
        while logger.is_training:
            # Get current status
            summary = logger.get_training_summary()
            
            if summary['status'] == 'Training':
                print(f"🔄 Epoch: {summary['current_epoch']}, "
                      f"Step: {summary['current_step']}, "
                      f"Loss: {summary['recent_loss']:.4f}, "
                      f"Duration: {summary['training_duration']:.1f}s")
            
            time.sleep(2)  # Update every 2 seconds
    
    try:
        logger.start_training("real_time_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2
        })
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_training, daemon=True)
        monitor_thread.start()
        
        # Training loop
        for epoch in range(2):
            logger.log_epoch_start(epoch, 2)
            
            for step in range(30):  # Shorter for demo
                loss = 1.0 / (step + 1) + random.uniform(0, 0.1)
                logger.log_step(step, loss)
                
                # Log memory usage
                if step % 10 == 0:
                    logger.log_memory_usage()
                
                time.sleep(0.1)  # Simulate training time
            
            logger.log_epoch_end(epoch, {"loss": 0.1})
        
        logger.stop_training()
        print("✅ Real-time monitoring example completed")
        
    except Exception as e:
        logger.log_error(e, context="Real-time monitoring example")
    finally:
        logger.close()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run all training logging examples."""
    print("🎯 Training Logging Examples for Video-OpusClip")
    print("=" * 60)
    print("This script demonstrates comprehensive training logging patterns")
    print("for progress tracking, error handling, and metrics collection.\n")
    
    # Setup environment
    setup_example_environment()
    
    # Run all examples
    examples = [
        example_basic_training_logging,
        example_error_recovery,
        example_comprehensive_metrics,
        example_checkpointing,
        example_hyperparameter_logging,
        example_training_analytics,
        example_external_integration,
        example_real_time_monitoring
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            print(f"\n{'='*60}")
            example()
            print(f"{'='*60}")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print(f"{'='*60}")
    
    print("\n🎉 All training logging examples completed!")
    print("\n📋 Summary:")
    print("• Basic training logging with progress tracking")
    print("• Error recovery and handling mechanisms")
    print("• Comprehensive metrics collection and visualization")
    print("• Checkpointing and model saving")
    print("• Hyperparameter logging and updates")
    print("• Training analytics and reporting")
    print("• External system integration")
    print("• Real-time monitoring capabilities")
    
    print(f"\n📁 Log files created in: example_training_logs/")
    print(f"📁 Model files created in: example_models/")

if __name__ == "__main__":
    main() 