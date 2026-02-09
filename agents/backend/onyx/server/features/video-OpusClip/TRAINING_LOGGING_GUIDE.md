# Training Logging Guide for Video-OpusClip

## Overview

This guide covers comprehensive logging implementation for training operations in the Video-OpusClip system, including progress tracking, error handling, metrics collection, and performance monitoring.

## 🎯 Training Logging Architecture

### **Core Components**

#### 1. **TrainingLogger**
Main logging class that handles all training-related logging operations.

#### 2. **TrainingMetrics**
Data structure for storing training metrics (loss, accuracy, learning rate, etc.).

#### 3. **TrainingEvent**
Data structure for logging training events (epoch start/end, checkpoints, etc.).

#### 4. **TrainingError**
Data structure for error logging with recovery tracking.

#### 5. **TrainingConfig**
Configuration class for customizing logging behavior.

## 🚀 Getting Started

### **Basic Setup**

```python
from training_logger import TrainingLogger, TrainingConfig

# Create configuration
config = TrainingConfig(
    log_dir="my_training_logs",
    enable_tensorboard=True,
    enable_wandb=False,
    save_checkpoints=True
)

# Initialize logger
logger = TrainingLogger(config)

# Start training
logger.start_training("my_model", {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
})
```

### **Complete Training Loop Example**

```python
def train_model_with_logging():
    """Complete training loop with comprehensive logging."""
    
    # Initialize logger
    config = TrainingConfig(
        log_dir="video_model_training",
        enable_tensorboard=True,
        save_checkpoints=True,
        error_recovery_enabled=True
    )
    
    logger = TrainingLogger(config)
    
    try:
        # Start training
        logger.start_training("video_generation_model", {
            "model_type": "transformer",
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 50
        })
        
        # Training loop
        for epoch in range(50):
            logger.log_epoch_start(epoch, 50)
            
            for step in range(1000):
                try:
                    # Training step
                    loss = train_step()
                    accuracy = calculate_accuracy()
                    
                    # Log step metrics
                    logger.log_step(
                        step=step,
                        loss=loss,
                        accuracy=accuracy,
                        learning_rate=current_lr
                    )
                    
                    # Log memory usage periodically
                    if step % 100 == 0:
                        logger.log_memory_usage()
                    
                    # Save checkpoint periodically
                    if step % 1000 == 0:
                        logger.log_checkpoint(
                            model_state=model.state_dict(),
                            step=step,
                            metrics={"loss": loss, "accuracy": accuracy}
                        )
                    
                except Exception as e:
                    logger.log_error(e, step, epoch, "Training step")
                    # Continue training or handle error
            
            # Validation
            val_metrics = validate_model()
            logger.log_validation(step, val_metrics)
            
            # Log epoch end
            logger.log_epoch_end(epoch, {
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"]
            })
        
        # Training completed
        logger.stop_training()
        
    except Exception as e:
        logger.log_error(e, context="Main training loop")
    finally:
        logger.close()
```

## 📊 Training Progress Logging

### **Epoch and Step Logging**

```python
def log_training_progress():
    """Log training progress at different levels."""
    
    logger = TrainingLogger()
    
    # Log epoch start
    logger.log_epoch_start(epoch=5, total_epochs=100)
    
    # Log individual steps
    for step in range(1000):
        loss = calculate_loss()
        accuracy = calculate_accuracy()
        
        logger.log_step(
            step=step,
            loss=loss,
            accuracy=accuracy,
            learning_rate=0.001,
            gradient_norm=0.5
        )
    
    # Log epoch end with summary metrics
    logger.log_epoch_end(epoch=5, metrics={
        "loss": 0.123,
        "accuracy": 0.95,
        "learning_rate": 0.001
    })
```

### **Validation Logging**

```python
def log_validation_results():
    """Log validation results during training."""
    
    logger = TrainingLogger()
    
    # Run validation
    val_loss = validate_model()
    val_accuracy = calculate_validation_accuracy()
    
    # Log validation metrics
    logger.log_validation(
        step=current_step,
        metrics={
            "loss": val_loss,
            "accuracy": val_accuracy,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall
        }
    )
```

### **Hyperparameter Logging**

```python
def log_hyperparameter_changes():
    """Log hyperparameter updates during training."""
    
    logger = TrainingLogger()
    
    # Log initial hyperparameters
    logger.log_hyperparameter_update({
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR"
    })
    
    # Log hyperparameter changes
    if should_reduce_lr:
        new_lr = current_lr * 0.1
        logger.log_hyperparameter_update({
            "learning_rate": new_lr,
            "reason": "Plateau detection"
        })
```

## 🚨 Error Logging and Recovery

### **Automatic Error Recovery**

```python
def training_with_error_recovery():
    """Training loop with automatic error recovery."""
    
    config = TrainingConfig(error_recovery_enabled=True)
    logger = TrainingLogger(config)
    
    for epoch in range(100):
        for step in range(1000):
            try:
                # Training step
                loss = train_step()
                logger.log_step(step, loss)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Logger will automatically attempt recovery
                    logger.log_error(e, step, epoch, "GPU memory error")
                    # Continue training
                else:
                    raise
            
            except Exception as e:
                # Log any other errors
                logger.log_error(e, step, epoch, "Training error")
                # Decide whether to continue or stop
```

### **Custom Error Recovery**

```python
def custom_error_recovery():
    """Implement custom error recovery strategies."""
    
    logger = TrainingLogger()
    
    def handle_training_error(error, step, epoch):
        """Custom error handler."""
        
        if isinstance(error, MemoryError):
            # Reduce batch size
            global batch_size
            batch_size = max(1, batch_size // 2)
            logger.log_hyperparameter_update({
                "batch_size": batch_size,
                "reason": "Memory error recovery"
            })
            
        elif isinstance(error, RuntimeError) and "gradient" in str(error):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            logger.log_event(TrainingEvent(
                event_type="GRADIENT_CLIPPING",
                message="Applied gradient clipping due to gradient error",
                severity="WARNING"
            ))
        
        # Log the error
        logger.log_error(error, step, epoch, "Custom recovery applied")
    
    # Use in training loop
    try:
        loss = train_step()
    except Exception as e:
        handle_training_error(e, current_step, current_epoch)
```

## 📈 Metrics and Performance Logging

### **Comprehensive Metrics Logging**

```python
def log_comprehensive_metrics():
    """Log comprehensive training metrics."""
    
    logger = TrainingLogger()
    
    # Training metrics
    training_metrics = TrainingMetrics(
        epoch=current_epoch,
        step=current_step,
        loss=loss,
        accuracy=accuracy,
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        memory_usage=memory_usage,
        gpu_usage=gpu_usage,
        training_time=step_time
    )
    
    logger.log_metrics(training_metrics)
    
    # Performance metrics
    performance_metrics = {
        "gpu_memory_used": gpu_memory_used,
        "gpu_memory_allocated": gpu_memory_allocated,
        "cpu_memory_used": cpu_memory_used,
        "batch_time": batch_time,
        "data_loading_time": data_loading_time,
        "forward_pass_time": forward_pass_time,
        "backward_pass_time": backward_pass_time
    }
    
    logger.log_performance_metrics(performance_metrics)
```

### **Memory Usage Monitoring**

```python
def monitor_memory_usage():
    """Monitor and log memory usage during training."""
    
    logger = TrainingLogger()
    
    # Log memory usage periodically
    if step % 100 == 0:
        logger.log_memory_usage()
    
    # Custom memory monitoring
    import psutil
    import torch
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        logger.log_performance_metrics({
            "system_memory_mb": memory_info.rss / (1024**2),
            "gpu_memory_allocated_gb": gpu_memory,
            "gpu_memory_reserved_gb": gpu_memory_reserved
        })
```

## 💾 Checkpointing and Model Saving

### **Automatic Checkpointing**

```python
def automatic_checkpointing():
    """Implement automatic checkpointing during training."""
    
    config = TrainingConfig(
        save_checkpoints=True,
        checkpoint_interval=1000
    )
    
    logger = TrainingLogger(config)
    
    for step in range(10000):
        # Training step
        loss = train_step()
        logger.log_step(step, loss)
        
        # Automatic checkpointing
        if step % config.checkpoint_interval == 0:
            logger.log_checkpoint(
                model_state=model.state_dict(),
                step=step,
                metrics={"loss": loss}
            )
```

### **Manual Model Saving**

```python
def manual_model_saving():
    """Manually save models with logging."""
    
    logger = TrainingLogger()
    
    # Save best model
    if is_best_model:
        model_path = f"models/best_model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), model_path)
        
        logger.log_model_save(
            model_path=model_path,
            metrics={
                "loss": best_loss,
                "accuracy": best_accuracy,
                "epoch": epoch
            }
        )
    
    # Save final model
    if epoch == final_epoch:
        final_model_path = "models/final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        
        logger.log_model_save(
            model_path=final_model_path,
            metrics=final_metrics
        )
```

## 🔍 Training Analytics and Reporting

### **Training Summary**

```python
def get_training_summary():
    """Get comprehensive training summary."""
    
    logger = TrainingLogger()
    
    # Get training summary
    summary = logger.get_training_summary()
    print(f"Training Status: {summary['status']}")
    print(f"Current Epoch: {summary['current_epoch']}")
    print(f"Current Step: {summary['current_step']}")
    print(f"Training Duration: {summary['training_duration']:.2f}s")
    print(f"Total Metrics: {summary['total_metrics']}")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Recent Loss: {summary['recent_loss']:.4f}")
```

### **Error Analysis**

```python
def analyze_training_errors():
    """Analyze training errors and recovery."""
    
    logger = TrainingLogger()
    
    # Get error summary
    error_summary = logger.get_errors_summary()
    
    print(f"Total Errors: {error_summary['total_errors']}")
    print(f"Error Types: {error_summary['error_types']}")
    print(f"Recovery Success Rate: {error_summary['recovery_success_rate']:.2%}")
    
    # Analyze recent errors
    for error in error_summary['recent_errors']:
        print(f"Error: {error['error_type']} - {error['error_message']}")
        print(f"Step: {error['step']}, Recovery: {error['recovery_successful']}")
```

### **Metrics History Analysis**

```python
def analyze_metrics_history():
    """Analyze training metrics history."""
    
    logger = TrainingLogger()
    
    # Get recent metrics
    recent_metrics = logger.get_metrics_history(limit=1000)
    
    # Calculate statistics
    losses = [m['loss'] for m in recent_metrics]
    accuracies = [m['accuracy'] for m in recent_metrics if m['accuracy'] is not None]
    
    print(f"Average Loss: {sum(losses) / len(losses):.4f}")
    print(f"Min Loss: {min(losses):.4f}")
    print(f"Max Loss: {max(losses):.4f}")
    
    if accuracies:
        print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")
        print(f"Best Accuracy: {max(accuracies):.4f}")
```

### **Export Training Logs**

```python
def export_training_data():
    """Export training logs for analysis."""
    
    logger = TrainingLogger()
    
    # Export all training data
    export_path = "training_export.json"
    logger.export_training_logs(export_path)
    
    print(f"Training data exported to: {export_path}")
    
    # Load and analyze exported data
    with open(export_path, 'r') as f:
        training_data = json.load(f)
    
    # Create visualizations
    import matplotlib.pyplot as plt
    
    metrics = training_data['metrics_history']
    steps = [m['step'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
```

## 🎛️ Advanced Configuration

### **Custom Training Configuration**

```python
def custom_training_config():
    """Create custom training configuration."""
    
    config = TrainingConfig(
        log_dir="custom_training_logs",
        log_level="DEBUG",
        max_log_files=20,
        log_file_size_mb=200,
        enable_tensorboard=True,
        enable_wandb=True,
        enable_progress_bars=True,
        save_checkpoints=True,
        checkpoint_interval=500,
        metrics_history_size=50000,
        error_recovery_enabled=True,
        performance_monitoring=True
    )
    
    logger = TrainingLogger(config)
    return logger
```

### **External Logging Integration**

```python
def integrate_external_logging():
    """Integrate with external logging systems."""
    
    config = TrainingConfig(
        enable_tensorboard=True,
        enable_wandb=True
    )
    
    logger = TrainingLogger(config)
    
    # TensorBoard will automatically log metrics
    # Weights & Biases will automatically log experiments
    
    # Custom external logging
    def log_to_custom_system(metrics):
        # Log to your custom system
        pass
    
    # Hook into logger
    original_log_metrics = logger.log_metrics
    
    def enhanced_log_metrics(metrics):
        original_log_metrics(metrics)
        log_to_custom_system(metrics)
    
    logger.log_metrics = enhanced_log_metrics
```

## 🔧 Integration with Existing Systems

### **Integration with Video-OpusClip**

```python
def integrate_with_video_opusclip():
    """Integrate training logger with Video-OpusClip system."""
    
    from optimized_config import get_config
    from error_handling import ErrorHandler
    
    # Get system configuration
    system_config = get_config()
    
    # Create training logger with system settings
    config = TrainingConfig(
        log_dir=system_config.get("training_log_dir", "training_logs"),
        enable_tensorboard=system_config.get("enable_tensorboard", True),
        error_recovery_enabled=system_config.get("error_recovery_enabled", True)
    )
    
    logger = TrainingLogger(config)
    
    # Integrate with error handler
    error_handler = ErrorHandler()
    
    def enhanced_error_logging(error, step, epoch, context):
        # Log to training logger
        logger.log_error(error, step, epoch, context)
        
        # Also log to system error handler
        error_handler.handle_error(error, ErrorType.TRAINING, ErrorSeverity.MEDIUM)
    
    logger.log_error = enhanced_error_logging
    
    return logger
```

### **Integration with Debug Tools**

```python
def integrate_with_debug_tools():
    """Integrate with debug tools for enhanced monitoring."""
    
    from debug_tools import DebugManager
    
    debug_manager = DebugManager()
    logger = TrainingLogger()
    
    # Enhanced logging with debug information
    def enhanced_log_step(step, loss, **kwargs):
        # Regular logging
        logger.log_step(step, loss, **kwargs)
        
        # Debug logging
        debug_manager.memory_analyzer.take_snapshot(f"training_step_{step}")
        
        # Performance profiling
        debug_manager.profiler.start_profile(f"training_step_{step}")
        # ... training step ...
        debug_manager.profiler.end_profile(f"training_step_{step}")
    
    logger.log_step = enhanced_log_step
```

## 📋 Best Practices

### **1. Comprehensive Error Handling**
```python
def robust_training_loop():
    """Robust training loop with comprehensive error handling."""
    
    logger = TrainingLogger()
    
    try:
        logger.start_training("robust_model")
        
        for epoch in range(epochs):
            try:
                logger.log_epoch_start(epoch, epochs)
                
                for step in range(steps_per_epoch):
                    try:
                        loss = train_step()
                        logger.log_step(step, loss)
                        
                    except Exception as e:
                        logger.log_error(e, step, epoch, "Training step")
                        # Continue training or implement recovery
                
                logger.log_epoch_end(epoch, epoch_metrics)
                
            except Exception as e:
                logger.log_error(e, epoch=epoch, context="Epoch training")
                # Handle epoch-level errors
        
        logger.stop_training()
        
    except Exception as e:
        logger.log_error(e, context="Main training loop")
    finally:
        logger.close()
```

### **2. Regular Checkpointing**
```python
def regular_checkpointing():
    """Implement regular checkpointing strategy."""
    
    logger = TrainingLogger()
    
    checkpoint_interval = 1000
    best_loss = float('inf')
    
    for step in range(total_steps):
        loss = train_step()
        logger.log_step(step, loss)
        
        # Regular checkpoints
        if step % checkpoint_interval == 0:
            logger.log_checkpoint(model.state_dict(), step, {"loss": loss})
        
        # Best model checkpoints
        if loss < best_loss:
            best_loss = loss
            logger.log_checkpoint(
                model.state_dict(), 
                step, 
                {"loss": loss, "best": True}
            )
```

### **3. Performance Monitoring**
```python
def performance_monitoring():
    """Monitor performance during training."""
    
    logger = TrainingLogger()
    
    for step in range(total_steps):
        start_time = time.time()
        
        # Training step
        loss = train_step()
        
        step_time = time.time() - start_time
        
        # Log with performance metrics
        logger.log_step(
            step=step,
            loss=loss,
            training_time=step_time
        )
        
        # Monitor memory every 100 steps
        if step % 100 == 0:
            logger.log_memory_usage()
```

## 🎯 Conclusion

The comprehensive training logging system for Video-OpusClip provides:

1. **📊 Detailed Progress Tracking**: Monitor every aspect of training
2. **🚨 Robust Error Handling**: Automatic recovery and detailed error logging
3. **📈 Performance Monitoring**: Memory, GPU, and timing metrics
4. **💾 Automatic Checkpointing**: Never lose training progress
5. **🔍 Comprehensive Analytics**: Detailed training analysis and reporting
6. **🎛️ Flexible Configuration**: Customize logging for your needs
7. **🔗 External Integration**: TensorBoard, Weights & Biases, custom systems

By following this guide, you can implement robust, comprehensive logging for all training operations in the Video-OpusClip system, ensuring reliable training progress tracking and error recovery. 