# Training Logging System Guide

## Overview

The training logging system provides comprehensive logging capabilities for all ads generation features, including fine-tuning, diffusion models, and tokenization. It offers real-time monitoring, error tracking, performance metrics, and historical analysis.

## Features

### 1. Multi-Output Logging
- **File Logging**: Rotated log files with timestamps
- **Console Logging**: Real-time output for development
- **Redis Logging**: Distributed access and persistence
- **TensorBoard Logging**: Visual metrics and training curves

### 2. Training Phases
- `DATA_PREPARATION`: Data loading and preprocessing
- `MODEL_LOADING`: Model initialization and caching
- `TRAINING`: Active training process
- `VALIDATION`: Model validation and evaluation
- `EVALUATION`: Performance assessment
- `INFERENCE`: Model inference and generation
- `CLEANUP`: Resource cleanup and finalization

### 3. Metrics Tracking
- Training loss and accuracy
- Learning rate progression
- Gradient norms
- Memory usage
- Generation time
- Custom metrics

### 4. Error Handling
- Detailed error logging with context
- Stack traces and debugging information
- Error categorization by phase
- Error history and analysis

## Usage

### Basic Training Logger

```python
from onyx.server.features.ads.training_logger import TrainingLogger, TrainingPhase

# Initialize logger
logger = TrainingLogger(
    user_id=123,
    model_name="microsoft/DialoGPT-medium",
    log_dir="logs/training/user_123"
)

# Start training session
logger.start_training(total_epochs=10, total_steps=1000)

# Log progress
logger.update_progress(
    epoch=1,
    step=100,
    loss=0.5,
    learning_rate=0.001,
    accuracy=0.85
)

# Log errors
try:
    # Training code
    pass
except Exception as e:
    logger.log_error(e, TrainingPhase.TRAINING, {"context": "additional info"})

# End training
logger.end_training("completed")
```

### Async Training Logger

```python
from onyx.server.features.ads.training_logger import AsyncTrainingLogger

# Initialize async logger
logger = AsyncTrainingLogger(
    user_id=123,
    model_name="stable-diffusion-v1-5",
    log_dir="logs/diffusion/user_123"
)

# Async logging
await logger.log_async("Starting image generation", TrainingPhase.INFERENCE)
await logger.update_progress_async(epoch=1, step=50, loss=0.3, learning_rate=0.001)

# Periodic log saving
await logger.save_logs_periodically(interval=60)  # Save every minute
```

### Integration with Services

#### Fine-tuning Service
```python
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService

service = OptimizedFineTuningService()

# Training automatically includes logging
results = await service.fine_tune_lora(
    user_id=123,
    base_model_name="microsoft/DialoGPT-medium",
    epochs=3,
    batch_size=4
)
```

#### Diffusion Service
```python
from onyx.server.features.ads.diffusion_service import DiffusionService

service = DiffusionService()

# Image generation with logging
images = await service.generate_text_to_image(
    params=generation_params,
    model_name="runwayml/stable-diffusion-v1-5",
    user_id=123  # Enables logging
)
```

#### Tokenization Service
```python
from onyx.server.features.ads.tokenization_service import TokenizationService

service = TokenizationService()

# Tokenization with logging
tokenized_data = await service.tokenize_ads_data(
    ads_data=data,
    max_length=512,
    user_id=123  # Enables logging
)
```

## API Endpoints

### Training Statistics
```bash
GET /training-logs/stats/{user_id}?model_name=optional_model
```

Response:
```json
{
  "user_id": 123,
  "model_name": "microsoft/DialoGPT-medium",
  "total_training_sessions": 5,
  "successful_sessions": 4,
  "failed_sessions": 1,
  "total_training_time": 3600.5,
  "avg_training_time": 720.1,
  "total_errors": 3,
  "recent_metrics": {
    "final_loss": 0.15,
    "final_accuracy": 0.92
  },
  "last_training_date": "2024-01-15T10:30:00"
}
```

### Training Progress
```bash
GET /training-logs/progress/{user_id}/{model_name}
```

Response:
```json
{
  "user_id": 123,
  "model_name": "microsoft/DialoGPT-medium",
  "current_epoch": 2,
  "total_epochs": 5,
  "current_step": 150,
  "total_steps": 500,
  "progress_percentage": 30.0,
  "elapsed_time": 1800.5,
  "estimated_completion": "2024-01-15T11:00:00",
  "status": "running",
  "recent_loss": 0.25,
  "recent_accuracy": 0.88,
  "recent_learning_rate": 0.001
}
```

### Training Metrics
```bash
GET /training-logs/metrics/{user_id}/{model_name}?limit=100
```

Response:
```json
{
  "user_id": 123,
  "model_name": "microsoft/DialoGPT-medium",
  "metrics": [
    {
      "epoch": 1,
      "step": 50,
      "loss": 0.5,
      "learning_rate": 0.001,
      "accuracy": 0.85,
      "timestamp": "2024-01-15T10:00:00"
    }
  ],
  "summary": {
    "total_metrics": 100,
    "avg_loss": 0.3,
    "min_loss": 0.15,
    "max_loss": 0.8,
    "avg_accuracy": 0.88
  }
}
```

### Error Logs
```bash
GET /training-logs/errors/{user_id}?model_name=optional&limit=50
```

Response:
```json
[
  {
    "timestamp": "2024-01-15T10:30:00",
    "error_type": "CUDAOutOfMemoryError",
    "error_message": "GPU memory exhausted",
    "phase": "training",
    "user_id": 123,
    "model_name": "microsoft/DialoGPT-medium",
    "traceback": "Full stack trace...",
    "context": {
      "batch_size": 8,
      "model_size": "large"
    }
  }
]
```

### Filtered Logs
```bash
POST /training-logs/logs/filter
```

Request:
```json
{
  "user_id": 123,
  "model_name": "microsoft/DialoGPT-medium",
  "start_date": "2024-01-15T00:00:00",
  "end_date": "2024-01-15T23:59:59",
  "phase": "training",
  "log_level": "INFO",
  "limit": 100,
  "offset": 0
}
```

## Configuration

### Logger Configuration
```python
# Basic configuration
logger = TrainingLogger(
    log_dir="logs/training",
    user_id=123,
    model_name="model_name",
    enable_tensorboard=True,
    enable_file_logging=True,
    enable_console_logging=True,
    enable_redis_logging=True,
    max_log_files=10,
    log_interval=10
)
```

### Environment Variables
```bash
# Redis configuration
REDIS_URL=redis://localhost:6379

# Logging configuration
LOG_LEVEL=INFO
LOG_DIR=logs
MAX_LOG_FILES=10
LOG_INTERVAL=10
```

## Monitoring and Debugging

### Real-time Monitoring
```python
# Get current training stats
stats = await training_logs_service.get_training_stats(user_id=123)
print(f"Progress: {stats.progress_percentage}%")

# Monitor errors
errors = await training_logs_service.get_error_logs(user_id=123)
for error in errors:
    print(f"Error: {error.error_type} - {error.error_message}")
```

### Performance Analysis
```python
# Get training metrics
metrics = await training_logs_service.get_training_metrics(user_id=123, model_name="model")
print(f"Average loss: {metrics.summary['avg_loss']}")

# Create training plots
plots = logger.create_training_plots(save_path="training_plots.png")
```

### Log Analysis
```python
# Get logs for specific time period
logs = await training_logs_service.get_file_logs(
    user_id=123,
    start_date=datetime.now() - timedelta(hours=1),
    end_date=datetime.now()
)

# Filter by phase
training_logs = [log for log in logs if log.phase == "training"]
```

## Best Practices

### 1. Error Handling
```python
try:
    # Training code
    pass
except Exception as e:
    # Log with context
    logger.log_error(e, TrainingPhase.TRAINING, {
        "user_id": user_id,
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })
    raise
```

### 2. Progress Logging
```python
# Log at regular intervals
if step % log_interval == 0:
    logger.update_progress(
        epoch=epoch,
        step=step,
        loss=loss,
        learning_rate=learning_rate,
        accuracy=accuracy,
        gradient_norm=gradient_norm
    )
```

### 3. Resource Management
```python
# Clean up old logs periodically
await training_logs_service.cleanup_old_logs(days=7)

# Close logger properly
logger.close()
```

### 4. Performance Monitoring
```python
# Monitor memory usage
memory_usage = logger.memory_tracker.get_memory_usage()
if memory_usage > threshold:
    logger.log_warning(f"High memory usage: {memory_usage}MB")

# Track generation time
start_time = time.time()
result = model.generate()
generation_time = time.time() - start_time
logger.log_info(f"Generation completed in {generation_time:.2f}s")
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Check memory usage in logs
   - Reduce batch size or model size
   - Enable gradient checkpointing

2. **Training Stalls**
   - Monitor learning rate progression
   - Check for gradient explosion/vanishing
   - Verify data pipeline

3. **Poor Performance**
   - Analyze loss curves
   - Check for overfitting/underfitting
   - Review hyperparameters

4. **Redis Connection Issues**
   - Verify Redis server is running
   - Check connection URL
   - Monitor Redis memory usage

### Debug Mode
```python
# Enable debug logging
logger = TrainingLogger(
    enable_console_logging=True,
    log_interval=1  # Log every step
)

# Enable detailed error logging
logger.log_debug("Detailed debug information")
```

### Log File Locations
- Training logs: `logs/training/user_{user_id}/`
- Diffusion logs: `logs/diffusion/user_{user_id}/`
- Tokenization logs: `logs/tokenization/user_{user_id}/`
- TensorBoard logs: `logs/training/tensorboard/`

## Integration Examples

### Custom Training Loop
```python
async def custom_training_loop(user_id: int, model_name: str):
    logger = AsyncTrainingLogger(user_id=user_id, model_name=model_name)
    
    try:
        logger.start_training(total_epochs=10, total_steps=1000)
        
        for epoch in range(10):
            for step in range(100):
                # Training step
                loss = train_step()
                
                # Log progress
                await logger.update_progress_async(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    learning_rate=0.001
                )
        
        logger.end_training("completed")
        
    except Exception as e:
        logger.log_error(e, TrainingPhase.TRAINING)
        logger.end_training("failed")
        raise
```

### Batch Processing
```python
async def batch_process_with_logging(user_id: int, items: List[Dict]):
    logger = AsyncTrainingLogger(user_id=user_id, model_name="batch_processor")
    
    logger.log_info(f"Starting batch processing of {len(items)} items")
    
    for i, item in enumerate(items):
        try:
            # Process item
            result = await process_item(item)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.log_info(f"Processed {i + 1}/{len(items)} items")
                
        except Exception as e:
            logger.log_error(e, TrainingPhase.DATA_PREPARATION, {"item_index": i})
            continue
    
    logger.log_info("Batch processing completed")
```

This comprehensive logging system provides full visibility into training processes, enabling effective monitoring, debugging, and optimization of ads generation features. 