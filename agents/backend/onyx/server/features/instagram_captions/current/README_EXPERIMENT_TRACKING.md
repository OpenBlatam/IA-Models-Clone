# Experiment Tracking and Model Checkpointing System

## Overview

The Experiment Tracking and Model Checkpointing System implements **Key Convention 4: "Implement proper experiment tracking and model checkpointing"** from the NLP system requirements. This system provides comprehensive experiment tracking, model checkpointing, and integration with popular tracking tools like TensorBoard and Weights & Biases.

## Key Features

### 🔬 **Experiment Tracking**
- **Metadata Management**: Track experiment names, descriptions, hyperparameters, and tags
- **Environment Tracking**: Automatically capture Python version, PyTorch version, CUDA info, and platform
- **Status Management**: Monitor experiment status (running, completed, failed, paused)
- **Git Integration**: Track git commit hashes for reproducibility

### 💾 **Model Checkpointing**
- **State Preservation**: Save model state, optimizer state, and scheduler state
- **Compression**: Automatic checkpoint compression to save disk space
- **Metadata Tracking**: Track checkpoint metrics, timestamps, and experiment associations
- **Cleanup Management**: Automatic cleanup of old checkpoints to maintain limits

### 📊 **Metrics Logging**
- **Multi-Platform Logging**: Log metrics to local files, TensorBoard, and W&B simultaneously
- **Step Tracking**: Associate metrics with training steps and epochs
- **History Management**: Maintain complete metric history for analysis
- **Real-time Monitoring**: Live metric tracking during training

### 🎯 **Integration Support**
- **TensorBoard**: Native TensorBoard integration for experiment visualization
- **Weights & Biases**: Full W&B integration for cloud-based experiment tracking
- **Model Artifacts**: Automatic model logging to W&B for version control
- **Cross-Platform**: Works with any PyTorch model and optimizer

## System Architecture

```
ExperimentTrackingSystem
├── ExperimentTracker        # Core experiment tracking
├── ModelCheckpointer       # Checkpoint management
├── TensorboardTracker      # TensorBoard integration
└── WandbTracker           # W&B integration
```

### Core Components

#### ExperimentTracker
- Experiment lifecycle management
- Metadata storage and retrieval
- Metrics logging and history
- Environment information capture

#### ModelCheckpointer
- Checkpoint saving and loading
- Compression and decompression
- Metadata management
- Automatic cleanup

#### TensorboardTracker
- TensorBoard writer management
- Metric logging to TensorBoard
- Model graph visualization
- Cross-platform compatibility

#### WandbTracker
- W&B run management
- Cloud-based experiment tracking
- Model artifact logging
- Configuration synchronization

## Installation

### Prerequisites
1. **Python 3.7+**: Required for dataclasses support
2. **PyTorch**: Core deep learning framework
3. **Internet Connection**: For W&B integration (optional)

### Dependencies
```bash
pip install -r requirements_experiment_tracking.txt
```

### Optional Dependencies
```bash
# For TensorBoard integration
pip install tensorboard

# For Weights & Biases integration
pip install wandb
```

### Setup
```python
from experiment_tracking_checkpointing_system import (
    ExperimentTrackingSystem, 
    ExperimentConfig
)

# Create configuration
config = ExperimentConfig(
    experiment_dir="./experiments",
    checkpoint_dir="./checkpoints",
    logs_dir="./logs",
    metrics_dir="./metrics",
    tensorboard_dir="./runs",
    wandb_project="my-nlp-project",
    save_frequency=1000,
    max_checkpoints=5,
    compression=True
)

# Initialize system
tracking_system = ExperimentTrackingSystem(config)
```

## Usage Examples

### Basic Experiment Tracking

```python
# Start experiment
experiment_id = tracking_system.start_experiment(
    name="bert-fine-tuning",
    description="Fine-tune BERT model on custom dataset",
    hyperparameters={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "warmup_steps": 500
    },
    model_config={
        "model_type": "bert",
        "hidden_size": 768,
        "num_layers": 12
    },
    dataset_info={
        "name": "custom-dataset",
        "size": 50000,
        "vocab_size": 30000
    },
    tags=["bert", "fine-tuning", "nlp"]
)

# Log metrics during training
for step in range(1000):
    metrics = {
        "loss": current_loss,
        "accuracy": current_accuracy,
        "learning_rate": current_lr
    }
    tracking_system.log_metrics(metrics, step)
    
    # Save checkpoint periodically
    if step % 100 == 0:
        checkpoint_path = tracking_system.save_checkpoint(
            model, optimizer, scheduler, epoch=0, step=step,
            metrics=metrics, is_best=(step == 0)
        )

# End experiment
tracking_system.end_experiment({
    "final_loss": final_loss,
    "final_accuracy": final_accuracy
})
```

### Checkpoint Management

```python
# Save checkpoint
checkpoint_path = tracking_system.save_checkpoint(
    model, optimizer, scheduler,
    epoch=epoch, step=step,
    metrics=metrics,
    is_best=is_best_model
)

# Load checkpoint
checkpoint_info = tracking_system.load_checkpoint(
    checkpoint_path, model, optimizer, scheduler, device="cuda"
)

# List available checkpoints
checkpoints = tracking_system.checkpointer.list_checkpoints(experiment_id)
for checkpoint in checkpoints:
    print(f"Step {checkpoint['step']}: {checkpoint['filename']}")

# Get best checkpoint
best_checkpoint = tracking_system.checkpointer.get_best_checkpoint(experiment_id)
```

### Advanced Integration

```python
# TensorBoard integration (automatic)
# Metrics are automatically logged to TensorBoard
# View with: tensorboard --logdir=./runs

# W&B integration (automatic)
# Experiments are automatically logged to W&B
# Models are logged as artifacts

# Custom metric logging
custom_metrics = {
    "custom_metric": custom_value,
    "gradient_norm": torch.norm(torch.stack([p.grad.norm() for p in model.parameters()])),
    "memory_usage": torch.cuda.memory_allocated() / 1024**3  # GB
}
tracking_system.log_metrics(custom_metrics, step)
```

## Configuration Options

### ExperimentConfig Parameters

```python
@dataclass
class ExperimentConfig:
    experiment_dir: str = "./experiments"      # Experiment storage directory
    checkpoint_dir: str = "./checkpoints"      # Checkpoint storage directory
    logs_dir: str = "./logs"                   # Log files directory
    metrics_dir: str = "./metrics"             # Metrics storage directory
    tensorboard_dir: str = "./runs"            # TensorBoard logs directory
    wandb_project: Optional[str] = None        # W&B project name
    save_frequency: int = 1000                 # Checkpoint save frequency
    max_checkpoints: int = 5                   # Maximum checkpoints to keep
    compression: bool = True                   # Enable checkpoint compression
```

### Directory Structure

```
project/
├── experiments/
│   └── experiment_name_timestamp/
│       ├── metadata.json          # Experiment metadata
│       ├── metrics.json           # Metric history
│       ├── summary.json           # Final summary
│       └── experiment.log         # Experiment log
├── checkpoints/
│   ├── checkpoints.json          # Checkpoint metadata
│   ├── checkpoint_timestamp_step.pt.gz
│   └── checkpoint_timestamp_step_best.pt.gz
├── logs/                         # Log files
├── metrics/                      # Metrics storage
└── runs/                         # TensorBoard logs
```

## Best Practices

### 1. Experiment Naming
- Use descriptive names that reflect the experiment purpose
- Include key hyperparameters in the name
- Use consistent naming conventions across team

### 2. Checkpoint Strategy
- Save checkpoints at regular intervals (e.g., every 1000 steps)
- Mark best checkpoints based on validation metrics
- Use compression to save disk space
- Set appropriate max_checkpoints limit

### 3. Metric Logging
- Log all relevant metrics (loss, accuracy, learning rate)
- Include custom metrics specific to your task
- Log metrics at consistent intervals
- Use meaningful metric names

### 4. Hyperparameter Tracking
- Log all hyperparameters that affect results
- Include model architecture details
- Track dataset information and statistics
- Document any preprocessing steps

## Integration with NLP System

The Experiment Tracking System integrates seamlessly with the broader NLP system:

### Configuration Management
- Uses the same configuration patterns
- Integrates with the configuration validation system
- Supports environment-specific settings

### Version Control
- Experiments are version-controlled
- Checkpoints are tracked with metadata
- Reproducibility through git commit tracking

### Modular Architecture
- Follows the same modular design principles
- Easy to extend with new tracking tools
- Consistent interface patterns

## Performance Considerations

### Checkpoint Optimization
- **Compression**: Reduces disk usage by 60-80%
- **Cleanup**: Automatic removal of old checkpoints
- **Metadata**: Efficient checkpoint indexing and search

### Memory Management
- **Streaming**: Metrics are written incrementally
- **Lazy Loading**: Checkpoints loaded only when needed
- **Efficient Storage**: Optimized data structures for large experiments

### Scalability
- **Multiple Experiments**: Support for concurrent experiments
- **Large Models**: Efficient handling of large model checkpoints
- **Long Training**: Support for extended training runs

## Error Handling

The system includes comprehensive error handling:

### Checkpoint Failures
- Graceful handling of save/load failures
- Automatic retry mechanisms
- Clear error messages and logging

### Integration Failures
- Fallback when external tools unavailable
- Graceful degradation of features
- Comprehensive error reporting

### Data Corruption
- Checkpoint validation
- Automatic backup creation
- Recovery mechanisms

## Troubleshooting

### Common Issues

1. **Checkpoint Save Fails**
   ```
   Error: No space left on device
   Solution: Enable compression, reduce max_checkpoints, or free disk space
   ```

2. **TensorBoard Not Working**
   ```
   Error: TensorBoard not available
   Solution: Install with: pip install tensorboard
   ```

3. **W&B Integration Fails**
   ```
   Error: W&B authentication failed
   Solution: Run: wandb login
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize system with debug logging
tracking_system = ExperimentTrackingSystem(config)
```

## Future Enhancements

### Planned Features
- **Distributed Training**: Support for multi-GPU and distributed training
- **Hyperparameter Optimization**: Integration with Optuna and Ray Tune
- **Model Versioning**: Semantic versioning for model artifacts
- **Automated Analysis**: Automatic experiment analysis and reporting

### Extension Points
- **Custom Trackers**: Easy addition of new tracking tools
- **Plugin System**: Extensible experiment tracking features
- **API Integration**: REST API for external tools
- **Web Dashboard**: Web-based experiment management interface

## Contributing

### Adding New Trackers
1. Create new tracker class
2. Implement required methods
3. Add to ExperimentTrackingSystem
4. Update configuration and tests

### Extending Functionality
1. Follow existing patterns
2. Add comprehensive error handling
3. Include type hints and documentation
4. Add unit tests for new features

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Include comprehensive docstrings
- Add type hints for all functions

## License

This experiment tracking system is part of the NLP System and follows the same licensing terms.

## Conclusion

The Experiment Tracking and Model Checkpointing System provides a comprehensive, production-ready solution for tracking experiments and managing model checkpoints. It implements all requirements from Key Convention 4 and integrates seamlessly with popular tools like TensorBoard and Weights & Biases.

Key benefits:
- **Complete Tracking**: Full experiment lifecycle management
- **Efficient Checkpointing**: Optimized checkpoint storage and retrieval
- **Multi-Platform**: Integration with TensorBoard and W&B
- **Production Ready**: Robust error handling and performance optimization
- **Extensible Architecture**: Easy to extend and customize

For questions and support, refer to the main NLP system documentation or the experiment tracking system logs.


