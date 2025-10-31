# 🔬 Experiment Tracking System

## Overview

The Experiment Tracking System provides comprehensive monitoring and visualization capabilities for deep learning experiments, integrating both TensorBoard and Weights & Biases (wandb). This system is specifically designed for numerical stability research and training experiments.

## 🚀 Features

### Core Capabilities
- **Unified Logging**: Single interface for both TensorBoard and Weights & Biases
- **Real-time Monitoring**: Live tracking of training metrics and numerical stability
- **Comprehensive Metrics**: Loss, accuracy, gradient norms, NaN/Inf counts, clipping statistics
- **Visualization**: Automatic plot generation and logging to tracking systems
- **Checkpointing**: Save and restore model states with metadata
- **Asynchronous Logging**: Non-blocking metric processing for optimal performance

### Tracking Systems
- **TensorBoard**: Local visualization and analysis
- **Weights & Biases**: Cloud-based experiment tracking and collaboration
- **Hybrid Mode**: Use both systems simultaneously with automatic synchronization

### Numerical Stability Focus
- **Gradient Monitoring**: Track gradient norms and clipping statistics
- **NaN/Inf Detection**: Monitor numerical stability issues in real-time
- **Stability Metrics**: Comprehensive logging of stability-related events
- **Performance Tracking**: Memory usage, GPU utilization, training time

## 📁 File Structure

```
experiment_tracking.py              # Core experiment tracking system
gradio_experiment_tracking.py       # Interactive Gradio interface
requirements_gradio.txt             # Dependencies including tensorboard and wandb
```

## 🛠️ Installation

### Prerequisites
```bash
pip install torch torchvision
pip install tensorboard>=2.13.0
pip install wandb>=0.15.0
pip install gradio matplotlib seaborn numpy
```

### Optional Dependencies
```bash
pip install psutil  # For system monitoring
```

## 🎯 Quick Start

### 1. Basic Usage

```python
from experiment_tracking import ExperimentTracker, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    experiment_name="my_experiment",
    project_name="numerical_stability_research",
    enable_tensorboard=True,
    enable_wandb=True
)

# Create tracker
tracker = ExperimentTracker(config)

# Log hyperparameters
tracker.log_hyperparameters({
    'learning_rate': 0.001,
    'batch_size': 32,
    'max_grad_norm': 1.0
})

# Log training step
tracker.log_training_step(
    loss=0.5,
    accuracy=0.85,
    gradient_norm=0.8,
    nan_count=0,
    inf_count=0
)

# Close tracker
tracker.close()
```

### 2. Training Loop Integration

```python
# In your training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Get gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Log training step
        tracker.log_training_step(
            loss=loss.item(),
            accuracy=calculate_accuracy(output, target),
            gradient_norm=grad_norm,
            nan_count=count_nans(model),
            inf_count=count_infs(model)
        )
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    # Log epoch metrics
    tracker.log_epoch(epoch, {
        'epoch_loss': epoch_loss,
        'epoch_accuracy': epoch_accuracy
    })
```

## 🔧 Configuration

### ExperimentConfig Options

```python
@dataclass
class ExperimentConfig:
    # Basic settings
    experiment_name: str = "gradient_clipping_nan_handling"
    project_name: str = "blatam_academy_facebook_posts"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Tracking settings
    enable_tensorboard: bool = True
    enable_wandb: bool = True
    log_interval: int = 100
    save_interval: int = 1000
    
    # Logging settings
    log_metrics: bool = True
    log_hyperparameters: bool = True
    log_model_architecture: bool = True
    log_gradients: bool = True
    log_images: bool = True
    log_text: bool = True
    
    # File paths
    tensorboard_dir: str = "runs/tensorboard"
    model_save_dir: str = "models"
    config_save_dir: str = "configs"
    
    # Advanced settings
    sync_tensorboard: bool = True
    resume_run: bool = False
    anonymous: bool = False
```

### Default Configurations

The system provides three preset configurations:

1. **Basic Tracking**: TensorBoard only, minimal logging
2. **Full Tracking**: Both TensorBoard and wandb with comprehensive logging
3. **Numerical Stability Focus**: Optimized for stability research with detailed metrics

## 📊 Metrics and Logging

### Training Metrics

```python
@dataclass
class TrainingMetrics:
    loss: float = 0.0
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    nan_count: int = 0
    inf_count: int = 0
    clipping_applied: bool = False
    clipping_threshold: Optional[float] = None
    training_time: float = 0.0
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
```

### Logging Methods

- **`log_training_step()`**: Log individual training steps
- **`log_epoch()`**: Log epoch-level metrics
- **`log_model_architecture()`**: Log model structure and parameters
- **`log_hyperparameters()`**: Log training configuration
- **`log_gradients()`**: Log gradient distributions
- **`log_images()`**: Log images for visualization
- **`log_text()`**: Log text data and notes

## 🎨 Visualization

### Automatic Plot Generation

The system automatically creates comprehensive training visualizations:

1. **Training Loss**: Loss progression over time
2. **Accuracy**: Training accuracy trends
3. **Gradient Norms**: Gradient magnitude monitoring
4. **Numerical Stability**: NaN/Inf occurrence tracking

### Custom Visualizations

```python
# Create custom visualization
viz_data = tracker.create_visualization("training_progress.png")

# Access the matplotlib figure
figure = viz_data['figure']

# Get metrics summary
summary = viz_data['metrics_summary']
```

## 💾 Checkpointing

### Save Checkpoints

```python
tracker.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    loss=loss,
    metrics=metrics,
    scheduler=scheduler
)
```

### Load Checkpoints

```python
checkpoint_data = tracker.load_checkpoint("checkpoint_epoch_10_step_1000.pt")

# Restore model state
model.load_state_dict(checkpoint_data['model_state_dict'])
optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

# Get metadata
epoch = checkpoint_data['epoch']
step = checkpoint_data['step']
loss = checkpoint_data['loss']
```

## 🌐 Gradio Interface

### Launch the Interface

```bash
python gradio_experiment_tracking.py
```

### Interface Features

1. **Configuration Tab**: Set up experiment parameters
2. **Training Tab**: Start/stop training simulations
3. **Monitoring Tab**: View visualizations and launch TensorBoard

### Quick Actions

- **Create Experiment Tracker**: Initialize with current settings
- **Start Training Simulation**: Run demo training with real-time logging
- **Create Visualization**: Generate training progress plots
- **Launch TensorBoard**: Open TensorBoard for detailed analysis

## 🔍 TensorBoard Integration

### Launch TensorBoard

```python
# Launch from the Gradio interface
# Or manually:
import subprocess
subprocess.run(["tensorboard", "--logdir=runs/tensorboard", "--port=6006"])
```

### Available Visualizations

- **Scalars**: Loss, accuracy, gradient norms, stability metrics
- **Histograms**: Parameter distributions, gradient distributions
- **Text**: Hyperparameters, experiment notes
- **Images**: Training visualizations
- **Graphs**: Model architecture (when available)

## ☁️ Weights & Biases Integration

### Setup

```python
# The system automatically handles wandb initialization
# Make sure you have wandb installed and configured
import wandb
wandb login()
```

### Features

- **Automatic Run Creation**: Unique runs for each experiment
- **Metric Logging**: Real-time metric tracking
- **Hyperparameter Logging**: Automatic configuration capture
- **Model Versioning**: Track model changes over time
- **Collaboration**: Share experiments with team members

## 📈 Performance Optimization

### Asynchronous Logging

The system uses background threads for non-blocking metric processing:

```python
# Metrics are queued and processed asynchronously
tracker.metrics_queue.put(metrics)

# Processing happens in background thread
# No impact on training performance
```

### Memory Management

- **Efficient Storage**: Optimized data structures for large experiments
- **Automatic Cleanup**: Resource management and cleanup
- **Batch Processing**: Efficient handling of multiple metrics

## 🛡️ Error Handling

### Robust Error Management

```python
try:
    tracker.log_training_step(...)
except Exception as e:
    logger.error(f"Failed to log training step: {e}")
    # System continues to function
    # Errors are logged but don't crash training
```

### Recovery Mechanisms

- **Graceful Degradation**: System continues with reduced functionality
- **Error Logging**: Comprehensive error tracking and reporting
- **Automatic Retry**: Retry mechanisms for transient failures

## 🔧 Advanced Usage

### Custom Metrics

```python
# Log custom metrics
tracker.log_custom_metric("custom_metric", value, step)

# Log multiple metrics at once
tracker.log_metrics_batch({
    "metric1": value1,
    "metric2": value2
})
```

### Experiment Groups

```python
# Group related experiments
config = ExperimentConfig(
    experiment_name="ablation_study",
    wandb_group="learning_rate_ablation"
)
```

### Resume Experiments

```python
# Resume from previous run
config = ExperimentConfig(
    resume_run=True,
    run_id="previous_run_id"
)
```

## 📊 Monitoring and Debugging

### System Health

```python
# Get experiment summary
summary = tracker.get_experiment_summary()

# Check system status
status = tracker.get_system_status()

# Monitor resource usage
resources = tracker.get_resource_usage()
```

### Debugging Tools

- **Log Analysis**: Comprehensive logging for troubleshooting
- **Performance Profiling**: Monitor system performance
- **Error Tracking**: Detailed error reporting and context

## 🚀 Best Practices

### 1. Configuration Management

```python
# Use configuration files for reproducibility
config = ExperimentConfig.from_file("experiment_config.json")

# Save configurations for future reference
tracker.save_config("final_config.json")
```

### 2. Metric Selection

```python
# Focus on relevant metrics for your research
# Don't log everything - be selective
tracker.log_training_step(
    loss=loss,
    gradient_norm=grad_norm,  # Essential for stability
    nan_count=nan_count       # Critical for numerical stability
)
```

### 3. Regular Checkpointing

```python
# Save checkpoints regularly
if step % save_interval == 0:
    tracker.save_checkpoint(...)
```

### 4. Resource Monitoring

```python
# Monitor system resources
if step % 100 == 0:
    memory_usage = get_memory_usage()
    gpu_utilization = get_gpu_utilization()
    tracker.log_training_step(
        loss=loss,
        memory_usage=memory_usage,
        gpu_utilization=gpu_utilization
    )
```

## 🔍 Troubleshooting

### Common Issues

1. **TensorBoard Not Starting**
   - Check if port 6006 is available
   - Verify log directory exists
   - Check TensorBoard installation

2. **Wandb Connection Issues**
   - Verify internet connection
   - Check wandb authentication
   - Ensure wandb is properly installed

3. **Performance Issues**
   - Reduce logging frequency
   - Use asynchronous logging
   - Monitor system resources

4. **Memory Issues**
   - Reduce batch size for logging
   - Clear old checkpoints
   - Monitor memory usage

### Debug Commands

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Check system status
tracker.get_system_status()

# Verify configuration
print(tracker.config)
```

## 📚 Examples

### Complete Training Example

See `example_usage()` function in `experiment_tracking.py` for a complete working example.

### Integration Examples

- **PyTorch Training Loop**: See training loop integration above
- **Custom Models**: Adapt for your specific model architectures
- **Multi-GPU Training**: Works with DataParallel and DistributedDataParallel

## 🤝 Contributing

### Development

1. **Code Style**: Follow PEP 8 and project conventions
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **Error Handling**: Implement robust error handling

### Feature Requests

- **New Metrics**: Suggest additional metrics to track
- **Visualizations**: Propose new visualization types
- **Integrations**: Request additional tracking system support

## 📄 License

This project is part of the Blatam Academy Facebook Posts feature.

## 🙏 Acknowledgments

- **TensorBoard**: Google's visualization toolkit
- **Weights & Biases**: Cloud-based experiment tracking
- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework

---

**🔬 Experiment Tracking System** | Built for Numerical Stability Research

Monitor your deep learning experiments with comprehensive logging, visualization, and tracking capabilities.
