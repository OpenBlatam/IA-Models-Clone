# Configuration Management and Experiment Tracking Guide

## Overview

This guide covers the comprehensive configuration management and experiment tracking system for the Onyx Ads Backend. The system provides:

- **Configuration Management**: YAML-based configuration files for all project settings
- **Experiment Tracking**: Multi-backend experiment tracking with automated checkpointing
- **Model Versioning**: Automated model checkpointing with versioning and cleanup
- **Reproducibility**: Complete experiment tracking for reproducible research
- **Performance Monitoring**: Real-time metrics tracking and visualization

## Table of Contents

1. [Configuration Management](#configuration-management)
2. [Experiment Tracking](#experiment-tracking)
3. [Model Checkpointing](#model-checkpointing)
4. [Integration with Existing Systems](#integration-with-existing-systems)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Configuration Management

### Configuration Types

The system supports six main configuration types:

#### 1. Model Configuration (`ModelConfig`)
```yaml
name: "ad_classification_model"
type: "transformer"
architecture: "bert-base-uncased"
input_size: 768
output_size: 10
hidden_sizes: [512, 256]
dropout_rate: 0.1
activation: "gelu"
batch_norm: true
pretrained: true
pretrained_path: "./pretrained/bert-base-uncased"
freeze_backbone: false
custom_parameters:
  attention_heads: 12
  hidden_dropout: 0.1
```

#### 2. Training Configuration (`TrainingConfig`)
```yaml
# Basic training settings
batch_size: 32
learning_rate: 2e-5
epochs: 10
validation_split: 0.2
test_split: 0.1
random_seed: 42

# Optimizer settings
optimizer: "adamw"
optimizer_params:
  weight_decay: 0.01
  betas: [0.9, 0.999]
scheduler: "cosine"
scheduler_params:
  warmup_steps: 1000
  max_steps: 10000

# Loss function settings
loss_function: "cross_entropy"
loss_params:
  label_smoothing: 0.1

# Regularization
weight_decay: 1e-4
gradient_clipping: 1.0

# Advanced training settings
mixed_precision: true
gradient_accumulation_steps: 4
early_stopping_patience: 10
early_stopping_min_delta: 1e-4
save_best_only: true
save_frequency: 1
```

#### 3. Data Configuration (`DataConfig`)
```yaml
# Data paths
train_data_path: "./data/train"
val_data_path: "./data/val"
test_data_path: "./data/test"
cache_dir: "./cache"

# Data processing
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2

# Data augmentation
augmentation: true
augmentation_params:
  rotation: 10
  horizontal_flip: true
  color_jitter: 0.1
  random_crop: [224, 224]

# Data validation
validate_data: true
max_samples: null
shuffle: true

# Data types
input_dtype: "float32"
target_dtype: "long"
```

#### 4. Experiment Configuration (`ExperimentConfig`)
```yaml
# Experiment identification
experiment_name: "ad_classification_v1"
experiment_id: "exp_001"
project_name: "ad_classification"
tags: ["transformer", "classification", "v1"]

# Tracking settings
track_experiments: true
tracking_backend: "wandb"  # wandb, mlflow, tensorboard, local
tracking_params:
  entity: "your_team"
  project: "ad_classification"

# Logging settings
log_level: "INFO"
log_frequency: 100
log_metrics: ["loss", "accuracy", "f1_score", "precision", "recall"]
log_gradients: false
log_hyperparameters: true

# Checkpointing settings
save_checkpoints: true
checkpoint_dir: "./checkpoints"
checkpoint_frequency: 1
max_checkpoints: 5
save_optimizer: true
save_scheduler: true
```

#### 5. Optimization Configuration (`OptimizationConfig`)
```yaml
# Performance optimization
enable_mixed_precision: true
enable_gradient_checkpointing: true
enable_model_compilation: true
enable_cudnn_benchmark: true
enable_cudnn_deterministic: false

# Memory optimization
memory_fraction: 0.8
gradient_accumulation_steps: 4
max_grad_norm: 1.0

# Multi-GPU settings
distributed_training: false
num_gpus: 1
backend: "nccl"

# Profiling settings
enable_profiling: false
profile_memory: true
profile_performance: true
```

#### 6. Deployment Configuration (`DeploymentConfig`)
```yaml
# Model serving
model_format: "torchscript"  # torchscript, onnx, tensorrt
optimization_level: "O1"
quantization: false
quantization_params:
  dtype: "int8"
  calibration_data: "./calibration_data"

# API settings
api_host: "0.0.0.0"
api_port: 8000
api_workers: 4
api_timeout: 30

# Monitoring
enable_monitoring: true
metrics_endpoint: "/metrics"
health_check_endpoint: "/health"
```

### Using the Configuration Manager

#### Creating Default Configurations
```python
from onyx.server.features.ads.config_manager import ConfigManager

# Create config manager
config_manager = ConfigManager("./configs")

# Create default configs for a project
config_files = config_manager.create_default_configs("ad_classification")

print("Created configuration files:")
for config_type, file_path in config_files.items():
    print(f"  {config_type}: {file_path}")
```

#### Loading Configurations
```python
# Load all configurations for a project
configs = config_manager.load_all_configs("ad_classification")

# Access specific configurations
model_config = configs['model']
training_config = configs['training']
data_config = configs['data']
experiment_config = configs['experiment']
optimization_config = configs['optimization']
deployment_config = configs['deployment']

# Get configuration summary
summary = config_manager.get_config_summary(configs)
print("Configuration Summary:")
print(json.dumps(summary, indent=2))
```

#### Updating Configurations
```python
# Update specific configuration values
updates = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 20
}

config_manager.update_config(
    "./configs/ad_classification_training_config.yaml",
    updates
)
```

#### Validating Configurations
```python
from onyx.server.features.ads.config_manager import ConfigType

# Validate configurations
for config_type, config in configs.items():
    config_enum = ConfigType[config_type.upper()]
    is_valid = config_manager.validate_config(config, config_enum)
    print(f"{config_type} config is valid: {is_valid}")
```

## Experiment Tracking

### Supported Backends

The system supports multiple experiment tracking backends:

1. **Weights & Biases (W&B)** - Cloud-based experiment tracking
2. **MLflow** - Open-source ML lifecycle management
3. **TensorBoard** - TensorFlow's visualization toolkit
4. **Local** - File-based local tracking

### Setting Up Experiment Tracking

#### Basic Setup
```python
from onyx.server.features.ads.experiment_tracker import (
    ExperimentTracker, ExperimentMetadata, create_experiment_tracker
)
from onyx.server.features.ads.config_manager import ExperimentConfig

# Create experiment configuration
experiment_config = ExperimentConfig(
    experiment_name="ad_classification_v1",
    project_name="ad_classification",
    track_experiments=True,
    tracking_backend="wandb",  # or "local", "tensorboard", "mlflow"
    save_checkpoints=True,
    checkpoint_dir="./checkpoints",
    log_metrics=["loss", "accuracy", "f1_score"]
)

# Create experiment tracker
tracker = create_experiment_tracker(experiment_config)

# Create experiment metadata
metadata = ExperimentMetadata(
    experiment_id="exp_001",
    experiment_name="ad_classification_v1",
    project_name="ad_classification",
    created_at=datetime.now(),
    tags=["transformer", "classification", "v1"],
    description="BERT-based ad classification model",
    git_commit="abc123",
    python_version="3.9.0"
)

# Start experiment
tracker.start_experiment(metadata)
```

#### Using Context Manager
```python
from onyx.server.features.ads.experiment_tracker import experiment_context

with experiment_context(experiment_config, metadata) as tracker:
    # Your training code here
    for epoch in range(training_config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Training step
            loss = model(data, target)
            loss.backward()
            optimizer.step()
            
            # Log metrics
            tracker.log_metrics({
                "loss": loss.item(),
                "accuracy": accuracy.item()
            }, step=batch_idx, epoch=epoch)
            
            # Save checkpoint periodically
            if batch_idx % 100 == 0:
                tracker.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={"loss": loss.item()},
                    is_best=False
                )
```

### Logging Different Types of Data

#### Metrics Logging
```python
# Log scalar metrics
tracker.log_metrics({
    "train_loss": 0.5,
    "train_accuracy": 0.85,
    "val_loss": 0.6,
    "val_accuracy": 0.82,
    "learning_rate": 1e-4
}, step=100, epoch=5)

# Log custom metrics
tracker.log_metrics({
    "f1_score": 0.78,
    "precision": 0.81,
    "recall": 0.75,
    "auc": 0.92
}, step=100, epoch=5)
```

#### Hyperparameters Logging
```python
# Log hyperparameters
hyperparameters = {
    "model": {
        "architecture": "bert-base-uncased",
        "hidden_size": 768,
        "num_classes": 10
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 10,
        "optimizer": "adamw"
    },
    "data": {
        "num_workers": 4,
        "augmentation": True
    }
}

tracker.log_hyperparameters(hyperparameters)
```

#### Model Architecture Logging
```python
# Log model architecture
tracker.log_model_architecture(model)
```

#### Gradient Logging
```python
# Log gradients (if enabled)
if training_config.log_gradients:
    tracker.log_gradients(model, step=100)
```

#### Image Logging
```python
# Log images for visualization
images = {
    "sample_predictions": sample_images,
    "attention_maps": attention_visualizations,
    "confusion_matrix": confusion_matrix_plot
}

tracker.log_images(images, step=100)
```

#### Text Logging
```python
# Log text data
text_data = {
    "sample_predictions": "Sample prediction results...",
    "model_summary": "Model architecture summary...",
    "training_log": "Training progress log..."
}

tracker.log_text(text_data, step=100)
```

## Model Checkpointing

### Automatic Checkpointing

The system provides automatic checkpointing with the following features:

- **Versioning**: Each checkpoint is versioned with timestamp and step information
- **Best Model Tracking**: Automatically tracks the best model based on metrics
- **Cleanup**: Automatically removes old checkpoints to save disk space
- **Metadata**: Stores comprehensive metadata with each checkpoint

### Checkpoint Management

#### Saving Checkpoints
```python
# Save regular checkpoint
checkpoint_path = tracker.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={"loss": loss.item(), "accuracy": accuracy.item()},
    is_best=False
)

# Save best checkpoint
if is_best_model:
    best_checkpoint_path = tracker.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"loss": loss.item(), "accuracy": accuracy.item()},
        is_best=True
    )
```

#### Loading Checkpoints
```python
# Load latest checkpoint
latest_checkpoint = tracker.checkpoint_manager.get_latest_checkpoint("exp_001")
if latest_checkpoint:
    checkpoint_info = tracker.load_checkpoint(
        model=model,
        checkpoint_path=latest_checkpoint,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")

# Load best checkpoint
best_checkpoint = tracker.checkpoint_manager.get_best_checkpoint("exp_001")
if best_checkpoint:
    checkpoint_info = tracker.load_checkpoint(
        model=model,
        checkpoint_path=best_checkpoint,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print(f"Loaded best checkpoint with metrics: {checkpoint_info['metrics']}")
```

#### Checkpoint Information
```python
# Get checkpoint information
checkpoint_infos = tracker.checkpoint_manager.get_checkpoint_info("exp_001")

for checkpoint_info in checkpoint_infos:
    print(f"Checkpoint: {checkpoint_info['checkpoint_id']}")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Step: {checkpoint_info['step']}")
    print(f"  Metrics: {checkpoint_info['metrics']}")
    print(f"  Is Best: {checkpoint_info['is_best']}")
    print(f"  File Size: {checkpoint_info['file_size']} bytes")
```

### Checkpoint Structure

Each checkpoint file contains:

```python
checkpoint_data = {
    'epoch': 5,
    'step': 1000,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # if save_optimizer=True
    'scheduler_state_dict': scheduler.state_dict(),  # if save_scheduler=True
    'metrics': {
        'loss': 0.5,
        'accuracy': 0.85,
        'f1_score': 0.78
    },
    'is_best': True,
    'timestamp': '2024-01-15T10:30:00',
    'checkpoint_id': 'checkpoint_epoch_5_step_1000_20240115_103000_best'
}
```

## Integration with Existing Systems

### Integration with Training Loops

#### Basic Integration
```python
def train_with_tracking(model, train_loader, val_loader, configs):
    # Create experiment tracker
    tracker = create_experiment_tracker(configs['experiment'])
    
    # Start experiment
    metadata = ExperimentMetadata(
        experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_name=configs['experiment'].experiment_name,
        project_name=configs['experiment'].project_name,
        created_at=datetime.now(),
        tags=configs['experiment'].tags
    )
    tracker.start_experiment(metadata)
    
    # Log hyperparameters
    hyperparameters = {
        "model": asdict(configs['model']),
        "training": asdict(configs['training']),
        "data": asdict(configs['data']),
        "optimization": asdict(configs['optimization'])
    }
    tracker.log_hyperparameters(hyperparameters)
    
    # Log model architecture
    tracker.log_model_architecture(model)
    
    # Training loop
    best_metric = float('inf')
    for epoch in range(configs['training'].epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Training step
            loss = train_step(model, data, target, optimizer)
            
            # Log metrics
            if batch_idx % configs['experiment'].log_frequency == 0:
                tracker.log_metrics({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                }, step=batch_idx, epoch=epoch)
            
            # Save checkpoint
            if batch_idx % configs['experiment'].checkpoint_frequency == 0:
                tracker.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={"loss": loss.item()},
                    is_best=False
                )
        
        # Validation
        val_metrics = validate(model, val_loader)
        tracker.log_metrics(val_metrics, step=batch_idx, epoch=epoch)
        
        # Save best checkpoint
        if val_metrics['val_loss'] < best_metric:
            best_metric = val_metrics['val_loss']
            tracker.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=val_metrics,
                is_best=True
            )
    
    # End experiment
    tracker.end_experiment()
```

#### Integration with Mixed Precision Training
```python
from onyx.server.features.ads.mixed_precision_training import MixedPrecisionTrainer

def train_with_mixed_precision_and_tracking(model, train_loader, val_loader, configs):
    # Create mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=configs['optimization']
    )
    
    # Create experiment tracker
    tracker = create_experiment_tracker(configs['experiment'])
    tracker.start_experiment()
    
    # Training loop with mixed precision
    for epoch in range(configs['training'].epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Mixed precision training step
            loss = mp_trainer.train_step(data, target)
            
            # Log metrics
            tracker.log_metrics({
                "loss": loss.item(),
                "scaler_scale": mp_trainer.scaler.get_scale()
            }, step=batch_idx, epoch=epoch)
            
            # Save checkpoint
            if batch_idx % configs['experiment'].checkpoint_frequency == 0:
                tracker.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={"loss": loss.item()}
                )
    
    tracker.end_experiment()
```

### Integration with Diffusion Models

```python
from onyx.server.features.ads.diffusion_service import DiffusionService

def train_diffusion_with_tracking(configs):
    # Create diffusion service
    diffusion_service = DiffusionService(configs['model'])
    
    # Create experiment tracker
    tracker = create_experiment_tracker(configs['experiment'])
    tracker.start_experiment()
    
    # Training loop
    for epoch in range(configs['training'].epochs):
        for batch_idx, (images, conditions) in enumerate(train_loader):
            # Diffusion training step
            loss = diffusion_service.train_step(images, conditions)
            
            # Log metrics
            tracker.log_metrics({
                "diffusion_loss": loss.item(),
                "noise_level": diffusion_service.current_noise_level
            }, step=batch_idx, epoch=epoch)
            
            # Log sample images
            if batch_idx % 100 == 0:
                sample_images = diffusion_service.generate_samples(conditions[:4])
                tracker.log_images({
                    "generated_samples": sample_images
                }, step=batch_idx)
    
    tracker.end_experiment()
```

## Best Practices

### Configuration Management

1. **Use Version Control**: Always commit configuration files to version control
2. **Environment-Specific Configs**: Create separate configs for different environments
3. **Validation**: Always validate configurations before use
4. **Documentation**: Document all configuration parameters
5. **Defaults**: Provide sensible defaults for all parameters

### Experiment Tracking

1. **Consistent Naming**: Use consistent naming conventions for experiments
2. **Meaningful Tags**: Use descriptive tags for easy filtering
3. **Regular Logging**: Log metrics at regular intervals
4. **Complete Metadata**: Include all relevant metadata
5. **Backup Strategy**: Use multiple tracking backends for redundancy

### Model Checkpointing

1. **Regular Checkpoints**: Save checkpoints at regular intervals
2. **Best Model Tracking**: Always track the best model
3. **Cleanup Policy**: Implement automatic cleanup of old checkpoints
4. **Verification**: Verify checkpoint integrity after loading
5. **Metadata**: Include comprehensive metadata with checkpoints

### Performance Optimization

1. **Async Logging**: Use async logging for better performance
2. **Batch Logging**: Batch multiple metrics together
3. **Selective Logging**: Only log essential metrics at high frequency
4. **Compression**: Compress checkpoint files for storage efficiency
5. **Caching**: Cache frequently accessed checkpoint metadata

## Troubleshooting

### Common Issues

#### Configuration Loading Errors
```python
# Problem: Configuration file not found
try:
    configs = config_manager.load_all_configs("project_name")
except FileNotFoundError as e:
    print(f"Configuration files not found: {e}")
    # Create default configs
    config_files = config_manager.create_default_configs("project_name")
```

#### Experiment Tracking Issues
```python
# Problem: Backend not available
try:
    tracker = create_experiment_tracker(experiment_config)
except ImportError as e:
    print(f"Tracking backend not available: {e}")
    # Fallback to local backend
    experiment_config.tracking_backend = "local"
    tracker = create_experiment_tracker(experiment_config)
```

#### Checkpoint Issues
```python
# Problem: Checkpoint loading fails
try:
    checkpoint_info = tracker.load_checkpoint(model, checkpoint_path)
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    # Try loading without optimizer/scheduler
    checkpoint_info = tracker.load_checkpoint(model, checkpoint_path, 
                                            optimizer=None, scheduler=None)
```

### Debugging Tips

1. **Enable Debug Logging**: Set log level to DEBUG for detailed information
2. **Check File Permissions**: Ensure proper file permissions for checkpoint directories
3. **Verify Dependencies**: Check that all required packages are installed
4. **Monitor Disk Space**: Ensure sufficient disk space for checkpoints
5. **Test Backends**: Test tracking backends individually

### Performance Monitoring

```python
# Monitor experiment tracking performance
import time

start_time = time.time()
tracker.log_metrics(metrics, step=step)
logging_time = time.time() - start_time

if logging_time > 1.0:  # More than 1 second
    logger.warning(f"Slow logging detected: {logging_time:.2f}s")
```

## Advanced Features

### Custom Tracking Backends

```python
class CustomBackend(TrackingBackendBase):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # Initialize your custom backend
    
    def start_experiment(self, metadata: ExperimentMetadata):
        # Start experiment in your backend
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        # Log metrics to your backend
    
    def end_experiment(self):
        # End experiment in your backend
```

### Custom Checkpoint Formats

```python
class CustomCheckpointManager(CheckpointManager):
    def save_checkpoint(self, **kwargs):
        # Custom checkpoint saving logic
        pass
    
    def load_checkpoint(self, **kwargs):
        # Custom checkpoint loading logic
        pass
```

### Integration with External Systems

```python
# Integration with MLflow
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_config.project_name)

# Integration with Weights & Biases
if WANDB_AVAILABLE:
    wandb.init(project=experiment_config.project_name)
```

This comprehensive guide provides everything needed to effectively use the configuration management and experiment tracking system in your ML projects. 