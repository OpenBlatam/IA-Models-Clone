# Experiment Tracking Guide: TensorBoard & Weights & Biases

A comprehensive guide for experiment tracking using TensorBoard and Weights & Biases (wandb) in the Video-OpusClip system.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [TensorBoard Guide](#tensorboard-guide)
4. [Weights & Biases Guide](#weights--biases-guide)
5. [Integration with Video-OpusClip](#integration-with-video-opusclip)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## Overview

Experiment tracking is crucial for machine learning projects to:
- Monitor training progress
- Compare different experiments
- Track hyperparameters and configurations
- Visualize metrics and results
- Manage model checkpoints
- Collaborate with team members

### TensorBoard vs Weights & Biases

| Feature | TensorBoard | Weights & Biases |
|---------|-------------|------------------|
| **Setup** | Local installation | Cloud-based service |
| **Cost** | Free | Free tier + paid plans |
| **Collaboration** | Limited | Excellent |
| **Version Control** | Manual | Automatic |
| **Model Registry** | No | Yes |
| **Hyperparameter Tuning** | Manual | Integrated |
| **Artifact Management** | Basic | Advanced |
| **Real-time Monitoring** | Yes | Yes |
| **Mobile App** | No | Yes |

## Installation

### TensorBoard Installation

```bash
# Already included in requirements_complete.txt
pip install tensorboard>=2.13.0

# For additional features
pip install tensorboardX>=2.6.0
```

### Weights & Biases Installation

```bash
# Already included in requirements_complete.txt
pip install wandb>=0.15.0

# Login to wandb
wandb login
```

### Verification

```python
# Test TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("test_logs")
writer.add_scalar("test", 1.0, 0)
writer.close()

# Test Weights & Biases
import wandb
wandb.init(project="test", mode="disabled")
wandb.log({"test": 1.0})
wandb.finish()
```

## TensorBoard Guide

### Basic Usage

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# Initialize writer
writer = SummaryWriter('runs/experiment_1')

# Log scalar values
for step in range(100):
    loss = 1.0 / (step + 1)
    accuracy = 0.8 + step * 0.001
    writer.add_scalar('Loss/train', loss, step)
    writer.add_scalar('Accuracy/train', accuracy, step)

# Log multiple scalars at once
for step in range(100):
    metrics = {
        'Loss/train': 1.0 / (step + 1),
        'Loss/val': 1.2 / (step + 1),
        'Accuracy/train': 0.8 + step * 0.001,
        'Accuracy/val': 0.75 + step * 0.001
    }
    writer.add_scalars('Training Metrics', metrics, step)

# Log hyperparameters
hparams = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}
writer.add_hparams(hparams, {'hparam/accuracy': 0.85})

# Log model graph
model = nn.Linear(10, 1)
dummy_input = torch.randn(1, 10)
writer.add_graph(model, dummy_input)

# Log images
for step in range(10):
    # Generate sample images
    images = torch.randn(4, 3, 64, 64)
    writer.add_images('Sample Images', images, step)

# Log histograms
for step in range(100):
    # Log weight distributions
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param.data, step)
        writer.add_histogram(f'gradients/{name}', param.grad, step)

# Log text
writer.add_text('Experiment Notes', 'This is a test experiment', 0)

# Close writer
writer.close()
```

### Advanced Features

#### Custom Scalars

```python
# Create custom scalar layout
layout = {
    'Loss': {
        'Training': ['multiline', ['Loss/train', 'Loss/val']],
        'Test': ['multiline', ['Loss/test']]
    },
    'Accuracy': {
        'Training': ['multiline', ['Accuracy/train', 'Accuracy/val']],
        'Test': ['multiline', ['Accuracy/test']]
    }
}
writer.add_custom_scalars(layout)
```

#### Embeddings Visualization

```python
# Log embeddings
embedding = torch.randn(100, 128)
metadata = [f'item_{i}' for i in range(100)]
writer.add_embedding(embedding, metadata=metadata, global_step=0)
```

#### Audio Logging

```python
# Log audio samples
for step in range(10):
    audio = torch.randn(1, 16000)  # 1 second at 16kHz
    writer.add_audio('Sample Audio', audio, step, sample_rate=16000)
```

#### Video Logging

```python
# Log video samples
for step in range(10):
    video = torch.randn(1, 3, 16, 64, 64)  # 16 frames, 64x64
    writer.add_video('Sample Video', video, step, fps=8)
```

### Launching TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir=runs --port=6006

# Access at http://localhost:6006
```

## Weights & Biases Guide

### Basic Usage

```python
import wandb
import torch
import torch.nn as nn

# Initialize wandb
wandb.init(
    project="video-opusclip",
    name="experiment_1",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "model": "transformer"
    }
)

# Log metrics
for step in range(100):
    loss = 1.0 / (step + 1)
    accuracy = 0.8 + step * 0.001
    
    wandb.log({
        "train_loss": loss,
        "train_accuracy": accuracy,
        "step": step
    })

# Log hyperparameters
wandb.config.update({
    "optimizer": "adam",
    "scheduler": "cosine"
})

# Log model
model = nn.Linear(10, 1)
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

# Log images
for step in range(10):
    images = torch.randn(4, 3, 64, 64)
    wandb.log({
        "sample_images": wandb.Image(images)
    }, step=step)

# Log tables
import pandas as pd
df = pd.DataFrame({
    "epoch": range(10),
    "loss": [1.0 / (i + 1) for i in range(10)],
    "accuracy": [0.8 + i * 0.01 for i in range(10)]
})
wandb.log({"results_table": wandb.Table(dataframe=df)})

# Finish run
wandb.finish()
```

### Advanced Features

#### Model Versioning

```python
# Log model with versioning
model_artifact = wandb.Artifact(
    name="video-model",
    type="model",
    description="Video generation model"
)
model_artifact.add_file("model.pth")
wandb.log_artifact(model_artifact)

# Load model from artifact
with wandb.init() as run:
    artifact = run.use_artifact("username/video-opusclip/video-model:latest")
    model_path = artifact.download()
```

#### Hyperparameter Sweeps

```python
# Define sweep configuration
sweep_config = {
    "method": "random",
    "name": "video-model-sweep",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "min": 0.0001,
            "max": 0.01
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "epochs": {
            "values": [50, 100, 200]
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="video-opusclip")

# Define training function
def train():
    wandb.init()
    config = wandb.config
    
    # Training loop
    for epoch in range(config.epochs):
        loss = train_epoch()
        wandb.log({"val_loss": loss, "epoch": epoch})

# Run sweep
wandb.agent(sweep_id, train, count=10)
```

#### Custom Plots

```python
# Create custom plots
import plotly.graph_objects as go

# Loss curve
fig = go.Figure()
fig.add_trace(go.Scatter(y=losses, name="Training Loss"))
fig.add_trace(go.Scatter(y=val_losses, name="Validation Loss"))
wandb.log({"loss_plot": wandb.plotly(fig)})

# Confusion matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=labels,
        preds=predictions
    )
})
```

#### System Monitoring

```python
# Monitor system resources
wandb.watch(model, log="all", log_freq=100)

# Log system metrics
import psutil
wandb.log({
    "cpu_usage": psutil.cpu_percent(),
    "memory_usage": psutil.virtual_memory().percent,
    "gpu_usage": get_gpu_usage()  # Custom function
})
```

## Integration with Video-OpusClip

### Experiment Tracker Class

```python
class VideoOpusClipTracker:
    """Unified experiment tracker for Video-OpusClip."""
    
    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        self.tensorboard_writer = None
        self._initialize_tracking()
    
    def _initialize_tracking(self):
        """Initialize tracking systems."""
        # TensorBoard
        if self.config.use_tensorboard:
            log_dir = f"runs/{self.config.experiment_name}"
            self.tensorboard_writer = SummaryWriter(log_dir)
        
        # Weights & Biases
        if self.config.use_wandb:
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.to_dict()
            )
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to all tracking systems."""
        # TensorBoard
        if self.tensorboard_writer:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Weights & Biases
        if self.wandb_run:
            wandb.log(metrics, step=step)
    
    def log_video_metrics(self, psnr, ssim, lpips, step=None):
        """Log video-specific metrics."""
        video_metrics = {
            "video/psnr": psnr,
            "video/ssim": ssim,
            "video/lpips": lpips
        }
        self.log_metrics(video_metrics, step)
    
    def log_generated_video(self, video_path, step=None):
        """Log generated video samples."""
        if self.wandb_run:
            wandb.log({
                "generated_video": wandb.Video(video_path)
            }, step=step)
    
    def close(self):
        """Close all tracking systems."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.wandb_run:
            wandb.finish()
```

### Training Integration

```python
class VideoTrainer:
    """Video training with experiment tracking."""
    
    def __init__(self, config):
        self.config = config
        self.tracker = VideoOpusClipTracker(config)
        self.model = create_model(config)
        self.optimizer = create_optimizer(self.model, config)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Log batch metrics
            if batch_idx % self.config.log_frequency == 0:
                self.tracker.log_metrics({
                    "train/batch_loss": loss,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr']
                }, step=epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss, psnr, ssim = self.validate_step(batch)
                total_loss += loss
                total_psnr += psnr
                total_ssim += ssim
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        # Log validation metrics
        self.tracker.log_metrics({
            "val/loss": avg_loss,
            "val/psnr": avg_psnr,
            "val/ssim": avg_ssim
        }, step=epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }
        
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Log checkpoint
        if self.config.use_wandb:
            wandb.save(checkpoint_path)
```

## Advanced Features

### Custom Metrics

```python
class VideoMetrics:
    """Video-specific metrics calculation."""
    
    @staticmethod
    def calculate_psnr(pred, target):
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = torch.mean((pred - target) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(pred, target):
        """Calculate Structural Similarity Index."""
        # Implementation of SSIM
        pass
    
    @staticmethod
    def calculate_lpips(pred, target, lpips_fn):
        """Calculate Learned Perceptual Image Patch Similarity."""
        return lpips_fn(pred, target)
```

### Experiment Comparison

```python
class ExperimentComparator:
    """Compare multiple experiments."""
    
    def __init__(self, experiment_names):
        self.experiments = experiment_names
    
    def compare_metrics(self, metric_name):
        """Compare specific metric across experiments."""
        api = wandb.Api()
        
        for exp_name in self.experiments:
            runs = api.runs(f"username/video-opusclip", filters={"display_name": exp_name})
            for run in runs:
                history = run.history()
                if metric_name in history.columns:
                    plt.plot(history[metric_name], label=exp_name)
        
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.legend()
        plt.show()
```

### Automated Reporting

```python
class ExperimentReporter:
    """Generate automated experiment reports."""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        report = {
            "experiment_name": self.tracker.config.experiment_name,
            "duration": self.tracker.get_duration(),
            "best_metrics": self.tracker.get_best_metrics(),
            "hyperparameters": self.tracker.config.to_dict(),
            "system_info": self.get_system_info()
        }
        
        # Log report
        if self.tracker.wandb_run:
            wandb.log({"experiment_report": wandb.Table(dataframe=pd.DataFrame([report]))})
        
        return report
```

## Best Practices

### 1. Consistent Naming

```python
# Use consistent metric naming
metrics = {
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "test/loss": test_loss,
    "test/accuracy": test_acc
}
```

### 2. Logging Frequency

```python
# Log at appropriate intervals
log_frequency = 100  # Log every 100 steps
eval_frequency = 1000  # Evaluate every 1000 steps
save_frequency = 5000  # Save checkpoint every 5000 steps
```

### 3. Resource Management

```python
# Monitor system resources
def log_system_metrics():
    return {
        "system/cpu_usage": psutil.cpu_percent(),
        "system/memory_usage": psutil.virtual_memory().percent,
        "system/gpu_usage": get_gpu_usage(),
        "system/gpu_memory": get_gpu_memory()
    }
```

### 4. Error Handling

```python
def safe_log_metrics(tracker, metrics, step):
    """Safely log metrics with error handling."""
    try:
        tracker.log_metrics(metrics, step)
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")
        # Fallback to local logging
        log_locally(metrics, step)
```

### 5. Configuration Management

```python
@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_name: str
    project_name: str = "video-opusclip"
    use_tensorboard: bool = True
    use_wandb: bool = True
    log_frequency: int = 100
    save_frequency: int = 1000
    max_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    
    def to_dict(self):
        return asdict(self)
```

## Troubleshooting

### Common Issues

#### TensorBoard Issues

```bash
# TensorBoard not starting
tensorboard --logdir=runs --port=6006 --bind_all

# Permission issues
sudo chmod -R 755 runs/

# Port already in use
tensorboard --logdir=runs --port=6007
```

#### Weights & Biases Issues

```python
# Authentication issues
wandb login

# Network issues
wandb.init(mode="offline")

# Memory issues
wandb.init(mode="disabled")
```

#### Performance Issues

```python
# Reduce logging frequency
config.log_frequency = 1000

# Use async logging
import asyncio

async def async_log_metrics(tracker, metrics, step):
    await asyncio.to_thread(tracker.log_metrics, metrics, step)
```

### Debug Mode

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug TensorBoard
writer = SummaryWriter('debug_logs', flush_secs=1)

# Debug Weights & Biases
wandb.init(mode="disabled", project="debug")
```

## Examples

### Complete Training Example

```python
def main():
    # Configuration
    config = ExperimentConfig(
        experiment_name="video_generation_v1",
        use_tensorboard=True,
        use_wandb=True,
        max_epochs=100
    )
    
    # Initialize trainer
    trainer = VideoTrainer(config)
    
    # Training loop
    for epoch in range(config.max_epochs):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate(epoch)
        
        # Log epoch metrics
        trainer.tracker.log_metrics({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss
        }, step=epoch)
        
        # Save checkpoint
        if epoch % 10 == 0:
            trainer.save_checkpoint(epoch, {"val_loss": val_loss})
    
    # Close tracker
    trainer.tracker.close()

if __name__ == "__main__":
    main()
```

### Video Generation Tracking

```python
def track_video_generation():
    """Track video generation process."""
    tracker = VideoOpusClipTracker(config)
    
    for step in range(100):
        # Generate video
        video = generate_video(model, prompt)
        
        # Calculate metrics
        psnr = calculate_psnr(video, target)
        ssim = calculate_ssim(video, target)
        lpips = calculate_lpips(video, target)
        
        # Log metrics
        tracker.log_video_metrics(psnr, ssim, lpips, step)
        
        # Log sample video
        if step % 10 == 0:
            save_video(video, f"sample_{step}.mp4")
            tracker.log_generated_video(f"sample_{step}.mp4", step)
    
    tracker.close()
```

This comprehensive guide provides everything you need to implement effective experiment tracking with TensorBoard and Weights & Biases in your Video-OpusClip system. 