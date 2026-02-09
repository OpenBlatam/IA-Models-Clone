#!/usr/bin/env python3
"""
Quick Start: Experiment Tracking with TensorBoard & Weights & Biases
===================================================================

This script provides a quick start guide for setting up and using
TensorBoard and Weights & Biases for experiment tracking in the
Video-OpusClip system.

Features:
- Installation verification
- Basic TensorBoard usage
- Basic Weights & Biases usage
- Video-specific tracking
- Integration examples
- Performance monitoring
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Install with: pip install psutil")

# Configuration
@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_name: str
    project_name: str = "video-opusclip"
    use_tensorboard: bool = True
    use_wandb: bool = True
    log_frequency: int = 100
    save_frequency: int = 1000
    max_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    device: str = "cpu"
    
    def to_dict(self):
        return asdict(self)

class ExperimentTracker:
    """Unified experiment tracker for Video-OpusClip."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.wandb_run = None
        self.tensorboard_writer = None
        self.start_time = time.time()
        self.current_step = 0
        self.metrics_history = {}
        
        self._initialize_tracking()
        logger.info(f"Experiment tracker initialized: {config.experiment_name}")
    
    def _initialize_tracking(self):
        """Initialize tracking systems."""
        # Create directories
        Path("runs").mkdir(exist_ok=True)
        Path("checkpoints").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = f"runs/{self.config.experiment_name}"
            self.tensorboard_writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard initialized: {log_dir}")
        
        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.project_name,
                    name=self.config.experiment_name,
                    config=self.config.to_dict(),
                    mode="online" if self.config.use_wandb else "disabled"
                )
                logger.info(f"Weights & Biases initialized: {self.wandb_run.id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self.wandb_run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all tracking systems."""
        if step is None:
            step = self.current_step
        
        # Update metrics history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # TensorBoard
        if self.tensorboard_writer:
            try:
                for name, value in metrics.items():
                    self.tensorboard_writer.add_scalar(name, value, step)
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")
        
        # Weights & Biases
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to Weights & Biases: {e}")
        
        self.current_step += 1
    
    def log_video_metrics(self, psnr: float, ssim: float, lpips: float, step: Optional[int] = None):
        """Log video-specific metrics."""
        video_metrics = {
            "video/psnr": psnr,
            "video/ssim": ssim,
            "video/lpips": lpips
        }
        self.log_metrics(video_metrics, step)
    
    def log_system_metrics(self):
        """Log system resource usage."""
        if not PSUTIL_AVAILABLE:
            return
        
        system_metrics = {
            "system/cpu_usage": psutil.cpu_percent(),
            "system/memory_usage": psutil.virtual_memory().percent,
            "system/memory_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            system_metrics.update({
                "system/gpu_usage": torch.cuda.utilization(),
                "system/gpu_memory_used_gb": torch.cuda.memory_allocated() / (1024**3),
                "system/gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        self.log_metrics(system_metrics)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_hparams(hyperparameters, {})
            except Exception as e:
                logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
        
        if self.wandb_run:
            try:
                wandb.config.update(hyperparameters)
            except Exception as e:
                logger.warning(f"Failed to log hyperparameters to Weights & Biases: {e}")
    
    def log_model_graph(self, model: nn.Module, dummy_input: torch.Tensor):
        """Log model architecture."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, dummy_input)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict()
        }
        
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Log checkpoint
        if self.wandb_run:
            try:
                wandb.save(checkpoint_path)
                logger.info(f"Checkpoint saved and logged: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to log checkpoint: {e}")
        
        return checkpoint_path
    
    def get_duration(self) -> float:
        """Get experiment duration in seconds."""
        return time.time() - self.start_time
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from history."""
        best_metrics = {}
        for key, values in self.metrics_history.items():
            if "loss" in key.lower():
                best_metrics[key] = min(values)
            else:
                best_metrics[key] = max(values)
        return best_metrics
    
    def close(self):
        """Close all tracking systems."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        if self.wandb_run:
            try:
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Failed to finish Weights & Biases run: {e}")

class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_dummy_data(num_samples: int = 1000, input_size: int = 10):
    """Create dummy data for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(num_samples, 1) * 0.1
    return X, y

def calculate_video_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate video-specific metrics."""
    # Simplified video metrics calculation
    mse = torch.mean((pred - target) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    
    # Simplified SSIM (in practice, use proper SSIM implementation)
    ssim = torch.exp(-mse)
    
    # Simplified LPIPS (in practice, use proper LPIPS implementation)
    lpips = mse
    
    return {
        "psnr": psnr.item(),
        "ssim": ssim.item(),
        "lpips": lpips.item()
    }

def demo_basic_tracking():
    """Demonstrate basic experiment tracking."""
    logger.info("=== Basic Experiment Tracking Demo ===")
    
    # Configuration
    config = ExperimentConfig(
        experiment_name="basic_demo",
        use_tensorboard=True,
        use_wandb=True,
        max_epochs=5,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(config)
    
    # Create model and data
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    X, y = create_dummy_data(1000, 10)
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "model_type": "SimpleModel",
        "optimizer": "Adam"
    })
    
    # Log model graph
    dummy_input = torch.randn(1, 10)
    tracker.log_model_graph(model, dummy_input)
    
    # Training loop
    model.train()
    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Simulate training batches
        for batch_idx in range(0, len(X), config.batch_size):
            batch_X = X[batch_idx:batch_idx + config.batch_size]
            batch_y = y[batch_idx:batch_idx + config.batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % config.log_frequency == 0:
                tracker.log_metrics({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })
        
        avg_loss = epoch_loss / num_batches
        
        # Validation (simplified)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X[:100])
            val_loss = criterion(val_outputs, y[:100]).item()
        
        # Log epoch metrics
        tracker.log_metrics({
            "train/epoch_loss": avg_loss,
            "val/loss": val_loss,
            "epoch": epoch
        })
        
        # Log system metrics
        tracker.log_system_metrics()
        
        # Save checkpoint
        if epoch % 2 == 0:
            tracker.save_checkpoint(model, optimizer, epoch, {
                "train_loss": avg_loss,
                "val_loss": val_loss
            })
        
        logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Get experiment summary
    duration = tracker.get_duration()
    best_metrics = tracker.get_best_metrics()
    
    logger.info(f"Experiment completed in {duration:.2f} seconds")
    logger.info(f"Best metrics: {best_metrics}")
    
    # Close tracker
    tracker.close()

def demo_video_tracking():
    """Demonstrate video-specific tracking."""
    logger.info("=== Video-Specific Tracking Demo ===")
    
    # Configuration
    config = ExperimentConfig(
        experiment_name="video_demo",
        use_tensorboard=True,
        use_wandb=True,
        max_epochs=3
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(config)
    
    # Simulate video generation process
    for step in range(50):
        # Simulate video generation
        pred_video = torch.randn(1, 3, 16, 64, 64)  # 16 frames, 64x64
        target_video = torch.randn(1, 3, 16, 64, 64)
        
        # Calculate video metrics
        video_metrics = calculate_video_metrics(pred_video, target_video)
        
        # Log video metrics
        tracker.log_video_metrics(
            psnr=video_metrics["psnr"],
            ssim=video_metrics["ssim"],
            lpips=video_metrics["lpips"],
            step=step
        )
        
        # Log generation metrics
        tracker.log_metrics({
            "generation/time_per_frame": 0.1 + np.random.random() * 0.05,
            "generation/memory_usage_gb": 2.0 + np.random.random() * 1.0,
            "generation/step": step
        })
        
        # Log system metrics
        if step % 10 == 0:
            tracker.log_system_metrics()
        
        time.sleep(0.1)  # Simulate processing time
    
    logger.info("Video tracking demo completed")
    tracker.close()

def demo_hyperparameter_sweep():
    """Demonstrate hyperparameter sweep with wandb."""
    if not WANDB_AVAILABLE:
        logger.warning("Weights & Biases not available for hyperparameter sweep")
        return
    
    logger.info("=== Hyperparameter Sweep Demo ===")
    
    # Define sweep configuration
    sweep_config = {
        "method": "random",
        "name": "video-opusclip-sweep",
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
            "hidden_size": {
                "values": [32, 64, 128]
            }
        }
    }
    
    def train_sweep():
        """Training function for sweep."""
        wandb.init()
        config = wandb.config
        
        # Create model with sweep parameters
        model = SimpleModel(hidden_size=config.hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        X, y = create_dummy_data(500, 10)
        
        # Training loop
        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx in range(0, len(X), config.batch_size):
                batch_X = X[batch_idx:batch_idx + config.batch_size]
                batch_y = y[batch_idx:batch_idx + config.batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X) // config.batch_size)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X[:50])
                val_loss = criterion(val_outputs, y[:50]).item()
            
            # Log metrics
            wandb.log({
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
    
    # Initialize sweep
    try:
        sweep_id = wandb.sweep(sweep_config, project="video-opusclip")
        logger.info(f"Sweep initialized: {sweep_id}")
        
        # Run sweep (limited to 3 runs for demo)
        wandb.agent(sweep_id, train_sweep, count=3)
        logger.info("Hyperparameter sweep completed")
        
    except Exception as e:
        logger.warning(f"Failed to run hyperparameter sweep: {e}")

def demo_tensorboard_only():
    """Demonstrate TensorBoard-only tracking."""
    logger.info("=== TensorBoard-Only Demo ===")
    
    # Configuration
    config = ExperimentConfig(
        experiment_name="tensorboard_demo",
        use_tensorboard=True,
        use_wandb=False,
        max_epochs=3
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(config)
    
    # Simple training simulation
    for step in range(100):
        # Simulate training metrics
        train_loss = 1.0 / (step + 1) + np.random.random() * 0.1
        val_loss = 1.2 / (step + 1) + np.random.random() * 0.1
        accuracy = 0.8 + step * 0.001 + np.random.random() * 0.01
        
        tracker.log_metrics({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "accuracy": accuracy
        })
        
        # Log histograms
        if tracker.tensorboard_writer and step % 20 == 0:
            weights = torch.randn(100)
            gradients = torch.randn(100)
            tracker.tensorboard_writer.add_histogram("weights", weights, step)
            tracker.tensorboard_writer.add_histogram("gradients", gradients, step)
    
    logger.info("TensorBoard demo completed")
    tracker.close()

def demo_wandb_only():
    """Demonstrate Weights & Biases-only tracking."""
    if not WANDB_AVAILABLE:
        logger.warning("Weights & Biases not available")
        return
    
    logger.info("=== Weights & Biases-Only Demo ===")
    
    # Configuration
    config = ExperimentConfig(
        experiment_name="wandb_demo",
        use_tensorboard=False,
        use_wandb=True,
        max_epochs=3
    )
    
    # Initialize tracker
    tracker = ExperimentTracker(config)
    
    # Simulate training with wandb-specific features
    for step in range(50):
        # Training metrics
        train_loss = 1.0 / (step + 1) + np.random.random() * 0.1
        val_loss = 1.2 / (step + 1) + np.random.random() * 0.1
        
        # Log metrics
        tracker.log_metrics({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "step": step
        })
        
        # Log images (simulated)
        if step % 10 == 0 and tracker.wandb_run:
            try:
                # Create sample images
                images = torch.randn(4, 3, 64, 64)
                wandb.log({
                    "sample_images": wandb.Image(images)
                }, step=step)
            except Exception as e:
                logger.warning(f"Failed to log images: {e}")
        
        # Log tables
        if step == 25 and tracker.wandb_run:
            try:
                import pandas as pd
                df = pd.DataFrame({
                    "epoch": range(5),
                    "loss": [1.0 / (i + 1) for i in range(5)],
                    "accuracy": [0.8 + i * 0.01 for i in range(5)]
                })
                wandb.log({"results_table": wandb.Table(dataframe=df)})
            except Exception as e:
                logger.warning(f"Failed to log table: {e}")
    
    logger.info("Weights & Biases demo completed")
    tracker.close()

def main():
    """Main function to run all demos."""
    logger.info("Starting Experiment Tracking Quick Start Guide")
    logger.info("=" * 50)
    
    # Check availability
    logger.info(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    logger.info(f"Weights & Biases available: {WANDB_AVAILABLE}")
    logger.info(f"psutil available: {PSUTIL_AVAILABLE}")
    
    # Run demos
    try:
        # Basic tracking demo
        demo_basic_tracking()
        print()
        
        # Video-specific tracking demo
        demo_video_tracking()
        print()
        
        # TensorBoard-only demo
        demo_tensorboard_only()
        print()
        
        # Weights & Biases-only demo
        demo_wandb_only()
        print()
        
        # Hyperparameter sweep demo (optional)
        if WANDB_AVAILABLE:
            demo_hyperparameter_sweep()
            print()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    
    logger.info("=" * 50)
    logger.info("Quick Start Guide completed!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Launch TensorBoard: tensorboard --logdir=runs --port=6006")
    logger.info("2. View Weights & Biases dashboard: https://wandb.ai")
    logger.info("3. Check the generated logs and checkpoints")
    logger.info("4. Integrate tracking into your Video-OpusClip training")

if __name__ == "__main__":
    main() 