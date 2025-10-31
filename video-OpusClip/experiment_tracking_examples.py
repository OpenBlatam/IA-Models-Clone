#!/usr/bin/env python3
"""
Experiment Tracking Examples: TensorBoard & Weights & Biases
===========================================================

Comprehensive examples demonstrating experiment tracking with TensorBoard
and Weights & Biases for the Video-OpusClip system.

Examples include:
- Basic training tracking
- Video generation tracking
- Hyperparameter optimization
- Model comparison
- Performance monitoring
- Custom metrics
- Artifact management
- Distributed training tracking
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import pickle
from datetime import datetime
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Configuration
@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""
    experiment_name: str
    project_name: str = "video-opusclip"
    use_tensorboard: bool = True
    use_wandb: bool = True
    log_frequency: int = 100
    save_frequency: int = 1000
    max_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    device: str = "cpu"
    tags: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self):
        return asdict(self)

class AdvancedExperimentTracker:
    """Advanced experiment tracker with comprehensive features."""
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.wandb_run = None
        self.tensorboard_writer = None
        self.start_time = time.time()
        self.current_step = 0
        self.current_epoch = 0
        self.metrics_history = {}
        self.best_metrics = {}
        self.checkpoints = []
        
        # Performance monitoring
        self.performance_metrics = {
            "gpu_utilization": [],
            "memory_usage": [],
            "training_speed": [],
            "throughput": []
        }
        
        # Threading for async logging
        self.log_queue = queue.Queue()
        self.log_thread = None
        self._start_logging_thread()
        
        self._initialize_tracking()
        logger.info(f"Advanced experiment tracker initialized: {config.experiment_name}")
    
    def _start_logging_thread(self):
        """Start background thread for async logging."""
        self.log_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.log_thread.start()
    
    def _logging_worker(self):
        """Background worker for async logging."""
        while True:
            try:
                log_data = self.log_queue.get(timeout=1)
                if log_data is None:  # Shutdown signal
                    break
                
                metrics, step = log_data
                self._log_metrics_sync(metrics, step)
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Logging worker error: {e}")
    
    def _initialize_tracking(self):
        """Initialize tracking systems."""
        # Create directories
        Path("runs").mkdir(exist_ok=True)
        Path("checkpoints").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("artifacts").mkdir(exist_ok=True)
        
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
                    tags=self.config.tags,
                    notes=self.config.notes,
                    mode="online" if self.config.use_wandb else "disabled"
                )
                logger.info(f"Weights & Biases initialized: {self.wandb_run.id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self.wandb_run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics asynchronously."""
        if step is None:
            step = self.current_step
        
        # Update metrics history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
            
            # Update best metrics
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                if "loss" in key.lower():
                    self.best_metrics[key] = min(self.best_metrics[key], value)
                else:
                    self.best_metrics[key] = max(self.best_metrics[key], value)
        
        # Queue for async logging
        self.log_queue.put((metrics, step))
        self.current_step += 1
    
    def _log_metrics_sync(self, metrics: Dict[str, float], step: int):
        """Synchronously log metrics to tracking systems."""
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
    
    def log_video_metrics(self, psnr: float, ssim: float, lpips: float, 
                         fid: Optional[float] = None, inception_score: Optional[float] = None,
                         step: Optional[int] = None):
        """Log video-specific metrics."""
        video_metrics = {
            "video/psnr": psnr,
            "video/ssim": ssim,
            "video/lpips": lpips
        }
        
        if fid is not None:
            video_metrics["video/fid"] = fid
        if inception_score is not None:
            video_metrics["video/inception_score"] = inception_score
        
        self.log_metrics(video_metrics, step)
    
    def log_performance_metrics(self):
        """Log system performance metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        # CPU and memory metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        performance_metrics = {
            "performance/cpu_usage": cpu_usage,
            "performance/memory_usage_percent": memory.percent,
            "performance/memory_available_gb": memory.available / (1024**3),
            "performance/memory_used_gb": memory.used / (1024**3)
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            gpu_utilization = torch.cuda.utilization()
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            
            performance_metrics.update({
                "performance/gpu_utilization": gpu_utilization,
                "performance/gpu_memory_allocated_gb": gpu_memory_allocated,
                "performance/gpu_memory_reserved_gb": gpu_memory_reserved
            })
        
        # GPUtil metrics if available
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    performance_metrics[f"performance/gpu_{i}_load"] = gpu.load * 100
                    performance_metrics[f"performance/gpu_{i}_memory_used"] = gpu.memoryUsed
                    performance_metrics[f"performance/gpu_{i}_memory_total"] = gpu.memoryTotal
                    performance_metrics[f"performance/gpu_{i}_temperature"] = gpu.temperature
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
        
        self.log_metrics(performance_metrics)
    
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
    
    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient statistics."""
        if self.tensorboard_writer:
            try:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.tensorboard_writer.add_histogram(f"gradients/{name}", param.grad, step)
                        self.tensorboard_writer.add_histogram(f"weights/{name}", param.data, step)
            except Exception as e:
                logger.warning(f"Failed to log gradients: {e}")
    
    def log_images(self, images: torch.Tensor, name: str, step: int):
        """Log images to tracking systems."""
        # TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_images(name, images, step)
            except Exception as e:
                logger.warning(f"Failed to log images to TensorBoard: {e}")
        
        # Weights & Biases
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Image(images)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log images to Weights & Biases: {e}")
    
    def log_video(self, video: torch.Tensor, name: str, step: int, fps: int = 8):
        """Log video to tracking systems."""
        # TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_video(name, video, step, fps=fps)
            except Exception as e:
                logger.warning(f"Failed to log video to TensorBoard: {e}")
        
        # Weights & Biases
        if self.wandb_run:
            try:
                # Save video to file first
                video_path = f"artifacts/{name}_step_{step}.mp4"
                # Note: In practice, you'd use a proper video writer here
                wandb.log({name: wandb.Video(video_path)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log video to Weights & Biases: {e}")
    
    def log_embeddings(self, embeddings: torch.Tensor, metadata: List[str], step: int):
        """Log embeddings for visualization."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_embedding(embeddings, metadata=metadata, global_step=step)
            except Exception as e:
                logger.warning(f"Failed to log embeddings: {e}")
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       epoch: int = None, step: int = None, 
                       metrics: Dict[str, float] = None, is_best: bool = False):
        """Save model checkpoint with comprehensive metadata."""
        if epoch is None:
            epoch = self.current_epoch
        if step is None:
            step = self.current_step
        if metrics is None:
            metrics = {}
        
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metrics": self.best_metrics,
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "is_best": is_best
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch}_step_{step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint separately
        if is_best:
            best_path = f"checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        # Log checkpoint
        if self.wandb_run:
            try:
                wandb.save(checkpoint_path)
                if is_best:
                    wandb.save(best_path)
            except Exception as e:
                logger.warning(f"Failed to log checkpoint: {e}")
        
        # Track checkpoints
        self.checkpoints.append({
            "path": checkpoint_path,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": checkpoint["timestamp"]
        })
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "artifact"):
        """Log artifact to Weights & Biases."""
        if self.wandb_run:
            try:
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type=artifact_type,
                    description=f"Artifact from {self.config.experiment_name}"
                )
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
                logger.info(f"Artifact logged: {artifact_name}")
            except Exception as e:
                logger.warning(f"Failed to log artifact: {e}")
    
    def create_custom_plots(self):
        """Create and log custom plots."""
        if not self.metrics_history:
            return
        
        # Create loss plot
        if "train/loss" in self.metrics_history and "val/loss" in self.metrics_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.metrics_history["train/loss"], label="Training Loss")
            ax.plot(self.metrics_history["val/loss"], label="Validation Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend()
            ax.grid(True)
            
            # Save and log plot
            plot_path = f"artifacts/loss_plot_{self.current_step}.png"
            plt.savefig(plot_path)
            plt.close()
            
            if self.wandb_run:
                try:
                    wandb.log({"loss_plot": wandb.Image(plot_path)})
                except Exception as e:
                    logger.warning(f"Failed to log loss plot: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        duration = time.time() - self.start_time
        
        summary = {
            "experiment_name": self.config.experiment_name,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
            "total_steps": self.current_step,
            "total_epochs": self.current_epoch,
            "best_metrics": self.best_metrics,
            "total_checkpoints": len(self.checkpoints),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def close(self):
        """Close all tracking systems."""
        # Stop logging thread
        if self.log_thread:
            self.log_queue.put(None)
            self.log_thread.join(timeout=5)
        
        # Create final summary
        summary = self.get_experiment_summary()
        
        # Log summary
        if self.wandb_run:
            try:
                wandb.log({"experiment_summary": wandb.Table(dataframe=pd.DataFrame([summary]))})
            except Exception as e:
                logger.warning(f"Failed to log experiment summary: {e}")
        
        # Close TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        # Close Weights & Biases
        if self.wandb_run:
            try:
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Failed to finish Weights & Biases run: {e}")
        
        # Save summary to file
        summary_path = f"logs/experiment_summary_{self.config.experiment_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment summary saved: {summary_path}")
        logger.info(f"Experiment completed: {self.config.experiment_name}")

# Example Models
class VideoGenerationModel(nn.Module):
    """Example video generation model."""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 1024, 
                 output_frames: int = 16, output_size: int = 64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_frames = output_frames
        self.output_size = output_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_frames * 3 * output_size * output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reshape to video format: (batch, frames, channels, height, width)
        video = decoded.view(-1, self.output_frames, 3, self.output_size, self.output_size)
        return video

class TransformerModel(nn.Module):
    """Example transformer model for video processing."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        transformed = self.transformer(x)
        output = self.output_proj(transformed)
        return output

# Example Functions
def example_basic_training():
    """Example of basic training with experiment tracking."""
    logger.info("=== Basic Training Example ===")
    
    # Configuration
    config = TrackingConfig(
        experiment_name="basic_training_example",
        use_tensorboard=True,
        use_wandb=True,
        max_epochs=5,
        learning_rate=0.001,
        batch_size=32,
        tags=["basic", "training", "example"]
    )
    
    # Initialize tracker
    tracker = AdvancedExperimentTracker(config)
    
    # Create model and data
    model = VideoGenerationModel()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    criterion = nn.MSELoss()
    
    # Create dummy data
    X = torch.randn(100, 512)
    y = torch.randn(100, 16, 3, 64, 64)
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "model_type": "VideoGenerationModel",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR"
    })
    
    # Log model graph
    dummy_input = torch.randn(1, 512)
    tracker.log_model_graph(model, dummy_input)
    
    # Training loop
    best_val_loss = float('inf')
    model.train()
    
    for epoch in range(config.max_epochs):
        tracker.current_epoch = epoch
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx in range(0, len(X), config.batch_size):
            batch_X = X[batch_idx:batch_idx + config.batch_size]
            batch_y = y[batch_idx:batch_idx + config.batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Log gradients
            if batch_idx % config.log_frequency == 0:
                tracker.log_gradients(model, tracker.current_step)
            
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X[:20])
            val_loss = criterion(val_outputs, y[:20]).item()
        
        # Log epoch metrics
        tracker.log_metrics({
            "train/epoch_loss": avg_loss,
            "val/loss": val_loss,
            "epoch": epoch
        })
        
        # Log performance metrics
        tracker.log_performance_metrics()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % 2 == 0 or is_best:
            tracker.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"val_loss": val_loss},
                is_best=is_best
            )
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Create custom plots
    tracker.create_custom_plots()
    
    # Close tracker
    tracker.close()

def example_video_generation_tracking():
    """Example of video generation tracking."""
    logger.info("=== Video Generation Tracking Example ===")
    
    # Configuration
    config = TrackingConfig(
        experiment_name="video_generation_tracking",
        use_tensorboard=True,
        use_wandb=True,
        tags=["video", "generation", "tracking"]
    )
    
    # Initialize tracker
    tracker = AdvancedExperimentTracker(config)
    
    # Create model
    model = VideoGenerationModel()
    model.eval()
    
    # Simulate video generation process
    for step in range(20):
        # Generate video
        prompt = torch.randn(1, 512)
        with torch.no_grad():
            generated_video = model(prompt)
        
        # Calculate video metrics (simplified)
        target_video = torch.randn_like(generated_video)
        
        # PSNR
        mse = torch.mean((generated_video - target_video) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        
        # SSIM (simplified)
        ssim = torch.exp(-mse)
        
        # LPIPS (simplified)
        lpips = mse
        
        # Log video metrics
        tracker.log_video_metrics(
            psnr=psnr.item(),
            ssim=ssim.item(),
            lpips=lpips.item(),
            step=step
        )
        
        # Log generation metrics
        tracker.log_metrics({
            "generation/step": step,
            "generation/video_length_frames": generated_video.shape[1],
            "generation/video_resolution": f"{generated_video.shape[3]}x{generated_video.shape[4]}"
        })
        
        # Log sample video frames
        if step % 5 == 0:
            # Log first frame as image
            first_frame = generated_video[0, 0]  # First batch, first frame
            tracker.log_images(first_frame.unsqueeze(0), "generated_frame", step)
            
            # Log video
            tracker.log_video(generated_video, "generated_video", step)
        
        # Log performance metrics
        tracker.log_performance_metrics()
        
        time.sleep(0.1)  # Simulate processing time
    
    tracker.close()

def example_hyperparameter_optimization():
    """Example of hyperparameter optimization with wandb."""
    if not WANDB_AVAILABLE:
        logger.warning("Weights & Biases not available for hyperparameter optimization")
        return
    
    logger.info("=== Hyperparameter Optimization Example ===")
    
    # Define sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "name": "video-model-optimization",
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "learning_rate": {
                "min": 0.0001,
                "max": 0.01,
                "distribution": "log_uniform"
            },
            "batch_size": {
                "values": [16, 32, 64, 128]
            },
            "hidden_size": {
                "values": [256, 512, 1024, 2048]
            },
            "dropout": {
                "min": 0.0,
                "max": 0.5
            }
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10
        }
    }
    
    def train_with_sweep():
        """Training function for hyperparameter sweep."""
        wandb.init()
        config = wandb.config
        
        # Create model with sweep parameters
        model = VideoGenerationModel(
            input_size=512,
            hidden_size=config.hidden_size,
            output_frames=16,
            output_size=64
        )
        
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        # Create dummy data
        X = torch.randn(200, 512)
        y = torch.randn(200, 16, 3, 64, 64)
        
        # Training loop
        for epoch in range(10):
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
                val_outputs = model(X[:20])
                val_loss = criterion(val_outputs, y[:20]).item()
            
            # Log metrics
            wandb.log({
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
    
    # Initialize sweep
    try:
        sweep_id = wandb.sweep(sweep_config, project="video-opusclip")
        logger.info(f"Hyperparameter sweep initialized: {sweep_id}")
        
        # Run sweep (limited to 5 runs for demo)
        wandb.agent(sweep_id, train_with_sweep, count=5)
        logger.info("Hyperparameter optimization completed")
        
    except Exception as e:
        logger.error(f"Failed to run hyperparameter optimization: {e}")

def example_model_comparison():
    """Example of comparing multiple models."""
    logger.info("=== Model Comparison Example ===")
    
    # Define models to compare
    models = {
        "VideoGenerationModel": VideoGenerationModel(),
        "TransformerModel": TransformerModel()
    }
    
    # Configuration
    config = TrackingConfig(
        experiment_name="model_comparison",
        use_tensorboard=True,
        use_wandb=True,
        tags=["comparison", "models"]
    )
    
    # Initialize tracker
    tracker = AdvancedExperimentTracker(config)
    
    # Create dummy data
    X = torch.randn(100, 512)
    y = torch.randn(100, 16, 3, 64, 64)
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Testing model: {model_name}")
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model_results = {
            "train_losses": [],
            "val_losses": [],
            "training_time": 0,
            "model_size": sum(p.numel() for p in model.parameters())
        }
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(3):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx in range(0, len(X), 32):
                batch_X = X[batch_idx:batch_idx + 32]
                batch_y = y[batch_idx:batch_idx + 32]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X) // 32)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X[:20])
                val_loss = criterion(val_outputs, y[:20]).item()
            
            model_results["train_losses"].append(avg_loss)
            model_results["val_losses"].append(val_loss)
            
            # Log metrics with model prefix
            tracker.log_metrics({
                f"{model_name}/train_loss": avg_loss,
                f"{model_name}/val_loss": val_loss,
                f"{model_name}/epoch": epoch
            })
        
        model_results["training_time"] = time.time() - start_time
        results[model_name] = model_results
        
        logger.info(f"{model_name}: Final Val Loss = {val_loss:.4f}, "
                   f"Training Time = {model_results['training_time']:.2f}s, "
                   f"Model Size = {model_results['model_size']:,} parameters")
    
    # Log comparison summary
    comparison_summary = {
        "model_comparison": {
            "best_model": min(results.keys(), key=lambda k: results[k]["val_losses"][-1]),
            "results": results
        }
    }
    
    if tracker.wandb_run:
        try:
            wandb.log({"comparison_summary": wandb.Table(
                dataframe=pd.DataFrame([comparison_summary])
            )})
        except Exception as e:
            logger.warning(f"Failed to log comparison summary: {e}")
    
    tracker.close()

def example_distributed_training_tracking():
    """Example of tracking distributed training."""
    logger.info("=== Distributed Training Tracking Example ===")
    
    # Configuration
    config = TrackingConfig(
        experiment_name="distributed_training",
        use_tensorboard=True,
        use_wandb=True,
        tags=["distributed", "training"]
    )
    
    # Initialize tracker
    tracker = AdvancedExperimentTracker(config)
    
    # Simulate distributed training metrics
    num_nodes = 4
    num_gpus_per_node = 2
    
    for step in range(50):
        # Simulate distributed training metrics
        distributed_metrics = {
            "distributed/total_nodes": num_nodes,
            "distributed/total_gpus": num_nodes * num_gpus_per_node,
            "distributed/global_step": step,
            "distributed/effective_batch_size": 32 * num_nodes * num_gpus_per_node
        }
        
        # Simulate per-node metrics
        for node in range(num_nodes):
            for gpu in range(num_gpus_per_node):
                distributed_metrics[f"distributed/node_{node}_gpu_{gpu}_utilization"] = np.random.uniform(0.7, 0.95)
                distributed_metrics[f"distributed/node_{node}_gpu_{gpu}_memory_used"] = np.random.uniform(0.6, 0.9)
        
        # Training metrics
        distributed_metrics.update({
            "train/loss": 1.0 / (step + 1) + np.random.random() * 0.1,
            "train/learning_rate": 0.001 * (0.95 ** step),
            "train/gradient_norm": np.random.uniform(0.1, 2.0)
        })
        
        tracker.log_metrics(distributed_metrics)
        
        # Log performance metrics
        tracker.log_performance_metrics()
        
        time.sleep(0.1)
    
    tracker.close()

def main():
    """Main function to run all examples."""
    logger.info("Starting Experiment Tracking Examples")
    logger.info("=" * 50)
    
    # Check availability
    logger.info(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    logger.info(f"Weights & Biases available: {WANDB_AVAILABLE}")
    logger.info(f"psutil available: {PSUTIL_AVAILABLE}")
    logger.info(f"GPUtil available: {GPUTIL_AVAILABLE}")
    
    # Run examples
    try:
        # Basic training example
        example_basic_training()
        print()
        
        # Video generation tracking example
        example_video_generation_tracking()
        print()
        
        # Model comparison example
        example_model_comparison()
        print()
        
        # Distributed training tracking example
        example_distributed_training_tracking()
        print()
        
        # Hyperparameter optimization example
        example_hyperparameter_optimization()
        print()
        
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.error(f"Examples failed: {e}")
    
    logger.info("=" * 50)
    logger.info("Experiment Tracking Examples completed!")
    logger.info("")
    logger.info("Generated files:")
    logger.info("- runs/: TensorBoard logs")
    logger.info("- checkpoints/: Model checkpoints")
    logger.info("- logs/: Experiment summaries")
    logger.info("- artifacts/: Generated plots and videos")

if __name__ == "__main__":
    main() 