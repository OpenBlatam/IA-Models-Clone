from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

    from experiment_tracking import (
from .experiment_tracker import (
from .metrics_tracker import (
from .checkpoint_manager import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Experiment Tracking Module
==========================

This module provides comprehensive experiment tracking and model checkpointing
for AI video generation experiments.

Features:
- WandB integration for experiment tracking
- TensorBoard integration for visualization
- Custom logging and metrics tracking
- Model checkpointing with versioning
- Experiment metadata management
- Performance monitoring
- Artifact management
- Video-specific metrics tracking
- Checkpoint management with compression

Usage:
        create_experiment_tracker,
        create_metrics_tracker,
        create_checkpoint_manager
    )
    
    # Create experiment tracker
    tracker = create_experiment_tracker("my_experiment", use_wandb=True)
    
    # Create metrics tracker
    metrics = create_metrics_tracker()
    
    # Create checkpoint manager
    checkpoint_mgr = create_checkpoint_manager("checkpoints")
    
    # Use in training loop
    for step in range(1000):
        loss = train_step(model, batch)
        tracker.log_metrics({"loss": loss}, step)
        
        if step % 100 == 0:
            checkpoint_mgr.save_checkpoint(model, optimizer, {"loss": loss}, step=step)
"""

    ExperimentConfig,
    ExperimentTracker,
    CheckpointInfo,
    create_experiment_tracker
)

    MetricValue,
    MetricDefinition,
    MetricsTracker,
    VideoMetricsTracker,
    PerformanceMonitor,
    create_metrics_tracker,
    create_video_metrics_tracker,
    create_performance_monitor
)

    CheckpointMetadata,
    CheckpointConfig,
    CheckpointManager,
    create_checkpoint_manager
)

# Convenience imports
__all__ = [
    # Experiment tracking
    "ExperimentConfig",
    "ExperimentTracker",
    "CheckpointInfo",
    "create_experiment_tracker",
    
    # Metrics tracking
    "MetricValue",
    "MetricDefinition",
    "MetricsTracker",
    "VideoMetricsTracker",
    "PerformanceMonitor",
    "create_metrics_tracker",
    "create_video_metrics_tracker",
    "create_performance_monitor",
    
    # Checkpoint management
    "CheckpointMetadata",
    "CheckpointConfig",
    "CheckpointManager",
    "create_checkpoint_manager"
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Video Team"
__description__ = "Experiment tracking and checkpointing for AI video generation"

# Quick start functions
def setup_experiment_tracking(
    experiment_name: str,
    project_name: str = "ai_video_generation",
    use_wandb: bool = False,
    use_tensorboard: bool = True,
    checkpoint_dir: str = "checkpoints",
    **kwargs
) -> tuple:
    """
    Set up complete experiment tracking system.
    
    Returns:
        tuple: (experiment_tracker, metrics_tracker, checkpoint_manager)
    """
    # Create experiment tracker
    tracker = create_experiment_tracker(
        experiment_name=experiment_name,
        project_name=project_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        **kwargs
    )
    
    # Create metrics tracker
    metrics = create_metrics_tracker()
    
    # Create checkpoint manager
    checkpoint_mgr = create_checkpoint_manager(
        checkpoint_dir=checkpoint_dir,
        save_frequency=kwargs.get("save_frequency", 1000),
        max_checkpoints=kwargs.get("max_checkpoints", 5)
    )
    
    return tracker, metrics, checkpoint_mgr


def create_video_experiment_tracker(
    experiment_name: str,
    use_wandb: bool = False,
    use_tensorboard: bool = True,
    **kwargs
) -> tuple:
    """
    Create experiment tracking system specifically for video generation.
    
    Returns:
        tuple: (experiment_tracker, video_metrics_tracker, checkpoint_manager)
    """
    # Create base trackers
    tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        **kwargs
    )
    
    # Create video-specific metrics tracker
    video_metrics = create_video_metrics_tracker(metrics)
    
    return tracker, video_metrics, checkpoint_mgr


def log_training_step(
    tracker: ExperimentTracker,
    metrics_tracker: MetricsTracker,
    checkpoint_manager: CheckpointManager,
    model,
    optimizer,
    loss: float,
    step: int,
    epoch: int = 0,
    additional_metrics: dict = None,
    save_checkpoint: bool = False
):
    """
    Log a single training step with all tracking systems.
    
    Args:
        tracker: Experiment tracker
        metrics_tracker: Metrics tracker
        checkpoint_manager: Checkpoint manager
        model: Model to save
        optimizer: Optimizer to save
        loss: Training loss
        step: Current step
        epoch: Current epoch
        additional_metrics: Additional metrics to log
        save_checkpoint: Whether to save checkpoint
    """
    # Prepare metrics
    metrics = {"loss": loss}
    if additional_metrics:
        metrics.update(additional_metrics)
    
    # Log to experiment tracker
    tracker.log_metrics(metrics, step)
    tracker.update_step(step)
    tracker.update_epoch(epoch)
    
    # Log to metrics tracker
    metrics_tracker.log_metrics(metrics, step, epoch)
    
    # Save checkpoint if requested
    if save_checkpoint:
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            epoch=epoch,
            step=step
        )


def log_video_generation(
    video_metrics_tracker: VideoMetricsTracker,
    psnr: float = None,
    ssim: float = None,
    lpips: float = None,
    generation_time: float = None,
    step: int = 0,
    epoch: int = 0
):
    """
    Log video generation metrics.
    
    Args:
        video_metrics_tracker: Video metrics tracker
        psnr: Peak Signal-to-Noise Ratio
        ssim: Structural Similarity Index
        lpips: Learned Perceptual Image Patch Similarity
        generation_time: Time taken to generate video
        step: Current step
        epoch: Current epoch
    """
    video_metrics_tracker.log_video_metrics(
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        generation_time=generation_time,
        step=step,
        epoch=epoch
    )


def get_experiment_summary(
    tracker: ExperimentTracker,
    metrics_tracker: MetricsTracker,
    checkpoint_manager: CheckpointManager
) -> dict:
    """
    Get comprehensive experiment summary.
    
    Returns:
        dict: Complete experiment summary
    """
    # Get summaries from each component
    tracker_summary = tracker.get_experiment_summary()
    metrics_summary = metrics_tracker.get_metric_summary()
    checkpoint_summary = checkpoint_manager.get_checkpoint_summary()
    
    # Combine summaries
    summary = {
        "experiment": tracker_summary,
        "metrics": metrics_summary,
        "checkpoints": checkpoint_summary,
        "timestamp": datetime.now().isoformat()
    }
    
    return summary


def export_experiment_data(
    tracker: ExperimentTracker,
    metrics_tracker: MetricsTracker,
    checkpoint_manager: CheckpointManager,
    export_dir: str = "experiment_exports"
):
    """
    Export all experiment data.
    
    Args:
        tracker: Experiment tracker
        metrics_tracker: Metrics tracker
        checkpoint_manager: Checkpoint manager
        export_dir: Directory to export data
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Export metrics
    metrics_tracker.export_metrics(export_path / "metrics.json", "json")
    
    # Export experiment summary
    summary = get_experiment_summary(tracker, metrics_tracker, checkpoint_manager)
    with open(export_path / "experiment_summary.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(summary, f, indent=2)
    
    # Export checkpoints list
    checkpoints = checkpoint_manager.list_checkpoints()
    checkpoints_data = [c.to_dict() for c in checkpoints]
    with open(export_path / "checkpoints.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(checkpoints_data, f, indent=2)
    
    logger.info(f"Experiment data exported to {export_path}")


# Example usage
if __name__ == "__main__":
    print("🔬 Experiment Tracking System")
    print("=" * 40)
    
    # Set up complete tracking system
    tracker, metrics, checkpoint_mgr = setup_experiment_tracking(
        "test_experiment",
        use_wandb=False,
        use_tensorboard=True
    )
    
    # Mock model and optimizer
    class MockModel:
        def state_dict(self) -> Any:
            return {"weights": [1, 2, 3]}
        
        def load_state_dict(self, state_dict) -> Any:
            pass
    
    class MockOptimizer:
        def state_dict(self) -> Any:
            return {"lr": 1e-4}
        
        def load_state_dict(self, state_dict) -> Any:
            pass
    
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Simulate training
    for step in range(100):
        loss = 1.0 / (step + 1)
        
        # Log training step
        log_training_step(
            tracker=tracker,
            metrics_tracker=metrics,
            checkpoint_manager=checkpoint_mgr,
            model=model,
            optimizer=optimizer,
            loss=loss,
            step=step,
            save_checkpoint=(step % 20 == 0)
        )
    
    # Get experiment summary
    summary = get_experiment_summary(tracker, metrics, checkpoint_mgr)
    print(f"Experiment summary: {summary}")
    
    # Export data
    export_experiment_data(tracker, metrics, checkpoint_mgr)
    
    # Close tracker
    tracker.close()
    
    print("✅ Experiment tracking example completed!") 