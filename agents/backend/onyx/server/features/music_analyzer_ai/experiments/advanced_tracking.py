"""
Advanced Experiment Tracking
Enhanced experiment tracking with wandb and tensorboard
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class AdvancedExperimentTracker:
    """
    Advanced experiment tracking with multiple backends
    """
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "music-analyzer-ai",
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "./logs"
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_run = None
        self.tensorboard_writer = None
        
        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=project_name,
                name=experiment_name,
                config={}
            )
            logger.info("Initialized wandb tracking")
        
        # Initialize tensorboard
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / experiment_name
            self.tensorboard_writer = SummaryWriter(str(tb_dir))
            logger.info(f"Initialized tensorboard at {tb_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        if self.wandb_run:
            self.wandb_run.config.update(config)
        
        # Save config to file
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics"""
        if self.wandb_run:
            if step is not None:
                self.wandb_run.log(metrics, step=step)
            else:
                self.wandb_run.log(metrics)
        
        if self.tensorboard_writer:
            for key, value in metrics.items():
                if step is not None:
                    self.tensorboard_writer.add_scalar(key, value, step)
                else:
                    self.tensorboard_writer.add_scalar(key, value)
    
    def log_model_architecture(
        self,
        model: Any,
        input_shape: tuple
    ):
        """Log model architecture"""
        if self.wandb_run:
            try:
                import torch
                dummy_input = torch.randn(*input_shape)
                self.wandb_run.watch(model, log="all", log_freq=100)
            except Exception as e:
                logger.warning(f"Could not log model architecture: {e}")
        
        if self.tensorboard_writer:
            try:
                import torch
                dummy_input = torch.randn(*input_shape)
                self.tensorboard_writer.add_graph(model, dummy_input)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
    
    def log_images(
        self,
        images: List[Any],
        name: str,
        step: Optional[int] = None
    ):
        """Log images"""
        if self.wandb_run:
            if step is not None:
                self.wandb_run.log({name: [wandb.Image(img) for img in images]}, step=step)
            else:
                self.wandb_run.log({name: [wandb.Image(img) for img in images]})
        
        if self.tensorboard_writer:
            for i, img in enumerate(images):
                tag = f"{name}/{i}"
                if step is not None:
                    self.tensorboard_writer.add_image(tag, img, step)
                else:
                    self.tensorboard_writer.add_image(tag, img)
    
    def log_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        name: str,
        step: Optional[int] = None
    ):
        """Log audio"""
        if self.wandb_run:
            if step is not None:
                self.wandb_run.log({name: wandb.Audio(audio, sample_rate=sample_rate)}, step=step)
            else:
                self.wandb_run.log({name: wandb.Audio(audio, sample_rate=sample_rate)})
    
    def finish(self):
        """Finish experiment tracking"""
        if self.wandb_run:
            self.wandb_run.finish()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info("Finished experiment tracking")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization helper
    """
    
    def __init__(self, tracker: AdvancedExperimentTracker):
        self.tracker = tracker
        self.best_metrics: Dict[str, float] = {}
        self.best_config: Optional[Dict[str, Any]] = None
    
    def log_hyperparameter_search(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """Log hyperparameter search results"""
        # Update best metrics
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
                self.best_config = config.copy()
        
        # Log to tracker
        self.tracker.log_config(config)
        self.tracker.log_metrics(metrics)
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get best configuration"""
        return self.best_config
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics"""
        return self.best_metrics.copy()

