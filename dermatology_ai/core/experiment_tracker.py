"""
Experiment Tracker - Track ML experiments and model performance
Enhanced with wandb and tensorboard support
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import experiment tracking libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.debug("wandb not available")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.debug("tensorboard not available")


class ExperimentStatus(Enum):
    """Experiment status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    experiment_id: str
    name: str
    description: str
    model_type: str
    hyperparameters: Dict[str, Any]
    dataset_info: Dict[str, Any]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment"""
    experiment_id: str
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ExperimentTracker:
    """
    Track ML experiments with wandb and tensorboard support:
    - Experiment configuration
    - Training metrics
    - Model checkpoints
    - Hyperparameter search
    - Real-time visualization
    """
    
    def __init__(
        self,
        experiments_dir: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None
    ):
        self.experiments_dir = Path(experiments_dir) if experiments_dir else Path("./experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.metrics: Dict[str, List[ExperimentMetrics]] = {}
        self.current_experiment: Optional[str] = None
        
        # Initialize tracking backends
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.wandb_runs: Dict[str, Any] = {}
        self.tensorboard_writers: Dict[str, Any] = {}
        self.wandb_project = wandb_project or "dermatology-ai"
        self.wandb_entity = wandb_entity
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment with tracking backends"""
        experiment_path = self.experiments_dir / config.experiment_id
        experiment_path.mkdir(exist_ok=True)
        
        # Save config
        config_path = experiment_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        self.experiments[config.experiment_id] = config
        self.metrics[config.experiment_id] = []
        self.current_experiment = config.experiment_id
        
        # Initialize wandb
        if self.use_wandb:
            try:
                wandb_run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=config.name,
                    id=config.experiment_id,
                    config=config.hyperparameters,
                    reinit=True
                )
                self.wandb_runs[config.experiment_id] = wandb_run
                logger.info(f"Initialized wandb for experiment: {config.experiment_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
        
        # Initialize tensorboard
        if self.use_tensorboard:
            try:
                tb_dir = experiment_path / "tensorboard"
                tb_writer = SummaryWriter(log_dir=str(tb_dir))
                self.tensorboard_writers[config.experiment_id] = tb_writer
                logger.info(f"Initialized tensorboard for experiment: {config.experiment_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize tensorboard: {e}")
        
        logger.info(f"Created experiment: {config.experiment_id} - {config.name}")
        return config.experiment_id
    
    def log_metrics(self, metrics: ExperimentMetrics):
        """Log metrics for current experiment with all backends"""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        if self.current_experiment not in self.metrics:
            self.metrics[self.current_experiment] = []
        
        self.metrics[self.current_experiment].append(metrics)
        
        # Save to file
        metrics_path = (
            self.experiments_dir / self.current_experiment / "metrics.jsonl"
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
        
        # Log to wandb
        if self.use_wandb and self.current_experiment in self.wandb_runs:
            try:
                wandb_log = {
                    "epoch": metrics.epoch,
                    "train_loss": metrics.train_loss,
                }
                if metrics.val_loss is not None:
                    wandb_log["val_loss"] = metrics.val_loss
                if metrics.train_accuracy is not None:
                    wandb_log["train_accuracy"] = metrics.train_accuracy
                if metrics.val_accuracy is not None:
                    wandb_log["val_accuracy"] = metrics.val_accuracy
                if metrics.learning_rate is not None:
                    wandb_log["learning_rate"] = metrics.learning_rate
                
                self.wandb_runs[self.current_experiment].log(wandb_log)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        # Log to tensorboard
        if self.use_tensorboard and self.current_experiment in self.tensorboard_writers:
            try:
                writer = self.tensorboard_writers[self.current_experiment]
                writer.add_scalar("Loss/Train", metrics.train_loss, metrics.epoch)
                
                if metrics.val_loss is not None:
                    writer.add_scalar("Loss/Validation", metrics.val_loss, metrics.epoch)
                if metrics.train_accuracy is not None:
                    writer.add_scalar("Accuracy/Train", metrics.train_accuracy, metrics.epoch)
                if metrics.val_accuracy is not None:
                    writer.add_scalar("Accuracy/Validation", metrics.val_accuracy, metrics.epoch)
                if metrics.learning_rate is not None:
                    writer.add_scalar("LearningRate", metrics.learning_rate, metrics.epoch)
                
                writer.flush()
            except Exception as e:
                logger.warning(f"Failed to log to tensorboard: {e}")
    
    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Save model checkpoint"""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        checkpoint_dir = self.experiments_dir / self.current_experiment / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        try:
            import torch
            torch.save({
                "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model,
                "epoch": epoch,
                "experiment_id": self.current_experiment,
                **(additional_info or {})
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        except ImportError:
            import pickle
            with open(checkpoint_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump({
                    "model": model,
                    "epoch": epoch,
                    "experiment_id": self.current_experiment,
                    **(additional_info or {})
                }, f)
            logger.info(f"Saved checkpoint: {checkpoint_path.with_suffix('.pkl')}")
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        experiment_metrics = self.metrics.get(experiment_id, [])
        
        summary = {
            "experiment_id": experiment_id,
            "name": config.name,
            "description": config.description,
            "model_type": config.model_type,
            "hyperparameters": config.hyperparameters,
            "created_at": config.created_at,
            "total_epochs": len(experiment_metrics),
            "metrics": [asdict(m) for m in experiment_metrics[-10:]]  # Last 10 metrics
        }
        
        if experiment_metrics:
            last_metric = experiment_metrics[-1]
            summary["latest_metrics"] = asdict(last_metric)
        
        return summary
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        return [
            {
                "experiment_id": exp_id,
                "name": config.name,
                "model_type": config.model_type,
                "created_at": config.created_at,
                "total_epochs": len(self.metrics.get(exp_id, []))
            }
            for exp_id, config in self.experiments.items()
        ]


