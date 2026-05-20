"""
Experiment Tracking for AI Models

Provides experiment tracking using wandb, tensorboard, or mlflow
for monitoring model performance and training metrics.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Base class for experiment tracking
    
    Supports multiple backends: wandb, tensorboard, mlflow
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        backend: str = "wandb",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment tracker
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment (auto-generated if None)
            backend: Tracking backend (wandb, tensorboard, mlflow)
            config: Configuration dictionary
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.backend = backend
        self.config = config or {}
        self.tracker = None
        self._init_tracker()
    
    def _init_tracker(self) -> None:
        """Initialize the tracking backend"""
        try:
            if self.backend == "wandb":
                self._init_wandb()
            elif self.backend == "tensorboard":
                self._init_tensorboard()
            elif self.backend == "mlflow":
                self._init_mlflow()
            else:
                logger.warning(f"Unknown backend: {self.backend}, using no-op tracker")
                self.tracker = None
        except ImportError as e:
            logger.warning(f"Failed to initialize {self.backend}: {e}. Using no-op tracker")
            self.tracker = None
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases"""
        try:
            import wandb
            
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                reinit=True
            )
            self.tracker = wandb
            logger.info(f"Initialized wandb tracking: {self.project_name}")
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")
    
    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = Path("runs") / self.project_name / (self.experiment_name or "experiment")
            self.tracker = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"Initialized TensorBoard: {log_dir}")
        except ImportError:
            raise ImportError("tensorboard not installed. Install with: pip install tensorboard")
    
    def _init_mlflow(self) -> None:
        """Initialize MLflow"""
        try:
            import mlflow
            
            mlflow.set_experiment(self.project_name)
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
            
            mlflow.start_run()
            if self.config:
                mlflow.log_params(self.config)
            
            self.tracker = mlflow
            logger.info(f"Initialized MLflow tracking: {self.project_name}")
        except ImportError:
            raise ImportError("mlflow not installed. Install with: pip install mlflow")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric"""
        if self.tracker is None:
            return
        
        try:
            if self.backend == "wandb":
                self.tracker.log({name: value}, step=step)
            elif self.backend == "tensorboard":
                self.tracker.add_scalar(name, value, step or 0)
            elif self.backend == "mlflow":
                self.tracker.log_metric(name, value, step=step)
        except Exception as e:
            logger.warning(f"Error logging metric {name}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        if self.tracker is None:
            return
        
        try:
            if self.backend == "wandb":
                self.tracker.config.update(params)
            elif self.backend == "tensorboard":
                # TensorBoard doesn't have a direct params API
                for key, value in params.items():
                    self.tracker.add_text(f"params/{key}", str(value))
            elif self.backend == "mlflow":
                self.tracker.log_params(params)
        except Exception as e:
            logger.warning(f"Error logging params: {e}")
    
    def log_model(self, model_path: str, model_name: str = "model") -> None:
        """Log a model artifact"""
        if self.tracker is None:
            return
        
        try:
            if self.backend == "wandb":
                self.tracker.save(model_path)
            elif self.backend == "tensorboard":
                # TensorBoard doesn't have model logging
                pass
            elif self.backend == "mlflow":
                self.tracker.log_artifact(model_path, artifact_path=model_name)
        except Exception as e:
            logger.warning(f"Error logging model: {e}")
    
    def finish(self) -> None:
        """Finish the experiment"""
        if self.tracker is None:
            return
        
        try:
            if self.backend == "wandb":
                self.tracker.finish()
            elif self.backend == "tensorboard":
                self.tracker.close()
            elif self.backend == "mlflow":
                self.tracker.end_run()
        except Exception as e:
            logger.warning(f"Error finishing experiment: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def load_model_config(config_path: str = "model_config.yaml") -> Dict[str, Any]:
    """
    Load model configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}", exc_info=True)
        return {}















