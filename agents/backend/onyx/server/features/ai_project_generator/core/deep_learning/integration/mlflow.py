"""
MLflow Integration
==================

Integration with MLflow for experiment tracking and model registry.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Try to import mlflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class MLflowIntegration:
    """
    Integration with MLflow.
    
    Provides functionality to:
    - Track experiments
    - Log metrics
    - Register models
    - Model versioning
    """
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize MLflow integration.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start MLflow run.
        
        Args:
            run_name: Name for the run
        """
        mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run_name}")
    
    def end_run(self) -> None:
        """End current MLflow run."""
        mlflow.end_run()
        logger.info("MLflow run ended")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Artifact path
            registered_model_name: Name for registered model
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Model logged to MLflow: {artifact_path}")
    
    def save_model(
        self,
        model: torch.nn.Module,
        path: Path,
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Save model to MLflow.
        
        Args:
            model: PyTorch model
            path: Path to save
            registered_model_name: Name for registered model
        """
        mlflow.pytorch.save_model(
            model,
            path=str(path),
            registered_model_name=registered_model_name
        )
        logger.info(f"Model saved to MLflow: {path}")



