"""
MLflow Integration
MLflow integration for experiment tracking
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class MLflowIntegration:
    """
    MLflow integration for experiment tracking
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow integration
        
        Args:
            experiment_name: Experiment name
            tracking_uri: Tracking URI (optional)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"Initialized MLflow: experiment={experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start MLflow run
        
        Args:
            run_name: Run name (optional)
        """
        mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        if step is not None:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path: str = "model") -> None:
        """
        Log model
        
        Args:
            model: Model to log
            artifact_path: Artifact path
        """
        mlflow.pytorch.log_model(model, artifact_path)
    
    def end_run(self) -> None:
        """End MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")



