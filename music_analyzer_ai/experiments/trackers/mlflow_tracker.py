"""
MLflow Tracker
Integration with MLflow for experiment tracking
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available")

from .base_tracker import BaseExperimentTracker


class MLflowTracker(BaseExperimentTracker):
    """MLflow experiment tracker"""
    
    def __init__(
        self,
        experiment_name: str,
        project_name: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        super().__init__(experiment_name, project_name)
        self.tracking_uri = tracking_uri
        self.run = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize MLflow run"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, tracker disabled")
            return False
        
        try:
            # Set tracking URI if provided
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment
            if self.project_name:
                mlflow.set_experiment(self.project_name)
            
            # Start run
            self.run = mlflow.start_run(run_name=self.experiment_name)
            self.initialized = True
            
            # Log config
            if config:
                self.log_params(config)
            
            logger.info(f"MLflow tracker initialized: {self.experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing MLflow: {str(e)}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow"""
        if not self.initialized or self.run is None:
            return
        
        try:
            if step is not None:
                mlflow.log_metrics(metrics, step=step)
            else:
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to MLflow"""
        if not self.initialized or self.run is None:
            return
        
        try:
            # Convert all values to strings for MLflow
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
        except Exception as e:
            logger.error(f"Error logging params to MLflow: {str(e)}")
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifact to MLflow"""
        if not self.initialized or self.run is None:
            return
        
        try:
            mlflow.log_artifact(model_path, artifact_path="models")
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {str(e)}")
    
    def finish(self):
        """End MLflow run"""
        if self.run is not None:
            mlflow.end_run()



