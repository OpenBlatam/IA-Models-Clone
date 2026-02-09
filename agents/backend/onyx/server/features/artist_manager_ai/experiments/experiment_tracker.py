"""
Experiment Tracker
==================

Experiment tracking following best practices.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Experiment tracker for ML experiments.
    
    Features:
    - Hyperparameter logging
    - Metric tracking
    - Model checkpointing
    - Configuration management
    - Experiment comparison
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory for experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment: Optional[str] = None
        self.experiment_path: Optional[Path] = None
        self.metrics: Dict[str, list] = {}
        self.hyperparameters: Dict[str, Any] = {}
        self._logger = logger
    
    def start_experiment(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start new experiment.
        
        Args:
            experiment_name: Name of experiment
            config: Configuration dictionary
        
        Returns:
            Experiment ID
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        self.experiment_path = self.experiment_dir / experiment_id
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = experiment_id
        self.metrics = {}
        self.hyperparameters = config or {}
        
        # Save configuration
        if config:
            config_path = self.experiment_path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        self._logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hyperparameters: Hyperparameter dictionary
        """
        self.hyperparameters.update(hyperparameters)
        
        # Save to file
        if self.experiment_path:
            hp_path = self.experiment_path / "hyperparameters.yaml"
            with open(hp_path, "w") as f:
                yaml.dump(hyperparameters, f, default_flow_style=False)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step/epoch number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "step": step if step is not None else len(self.metrics[name]),
            "timestamp": datetime.now().isoformat()
        })
        
        # Save metrics periodically
        if self.experiment_path and len(self.metrics[name]) % 10 == 0:
            self._save_metrics()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def _save_metrics(self):
        """Save metrics to file."""
        if self.experiment_path:
            metrics_path = self.experiment_path / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)
    
    def save_model_info(self, model_info: Dict[str, Any]):
        """
        Save model information.
        
        Args:
            model_info: Model information dictionary
        """
        if self.experiment_path:
            model_path = self.experiment_path / "model_info.yaml"
            with open(model_path, "w") as f:
                yaml.dump(model_info, f, default_flow_style=False)
    
    def finish_experiment(self, summary: Optional[Dict[str, Any]] = None):
        """
        Finish experiment.
        
        Args:
            summary: Final summary dictionary
        """
        # Save final metrics
        self._save_metrics()
        
        # Save summary
        if summary and self.experiment_path:
            summary_path = self.experiment_path / "summary.yaml"
            with open(summary_path, "w") as f:
                yaml.dump(summary, f, default_flow_style=False)
        
        self._logger.info(f"Finished experiment: {self.current_experiment}")
        self.current_experiment = None
    
    def get_experiment_path(self) -> Optional[Path]:
        """Get current experiment path."""
        return self.experiment_path




