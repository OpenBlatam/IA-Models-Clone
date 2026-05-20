"""
Experiment Utilities
=====================

Utilities for experiment management and tracking.
"""

import torch
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Experiment logger for tracking experiments.
    
    Features:
    - Hyperparameter logging
    - Metrics tracking
    - Model checkpoints
    - Experiment comparison
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_dir: Experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[str] = None
        self._logger = logger
    
    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Create new experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
        
        Returns:
            Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        experiment_path = self.experiment_dir / experiment_id
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = experiment_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.current_experiment = experiment_id
        self._logger.info(f"Created experiment: {experiment_id}")
        
        return experiment_id
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int
    ):
        """
        Log metrics.
        
        Args:
            metrics: Metrics dictionary
            step: Step number
        """
        if self.current_experiment is None:
            self._logger.warning("No active experiment")
            return
        
        metrics_path = self.experiment_dir / self.current_experiment / "metrics.json"
        
        # Load existing metrics
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add new metrics
        all_metrics[str(step)] = metrics
        
        # Save
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            epoch: Epoch number
            metrics: Metrics (optional)
        """
        if self.current_experiment is None:
            self._logger.warning("No active experiment")
            return
        
        checkpoint_dir = self.experiment_dir / self.current_experiment / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {}
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self._logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            epoch: Epoch to load (latest if None)
        
        Returns:
            Checkpoint data
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        checkpoint_dir = self.experiment_dir / self.current_experiment / "checkpoints"
        
        if epoch is None:
            # Load latest
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self._logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return checkpoint
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "val_loss"
    ) -> Dict[str, Any]:
        """
        Compare experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            metric: Metric to compare
        
        Returns:
            Comparison results
        """
        comparisons = {}
        
        for exp_id in experiment_ids:
            metrics_path = self.experiment_dir / exp_id / "metrics.json"
            if not metrics_path.exists():
                continue
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract metric values
            values = [
                step_metrics.get(metric)
                for step_metrics in metrics.values()
                if metric in step_metrics
            ]
            
            if values:
                comparisons[exp_id] = {
                    "mean": float(sum(values) / len(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "final": float(values[-1])
                }
        
        return comparisons




