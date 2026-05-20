"""
Experiment Tracker Module
=========================

Track training experiments with metadata and metrics.

Author: BUL System
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Track training experiments with metadata, hyperparameters, and results.
    
    Provides:
    - Experiment logging
    - Hyperparameter tracking
    - Metric tracking
    - Experiment comparison
    - Export/import functionality
    
    Example:
        >>> tracker = ExperimentTracker("experiments/")
        >>> tracker.start_experiment("gpt2_fine_tune_v1")
        >>> tracker.log_params({"lr": 3e-5, "batch_size": 8})
        >>> tracker.log_metrics({"loss": 0.5, "perplexity": 2.7})
        >>> tracker.end_experiment()
    """
    
    def __init__(self, experiments_dir: Path):
        """
        Initialize ExperimentTracker.
        
        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.current_experiment_path: Optional[Path] = None
    
    def start_experiment(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Optional description
            tags: Optional tags for categorization
            **kwargs: Additional metadata
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "description": description,
            "tags": tags or [],
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "hyperparameters": {},
            "metrics": {},
            "metadata": kwargs,
        }
        
        self.current_experiment_path = experiment_dir
        
        # Save initial experiment data
        self._save_experiment()
        
        logger.info(f"Started experiment: {experiment_id}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["hyperparameters"].update(params)
        self._save_experiment()
        logger.debug(f"Logged parameters: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        if step is not None:
            if "metrics_history" not in self.current_experiment:
                self.current_experiment["metrics_history"] = []
            self.current_experiment["metrics_history"].append({
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            })
        
        # Update latest metrics
        self.current_experiment["metrics"].update(metrics)
        self._save_experiment()
        logger.debug(f"Logged metrics at step {step}: {metrics}")
    
    def log_artifact(self, file_path: Path, artifact_name: Optional[str] = None) -> None:
        """
        Log an artifact (file) with the experiment.
        
        Args:
            file_path: Path to the file to log
            artifact_name: Optional name for the artifact
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        if not self.current_experiment_path:
            raise ValueError("No experiment path set")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        
        artifact_name = artifact_name or file_path.name
        artifacts_dir = self.current_experiment_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Copy file
        import shutil
        dest_path = artifacts_dir / artifact_name
        shutil.copy2(file_path, dest_path)
        
        # Log artifact reference
        if "artifacts" not in self.current_experiment:
            self.current_experiment["artifacts"] = []
        self.current_experiment["artifacts"].append({
            "name": artifact_name,
            "path": str(dest_path),
            "timestamp": datetime.now().isoformat(),
        })
        
        self._save_experiment()
        logger.info(f"Logged artifact: {artifact_name}")
    
    def end_experiment(self, status: str = "completed") -> None:
        """
        End the current experiment.
        
        Args:
            status: Final status (completed, failed, cancelled)
        """
        if not self.current_experiment:
            raise ValueError("No active experiment to end.")
        
        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = status
        
        # Calculate duration
        start = datetime.fromisoformat(self.current_experiment["start_time"])
        end = datetime.now()
        duration = (end - start).total_seconds()
        self.current_experiment["duration_seconds"] = duration
        
        self._save_experiment()
        logger.info(f"Ended experiment: {self.current_experiment['experiment_id']} (status: {status})")
        
        self.current_experiment = None
        self.current_experiment_path = None
    
    def _save_experiment(self) -> None:
        """Save current experiment data to disk."""
        if not self.current_experiment or not self.current_experiment_path:
            return
        
        experiment_file = self.current_experiment_path / "experiment.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            experiment_file = exp_dir / "experiment.json"
            if experiment_file.exists():
                try:
                    with open(experiment_file, 'r') as f:
                        exp_data = json.load(f)
                        experiments.append({
                            "experiment_id": exp_data.get("experiment_id"),
                            "name": exp_data.get("experiment_name"),
                            "status": exp_data.get("status"),
                            "start_time": exp_data.get("start_time"),
                            "metrics": exp_data.get("metrics", {}),
                        })
                except Exception as e:
                    logger.warning(f"Could not load experiment from {exp_dir}: {e}")
        
        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full experiment data.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment data or None if not found
        """
        experiment_file = self.experiments_dir / experiment_id / "experiment.json"
        if not experiment_file.exists():
            return None
        
        with open(experiment_file, 'r') as f:
            return json.load(f)
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison summary
        """
        experiments = []
        for exp_id in experiment_ids:
            exp_data = self.get_experiment(exp_id)
            if exp_data:
                experiments.append(exp_data)
        
        if not experiments:
            return {"error": "No valid experiments found"}
        
        comparison = {
            "experiments": [exp["experiment_id"] for exp in experiments],
            "metrics_comparison": {},
            "hyperparameters_comparison": {},
        }
        
        # Compare metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.get("metrics", {}).keys())
        
        for metric in all_metrics:
            values = [exp.get("metrics", {}).get(metric) for exp in experiments]
            comparison["metrics_comparison"][metric] = {
                "values": values,
                "min": min(v for v in values if v is not None),
                "max": max(v for v in values if v is not None),
                "avg": sum(v for v in values if v is not None) / len([v for v in values if v is not None]),
            }
        
        # Compare hyperparameters
        all_params = set()
        for exp in experiments:
            all_params.update(exp.get("hyperparameters", {}).keys())
        
        for param in all_params:
            values = [exp.get("hyperparameters", {}).get(param) for exp in experiments]
            comparison["hyperparameters_comparison"][param] = {
                "values": values,
                "unique": list(set(v for v in values if v is not None)),
            }
        
        return comparison

