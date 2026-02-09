"""
Experiment Manager
Manage and track experiments
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Experiment run configuration"""
    experiment_name: str
    run_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list] = None
    config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set defaults"""
        if self.run_name is None:
            self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.tags is None:
            self.tags = []
        if self.config is None:
            self.config = {}


class ExperimentManager:
    """
    Manage experiments and runs
    """
    
    def __init__(self, base_dir: Path = Path("experiments")):
        """
        Initialize experiment manager
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_run = None
    
    def create_run(self, config: RunConfig) -> Path:
        """
        Create new experiment run
        
        Args:
            config: Run configuration
            
        Returns:
            Path to run directory
        """
        # Create experiment directory
        exp_dir = self.base_dir / config.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory
        run_dir = exp_dir / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Create subdirectories
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        
        self.current_run = run_dir
        logger.info(f"Created experiment run: {run_dir}")
        
        return run_dir
    
    def get_run_path(self, experiment_name: str, run_name: str) -> Path:
        """
        Get path to specific run
        
        Args:
            experiment_name: Experiment name
            run_name: Run name
            
        Returns:
            Path to run directory
        """
        return self.base_dir / experiment_name / run_name
    
    def list_runs(self, experiment_name: Optional[str] = None) -> list:
        """
        List all runs
        
        Args:
            experiment_name: Filter by experiment name (optional)
            
        Returns:
            List of run paths
        """
        if experiment_name:
            exp_dir = self.base_dir / experiment_name
            if exp_dir.exists():
                return [exp_dir / d for d in exp_dir.iterdir() if d.is_dir()]
        else:
            runs = []
            for exp_dir in self.base_dir.iterdir():
                if exp_dir.is_dir():
                    runs.extend([exp_dir / d for d in exp_dir.iterdir() if d.is_dir()])
            return runs
    
    def get_best_run(self, experiment_name: str, metric: str = "val_accuracy") -> Optional[Path]:
        """
        Get best run based on metric
        
        Args:
            experiment_name: Experiment name
            metric: Metric to compare
            
        Returns:
            Path to best run or None
        """
        runs = self.list_runs(experiment_name)
        if not runs:
            return None
        
        best_run = None
        best_value = float('-inf')
        
        for run in runs:
            metrics_file = run / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if metric in metrics:
                        value = metrics[metric]
                        if value > best_value:
                            best_value = value
                            best_run = run
        
        return best_run


class ExperimentLogger:
    """
    Logger for experiment metrics and artifacts
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize experiment logger
        
        Args:
            run_dir: Run directory
        """
        self.run_dir = Path(run_dir)
        self.metrics = {}
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log metric
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (optional)
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'step': step,
        })
        
        # Save to file
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_artifact(self, file_path: Path, name: Optional[str] = None) -> None:
        """
        Log artifact
        
        Args:
            file_path: Path to artifact file
            name: Artifact name (optional)
        """
        artifacts_dir = self.run_dir / "artifacts"
        if name:
            dest_path = artifacts_dir / name
        else:
            dest_path = artifacts_dir / file_path.name
        
        import shutil
        shutil.copy2(file_path, dest_path)
        logger.info(f"Logged artifact: {dest_path}")



