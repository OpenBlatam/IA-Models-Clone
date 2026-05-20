"""
Advanced Experiment Management
Manage experiments with versioning, comparison, and analysis
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class Experiment:
    """Experiment representation"""
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        config: Dict[str, Any],
        status: str = "running"  # "running", "completed", "failed", "cancelled"
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.config = config
        self.status = status
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.metrics: Dict[str, List[float]] = {}
        self.checkpoints: List[str] = []
        self.notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metrics": self.metrics,
            "checkpoints": self.checkpoints,
            "notes": self.notes
        }
    
    def add_metric(self, name: str, value: float):
        """Add metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        self.updated_at = datetime.now().isoformat()
    
    def get_best_metric(self, metric_name: str, maximize: bool = True) -> Optional[float]:
        """Get best metric value"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        values = self.metrics[metric_name]
        return max(values) if maximize else min(values)


class ExperimentManager:
    """
    Advanced experiment management system
    """
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
    
    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None
    ) -> Experiment:
        """Create a new experiment"""
        if experiment_id is None:
            # Generate ID from name and config
            config_str = json.dumps(config, sort_keys=True)
            experiment_id = hashlib.md5(f"{name}_{config_str}".encode()).hexdigest()[:8]
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            config=config
        )
        
        self.experiments[experiment_id] = experiment
        self.active_experiments.append(experiment_id)
        
        # Save experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created experiment: {name} ({experiment_id})")
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)
    
    def update_experiment(
        self,
        experiment_id: str,
        metrics: Optional[Dict[str, float]] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """Update experiment"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if metrics:
            for name, value in metrics.items():
                experiment.add_metric(name, value)
        
        if status:
            experiment.status = status
        
        if notes:
            experiment.notes = notes
        
        experiment.updated_at = datetime.now().isoformat()
        self._save_experiment(experiment)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_name: str = "val_loss"
    ) -> Dict[str, Any]:
        """Compare multiple experiments"""
        results = {}
        
        for exp_id in experiment_ids:
            experiment = self.get_experiment(exp_id)
            if experiment:
                best_metric = experiment.get_best_metric(metric_name, maximize=False)
                results[exp_id] = {
                    "name": experiment.name,
                    "best_metric": best_metric,
                    "status": experiment.status,
                    "config": experiment.config
                }
        
        # Find best
        valid_results = {k: v for k, v in results.items() if v["best_metric"] is not None}
        if valid_results:
            best_id = min(valid_results, key=lambda x: valid_results[x]["best_metric"])
            results["best"] = {
                "experiment_id": best_id,
                "metric_value": valid_results[best_id]["best_metric"]
            }
        
        return results
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List experiments"""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        # Sort by updated_at
        experiments.sort(key=lambda x: x.updated_at, reverse=True)
        
        if limit:
            experiments = experiments[:limit]
        
        return [e.to_dict() for e in experiments]
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk"""
        exp_file = self.experiments_dir / f"{experiment.experiment_id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
    
    def load_experiments(self):
        """Load experiments from disk"""
        for exp_file in self.experiments_dir.glob("*.json"):
            try:
                with open(exp_file, "r") as f:
                    data = json.load(f)
                    experiment = Experiment(
                        experiment_id=data["experiment_id"],
                        name=data["name"],
                        config=data["config"],
                        status=data.get("status", "completed")
                    )
                    experiment.metrics = data.get("metrics", {})
                    experiment.checkpoints = data.get("checkpoints", [])
                    experiment.notes = data.get("notes", "")
                    experiment.created_at = data.get("created_at", datetime.now().isoformat())
                    experiment.updated_at = data.get("updated_at", datetime.now().isoformat())
                    
                    self.experiments[experiment.experiment_id] = experiment
            except Exception as e:
                logger.error(f"Error loading experiment {exp_file}: {str(e)}")

