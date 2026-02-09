"""
Experiments Module - Experiment tracking and versioning.

Provides:
- Experiment management
- Version control for models and benchmarks
- Experiment comparison
- Reproducibility tracking
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str = ""
    model_name: str = ""
    benchmark_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get hash for reproducibility."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class Experiment:
    """Experiment record."""
    id: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": self.results,
            "error": self.error,
            "duration": self.duration,
            "version": self.version,
        }


class ExperimentManager:
    """Manages experiments and versioning."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize experiment manager.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path) if storage_path else Path("experiments")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """Load experiments from storage."""
        experiments_file = self.storage_path / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file, 'r') as f:
                    data = json.load(f)
                    for exp_data in data:
                        exp = Experiment(
                            id=exp_data["id"],
                            config=ExperimentConfig(**exp_data["config"]),
                            status=ExperimentStatus(exp_data["status"]),
                            created_at=exp_data["created_at"],
                            started_at=exp_data.get("started_at"),
                            completed_at=exp_data.get("completed_at"),
                            results=exp_data.get("results"),
                            error=exp_data.get("error"),
                            duration=exp_data.get("duration", 0.0),
                            version=exp_data.get("version", "1.0.0"),
                        )
                        self.experiments[exp.id] = exp
            except Exception as e:
                logger.error(f"Error loading experiments: {e}")
    
    def _save_experiments(self) -> None:
        """Save experiments to storage."""
        experiments_file = self.storage_path / "experiments.json"
        data = [exp.to_dict() for exp in self.experiments.values()]
        with open(experiments_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Created experiment
        """
        exp_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_id = f"{exp_id}_{config.get_hash()[:8]}"
        
        experiment = Experiment(
            id=exp_id,
            config=config,
            status=ExperimentStatus.DRAFT,
        )
        
        self.experiments[exp_id] = experiment
        self._save_experiments()
        
        # Save experiment config
        exp_dir = self.storage_path / exp_id
        exp_dir.mkdir(exist_ok=True)
        with open(exp_dir / "config.yaml", 'w') as f:
            yaml.dump(config.to_dict(), f)
        
        logger.info(f"Created experiment: {exp_id}")
        return experiment
    
    def start_experiment(self, exp_id: str) -> Experiment:
        """
        Start an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Updated experiment
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        exp = self.experiments[exp_id]
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now().isoformat()
        self._save_experiments()
        
        return exp
    
    def complete_experiment(
        self,
        exp_id: str,
        results: Dict[str, Any],
    ) -> Experiment:
        """
        Complete an experiment.
        
        Args:
            exp_id: Experiment ID
            results: Experiment results
            
        Returns:
            Updated experiment
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        exp = self.experiments[exp_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now().isoformat()
        exp.results = results
        
        if exp.started_at:
            start = datetime.fromisoformat(exp.started_at)
            end = datetime.fromisoformat(exp.completed_at)
            exp.duration = (end - start).total_seconds()
        
        # Save results
        exp_dir = self.storage_path / exp_id
        with open(exp_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        self._save_experiments()
        
        logger.info(f"Completed experiment: {exp_id}")
        return exp
    
    def fail_experiment(self, exp_id: str, error: str) -> Experiment:
        """
        Mark experiment as failed.
        
        Args:
            exp_id: Experiment ID
            error: Error message
            
        Returns:
            Updated experiment
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        exp = self.experiments[exp_id]
        exp.status = ExperimentStatus.FAILED
        exp.completed_at = datetime.now().isoformat()
        exp.error = error
        
        if exp.started_at:
            start = datetime.fromisoformat(exp.started_at)
            end = datetime.fromisoformat(exp.completed_at)
            exp.duration = (end - start).total_seconds()
        
        self._save_experiments()
        
        logger.error(f"Failed experiment: {exp_id} - {error}")
        return exp
    
    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.experiments.get(exp_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Experiment]:
        """
        List experiments with filters.
        
        Args:
            status: Filter by status
            tags: Filter by tags
            
        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if tags:
            experiments = [
                e for e in experiments
                if any(tag in e.config.tags for tag in tags)
            ]
        
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)
    
    def compare_experiments(
        self,
        exp_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs
            
        Returns:
            Comparison results
        """
        experiments = [self.experiments[eid] for eid in exp_ids if eid in self.experiments]
        
        if not experiments:
            return {}
        
        comparison = {
            "experiments": [e.to_dict() for e in experiments],
            "metrics": {},
        }
        
        # Compare results if available
        completed = [e for e in experiments if e.results]
        if completed:
            comparison["metrics"] = {
                "best_accuracy": max(
                    (e.results.get("accuracy", 0.0) for e in completed),
                    default=0.0
                ),
                "best_throughput": max(
                    (e.results.get("throughput", 0.0) for e in completed),
                    default=0.0
                ),
                "avg_accuracy": sum(
                    e.results.get("accuracy", 0.0) for e in completed
                ) / len(completed) if completed else 0.0,
            }
        
        return comparison












