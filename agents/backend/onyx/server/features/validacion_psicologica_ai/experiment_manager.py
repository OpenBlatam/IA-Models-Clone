"""
Experiment Manager
==================
Comprehensive experiment management system
"""

from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path
import json
import structlog
import yaml

logger = structlog.get_logger()


class Experiment:
    """Experiment container"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment
        
        Args:
            name: Experiment name
            description: Description
            config: Configuration dictionary
        """
        self.id = uuid4()
        self.name = name
        self.description = description
        self.config = config or {}
        self.status = "created"
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.metrics: Dict[str, List[float]] = {}
        self.artifacts: List[str] = []
        self.tags: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags
        }


class ExperimentManager:
    """
    Manager for experiments
    """
    
    def __init__(self, experiments_dir: str = "./experiments"):
        """
        Initialize experiment manager
        
        Args:
            experiments_dir: Directory for experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[UUID, Experiment] = {}
        
        logger.info("ExperimentManager initialized", experiments_dir=str(self.experiments_dir))
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """
        Create new experiment
        
        Args:
            name: Experiment name
            description: Description
            config: Configuration
            tags: Tags
            
        Returns:
            Created experiment
        """
        experiment = Experiment(name, description, config)
        if tags:
            experiment.tags = tags
        
        self._experiments[experiment.id] = experiment
        
        # Save experiment
        self._save_experiment(experiment)
        
        logger.info("Experiment created", experiment_id=str(experiment.id), name=name)
        
        return experiment
    
    def get_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        """
        Get experiment by ID
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment or None
        """
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> List[Experiment]:
        """
        List experiments
        
        Args:
            tags: Filter by tags
            status: Filter by status
            
        Returns:
            List of experiments
        """
        experiments = list(self._experiments.values())
        
        if tags:
            experiments = [e for e in experiments if any(tag in e.tags for tag in tags)]
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return sorted(experiments, key=lambda x: x.created_at, reverse=True)
    
    def update_experiment_metrics(
        self,
        experiment_id: UUID,
        metrics: Dict[str, float]
    ) -> None:
        """
        Update experiment metrics
        
        Args:
            experiment_id: Experiment ID
            metrics: Metrics dictionary
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        for key, value in metrics.items():
            if key not in experiment.metrics:
                experiment.metrics[key] = []
            experiment.metrics[key].append(value)
        
        self._save_experiment(experiment)
    
    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk"""
        experiment_dir = self.experiments_dir / str(experiment.id)
        experiment_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)
        
        # Save config
        if experiment.config:
            config_path = experiment_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(experiment.config, f)


# Global experiment manager
experiment_manager = ExperimentManager()




