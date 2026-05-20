"""
Experiment Tracker
==================

Sistema de tracking de experimentos.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Experimento."""
    id: str
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, List[float]]
    status: str = "running"
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class ExperimentTracker:
    """Tracker de experimentos."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Inicializar tracker.
        
        Args:
            experiments_dir: Directorio para experimentos
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self._logger = logger
    
    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> Experiment:
        """
        Crear experimento.
        
        Args:
            name: Nombre del experimento
            config: Configuración
        
        Returns:
            Experimento creado
        """
        import uuid
        
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            config=config,
            metrics={}
        )
        
        self.experiments[experiment.id] = experiment
        
        # Guardar en disco
        self._save_experiment(experiment)
        
        self._logger.info(f"Created experiment: {name} (id: {experiment.id})")
        return experiment
    
    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ):
        """
        Registrar métrica.
        
        Args:
            experiment_id: ID del experimento
            metric_name: Nombre de la métrica
            value: Valor
            step: Paso (opcional)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if metric_name not in experiment.metrics:
            experiment.metrics[metric_name] = []
        
        experiment.metrics[metric_name].append(value)
        
        # Guardar actualización
        self._save_experiment(experiment)
    
    def complete_experiment(self, experiment_id: str):
        """
        Completar experimento.
        
        Args:
            experiment_id: ID del experimento
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment.status = "completed"
        experiment.completed_at = datetime.now()
        
        self._save_experiment(experiment)
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Obtener experimento."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Experiment]:
        """Listar experimentos."""
        return list(self.experiments.values())
    
    def _save_experiment(self, experiment: Experiment):
        """Guardar experimento en disco."""
        filepath = self.experiments_dir / f"{experiment.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)




