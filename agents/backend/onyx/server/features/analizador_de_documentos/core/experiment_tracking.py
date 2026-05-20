"""
Sistema de Experiment Tracking
================================

Sistema para seguimiento de experimentos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Estado del experimento"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Experiment:
    """Experimento"""
    experiment_id: str
    name: str
    description: str
    tags: List[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: ExperimentStatus
    created_at: str
    updated_at: str


@dataclass
class ExperimentRun:
    """Ejecución de experimento"""
    run_id: str
    experiment_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    status: ExperimentStatus
    started_at: str
    completed_at: Optional[str]


class ExperimentTracking:
    """
    Sistema de Experiment Tracking
    
    Proporciona:
    - Seguimiento de experimentos
    - Versionado de código y datos
    - Logging de métricas
    - Comparación de experimentos
    - Búsqueda de experimentos
    - Visualización de resultados
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.experiments: Dict[str, Experiment] = {}
        self.runs: Dict[str, ExperimentRun] = {}
        logger.info("ExperimentTracking inicializado")
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """
        Crear experimento
        
        Args:
            name: Nombre del experimento
            description: Descripción
            tags: Tags
        
        Returns:
            Experimento creado
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tags or [],
            parameters={},
            metrics={},
            status=ExperimentStatus.CREATED,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.experiments[experiment_id] = experiment
        
        logger.info(f"Experimento creado: {experiment_id}")
        
        return experiment
    
    def log_run(
        self,
        experiment_id: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[List[str]] = None
    ) -> ExperimentRun:
        """
        Registrar ejecución
        
        Args:
            experiment_id: ID del experimento
            parameters: Parámetros
            metrics: Métricas
            artifacts: Artifacts
        
        Returns:
            Ejecución registrada
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experimento no encontrado: {experiment_id}")
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            parameters=parameters,
            metrics=metrics,
            artifacts=artifacts or [],
            status=ExperimentStatus.COMPLETED,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat()
        )
        
        self.runs[run_id] = run
        
        # Actualizar experimento
        experiment = self.experiments[experiment_id]
        experiment.parameters.update(parameters)
        experiment.metrics.update(metrics)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.updated_at = datetime.now().isoformat()
        
        logger.info(f"Ejecución registrada: {run_id}")
        
        return run
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Comparar experimentos
        
        Args:
            experiment_ids: IDs de experimentos
            metric: Métrica para comparar
        
        Returns:
            Comparación
        """
        comparison = {
            "metric": metric,
            "experiments": []
        }
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                comparison["experiments"].append({
                    "experiment_id": exp_id,
                    "name": exp.name,
                    metric: exp.metrics.get(metric, 0.0)
                })
        
        # Ordenar por métrica
        comparison["experiments"].sort(key=lambda x: x[metric], reverse=True)
        comparison["best"] = comparison["experiments"][0] if comparison["experiments"] else None
        
        logger.info(f"Comparación completada: {len(experiment_ids)} experimentos")
        
        return comparison


# Instancia global
_experiment_tracking: Optional[ExperimentTracking] = None


def get_experiment_tracking() -> ExperimentTracking:
    """Obtener instancia global del sistema"""
    global _experiment_tracking
    if _experiment_tracking is None:
        _experiment_tracking = ExperimentTracking()
    return _experiment_tracking


