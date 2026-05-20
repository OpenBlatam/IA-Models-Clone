"""
MLOps Manager - Gestor de MLOps
================================

Sistema de gestión de operaciones de ML con versionado de modelos, experimentos y monitoreo.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Estado de experimento."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(Enum):
    """Estado del modelo."""
    TRAINING = "training"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    RETIRED = "retired"


@dataclass
class MLExperiment:
    """Experimento de ML."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    dataset_version: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLModel:
    """Modelo de ML."""
    model_id: str
    name: str
    version: str
    model_type: str
    status: ModelStatus
    experiment_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelDrift:
    """Drift de modelo."""
    drift_id: str
    model_id: str
    metric_name: str
    baseline_value: float
    current_value: float
    drift_score: float
    detected_at: datetime = field(default_factory=datetime.now)
    severity: str = "medium"  # low, medium, high, critical


class MLOpsManager:
    """Gestor de MLOps."""
    
    def __init__(self):
        self.experiments: Dict[str, MLExperiment] = {}
        self.models: Dict[str, MLModel] = {}
        self.model_versions: Dict[str, List[str]] = defaultdict(list)
        self.drift_detections: List[ModelDrift] = []
        self._lock = asyncio.Lock()
    
    async def create_experiment(
        self,
        experiment_id: str,
        name: str,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_version: Optional[str] = None,
    ) -> str:
        """Crear experimento de ML."""
        experiment = MLExperiment(
            experiment_id=experiment_id,
            name=name,
            status=ExperimentStatus.PENDING,
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            dataset_version=dataset_version or "",
        )
        
        async with self._lock:
            self.experiments[experiment_id] = experiment
        
        logger.info(f"Created experiment: {experiment_id} - {name}")
        return experiment_id
    
    async def update_experiment(
        self,
        experiment_id: str,
        status: Optional[ExperimentStatus] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Actualizar experimento."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        async with self._lock:
            if status:
                experiment.status = status
                if status == ExperimentStatus.COMPLETED:
                    experiment.completed_at = datetime.now()
            
            if metrics:
                experiment.metrics.update(metrics)
    
    async def register_model(
        self,
        model_id: str,
        name: str,
        version: str,
        model_type: str,
        experiment_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Registrar modelo."""
        model = MLModel(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            status=ModelStatus.VALIDATED,
            experiment_id=experiment_id,
            metrics=metrics or {},
        )
        
        async with self._lock:
            self.models[model_id] = model
            model_key = f"{name}_{version}"
            self.model_versions[model_key].append(model_id)
        
        logger.info(f"Registered model: {model_id} - {name} v{version}")
        return model_id
    
    async def deploy_model(self, model_id: str) -> bool:
        """Desplegar modelo."""
        model = self.models.get(model_id)
        if not model:
            return False
        
        async with self._lock:
            model.status = ModelStatus.DEPLOYED
            model.deployed_at = datetime.now()
        
        logger.info(f"Deployed model: {model_id}")
        return True
    
    async def record_model_performance(
        self,
        model_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ):
        """Registrar rendimiento del modelo."""
        model = self.models.get(model_id)
        if not model:
            return
        
        async with self._lock:
            model.performance_history.append({
                "metrics": metrics,
                "timestamp": timestamp or datetime.now(),
            })
    
    async def detect_drift(
        self,
        model_id: str,
        metric_name: str,
        current_value: float,
        baseline_value: Optional[float] = None,
    ) -> Optional[ModelDrift]:
        """Detectar drift en modelo."""
        model = self.models.get(model_id)
        if not model:
            return None
        
        if baseline_value is None:
            # Usar métrica inicial del modelo
            baseline_value = model.metrics.get(metric_name, 0.0)
        
        # Calcular drift score (simplificado)
        if baseline_value == 0:
            drift_score = abs(current_value)
        else:
            drift_score = abs((current_value - baseline_value) / baseline_value)
        
        severity = "low"
        if drift_score >= 0.5:
            severity = "critical"
        elif drift_score >= 0.3:
            severity = "high"
        elif drift_score >= 0.15:
            severity = "medium"
        
        drift = ModelDrift(
            drift_id=f"drift_{model_id}_{datetime.now().timestamp()}",
            model_id=model_id,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            drift_score=drift_score,
            severity=severity,
        )
        
        async with self._lock:
            self.drift_detections.append(drift)
        
        if severity in ["high", "critical"]:
            logger.warning(
                f"Drift detected in model {model_id}: {metric_name} = {drift_score:.2f} ({severity})"
            )
        
        return drift
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Obtener experimento."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None
        
        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "status": exp.status.value,
            "model_type": exp.model_type,
            "hyperparameters": exp.hyperparameters,
            "metrics": exp.metrics,
            "dataset_version": exp.dataset_version,
            "created_at": exp.created_at.isoformat(),
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        }
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtener modelo."""
        model = self.models.get(model_id)
        if not model:
            return None
        
        return {
            "model_id": model.model_id,
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
            "status": model.status.value,
            "experiment_id": model.experiment_id,
            "metrics": model.metrics,
            "created_at": model.created_at.isoformat(),
            "deployed_at": model.deployed_at.isoformat() if model.deployed_at else None,
            "performance_history_count": len(model.performance_history),
        }
    
    def get_drift_detections(
        self,
        model_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener detecciones de drift."""
        drifts = self.drift_detections
        
        if model_id:
            drifts = [d for d in drifts if d.model_id == model_id]
        
        if severity:
            drifts = [d for d in drifts if d.severity == severity]
        
        return [
            {
                "drift_id": d.drift_id,
                "model_id": d.model_id,
                "metric_name": d.metric_name,
                "baseline_value": d.baseline_value,
                "current_value": d.current_value,
                "drift_score": d.drift_score,
                "severity": d.severity,
                "detected_at": d.detected_at.isoformat(),
            }
            for d in drifts[-limit:]
        ]
    
    def get_mlops_summary(self) -> Dict[str, Any]:
        """Obtener resumen de MLOps."""
        experiments_by_status: Dict[str, int] = defaultdict(int)
        models_by_status: Dict[str, int] = defaultdict(int)
        drifts_by_severity: Dict[str, int] = defaultdict(int)
        
        for exp in self.experiments.values():
            experiments_by_status[exp.status.value] += 1
        
        for model in self.models.values():
            models_by_status[model.status.value] += 1
        
        for drift in self.drift_detections:
            drifts_by_severity[drift.severity] += 1
        
        return {
            "total_experiments": len(self.experiments),
            "experiments_by_status": dict(experiments_by_status),
            "total_models": len(self.models),
            "models_by_status": dict(models_by_status),
            "total_drift_detections": len(self.drift_detections),
            "drifts_by_severity": dict(drifts_by_severity),
        }
















