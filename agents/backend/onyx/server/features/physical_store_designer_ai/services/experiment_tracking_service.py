"""
Experiment Tracking Service - Sistema de experimentación y tracking
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Estados de experimento"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentTrackingService:
    """Servicio para tracking de experimentos"""
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.runs: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_experiment(
        self,
        experiment_name: str,
        description: str,
        hyperparameters: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Crear experimento"""
        
        experiment_id = f"exp_{experiment_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        experiment = {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "description": description,
            "hyperparameters": hyperparameters,
            "tags": tags or [],
            "status": ExperimentStatus.DRAFT.value,
            "created_at": datetime.now().isoformat(),
            "runs": []
        }
        
        self.experiments[experiment_id] = experiment
        
        return experiment
    
    def create_run(
        self,
        experiment_id: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Crear run de experimento"""
        
        experiment = self.experiments.get(experiment_id)
        
        if not experiment:
            raise ValueError(f"Experimento {experiment_id} no encontrado")
        
        run_id = f"run_{experiment_id}_{len(experiment['runs']) + 1}"
        
        run = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "name": run_name,
            "config": config or {},
            "status": ExperimentStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "metrics": {},
            "artifacts": []
        }
        
        experiment["runs"].append(run_id)
        
        if experiment_id not in self.runs:
            self.runs[experiment_id] = []
        
        self.runs[experiment_id].append(run)
        
        return run
    
    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Registrar métrica"""
        
        metric = {
            "metric_id": f"metric_{run_id}_{len(self.metrics.get(run_id, [])) + 1}",
            "run_id": run_id,
            "name": metric_name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        
        if run_id not in self.metrics:
            self.metrics[run_id] = []
        
        self.metrics[run_id].append(metric)
        
        # Actualizar en run
        for experiment_runs in self.runs.values():
            for run in experiment_runs:
                if run["run_id"] == run_id:
                    if "metrics" not in run:
                        run["metrics"] = {}
                    run["metrics"][metric_name] = value
                    break
        
        return metric
    
    def log_parameters(
        self,
        run_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Registrar parámetros"""
        
        for experiment_runs in self.runs.values():
            for run in experiment_runs:
                if run["run_id"] == run_id:
                    if "parameters" not in run:
                        run["parameters"] = {}
                    run["parameters"].update(parameters)
                    return {"run_id": run_id, "parameters": run["parameters"]}
        
        raise ValueError(f"Run {run_id} no encontrado")
    
    def complete_run(
        self,
        run_id: str,
        final_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Completar run"""
        
        for experiment_runs in self.runs.values():
            for run in experiment_runs:
                if run["run_id"] == run_id:
                    run["status"] = ExperimentStatus.COMPLETED.value
                    run["completed_at"] = datetime.now().isoformat()
                    
                    if final_metrics:
                        run["final_metrics"] = final_metrics
                    
                    return True
        
        return False
    
    def get_experiment_results(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """Obtener resultados del experimento"""
        
        experiment = self.experiments.get(experiment_id)
        
        if not experiment:
            raise ValueError(f"Experimento {experiment_id} no encontrado")
        
        runs = self.runs.get(experiment_id, [])
        
        return {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "total_runs": len(runs),
            "completed_runs": len([r for r in runs if r["status"] == ExperimentStatus.COMPLETED.value]),
            "runs": runs,
            "best_run": self._find_best_run(runs),
            "summary": self._generate_summary(runs)
        }
    
    def _find_best_run(self, runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Encontrar mejor run"""
        completed_runs = [r for r in runs if r["status"] == ExperimentStatus.COMPLETED.value]
        
        if not completed_runs:
            return None
        
        # Encontrar run con mejor métrica (asumiendo "accuracy" o "loss")
        best_run = None
        best_score = -float('inf')
        
        for run in completed_runs:
            metrics = run.get("final_metrics", run.get("metrics", {}))
            score = metrics.get("accuracy", metrics.get("val_accuracy", -metrics.get("loss", -float('inf'))))
            
            if score > best_score:
                best_score = score
                best_run = run
        
        return best_run
    
    def _generate_summary(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar resumen de runs"""
        completed = [r for r in runs if r["status"] == ExperimentStatus.COMPLETED.value]
        
        if not completed:
            return {"message": "No hay runs completados"}
        
        all_metrics = {}
        for run in completed:
            metrics = run.get("final_metrics", run.get("metrics", {}))
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        summary = {}
        for key, values in all_metrics.items():
            summary[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5
            }
        
        return summary




