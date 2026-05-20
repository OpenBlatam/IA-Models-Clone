"""
Experiment Tracker - Tracking de experimentos y configuraciones.

Sigue principios de experiment tracking como wandb/tensorboard.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import hashlib

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuración de un experimento."""
    name: str
    description: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    hyperparameters: Dict[str, Any] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Obtener hash de la configuración."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class ExperimentResult:
    """Resultado de un experimento."""
    experiment_id: str
    config: ExperimentConfig
    response: str
    metrics: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    tokens_used: int
    cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ExperimentTracker:
    """
    Tracker de experimentos para LLMs.
    
    Sigue principios de experiment tracking (wandb/tensorboard).
    """
    
    def __init__(self, storage_path: str = "./storage/experiments"):
        """
        Inicializar tracker.
        
        Args:
            storage_path: Ruta para almacenar experimentos
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentResult] = {}
        self.current_experiment: Optional[str] = None
    
    def start_experiment(
        self,
        config: ExperimentConfig,
        experiment_id: Optional[str] = None
    ) -> str:
        """
        Iniciar un nuevo experimento.
        
        Args:
            config: Configuración del experimento
            experiment_id: ID del experimento (generado si no se proporciona)
            
        Returns:
            ID del experimento
        """
        if experiment_id is None:
            experiment_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.get_hash()}"
        
        self.current_experiment = experiment_id
        logger.info(f"Experimento '{experiment_id}' iniciado: {config.name}")
        
        return experiment_id
    
    def log_result(
        self,
        experiment_id: str,
        response: str,
        metrics: Dict[str, Any],
        duration_ms: float,
        tokens_used: int,
        cost: float = 0.0
    ) -> None:
        """
        Registrar resultado de un experimento.
        
        Args:
            experiment_id: ID del experimento
            response: Respuesta del modelo
            metrics: Métricas adicionales
            duration_ms: Duración en milisegundos
            tokens_used: Tokens usados
            cost: Costo del request
        """
        # Obtener configuración (debe estar guardada)
        # En una implementación completa, se guardaría al iniciar
        config = ExperimentConfig(
            name=experiment_id,
            model=metrics.get("model", ""),
            temperature=metrics.get("temperature", 0.7)
        )
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            response=response,
            metrics=metrics,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            cost=cost
        )
        
        self.experiments[experiment_id] = result
        self._save_experiment(result)
        
        logger.debug(f"Resultado registrado para experimento '{experiment_id}'")
    
    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Registrar una métrica.
        
        Args:
            experiment_id: ID del experimento
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            step: Paso/iteración (opcional)
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experimento '{experiment_id}' no encontrado")
            return
        
        result = self.experiments[experiment_id]
        metric_key = f"{metric_name}_{step}" if step is not None else metric_name
        result.metrics[metric_key] = value
        
        self._save_experiment(result)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "duration_ms"
    ) -> Dict[str, Any]:
        """
        Comparar múltiples experimentos.
        
        Args:
            experiment_ids: IDs de experimentos a comparar
            metric: Métrica a comparar
            
        Returns:
            Diccionario con comparación
        """
        results = []
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                result = self.experiments[exp_id]
                value = result.metrics.get(metric, getattr(result, metric, None))
                results.append({
                    "experiment_id": exp_id,
                    "config": result.config.name,
                    "value": value
                })
        
        if not results:
            return {"error": "No se encontraron experimentos"}
        
        values = [r["value"] for r in results if r["value"] is not None]
        
        return {
            "experiments": results,
            "metric": metric,
            "best": min(results, key=lambda x: x["value"] or float('inf')) if values else None,
            "worst": max(results, key=lambda x: x["value"] or 0) if values else None,
            "average": sum(values) / len(values) if values else None
        }
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Obtener resultado de un experimento."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[str] = None
    ) -> List[ExperimentResult]:
        """
        Listar experimentos con filtros.
        
        Args:
            name_filter: Filtrar por nombre
            tag_filter: Filtrar por tag
            
        Returns:
            Lista de resultados
        """
        results = list(self.experiments.values())
        
        if name_filter:
            results = [r for r in results if name_filter.lower() in r.config.name.lower()]
        
        if tag_filter:
            results = [r for r in results if tag_filter in r.config.tags]
        
        return results
    
    def _save_experiment(self, result: ExperimentResult) -> None:
        """Guardar experimento a disco."""
        filepath = self.storage_path / f"{result.experiment_id}.json"
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_experiments(self) -> None:
        """Cargar experimentos desde disco."""
        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Reconstruir objeto
                config_data = data.get("config", {})
                config = ExperimentConfig(**config_data)
                
                result = ExperimentResult(
                    experiment_id=data["experiment_id"],
                    config=config,
                    response=data["response"],
                    metrics=data["metrics"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    duration_ms=data["duration_ms"],
                    tokens_used=data["tokens_used"],
                    cost=data.get("cost", 0.0)
                )
                
                self.experiments[result.experiment_id] = result
            except Exception as e:
                logger.error(f"Error cargando experimento {filepath}: {e}")
        
        logger.info(f"Cargados {len(self.experiments)} experimentos")


# Instancia global
_experiment_tracker = ExperimentTracker()


def get_experiment_tracker() -> ExperimentTracker:
    """Obtener instancia global del tracker."""
    return _experiment_tracker



