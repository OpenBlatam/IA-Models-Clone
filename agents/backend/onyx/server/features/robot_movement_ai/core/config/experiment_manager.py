"""
Experiment Manager System
=========================

Sistema de gestión de experimentos A/B.
"""

import logging
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExperimentVariant:
    """Variante de experimento."""
    variant_id: str
    name: str
    weight: float  # 0-1, probabilidad de asignación
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Experimento."""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant] = field(default_factory=list)
    enabled: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Resultado de experimento."""
    experiment_id: str
    variant_id: str
    user_id: Optional[str] = None
    success: bool = False
    metric_value: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentManager:
    """
    Gestor de experimentos.
    
    Gestiona experimentos A/B.
    """
    
    def __init__(self):
        """Inicializar gestor de experimentos."""
        self.experiments: Dict[str, Experiment] = {}
        self.assignments: Dict[str, str] = {}  # user_id -> variant_id
        self.results: List[ExperimentResult] = []
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        enabled: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Experiment:
        """
        Crear experimento.
        
        Args:
            experiment_id: ID único del experimento
            name: Nombre del experimento
            description: Descripción
            variants: Lista de variantes
            enabled: Si está habilitado
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Experimento creado
        """
        experiment_variants = []
        for variant_data in variants:
            variant = ExperimentVariant(
                variant_id=variant_data["variant_id"],
                name=variant_data["name"],
                weight=variant_data.get("weight", 1.0 / len(variants)),
                config=variant_data.get("config", {})
            )
            experiment_variants.append(variant)
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=experiment_variants,
            enabled=enabled,
            start_date=start_date,
            end_date=end_date
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {name} ({experiment_id})")
        
        return experiment
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Asignar variante a usuario.
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario (opcional)
            
        Returns:
            ID de variante asignada
        """
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if not experiment.enabled:
            return None
        
        # Verificar fechas
        now = datetime.now().isoformat()
        if experiment.start_date and now < experiment.start_date:
            return None
        if experiment.end_date and now > experiment.end_date:
            return None
        
        # Si ya está asignado, retornar asignación existente
        if user_id and user_id in self.assignments:
            return self.assignments[user_id]
        
        # Asignar según pesos
        total_weight = sum(v.weight for v in experiment.variants)
        r = random.random() * total_weight
        
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.weight
            if r <= cumulative:
                if user_id:
                    self.assignments[user_id] = variant.variant_id
                return variant.variant_id
        
        # Fallback a primera variante
        if experiment.variants:
            variant_id = experiment.variants[0].variant_id
            if user_id:
                self.assignments[user_id] = variant_id
            return variant_id
        
        return None
    
    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        success: bool = True,
        metric_value: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> ExperimentResult:
        """
        Registrar resultado de experimento.
        
        Args:
            experiment_id: ID del experimento
            variant_id: ID de la variante
            success: Si fue exitoso
            metric_value: Valor de métrica
            user_id: ID del usuario
            
        Returns:
            Resultado registrado
        """
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            success=success,
            metric_value=metric_value
        )
        
        self.results.append(result)
        return result
    
    def get_experiment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de experimento."""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        variant_stats = {}
        for variant in experiment.variants:
            variant_results = [r for r in experiment_results if r.variant_id == variant.variant_id]
            
            if variant_results:
                success_count = sum(1 for r in variant_results if r.success)
                total_count = len(variant_results)
                avg_metric = sum(r.metric_value for r in variant_results if r.metric_value is not None) / total_count if total_count > 0 else 0.0
                
                variant_stats[variant.variant_id] = {
                    "name": variant.name,
                    "total_participants": total_count,
                    "success_count": success_count,
                    "success_rate": success_count / total_count if total_count > 0 else 0.0,
                    "average_metric": avg_metric
                }
            else:
                variant_stats[variant.variant_id] = {
                    "name": variant.name,
                    "total_participants": 0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "average_metric": 0.0
                }
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "total_participants": len(experiment_results),
            "variants": variant_stats
        }
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Obtener experimento."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Experiment]:
        """Listar todos los experimentos."""
        return list(self.experiments.values())


# Instancia global
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Obtener instancia global del gestor de experimentos."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager






