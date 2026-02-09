"""
Sistema de A/B Testing

Proporciona:
- Experimentos A/B
- Asignación de variantes
- Tracking de métricas
- Análisis estadístico
- Significancia estadística
"""

import logging
import hashlib
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class Variant(Enum):
    """Variantes de experimento"""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"


@dataclass
class Experiment:
    """Experimento A/B"""
    id: str
    name: str
    description: Optional[str] = None
    variants: List[str] = field(default_factory=lambda: ["control", "variant_a"])
    traffic_split: Dict[str, float] = field(default_factory=lambda: {"control": 0.5, "variant_a": 0.5})
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metrics: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Resultado de un experimento"""
    experiment_id: str
    variant: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)


class ABTestingService:
    """Servicio de A/B testing"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.results: List[ExperimentResult] = []
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        logger.info("ABTestingService initialized")
    
    def create_experiment(
        self,
        name: str,
        variants: List[str] = None,
        traffic_split: Dict[str, float] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Crea un nuevo experimento
        
        Args:
            name: Nombre del experimento
            variants: Lista de variantes
            traffic_split: División de tráfico (debe sumar 1.0)
            description: Descripción
        
        Returns:
            ID del experimento
        """
        import uuid
        experiment_id = str(uuid.uuid4())
        
        if variants is None:
            variants = ["control", "variant_a"]
        
        if traffic_split is None:
            # División equitativa
            split = 1.0 / len(variants)
            traffic_split = {v: split for v in variants}
        
        # Validar que la suma sea 1.0
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Experiment created: {experiment_id} - {name}")
        
        return experiment_id
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        force_variant: Optional[str] = None
    ) -> str:
        """
        Asigna una variante a un usuario
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
            force_variant: Forzar una variante específica (para testing)
        
        Returns:
            Variante asignada
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if not experiment.active:
            return experiment.variants[0]  # Devolver control si está inactivo
        
        # Verificar si ya tiene asignación
        if experiment_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][experiment_id]
        
        # Forzar variante si se especifica
        if force_variant:
            if force_variant in experiment.variants:
                self.user_assignments[user_id][experiment_id] = force_variant
                return force_variant
        
        # Asignar basado en hash consistente
        variant = self._hash_assign(user_id, experiment_id, experiment)
        self.user_assignments[user_id][experiment_id] = variant
        
        return variant
    
    def _hash_assign(
        self,
        user_id: str,
        experiment_id: str,
        experiment: Experiment
    ) -> str:
        """Asigna variante usando hash consistente"""
        # Crear hash determinístico
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000.0
        
        # Asignar basado en traffic split
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += experiment.traffic_split.get(variant, 0)
            if random_value < cumulative:
                return variant
        
        # Fallback a primera variante
        return experiment.variants[0]
    
    def record_result(
        self,
        experiment_id: str,
        user_id: str,
        variant: str,
        metrics: Dict[str, float]
    ):
        """
        Registra resultado de un experimento
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
            variant: Variante asignada
            metrics: Métricas del resultado
        """
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            metrics=metrics
        )
        
        self.results.append(result)
        logger.debug(f"Result recorded for experiment {experiment_id}, variant {variant}")
    
    def analyze_experiment(
        self,
        experiment_id: str,
        metric: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Analiza un experimento
        
        Args:
            experiment_id: ID del experimento
            metric: Métrica a analizar
            confidence_level: Nivel de confianza (0.95 = 95%)
        
        Returns:
            Análisis del experimento
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Filtrar resultados del experimento
        experiment_results = [
            r for r in self.results
            if r.experiment_id == experiment_id and metric in r.metrics
        ]
        
        if not experiment_results:
            return {
                "experiment_id": experiment_id,
                "metric": metric,
                "error": "No results found"
            }
        
        # Agrupar por variante
        variant_data = defaultdict(list)
        for result in experiment_results:
            variant_data[result.variant].append(result.metrics[metric])
        
        # Calcular estadísticas por variante
        variant_stats = {}
        for variant, values in variant_data.items():
            if values:
                variant_stats[variant] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "std": self._calculate_std(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Calcular significancia estadística
        if "control" in variant_stats and len(variant_stats) > 1:
            control_stats = variant_stats["control"]
            significance = {}
            
            for variant, stats in variant_stats.items():
                if variant != "control":
                    significance[variant] = self._calculate_significance(
                        control_stats,
                        stats,
                        confidence_level
                    )
        
        return {
            "experiment_id": experiment_id,
            "metric": metric,
            "variant_stats": variant_stats,
            "significance": significance if "significance" in locals() else {},
            "total_results": len(experiment_results),
            "confidence_level": confidence_level
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calcula desviación estándar"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _calculate_significance(
        self,
        control_stats: Dict[str, float],
        variant_stats: Dict[str, float],
        confidence_level: float
    ) -> Dict[str, Any]:
        """Calcula significancia estadística (t-test simplificado)"""
        n1, mean1, std1 = (
            control_stats["count"],
            control_stats["mean"],
            control_stats["std"]
        )
        n2, mean2, std2 = (
            variant_stats["count"],
            variant_stats["mean"],
            variant_stats["std"]
        )
        
        # Calcular diferencia
        difference = mean2 - mean1
        percent_change = (difference / mean1 * 100) if mean1 > 0 else 0
        
        # Calcular error estándar combinado
        se1 = std1 / math.sqrt(n1) if n1 > 0 else 0
        se2 = std2 / math.sqrt(n2) if n2 > 0 else 0
        combined_se = math.sqrt(se1**2 + se2**2)
        
        # Calcular t-statistic (simplificado)
        if combined_se > 0:
            t_stat = abs(difference) / combined_se
            # Para simplificar, asumimos significancia si t > 2 (aproximadamente 95% confianza)
            is_significant = t_stat > 2.0
        else:
            t_stat = 0
            is_significant = False
        
        return {
            "difference": difference,
            "percent_change": round(percent_change, 2),
            "t_statistic": round(t_stat, 3),
            "is_significant": is_significant,
            "confidence_level": confidence_level
        }
    
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de un experimento"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {}
        
        # Contar asignaciones por variante
        variant_counts = defaultdict(int)
        for user_assignments in self.user_assignments.values():
            if experiment_id in user_assignments:
                variant_counts[user_assignments[experiment_id]] += 1
        
        # Contar resultados por variante
        result_counts = defaultdict(int)
        for result in self.results:
            if result.experiment_id == experiment_id:
                result_counts[result.variant] += 1
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "active": experiment.active,
            "assignments_by_variant": dict(variant_counts),
            "results_by_variant": dict(result_counts),
            "total_assignments": sum(variant_counts.values()),
            "total_results": sum(result_counts.values())
        }


# Instancia global
_ab_testing_service: Optional[ABTestingService] = None


def get_ab_testing_service() -> ABTestingService:
    """Obtiene la instancia global del servicio de A/B testing"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService()
    return _ab_testing_service

