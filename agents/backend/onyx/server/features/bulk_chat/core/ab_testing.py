"""
A/B Testing - Framework de A/B Testing
=======================================

Sistema completo de A/B testing para experimentos.
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


class Variant(Enum):
    """Variantes de test."""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


@dataclass
class Experiment:
    """Experimento A/B."""
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    traffic_split: Dict[Variant, float] = field(default_factory=dict)
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Resultado de experimento."""
    experiment_id: str
    variant: Variant
    user_id: str
    timestamp: datetime
    metric_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework:
    """Framework de A/B testing."""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.results: List[ExperimentResult] = []
        self.user_assignments: Dict[str, Dict[str, Variant]] = {}  # {user_id: {experiment_id: variant}}
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: List[Variant],
        traffic_split: Optional[Dict[Variant, float]] = None,
    ) -> Experiment:
        """
        Crear nuevo experimento.
        
        Args:
            experiment_id: ID único del experimento
            name: Nombre del experimento
            description: Descripción
            variants: Lista de variantes
            traffic_split: División de tráfico (por defecto igual)
        
        Returns:
            Experimento creado
        """
        if traffic_split is None:
            # División igual
            split_per_variant = 1.0 / len(variants)
            traffic_split = {v: split_per_variant for v in variants}
        
        # Normalizar
        total = sum(traffic_split.values())
        traffic_split = {k: v / total for k, v in traffic_split.items()}
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {experiment_id}")
        
        return experiment
    
    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Optional[Variant]:
        """
        Obtener variante asignada a usuario.
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
        
        Returns:
            Variante asignada
        """
        # Verificar si ya está asignado
        if user_id in self.user_assignments:
            if experiment_id in self.user_assignments[user_id]:
                return self.user_assignments[user_id][experiment_id]
        
        # Verificar si el experimento existe y está activo
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return None
        
        # Verificar fechas
        now = datetime.now()
        if experiment.start_date > now:
            return None
        
        if experiment.end_date and experiment.end_date < now:
            return None
        
        # Asignar variante usando consistent hashing
        variant = self._assign_variant(experiment, user_id)
        
        # Guardar asignación
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant
        
        return variant
    
    def _assign_variant(
        self,
        experiment: Experiment,
        user_id: str,
    ) -> Variant:
        """Asignar variante usando consistent hashing."""
        # Hash del user_id + experiment_id para consistencia
        hash_input = f"{experiment.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalizar a 0-1
        normalized = (hash_value % 10000) / 10000.0
        
        # Asignar basado en traffic_split
        cumulative = 0.0
        for variant, split in experiment.traffic_split.items():
            cumulative += split
            if normalized <= cumulative:
                return variant
        
        # Fallback a primera variante
        return experiment.variants[0]
    
    async def record_result(
        self,
        experiment_id: str,
        variant: Variant,
        user_id: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registrar resultado de experimento."""
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            timestamp=datetime.now(),
            metric_value=metric_value,
            metadata=metadata or {},
        )
        
        self.results.append(result)
        logger.debug(f"Recorded result for {experiment_id}: {variant.value} = {metric_value}")
    
    async def get_experiment_stats(
        self,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """Obtener estadísticas de experimento."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {}
        
        experiment_results = [
            r for r in self.results
            if r.experiment_id == experiment_id
        ]
        
        stats = {}
        for variant in experiment.variants:
            variant_results = [
                r.metric_value
                for r in experiment_results
                if r.variant == variant
            ]
            
            if variant_results:
                stats[variant.value] = {
                    "count": len(variant_results),
                    "mean": sum(variant_results) / len(variant_results),
                    "min": min(variant_results),
                    "max": max(variant_results),
                }
            else:
                stats[variant.value] = {
                    "count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "is_active": experiment.is_active,
            "total_results": len(experiment_results),
            "variants": stats,
        }
    
    def stop_experiment(self, experiment_id: str):
        """Detener experimento."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].is_active = False
            self.experiments[experiment_id].end_date = datetime.now()
            logger.info(f"Stopped experiment: {experiment_id}")
































