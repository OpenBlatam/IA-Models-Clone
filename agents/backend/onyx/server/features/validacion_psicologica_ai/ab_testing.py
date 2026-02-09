"""
Sistema de Pruebas A/B
======================
Testing A/B para optimización
"""

from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum
import structlog
import random
from collections import defaultdict

logger = structlog.get_logger()


class Variant(str, Enum):
    """Variantes de prueba"""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


class ExperimentStatus(str, Enum):
    """Estado de experimento"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABExperiment:
    """Experimento A/B"""
    
    def __init__(
        self,
        id: UUID,
        name: str,
        description: str,
        variants: List[Variant],
        traffic_split: Dict[Variant, float],
        start_date: datetime,
        end_date: Optional[datetime] = None
    ):
        self.id = id
        self.name = name
        self.description = description
        self.variants = variants
        self.traffic_split = traffic_split
        self.start_date = start_date
        self.end_date = end_date
        self.status = ExperimentStatus.DRAFT
        self.created_at = datetime.utcnow()
        self.metrics: Dict[Variant, Dict[str, Any]] = defaultdict(lambda: {
            "participants": 0,
            "conversions": 0,
            "conversion_rate": 0.0
        })


class ABTestManager:
    """Gestor de pruebas A/B"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._experiments: Dict[UUID, ABExperiment] = {}
        self._user_assignments: Dict[UUID, Dict[UUID, Variant]] = defaultdict(dict)
        logger.info("ABTestManager initialized")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        traffic_split: Dict[Variant, float],
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> ABExperiment:
        """
        Crear experimento A/B
        
        Args:
            name: Nombre del experimento
            description: Descripción
            variants: Lista de variantes
            traffic_split: División de tráfico
            start_date: Fecha de inicio
            end_date: Fecha de fin (opcional)
            
        Returns:
            Experimento creado
        """
        # Validar que la suma de traffic_split sea 1.0
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
        
        experiment = ABExperiment(
            id=uuid4(),
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            start_date=start_date,
            end_date=end_date
        )
        
        self._experiments[experiment.id] = experiment
        
        logger.info("AB experiment created", experiment_id=str(experiment.id), name=name)
        
        return experiment
    
    def start_experiment(self, experiment_id: UUID) -> None:
        """Iniciar experimento"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.RUNNING
        logger.info("AB experiment started", experiment_id=str(experiment_id))
    
    def assign_variant(
        self,
        experiment_id: UUID,
        user_id: UUID
    ) -> Variant:
        """
        Asignar variante a usuario
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
            
        Returns:
            Variante asignada
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment.status != ExperimentStatus.RUNNING:
            return Variant.CONTROL
        
        # Verificar si ya está asignado
        if user_id in self._user_assignments[experiment_id]:
            return self._user_assignments[experiment_id][user_id]
        
        # Asignar según traffic split
        rand = random.random()
        cumulative = 0.0
        
        for variant, percentage in experiment.traffic_split.items():
            cumulative += percentage
            if rand <= cumulative:
                self._user_assignments[experiment_id][user_id] = variant
                experiment.metrics[variant]["participants"] += 1
                logger.debug(
                    "Variant assigned",
                    experiment_id=str(experiment_id),
                    user_id=str(user_id),
                    variant=variant.value
                )
                return variant
        
        # Fallback a control
        variant = Variant.CONTROL
        self._user_assignments[experiment_id][user_id] = variant
        return variant
    
    def record_conversion(
        self,
        experiment_id: UUID,
        user_id: UUID
    ) -> None:
        """
        Registrar conversión
        
        Args:
            experiment_id: ID del experimento
            user_id: ID del usuario
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return
        
        variant = self._user_assignments[experiment_id].get(user_id)
        if not variant:
            return
        
        experiment.metrics[variant]["conversions"] += 1
        
        # Recalcular conversion rate
        participants = experiment.metrics[variant]["participants"]
        conversions = experiment.metrics[variant]["conversions"]
        experiment.metrics[variant]["conversion_rate"] = (
            conversions / participants if participants > 0 else 0.0
        )
        
        logger.debug(
            "Conversion recorded",
            experiment_id=str(experiment_id),
            user_id=str(user_id),
            variant=variant.value
        )
    
    def get_experiment_results(self, experiment_id: UUID) -> Dict[str, Any]:
        """
        Obtener resultados del experimento
        
        Args:
            experiment_id: ID del experimento
            
        Returns:
            Resultados del experimento
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return {
            "experiment_id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status.value,
            "variants": {
                variant.value: {
                    "participants": metrics["participants"],
                    "conversions": metrics["conversions"],
                    "conversion_rate": metrics["conversion_rate"]
                }
                for variant, metrics in experiment.metrics.items()
            },
            "start_date": experiment.start_date.isoformat(),
            "end_date": experiment.end_date.isoformat() if experiment.end_date else None
        }


# Instancia global del gestor de pruebas A/B
ab_test_manager = ABTestManager()




