"""
Routing A/B Testing Optimizations
==================================

Optimizaciones para A/B testing y feature flags.
Incluye: Experiment management, Feature flags, Statistical analysis, etc.
"""

import logging
import random
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Experimento A/B."""
    name: str
    variants: List[str]
    traffic_split: Dict[str, float]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    active: bool = True
    metrics: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))


class ExperimentManager:
    """Gestor de experimentos A/B."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.experiments: Dict[str, Experiment] = {}
        self.assignments: Dict[str, Dict[str, str]] = {}  # user_id -> experiment -> variant
        self.lock = threading.Lock()
    
    def create_experiment(
        self,
        name: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None
    ) -> Experiment:
        """
        Crear experimento.
        
        Args:
            name: Nombre del experimento
            variants: Lista de variantes
            traffic_split: División de tráfico (por defecto: igual)
        
        Returns:
            Experimento creado
        """
        if traffic_split is None:
            traffic_split = {v: 1.0 / len(variants) for v in variants}
        
        experiment = Experiment(
            name=name,
            variants=variants,
            traffic_split=traffic_split
        )
        
        with self.lock:
            self.experiments[name] = experiment
        
        logger.info(f"Experiment created: {name}")
        return experiment
    
    def assign_variant(self, experiment_name: str, user_id: str) -> str:
        """
        Asignar variante a usuario.
        
        Args:
            experiment_name: Nombre del experimento
            user_id: ID del usuario
        
        Returns:
            Variante asignada
        """
        with self.lock:
            if user_id not in self.assignments:
                self.assignments[user_id] = {}
            
            if experiment_name in self.assignments[user_id]:
                return self.assignments[user_id][experiment_name]
            
            experiment = self.experiments.get(experiment_name)
            if not experiment or not experiment.active:
                return "control"
            
            # Asignar basado en traffic split
            rand = random.random()
            cumulative = 0.0
            for variant, split in experiment.traffic_split.items():
                cumulative += split
                if rand <= cumulative:
                    self.assignments[user_id][experiment_name] = variant
                    return variant
            
            # Fallback
            variant = experiment.variants[0]
            self.assignments[user_id][experiment_name] = variant
            return variant
    
    def record_metric(self, experiment_name: str, variant: str, metric_name: str, value: float):
        """Registrar métrica."""
        with self.lock:
            if experiment_name in self.experiments:
                experiment = self.experiments[experiment_name]
                experiment.metrics[variant][metric_name] += value
    
    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """Obtener resultados del experimento."""
        with self.lock:
            experiment = self.experiments.get(experiment_name)
            if not experiment:
                return {}
            
            return {
                'name': experiment.name,
                'variants': experiment.variants,
                'traffic_split': experiment.traffic_split,
                'metrics': dict(experiment.metrics),
                'active': experiment.active
            }


class FeatureFlagManager:
    """Gestor de feature flags."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.flags: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def set_flag(self, flag_name: str, enabled: bool, rollout_percentage: float = 100.0):
        """
        Establecer feature flag.
        
        Args:
            flag_name: Nombre del flag
            enabled: Si está habilitado
            rollout_percentage: Porcentaje de rollout (0-100)
        """
        with self.lock:
            self.flags[flag_name] = {
                'enabled': enabled,
                'rollout_percentage': rollout_percentage,
                'updated_at': time.time()
            }
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """
        Verificar si feature flag está habilitado.
        
        Args:
            flag_name: Nombre del flag
            user_id: ID del usuario (para rollout gradual)
        
        Returns:
            True si está habilitado
        """
        with self.lock:
            flag = self.flags.get(flag_name)
            if not flag:
                return False
            
            if not flag['enabled']:
                return False
            
            # Rollout gradual
            if flag['rollout_percentage'] < 100.0 and user_id:
                # Hash user_id para consistencia
                user_hash = hash(user_id) % 100
                return user_hash < flag['rollout_percentage']
            
            return True


class ABTestingOptimizer:
    """Optimizador completo de A/B testing."""
    
    def __init__(self):
        """Inicializar optimizador."""
        self.experiment_manager = ExperimentManager()
        self.feature_flag_manager = FeatureFlagManager()
    
    def create_experiment(self, name: str, variants: List[str], traffic_split: Optional[Dict[str, float]] = None):
        """Crear experimento."""
        return self.experiment_manager.create_experiment(name, variants, traffic_split)
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Obtener variante para usuario."""
        return self.experiment_manager.assign_variant(experiment_name, user_id)
    
    def set_feature_flag(self, flag_name: str, enabled: bool, rollout_percentage: float = 100.0):
        """Establecer feature flag."""
        self.feature_flag_manager.set_flag(flag_name, enabled, rollout_percentage)
    
    def is_feature_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Verificar si feature está habilitado."""
        return self.feature_flag_manager.is_enabled(flag_name, user_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'num_experiments': len(self.experiment_manager.experiments),
            'num_assignments': len(self.experiment_manager.assignments),
            'num_feature_flags': len(self.feature_flag_manager.flags)
        }

