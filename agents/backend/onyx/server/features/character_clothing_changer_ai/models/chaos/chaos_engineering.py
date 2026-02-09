"""
Chaos Engineering System
========================
Sistema de chaos engineering para testing de resiliencia
"""

import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class ChaosExperimentType(Enum):
    """Tipos de experimentos de chaos"""
    LATENCY = "latency"
    FAILURE = "failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    CUSTOM = "custom"


class ExperimentStatus(Enum):
    """Estados de experimento"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ChaosExperiment:
    """Experimento de chaos"""
    id: str
    name: str
    description: str
    experiment_type: ChaosExperimentType
    target_service: str
    config: Dict[str, Any]
    status: ExperimentStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}


class ChaosEngineering:
    """
    Sistema de chaos engineering
    """
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_handlers: Dict[ChaosExperimentType, Callable] = {}
        self._init_default_handlers()
    
    def _init_default_handlers(self):
        """Inicializar handlers por defecto"""
        self.experiment_handlers[ChaosExperimentType.LATENCY] = self._inject_latency
        self.experiment_handlers[ChaosExperimentType.FAILURE] = self._inject_failure
        self.experiment_handlers[ChaosExperimentType.RESOURCE_EXHAUSTION] = self._exhaust_resources
        self.experiment_handlers[ChaosExperimentType.NETWORK_PARTITION] = self._simulate_network_partition
    
    def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ChaosExperimentType,
        target_service: str,
        config: Dict[str, Any]
    ) -> ChaosExperiment:
        """
        Crear experimento de chaos
        
        Args:
            name: Nombre del experimento
            description: Descripción
            experiment_type: Tipo de experimento
            target_service: Servicio objetivo
            config: Configuración del experimento
        """
        experiment_id = f"chaos_{int(time.time())}"
        
        experiment = ChaosExperiment(
            id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            target_service=target_service,
            config=config,
            status=ExperimentStatus.SCHEDULED,
            created_at=time.time()
        )
        
        self.experiments[experiment_id] = experiment
        return experiment
    
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Ejecutar experimento
        
        Args:
            experiment_id: ID del experimento
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()
        self.active_experiments[experiment_id] = experiment
        
        try:
            handler = self.experiment_handlers.get(experiment.experiment_type)
            if not handler:
                raise ValueError(f"No handler for experiment type {experiment.experiment_type}")
            
            # Ejecutar experimento
            result = handler(experiment.target_service, experiment.config)
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = time.time()
            experiment.results = result
            
            return result
            
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.completed_at = time.time()
            experiment.results = {'error': str(e)}
            raise
        finally:
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
    
    def _inject_latency(self, target_service: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Inyectar latencia"""
        latency_ms = config.get('latency_ms', 1000)
        duration = config.get('duration', 60)
        
        # En implementación real, inyectar latencia en el servicio
        # Por ahora, simular
        time.sleep(0.1)
        
        return {
            'type': 'latency',
            'latency_injected_ms': latency_ms,
            'duration_seconds': duration,
            'target_service': target_service
        }
    
    def _inject_failure(self, target_service: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Inyectar fallos"""
        failure_rate = config.get('failure_rate', 0.5)
        duration = config.get('duration', 60)
        
        return {
            'type': 'failure',
            'failure_rate': failure_rate,
            'duration_seconds': duration,
            'target_service': target_service
        }
    
    def _exhaust_resources(self, target_service: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Agotar recursos"""
        resource_type = config.get('resource_type', 'memory')
        percentage = config.get('percentage', 80)
        
        return {
            'type': 'resource_exhaustion',
            'resource_type': resource_type,
            'percentage': percentage,
            'target_service': target_service
        }
    
    def _simulate_network_partition(self, target_service: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simular partición de red"""
        partition_duration = config.get('duration', 60)
        
        return {
            'type': 'network_partition',
            'duration_seconds': partition_duration,
            'target_service': target_service
        }
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Detener experimento"""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            experiment.status = ExperimentStatus.CANCELLED
            experiment.completed_at = time.time()
            del self.active_experiments[experiment_id]
            return True
        return False
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Obtener resultados de experimento"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        return {
            'id': experiment.id,
            'name': experiment.name,
            'status': experiment.status.value,
            'results': experiment.results,
            'started_at': experiment.started_at,
            'completed_at': experiment.completed_at,
            'duration': (
                (experiment.completed_at or time.time()) - (experiment.started_at or experiment.created_at)
                if experiment.started_at else None
            )
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de chaos engineering"""
        status_counts = {}
        for experiment in self.experiments.values():
            status = experiment.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_experiments': len(self.experiments),
            'active_experiments': len(self.active_experiments),
            'status_counts': status_counts,
            'experiment_types': {
                exp_type.value: len([
                    e for e in self.experiments.values()
                    if e.experiment_type == exp_type
                ])
                for exp_type in ChaosExperimentType
            }
        }


# Instancia global
chaos_engineering = ChaosEngineering()

