"""
Chaos Engineering - Ingeniería del caos para testing de resiliencia
===================================================================
"""

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class ChaosExperimentType(Enum):
    """Tipos de experimentos de caos"""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SERVICE_DOWN = "service_down"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"


class ExperimentStatus(Enum):
    """Estados de experimento"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ChaosExperiment:
    """Experimento de caos"""
    id: str
    name: str
    experiment_type: ChaosExperimentType
    target_service: str
    duration: float  # segundos
    intensity: float = 0.5  # 0.0 - 1.0
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "experiment_type": self.experiment_type.value,
            "target_service": self.target_service,
            "duration": self.duration,
            "intensity": self.intensity,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


class ChaosEngineer:
    """Ingeniero del caos"""
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: Dict[str, asyncio.Task] = {}
        self.monitoring_callbacks: List[Callable] = []
    
    def create_experiment(
        self,
        name: str,
        experiment_type: ChaosExperimentType,
        target_service: str,
        duration: float,
        intensity: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChaosExperiment:
        """Crea un experimento de caos"""
        import uuid
        
        experiment = ChaosExperiment(
            id=str(uuid.uuid4()),
            name=name,
            experiment_type=experiment_type,
            target_service=target_service,
            duration=duration,
            intensity=intensity,
            metadata=metadata or {}
        )
        
        self.experiments[experiment.id] = experiment
        logger.info(f"Experimento de caos {experiment.id} creado: {name}")
        return experiment
    
    async def run_experiment(self, experiment_id: str) -> bool:
        """Ejecuta un experimento"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        
        try:
            # Ejecutar experimento según tipo
            if experiment.experiment_type == ChaosExperimentType.LATENCY:
                await self._inject_latency(experiment)
            elif experiment.experiment_type == ChaosExperimentType.ERROR_RATE:
                await self._inject_errors(experiment)
            elif experiment.experiment_type == ChaosExperimentType.SERVICE_DOWN:
                await self._simulate_service_down(experiment)
            elif experiment.experiment_type == ChaosExperimentType.RESOURCE_EXHAUSTION:
                await self._simulate_resource_exhaustion(experiment)
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now()
            return True
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            logger.error(f"Error ejecutando experimento {experiment_id}: {e}")
            return False
    
    async def _inject_latency(self, experiment: ChaosExperiment):
        """Inyecta latencia"""
        # Simular inyección de latencia
        latency_ms = int(experiment.intensity * 1000)  # 0-1000ms
        logger.info(f"Inyectando {latency_ms}ms de latencia en {experiment.target_service}")
        await asyncio.sleep(experiment.duration)
    
    async def _inject_errors(self, experiment: ChaosExperiment):
        """Inyecta errores"""
        error_rate = experiment.intensity  # 0-100%
        logger.info(f"Inyectando {error_rate*100}% de errores en {experiment.target_service}")
        await asyncio.sleep(experiment.duration)
    
    async def _simulate_service_down(self, experiment: ChaosExperiment):
        """Simula caída de servicio"""
        logger.warning(f"Simulando caída de {experiment.target_service}")
        await asyncio.sleep(experiment.duration)
    
    async def _simulate_resource_exhaustion(self, experiment: ChaosExperiment):
        """Simula agotamiento de recursos"""
        logger.warning(f"Simulando agotamiento de recursos en {experiment.target_service}")
        await asyncio.sleep(experiment.duration)
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancela un experimento"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        if experiment.status == ExperimentStatus.RUNNING:
            if experiment_id in self.active_experiments:
                self.active_experiments[experiment_id].cancel()
                del self.active_experiments[experiment_id]
            
            experiment.status = ExperimentStatus.CANCELLED
            experiment.completed_at = datetime.now()
            return True
        
        return False
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[Dict[str, Any]]:
        """Lista experimentos"""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return [e.to_dict() for e in experiments]
    
    def register_monitoring_callback(self, callback: Callable):
        """Registra callback para monitoreo"""
        self.monitoring_callbacks.append(callback)




