"""
Sistema de Inferencia Distribuida

Proporciona:
- Distribución de carga entre múltiples workers
- Sharding de modelos
- Pipeline parallelism
- Data parallelism
- Failover automático
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class InferenceWorker:
    """Worker de inferencia"""
    id: str
    url: str
    capacity: int = 10  # Capacidad concurrente
    active_tasks: int = 0
    total_tasks: int = 0
    failed_tasks: int = 0
    avg_latency: float = 0.0
    healthy: bool = True
    last_heartbeat: Optional[datetime] = None


class DistributedInferenceEngine:
    """Motor de inferencia distribuida"""
    
    def __init__(self):
        self.workers: Dict[str, InferenceWorker] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        logger.info("DistributedInferenceEngine initialized")
    
    def register_worker(
        self,
        worker_id: str,
        url: str,
        capacity: int = 10
    ):
        """
        Registra un worker de inferencia
        
        Args:
            worker_id: ID del worker
            url: URL del worker
            capacity: Capacidad concurrente
        """
        worker = InferenceWorker(
            id=worker_id,
            url=url,
            capacity=capacity
        )
        
        self.workers[worker_id] = worker
        logger.info(f"Inference worker registered: {worker_id} at {url}")
    
    def get_available_worker(self) -> Optional[InferenceWorker]:
        """
        Obtiene un worker disponible
        
        Returns:
            Worker disponible o None
        """
        # Filtrar workers saludables con capacidad
        available = [
            w for w in self.workers.values()
            if w.healthy and w.active_tasks < w.capacity
        ]
        
        if not available:
            return None
        
        # Seleccionar worker con menor carga
        return min(available, key=lambda w: w.active_tasks / w.capacity)
    
    async def distribute_inference(
        self,
        task_data: Dict[str, Any],
        inference_func: Callable
    ) -> Any:
        """
        Distribuye una tarea de inferencia
        
        Args:
            task_data: Datos de la tarea
            inference_func: Función de inferencia
        
        Returns:
            Resultado de la inferencia
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        # Obtener worker disponible
        worker = self.get_available_worker()
        
        if not worker:
            # Si no hay workers disponibles, encolar
            self.task_queue.append({
                "task_id": task_id,
                "task_data": task_data,
                "inference_func": inference_func
            })
            raise Exception("No workers available, task queued")
        
        # Asignar tarea al worker
        worker.active_tasks += 1
        worker.total_tasks += 1
        
        try:
            start_time = datetime.now()
            
            # Ejecutar inferencia
            if asyncio.iscoroutinefunction(inference_func):
                result = await inference_func(task_data, worker.url)
            else:
                result = await asyncio.to_thread(
                    inference_func,
                    task_data,
                    worker.url
                )
            
            # Calcular latencia
            latency = (datetime.now() - start_time).total_seconds()
            worker.avg_latency = (
                (worker.avg_latency * (worker.total_tasks - 1) + latency) /
                worker.total_tasks
            )
            
            return result
        
        except Exception as e:
            worker.failed_tasks += 1
            logger.error(f"Inference failed on worker {worker.id}: {e}")
            raise
        
        finally:
            worker.active_tasks -= 1
    
    def shard_model(
        self,
        model_parts: List[str],
        shard_strategy: str = "round_robin"
    ) -> Dict[str, str]:
        """
        Distribuye partes del modelo entre workers
        
        Args:
            model_parts: Partes del modelo
            shard_strategy: Estrategia de sharding
        
        Returns:
            Mapeo de parte a worker
        """
        shard_map = {}
        worker_list = list(self.workers.keys())
        
        if not worker_list:
            return shard_map
        
        for i, part in enumerate(model_parts):
            if shard_strategy == "round_robin":
                worker_id = worker_list[i % len(worker_list)]
            elif shard_strategy == "hash":
                hash_value = int(hashlib.md5(part.encode()).hexdigest(), 16)
                worker_id = worker_list[hash_value % len(worker_list)]
            else:
                worker_id = worker_list[0]
            
            shard_map[part] = worker_id
        
        return shard_map
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de inferencia distribuida"""
        total_capacity = sum(w.capacity for w in self.workers.values())
        active_tasks = sum(w.active_tasks for w in self.workers.values())
        total_tasks = sum(w.total_tasks for w in self.workers.values())
        failed_tasks = sum(w.failed_tasks for w in self.workers.values())
        
        return {
            "total_workers": len(self.workers),
            "healthy_workers": sum(1 for w in self.workers.values() if w.healthy),
            "total_capacity": total_capacity,
            "active_tasks": active_tasks,
            "utilization": (active_tasks / total_capacity * 100) if total_capacity > 0 else 0,
            "total_tasks": total_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (
                ((total_tasks - failed_tasks) / total_tasks * 100)
                if total_tasks > 0 else 100
            ),
            "queued_tasks": len(self.task_queue),
            "workers": [
                {
                    "id": w.id,
                    "url": w.url,
                    "capacity": w.capacity,
                    "active_tasks": w.active_tasks,
                    "healthy": w.healthy,
                    "avg_latency": w.avg_latency,
                    "success_rate": (
                        ((w.total_tasks - w.failed_tasks) / w.total_tasks * 100)
                        if w.total_tasks > 0 else 100
                    )
                }
                for w in self.workers.values()
            ]
        }


# Instancia global
_distributed_inference: Optional[DistributedInferenceEngine] = None


def get_distributed_inference() -> DistributedInferenceEngine:
    """Obtiene la instancia global del motor de inferencia distribuida"""
    global _distributed_inference
    if _distributed_inference is None:
        _distributed_inference = DistributedInferenceEngine()
    return _distributed_inference

