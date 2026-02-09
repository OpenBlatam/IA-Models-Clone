"""
Routing Scalability Optimizations
==================================

Optimizaciones para escalabilidad horizontal.
Incluye: Load balancing, Distributed processing, Sharding, etc.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class LoadBalancer:
    """Balanceador de carga."""
    
    def __init__(self, strategy: str = "round_robin"):
        """
        Inicializar balanceador de carga.
        
        Args:
            strategy: Estrategia ('round_robin', 'least_connections', 'hash')
        """
        self.strategy = strategy
        self.servers: List[str] = []
        self.server_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'connections': 0,
            'requests': 0,
            'errors': 0
        })
        self.current_index = 0
        self.lock = threading.Lock()
    
    def add_server(self, server: str):
        """Agregar servidor al pool."""
        with self.lock:
            if server not in self.servers:
                self.servers.append(server)
                logger.info(f"Server added to load balancer: {server}")
    
    def remove_server(self, server: str):
        """Remover servidor del pool."""
        with self.lock:
            if server in self.servers:
                self.servers.remove(server)
                logger.info(f"Server removed from load balancer: {server}")
    
    def select_server(self, key: Optional[str] = None) -> Optional[str]:
        """
        Seleccionar servidor.
        
        Args:
            key: Clave para hash-based selection
        
        Returns:
            Servidor seleccionado
        """
        with self.lock:
            if not self.servers:
                return None
            
            if self.strategy == "round_robin":
                server = self.servers[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.servers)
                return server
            
            elif self.strategy == "least_connections":
                server = min(
                    self.servers,
                    key=lambda s: self.server_stats[s]['connections']
                )
                return server
            
            elif self.strategy == "hash" and key:
                index = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(self.servers)
                return self.servers[index]
            
            return self.servers[0]
    
    def record_request(self, server: str, success: bool = True):
        """Registrar request."""
        with self.lock:
            self.server_stats[server]['requests'] += 1
            if not success:
                self.server_stats[server]['errors'] += 1


class ShardingManager:
    """Gestor de sharding."""
    
    def __init__(self, num_shards: int = 4):
        """
        Inicializar gestor de sharding.
        
        Args:
            num_shards: Número de shards
        """
        self.num_shards = num_shards
        self.shards: Dict[int, Dict[str, Any]] = {i: {} for i in range(num_shards)}
        self.lock = threading.Lock()
    
    def get_shard(self, key: str) -> int:
        """
        Obtener shard para una clave.
        
        Args:
            key: Clave
        
        Returns:
            Índice del shard
        """
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_shards
    
    def get_shard_data(self, shard_id: int) -> Dict[str, Any]:
        """Obtener datos de un shard."""
        with self.lock:
            return self.shards[shard_id].copy()
    
    def put_shard_data(self, shard_id: int, key: str, value: Any):
        """Guardar datos en un shard."""
        with self.lock:
            self.shards[shard_id][key] = value
    
    def get_all_shards(self) -> Dict[int, Dict[str, Any]]:
        """Obtener todos los shards."""
        with self.lock:
            return {k: v.copy() for k, v in self.shards.items()}


class DistributedProcessor:
    """Procesador distribuido."""
    
    def __init__(self, load_balancer: LoadBalancer):
        """
        Inicializar procesador distribuido.
        
        Args:
            load_balancer: Balanceador de carga
        """
        self.load_balancer = load_balancer
        self.task_queue: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def submit_task(self, task_id: str, task: Callable, *args, **kwargs) -> str:
        """
        Enviar tarea para procesamiento distribuido.
        
        Args:
            task_id: ID de la tarea
            task: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            ID de la tarea
        """
        with self.lock:
            self.task_queue.append({
                'task_id': task_id,
                'task': task,
                'args': args,
                'kwargs': kwargs,
                'status': 'pending'
            })
        
        # En un sistema real, esto enviaría la tarea a un servidor
        # Por ahora, simulamos el procesamiento
        logger.info(f"Task {task_id} submitted for distributed processing")
        return task_id
    
    def get_result(self, task_id: str) -> Optional[Any]:
        """Obtener resultado de una tarea."""
        with self.lock:
            return self.results.get(task_id)


class ScalabilityOptimizer:
    """Optimizador completo de escalabilidad."""
    
    def __init__(self, num_shards: int = 4):
        """
        Inicializar optimizador de escalabilidad.
        
        Args:
            num_shards: Número de shards
        """
        self.load_balancer = LoadBalancer()
        self.sharding_manager = ShardingManager(num_shards)
        self.distributed_processor = DistributedProcessor(self.load_balancer)
    
    def add_server(self, server: str):
        """Agregar servidor."""
        self.load_balancer.add_server(server)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'num_servers': len(self.load_balancer.servers),
            'num_shards': self.sharding_manager.num_shards,
            'pending_tasks': len(self.distributed_processor.task_queue),
            'completed_tasks': len(self.distributed_processor.results)
        }

