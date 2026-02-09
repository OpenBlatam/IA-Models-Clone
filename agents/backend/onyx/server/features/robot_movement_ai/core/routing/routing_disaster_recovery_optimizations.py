"""
Routing Disaster Recovery Optimizations
=======================================

Optimizaciones para disaster recovery.
Incluye: Backup strategies, Failover, Recovery procedures, etc.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Estrategias de recuperación."""
    HOT_STANDBY = "hot_standby"
    WARM_STANDBY = "warm_standby"
    COLD_STANDBY = "cold_standby"
    MULTI_SITE = "multi_site"


class BackupStrategy:
    """Estrategia de backup."""
    
    def __init__(self, strategy: str = "incremental", interval: float = 3600.0):
        """
        Inicializar estrategia.
        
        Args:
            strategy: Tipo de backup (full, incremental, differential)
            interval: Intervalo entre backups en segundos
        """
        self.strategy = strategy
        self.interval = interval
        self.last_backup: Optional[float] = None
        self.backup_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def should_backup(self) -> bool:
        """Verificar si se debe hacer backup."""
        with self.lock:
            if self.last_backup is None:
                return True
            return time.time() - self.last_backup >= self.interval
    
    def record_backup(self, backup_id: str, size: float):
        """Registrar backup."""
        with self.lock:
            self.last_backup = time.time()
            self.backup_history.append({
                'backup_id': backup_id,
                'timestamp': self.last_backup,
                'size': size,
                'strategy': self.strategy
            })


class FailoverManager:
    """Gestor de failover."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.primary_node: Optional[str] = None
        self.secondary_nodes: List[str] = []
        self.current_node: Optional[str] = None
        self.failover_history: deque = deque(maxlen=100)
        self.health_checks: Dict[str, bool] = {}
        self.lock = threading.Lock()
    
    def set_primary(self, node_id: str):
        """Establecer nodo primario."""
        with self.lock:
            self.primary_node = node_id
            self.current_node = node_id
    
    def add_secondary(self, node_id: str):
        """Agregar nodo secundario."""
        with self.lock:
            if node_id not in self.secondary_nodes:
                self.secondary_nodes.append(node_id)
    
    def check_health(self, node_id: str) -> bool:
        """Verificar salud de nodo."""
        with self.lock:
            return self.health_checks.get(node_id, True)
    
    def update_health(self, node_id: str, healthy: bool):
        """Actualizar salud de nodo."""
        with self.lock:
            self.health_checks[node_id] = healthy
    
    def failover(self) -> Optional[str]:
        """Ejecutar failover."""
        with self.lock:
            if not self.secondary_nodes:
                return None
            
            # Seleccionar primer nodo secundario saludable
            for node_id in self.secondary_nodes:
                if self.check_health(node_id):
                    old_node = self.current_node
                    self.current_node = node_id
                    self.failover_history.append({
                        'from_node': old_node,
                        'to_node': node_id,
                        'timestamp': time.time()
                    })
                    logger.warning(f"Failover executed: {old_node} -> {node_id}")
                    return node_id
            
            return None


class RecoveryProcedure:
    """Procedimiento de recuperación."""
    
    def __init__(self):
        """Inicializar procedimiento."""
        self.procedures: Dict[str, List[Any]] = {}
        self.recovery_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def register_procedure(self, scenario: str, steps: List[Any]):
        """Registrar procedimiento de recuperación."""
        with self.lock:
            self.procedures[scenario] = steps
    
    def execute_recovery(self, scenario: str) -> bool:
        """Ejecutar recuperación."""
        with self.lock:
            steps = self.procedures.get(scenario, [])
        
        if not steps:
            logger.error(f"No recovery procedure found for scenario: {scenario}")
            return False
        
        start_time = time.time()
        try:
            for step in steps:
                step()
            
            recovery_time = time.time() - start_time
            with self.lock:
                self.recovery_history.append({
                    'scenario': scenario,
                    'timestamp': time.time(),
                    'recovery_time': recovery_time,
                    'success': True
                })
            
            logger.info(f"Recovery completed for scenario: {scenario} in {recovery_time:.2f}s")
            return True
        except Exception as e:
            recovery_time = time.time() - start_time
            with self.lock:
                self.recovery_history.append({
                    'scenario': scenario,
                    'timestamp': time.time(),
                    'recovery_time': recovery_time,
                    'success': False,
                    'error': str(e)
                })
            
            logger.error(f"Recovery failed for scenario: {scenario}: {e}")
            return False


class DisasterRecoveryOptimizer:
    """Optimizador completo de disaster recovery."""
    
    def __init__(self, recovery_strategy: RecoveryStrategy = RecoveryStrategy.HOT_STANDBY):
        """Inicializar optimizador."""
        self.recovery_strategy = recovery_strategy
        self.backup_strategy = BackupStrategy()
        self.failover_manager = FailoverManager()
        self.recovery_procedure = RecoveryProcedure()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'recovery_strategy': self.recovery_strategy.value,
            'backup_stats': {
                'strategy': self.backup_strategy.strategy,
                'last_backup': self.backup_strategy.last_backup,
                'backup_count': len(self.backup_strategy.backup_history)
            },
            'failover_stats': {
                'primary_node': self.failover_manager.primary_node,
                'current_node': self.failover_manager.current_node,
                'secondary_nodes': len(self.failover_manager.secondary_nodes),
                'failover_count': len(self.failover_manager.failover_history)
            },
            'recovery_stats': {
                'procedures_registered': len(self.recovery_procedure.procedures),
                'recovery_count': len(self.recovery_procedure.recovery_history)
            }
        }

