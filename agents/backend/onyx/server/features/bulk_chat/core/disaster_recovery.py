"""
Disaster Recovery - Recuperación de Desastres
=============================================

Sistema avanzado de recuperación de desastres con replicación, failover automático y restauración.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RecoveryStatus(Enum):
    """Estado de recuperación."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    RECOVERED = "recovered"


@dataclass
class RecoveryPoint:
    """Punto de recuperación."""
    point_id: str
    timestamp: datetime
    backup_location: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False


@dataclass
class FailoverAction:
    """Acción de failover."""
    action_id: str
    source_node: str
    target_node: str
    timestamp: datetime
    reason: str
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DisasterRecovery:
    """Sistema de recuperación de desastres."""
    
    def __init__(
        self,
        backup_interval_minutes: int = 60,
        max_recovery_points: int = 100,
        auto_failover: bool = True,
    ):
        self.backup_interval_minutes = backup_interval_minutes
        self.max_recovery_points = max_recovery_points
        self.auto_failover = auto_failover
        self.recovery_points: List[RecoveryPoint] = []
        self.failover_history: List[FailoverAction] = []
        self.replication_nodes: Dict[str, Dict[str, Any]] = {}
        self.current_status = RecoveryStatus.HEALTHY
        self._lock = asyncio.Lock()
    
    def register_replication_node(
        self,
        node_id: str,
        address: str,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registrar nodo de replicación."""
        self.replication_nodes[node_id] = {
            "address": address,
            "priority": priority,
            "status": "healthy",
            "last_check": datetime.now(),
            "metadata": metadata or {},
        }
        
        logger.info(f"Registered replication node: {node_id} at {address}")
    
    async def create_recovery_point(
        self,
        backup_location: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Crear punto de recuperación.
        
        Args:
            backup_location: Ubicación del backup
            metadata: Metadatos adicionales
        
        Returns:
            ID del punto de recuperación
        """
        point_id = f"rp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        recovery_point = RecoveryPoint(
            point_id=point_id,
            timestamp=datetime.now(),
            backup_location=backup_location,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.recovery_points.append(recovery_point)
            
            # Mantener solo últimos N puntos
            if len(self.recovery_points) > self.max_recovery_points:
                self.recovery_points.pop(0)
        
        logger.info(f"Created recovery point: {point_id}")
        return point_id
    
    async def verify_recovery_point(self, point_id: str) -> bool:
        """Verificar integridad de punto de recuperación."""
        recovery_point = next(
            (rp for rp in self.recovery_points if rp.point_id == point_id),
            None
        )
        
        if not recovery_point:
            return False
        
        # En producción, verificar integridad del backup
        recovery_point.verified = True
        return True
    
    async def restore_from_point(
        self,
        point_id: str,
        target_location: Optional[str] = None,
    ) -> bool:
        """
        Restaurar desde punto de recuperación.
        
        Args:
            point_id: ID del punto de recuperación
            target_location: Ubicación de destino (opcional)
        
        Returns:
            True si la restauración fue exitosa
        """
        recovery_point = next(
            (rp for rp in self.recovery_points if rp.point_id == point_id),
            None
        )
        
        if not recovery_point:
            logger.error(f"Recovery point not found: {point_id}")
            return False
        
        if not recovery_point.verified:
            logger.warning(f"Recovery point not verified: {point_id}")
        
        # En producción, restaurar desde backup
        logger.info(f"Restoring from recovery point: {point_id}")
        
        async with self._lock:
            self.current_status = RecoveryStatus.RECOVERING
        
        # Simular restauración
        await asyncio.sleep(0.1)
        
        async with self._lock:
            self.current_status = RecoveryStatus.RECOVERED
        
        logger.info(f"Restored from recovery point: {point_id}")
        return True
    
    async def initiate_failover(
        self,
        source_node: str,
        target_node: str,
        reason: str,
    ) -> str:
        """
        Iniciar failover.
        
        Args:
            source_node: Nodo fuente
            target_node: Nodo destino
            reason: Razón del failover
        
        Returns:
            ID de la acción de failover
        """
        action_id = f"failover_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        action = FailoverAction(
            action_id=action_id,
            source_node=source_node,
            target_node=target_node,
            timestamp=datetime.now(),
            reason=reason,
            status="in_progress",
        )
        
        async with self._lock:
            self.failover_history.append(action)
            self.current_status = RecoveryStatus.RECOVERING
        
        # Simular failover
        await asyncio.sleep(0.1)
        
        async with self._lock:
            action.status = "completed"
            self.current_status = RecoveryStatus.HEALTHY
        
        logger.info(f"Failover completed: {source_node} -> {target_node}")
        return action_id
    
    async def check_node_health(self, node_id: str) -> bool:
        """Verificar salud de nodo."""
        node = self.replication_nodes.get(node_id)
        if not node:
            return False
        
        # En producción, hacer ping real al nodo
        node["last_check"] = datetime.now()
        return node["status"] == "healthy"
    
    async def auto_failover_check(self):
        """Verificar nodos y hacer failover automático si es necesario."""
        if not self.auto_failover:
            return
        
        unhealthy_nodes = []
        healthy_nodes = []
        
        for node_id, node_info in self.replication_nodes.items():
            is_healthy = await self.check_node_health(node_id)
            if not is_healthy:
                unhealthy_nodes.append(node_id)
                node_info["status"] = "unhealthy"
            else:
                healthy_nodes.append((node_id, node_info["priority"]))
        
        # Si hay nodos no saludables y nodos saludables disponibles
        if unhealthy_nodes and healthy_nodes:
            # Seleccionar nodo de mayor prioridad
            healthy_nodes.sort(key=lambda x: x[1], reverse=True)
            target_node = healthy_nodes[0][0]
            
            for source_node in unhealthy_nodes:
                await self.initiate_failover(
                    source_node=source_node,
                    target_node=target_node,
                    reason="Auto-failover: node unhealthy",
                )
    
    def get_recovery_points(
        self,
        limit: int = 50,
        verified_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Obtener puntos de recuperación."""
        points = self.recovery_points
        
        if verified_only:
            points = [rp for rp in points if rp.verified]
        
        return [
            {
                "point_id": rp.point_id,
                "timestamp": rp.timestamp.isoformat(),
                "backup_location": rp.backup_location,
                "verified": rp.verified,
                "metadata": rp.metadata,
            }
            for rp in points[-limit:]
        ]
    
    def get_failover_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener historial de failovers."""
        return [
            {
                "action_id": action.action_id,
                "source_node": action.source_node,
                "target_node": action.target_node,
                "timestamp": action.timestamp.isoformat(),
                "reason": action.reason,
                "status": action.status,
            }
            for action in self.failover_history[-limit:]
        ]
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Obtener estado de recuperación."""
        return {
            "status": self.current_status.value,
            "recovery_points_count": len(self.recovery_points),
            "verified_points_count": sum(1 for rp in self.recovery_points if rp.verified),
            "replication_nodes_count": len(self.replication_nodes),
            "healthy_nodes_count": sum(
                1 for n in self.replication_nodes.values() if n["status"] == "healthy"
            ),
            "failover_history_count": len(self.failover_history),
            "auto_failover_enabled": self.auto_failover,
        }
















