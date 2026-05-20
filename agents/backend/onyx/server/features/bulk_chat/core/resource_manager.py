"""
Resource Manager - Gestor de Recursos
=====================================

Sistema avanzado de gestión de recursos con límites, cuotas y asignación inteligente.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Tipo de recurso."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    SESSIONS = "sessions"
    REQUESTS = "requests"


@dataclass
class ResourceQuota:
    """Cuota de recurso."""
    resource_type: ResourceType
    limit: float
    unit: str = ""
    used: float = 0.0
    reserved: float = 0.0
    available: float = 0.0


@dataclass
class ResourceAllocation:
    """Asignación de recurso."""
    allocation_id: str
    resource_type: ResourceType
    amount: float
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class ResourceManager:
    """Gestor de recursos."""
    
    def __init__(self):
        self.quotas: Dict[ResourceType, ResourceQuota] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.usage_history: Dict[ResourceType, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def set_quota(
        self,
        resource_type: ResourceType,
        limit: float,
        unit: str = "",
    ):
        """Establecer cuota de recurso."""
        quota = ResourceQuota(
            resource_type=resource_type,
            limit=limit,
            unit=unit,
            available=limit,
        )
        
        self.quotas[resource_type] = quota
        
        logger.info(f"Set quota for {resource_type.value}: {limit} {unit}")
    
    async def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Asignar recurso.
        
        Args:
            resource_type: Tipo de recurso
            amount: Cantidad a asignar
            user_id: ID de usuario (opcional)
            expires_at: Fecha de expiración (opcional)
        
        Returns:
            ID de asignación o None si no hay recursos disponibles
        """
        quota = self.quotas.get(resource_type)
        if not quota:
            logger.warning(f"No quota set for {resource_type.value}")
            return None
        
        async with self._lock:
            available = quota.limit - quota.used - quota.reserved
            
            if amount > available:
                logger.warning(
                    f"Insufficient {resource_type.value}: "
                    f"requested {amount}, available {available}"
                )
                return None
            
            allocation_id = f"alloc_{resource_type.value}_{datetime.now().timestamp()}"
            
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=resource_type,
                amount=amount,
                user_id=user_id,
                expires_at=expires_at,
            )
            
            self.allocations[allocation_id] = allocation
            quota.used += amount
            quota.available = quota.limit - quota.used - quota.reserved
            
            # Registrar en historial
            self.usage_history[resource_type].append({
                "timestamp": datetime.now(),
                "action": "allocate",
                "amount": amount,
                "user_id": user_id,
                "allocation_id": allocation_id,
            })
            
            logger.info(
                f"Allocated {amount} {quota.unit} of {resource_type.value} "
                f"to {user_id or 'system'}"
            )
        
        return allocation_id
    
    async def release_resource(self, allocation_id: str) -> bool:
        """Liberar recurso asignado."""
        allocation = self.allocations.get(allocation_id)
        if not allocation:
            return False
        
        quota = self.quotas.get(allocation.resource_type)
        if not quota:
            return False
        
        async with self._lock:
            quota.used -= allocation.amount
            quota.available = quota.limit - quota.used - quota.reserved
            
            del self.allocations[allocation_id]
            
            # Registrar en historial
            self.usage_history[allocation.resource_type].append({
                "timestamp": datetime.now(),
                "action": "release",
                "amount": allocation.amount,
                "user_id": allocation.user_id,
                "allocation_id": allocation_id,
            })
            
            logger.info(f"Released allocation {allocation_id}")
        
        return True
    
    async def reserve_resource(
        self,
        resource_type: ResourceType,
        amount: float,
    ) -> bool:
        """Reservar recurso (sin asignar aún)."""
        quota = self.quotas.get(resource_type)
        if not quota:
            return False
        
        async with self._lock:
            available = quota.limit - quota.used - quota.reserved
            
            if amount > available:
                return False
            
            quota.reserved += amount
            quota.available = quota.limit - quota.used - quota.reserved
        
        return True
    
    async def release_reservation(
        self,
        resource_type: ResourceType,
        amount: float,
    ) -> bool:
        """Liberar reserva."""
        quota = self.quotas.get(resource_type)
        if not quota:
            return False
        
        async with self._lock:
            if quota.reserved < amount:
                return False
            
            quota.reserved -= amount
            quota.available = quota.limit - quota.used - quota.reserved
        
        return True
    
    def get_quota(self, resource_type: ResourceType) -> Optional[Dict[str, Any]]:
        """Obtener información de cuota."""
        quota = self.quotas.get(resource_type)
        if not quota:
            return None
        
        return {
            "resource_type": resource_type.value,
            "limit": quota.limit,
            "used": quota.used,
            "reserved": quota.reserved,
            "available": quota.available,
            "usage_percent": (quota.used / quota.limit * 100) if quota.limit > 0 else 0.0,
            "unit": quota.unit,
        }
    
    def get_all_quotas(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las cuotas."""
        return {
            rt.value: self.get_quota(rt)
            for rt in self.quotas.keys()
        }
    
    def get_usage_history(
        self,
        resource_type: Optional[ResourceType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de uso."""
        if resource_type:
            history = self.usage_history.get(resource_type, [])
        else:
            # Combinar todos los tipos
            history = []
            for rt_history in self.usage_history.values():
                history.extend(rt_history)
            history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return [
            {
                "timestamp": h["timestamp"].isoformat(),
                "action": h["action"],
                "amount": h["amount"],
                "user_id": h.get("user_id"),
                "allocation_id": h.get("allocation_id"),
                "resource_type": resource_type.value if resource_type else None,
            }
            for h in history[-limit:]
        ]
    
    async def cleanup_expired_allocations(self):
        """Limpiar asignaciones expiradas."""
        now = datetime.now()
        expired = [
            alloc_id
            for alloc_id, alloc in self.allocations.items()
            if alloc.expires_at and alloc.expires_at < now
        ]
        
        for alloc_id in expired:
            await self.release_resource(alloc_id)
        
        return len(expired)
















