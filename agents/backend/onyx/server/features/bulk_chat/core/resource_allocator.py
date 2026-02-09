"""
Resource Allocator - Asignador de Recursos
==========================================

Sistema avanzado de asignación de recursos con cuotas, prioridades y balanceo automático.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Tipo de recurso."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    CUSTOM = "custom"


class AllocationStatus(Enum):
    """Estado de asignación."""
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    RELEASED = "released"
    THROTTLED = "throttled"
    DENIED = "denied"


@dataclass
class ResourceQuota:
    """Cuota de recurso."""
    quota_id: str
    resource_type: ResourceType
    max_amount: float
    current_usage: float = 0.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Asignación de recurso."""
    allocation_id: str
    resource_type: ResourceType
    amount: float
    requester_id: str
    status: AllocationStatus
    allocated_at: datetime = field(default_factory=datetime.now)
    released_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceAllocator:
    """Asignador de recursos."""
    
    def __init__(self):
        self.quotas: Dict[str, ResourceQuota] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.requester_allocations: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def create_quota(
        self,
        quota_id: str,
        resource_type: ResourceType,
        max_amount: float,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear cuota de recurso."""
        quota = ResourceQuota(
            quota_id=quota_id,
            resource_type=resource_type,
            max_amount=max_amount,
            priority=priority,
            metadata=metadata or {},
        )
        
        async def save_quota():
            async with self._lock:
                self.quotas[quota_id] = quota
        
        asyncio.create_task(save_quota())
        
        logger.info(f"Created quota: {quota_id} for {resource_type.value}")
        return quota_id
    
    async def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        requester_id: str,
        quota_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Asignar recurso."""
        allocation_id = f"alloc_{requester_id}_{datetime.now().timestamp()}"
        
        # Buscar cuota disponible
        quota = None
        if quota_id:
            quota = self.quotas.get(quota_id)
        else:
            # Buscar cuota por tipo
            quotas = [q for q in self.quotas.values() if q.resource_type == resource_type]
            if quotas:
                quota = max(quotas, key=lambda q: q.priority)
        
        async with self._lock:
            if not quota:
                # Sin cuota, asignar directamente
                status = AllocationStatus.ALLOCATED
            else:
                # Verificar disponibilidad
                available = quota.max_amount - quota.current_usage
                if available >= amount:
                    quota.current_usage += amount
                    status = AllocationStatus.ALLOCATED
                else:
                    status = AllocationStatus.DENIED
            
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=resource_type,
                amount=amount,
                requester_id=requester_id,
                status=status,
                metadata=metadata or {},
            )
            
            self.allocations[allocation_id] = allocation
            self.requester_allocations[requester_id].append(allocation_id)
        
        return allocation_id if status == AllocationStatus.ALLOCATED else None
    
    async def release_resource(self, allocation_id: str) -> bool:
        """Liberar recurso."""
        allocation = self.allocations.get(allocation_id)
        if not allocation:
            return False
        
        async with self._lock:
            if allocation.status == AllocationStatus.RELEASED:
                return False
            
            # Buscar cuota asociada
            quotas = [q for q in self.quotas.values() if q.resource_type == allocation.resource_type]
            for quota in quotas:
                if quota.current_usage >= allocation.amount:
                    quota.current_usage -= allocation.amount
                    break
            
            allocation.status = AllocationStatus.RELEASED
            allocation.released_at = datetime.now()
            
            # Remover de requester
            if allocation_id in self.requester_allocations[allocation.requester_id]:
                self.requester_allocations[allocation.requester_id].remove(allocation_id)
        
        logger.info(f"Released allocation: {allocation_id}")
        return True
    
    def get_quota_usage(self, quota_id: str) -> Optional[Dict[str, Any]]:
        """Obtener uso de cuota."""
        quota = self.quotas.get(quota_id)
        if not quota:
            return None
        
        return {
            "quota_id": quota.quota_id,
            "resource_type": quota.resource_type.value,
            "max_amount": quota.max_amount,
            "current_usage": quota.current_usage,
            "available": quota.max_amount - quota.current_usage,
            "usage_percentage": (quota.current_usage / quota.max_amount * 100) if quota.max_amount > 0 else 0.0,
        }
    
    def get_requester_allocations(self, requester_id: str) -> List[Dict[str, Any]]:
        """Obtener asignaciones de requester."""
        allocation_ids = self.requester_allocations.get(requester_id, [])
        
        return [
            {
                "allocation_id": alloc.allocation_id,
                "resource_type": alloc.resource_type.value,
                "amount": alloc.amount,
                "status": alloc.status.value,
                "allocated_at": alloc.allocated_at.isoformat(),
                "released_at": alloc.released_at.isoformat() if alloc.released_at else None,
            }
            for alloc_id in allocation_ids
            if (alloc := self.allocations.get(alloc_id))
        ]
    
    def get_resource_allocator_summary(self) -> Dict[str, Any]:
        """Obtener resumen del asignador."""
        by_type: Dict[str, float] = defaultdict(float)
        by_status: Dict[str, int] = defaultdict(int)
        
        for allocation in self.allocations.values():
            by_type[allocation.resource_type.value] += allocation.amount
            by_status[allocation.status.value] += 1
        
        return {
            "total_quotas": len(self.quotas),
            "total_allocations": len(self.allocations),
            "allocations_by_type": dict(by_type),
            "allocations_by_status": dict(by_status),
            "total_requesters": len(self.requester_allocations),
        }



