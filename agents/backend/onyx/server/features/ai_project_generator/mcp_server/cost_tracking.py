"""
MCP Cost Tracking - Tracking de costos
========================================
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostEntry(BaseModel):
    """Entrada de costo"""
    entry_id: str = Field(..., description="ID único de la entrada")
    resource_id: str = Field(..., description="ID del recurso")
    operation: str = Field(..., description="Operación")
    cost: float = Field(..., description="Costo")
    currency: str = Field(default="USD", description="Moneda")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    tenant_id: Optional[str] = Field(None, description="ID del tenant")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """
    Tracker de costos
    
    Rastrea costos de operaciones por usuario/tenant/recurso.
    """
    
    def __init__(self):
        self._costs: List[CostEntry] = []
        self._cost_rates: Dict[str, float] = {}  # resource_id:operation -> cost
    
    def set_cost_rate(
        self,
        resource_id: str,
        operation: str,
        cost: float,
    ):
        """
        Configura tasa de costo para operación
        
        Args:
            resource_id: ID del recurso
            operation: Operación
            cost: Costo por operación
        """
        key = f"{resource_id}:{operation}"
        self._cost_rates[key] = cost
        logger.info(f"Set cost rate for {key}: {cost}")
    
    def record_cost(
        self,
        resource_id: str,
        operation: str,
        cost: Optional[float] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Registra un costo
        
        Args:
            resource_id: ID del recurso
            operation: Operación
            cost: Costo (usa tasa si None)
            user_id: ID del usuario (opcional)
            tenant_id: ID del tenant (opcional)
            metadata: Metadata adicional (opcional)
            
        Returns:
            ID de la entrada de costo
        """
        import uuid
        
        # Obtener costo
        if cost is None:
            key = f"{resource_id}:{operation}"
            cost = self._cost_rates.get(key, 0.0)
        
        entry = CostEntry(
            entry_id=str(uuid.uuid4()),
            resource_id=resource_id,
            operation=operation,
            cost=cost,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )
        
        self._costs.append(entry)
        
        # Mantener solo últimos 10000 entradas
        if len(self._costs) > 10000:
            self._costs = self._costs[-10000:]
        
        logger.debug(f"Recorded cost: {cost} for {resource_id}:{operation}")
        
        return entry.entry_id
    
    def get_total_cost(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> float:
        """
        Obtiene costo total
        
        Args:
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            user_id: Filtrar por usuario (opcional)
            tenant_id: Filtrar por tenant (opcional)
            resource_id: Filtrar por recurso (opcional)
            
        Returns:
            Costo total
        """
        costs = self._costs
        
        if start_time:
            costs = [c for c in costs if c.timestamp >= start_time]
        if end_time:
            costs = [c for c in costs if c.timestamp <= end_time]
        if user_id:
            costs = [c for c in costs if c.user_id == user_id]
        if tenant_id:
            costs = [c for c in costs if c.tenant_id == tenant_id]
        if resource_id:
            costs = [c for c in costs if c.resource_id == resource_id]
        
        return sum(c.cost for c in costs)
    
    def get_cost_breakdown(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene desglose de costos
        
        Args:
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            
        Returns:
            Diccionario con desglose de costos
        """
        costs = self._costs
        
        if start_time:
            costs = [c for c in costs if c.timestamp >= start_time]
        if end_time:
            costs = [c for c in costs if c.timestamp <= end_time]
        
        # Por recurso
        by_resource = defaultdict(float)
        for cost in costs:
            by_resource[cost.resource_id] += cost.cost
        
        # Por operación
        by_operation = defaultdict(float)
        for cost in costs:
            by_operation[cost.operation] += cost.cost
        
        # Por usuario
        by_user = defaultdict(float)
        for cost in costs:
            if cost.user_id:
                by_user[cost.user_id] += cost.cost
        
        return {
            "total": sum(c.cost for c in costs),
            "by_resource": dict(by_resource),
            "by_operation": dict(by_operation),
            "by_user": dict(by_user),
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        }

