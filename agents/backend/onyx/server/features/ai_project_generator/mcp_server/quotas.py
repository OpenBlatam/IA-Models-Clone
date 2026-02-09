"""
MCP Resource Quotas - Cuotas de recursos
==========================================
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class QuotaLimit(BaseModel):
    """Límite de cuota"""
    resource_type: str = Field(..., description="Tipo de recurso")
    limit: int = Field(..., description="Límite")
    period: str = Field(default="day", description="Período: day, week, month")
    reset_at: Optional[datetime] = None


class ResourceQuota(BaseModel):
    """Cuota de recursos"""
    quota_id: str = Field(..., description="ID único de la cuota")
    entity_id: str = Field(..., description="ID de la entidad (usuario/tenant)")
    entity_type: str = Field(default="user", description="Tipo: user, tenant")
    limits: Dict[str, QuotaLimit] = Field(default_factory=dict)
    usage: Dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuotaManager:
    """
    Gestor de cuotas de recursos
    
    Controla el uso de recursos por usuario/tenant.
    """
    
    def __init__(self):
        self._quotas: Dict[str, ResourceQuota] = {}
    
    def set_quota(
        self,
        entity_id: str,
        resource_type: str,
        limit: int,
        period: str = "day",
        entity_type: str = "user",
    ):
        """
        Configura cuota para una entidad
        
        Args:
            entity_id: ID de la entidad
            resource_type: Tipo de recurso
            limit: Límite
            period: Período (day, week, month)
            entity_type: Tipo de entidad
        """
        quota_id = f"{entity_type}:{entity_id}"
        
        if quota_id not in self._quotas:
            self._quotas[quota_id] = ResourceQuota(
                quota_id=quota_id,
                entity_id=entity_id,
                entity_type=entity_type,
            )
        
        quota = self._quotas[quota_id]
        
        # Calcular reset_at
        reset_at = self._calculate_reset_at(period)
        
        quota.limits[resource_type] = QuotaLimit(
            resource_type=resource_type,
            limit=limit,
            period=period,
            reset_at=reset_at,
        )
        
        logger.info(f"Set quota for {entity_id}: {resource_type} = {limit}/{period}")
    
    def check_quota(
        self,
        entity_id: str,
        resource_type: str,
        amount: int = 1,
        entity_type: str = "user",
    ) -> tuple[bool, Optional[str]]:
        """
        Verifica si hay cuota disponible
        
        Args:
            entity_id: ID de la entidad
            resource_type: Tipo de recurso
            amount: Cantidad a verificar
            entity_type: Tipo de entidad
            
        Returns:
            Tuple (allowed, error_message)
        """
        quota_id = f"{entity_type}:{entity_id}"
        quota = self._quotas.get(quota_id)
        
        if not quota:
            # Sin cuota configurada, permitir
            return True, None
        
        limit = quota.limits.get(resource_type)
        if not limit:
            # Sin límite para este recurso, permitir
            return True, None
        
        # Verificar si necesita reset
        if limit.reset_at and datetime.utcnow() > limit.reset_at:
            self._reset_quota(quota, resource_type, limit.period)
        
        current_usage = quota.usage.get(resource_type, 0)
        
        if current_usage + amount > limit.limit:
            return False, f"Quota exceeded for {resource_type}: {current_usage}/{limit.limit}"
        
        return True, None
    
    def consume_quota(
        self,
        entity_id: str,
        resource_type: str,
        amount: int = 1,
        entity_type: str = "user",
    ) -> bool:
        """
        Consume cuota
        
        Args:
            entity_id: ID de la entidad
            resource_type: Tipo de recurso
            amount: Cantidad a consumir
            entity_type: Tipo de entidad
            
        Returns:
            True si se consumió exitosamente
        """
        allowed, error = self.check_quota(entity_id, resource_type, amount, entity_type)
        
        if not allowed:
            raise MCPError(error or "Quota exceeded")
        
        quota_id = f"{entity_type}:{entity_id}"
        quota = self._quotas.get(quota_id)
        
        if quota:
            quota.usage[resource_type] = quota.usage.get(resource_type, 0) + amount
        
        return True
    
    def _calculate_reset_at(self, period: str) -> datetime:
        """Calcula fecha de reset"""
        now = datetime.utcnow()
        
        if period == "day":
            return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            days_until_monday = (7 - now.weekday()) % 7
            return (now + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "month":
            if now.month == 12:
                return datetime(now.year + 1, 1, 1)
            return datetime(now.year, now.month + 1, 1)
        
        return now + timedelta(days=1)
    
    def _reset_quota(self, quota: ResourceQuota, resource_type: str, period: str):
        """Resetea cuota"""
        quota.usage[resource_type] = 0
        limit = quota.limits[resource_type]
        limit.reset_at = self._calculate_reset_at(period)
        logger.info(f"Reset quota for {quota.entity_id}: {resource_type}")
    
    def get_quota_status(
        self,
        entity_id: str,
        entity_type: str = "user",
    ) -> Dict[str, Any]:
        """
        Obtiene estado de cuotas
        
        Args:
            entity_id: ID de la entidad
            entity_type: Tipo de entidad
            
        Returns:
            Diccionario con estado de cuotas
        """
        quota_id = f"{entity_type}:{entity_id}"
        quota = self._quotas.get(quota_id)
        
        if not quota:
            return {"entity_id": entity_id, "quotas": {}}
        
        status = {}
        for resource_type, limit in quota.limits.items():
            usage = quota.usage.get(resource_type, 0)
            status[resource_type] = {
                "limit": limit.limit,
                "usage": usage,
                "remaining": max(0, limit.limit - usage),
                "period": limit.period,
                "reset_at": limit.reset_at.isoformat() if limit.reset_at else None,
            }
        
        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "quotas": status,
        }

