"""
Sistema de Cuotas y Límites
============================
Gestión de cuotas y límites por usuario
"""

from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .models import ValidationStatus

logger = structlog.get_logger()


class QuotaType(str, Enum):
    """Tipos de cuota"""
    VALIDATIONS_PER_DAY = "validations_per_day"
    VALIDATIONS_PER_MONTH = "validations_per_month"
    EXPORTS_PER_DAY = "exports_per_day"
    CONNECTIONS_MAX = "connections_max"
    PLATFORMS_PER_VALIDATION = "platforms_per_validation"
    DATA_RETENTION_DAYS = "data_retention_days"


class QuotaLimit:
    """Límite de cuota"""
    
    def __init__(
        self,
        quota_type: QuotaType,
        limit: int,
        period_days: Optional[int] = None
    ):
        self.quota_type = quota_type
        self.limit = limit
        self.period_days = period_days
        self.reset_at: Optional[datetime] = None


class QuotaManager:
    """Gestor de cuotas"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._quotas: Dict[UUID, Dict[QuotaType, QuotaLimit]] = defaultdict(dict)
        self._usage: Dict[UUID, Dict[QuotaType, List[datetime]]] = defaultdict(lambda: defaultdict(list))
        
        # Cuotas por defecto
        self._default_quotas = {
            QuotaType.VALIDATIONS_PER_DAY: QuotaLimit(QuotaType.VALIDATIONS_PER_DAY, 5, 1),
            QuotaType.VALIDATIONS_PER_MONTH: QuotaLimit(QuotaType.VALIDATIONS_PER_MONTH, 50, 30),
            QuotaType.EXPORTS_PER_DAY: QuotaLimit(QuotaType.EXPORTS_PER_DAY, 20, 1),
            QuotaType.CONNECTIONS_MAX: QuotaLimit(QuotaType.CONNECTIONS_MAX, 5, None),
            QuotaType.PLATFORMS_PER_VALIDATION: QuotaLimit(QuotaType.PLATFORMS_PER_VALIDATION, 3, None),
            QuotaType.DATA_RETENTION_DAYS: QuotaLimit(QuotaType.DATA_RETENTION_DAYS, 90, None),
        }
        
        logger.info("QuotaManager initialized")
    
    def set_quota(
        self,
        user_id: UUID,
        quota_type: QuotaType,
        limit: int,
        period_days: Optional[int] = None
    ) -> None:
        """
        Establecer cuota para usuario
        
        Args:
            user_id: ID del usuario
            quota_type: Tipo de cuota
            limit: Límite
            period_days: Período en días (opcional)
        """
        quota = QuotaLimit(quota_type, limit, period_days)
        self._quotas[user_id][quota_type] = quota
        logger.info(
            "Quota set",
            user_id=str(user_id),
            quota_type=quota_type.value,
            limit=limit
        )
    
    def get_quota(
        self,
        user_id: UUID,
        quota_type: QuotaType
    ) -> QuotaLimit:
        """
        Obtener cuota del usuario
        
        Args:
            user_id: ID del usuario
            quota_type: Tipo de cuota
            
        Returns:
            Límite de cuota
        """
        if quota_type in self._quotas[user_id]:
            return self._quotas[user_id][quota_type]
        
        return self._default_quotas.get(quota_type, QuotaLimit(quota_type, 0))
    
    def check_quota(
        self,
        user_id: UUID,
        quota_type: QuotaType
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Verificar si usuario puede realizar acción
        
        Args:
            user_id: ID del usuario
            quota_type: Tipo de cuota
            
        Returns:
            (allowed, info)
        """
        quota = self.get_quota(user_id, quota_type)
        current_time = datetime.utcnow()
        
        # Limpiar uso antiguo
        if quota.period_days:
            cutoff = current_time - timedelta(days=quota.period_days)
            self._usage[user_id][quota_type] = [
                dt for dt in self._usage[user_id][quota_type]
                if dt > cutoff
            ]
        
        usage_count = len(self._usage[user_id][quota_type])
        
        if usage_count >= quota.limit:
            return False, {
                "allowed": False,
                "limit": quota.limit,
                "used": usage_count,
                "remaining": 0,
                "quota_type": quota_type.value
            }
        
        return True, {
            "allowed": True,
            "limit": quota.limit,
            "used": usage_count,
            "remaining": quota.limit - usage_count,
            "quota_type": quota_type.value
        }
    
    def record_usage(
        self,
        user_id: UUID,
        quota_type: QuotaType
    ) -> None:
        """
        Registrar uso de cuota
        
        Args:
            user_id: ID del usuario
            quota_type: Tipo de cuota
        """
        self._usage[user_id][quota_type].append(datetime.utcnow())
        logger.debug(
            "Quota usage recorded",
            user_id=str(user_id),
            quota_type=quota_type.value
        )
    
    def get_usage_info(
        self,
        user_id: UUID,
        quota_type: QuotaType
    ) -> Dict[str, Any]:
        """
        Obtener información de uso
        
        Args:
            user_id: ID del usuario
            quota_type: Tipo de cuota
            
        Returns:
            Información de uso
        """
        quota = self.get_quota(user_id, quota_type)
        current_time = datetime.utcnow()
        
        # Limpiar uso antiguo
        if quota.period_days:
            cutoff = current_time - timedelta(days=quota.period_days)
            self._usage[user_id][quota_type] = [
                dt for dt in self._usage[user_id][quota_type]
                if dt > cutoff
            ]
        
        usage_count = len(self._usage[user_id][quota_type])
        
        return {
            "quota_type": quota_type.value,
            "limit": quota.limit,
            "used": usage_count,
            "remaining": max(0, quota.limit - usage_count),
            "usage_percentage": (usage_count / quota.limit * 100) if quota.limit > 0 else 0.0
        }
    
    def get_all_quotas(self, user_id: UUID) -> Dict[str, Any]:
        """
        Obtener todas las cuotas del usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Información de todas las cuotas
        """
        quotas_info = {}
        
        for quota_type in QuotaType:
            quotas_info[quota_type.value] = self.get_usage_info(user_id, quota_type)
        
        return {
            "user_id": str(user_id),
            "quotas": quotas_info
        }


# Instancia global del gestor de cuotas
quota_manager = QuotaManager()




