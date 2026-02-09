"""
Sistema de alertas avanzado
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, Integer
from sqlalchemy.orm import Session

from ..db.base import Base, get_db_session

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Tipos de alertas"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SECURITY = "security"


class AlertSeverity(str, Enum):
    """Severidad de alertas"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertModel(Base):
    """Modelo de alerta en BD"""
    __tablename__ = "alerts"
    
    id = Column(String(64), primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSON, nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)


@dataclass
class Alert:
    """Alerta"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class AlertService:
    """Servicio de alertas"""
    
    def __init__(self):
        self._init_table()
    
    def _init_table(self):
        """Inicializa tabla de alertas"""
        from ..db.base import init_db
        init_db()
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crea una alerta
        
        Args:
            alert_type: Tipo de alerta
            severity: Severidad
            title: Título
            message: Mensaje
            data: Datos adicionales
            
        Returns:
            ID de la alerta
        """
        alert_id = str(uuid.uuid4())
        
        with get_db_session() as db:
            alert = AlertModel(
                id=alert_id,
                alert_type=alert_type.value,
                severity=severity.value,
                title=title,
                message=message,
                data=data,
                acknowledged=False
            )
            db.add(alert)
            db.commit()
        
        logger.warning(f"Alerta creada: {title} ({severity.value})")
        return alert_id
    
    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        acknowledged_only: bool = False,
        unacknowledged_only: bool = False,
        limit: int = 50
    ) -> List[Alert]:
        """Obtiene alertas con filtros"""
        with get_db_session() as db:
            query = db.query(AlertModel)
            
            if alert_type:
                query = query.filter_by(alert_type=alert_type.value)
            
            if severity:
                query = query.filter_by(severity=severity.value)
            
            if acknowledged_only:
                query = query.filter_by(acknowledged=True)
            
            if unacknowledged_only:
                query = query.filter_by(acknowledged=False)
            
            results = query.order_by(
                AlertModel.created_at.desc()
            ).limit(limit).all()
            
            return [
                Alert(
                    alert_id=a.id,
                    alert_type=AlertType(a.alert_type),
                    severity=AlertSeverity(a.severity),
                    title=a.title,
                    message=a.message,
                    data=a.data,
                    acknowledged=a.acknowledged,
                    created_at=a.created_at
                )
                for a in results
            ]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Marca alerta como reconocida"""
        with get_db_session() as db:
            alert = db.query(AlertModel).filter_by(id=alert_id).first()
            if not alert:
                return False
            
            alert.acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            db.commit()
            
            return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Marca alerta como resuelta"""
        with get_db_session() as db:
            alert = db.query(AlertModel).filter_by(id=alert_id).first()
            if not alert:
                return False
            
            alert.resolved_at = datetime.utcnow()
            db.commit()
            
            return True
    
    def get_critical_alerts_count(self) -> int:
        """Obtiene conteo de alertas críticas no reconocidas"""
        with get_db_session() as db:
            return db.query(AlertModel).filter_by(
                severity=AlertSeverity.CRITICAL.value,
                acknowledged=False
            ).count()


# Singleton global
_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """Obtiene instancia singleton del servicio de alertas"""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service




