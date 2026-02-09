"""
Routing Compliance and Audit Optimizations
===========================================

Optimizaciones para compliance y auditoría.
Incluye: Audit logging, Compliance checks, Data governance, etc.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Evento de auditoría."""
    event_type: str
    user: str
    action: str
    resource: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    ip_address: Optional[str] = None


class AuditLogger:
    """Logger de auditoría."""
    
    def __init__(self, max_events: int = 100000):
        """
        Inicializar logger de auditoría.
        
        Args:
            max_events: Máximo de eventos a mantener
        """
        self.events: deque = deque(maxlen=max_events)
        self.lock = threading.Lock()
    
    def log_event(
        self,
        event_type: str,
        user: str,
        action: str,
        resource: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """
        Registrar evento de auditoría.
        
        Args:
            event_type: Tipo de evento
            user: Usuario
            action: Acción realizada
            resource: Recurso afectado
            success: Si la acción fue exitosa
            details: Detalles adicionales
            ip_address: Dirección IP
        """
        event = AuditEvent(
            event_type=event_type,
            user=user,
            action=action,
            resource=resource,
            success=success,
            details=details or {},
            ip_address=ip_address
        )
        
        with self.lock:
            self.events.append(event)
    
    def get_events(
        self,
        user: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[AuditEvent]:
        """Obtener eventos filtrados."""
        with self.lock:
            events = list(self.events)
        
        if user:
            events = [e for e in events if e.user == user]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def export_events(self, format: str = "json") -> str:
        """Exportar eventos."""
        with self.lock:
            events = [{
                'event_type': e.event_type,
                'user': e.user,
                'action': e.action,
                'resource': e.resource,
                'timestamp': e.timestamp,
                'success': e.success,
                'details': e.details,
                'ip_address': e.ip_address
            } for e in self.events]
        
        if format == "json":
            return json.dumps(events, indent=2)
        else:
            # Implementar otros formatos si es necesario
            return str(events)


class ComplianceChecker:
    """Verificador de compliance."""
    
    def __init__(self):
        """Inicializar verificador."""
        self.compliance_rules: List[Any] = []
        self.violations: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def add_rule(self, rule_name: str, rule_func: Any):
        """
        Agregar regla de compliance.
        
        Args:
            rule_name: Nombre de la regla
            rule_func: Función que verifica compliance
        """
        self.compliance_rules.append((rule_name, rule_func))
    
    def check_compliance(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verificar compliance.
        
        Args:
            data: Datos a verificar
        
        Returns:
            Lista de violaciones
        """
        violations = []
        
        for rule_name, rule_func in self.compliance_rules:
            try:
                is_compliant, message = rule_func(data)
                if not is_compliant:
                    violation = {
                        'rule': rule_name,
                        'message': message,
                        'timestamp': time.time(),
                        'data': data
                    }
                    violations.append(violation)
                    
                    with self.lock:
                        self.violations.append(violation)
            except Exception as e:
                logger.error(f"Error checking compliance rule {rule_name}: {e}")
        
        return violations
    
    def get_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener violaciones recientes."""
        with self.lock:
            return self.violations[-limit:]


class DataGovernance:
    """Gestor de gobernanza de datos."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.data_classifications: Dict[str, str] = {}
        self.retention_policies: Dict[str, float] = {}
        self.access_controls: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
    
    def classify_data(self, data_id: str, classification: str):
        """
        Clasificar datos.
        
        Args:
            data_id: ID de los datos
            classification: Clasificación (public, internal, confidential, restricted)
        """
        with self.lock:
            self.data_classifications[data_id] = classification
    
    def set_retention_policy(self, data_type: str, retention_days: float):
        """
        Establecer política de retención.
        
        Args:
            data_type: Tipo de datos
            retention_days: Días de retención
        """
        with self.lock:
            self.retention_policies[data_type] = retention_days
    
    def check_access(self, user: str, resource: str) -> bool:
        """
        Verificar acceso a recurso.
        
        Args:
            user: Usuario
            resource: Recurso
        
        Returns:
            True si tiene acceso
        """
        with self.lock:
            if resource in self.access_controls:
                return user in self.access_controls[resource]
            return True  # Por defecto, permitir acceso


class ComplianceOptimizer:
    """Optimizador completo de compliance."""
    
    def __init__(self):
        """Inicializar optimizador de compliance."""
        self.audit_logger = AuditLogger()
        self.compliance_checker = ComplianceChecker()
        self.data_governance = DataGovernance()
    
    def audit_action(
        self,
        user: str,
        action: str,
        resource: str,
        success: bool = True,
        **kwargs
    ):
        """Registrar acción para auditoría."""
        self.audit_logger.log_event(
            event_type="user_action",
            user=user,
            action=action,
            resource=resource,
            success=success,
            details=kwargs
        )
    
    def check_compliance(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar compliance."""
        return self.compliance_checker.check_compliance(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'total_audit_events': len(self.audit_logger.events),
            'total_violations': len(self.compliance_checker.violations),
            'data_classifications': len(self.data_governance.data_classifications),
            'retention_policies': len(self.data_governance.retention_policies)
        }

