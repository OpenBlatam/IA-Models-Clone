"""
Compliance Manager - Sistema de compliance y auditoría
=======================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ComplianceManager:
    """
    Gestiona compliance, auditoría y regulaciones.
    """
    
    def __init__(self, compliance_dir: str = "data/compliance"):
        """
        Inicializar gestor de compliance.
        
        Args:
            compliance_dir: Directorio para datos de compliance
        """
        self.compliance_dir = Path(compliance_dir)
        self.compliance_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_log: List[Dict[str, Any]] = []
        self.compliance_rules: Dict[str, List[Dict[str, Any]]] = {}
    
    def log_audit_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource_type: str,
        resource_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Registra un evento de auditoría.
        
        Args:
            event_type: Tipo de evento
            user_id: ID del usuario (opcional)
            resource_type: Tipo de recurso
            resource_id: ID del recurso
            action: Acción realizada
            details: Detalles adicionales
        """
        audit_event = {
            "event_id": f"audit_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "event_type": event_type,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "ip_address": None,  # Se llenaría en producción
            "user_agent": None   # Se llenaría en producción
        }
        
        self.audit_log.append(audit_event)
        
        # Mantener solo últimos 10000 eventos
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        # Guardar en disco
        self._save_audit_event(audit_event)
        
        logger.info(f"Evento de auditoría registrado: {event_type} - {action}")
    
    def check_compliance(
        self,
        operation: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verifica compliance de una operación.
        
        Args:
            operation: Operación a verificar
            data: Datos de la operación
            
        Returns:
            Resultado de verificación de compliance
        """
        compliance_result = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        # Verificar reglas de compliance
        rules = self.compliance_rules.get(operation, [])
        
        for rule in rules:
            rule_type = rule.get("type")
            rule_check = rule.get("check")
            
            if rule_type == "data_retention":
                # Verificar retención de datos
                if "created_at" in data:
                    retention_days = rule.get("days", 365)
                    created = datetime.fromisoformat(data["created_at"])
                    age_days = (datetime.now() - created).days
                    
                    if age_days > retention_days:
                        compliance_result["violations"].append(
                            f"Datos exceden período de retención: {age_days} días"
                        )
                        compliance_result["compliant"] = False
            
            elif rule_type == "data_privacy":
                # Verificar privacidad de datos
                sensitive_fields = rule.get("sensitive_fields", [])
                for field in sensitive_fields:
                    if field in data and data[field]:
                        compliance_result["warnings"].append(
                            f"Campo sensible detectado: {field}"
                        )
            
            elif rule_type == "access_control":
                # Verificar control de acceso
                required_permissions = rule.get("required_permissions", [])
                user_permissions = data.get("user_permissions", [])
                
                missing = set(required_permissions) - set(user_permissions)
                if missing:
                    compliance_result["violations"].append(
                        f"Permisos faltantes: {missing}"
                    )
                    compliance_result["compliant"] = False
        
        return compliance_result
    
    def generate_audit_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera reporte de auditoría.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            user_id: Filtrar por usuario (opcional)
            
        Returns:
            Reporte de auditoría
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Filtrar eventos
        filtered_events = [
            e for e in self.audit_log
            if start_date <= datetime.fromisoformat(e["timestamp"]) <= end_date
        ]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.get("user_id") == user_id]
        
        # Agrupar por tipo
        events_by_type = {}
        for event in filtered_events:
            event_type = event["event_type"]
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        return {
            "report_type": "audit",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "user_id": user_id,
            "total_events": len(filtered_events),
            "events_by_type": {
                event_type: len(events)
                for event_type, events in events_by_type.items()
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def add_compliance_rule(
        self,
        operation: str,
        rule: Dict[str, Any]
    ):
        """
        Agrega una regla de compliance.
        
        Args:
            operation: Operación
            rule: Regla de compliance
        """
        if operation not in self.compliance_rules:
            self.compliance_rules[operation] = []
        
        self.compliance_rules[operation].append(rule)
        logger.info(f"Regla de compliance agregada: {operation}")
    
    def _save_audit_event(self, event: Dict[str, Any]):
        """Guarda evento de auditoría en disco"""
        try:
            audit_file = self.compliance_dir / "audit_log.jsonl"
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Error guardando evento de auditoría: {e}")

