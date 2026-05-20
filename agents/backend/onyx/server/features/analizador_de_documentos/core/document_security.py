"""
Document Security - Sistema de Seguridad Avanzado
==================================================

Sistema de seguridad para análisis de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import hmac
import secrets

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Política de seguridad."""
    policy_id: str
    name: str
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_formats: List[str] = field(default_factory=lambda: ["pdf", "docx", "txt"])
    scan_for_malware: bool = True
    encrypt_content: bool = False
    require_authentication: bool = False
    rate_limit: Optional[int] = None  # requests per minute
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAuditLog:
    """Registro de auditoría de seguridad."""
    log_id: str
    event_type: str
    document_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # 'info', 'warning', 'error', 'critical'


class SecurityManager:
    """Gestor de seguridad."""
    
    def __init__(self, analyzer):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.policies: Dict[str, SecurityPolicy] = {}
        self.audit_logs: List[SecurityAuditLog] = []
        self.secret_key = secrets.token_hex(32)
        self.max_logs = 10000
    
    def register_policy(self, policy: SecurityPolicy):
        """Registrar política de seguridad."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Política de seguridad registrada: {policy.name}")
    
    async def validate_document(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        policy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validar documento según políticas de seguridad.
        
        Args:
            file_path: Ruta al archivo
            content: Contenido del documento
            policy_id: ID de política (opcional)
        
        Returns:
            Diccionario con resultado de validación
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_performed": []
        }
        
        # Obtener política
        policy = None
        if policy_id:
            policy = self.policies.get(policy_id)
        
        # Si no hay política específica, usar default
        if not policy:
            policy = SecurityPolicy(
                policy_id="default",
                name="Default Policy"
            )
        
        # Verificar tamaño
        if file_path:
            import os
            file_size = os.path.getsize(file_path)
            result["checks_performed"].append("file_size_check")
            
            if file_size > policy.max_file_size:
                result["valid"] = False
                result["errors"].append(
                    f"Archivo excede tamaño máximo: {policy.max_file_size} bytes"
                )
            else:
                result["warnings"].append(f"Tamaño de archivo: {file_size} bytes")
        
        # Verificar formato
        if file_path and hasattr(self.analyzer, 'format_handler'):
            format_type = self.analyzer.format_handler.detect_format(file_path)
            result["checks_performed"].append("format_check")
            
            if format_type and format_type not in policy.allowed_formats:
                result["valid"] = False
                result["errors"].append(
                    f"Formato no permitido: {format_type}. Permitidos: {policy.allowed_formats}"
                )
        
        # Verificar contenido (básico)
        if content:
            result["checks_performed"].append("content_check")
            
            # Verificar contenido sospechoso (simplificado)
            suspicious_patterns = ["<script>", "javascript:", "eval("]
            for pattern in suspicious_patterns:
                if pattern.lower() in content.lower():
                    result["warnings"].append(f"Patrón sospechoso detectado: {pattern}")
        
        # Registrar auditoría
        self.log_security_event(
            "document_validation",
            document_id=None,
            details=result
        )
        
        return result
    
    def generate_secure_hash(self, content: str) -> str:
        """Generar hash seguro del contenido."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_content_integrity(
        self,
        content: str,
        expected_hash: str
    ) -> bool:
        """Verificar integridad del contenido."""
        actual_hash = self.generate_secure_hash(content)
        return hmac.compare_digest(actual_hash, expected_hash)
    
    def log_security_event(
        self,
        event_type: str,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ):
        """Registrar evento de seguridad."""
        log = SecurityAuditLog(
            log_id=f"log_{len(self.audit_logs) + 1}",
            event_type=event_type,
            document_id=document_id,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            severity=severity
        )
        
        self.audit_logs.append(log)
        
        # Mantener solo últimos N logs
        if len(self.audit_logs) > self.max_logs:
            self.audit_logs = self.audit_logs[-self.max_logs:]
        
        logger.info(f"Evento de seguridad registrado: {event_type}")
    
    def get_audit_logs(
        self,
        event_type: Optional[str] = None,
        document_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityAuditLog]:
        """Obtener logs de auditoría."""
        logs = self.audit_logs
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if document_id:
            logs = [l for l in logs if l.document_id == document_id]
        
        if severity:
            logs = [l for l in logs if l.severity == severity]
        
        return logs[-limit:]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de seguridad."""
        total_logs = len(self.audit_logs)
        by_severity = {
            severity: len([l for l in self.audit_logs if l.severity == severity])
            for severity in ["info", "warning", "error", "critical"]
        }
        
        return {
            "total_audit_logs": total_logs,
            "by_severity": by_severity,
            "total_policies": len(self.policies),
            "active_policies": len([p for _id, p in self.policies.items()])
        }


__all__ = [
    "SecurityManager",
    "SecurityPolicy",
    "SecurityAuditLog"
]
















