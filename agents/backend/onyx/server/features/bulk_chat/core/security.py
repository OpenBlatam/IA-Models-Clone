"""
Security - Seguridad Avanzada
==============================

Sistema de seguridad con encryption, audit logs y más.
"""

import asyncio
import logging
import hashlib
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Niveles de seguridad."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditLog:
    """Log de auditoría."""
    log_id: str
    user_id: Optional[str]
    action: str
    resource: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """Gestor de seguridad avanzado."""
    
    def __init__(self):
        self.audit_logs: List[AuditLog] = []
        self.max_logs: int = 10000
        self.encryption_key: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def log_action(
        self,
        user_id: Optional[str],
        action: str,
        resource: str,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Registrar acción en audit log."""
        log_id = hashlib.md5(
            f"{user_id}:{action}:{resource}:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        audit_log = AuditLog(
            log_id=log_id,
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
        )
        
        async with self._lock:
            self.audit_logs.append(audit_log)
            if len(self.audit_logs) > self.max_logs:
                self.audit_logs.pop(0)
        
        logger.info(
            f"Audit log: {action} on {resource} by {user_id} - "
            f"{'SUCCESS' if success else 'FAILED'}"
        )
    
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Obtener logs de auditoría."""
        logs = self.audit_logs
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if action:
            logs = [l for l in logs if l.action == action]
        
        if resource:
            logs = [l for l in logs if l.resource == resource]
        
        return logs[-limit:]
    
    def sanitize_input(self, input_str: str) -> str:
        """Sanitizar input del usuario."""
        # Remover caracteres peligrosos
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '(', ')', '{', '}']
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limitar longitud
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def validate_session_access(
        self,
        user_id: str,
        session_id: str,
        session_user_id: Optional[str],
    ) -> bool:
        """Validar acceso a sesión."""
        # Si la sesión no tiene usuario, es pública
        if session_user_id is None:
            return True
        
        # Verificar que el usuario coincida
        return user_id == session_user_id
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generar token seguro."""
        import secrets
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str) -> str:
        """Hash de contraseña (SHA256)."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verificar contraseña."""
        return self.hash_password(password) == password_hash
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de seguridad."""
        total_logs = len(self.audit_logs)
        failed_actions = sum(1 for log in self.audit_logs if not log.success)
        
        # Acciones más comunes
        action_counts = {}
        for log in self.audit_logs:
            action_counts[log.action] = action_counts.get(log.action, 0) + 1
        
        return {
            "total_audit_logs": total_logs,
            "failed_actions": failed_actions,
            "success_rate": (total_logs - failed_actions) / total_logs if total_logs > 0 else 0.0,
            "top_actions": dict(sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
































