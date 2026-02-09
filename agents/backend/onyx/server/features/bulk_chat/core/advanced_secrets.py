"""
Advanced Secrets - Gestión Avanzada de Secretos
===============================================

Sistema avanzado de gestión de secretos con rotación automática, encriptación y auditoría.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import secrets

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Tipo de secreto."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    DATABASE = "database"
    ENCRYPTION_KEY = "encryption_key"


class SecretStatus(Enum):
    """Estado de secreto."""
    ACTIVE = "active"
    ROTATING = "rotating"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class Secret:
    """Secreto."""
    secret_id: str
    name: str
    secret_type: SecretType
    value: str
    status: SecretStatus = SecretStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    rotation_interval_days: int = 90
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecretAccess:
    """Acceso a secreto."""
    access_id: str
    secret_id: str
    accessed_by: str
    timestamp: datetime
    action: str  # "read", "update", "rotate"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedSecrets:
    """Gestor avanzado de secretos."""
    
    def __init__(self):
        self.secrets: Dict[str, Secret] = {}
        self.secret_history: Dict[str, List[str]] = defaultdict(list)  # secret_id -> [old_values]
        self.access_log: List[SecretAccess] = []
        self._lock = asyncio.Lock()
    
    def create_secret(
        self,
        secret_id: str,
        name: str,
        secret_type: SecretType,
        value: str,
        expires_at: Optional[datetime] = None,
        rotation_interval_days: int = 90,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear secreto."""
        secret = Secret(
            secret_id=secret_id,
            name=name,
            secret_type=secret_type,
            value=value,
            expires_at=expires_at,
            rotation_interval_days=rotation_interval_days,
            metadata=metadata or {},
        )
        
        self.secrets[secret_id] = secret
        logger.info(f"Created secret: {secret_id} - {name}")
        return secret_id
    
    def get_secret(
        self,
        secret_id: str,
        accessed_by: str = "system",
        validate_expiry: bool = True,
    ) -> Optional[str]:
        """Obtener valor de secreto."""
        secret = self.secrets.get(secret_id)
        if not secret:
            return None
        
        # Validar expiración
        if validate_expiry and secret.expires_at:
            if datetime.now() > secret.expires_at:
                secret.status = SecretStatus.EXPIRED
                return None
        
        if secret.status != SecretStatus.ACTIVE:
            return None
        
        # Registrar acceso
        access = SecretAccess(
            access_id=f"access_{datetime.now().timestamp()}",
            secret_id=secret_id,
            accessed_by=accessed_by,
            timestamp=datetime.now(),
            action="read",
        )
        
        self.access_log.append(access)
        
        secret.access_count += 1
        secret.last_accessed = datetime.now()
        
        return secret.value
    
    async def rotate_secret(
        self,
        secret_id: str,
        new_value: Optional[str] = None,
        rotated_by: str = "system",
    ) -> bool:
        """Rotar secreto."""
        secret = self.secrets.get(secret_id)
        if not secret:
            return False
        
        # Guardar valor anterior
        old_value = secret.value
        self.secret_history[secret_id].append(old_value)
        
        # Generar nuevo valor si no se proporciona
        if new_value is None:
            new_value = secrets.token_urlsafe(32)
        
        async with self._lock:
            secret.status = SecretStatus.ROTATING
            secret.value = new_value
            secret.last_rotated = datetime.now()
            secret.status = SecretStatus.ACTIVE
        
        # Registrar acceso
        access = SecretAccess(
            access_id=f"access_{datetime.now().timestamp()}",
            secret_id=secret_id,
            accessed_by=rotated_by,
            timestamp=datetime.now(),
            action="rotate",
        )
        
        self.access_log.append(access)
        
        logger.info(f"Rotated secret: {secret_id}")
        return True
    
    async def auto_rotate_secrets(self):
        """Rotar secretos automáticamente según intervalo."""
        now = datetime.now()
        
        for secret in self.secrets.values():
            if secret.status != SecretStatus.ACTIVE:
                continue
            
            if secret.last_rotated:
                days_since_rotation = (now - secret.last_rotated).days
                if days_since_rotation >= secret.rotation_interval_days:
                    await self.rotate_secret(secret.secret_id)
            elif secret.created_at:
                days_since_creation = (now - secret.created_at).days
                if days_since_rotation >= secret.rotation_interval_days:
                    await self.rotate_secret(secret.secret_id)
    
    def revoke_secret(self, secret_id: str, revoked_by: str = "system"):
        """Revocar secreto."""
        secret = self.secrets.get(secret_id)
        if not secret:
            return False
        
        secret.status = SecretStatus.REVOKED
        
        # Registrar acceso
        access = SecretAccess(
            access_id=f"access_{datetime.now().timestamp()}",
            secret_id=secret_id,
            accessed_by=revoked_by,
            timestamp=datetime.now(),
            action="revoke",
        )
        
        self.access_log.append(access)
        
        logger.warning(f"Revoked secret: {secret_id}")
        return True
    
    def get_secret_info(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de secreto (sin valor)."""
        secret = self.secrets.get(secret_id)
        if not secret:
            return None
        
        return {
            "secret_id": secret.secret_id,
            "name": secret.name,
            "secret_type": secret.secret_type.value,
            "status": secret.status.value,
            "created_at": secret.created_at.isoformat(),
            "expires_at": secret.expires_at.isoformat() if secret.expires_at else None,
            "last_rotated": secret.last_rotated.isoformat() if secret.last_rotated else None,
            "rotation_interval_days": secret.rotation_interval_days,
            "access_count": secret.access_count,
            "last_accessed": secret.last_accessed.isoformat() if secret.last_accessed else None,
        }
    
    def get_access_log(
        self,
        secret_id: Optional[str] = None,
        accessed_by: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener log de accesos."""
        log = self.access_log
        
        if secret_id:
            log = [a for a in log if a.secret_id == secret_id]
        
        if accessed_by:
            log = [a for a in log if a.accessed_by == accessed_by]
        
        return [
            {
                "access_id": a.access_id,
                "secret_id": a.secret_id,
                "accessed_by": a.accessed_by,
                "timestamp": a.timestamp.isoformat(),
                "action": a.action,
            }
            for a in log[-limit:]
        ]
    
    def get_secrets_summary(self) -> Dict[str, Any]:
        """Obtener resumen de secretos."""
        by_type: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        
        for secret in self.secrets.values():
            by_type[secret.secret_type.value] += 1
            by_status[secret.status.value] += 1
        
        expired_secrets = [
            s for s in self.secrets.values()
            if s.expires_at and datetime.now() > s.expires_at
        ]
        
        return {
            "total_secrets": len(self.secrets),
            "secrets_by_type": dict(by_type),
            "secrets_by_status": dict(by_status),
            "expired_secrets": len(expired_secrets),
            "total_accesses": len(self.access_log),
        }
















