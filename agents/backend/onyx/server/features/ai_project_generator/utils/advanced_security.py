"""
Advanced Security - Seguridad Avanzada
======================================

Sistema avanzado de seguridad y protección.
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AdvancedSecurity:
    """Sistema avanzado de seguridad"""

    def __init__(self):
        """Inicializa el sistema de seguridad"""
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Dict[str, datetime] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.max_failed_attempts = 5
        self.block_duration_minutes = 15

    def generate_api_key(
        self,
        user_id: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None,
    ) -> str:
        """
        Genera una API key.

        Args:
            user_id: ID del usuario
            permissions: Permisos
            expires_in_days: Días hasta expiración

        Returns:
            API key generada
        """
        api_key = secrets.token_urlsafe(32)
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "last_used": None,
        }

        logger.info(f"API key generada para usuario {user_id}")
        return api_key

    def validate_api_key(
        self,
        api_key: str,
        required_permission: Optional[str] = None,
    ) -> bool:
        """
        Valida una API key.

        Args:
            api_key: API key a validar
            required_permission: Permiso requerido

        Returns:
            Si es válida
        """
        if api_key not in self.api_keys:
            return False

        key_info = self.api_keys[api_key]

        # Verificar expiración
        if key_info["expires_at"]:
            expires_at = datetime.fromisoformat(key_info["expires_at"])
            if datetime.now() > expires_at:
                del self.api_keys[api_key]
                return False

        # Verificar permiso
        if required_permission:
            if required_permission not in key_info["permissions"]:
                return False

        # Actualizar último uso
        key_info["last_used"] = datetime.now().isoformat()
        return True

    def record_failed_attempt(
        self,
        identifier: str,
    ):
        """Registra un intento fallido"""
        now = datetime.now()
        self.failed_attempts[identifier].append(now)

        # Limpiar intentos antiguos
        cutoff = now - timedelta(minutes=self.block_duration_minutes)
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]

        # Bloquear si excede límite
        if len(self.failed_attempts[identifier]) >= self.max_failed_attempts:
            self.blocked_ips[identifier] = now
            logger.warning(f"IP {identifier} bloqueada por múltiples intentos fallidos")

    def is_blocked(
        self,
        identifier: str,
    ) -> bool:
        """Verifica si un identificador está bloqueado"""
        if identifier not in self.blocked_ips:
            return False

        block_time = self.blocked_ips[identifier]
        if datetime.now() - block_time > timedelta(minutes=self.block_duration_minutes):
            del self.blocked_ips[identifier]
            return False

        return True

    def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 100,
        window_minutes: int = 60,
    ) -> bool:
        """
        Verifica rate limit.

        Args:
            identifier: Identificador
            max_requests: Máximo de requests
            window_minutes: Ventana en minutos

        Returns:
            Si está dentro del límite
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)

        # Limpiar requests antiguos
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if req_time > cutoff
        ]

        # Verificar límite
        if len(self.rate_limits[identifier]) >= max_requests:
            return False

        # Registrar request
        self.rate_limits[identifier].append(now)
        return True

    def hash_sensitive_data(
        self,
        data: str,
    ) -> str:
        """Hashea datos sensibles"""
        return hashlib.sha256(data.encode()).hexdigest()

    def get_security_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de seguridad"""
        return {
            "blocked_ips": len(self.blocked_ips),
            "active_api_keys": len(self.api_keys),
            "total_failed_attempts": sum(len(attempts) for attempts in self.failed_attempts.values()),
        }


