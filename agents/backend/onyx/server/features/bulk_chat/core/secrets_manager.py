"""
Secrets Manager - Gestión de Secretos y Configuración
====================================================

Sistema de gestión segura de secretos y configuración.
"""

import asyncio
import logging
import os
import hashlib
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class Secret:
    """Secreto."""
    secret_id: str
    name: str
    value: str
    secret_type: str = "api_key"  # "api_key", "password", "token", "certificate"
    encrypted: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecretsManager:
    """Gestor de secretos."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.secrets: Dict[str, Secret] = {}
        self.encryption_key = encryption_key or os.getenv("SECRETS_ENCRYPTION_KEY", "")
        self._lock = asyncio.Lock()
        self._load_from_env()
    
    def _load_from_env(self):
        """Cargar secretos desde variables de entorno."""
        # Cargar secretos comunes
        common_secrets = [
            ("OPENAI_API_KEY", "OpenAI API Key"),
            ("ANTHROPIC_API_KEY", "Anthropic API Key"),
            ("REDIS_URL", "Redis URL"),
            ("DATABASE_URL", "Database URL"),
        ]
        
        for env_key, name in common_secrets:
            value = os.getenv(env_key)
            if value:
                secret_id = hashlib.md5(f"{env_key}_secret".encode()).hexdigest()
                self.secrets[secret_id] = Secret(
                    secret_id=secret_id,
                    name=name,
                    value=value,
                    secret_type="api_key",
                )
                logger.debug(f"Loaded secret from env: {name}")
    
    async def store_secret(
        self,
        secret_id: str,
        name: str,
        value: str,
        secret_type: str = "api_key",
        encrypted: bool = False,
    ) -> Secret:
        """
        Almacenar secreto.
        
        Args:
            secret_id: ID único del secreto
            name: Nombre del secreto
            value: Valor del secreto
            secret_type: Tipo de secreto
            encrypted: Si está encriptado
        
        Returns:
            Secreto almacenado
        """
        secret = Secret(
            secret_id=secret_id,
            name=name,
            value=self._encrypt(value) if encrypted else value,
            secret_type=secret_type,
            encrypted=encrypted,
        )
        
        async with self._lock:
            self.secrets[secret_id] = secret
        
        logger.info(f"Stored secret: {name}")
        return secret
    
    def _encrypt(self, value: str) -> str:
        """Encriptar valor (simplificado)."""
        # En producción, usar una librería de encriptación real
        if self.encryption_key:
            # Hash simple para demostración
            return hashlib.sha256(f"{self.encryption_key}{value}".encode()).hexdigest()
        return value
    
    def _decrypt(self, encrypted_value: str) -> str:
        """Desencriptar valor (simplificado)."""
        # En producción, usar una librería de encriptación real
        # Esta implementación es solo para demostración
        return encrypted_value
    
    async def get_secret(self, secret_id: str, decrypt: bool = True) -> Optional[str]:
        """
        Obtener secreto.
        
        Args:
            secret_id: ID del secreto
            decrypt: Si debe desencriptar
        
        Returns:
            Valor del secreto
        """
        secret = self.secrets.get(secret_id)
        if not secret:
            return None
        
        if decrypt and secret.encrypted:
            return self._decrypt(secret.value)
        
        return secret.value
    
    async def get_secret_info(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de secreto (sin el valor)."""
        secret = self.secrets.get(secret_id)
        if not secret:
            return None
        
        return {
            "secret_id": secret.secret_id,
            "name": secret.name,
            "secret_type": secret.secret_type,
            "encrypted": secret.encrypted,
            "created_at": secret.created_at.isoformat(),
            "updated_at": secret.updated_at.isoformat(),
        }
    
    async def list_secrets(self) -> List[Dict[str, Any]]:
        """Listar secretos (sin valores)."""
        return [
            {
                "secret_id": s.secret_id,
                "name": s.name,
                "secret_type": s.secret_type,
                "encrypted": s.encrypted,
            }
            for s in self.secrets.values()
        ]
    
    async def delete_secret(self, secret_id: str):
        """Eliminar secreto."""
        async with self._lock:
            if secret_id in self.secrets:
                del self.secrets[secret_id]
                logger.info(f"Deleted secret: {secret_id}")
    
    def get_config_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Obtener valor de configuración."""
        # Primero buscar en secretos
        for secret in self.secrets.values():
            if secret.name == key:
                return secret.value
        
        # Luego buscar en variables de entorno
        return os.getenv(key, default)
































