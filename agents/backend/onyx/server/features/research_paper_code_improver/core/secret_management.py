"""
Secret Management - Gestión segura de secretos
===============================================
"""

import logging
import base64
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography no disponible, usando encriptación básica")

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Tipos de secretos"""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    DATABASE_CREDENTIAL = "database_credential"
    CUSTOM = "custom"


@dataclass
class Secret:
    """Secreto individual"""
    id: str
    name: str
    secret_type: SecretType
    encrypted_value: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Verifica si el secreto expiró"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class SecretManager:
    """Gestor de secretos"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.secrets: Dict[str, Secret] = {}
        self.master_key = master_key or self._generate_master_key()
        self.cipher = self._create_cipher()
    
    def _generate_master_key(self) -> str:
        """Genera una master key"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _create_cipher(self):
        """Crea el cipher para encriptación"""
        if not CRYPTO_AVAILABLE:
            return None
        
        try:
            # Derivar key de master_key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'secret_salt',  # En producción, usar salt único
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            return Fernet(key)
        except Exception as e:
            logger.error(f"Error creando cipher: {e}")
            return None
    
    def _encrypt(self, value: str) -> bytes:
        """Encripta un valor"""
        if self.cipher:
            return self.cipher.encrypt(value.encode())
        else:
            # Fallback básico (no seguro para producción)
            return base64.b64encode(value.encode())
    
    def _decrypt(self, encrypted_value: bytes) -> str:
        """Desencripta un valor"""
        if self.cipher:
            return self.cipher.decrypt(encrypted_value).decode()
        else:
            # Fallback básico
            return base64.b64decode(encrypted_value).decode()
    
    def store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.CUSTOM,
        expires_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Secret:
        """Almacena un secreto"""
        import uuid
        
        secret_id = str(uuid.uuid4())
        encrypted_value = self._encrypt(value)
        
        secret = Secret(
            id=secret_id,
            name=name,
            secret_type=secret_type,
            encrypted_value=encrypted_value,
            metadata=metadata or {},
            expires_at=expires_at,
            tags=tags or []
        )
        
        self.secrets[secret_id] = secret
        logger.info(f"Secreto {name} almacenado")
        return secret
    
    def get_secret(self, secret_id: str) -> Optional[str]:
        """Obtiene un secreto (desencriptado)"""
        if secret_id not in self.secrets:
            return None
        
        secret = self.secrets[secret_id]
        
        if secret.is_expired():
            logger.warning(f"Secreto {secret_id} expirado")
            return None
        
        try:
            return self._decrypt(secret.encrypted_value)
        except Exception as e:
            logger.error(f"Error desencriptando secreto {secret_id}: {e}")
            return None
    
    def get_secret_metadata(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene metadata de un secreto (sin el valor)"""
        if secret_id not in self.secrets:
            return None
        
        secret = self.secrets[secret_id]
        return {
            "id": secret.id,
            "name": secret.name,
            "type": secret.secret_type.value,
            "created_at": secret.created_at.isoformat(),
            "updated_at": secret.updated_at.isoformat(),
            "expires_at": secret.expires_at.isoformat() if secret.expires_at else None,
            "is_expired": secret.is_expired(),
            "tags": secret.tags,
            "metadata": secret.metadata
        }
    
    def update_secret(
        self,
        secret_id: str,
        value: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Actualiza un secreto"""
        if secret_id not in self.secrets:
            return False
        
        secret = self.secrets[secret_id]
        
        if value is not None:
            secret.encrypted_value = self._encrypt(value)
            secret.updated_at = datetime.now()
        
        if metadata is not None:
            secret.metadata.update(metadata)
        
        if expires_at is not None:
            secret.expires_at = expires_at
        
        return True
    
    def delete_secret(self, secret_id: str) -> bool:
        """Elimina un secreto"""
        if secret_id not in self.secrets:
            return False
        
        del self.secrets[secret_id]
        logger.info(f"Secreto {secret_id} eliminado")
        return True
    
    def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
        tags: Optional[List[str]] = None,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Lista secretos"""
        secrets = list(self.secrets.values())
        
        if secret_type:
            secrets = [s for s in secrets if s.secret_type == secret_type]
        
        if tags:
            secrets = [
                s for s in secrets
                if any(tag in s.tags for tag in tags)
            ]
        
        if not include_expired:
            secrets = [s for s in secrets if not s.is_expired()]
        
        return [self.get_secret_metadata(s.id) for s in secrets]
    
    def rotate_secret(self, secret_id: str, new_value: str) -> bool:
        """Rota un secreto (actualiza con nuevo valor)"""
        return self.update_secret(secret_id, value=new_value)
    
    def search_secrets(self, query: str) -> List[Dict[str, Any]]:
        """Busca secretos por nombre o tags"""
        results = []
        query_lower = query.lower()
        
        for secret in self.secrets.values():
            if query_lower in secret.name.lower():
                results.append(self.get_secret_metadata(secret.id))
            elif any(query_lower in tag.lower() for tag in secret.tags):
                results.append(self.get_secret_metadata(secret.id))
        
        return results




