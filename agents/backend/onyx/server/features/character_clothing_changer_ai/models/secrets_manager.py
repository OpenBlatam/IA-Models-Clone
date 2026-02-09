"""
Secrets Manager for Flux2 Clothing Changer
===========================================

Secrets and credentials management.
"""

import os
import base64
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class Secret:
    """Secret information."""
    secret_id: str
    name: str
    value: str
    encrypted: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecretsManager:
    """Secrets management system."""
    
    def __init__(
        self,
        master_key: Optional[bytes] = None,
    ):
        """
        Initialize secrets manager.
        
        Args:
            master_key: Optional master encryption key
        """
        if master_key is None:
            # Generate or load from environment
            master_key = os.getenv("SECRETS_MASTER_KEY")
            if master_key:
                master_key = master_key.encode()
            elif CRYPTOGRAPHY_AVAILABLE:
                # Generate new key (should be stored securely)
                master_key = Fernet.generate_key()
                logger.warning("Generated new master key - store it securely!")
            else:
                master_key = b"default_key_not_secure"  # Fallback
                logger.warning("Using default key - not secure! Install cryptography.")
        
        self.master_key = master_key
        self.cipher = Fernet(master_key) if CRYPTOGRAPHY_AVAILABLE else None
        self.secrets: Dict[str, Secret] = {}
    
    def store_secret(
        self,
        secret_id: str,
        name: str,
        value: str,
        encrypt: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Secret:
        """
        Store secret.
        
        Args:
            secret_id: Secret identifier
            name: Secret name
            value: Secret value
            encrypt: Whether to encrypt
            metadata: Optional metadata
            
        Returns:
            Stored secret
        """
        if encrypt and self.cipher:
            try:
                encrypted_value = self.cipher.encrypt(value.encode())
                stored_value = base64.b64encode(encrypted_value).decode()
            except Exception as e:
                logger.error(f"Encryption failed: {e}")
                stored_value = value
                encrypt = False
        else:
            stored_value = value
        
        secret = Secret(
            secret_id=secret_id,
            name=name,
            value=stored_value,
            encrypted=encrypt,
            metadata=metadata or {},
        )
        
        self.secrets[secret_id] = secret
        logger.info(f"Stored secret: {secret_id}")
        return secret
    
    def get_secret(
        self,
        secret_id: str,
    ) -> Optional[str]:
        """
        Get secret value.
        
        Args:
            secret_id: Secret identifier
            
        Returns:
            Decrypted secret value or None
        """
        if secret_id not in self.secrets:
            return None
        
        secret = self.secrets[secret_id]
        
        if secret.encrypted and self.cipher:
            try:
                encrypted_value = base64.b64decode(secret.value.encode())
                decrypted_value = self.cipher.decrypt(encrypted_value)
                return decrypted_value.decode()
            except Exception as e:
                logger.error(f"Failed to decrypt secret {secret_id}: {e}")
                return None
        else:
            return secret.value
    
    def delete_secret(self, secret_id: str) -> bool:
        """
        Delete secret.
        
        Args:
            secret_id: Secret identifier
            
        Returns:
            True if deleted
        """
        if secret_id in self.secrets:
            del self.secrets[secret_id]
            logger.info(f"Deleted secret: {secret_id}")
            return True
        return False
    
    def list_secrets(self) -> List[str]:
        """List all secret IDs."""
        return list(self.secrets.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get secrets manager statistics."""
        return {
            "total_secrets": len(self.secrets),
            "encrypted_secrets": len([s for s in self.secrets.values() if s.encrypted]),
            "unencrypted_secrets": len([s for s in self.secrets.values() if not s.encrypted]),
        }

