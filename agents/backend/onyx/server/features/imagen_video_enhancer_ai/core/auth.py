"""
Authentication and Authorization for Imagen Video Enhancer AI
=============================================================

Simple API key authentication system.
"""

import hashlib
import secrets
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key information."""
    key_hash: str
    name: str
    permissions: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    enabled: bool = True
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if not self.enabled:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


class AuthManager:
    """
    Manages API key authentication.
    
    Features:
    - API key generation
    - Key validation
    - Permission checking
    - Key expiration
    """
    
    def __init__(self):
        """Initialize auth manager."""
        self._keys: Dict[str, APIKey] = {}
        self._key_to_hash: Dict[str, str] = {}  # Map plain key to hash
    
    def generate_key(
        self,
        name: str,
        permissions: List[str],
        expires_days: Optional[int] = None
    ) -> str:
        """
        Generate a new API key.
        
        Args:
            name: Key name/identifier
            permissions: List of permissions
            expires_days: Optional expiration in days
            
        Returns:
            Plain API key (store this securely, it won't be shown again)
        """
        # Generate random key
        plain_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self._keys[key_hash] = api_key
        self._key_to_hash[plain_key] = key_hash
        
        logger.info(f"Generated API key: {name} with permissions: {permissions}")
        return plain_key
    
    def validate_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            api_key: Plain API key
            
        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Check if we have this key
        stored_key = self._keys.get(key_hash)
        if not stored_key:
            return None
        
        # Check if valid
        if not stored_key.is_valid():
            return None
        
        # Update last used
        stored_key.last_used = datetime.now()
        
        return stored_key
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: Plain API key
            
        Returns:
            True if revoked, False if not found
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self._keys:
            self._keys[key_hash].enabled = False
            logger.info(f"Revoked API key: {self._keys[key_hash].name}")
            return True
        
        return False
    
    def check_permission(self, api_key: str, permission: str) -> bool:
        """
        Check if API key has a permission.
        
        Args:
            api_key: Plain API key
            permission: Permission to check
            
        Returns:
            True if has permission
        """
        key_info = self.validate_key(api_key)
        if not key_info:
            return False
        
        # Check for wildcard permission
        if "*" in key_info.permissions:
            return True
        
        return permission in key_info.permissions
    
    def list_keys(self) -> List[Dict]:
        """List all API keys (without plain keys)."""
        return [
            {
                "name": key.name,
                "permissions": key.permissions,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "enabled": key.enabled
            }
            for key in self._keys.values()
        ]




