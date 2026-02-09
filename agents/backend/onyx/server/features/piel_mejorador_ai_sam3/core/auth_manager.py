"""
Authentication Manager for Piel Mejorador AI SAM3
================================================

Basic authentication system for API.
"""

import hashlib
import secrets
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt

logger = None  # Will be set when logging is available


@dataclass
class APIKey:
    """API key data structure."""
    key_id: str
    key_hash: str
    name: str
    permissions: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    enabled: bool = True


class AuthManager:
    """
    Manages API authentication.
    
    Features:
    - API key management
    - JWT tokens
    - Permission-based access
    - Key rotation
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize auth manager.
        
        Args:
            secret_key: Secret key for JWT (defaults to random)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self._api_keys: Dict[str, APIKey] = {}
        self._key_lookup: Dict[str, str] = {}  # hash -> key_id
    
    def create_api_key(
        self,
        name: str,
        permissions: List[str],
        expires_days: Optional[int] = None
    ) -> tuple[str, str]:
        """
        Create a new API key.
        
        Args:
            name: Key name/description
            permissions: List of permissions
            expires_days: Optional expiration in days
            
        Returns:
            Tuple of (key_id, plain_key)
        """
        # Generate key
        plain_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()
        key_id = secrets.token_urlsafe(16)
        
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self._api_keys[key_id] = api_key
        self._key_lookup[key_hash] = key_id
        
        return key_id, plain_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            api_key: Plain API key
            
        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_id = self._key_lookup.get(key_hash)
        
        if not key_id:
            return None
        
        api_key_obj = self._api_keys.get(key_id)
        if not api_key_obj or not api_key_obj.enabled:
            return None
        
        # Check expiration
        if api_key_obj.expires_at and datetime.now() > api_key_obj.expires_at:
            return None
        
        # Update last used
        api_key_obj.last_used = datetime.now()
        
        return api_key_obj
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if revoked
        """
        if key_id in self._api_keys:
            self._api_keys[key_id].enabled = False
            return True
        return False
    
    def generate_jwt(self, key_id: str, expires_minutes: int = 60) -> str:
        """
        Generate JWT token.
        
        Args:
            key_id: API key ID
            expires_minutes: Token expiration
            
        Returns:
            JWT token
        """
        api_key = self._api_keys.get(key_id)
        if not api_key:
            raise ValueError(f"API key not found: {key_id}")
        
        payload = {
            "key_id": key_id,
            "permissions": api_key.permissions,
            "exp": int(time.time()) + (expires_minutes * 60),
            "iat": int(time.time())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def validate_jwt(self, token: str) -> Optional[Dict]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            key_id = payload.get("key_id")
            
            # Verify key is still valid
            api_key = self._api_keys.get(key_id)
            if not api_key or not api_key.enabled:
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def list_api_keys(self) -> List[Dict]:
        """List all API keys (without hashes)."""
        return [
            {
                "key_id": k.key_id,
                "name": k.name,
                "permissions": k.permissions,
                "created_at": k.created_at.isoformat(),
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "enabled": k.enabled,
            }
            for k in self._api_keys.values()
        ]




