"""
Authentication Manager for Color Grading AI
===========================================

Manages API authentication and authorization.
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import jwt
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key data structure."""
    key_id: str
    key_hash: str
    name: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True


class AuthManager:
    """
    Manages authentication and authorization.
    
    Features:
    - API key generation and validation
    - JWT token support
    - Permission-based access control
    - Usage tracking
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize auth manager.
        
        Args:
            secret_key: Secret key for JWT (auto-generated if None)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self._api_keys: Dict[str, APIKey] = {}
        self._key_lookup: Dict[str, str] = {}  # key_hash -> key_id
    
    def generate_api_key(
        self,
        name: str,
        permissions: List[str],
        expires_days: Optional[int] = None
    ) -> str:
        """
        Generate a new API key.
        
        Args:
            name: Key name/description
            permissions: List of permissions
            expires_days: Optional expiration in days
            
        Returns:
            API key string (only shown once)
        """
        import uuid
        key_id = str(uuid.uuid4())
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self._api_keys[key_id] = api_key_obj
        self._key_lookup[key_hash] = key_id
        
        logger.info(f"Generated API key: {name} ({key_id})")
        return api_key  # Return plain key (only time it's shown)
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_id = self._key_lookup.get(key_hash)
        
        if not key_id:
            return None
        
        api_key_obj = self._api_keys.get(key_id)
        if not api_key_obj or not api_key_obj.is_active:
            return None
        
        # Check expiration
        if api_key_obj.expires_at and datetime.now() > api_key_obj.expires_at:
            return None
        
        # Update usage
        api_key_obj.last_used = datetime.now()
        api_key_obj.usage_count += 1
        
        return api_key_obj
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if revoked
        """
        api_key_obj = self._api_keys.get(key_id)
        if api_key_obj:
            api_key_obj.is_active = False
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without actual keys)."""
        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "permissions": key.permissions,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "usage_count": key.usage_count,
                "is_active": key.is_active,
            }
            for key in self._api_keys.values()
        ]
    
    def check_permission(self, api_key_obj: APIKey, permission: str) -> bool:
        """
        Check if API key has permission.
        
        Args:
            api_key_obj: API key object
            permission: Permission to check
            
        Returns:
            True if has permission
        """
        return permission in api_key_obj.permissions or "admin" in api_key_obj.permissions
    
    def generate_jwt_token(
        self,
        user_id: str,
        permissions: List[str],
        expires_hours: int = 24
    ) -> str:
        """
        Generate JWT token.
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            expires_hours: Expiration in hours
            
        Returns:
            JWT token string
        """
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
            "iat": datetime.utcnow(),
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None




