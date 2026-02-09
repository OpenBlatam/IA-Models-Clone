"""
Basic authentication system for API access.
"""

import hashlib
import secrets
import time
from typing import Optional, Dict
from datetime import timedelta
import logging

from ..utils.file_helpers import get_iso_timestamp, parse_iso_date

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Simple authentication manager with API keys.
    """
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        self.token_secret = secrets.token_urlsafe(32)
    
    def create_api_key(self, user_id: str, permissions: Optional[list] = None) -> str:
        """
        Create a new API key for a user.
        
        Args:
            user_id: User identifier
            permissions: List of permissions
        
        Returns:
            API key string
        """
        api_key = f"rmai_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions or ["read", "write"],
            "created_at": get_iso_timestamp(),
            "last_used": None
        }
        logger.info(f"API key created for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
        
        Returns:
            User info if valid, None otherwise
        """
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            key_info["last_used"] = get_iso_timestamp()
            return key_info
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
        
        Returns:
            True if revoked, False if not found
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"API key revoked: {api_key[:10]}...")
            return True
        return False
    
    def create_session_token(self, user_id: str, expires_in: int = 3600) -> str:
        """
        Create a session token.
        
        Args:
            user_id: User identifier
            expires_in: Expiration time in seconds
        
        Returns:
            Session token
        """
        token = secrets.token_urlsafe(32)
        from ..utils.file_helpers import datetime_to_iso
        self.sessions[token] = {
            "user_id": user_id,
            "created_at": get_iso_timestamp(),
            "expires_at": datetime_to_iso(datetime.now() + timedelta(seconds=expires_in))
        }
        return token
    
    def validate_session_token(self, token: str) -> Optional[Dict]:
        """
        Validate a session token.
        
        Args:
            token: Session token to validate
        
        Returns:
            Session info if valid, None otherwise
        """
        if token in self.sessions:
            session = self.sessions[token]
            expires_at = parse_iso_date(session.get("expires_at"))
            if not expires_at:
                return None
            if datetime.now() < expires_at:
                return session
            else:
                del self.sessions[token]
        return None
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password (basic implementation).
        In production, use bcrypt or argon2.
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()






