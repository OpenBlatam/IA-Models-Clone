"""JWT authentication and authorization"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
import secrets
import logging

logger = logging.getLogger(__name__)


class JWTAuth:
    """JWT authentication manager"""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire: int = 3600,
        refresh_token_expire: int = 86400
    ):
        """
        Initialize JWT auth
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm
            access_token_expire: Access token expiration in seconds
            refresh_token_expire: Refresh token expiration in seconds
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire = access_token_expire
        self.refresh_token_expire = refresh_token_expire
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        roles: Optional[list] = None
    ) -> str:
        """
        Create access token
        
        Args:
            user_id: User ID
            username: Username
            roles: Optional list of roles
            
        Returns:
            JWT token
        """
        expire = datetime.utcnow() + timedelta(seconds=self.access_token_expire)
        
        payload = {
            "sub": user_id,
            "username": username,
            "roles": roles or [],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create refresh token
        
        Args:
            user_id: User ID
            
        Returns:
            JWT refresh token
        """
        expire = datetime.utcnow() + timedelta(seconds=self.refresh_token_expire)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode token
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload or None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh access token
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New access token or None
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Create new access token
        return self.create_access_token(user_id, payload.get("username", ""), payload.get("roles", []))


# Global JWT auth
_jwt_auth: Optional[JWTAuth] = None


def get_jwt_auth() -> JWTAuth:
    """Get global JWT auth"""
    global _jwt_auth
    if _jwt_auth is None:
        import os
        secret_key = os.getenv("JWT_SECRET_KEY")
        _jwt_auth = JWTAuth(secret_key=secret_key)
    return _jwt_auth

