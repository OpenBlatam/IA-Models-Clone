"""
JWT Authentication Handler
"""

import jwt
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class JWTHandler:
    """Handles JWT token generation and validation"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = algorithm
        self.token_expiry = int(os.getenv("JWT_EXPIRY_SECONDS", "86400"))  # 24 hours default
    
    def generate_token(
        self,
        user_id: str,
        email: str,
        roles: list = None,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JWT token
        
        Args:
            user_id: User identifier
            email: User email
            roles: List of user roles
            additional_claims: Additional claims to include
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "email": email,
            "roles": roles or ["user"],
            "iat": now,
            "exp": now + timedelta(seconds=self.token_expiry),
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Generated JWT token for user {user_id}")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh JWT token
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token if valid, None otherwise
        """
        payload = self.validate_token(token)
        if not payload:
            return None
        
        # Generate new token with same claims
        return self.generate_token(
            user_id=payload.get("user_id"),
            email=payload.get("email"),
            roles=payload.get("roles", []),
            additional_claims={k: v for k, v in payload.items() if k not in ["iat", "exp", "user_id", "email", "roles"]}
        )


_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get JWT handler instance (singleton)"""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler

