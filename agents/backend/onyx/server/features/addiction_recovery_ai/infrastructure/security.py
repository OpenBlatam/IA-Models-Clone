"""
Security Service Implementations
Provides implementations for authentication and authorization
"""

import logging
from typing import Dict, Any, Optional

from core.interfaces import IAuthenticationService, IServiceFactory
from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)


class JWTAuthenticationService(IAuthenticationService):
    """JWT-based authentication service"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self._secret_key: Optional[str] = None
        self.algorithm = "HS256"
    
    def _get_secret_key(self) -> str:
        """Get JWT secret key"""
        if self._secret_key:
            return self._secret_key
        
        try:
            # Try Secrets Manager
            if self.settings.secrets_manager_secret_name:
                from aws.aws_services import SecretsManagerService
                secrets = SecretsManagerService()
                secret_data = secrets.get_secret()
                self._secret_key = secret_data.get("JWT_SECRET_KEY")
            
            # Fallback to environment
            if not self._secret_key:
                import os
                self._secret_key = os.getenv("JWT_SECRET_KEY")
            
            if not self._secret_key:
                raise ValueError("JWT secret key not found")
            
            return self._secret_key
        except Exception as e:
            logger.error(f"Error getting JWT secret: {str(e)}")
            raise
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user and return token"""
        # Implementation for authentication
        # This would typically validate username/password and return JWT
        import jwt
        from datetime import datetime, timedelta
        
        user_id = credentials.get("user_id")
        if not user_id:
            raise ValueError("User ID required")
        
        payload = {
            "sub": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=60)
        }
        
        token = jwt.encode(payload, self._get_secret_key(), algorithm=self.algorithm)
        
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        import jwt
        
        try:
            payload = jwt.decode(
                token,
                self._get_secret_key(),
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    async def refresh_token(self, token: str) -> str:
        """Refresh JWT token"""
        # Validate current token
        payload = await self.validate_token(token)
        
        # Create new token with extended expiration
        import jwt
        from datetime import datetime, timedelta
        
        new_payload = {
            **payload,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=60)
        }
        
        return jwt.encode(new_payload, self._get_secret_key(), algorithm=self.algorithm)


class SecurityServiceFactory(IServiceFactory):
    """Factory for creating security services"""
    
    @staticmethod
    def create_authentication_service(backend: str = "jwt") -> IAuthenticationService:
        """Create authentication service"""
        if backend == "jwt":
            return JWTAuthenticationService()
        else:
            raise ValueError(f"Unsupported auth backend: {backend}")
    
    def create_authentication_service(self) -> IAuthenticationService:
        """Create authentication service (factory method)"""
        return self.create_authentication_service()
    
    def create_storage_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError
    
    def create_cache_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError
    
    def create_file_storage_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError
    
    def create_message_queue_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError
    
    def create_notification_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError
    
    def create_metrics_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError
    
    def create_tracing_service(self):
        """Not implemented in security factory"""
        raise NotImplementedError















