"""
Refresh Token Management
"""

from typing import Optional
from datetime import datetime, timedelta
import logging
import jwt

from .base import User, Token, AuthProvider

logger = logging.getLogger(__name__)


class RefreshTokenManager:
    """Manages refresh tokens for authentication"""
    
    def __init__(
        self,
        secret_key: str,
        access_token_ttl: timedelta = timedelta(hours=1),
        refresh_token_ttl: timedelta = timedelta(days=30),
        redis_client=None
    ):
        self.secret_key = secret_key
        self.access_token_ttl = access_token_ttl
        self.refresh_token_ttl = refresh_token_ttl
        self.redis_client = redis_client
        self._refresh_tokens: dict = {}
    
    async def generate_token_pair(self, user: User) -> Token:
        """Generate access and refresh token pair"""
        now = datetime.utcnow()
        
        # Access token
        access_payload = {
            "user_id": user.id,
            "email": user.email,
            "type": "access",
            "iat": now,
            "exp": now + self.access_token_ttl
        }
        access_token = jwt.encode(access_payload, self.secret_key, algorithm="HS256")
        
        # Refresh token
        refresh_payload = {
            "user_id": user.id,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_token_ttl,
            "jti": f"{user.id}_{now.timestamp()}"  # JWT ID for revocation
        }
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm="HS256")
        
        # Store refresh token
        token = Token(
            user_id=user.id,
            token_type="bearer"
        )
        token.access_token = access_token
        token.refresh_token = refresh_token
        token.expires_at = now + self.access_token_ttl
        
        # Store in Redis or memory
        if self.redis_client:
            await self.redis_client.setex(
                f"refresh_token:{refresh_payload['jti']}",
                int(self.refresh_token_ttl.total_seconds()),
                user.id
            )
        else:
            self._refresh_tokens[refresh_payload['jti']] = {
                "user_id": user.id,
                "expires_at": now + self.refresh_token_ttl
            }
        
        return token
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[Token]:
        """Generate new access token from refresh token"""
        try:
            # Decode refresh token
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=["HS256"])
            
            if payload.get("type") != "refresh":
                logger.warning("Invalid token type for refresh")
                return None
            
            jti = payload.get("jti")
            user_id = payload.get("user_id")
            
            # Verify token exists and is valid
            if self.redis_client:
                stored_user_id = await self.redis_client.get(f"refresh_token:{jti}")
                if not stored_user_id or stored_user_id.decode() != user_id:
                    logger.warning("Refresh token not found or invalid")
                    return None
            else:
                if jti not in self._refresh_tokens:
                    logger.warning("Refresh token not found")
                    return None
                
                token_data = self._refresh_tokens[jti]
                if token_data["expires_at"] < datetime.utcnow():
                    del self._refresh_tokens[jti]
                    logger.warning("Refresh token expired")
                    return None
            
            # Generate new access token
            now = datetime.utcnow()
            access_payload = {
                "user_id": user_id,
                "type": "access",
                "iat": now,
                "exp": now + self.access_token_ttl
            }
            access_token = jwt.encode(access_payload, self.secret_key, algorithm="HS256")
            
            token = Token(user_id=user_id, token_type="bearer")
            token.access_token = access_token
            token.expires_at = now + self.access_token_ttl
            
            return token
            
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid refresh token: {e}")
            return None
    
    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke a refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=["HS256"])
            jti = payload.get("jti")
            
            if self.redis_client:
                await self.redis_client.delete(f"refresh_token:{jti}")
            else:
                if jti in self._refresh_tokens:
                    del self._refresh_tokens[jti]
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking refresh token: {e}")
            return False
    
    async def revoke_all_user_tokens(self, user_id: str) -> bool:
        """Revoke all refresh tokens for a user"""
        try:
            # TODO: Implement efficient revocation of all user tokens
            # This would require tracking tokens by user_id
            return True
            
        except Exception as e:
            logger.error(f"Error revoking all user tokens: {e}")
            return False

