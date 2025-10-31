"""
Authentication Management
=========================

OAuth2, JWT, and authentication management.
"""

import asyncio
import logging
import secrets
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import jwt
import httpx
from cryptography.fernet import Fernet
import uuid

from .types import (
    AuthProvider, TokenType, TokenInfo, OAuth2Config, JWTConfig,
    UserSession, SecurityLevel, AuditEvent
)

logger = logging.getLogger(__name__)

class JWTManager:
    """JWT token management."""
    
    def __init__(self, config: JWTConfig):
        self.config = config
        self.blacklisted_tokens: set = set()
    
    def create_access_token(self, user_id: str, scopes: List[str] = None) -> TokenInfo:
        """Create an access token."""
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(minutes=self.config.access_token_expire_minutes)
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": expires_at,
                "type": TokenType.ACCESS.value,
                "iss": self.config.issuer,
                "aud": self.config.audience,
                "scopes": scopes or []
            }
            
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
            
            return TokenInfo(
                token=token,
                token_type=TokenType.ACCESS,
                user_id=user_id,
                expires_at=expires_at,
                scopes=scopes or []
            )
            
        except Exception as e:
            logger.error(f"Failed to create access token: {str(e)}")
            raise
    
    def create_refresh_token(self, user_id: str) -> TokenInfo:
        """Create a refresh token."""
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(days=self.config.refresh_token_expire_days)
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": expires_at,
                "type": TokenType.REFRESH.value,
                "iss": self.config.issuer,
                "aud": self.config.audience
            }
            
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
            
            return TokenInfo(
                token=token,
                token_type=TokenType.REFRESH,
                user_id=user_id,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            if token in self.blacklisted_tokens:
                return None
            
            payload = jwt.decode(
                token, 
                self.config.secret_key, 
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            return None
    
    def blacklist_token(self, token: str):
        """Add token to blacklist."""
        self.blacklisted_tokens.add(token)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[TokenInfo]:
        """Refresh an access token using a refresh token."""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get("type") != TokenType.REFRESH.value:
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            return self.create_access_token(user_id)
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {str(e)}")
            return None

class OAuth2Provider:
    """OAuth2 provider implementation."""
    
    def __init__(self, config: OAuth2Config):
        self.config = config
        self.http_client = httpx.AsyncClient()
    
    async def get_authorization_url(self, state: str = None) -> str:
        """Get OAuth2 authorization URL."""
        try:
            if not state:
                state = secrets.token_urlsafe(32)
            
            params = {
                "client_id": self.config.client_id,
                "redirect_uri": self.config.redirect_uri,
                "response_type": "code",
                "scope": " ".join(self.config.scopes),
                "state": state
            }
            
            params.update(self.config.additional_params)
            
            # Build URL with parameters
            url = f"{self.config.authorization_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to get authorization URL: {str(e)}")
            raise
    
    async def exchange_code_for_token(self, code: str, state: str = None) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            data = {
                "grant_type": "authorization_code",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "redirect_uri": self.config.redirect_uri
            }
            
            response = await self.http_client.post(
                self.config.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            return token_data
            
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {str(e)}")
            raise
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information using access token."""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await self.http_client.get(
                self.config.user_info_url,
                headers=headers
            )
            
            response.raise_for_status()
            user_info = response.json()
            
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to get user info: {str(e)}")
            raise
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token."""
        try:
            data = {
                "grant_type": "refresh_token",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "refresh_token": refresh_token
            }
            
            response = await self.http_client.post(
                self.config.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            return token_data
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {str(e)}")
            raise
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()

class AuthManager:
    """Main authentication manager."""
    
    def __init__(self, jwt_config: JWTConfig, oauth2_providers: Dict[str, OAuth2Config] = None):
        self.jwt_manager = JWTManager(jwt_config)
        self.oauth2_providers: Dict[str, OAuth2Provider] = {}
        self.user_sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self._lock = asyncio.Lock()
        
        # Initialize OAuth2 providers
        if oauth2_providers:
            for name, config in oauth2_providers.items():
                self.oauth2_providers[name] = OAuth2Provider(config)
    
    async def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Optional[TokenInfo]:
        """Authenticate user with username and password."""
        try:
            # Check for brute force attempts
            if await self._is_account_locked(username):
                logger.warning(f"Account locked for user: {username}")
                return None
            
            # Verify credentials (this would integrate with your user store)
            user_id = await self._verify_credentials(username, password)
            if not user_id:
                await self._record_failed_attempt(username, ip_address)
                return None
            
            # Clear failed attempts on successful login
            await self._clear_failed_attempts(username)
            
            # Create tokens
            access_token = self.jwt_manager.create_access_token(user_id)
            refresh_token = self.jwt_manager.create_refresh_token(user_id)
            
            # Create session
            await self._create_session(user_id, ip_address)
            
            logger.info(f"User authenticated successfully: {username}")
            return access_token
            
        except Exception as e:
            logger.error(f"Authentication failed for user {username}: {str(e)}")
            return None
    
    async def authenticate_oauth2(self, provider: str, code: str, state: str = None) -> Optional[TokenInfo]:
        """Authenticate user via OAuth2."""
        try:
            if provider not in self.oauth2_providers:
                raise ValueError(f"OAuth2 provider not found: {provider}")
            
            oauth2_provider = self.oauth2_providers[provider]
            
            # Exchange code for token
            token_data = await oauth2_provider.exchange_code_for_token(code, state)
            access_token = token_data.get("access_token")
            
            if not access_token:
                raise ValueError("No access token received")
            
            # Get user info
            user_info = await oauth2_provider.get_user_info(access_token)
            
            # Create or get user
            user_id = await self._get_or_create_oauth2_user(provider, user_info)
            
            # Create JWT tokens
            jwt_token = self.jwt_manager.create_access_token(user_id)
            
            # Create session
            await self._create_session(user_id)
            
            logger.info(f"OAuth2 authentication successful for user: {user_id}")
            return jwt_token
            
        except Exception as e:
            logger.error(f"OAuth2 authentication failed: {str(e)}")
            return None
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[TokenInfo]:
        """Refresh access token."""
        try:
            new_token = self.jwt_manager.refresh_access_token(refresh_token)
            if new_token:
                logger.info(f"Access token refreshed for user: {new_token.user_id}")
            return new_token
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {str(e)}")
            return None
    
    async def logout(self, token: str, user_id: str):
        """Logout user and invalidate tokens."""
        try:
            # Blacklist token
            self.jwt_manager.blacklist_token(token)
            
            # Remove session
            await self._remove_session(user_id)
            
            logger.info(f"User logged out: {user_id}")
            
        except Exception as e:
            logger.error(f"Logout failed for user {user_id}: {str(e)}")
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        return self.jwt_manager.verify_token(token)
    
    async def get_oauth2_authorization_url(self, provider: str, state: str = None) -> Optional[str]:
        """Get OAuth2 authorization URL."""
        try:
            if provider not in self.oauth2_providers:
                return None
            
            oauth2_provider = self.oauth2_providers[provider]
            return await oauth2_provider.get_authorization_url(state)
            
        except Exception as e:
            logger.error(f"Failed to get OAuth2 authorization URL: {str(e)}")
            return None
    
    async def _verify_credentials(self, username: str, password: str) -> Optional[str]:
        """Verify user credentials."""
        # This would integrate with your user store/database
        # For now, return a mock user ID
        if username == "admin" and password == "admin123":
            return "user_001"
        return None
    
    async def _get_or_create_oauth2_user(self, provider: str, user_info: Dict[str, Any]) -> str:
        """Get or create user from OAuth2 user info."""
        # This would integrate with your user store
        # For now, return a mock user ID
        return f"oauth2_user_{user_info.get('id', 'unknown')}"
    
    async def _create_session(self, user_id: str, ip_address: str = None):
        """Create user session."""
        try:
            session_id = str(uuid.uuid4())
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address
            )
            
            async with self._lock:
                self.user_sessions[user_id] = session
            
            logger.debug(f"Created session for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {str(e)}")
    
    async def _remove_session(self, user_id: str):
        """Remove user session."""
        try:
            async with self._lock:
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
            
            logger.debug(f"Removed session for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove session for user {user_id}: {str(e)}")
    
    async def _record_failed_attempt(self, username: str, ip_address: str = None):
        """Record failed login attempt."""
        try:
            async with self._lock:
                if username not in self.failed_attempts:
                    self.failed_attempts[username] = []
                
                self.failed_attempts[username].append(datetime.now())
                
                # Keep only last 10 attempts
                if len(self.failed_attempts[username]) > 10:
                    self.failed_attempts[username] = self.failed_attempts[username][-10:]
            
            logger.warning(f"Failed login attempt for user: {username}")
            
        except Exception as e:
            logger.error(f"Failed to record failed attempt: {str(e)}")
    
    async def _clear_failed_attempts(self, username: str):
        """Clear failed login attempts."""
        try:
            async with self._lock:
                if username in self.failed_attempts:
                    del self.failed_attempts[username]
            
        except Exception as e:
            logger.error(f"Failed to clear failed attempts: {str(e)}")
    
    async def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        try:
            if username not in self.failed_attempts:
                return False
            
            attempts = self.failed_attempts[username]
            recent_attempts = [a for a in attempts if datetime.now() - a < timedelta(minutes=15)]
            
            return len(recent_attempts) >= 5
            
        except Exception as e:
            logger.error(f"Failed to check account lock status: {str(e)}")
            return False
    
    async def get_active_sessions(self, user_id: str) -> List[UserSession]:
        """Get active sessions for user."""
        try:
            async with self._lock:
                if user_id in self.user_sessions:
                    session = self.user_sessions[user_id]
                    if session.is_active and session.expires_at > datetime.now():
                        return [session]
            return []
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {str(e)}")
            return []
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            now = datetime.now()
            expired_users = []
            
            async with self._lock:
                for user_id, session in self.user_sessions.items():
                    if session.expires_at <= now:
                        expired_users.append(user_id)
                
                for user_id in expired_users:
                    del self.user_sessions[user_id]
            
            if expired_users:
                logger.info(f"Cleaned up {len(expired_users)} expired sessions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
    
    async def close(self):
        """Close authentication manager."""
        try:
            for provider in self.oauth2_providers.values():
                await provider.close()
            
            logger.info("Authentication manager closed")
            
        except Exception as e:
            logger.error(f"Failed to close authentication manager: {str(e)}")
