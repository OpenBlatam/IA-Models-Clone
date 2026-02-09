"""
Authentication Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import uuid4


class AuthProvider(str, Enum):
    """Authentication providers"""
    LOCAL = "local"
    OAUTH_GOOGLE = "oauth_google"
    OAUTH_GITHUB = "oauth_github"
    OAUTH_SLACK = "oauth_slack"
    JWT = "jwt"


class User:
    """User model"""
    
    def __init__(
        self,
        email: str,
        password_hash: Optional[str] = None,
        provider: AuthProvider = AuthProvider.LOCAL
    ):
        self.id = str(uuid4())
        self.email = email
        self.password_hash = password_hash
        self.provider = provider
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.last_login: Optional[datetime] = None


class Token:
    """Token model"""
    
    def __init__(
        self,
        user_id: str,
        token_type: str = "bearer",
        expires_in: int = 3600
    ):
        self.id = str(uuid4())
        self.user_id = user_id
        self.token_type = token_type
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        self.created_at = datetime.utcnow()


class Session:
    """Session model"""
    
    def __init__(self, user_id: str, session_data: Optional[Dict[str, Any]] = None):
        self.id = str(uuid4())
        self.user_id = user_id
        self.session_data = session_data or {}
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=24)


class AuthBase(ABC):
    """Base interface for authentication"""
    
    @abstractmethod
    async def authenticate(
        self,
        email: str,
        password: Optional[str] = None,
        provider: Optional[AuthProvider] = None
    ) -> Optional[User]:
        """Authenticate user"""
        pass
    
    @abstractmethod
    async def generate_token(self, user: User) -> Token:
        """Generate authentication token"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate token and return user"""
        pass
    
    @abstractmethod
    async def create_session(self, user: User) -> Session:
        """Create user session"""
        pass

