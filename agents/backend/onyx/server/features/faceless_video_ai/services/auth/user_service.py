"""
User Service
Manages users and authentication
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


class User:
    """User model"""
    
    def __init__(
        self,
        user_id: str,
        email: str,
        password_hash: str,
        roles: List[str] = None,
        api_key: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        self.user_id = user_id
        self.email = email
        self.password_hash = password_hash
        self.roles = roles or ["user"]
        self.api_key = api_key
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "roles": self.roles,
            "api_key": self.api_key,
            "created_at": self.created_at.isoformat(),
        }


class UserService:
    """Manages users and authentication"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.users: Dict[str, User] = {}
        self.users_by_email: Dict[str, User] = {}
        self.users_by_api_key: Dict[str, User] = {}
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate random API key"""
        return secrets.token_urlsafe(32)
    
    def create_user(
        self,
        email: str,
        password: str,
        roles: List[str] = None
    ) -> User:
        """
        Create new user
        
        Args:
            email: User email
            password: User password
            roles: User roles
            
        Returns:
            Created user
        """
        if email in self.users_by_email:
            raise ValueError("User with this email already exists")
        
        user_id = hashlib.md5(email.encode()).hexdigest()
        password_hash = self.hash_password(password)
        api_key = self.generate_api_key()
        
        user = User(
            user_id=user_id,
            email=email,
            password_hash=password_hash,
            roles=roles or ["user"],
            api_key=api_key
        )
        
        self.users[user_id] = user
        self.users_by_email[email] = user
        self.users_by_api_key[api_key] = user
        
        logger.info(f"Created user: {email}")
        return user
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate user
        
        Args:
            email: User email
            password: User password
            
        Returns:
            User if authenticated, None otherwise
        """
        user = self.users_by_email.get(email)
        if not user:
            return None
        
        password_hash = self.hash_password(password)
        if user.password_hash != password_hash:
            return None
        
        logger.debug(f"User authenticated: {email}")
        return user
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        return self.users_by_api_key.get(api_key)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def update_user_roles(self, user_id: str, roles: List[str]):
        """Update user roles"""
        user = self.users.get(user_id)
        if user:
            user.roles = roles
            logger.info(f"Updated roles for user {user_id}: {roles}")
    
    def regenerate_api_key(self, user_id: str) -> str:
        """Regenerate API key for user"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Remove old API key
        if user.api_key:
            del self.users_by_api_key[user.api_key]
        
        # Generate new API key
        new_api_key = self.generate_api_key()
        user.api_key = new_api_key
        self.users_by_api_key[new_api_key] = user
        
        logger.info(f"Regenerated API key for user {user_id}")
        return new_api_key


_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get user service instance (singleton)"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service

