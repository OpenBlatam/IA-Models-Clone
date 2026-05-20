"""
Authentication and authorization system.
"""

import logging
import hashlib
import secrets
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Represents a user in the system."""
    user_id: str
    email: str
    username: str
    role: str = "student"
    created_at: datetime = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AuthManager:
    """
    Manages authentication and authorization.
    """
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.password_hashes: Dict[str, str] = {}
    
    def register_user(
        self,
        email: str,
        username: str,
        password: str,
        role: str = "student"
    ) -> User:
        """
        Register a new user.
        
        Args:
            email: User email
            username: Username
            password: Plain text password
            role: User role (student, teacher, admin)
        
        Returns:
            Created user
        """
        user_id = self._generate_user_id(email)
        
        if user_id in self.users:
            raise ValueError("User already exists")
        
        user = User(
            user_id=user_id,
            email=email,
            username=username,
            role=role
        )
        
        self.users[user_id] = user
        self.password_hashes[user_id] = self._hash_password(password)
        
        logger.info(f"Registered user {user_id} with role {role}")
        
        return user
    
    def authenticate(self, email: str, password: str) -> Optional[str]:
        """
        Authenticate a user and return session token.
        
        Args:
            email: User email
            password: Plain text password
        
        Returns:
            Session token if successful, None otherwise
        """
        user_id = self._generate_user_id(email)
        
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        
        if not user.is_active:
            return None
        
        if not self._verify_password(password, self.password_hashes[user_id]):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        # Create session
        session_token = self._generate_session_token()
        self.sessions[session_token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24)
        }
        
        logger.info(f"User {user_id} authenticated")
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """
        Validate a session token and return user.
        
        Args:
            session_token: Session token
        
        Returns:
            User if valid, None otherwise
        """
        if session_token not in self.sessions:
            return None
        
        session = self.sessions[session_token]
        
        if datetime.now() > session["expires_at"]:
            del self.sessions[session_token]
            return None
        
        user_id = session["user_id"]
        return self.users.get(user_id)
    
    def logout(self, session_token: str):
        """Logout a user by removing session."""
        if session_token in self.sessions:
            del self.sessions[session_token]
            logger.info(f"User logged out")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def has_permission(self, user: User, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: User object
            permission: Permission to check
        
        Returns:
            True if user has permission
        """
        role_permissions = {
            "student": ["ask_question", "view_own_reports", "view_own_profile"],
            "teacher": ["ask_question", "view_own_reports", "view_own_profile", "view_student_reports", "create_quiz"],
            "admin": ["*"]  # All permissions
        }
        
        user_permissions = role_permissions.get(user.role, [])
        
        return "*" in user_permissions or permission in user_permissions
    
    def _generate_user_id(self, email: str) -> str:
        """Generate user ID from email."""
        return hashlib.md5(email.encode()).hexdigest()
    
    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against hash."""
        return self._hash_password(password) == password_hash
    
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return secrets.token_urlsafe(32)






