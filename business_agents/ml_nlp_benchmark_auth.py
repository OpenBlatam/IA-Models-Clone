"""
ML NLP Benchmark Authentication System
Real, working authentication and authorization for ML NLP Benchmark system
"""

import time
import hashlib
import secrets
import jwt
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: Set[str]
    permissions: Set[str]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    api_key: Optional[str]
    rate_limit: int
    quota_used: int
    quota_limit: int

@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    key_value: str
    user_id: str
    name: str
    permissions: Set[str]
    rate_limit: int
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool

class MLNLPBenchmarkAuth:
    """Authentication and authorization system"""
    
    def __init__(self, secret_key: str = None, token_expiry: int = 3600):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry = token_expiry
        
        # In-memory storage (in production, use database)
        self.users = {}
        self.api_keys = {}
        self.sessions = {}
        self.rate_limits = {}
        
        # Default roles and permissions
        self.roles = {
            "admin": {
                "permissions": {"*"},  # All permissions
                "rate_limit": 10000,
                "quota_limit": 1000000
            },
            "user": {
                "permissions": {
                    "analyze_text", "analyze_batch", "get_stats", "get_health"
                },
                "rate_limit": 1000,
                "quota_limit": 100000
            },
            "guest": {
                "permissions": {
                    "analyze_text", "get_health"
                },
                "rate_limit": 100,
                "quota_limit": 10000
            }
        }
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = "admin123"
        password_hash, salt = self._hash_password(admin_password)
        
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@example.com",
            password_hash=password_hash,
            salt=salt,
            roles={"admin"},
            permissions={"*"},
            created_at=datetime.now(),
            last_login=None,
            is_active=True,
            api_key=self._generate_api_key(),
            rate_limit=10000,
            quota_used=0,
            quota_limit=1000000
        )
        
        self.users["admin"] = admin_user
        logger.info("Default admin user created")
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = password + salt
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        
        return hashed, salt
    
    def _verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash"""
        combined = password + salt
        computed_hash = hashlib.sha256(combined.encode()).hexdigest()
        return computed_hash == hashed
    
    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"ml_nlp_{secrets.token_urlsafe(32)}"
    
    def _generate_token(self, user_id: str, expires_in: int = None) -> str:
        """Generate JWT token"""
        if expires_in is None:
            expires_in = self.token_expiry
        
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def register_user(self, username: str, email: str, password: str, 
                     roles: Set[str] = None) -> tuple:
        """Register new user"""
        if username in self.users:
            return False, "Username already exists"
        
        if email in [user.email for user in self.users.values()]:
            return False, "Email already exists"
        
        # Validate password strength
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        # Hash password
        password_hash, salt = self._hash_password(password)
        
        # Set default roles
        if roles is None:
            roles = {"user"}
        
        # Get role permissions and limits
        user_permissions = set()
        rate_limit = 1000
        quota_limit = 100000
        
        for role in roles:
            if role in self.roles:
                role_data = self.roles[role]
                user_permissions.update(role_data["permissions"])
                rate_limit = max(rate_limit, role_data["rate_limit"])
                quota_limit = max(quota_limit, role_data["quota_limit"])
        
        # Create user
        user = User(
            user_id=secrets.token_hex(8),
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles,
            permissions=user_permissions,
            created_at=datetime.now(),
            last_login=None,
            is_active=True,
            api_key=self._generate_api_key(),
            rate_limit=rate_limit,
            quota_used=0,
            quota_limit=quota_limit
        )
        
        self.users[username] = user
        logger.info(f"User registered: {username}")
        
        return True, "User registered successfully"
    
    def authenticate_user(self, username: str, password: str) -> tuple:
        """Authenticate user with username and password"""
        if username not in self.users:
            return False, "Invalid username or password"
        
        user = self.users[username]
        
        if not user.is_active:
            return False, "User account is disabled"
        
        if not self._verify_password(password, user.password_hash, user.salt):
            return False, "Invalid username or password"
        
        # Update last login
        user.last_login = datetime.now()
        
        # Generate token
        token = self._generate_token(user.user_id)
        
        # Store session
        self.sessions[token] = {
            "user_id": user.user_id,
            "username": username,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        logger.info(f"User authenticated: {username}")
        
        return True, {
            "token": token,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": list(user.roles),
                "permissions": list(user.permissions),
                "rate_limit": user.rate_limit,
                "quota_used": user.quota_used,
                "quota_limit": user.quota_limit
            }
        }
    
    def authenticate_api_key(self, api_key: str) -> tuple:
        """Authenticate user with API key"""
        # Find user by API key
        user = None
        for u in self.users.values():
            if u.api_key == api_key:
                user = u
                break
        
        if not user:
            return False, "Invalid API key"
        
        if not user.is_active:
            return False, "User account is disabled"
        
        # Update last login
        user.last_login = datetime.now()
        
        logger.info(f"User authenticated with API key: {user.username}")
        
        return True, {
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": list(user.roles),
                "permissions": list(user.permissions),
                "rate_limit": user.rate_limit,
                "quota_used": user.quota_used,
                "quota_limit": user.quota_limit
            }
        }
    
    def verify_token(self, token: str) -> tuple:
        """Verify JWT token"""
        payload = self._verify_token(token)
        if not payload:
            return False, "Invalid or expired token"
        
        user_id = payload.get("user_id")
        if not user_id:
            return False, "Invalid token payload"
        
        # Find user
        user = None
        for u in self.users.values():
            if u.user_id == user_id:
                user = u
                break
        
        if not user:
            return False, "User not found"
        
        if not user.is_active:
            return False, "User account is disabled"
        
        # Update session activity
        if token in self.sessions:
            self.sessions[token]["last_activity"] = datetime.now()
        
        return True, {
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": list(user.roles),
                "permissions": list(user.permissions),
                "rate_limit": user.rate_limit,
                "quota_used": user.quota_used,
                "quota_limit": user.quota_limit
            }
        }
    
    def check_permission(self, user: Dict[str, Any], permission: str) -> bool:
        """Check if user has permission"""
        user_permissions = set(user.get("permissions", []))
        
        # Check for wildcard permission
        if "*" in user_permissions:
            return True
        
        # Check specific permission
        return permission in user_permissions
    
    def check_rate_limit(self, user: Dict[str, Any], endpoint: str) -> tuple:
        """Check rate limit for user"""
        user_id = user["user_id"]
        rate_limit = user["rate_limit"]
        
        current_time = time.time()
        window_start = current_time - 3600  # 1 hour window
        
        # Initialize rate limit tracking
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {}
        
        if endpoint not in self.rate_limits[user_id]:
            self.rate_limits[user_id][endpoint] = []
        
        # Clean old entries
        self.rate_limits[user_id][endpoint] = [
            timestamp for timestamp in self.rate_limits[user_id][endpoint]
            if timestamp > window_start
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[user_id][endpoint]) >= rate_limit:
            return False, f"Rate limit exceeded. Limit: {rate_limit} requests per hour"
        
        # Add current request
        self.rate_limits[user_id][endpoint].append(current_time)
        
        return True, "Rate limit OK"
    
    def check_quota(self, user: Dict[str, Any], usage: int = 1) -> tuple:
        """Check quota for user"""
        quota_used = user["quota_used"]
        quota_limit = user["quota_limit"]
        
        if quota_used + usage > quota_limit:
            return False, f"Quota exceeded. Used: {quota_used}, Limit: {quota_limit}"
        
        return True, "Quota OK"
    
    def update_quota(self, user_id: str, usage: int):
        """Update user quota usage"""
        for user in self.users.values():
            if user.user_id == user_id:
                user.quota_used += usage
                break
    
    def create_api_key(self, user_id: str, name: str, permissions: Set[str] = None,
                      rate_limit: int = 1000, expires_in: int = None) -> tuple:
        """Create API key for user"""
        # Find user
        user = None
        for u in self.users.values():
            if u.user_id == user_id:
                user = u
                break
        
        if not user:
            return False, "User not found"
        
        # Set default permissions
        if permissions is None:
            permissions = user.permissions
        
        # Generate API key
        api_key_value = self._generate_api_key()
        
        # Create API key object
        api_key = APIKey(
            key_id=secrets.token_hex(8),
            key_value=api_key_value,
            user_id=user_id,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=expires_in) if expires_in else None,
            last_used=None,
            is_active=True
        )
        
        self.api_keys[api_key_value] = api_key
        logger.info(f"API key created for user {user.username}: {name}")
        
        return True, {
            "api_key": api_key_value,
            "key_id": api_key.key_id,
            "name": name,
            "permissions": list(permissions),
            "rate_limit": rate_limit,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
        }
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key].is_active = False
            logger.info(f"API key revoked: {api_key}")
            return True
        return False
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u.is_active])
        total_api_keys = len(self.api_keys)
        active_api_keys = len([k for k in self.api_keys.values() if k.is_active])
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_api_keys": total_api_keys,
            "active_api_keys": active_api_keys,
            "roles": list(self.roles.keys()),
            "permissions": list(set().union(*[role["permissions"] for role in self.roles.values()]))
        }
    
    def logout(self, token: str) -> bool:
        """Logout user"""
        if token in self.sessions:
            del self.sessions[token]
            logger.info("User logged out")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        current_time = datetime.now()
        expired_tokens = []
        
        for token, session in self.sessions.items():
            if current_time - session["last_activity"] > timedelta(hours=24):
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.sessions[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

# Global auth instance
ml_nlp_benchmark_auth = MLNLPBenchmarkAuth()

def register_user(username: str, email: str, password: str, roles: Set[str] = None) -> tuple:
    """Register new user"""
    return ml_nlp_benchmark_auth.register_user(username, email, password, roles)

def authenticate_user(username: str, password: str) -> tuple:
    """Authenticate user with username and password"""
    return ml_nlp_benchmark_auth.authenticate_user(username, password)

def authenticate_api_key(api_key: str) -> tuple:
    """Authenticate user with API key"""
    return ml_nlp_benchmark_auth.authenticate_api_key(api_key)

def verify_token(token: str) -> tuple:
    """Verify JWT token"""
    return ml_nlp_benchmark_auth.verify_token(token)

def check_permission(user: Dict[str, Any], permission: str) -> bool:
    """Check if user has permission"""
    return ml_nlp_benchmark_auth.check_permission(user, permission)

def check_rate_limit(user: Dict[str, Any], endpoint: str) -> tuple:
    """Check rate limit for user"""
    return ml_nlp_benchmark_auth.check_rate_limit(user, endpoint)

def check_quota(user: Dict[str, Any], usage: int = 1) -> tuple:
    """Check quota for user"""
    return ml_nlp_benchmark_auth.check_quota(user, usage)

def update_quota(user_id: str, usage: int):
    """Update user quota usage"""
    ml_nlp_benchmark_auth.update_quota(user_id, usage)

def create_api_key(user_id: str, name: str, permissions: Set[str] = None,
                  rate_limit: int = 1000, expires_in: int = None) -> tuple:
    """Create API key for user"""
    return ml_nlp_benchmark_auth.create_api_key(user_id, name, permissions, rate_limit, expires_in)

def revoke_api_key(api_key: str) -> bool:
    """Revoke API key"""
    return ml_nlp_benchmark_auth.revoke_api_key(api_key)

def get_user_stats() -> Dict[str, Any]:
    """Get user statistics"""
    return ml_nlp_benchmark_auth.get_user_stats()

def logout(token: str) -> bool:
    """Logout user"""
    return ml_nlp_benchmark_auth.logout(token)

def cleanup_expired_sessions():
    """Cleanup expired sessions"""
    ml_nlp_benchmark_auth.cleanup_expired_sessions()











