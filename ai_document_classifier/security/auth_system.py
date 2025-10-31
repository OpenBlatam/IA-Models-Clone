"""
Authentication and Authorization System
======================================

Advanced security system for the AI Document Classifier with JWT tokens,
role-based access control, and API key management.
"""

import jwt
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path
import bcrypt
import uuid
import json

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles and permissions"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    API_USER = "api_user"
    GUEST = "guest"

class Permission(Enum):
    """System permissions"""
    CLASSIFY_DOCUMENTS = "classify_documents"
    GENERATE_TEMPLATES = "generate_templates"
    BATCH_PROCESSING = "batch_processing"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    CONFIGURE_SERVICES = "configure_services"
    VIEW_ALERTS = "view_alerts"
    MANAGE_ALERTS = "manage_alerts"
    EXPORT_DATA = "export_data"
    ADMIN_ACCESS = "admin_access"

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIKey:
    """API Key model"""
    id: str
    user_id: str
    key_hash: str
    name: str
    permissions: List[Permission] = field(default_factory=list)
    rate_limit: int = 1000  # requests per hour
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """User session model"""
    id: str
    user_id: str
    token: str
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

class AuthSystem:
    """
    Advanced authentication and authorization system
    """
    
    def __init__(self, db_path: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize authentication system
        
        Args:
            db_path: Path to authentication database
            secret_key: Secret key for JWT tokens
        """
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent / "data" / "auth.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.secret_key = secret_key or self._generate_secret_key()
        self.token_expiry = timedelta(hours=24)
        self.refresh_token_expiry = timedelta(days=30)
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: list(Permission),
            UserRole.USER: [
                Permission.CLASSIFY_DOCUMENTS,
                Permission.GENERATE_TEMPLATES,
                Permission.BATCH_PROCESSING,
                Permission.VIEW_ANALYTICS,
                Permission.VIEW_ALERTS,
                Permission.EXPORT_DATA
            ],
            UserRole.READONLY: [
                Permission.CLASSIFY_DOCUMENTS,
                Permission.VIEW_ANALYTICS,
                Permission.VIEW_ALERTS
            ],
            UserRole.API_USER: [
                Permission.CLASSIFY_DOCUMENTS,
                Permission.GENERATE_TEMPLATES,
                Permission.BATCH_PROCESSING,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.GUEST: [
                Permission.CLASSIFY_DOCUMENTS
            ]
        }
        
        # Initialize database
        self._init_database()
        
        # Create default admin user
        self._create_default_admin()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    def _init_database(self):
        """Initialize authentication database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT,
                    api_keys TEXT,
                    created_at TIMESTAMP NOT NULL,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT,
                    rate_limit INTEGER DEFAULT 1000,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        try:
            admin_user = self.get_user_by_username("admin")
            if not admin_user:
                self.create_user(
                    username="admin",
                    email="admin@example.com",
                    password="admin123",
                    role=UserRole.ADMIN
                )
                logger.info("Default admin user created (username: admin, password: admin123)")
        except Exception as e:
            logger.error(f"Error creating default admin user: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        role: UserRole = UserRole.USER,
        permissions: Optional[List[Permission]] = None
    ) -> User:
        """
        Create a new user
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role
            permissions: Optional custom permissions
            
        Returns:
            Created user object
        """
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)
        
        # Set permissions based on role
        if permissions is None:
            permissions = self.role_permissions.get(role, [])
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users 
                    (id, username, email, password_hash, role, permissions, 
                     api_keys, created_at, is_active, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.id, user.username, user.email, password_hash, user.role.value,
                    json.dumps([p.value for p in user.permissions]),
                    json.dumps(user.api_keys), user.created_at.isoformat(),
                    user.is_active, json.dumps(user.metadata)
                ))
                conn.commit()
            
            logger.info(f"User created: {username}")
            return user
            
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            else:
                raise ValueError("User creation failed")
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, password_hash, role, permissions,
                           api_keys, created_at, last_login, is_active, metadata
                    FROM users 
                    WHERE username = ? AND is_active = TRUE
                """, (username,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Verify password
                if not self._verify_password(password, row[3]):
                    return None
                
                # Update last login
                conn.execute("""
                    UPDATE users SET last_login = ? WHERE id = ?
                """, (datetime.now().isoformat(), row[0]))
                conn.commit()
                
                # Create user object
                user = User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    role=UserRole(row[4]),
                    permissions=[Permission(p) for p in json.loads(row[5] or "[]")],
                    api_keys=json.loads(row[6] or "[]"),
                    created_at=datetime.fromisoformat(row[7]),
                    last_login=datetime.fromisoformat(row[8]) if row[8] else None,
                    is_active=bool(row[9]),
                    metadata=json.loads(row[10] or "{}")
                )
                
                return user
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def generate_jwt_token(self, user: User) -> Tuple[str, str]:
        """
        Generate JWT access and refresh tokens
        
        Args:
            user: User object
            
        Returns:
            Tuple of (access_token, refresh_token)
        """
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": now,
            "exp": now + self.token_expiry,
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user.id,
            "iat": now,
            "exp": now + self.refresh_token_expiry,
            "type": "refresh"
        }
        
        access_token = jwt.encode(access_payload, self.secret_key, algorithm="HS256")
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm="HS256")
        
        return access_token, refresh_token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def create_api_key(
        self, 
        user_id: str, 
        name: str, 
        permissions: Optional[List[Permission]] = None,
        rate_limit: int = 1000,
        expires_at: Optional[datetime] = None
    ) -> Tuple[str, APIKey]:
        """
        Create a new API key for user
        
        Args:
            user_id: User ID
            name: API key name
            permissions: Optional custom permissions
            rate_limit: Rate limit (requests per hour)
            expires_at: Optional expiration date
            
        Returns:
            Tuple of (api_key, APIKey object)
        """
        # Generate API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_api_key(api_key)
        
        # Get user to determine permissions
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        if permissions is None:
            permissions = user.permissions
        
        api_key_obj = APIKey(
            id=str(uuid.uuid4()),
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=expires_at
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_keys 
                    (id, user_id, key_hash, name, permissions, rate_limit,
                     expires_at, created_at, is_active, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    api_key_obj.id, api_key_obj.user_id, api_key_obj.key_hash,
                    api_key_obj.name, json.dumps([p.value for p in api_key_obj.permissions]),
                    api_key_obj.rate_limit, api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None,
                    api_key_obj.created_at.isoformat(), api_key_obj.is_active,
                    json.dumps(api_key_obj.metadata)
                ))
                conn.commit()
            
            # Add API key to user's list
            user.api_keys.append(api_key)
            self._update_user_api_keys(user)
            
            logger.info(f"API key created for user {user.username}: {name}")
            return api_key, api_key_obj
            
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise
    
    def verify_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
        """
        Verify API key and return user and API key objects
        
        Args:
            api_key: API key string
            
        Returns:
            Tuple of (User, APIKey) if valid, None otherwise
        """
        key_hash = self._hash_api_key(api_key)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get API key
                cursor = conn.execute("""
                    SELECT id, user_id, key_hash, name, permissions, rate_limit,
                           expires_at, created_at, last_used, is_active, metadata
                    FROM api_keys 
                    WHERE key_hash = ? AND is_active = TRUE
                """, (key_hash,))
                
                api_key_row = cursor.fetchone()
                if not api_key_row:
                    return None
                
                # Check expiration
                if api_key_row[6]:  # expires_at
                    expires_at = datetime.fromisoformat(api_key_row[6])
                    if datetime.now() > expires_at:
                        return None
                
                # Update last used
                conn.execute("""
                    UPDATE api_keys SET last_used = ? WHERE id = ?
                """, (datetime.now().isoformat(), api_key_row[0]))
                
                # Get user
                cursor = conn.execute("""
                    SELECT id, username, email, password_hash, role, permissions,
                           api_keys, created_at, last_login, is_active, metadata
                    FROM users 
                    WHERE id = ? AND is_active = TRUE
                """, (api_key_row[1],))
                
                user_row = cursor.fetchone()
                if not user_row:
                    return None
                
                conn.commit()
                
                # Create objects
                user = User(
                    id=user_row[0],
                    username=user_row[1],
                    email=user_row[2],
                    role=UserRole(user_row[4]),
                    permissions=[Permission(p) for p in json.loads(user_row[5] or "[]")],
                    api_keys=json.loads(user_row[6] or "[]"),
                    created_at=datetime.fromisoformat(user_row[7]),
                    last_login=datetime.fromisoformat(user_row[8]) if user_row[8] else None,
                    is_active=bool(user_row[9]),
                    metadata=json.loads(user_row[10] or "{}")
                )
                
                api_key_obj = APIKey(
                    id=api_key_row[0],
                    user_id=api_key_row[1],
                    key_hash=api_key_row[2],
                    name=api_key_row[3],
                    permissions=[Permission(p) for p in json.loads(api_key_row[4] or "[]")],
                    rate_limit=api_key_row[5],
                    expires_at=datetime.fromisoformat(api_key_row[6]) if api_key_row[6] else None,
                    created_at=datetime.fromisoformat(api_key_row[7]),
                    last_used=datetime.fromisoformat(api_key_row[8]) if api_key_row[8] else None,
                    is_active=bool(api_key_row[9]),
                    metadata=json.loads(api_key_row[10] or "{}")
                )
                
                return user, api_key_obj
                
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has specific permission
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        return permission in user.permissions
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, password_hash, role, permissions,
                           api_keys, created_at, last_login, is_active, metadata
                    FROM users 
                    WHERE id = ? AND is_active = TRUE
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    role=UserRole(row[4]),
                    permissions=[Permission(p) for p in json.loads(row[5] or "[]")],
                    api_keys=json.loads(row[6] or "[]"),
                    created_at=datetime.fromisoformat(row[7]),
                    last_login=datetime.fromisoformat(row[8]) if row[8] else None,
                    is_active=bool(row[9]),
                    metadata=json.loads(row[10] or "{}")
                )
                
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, password_hash, role, permissions,
                           api_keys, created_at, last_login, is_active, metadata
                    FROM users 
                    WHERE username = ? AND is_active = TRUE
                """, (username,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    role=UserRole(row[4]),
                    permissions=[Permission(p) for p in json.loads(row[5] or "[]")],
                    api_keys=json.loads(row[6] or "[]"),
                    created_at=datetime.fromisoformat(row[7]),
                    last_login=datetime.fromisoformat(row[8]) if row[8] else None,
                    is_active=bool(row[9]),
                    metadata=json.loads(row[10] or "{}")
                )
                
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def _update_user_api_keys(self, user: User):
        """Update user's API keys list"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE users SET api_keys = ? WHERE id = ?
                """, (json.dumps(user.api_keys), user.id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating user API keys: {e}")
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        key_hash = self._hash_api_key(api_key)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE api_keys SET is_active = FALSE WHERE key_hash = ?
                """, (key_hash,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False
    
    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, key_hash, name, permissions, rate_limit,
                           expires_at, created_at, last_used, is_active, metadata
                    FROM api_keys 
                    WHERE user_id = ? AND is_active = TRUE
                    ORDER BY created_at DESC
                """, (user_id,))
                
                api_keys = []
                for row in cursor.fetchall():
                    api_key = APIKey(
                        id=row[0],
                        user_id=row[1],
                        key_hash=row[2],
                        name=row[3],
                        permissions=[Permission(p) for p in json.loads(row[4] or "[]")],
                        rate_limit=row[5],
                        expires_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        created_at=datetime.fromisoformat(row[7]),
                        last_used=datetime.fromisoformat(row[8]) if row[8] else None,
                        is_active=bool(row[9]),
                        metadata=json.loads(row[10] or "{}")
                    )
                    api_keys.append(api_key)
                
                return api_keys
                
        except Exception as e:
            logger.error(f"Error getting user API keys: {e}")
            return []
    
    def list_users(self) -> List[User]:
        """List all active users"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, password_hash, role, permissions,
                           api_keys, created_at, last_login, is_active, metadata
                    FROM users 
                    WHERE is_active = TRUE
                    ORDER BY created_at DESC
                """)
                
                users = []
                for row in cursor.fetchall():
                    user = User(
                        id=row[0],
                        username=row[1],
                        email=row[2],
                        role=UserRole(row[4]),
                        permissions=[Permission(p) for p in json.loads(row[5] or "[]")],
                        api_keys=json.loads(row[6] or "[]"),
                        created_at=datetime.fromisoformat(row[7]),
                        last_login=datetime.fromisoformat(row[8]) if row[8] else None,
                        is_active=bool(row[9]),
                        metadata=json.loads(row[10] or "{}")
                    )
                    users.append(user)
                
                return users
                
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE users SET is_active = FALSE WHERE id = ?
                """, (user_id,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            return False

# Global auth system instance
auth_system = AuthSystem()

# Example usage
if __name__ == "__main__":
    # Initialize auth system
    auth = AuthSystem()
    
    # Create a test user
    try:
        user = auth.create_user("testuser", "test@example.com", "password123")
        print(f"User created: {user.username}")
        
        # Authenticate user
        authenticated_user = auth.authenticate_user("testuser", "password123")
        if authenticated_user:
            print(f"User authenticated: {authenticated_user.username}")
            
            # Generate tokens
            access_token, refresh_token = auth.generate_jwt_token(authenticated_user)
            print(f"Access token generated: {access_token[:50]}...")
            
            # Create API key
            api_key, api_key_obj = auth.create_api_key(
                authenticated_user.id, 
                "Test API Key",
                rate_limit=500
            )
            print(f"API key created: {api_key[:20]}...")
            
            # Verify API key
            user_from_key, api_key_from_verify = auth.verify_api_key(api_key)
            if user_from_key:
                print(f"API key verified for user: {user_from_key.username}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("Auth system initialized successfully")



























