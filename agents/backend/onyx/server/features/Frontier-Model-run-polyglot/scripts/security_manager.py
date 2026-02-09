#!/usr/bin/env python3
"""
Advanced Security and Encryption Features for Frontier Model Training
Provides comprehensive security, encryption, access control, and audit logging.
"""

import os
import json
import hashlib
import hmac
import secrets
import base64
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import bcrypt
from passlib.context import CryptContext
import redis
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from contextlib import contextmanager

console = Console()

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EncryptionType(Enum):
    """Types of encryption."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"

class AccessLevel(Enum):
    """Access levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100

@dataclass
class User:
    """User information."""
    username: str
    email: str
    password_hash: str
    access_level: AccessLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    permissions: List[str] = None

@dataclass
class AuditLog:
    """Audit log entry."""
    timestamp: datetime
    user: str
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = None

class EncryptionManager:
    """Advanced encryption and decryption manager."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption keys
        self.symmetric_key = self._generate_symmetric_key()
        self.private_key, self.public_key = self._generate_asymmetric_keys()
        
        # Initialize Fernet for symmetric encryption
        self.fernet = Fernet(self.symmetric_key)
    
    def _generate_symmetric_key(self) -> bytes:
        """Generate symmetric encryption key."""
        if self.config.encryption_key:
            # Derive key from password
            password = self.config.encryption_key.encode()
            salt = b'frontier_model_salt'  # In practice, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            return base64.urlsafe_b64encode(kdf.derive(password))
        else:
            return Fernet.generate_key()
    
    def _generate_asymmetric_keys(self) -> Tuple[bytes, bytes]:
        """Generate asymmetric encryption keys."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_symmetric(self, data: Union[str, bytes]) -> str:
        """Encrypt data using symmetric encryption."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = self.fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode('utf-8')
    
    def encrypt_asymmetric(self, data: Union[str, bytes], public_key: bytes = None) -> str:
        """Encrypt data using asymmetric encryption."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if public_key is None:
            public_key = self.public_key
        
        # Load public key
        pub_key = serialization.load_pem_public_key(public_key, backend=default_backend())
        
        # Encrypt data
        encrypted_data = pub_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using asymmetric encryption."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        
        # Load private key
        private_key = serialization.load_pem_private_key(
            self.private_key,
            password=None,
            backend=default_backend()
        )
        
        # Decrypt data
        decrypted_data = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return decrypted_data.decode('utf-8')
    
    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """Encrypt a file."""
        if output_path is None:
            output_path = file_path + '.encrypted'
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = self.fernet.encrypt(file_data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """Decrypt a file."""
        if output_path is None:
            output_path = encrypted_file_path.replace('.encrypted', '')
        
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path
    
    def generate_hash(self, data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """Generate hash of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def generate_hmac(self, data: Union[str, bytes], key: Union[str, bytes]) -> str:
        """Generate HMAC of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()

class AuthenticationManager:
    """Advanced authentication and authorization manager."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Password context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # User storage (in practice, use a proper database)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Initialize default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_password = self.hash_password("admin123")
        admin_user = User(
            username="admin",
            email="admin@frontier-model.com",
            password_hash=admin_password,
            access_level=AccessLevel.SUPER_ADMIN,
            created_at=datetime.now(),
            permissions=["*"]
        )
        self.users["admin"] = admin_user
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        return len(errors) == 0, errors
    
    def create_user(self, username: str, email: str, password: str, 
                   access_level: AccessLevel, permissions: List[str] = None) -> Tuple[bool, str]:
        """Create new user."""
        # Validate password
        is_valid, errors = self.validate_password(password)
        if not is_valid:
            return False, f"Password validation failed: {', '.join(errors)}"
        
        # Check if user exists
        if username in self.users:
            return False, "Username already exists"
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            access_level=access_level,
            created_at=datetime.now(),
            permissions=permissions or []
        )
        
        self.users[username] = user
        
        # Log user creation
        self._log_audit_event(username, "user_created", f"user:{username}", "127.0.0.1", "system", True)
        
        return True, "User created successfully"
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = "127.0.0.1", user_agent: str = "unknown") -> Tuple[bool, str, Optional[str]]:
        """Authenticate user."""
        # Check rate limiting
        if not self.rate_limiter.is_allowed(ip_address):
            self._log_audit_event(username, "login_attempt", "auth", ip_address, user_agent, False, 
                                {"reason": "rate_limited"})
            return False, "Too many login attempts", None
        
        # Check if user exists
        if username not in self.users:
            self._log_audit_event(username, "login_attempt", "auth", ip_address, user_agent, False, 
                                {"reason": "user_not_found"})
            return False, "Invalid credentials", None
        
        user = self.users[username]
        
        # Check if user is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_audit_event(username, "login_attempt", "auth", ip_address, user_agent, False, 
                                {"reason": "account_locked"})
            return False, "Account is locked", None
        
        # Check if user is active
        if not user.is_active:
            self._log_audit_event(username, "login_attempt", "auth", ip_address, user_agent, False, 
                                {"reason": "account_inactive"})
            return False, "Account is inactive", None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)
                self._log_audit_event(username, "account_locked", f"user:{username}", ip_address, user_agent, False, 
                                    {"reason": "too_many_failed_attempts"})
            
            self._log_audit_event(username, "login_attempt", "auth", ip_address, user_agent, False, 
                                {"reason": "invalid_password"})
            return False, "Invalid credentials", None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Generate JWT token
        token = self._generate_jwt_token(user)
        
        # Create session
        session_id = secrets.token_hex(16)
        self.sessions[session_id] = {
            "user": username,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=self.config.session_timeout_minutes),
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        self._log_audit_event(username, "login_success", "auth", ip_address, user_agent, True)
        
        return True, "Authentication successful", token
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "username": user.username,
            "email": user.email,
            "access_level": user.access_level.value,
            "permissions": user.permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours)
        }
        
        secret = self.config.jwt_secret or secrets.token_hex(32)
        return jwt.encode(payload, secret, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token."""
        try:
            secret = self.config.jwt_secret or secrets.token_hex(32)
            payload = jwt.decode(token, secret, algorithms=[self.config.jwt_algorithm])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}
    
    def check_permission(self, username: str, resource: str, action: str) -> bool:
        """Check if user has permission for resource and action."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Super admin has all permissions
        if user.access_level == AccessLevel.SUPER_ADMIN:
            return True
        
        # Check specific permissions
        required_permission = f"{resource}:{action}"
        if "*" in user.permissions or required_permission in user.permissions:
            return True
        
        # Check wildcard permissions
        for permission in user.permissions:
            if permission.endswith("*") and required_permission.startswith(permission[:-1]):
                return True
        
        return False
    
    def _log_audit_event(self, user: str, action: str, resource: str, 
                        ip_address: str, user_agent: str, success: bool, details: Dict[str, Any] = None):
        """Log audit event."""
        if not self.config.enable_audit_logging:
            return
        
        audit_log = AuditLog(
            timestamp=datetime.now(),
            user=user,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
        
        # In practice, save to database
        self.logger.info(f"Audit: {asdict(audit_log)}")

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = datetime.now()
            
            # Clean old requests
            if identifier in self.requests:
                self.requests[identifier] = [
                    req_time for req_time in self.requests[identifier]
                    if now - req_time < timedelta(minutes=1)
                ]
            else:
                self.requests[identifier] = []
            
            # Check if under limit
            if len(self.requests[identifier]) < self.requests_per_minute:
                self.requests[identifier].append(now)
                return True
            
            return False

class SecurityManager:
    """Main security manager."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.encryption_manager = EncryptionManager(config)
        self.auth_manager = AuthenticationManager(config)
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.failed_attempts: Dict[str, int] = {}
        
        # Initialize security database
        self._init_security_db()
    
    def _init_security_db(self):
        """Initialize security database."""
        self.db_path = Path("./security.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    details TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT
                )
            """)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.encryption_manager.encrypt_symmetric(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption_manager.decrypt_symmetric(encrypted_data)
    
    def create_user(self, username: str, email: str, password: str, 
                   access_level: AccessLevel, permissions: List[str] = None) -> Tuple[bool, str]:
        """Create new user."""
        return self.auth_manager.create_user(username, email, password, access_level, permissions)
    
    def authenticate_user(self, username: str, password: str, 
                        ip_address: str = "127.0.0.1", user_agent: str = "unknown") -> Tuple[bool, str, Optional[str]]:
        """Authenticate user."""
        return self.auth_manager.authenticate_user(username, password, ip_address, user_agent)
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token."""
        return self.auth_manager.verify_jwt_token(token)
    
    def check_permission(self, username: str, resource: str, action: str) -> bool:
        """Check user permission."""
        return self.auth_manager.check_permission(username, resource, action)
    
    def log_security_event(self, event_type: str, severity: SecurityLevel, 
                          description: str, details: Dict[str, Any] = None):
        """Log security event."""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "severity": severity.value,
            "description": description,
            "details": details or {}
        }
        
        self.security_events.append(event)
        
        # Save to database
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO security_events (timestamp, event_type, severity, description, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event["timestamp"].isoformat(),
                event["event_type"],
                event["severity"],
                event["description"],
                json.dumps(event["details"])
            ))
        
        # Log to console
        console.print(f"[red]Security Event: {description}[/red]")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security report."""
        return {
            "total_users": len(self.auth_manager.users),
            "active_sessions": len(self.auth_manager.sessions),
            "security_events": len(self.security_events),
            "failed_attempts": sum(self.failed_attempts.values()),
            "encryption_enabled": self.config.enable_encryption,
            "audit_logging_enabled": self.config.enable_audit_logging,
            "rate_limiting_enabled": self.config.enable_rate_limiting
        }
    
    def backup_security_data(self, backup_path: str):
        """Backup security data."""
        backup_file = Path(backup_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Encrypt and backup database
        encrypted_path = self.encryption_manager.encrypt_file(str(self.db_path), str(backup_file))
        
        console.print(f"[green]Security data backed up to: {encrypted_path}[/green]")
    
    def restore_security_data(self, backup_path: str):
        """Restore security data."""
        # Decrypt and restore database
        decrypted_path = self.encryption_manager.decrypt_file(backup_path, str(self.db_path))
        
        console.print(f"[green]Security data restored from: {backup_path}[/green]")

def main():
    """Main function for security CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security and Encryption Tools")
    parser.add_argument("--action", type=str, 
                       choices=["encrypt", "decrypt", "create-user", "authenticate", "report"],
                       required=True, help="Action to perform")
    parser.add_argument("--data", type=str, help="Data to encrypt/decrypt")
    parser.add_argument("--file", type=str, help="File to encrypt/decrypt")
    parser.add_argument("--username", type=str, help="Username")
    parser.add_argument("--password", type=str, help="Password")
    parser.add_argument("--email", type=str, help="Email")
    parser.add_argument("--access-level", type=str, 
                       choices=["read", "write", "admin", "super_admin"],
                       default="read", help="Access level")
    
    args = parser.parse_args()
    
    # Create security configuration
    config = SecurityConfig(
        encryption_key="frontier_model_secret_key",
        jwt_secret="frontier_model_jwt_secret",
        enable_encryption=True,
        enable_audit_logging=True,
        enable_rate_limiting=True
    )
    
    # Create security manager
    security_manager = SecurityManager(config)
    
    if args.action == "encrypt":
        if args.data:
            encrypted = security_manager.encrypt_sensitive_data(args.data)
            console.print(f"[green]Encrypted data: {encrypted}[/green]")
        elif args.file:
            encrypted_path = security_manager.encryption_manager.encrypt_file(args.file)
            console.print(f"[green]Encrypted file: {encrypted_path}[/green]")
        else:
            console.print("[red]No data or file specified[/red]")
    
    elif args.action == "decrypt":
        if args.data:
            decrypted = security_manager.decrypt_sensitive_data(args.data)
            console.print(f"[green]Decrypted data: {decrypted}[/green]")
        elif args.file:
            decrypted_path = security_manager.encryption_manager.decrypt_file(args.file)
            console.print(f"[green]Decrypted file: {decrypted_path}[/green]")
        else:
            console.print("[red]No data or file specified[/red]")
    
    elif args.action == "create-user":
        if not all([args.username, args.password, args.email]):
            console.print("[red]Username, password, and email are required[/red]")
            return
        
        success, message = security_manager.create_user(
            args.username,
            args.email,
            args.password,
            AccessLevel(args.access_level)
        )
        
        if success:
            console.print(f"[green]{message}[/green]")
        else:
            console.print(f"[red]{message}[/red]")
    
    elif args.action == "authenticate":
        if not all([args.username, args.password]):
            console.print("[red]Username and password are required[/red]")
            return
        
        success, message, token = security_manager.authenticate_user(args.username, args.password)
        
        if success:
            console.print(f"[green]{message}[/green]")
            console.print(f"[blue]Token: {token}[/blue]")
        else:
            console.print(f"[red]{message}[/red]")
    
    elif args.action == "report":
        report = security_manager.get_security_report()
        
        table = Table(title="Security Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in report.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)

if __name__ == "__main__":
    main()
