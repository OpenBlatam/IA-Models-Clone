"""
BUL Advanced Security Manager
============================

Advanced security management system for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
import base64
import sqlite3
from dataclasses import dataclass
from enum import Enum
import yaml
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import bcrypt

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEventType(Enum):
    """Security event types."""
    LOGIN_ATTEMPT = "login_attempt"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    API_ACCESS = "api_access"
    FILE_ACCESS = "file_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_ALERT = "security_alert"

@dataclass
class SecurityPolicy:
    """Security policy definition."""
    id: str
    name: str
    description: str
    security_level: SecurityLevel
    rules: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime = None

@dataclass
class SecurityEvent:
    """Security event definition."""
    id: str
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    timestamp: datetime
    severity: SecurityLevel

@dataclass
class SecurityUser:
    """Security user definition."""
    id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: List[str]
    permissions: List[str]
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = None

class AdvancedSecurityManager:
    """Advanced security management system for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.policies = {}
        self.users = {}
        self.security_events = []
        self.encryption_key = None
        self.jwt_secret = None
        self.init_security_environment()
        self.load_policies()
        self.load_users()
        self.setup_encryption()
    
    def init_security_environment(self):
        """Initialize security environment."""
        print("🔒 Initializing advanced security environment...")
        
        # Create security directories
        self.security_dir = Path("security")
        self.security_dir.mkdir(exist_ok=True)
        
        self.keys_dir = Path("security_keys")
        self.keys_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("security_logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_security_database()
        
        print("✅ Advanced security environment initialized")
    
    def init_security_database(self):
        """Initialize security database."""
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_policies (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                security_level TEXT,
                rules TEXT,
                enabled BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                salt TEXT,
                roles TEXT,
                permissions TEXT,
                last_login DATETIME,
                failed_attempts INTEGER DEFAULT 0,
                locked_until DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id TEXT PRIMARY KEY,
                event_type TEXT,
                user_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                resource TEXT,
                action TEXT,
                result TEXT,
                details TEXT,
                timestamp DATETIME,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_policies(self):
        """Load existing security policies."""
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM security_policies")
        rows = cursor.fetchall()
        
        for row in rows:
            policy = SecurityPolicy(
                id=row[0],
                name=row[1],
                description=row[2],
                security_level=SecurityLevel(row[3]),
                rules=json.loads(row[4]),
                enabled=bool(row[5]),
                created_at=datetime.fromisoformat(row[6])
            )
            self.policies[policy.id] = policy
        
        conn.close()
        
        # Create default policies if none exist
        if not self.policies:
            self.create_default_policies()
        
        print(f"✅ Loaded {len(self.policies)} security policies")
    
    def load_users(self):
        """Load existing security users."""
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM security_users")
        rows = cursor.fetchall()
        
        for row in rows:
            user = SecurityUser(
                id=row[0],
                username=row[1],
                email=row[2],
                password_hash=row[3],
                salt=row[4],
                roles=json.loads(row[5]),
                permissions=json.loads(row[6]),
                last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                failed_attempts=row[8] or 0,
                locked_until=datetime.fromisoformat(row[9]) if row[9] else None,
                created_at=datetime.fromisoformat(row[10])
            )
            self.users[user.id] = user
        
        conn.close()
        
        # Create default admin user if none exist
        if not self.users:
            self.create_default_admin_user()
        
        print(f"✅ Loaded {len(self.users)} security users")
    
    def setup_encryption(self):
        """Setup encryption keys."""
        # Load or generate encryption key
        key_file = self.keys_dir / "encryption.key"
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
        
        # Load or generate JWT secret
        jwt_file = self.keys_dir / "jwt.secret"
        if jwt_file.exists():
            with open(jwt_file, 'r') as f:
                self.jwt_secret = f.read().strip()
        else:
            self.jwt_secret = secrets.token_urlsafe(32)
            with open(jwt_file, 'w') as f:
                f.write(self.jwt_secret)
        
        print("✅ Encryption keys setup completed")
    
    def create_default_policies(self):
        """Create default security policies."""
        default_policies = [
            {
                'id': 'password_policy',
                'name': 'Password Policy',
                'description': 'Password complexity and security requirements',
                'security_level': SecurityLevel.HIGH,
                'rules': [
                    {'type': 'min_length', 'value': 8},
                    {'type': 'require_uppercase', 'value': True},
                    {'type': 'require_lowercase', 'value': True},
                    {'type': 'require_numbers', 'value': True},
                    {'type': 'require_special_chars', 'value': True},
                    {'type': 'max_age_days', 'value': 90}
                ]
            },
            {
                'id': 'login_policy',
                'name': 'Login Policy',
                'description': 'Login attempt and lockout policies',
                'security_level': SecurityLevel.MEDIUM,
                'rules': [
                    {'type': 'max_failed_attempts', 'value': 5},
                    {'type': 'lockout_duration_minutes', 'value': 30},
                    {'type': 'session_timeout_minutes', 'value': 60}
                ]
            },
            {
                'id': 'api_access_policy',
                'name': 'API Access Policy',
                'description': 'API access and rate limiting policies',
                'security_level': SecurityLevel.HIGH,
                'rules': [
                    {'type': 'rate_limit_requests_per_minute', 'value': 100},
                    {'type': 'require_authentication', 'value': True},
                    {'type': 'require_https', 'value': True}
                ]
            },
            {
                'id': 'data_access_policy',
                'name': 'Data Access Policy',
                'description': 'Data access and modification policies',
                'security_level': SecurityLevel.CRITICAL,
                'rules': [
                    {'type': 'audit_all_access', 'value': True},
                    {'type': 'require_authorization', 'value': True},
                    {'type': 'encrypt_sensitive_data', 'value': True}
                ]
            }
        ]
        
        for policy_data in default_policies:
            self.create_policy(
                policy_id=policy_data['id'],
                name=policy_data['name'],
                description=policy_data['description'],
                security_level=policy_data['security_level'],
                rules=policy_data['rules']
            )
    
    def create_default_admin_user(self):
        """Create default admin user."""
        admin_user = self.create_user(
            user_id="admin",
            username="admin",
            email="admin@company.com",
            password="admin123!",
            roles=["admin"],
            permissions=["*"]
        )
        print(f"✅ Created default admin user: {admin_user.username}")
    
    def create_policy(self, policy_id: str, name: str, description: str,
                     security_level: SecurityLevel, rules: List[Dict[str, Any]]) -> SecurityPolicy:
        """Create a new security policy."""
        policy = SecurityPolicy(
            id=policy_id,
            name=name,
            description=description,
            security_level=security_level,
            rules=rules,
            enabled=True,
            created_at=datetime.now()
        )
        
        self.policies[policy_id] = policy
        
        # Save to database
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO security_policies 
            (id, name, description, security_level, rules, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (policy_id, name, description, security_level.value, 
              json.dumps(rules), True, policy.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created security policy: {name}")
        return policy
    
    def create_user(self, user_id: str, username: str, email: str, password: str,
                   roles: List[str], permissions: List[str]) -> SecurityUser:
        """Create a new security user."""
        # Generate salt and hash password
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        
        user = SecurityUser(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles,
            permissions=permissions,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        
        # Save to database
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO security_users 
            (id, username, email, password_hash, salt, roles, permissions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, username, email, password_hash, salt, 
              json.dumps(roles), json.dumps(permissions), user.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created security user: {username}")
        return user
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "unknown") -> Dict[str, Any]:
        """Authenticate a user."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                None, ip_address, "unknown", "authentication", "login",
                "failed", {"reason": "user_not_found", "username": username},
                SecurityLevel.MEDIUM
            )
            return {"success": False, "error": "Invalid credentials"}
        
        # Check if user is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                user.id, ip_address, "unknown", "authentication", "login",
                "failed", {"reason": "account_locked", "locked_until": user.locked_until.isoformat()},
                SecurityLevel.HIGH
            )
            return {"success": False, "error": "Account is locked"}
        
        # Verify password
        if not self._verify_password(password, user.password_hash, user.salt):
            # Increment failed attempts
            user.failed_attempts += 1
            
            # Check if should lock account
            login_policy = self.policies.get('login_policy')
            if login_policy:
                max_attempts = next((rule['value'] for rule in login_policy.rules if rule['type'] == 'max_failed_attempts'), 5)
                if user.failed_attempts >= max_attempts:
                    lockout_duration = next((rule['value'] for rule in login_policy.rules if rule['type'] == 'lockout_duration_minutes'), 30)
                    user.locked_until = datetime.now() + timedelta(minutes=lockout_duration)
            
            self._save_user(user)
            
            self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                user.id, ip_address, "unknown", "authentication", "login",
                "failed", {"reason": "invalid_password", "failed_attempts": user.failed_attempts},
                SecurityLevel.MEDIUM
            )
            return {"success": False, "error": "Invalid credentials"}
        
        # Reset failed attempts and update last login
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        self._save_user(user)
        
        # Generate JWT token
        token = self._generate_jwt_token(user)
        
        self._log_security_event(
            SecurityEventType.LOGIN_ATTEMPT,
            user.id, ip_address, "unknown", "authentication", "login",
            "success", {"username": username},
            SecurityLevel.LOW
        )
        
        return {
            "success": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": user.roles,
                "permissions": user.permissions
            },
            "token": token
        }
    
    def authorize_user(self, user_id: str, resource: str, action: str) -> bool:
        """Authorize a user for a resource and action."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check if user has wildcard permission
        if "*" in user.permissions:
            return True
        
        # Check specific permissions
        permission = f"{resource}:{action}"
        if permission in user.permissions:
            return True
        
        # Check role-based permissions
        for role in user.roles:
            role_permission = f"{role}:{resource}:{action}"
            if role_permission in user.permissions:
                return True
        
        return False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        fernet = Fernet(self.encryption_key)
        decoded_data = base64.b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(decoded_data)
        return decrypted_data.decode()
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security policies."""
        password_policy = self.policies.get('password_policy')
        if not password_policy:
            return {"valid": True, "errors": []}
        
        errors = []
        
        for rule in password_policy.rules:
            rule_type = rule['type']
            rule_value = rule['value']
            
            if rule_type == 'min_length' and len(password) < rule_value:
                errors.append(f"Password must be at least {rule_value} characters long")
            
            elif rule_type == 'require_uppercase' and rule_value and not re.search(r'[A-Z]', password):
                errors.append("Password must contain at least one uppercase letter")
            
            elif rule_type == 'require_lowercase' and rule_value and not re.search(r'[a-z]', password):
                errors.append("Password must contain at least one lowercase letter")
            
            elif rule_type == 'require_numbers' and rule_value and not re.search(r'\d', password):
                errors.append("Password must contain at least one number")
            
            elif rule_type == 'require_special_chars' and rule_value and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("Password must contain at least one special character")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def audit_access(self, user_id: str, resource: str, action: str, 
                    ip_address: str = "unknown", user_agent: str = "unknown") -> None:
        """Audit user access to resources."""
        self._log_security_event(
            SecurityEventType.DATA_ACCESS,
            user_id, ip_address, user_agent, resource, action,
            "success", {"user_id": user_id, "resource": resource, "action": action},
            SecurityLevel.LOW
        )
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with salt."""
        return bcrypt.hashpw(password.encode(), salt.encode()).decode()
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode(), password_hash.encode())
        except:
            return False
    
    def _generate_jwt_token(self, user: SecurityUser) -> str:
        """Generate JWT token for user."""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'roles': user.roles,
            'permissions': user.permissions,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return {"valid": True, "payload": payload}
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "error": "Invalid token"}
    
    def _save_user(self, user: SecurityUser):
        """Save user to database."""
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE security_users 
            SET failed_attempts = ?, locked_until = ?, last_login = ?
            WHERE id = ?
        ''', (user.failed_attempts, 
              user.locked_until.isoformat() if user.locked_until else None,
              user.last_login.isoformat() if user.last_login else None,
              user.id))
        
        conn.commit()
        conn.close()
    
    def _log_security_event(self, event_type: SecurityEventType, user_id: Optional[str],
                           ip_address: str, user_agent: str, resource: str, action: str,
                           result: str, details: Dict[str, Any], severity: SecurityLevel):
        """Log a security event."""
        event = SecurityEvent(
            id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details,
            timestamp=datetime.now(),
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Save to database
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO security_events 
            (id, event_type, user_id, ip_address, user_agent, resource, action, result, details, timestamp, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (event.id, event.event_type.value, event.user_id, event.ip_address,
              event.user_agent, event.resource, event.action, event.result,
              json.dumps(event.details), event.timestamp.isoformat(), event.severity.value))
        
        conn.commit()
        conn.close()
        
        # Log to file
        self._log_to_file(event)
    
    def _log_to_file(self, event: SecurityEvent):
        """Log security event to file."""
        log_file = self.logs_dir / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a') as f:
            f.write(f"{event.timestamp.isoformat()} | {event.severity.value.upper()} | {event.event_type.value} | {event.user_id or 'N/A'} | {event.ip_address} | {event.resource} | {event.action} | {event.result} | {json.dumps(event.details)}\n")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        total_events = len(self.security_events)
        failed_logins = len([e for e in self.security_events if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE])
        successful_logins = len([e for e in self.security_events if e.event_type == SecurityEventType.LOGIN_ATTEMPT and e.result == "success"])
        
        # Get events by severity
        events_by_severity = {}
        for level in SecurityLevel:
            events_by_severity[level.value] = len([e for e in self.security_events if e.severity == level])
        
        return {
            'total_users': len(self.users),
            'total_policies': len(self.policies),
            'total_events': total_events,
            'failed_logins': failed_logins,
            'successful_logins': successful_logins,
            'events_by_severity': events_by_severity,
            'locked_users': len([u for u in self.users.values() if u.locked_until and datetime.now() < u.locked_until])
        }
    
    def generate_security_report(self) -> str:
        """Generate security report."""
        stats = self.get_security_stats()
        
        report = f"""
BUL Advanced Security Manager Report
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

USERS
-----
Total Users: {stats['total_users']}
Locked Users: {stats['locked_users']}
"""
        
        for user_id, user in self.users.items():
            report += f"""
{user.username} ({user_id}):
  Email: {user.email}
  Roles: {', '.join(user.roles)}
  Last Login: {user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never'}
  Failed Attempts: {user.failed_attempts}
  Status: {'Locked' if user.locked_until and datetime.now() < user.locked_until else 'Active'}
"""
        
        report += f"""
POLICIES
--------
Total Policies: {stats['total_policies']}
"""
        
        for policy_id, policy in self.policies.items():
            report += f"""
{policy.name} ({policy_id}):
  Level: {policy.security_level.value}
  Rules: {len(policy.rules)}
  Enabled: {policy.enabled}
  Created: {policy.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
SECURITY EVENTS
---------------
Total Events: {stats['total_events']}
Successful Logins: {stats['successful_logins']}
Failed Logins: {stats['failed_logins']}

Events by Severity:
"""
        
        for severity, count in stats['events_by_severity'].items():
            report += f"  {severity.title()}: {count}\n"
        
        # Show recent events
        recent_events = sorted(
            self.security_events,
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        if recent_events:
            report += f"""
RECENT SECURITY EVENTS
----------------------
"""
            for event in recent_events:
                report += f"""
{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {event.severity.value.upper()} | {event.event_type.value}
  User: {event.user_id or 'N/A'}
  IP: {event.ip_address}
  Resource: {event.resource}
  Action: {event.action}
  Result: {event.result}
"""
        
        return report

def main():
    """Main advanced security manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Advanced Security Manager")
    parser.add_argument("--create-policy", help="Create a new security policy")
    parser.add_argument("--create-user", help="Create a new security user")
    parser.add_argument("--authenticate", help="Authenticate a user")
    parser.add_argument("--authorize", help="Authorize a user")
    parser.add_argument("--encrypt", help="Encrypt data")
    parser.add_argument("--decrypt", help="Decrypt data")
    parser.add_argument("--validate-password", help="Validate password")
    parser.add_argument("--list-users", action="store_true", help="List all users")
    parser.add_argument("--list-policies", action="store_true", help="List all policies")
    parser.add_argument("--stats", action="store_true", help="Show security statistics")
    parser.add_argument("--report", action="store_true", help="Generate security report")
    parser.add_argument("--name", help="Name for policy/user")
    parser.add_argument("--description", help="Description for policy")
    parser.add_argument("--security-level", choices=['low', 'medium', 'high', 'critical'],
                       help="Security level")
    parser.add_argument("--username", help="Username")
    parser.add_argument("--email", help="Email address")
    parser.add_argument("--password", help="Password")
    parser.add_argument("--roles", help="Comma-separated list of roles")
    parser.add_argument("--permissions", help="Comma-separated list of permissions")
    parser.add_argument("--resource", help="Resource for authorization")
    parser.add_argument("--action", help="Action for authorization")
    parser.add_argument("--data", help="Data to encrypt/decrypt")
    parser.add_argument("--rules", help="JSON string of policy rules")
    
    args = parser.parse_args()
    
    security_manager = AdvancedSecurityManager()
    
    print("🔒 BUL Advanced Security Manager")
    print("=" * 40)
    
    if args.create_policy:
        if not all([args.name, args.description, args.security_level]):
            print("❌ Error: --name, --description, and --security-level are required")
            return 1
        
        rules = []
        if args.rules:
            try:
                rules = json.loads(args.rules)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --rules")
                return 1
        
        policy = security_manager.create_policy(
            policy_id=args.create_policy,
            name=args.name,
            description=args.description,
            security_level=SecurityLevel(args.security_level),
            rules=rules
        )
        print(f"✅ Created policy: {policy.name}")
    
    elif args.create_user:
        if not all([args.username, args.email, args.password]):
            print("❌ Error: --username, --email, and --password are required")
            return 1
        
        roles = args.roles.split(',') if args.roles else ['user']
        permissions = args.permissions.split(',') if args.permissions else []
        
        user = security_manager.create_user(
            user_id=args.create_user,
            username=args.username,
            email=args.email,
            password=args.password,
            roles=roles,
            permissions=permissions
        )
        print(f"✅ Created user: {user.username}")
    
    elif args.authenticate:
        if not args.password:
            print("❌ Error: --password is required for authentication")
            return 1
        
        result = security_manager.authenticate_user(args.authenticate, args.password)
        if result['success']:
            print(f"✅ Authentication successful")
            print(f"   User: {result['user']['username']}")
            print(f"   Roles: {', '.join(result['user']['roles'])}")
        else:
            print(f"❌ Authentication failed: {result['error']}")
    
    elif args.authorize:
        if not all([args.resource, args.action]):
            print("❌ Error: --resource and --action are required for authorization")
            return 1
        
        authorized = security_manager.authorize_user(args.authorize, args.resource, args.action)
        if authorized:
            print(f"✅ Authorization granted")
        else:
            print(f"❌ Authorization denied")
    
    elif args.encrypt:
        if not args.data:
            print("❌ Error: --data is required for encryption")
            return 1
        
        encrypted = security_manager.encrypt_data(args.data)
        print(f"✅ Data encrypted: {encrypted}")
    
    elif args.decrypt:
        if not args.data:
            print("❌ Error: --data is required for decryption")
            return 1
        
        decrypted = security_manager.decrypt_data(args.data)
        print(f"✅ Data decrypted: {decrypted}")
    
    elif args.validate_password:
        result = security_manager.validate_password(args.validate_password)
        if result['valid']:
            print(f"✅ Password is valid")
        else:
            print(f"❌ Password validation failed:")
            for error in result['errors']:
                print(f"   - {error}")
    
    elif args.list_users:
        users = security_manager.users
        if users:
            print(f"\n👥 Security Users ({len(users)}):")
            print("-" * 60)
            for user_id, user in users.items():
                print(f"{user.username} ({user_id}):")
                print(f"  Email: {user.email}")
                print(f"  Roles: {', '.join(user.roles)}")
                print(f"  Last Login: {user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never'}")
                print(f"  Status: {'Locked' if user.locked_until and datetime.now() < user.locked_until else 'Active'}")
                print()
        else:
            print("No users found.")
    
    elif args.list_policies:
        policies = security_manager.policies
        if policies:
            print(f"\n🔒 Security Policies ({len(policies)}):")
            print("-" * 60)
            for policy_id, policy in policies.items():
                print(f"{policy.name} ({policy_id}):")
                print(f"  Level: {policy.security_level.value}")
                print(f"  Rules: {len(policy.rules)}")
                print(f"  Enabled: {policy.enabled}")
                print()
        else:
            print("No policies found.")
    
    elif args.stats:
        stats = security_manager.get_security_stats()
        print(f"\n📊 Security Statistics:")
        print(f"   Total Users: {stats['total_users']}")
        print(f"   Total Policies: {stats['total_policies']}")
        print(f"   Total Events: {stats['total_events']}")
        print(f"   Successful Logins: {stats['successful_logins']}")
        print(f"   Failed Logins: {stats['failed_logins']}")
        print(f"   Locked Users: {stats['locked_users']}")
        print(f"   Events by Severity:")
        for severity, count in stats['events_by_severity'].items():
            print(f"     {severity.title()}: {count}")
    
    elif args.report:
        report = security_manager.generate_security_report()
        print(report)
        
        # Save report
        report_file = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        stats = security_manager.get_security_stats()
        print(f"👥 Users: {stats['total_users']}")
        print(f"🔒 Policies: {stats['total_policies']}")
        print(f"📊 Events: {stats['total_events']}")
        print(f"🔓 Locked Users: {stats['locked_users']}")
        print(f"\n💡 Use --create-user to create a new user")
        print(f"💡 Use --authenticate to authenticate a user")
        print(f"💡 Use --list-users to see all users")
        print(f"💡 Use --report to generate security report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
