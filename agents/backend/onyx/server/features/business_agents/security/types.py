"""
Security Types and Definitions
==============================

Type definitions for security components.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import uuid

class AuthProvider(Enum):
    """Authentication provider types."""
    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    SAML = "saml"
    LDAP = "ldap"
    OAUTH2 = "oauth2"

class TokenType(Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id"
    API_KEY = "api_key"
    SESSION = "session"

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditEvent(Enum):
    """Security audit events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    API_ACCESS = "api_access"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityPolicy:
    """Security policy definition."""
    name: str
    description: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class AccessControl:
    """Access control definition."""
    resource: str
    action: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TokenInfo:
    """Token information."""
    token: str
    token_type: TokenType
    user_id: str
    expires_at: datetime
    issued_at: datetime = field(default_factory=datetime.now)
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OAuth2Config:
    """OAuth2 configuration."""
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    user_info_url: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=list)
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JWTConfig:
    """JWT configuration."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "business_agents_system"
    audience: str = "business_agents_users"

@dataclass
class SecurityMetrics:
    """Security metrics."""
    total_logins: int = 0
    failed_logins: int = 0
    active_sessions: int = 0
    blocked_ips: int = 0
    security_violations: int = 0
    last_security_scan: Optional[datetime] = None
    password_strength_score: float = 0.0
    two_factor_enabled_users: int = 0

@dataclass
class SecurityAlert:
    """Security alert."""
    alert_id: str
    alert_type: str
    severity: SecurityLevel
    title: str
    description: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PasswordPolicy:
    """Password policy configuration."""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    max_age_days: int = 90
    history_count: int = 5
    lockout_attempts: int = 5
    lockout_duration_minutes: int = 30

@dataclass
class TwoFactorConfig:
    """Two-factor authentication configuration."""
    enabled: bool = False
    method: str = "totp"  # totp, sms, email
    backup_codes_count: int = 10
    grace_period_days: int = 7
    required_for_admin: bool = True

@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    algorithm: str = "AES-256-GCM"
    key_size: int = 32
    iv_size: int = 12
    tag_size: int = 16
    key_rotation_days: int = 90
    backup_keys: bool = True

@dataclass
class SessionConfig:
    """Session configuration."""
    timeout_minutes: int = 1440  # 24 hours
    max_concurrent_sessions: int = 5
    require_https: bool = True
    secure_cookies: bool = True
    same_site: str = "strict"
    http_only: bool = True

@dataclass
class AuditConfig:
    """Audit logging configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    retention_days: int = 365
    include_request_body: bool = False
    include_response_body: bool = False
    sensitive_fields: List[str] = field(default_factory=lambda: ["password", "token", "secret"])
    alert_on_violations: bool = True

@dataclass
class SecurityConfig:
    """Overall security configuration."""
    password_policy: PasswordPolicy = field(default_factory=PasswordPolicy)
    two_factor: TwoFactorConfig = field(default_factory=TwoFactorConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    jwt: Optional[JWTConfig] = None
    oauth2_providers: Dict[str, OAuth2Config] = field(default_factory=dict)
    security_policies: List[SecurityPolicy] = field(default_factory=list)
    access_controls: List[AccessControl] = field(default_factory=list)
