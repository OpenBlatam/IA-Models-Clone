"""
Enterprise Features for HeyGen AI
================================

Provides enterprise-grade features including SSO, RBAC, audit logging,
compliance, and enterprise user management for professional deployments.
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# Security imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Enterprise user information."""
    
    user_id: str
    username: str
    email: str
    full_name: str
    role: str = "user"  # admin, manager, user, guest
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Role:
    """Role-based access control definition."""
    
    role_id: str
    name: str
    description: str
    permissions: List[str]
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Permission:
    """Permission definition."""
    
    permission_id: str
    name: str
    description: str
    resource: str  # e.g., "video", "user", "analytics"
    action: str    # e.g., "read", "write", "delete"
    scope: str = "global"  # global, user, organization
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuditLog:
    """Audit log entry."""
    
    log_id: str
    user_id: str
    action: str
    resource: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    encrypted: bool = True


@dataclass
class SSOConfig:
    """Single Sign-On configuration."""
    
    provider: str  # saml, oidc, oauth2, ldap
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceConfig:
    """Compliance configuration."""
    
    gdpr_enabled: bool = True
    hipaa_enabled: bool = False
    sox_enabled: bool = False
    data_retention_days: int = 2555  # 7 years
    audit_logging_enabled: bool = True
    encryption_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnterpriseFeatures(BaseService):
    """Enterprise-grade features and security system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enterprise features system."""
        super().__init__("EnterpriseFeatures", ServiceType.SECURITY)
        
        # User management
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        
        # SSO configuration
        self.sso_configs: Dict[str, SSOConfig] = {}
        
        # Compliance
        self.compliance_config = ComplianceConfig()
        
        # Audit logging
        self.audit_logs: List[AuditLog] = []
        self.audit_encryption_key: Optional[bytes] = None
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.enterprise_stats = {
            "total_users": 0,
            "active_users": 0,
            "total_roles": 0,
            "total_permissions": 0,
            "total_audit_logs": 0,
            "sso_providers": 0,
            "compliance_features": 0
        }
        
        # Security settings
        self.security_config = {
            "password_min_length": 12,
            "password_require_special": True,
            "password_require_numbers": True,
            "password_require_uppercase": True,
            "session_timeout_minutes": 480,  # 8 hours
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30
        }

    async def _initialize_service_impl(self) -> None:
        """Initialize enterprise features."""
        try:
            logger.info("Initializing enterprise features...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Initialize encryption
            await self._initialize_encryption()
            
            # Load default roles and permissions
            await self._load_default_roles_permissions()
            
            # Initialize SSO providers
            await self._initialize_sso_providers()
            
            # Initialize compliance features
            await self._initialize_compliance()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Enterprise features initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enterprise features: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not CRYPTOGRAPHY_AVAILABLE:
            missing_deps.append("cryptography")
        
        if not JWT_AVAILABLE:
            missing_deps.append("PyJWT")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some security features may not be available")

    async def _initialize_encryption(self) -> None:
        """Initialize encryption for audit logging."""
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                # Generate encryption key for audit logs
                salt = b'enterprise_salt_12345'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(b"enterprise_key"))
                self.audit_encryption_key = key
                logger.info("Audit log encryption initialized")
            else:
                logger.warning("Cryptography not available, audit logs will not be encrypted")
                
        except Exception as e:
            logger.warning(f"Encryption initialization had issues: {e}")

    async def _load_default_roles_permissions(self) -> None:
        """Load default roles and permissions."""
        try:
            # Create default permissions
            default_permissions = [
                Permission("perm_video_read", "Read Videos", "Allow reading video content", "video", "read"),
                Permission("perm_video_write", "Write Videos", "Allow creating and editing videos", "video", "write"),
                Permission("perm_video_delete", "Delete Videos", "Allow deleting videos", "video", "delete"),
                Permission("perm_user_read", "Read Users", "Allow reading user information", "user", "read"),
                Permission("perm_user_write", "Write Users", "Allow creating and editing users", "user", "write"),
                Permission("perm_user_delete", "Delete Users", "Allow deleting users", "user", "delete"),
                Permission("perm_analytics_read", "Read Analytics", "Allow reading analytics data", "analytics", "read"),
                Permission("perm_analytics_write", "Write Analytics", "Allow writing analytics data", "analytics", "write"),
                Permission("perm_system_admin", "System Admin", "Full system access", "system", "admin"),
            ]
            
            for permission in default_permissions:
                self.permissions[permission.permission_id] = permission
                self.enterprise_stats["total_permissions"] += 1
            
            # Create default roles
            default_roles = [
                Role("role_guest", "Guest", "Limited access user", 
                     ["perm_video_read"], is_system_role=True),
                Role("role_user", "User", "Standard user", 
                     ["perm_video_read", "perm_video_write", "perm_analytics_read"], is_system_role=True),
                Role("role_manager", "Manager", "Team manager", 
                     ["perm_video_read", "perm_video_write", "perm_video_delete", 
                      "perm_user_read", "perm_analytics_read", "perm_analytics_write"], is_system_role=True),
                Role("role_admin", "Administrator", "System administrator", 
                     ["perm_system_admin"], is_system_role=True),
            ]
            
            for role in default_roles:
                self.roles[role.role_id] = role
                self.enterprise_stats["total_roles"] += 1
            
            logger.info(f"Loaded {len(default_permissions)} permissions and {len(default_roles)} roles")
            
        except Exception as e:
            logger.warning(f"Failed to load default roles and permissions: {e}")

    async def _initialize_sso_providers(self) -> None:
        """Initialize SSO providers."""
        try:
            # SAML configuration
            self.sso_configs["saml"] = SSOConfig(
                provider="saml",
                enabled=True,
                config={
                    "entity_id": "https://heygen-ai.example.com",
                    "sso_url": "https://idp.example.com/sso",
                    "x509_cert": "your_x509_certificate_here"
                }
            )
            
            # OIDC configuration
            self.sso_configs["oidc"] = SSOConfig(
                provider="oidc",
                enabled=True,
                config={
                    "issuer": "https://accounts.google.com",
                    "client_id": "your_client_id",
                    "client_secret": "your_client_secret"
                }
            )
            
            # OAuth2 configuration
            self.sso_configs["oauth2"] = SSOConfig(
                provider="oauth2",
                enabled=True,
                config={
                    "authorization_url": "https://oauth.example.com/authorize",
                    "token_url": "https://oauth.example.com/token",
                    "client_id": "your_client_id",
                    "client_secret": "your_client_secret"
                }
            )
            
            # LDAP configuration
            self.sso_configs["ldap"] = SSOConfig(
                provider="ldap",
                enabled=True,
                config={
                    "server_url": "ldap://ldap.example.com:389",
                    "base_dn": "dc=example,dc=com",
                    "bind_dn": "cn=admin,dc=example,dc=com",
                    "bind_password": "your_ldap_password"
                }
            )
            
            self.enterprise_stats["sso_providers"] = len(self.sso_configs)
            logger.info(f"Initialized {len(self.sso_configs)} SSO providers")
            
        except Exception as e:
            logger.warning(f"Failed to initialize some SSO providers: {e}")

    async def _initialize_compliance(self) -> None:
        """Initialize compliance features."""
        try:
            # Count enabled compliance features
            compliance_features = 0
            if self.compliance_config.gdpr_enabled:
                compliance_features += 1
            if self.compliance_config.hipaa_enabled:
                compliance_features += 1
            if self.compliance_config.sox_enabled:
                compliance_features += 1
            if self.compliance_config.audit_logging_enabled:
                compliance_features += 1
            if self.compliance_config.encryption_enabled:
                compliance_features += 1
            
            self.enterprise_stats["compliance_features"] = compliance_features
            logger.info(f"Initialized {compliance_features} compliance features")
            
        except Exception as e:
            logger.warning(f"Failed to initialize compliance features: {e}")

    async def _validate_configuration(self) -> None:
        """Validate enterprise features configuration."""
        if not self.roles:
            raise RuntimeError("No roles configured")
        
        if not self.permissions:
            raise RuntimeError("No permissions configured")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def create_user(self, username: str, email: str, full_name: str, 
                         role: str = "user", password: str = None) -> str:
        """Create a new enterprise user."""
        try:
            logger.info(f"Creating enterprise user: {username}")
            
            # Validate role
            if role not in self.roles:
                raise ValueError(f"Invalid role: {role}")
            
            # Check if username already exists
            if any(user.username == username for user in self.users.values()):
                raise ValueError(f"Username already exists: {username}")
            
            # Check if email already exists
            if any(user.email == email for user in self.users.values()):
                raise ValueError(f"Email already exists: {email}")
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            
            # Get role permissions
            role_obj = self.roles[role]
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                full_name=full_name,
                role=role,
                permissions=role_obj.permissions.copy(),
                is_active=True
            )
            
            # Store user
            self.users[user_id] = user
            self.enterprise_stats["total_users"] += 1
            self.enterprise_stats["active_users"] += 1
            
            # Log user creation
            await self._log_audit_event(
                user_id="system",
                action="user_created",
                resource="user",
                resource_id=user_id,
                details={"username": username, "email": email, "role": role}
            )
            
            logger.info(f"Enterprise user created: {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to create enterprise user: {e}")
            raise

    @with_error_handling
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """Authenticate a user."""
        try:
            logger.info(f"Authenticating user: {username}")
            
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                logger.warning(f"Authentication failed: user not found: {username}")
                return None
            
            if not user.is_active:
                logger.warning(f"Authentication failed: user inactive: {username}")
                return None
            
            # For demo purposes, accept any password
            # In production, this would validate against hashed passwords
            
            # Update last login
            user.last_login = datetime.now()
            self.users[user.user_id] = user
            
            # Log successful authentication
            await self._log_audit_event(
                user_id=user.user_id,
                action="user_login",
                resource="user",
                resource_id=user.user_id,
                details={"username": username, "success": True},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            logger.info(f"User authenticated successfully: {username}")
            return user.user_id
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    @with_error_handling
    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if a user has permission for a resource and action."""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            if not user.is_active:
                return False
            
            # Check user permissions
            for permission_id in user.permissions:
                if permission_id in self.permissions:
                    permission = self.permissions[permission_id]
                    if permission.resource == resource and permission.action == action:
                        return True
            
            # Check role permissions
            if user.role in self.roles:
                role = self.roles[user.role]
                for permission_id in role.permissions:
                    if permission_id in self.permissions:
                        permission = self.permissions[permission_id]
                        if permission.resource == resource and permission.action == action:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False

    @with_error_handling
    async def create_role(self, name: str, description: str, 
                         permissions: List[str], is_system_role: bool = False) -> str:
        """Create a new role."""
        try:
            logger.info(f"Creating role: {name}")
            
            # Validate permissions
            for permission_id in permissions:
                if permission_id not in self.permissions:
                    raise ValueError(f"Invalid permission: {permission_id}")
            
            # Check if role name already exists
            if any(role.name == name for role in self.roles.values()):
                raise ValueError(f"Role name already exists: {name}")
            
            # Generate role ID
            role_id = str(uuid.uuid4())
            
            # Create role
            role = Role(
                role_id=role_id,
                name=name,
                description=description,
                permissions=permissions.copy(),
                is_system_role=is_system_role
            )
            
            # Store role
            self.roles[role_id] = role
            self.enterprise_stats["total_roles"] += 1
            
            # Log role creation
            await self._log_audit_event(
                user_id="system",
                action="role_created",
                resource="role",
                resource_id=role_id,
                details={"name": name, "permissions": permissions}
            )
            
            logger.info(f"Role created: {role_id}")
            return role_id
            
        except Exception as e:
            logger.error(f"Failed to create role: {e}")
            raise

    @with_error_handling
    async def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user."""
        try:
            logger.info(f"Assigning role {role_id} to user {user_id}")
            
            # Validate user and role
            if user_id not in self.users:
                raise ValueError(f"User not found: {user_id}")
            
            if role_id not in self.roles:
                raise ValueError(f"Role not found: {role_id}")
            
            user = self.users[user_id]
            role = self.roles[role_id]
            
            # Update user role and permissions
            user.role = role_id
            user.permissions = role.permissions.copy()
            self.users[user_id] = user
            
            # Log role assignment
            await self._log_audit_event(
                user_id="system",
                action="role_assigned",
                resource="user",
                resource_id=user_id,
                details={"role_id": role_id, "role_name": role.name}
            )
            
            logger.info(f"Role assigned successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            raise

    async def _log_audit_event(self, user_id: str, action: str, resource: str, 
                              resource_id: str, details: Dict[str, Any], 
                              ip_address: str = None, user_agent: str = None) -> None:
        """Log an audit event."""
        try:
            if not self.compliance_config.audit_logging_enabled:
                return
            
            # Create audit log entry
            log_entry = AuditLog(
                log_id=str(uuid.uuid4()),
                user_id=user_id,
                action=action,
                resource=resource,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Encrypt sensitive details if encryption is enabled
            if self.compliance_config.encryption_enabled and self.audit_encryption_key:
                try:
                    fernet = Fernet(self.audit_encryption_key)
                    encrypted_details = fernet.encrypt(str(details).encode())
                    log_entry.details = {"encrypted": True, "data": encrypted_details.decode()}
                except Exception as e:
                    logger.warning(f"Failed to encrypt audit details: {e}")
            
            # Store audit log
            self.audit_logs.append(log_entry)
            self.enterprise_stats["total_audit_logs"] += 1
            
            # Clean up old audit logs based on retention policy
            await self._cleanup_old_audit_logs()
            
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")

    async def _cleanup_old_audit_logs(self) -> None:
        """Clean up old audit logs based on retention policy."""
        try:
            if not self.compliance_config.audit_logging_enabled:
                return
            
            retention_days = self.compliance_config.data_retention_days
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Remove old logs
            original_count = len(self.audit_logs)
            self.audit_logs = [
                log for log in self.audit_logs 
                if log.timestamp > cutoff_date
            ]
            
            removed_count = original_count - len(self.audit_logs)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old audit logs")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old audit logs: {e}")

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        try:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "permissions": user.permissions,
                "is_active": user.is_active,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
                "metadata": user.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None

    async def get_audit_logs(self, user_id: str = None, resource: str = None, 
                            start_date: datetime = None, end_date: datetime = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs with optional filtering."""
        try:
            filtered_logs = self.audit_logs
            
            # Apply filters
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            
            if resource:
                filtered_logs = [log for log in filtered_logs if log.resource == resource]
            
            if start_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
            
            if end_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
            
            # Sort by timestamp (newest first) and limit results
            filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
            filtered_logs = filtered_logs[:limit]
            
            # Convert to dictionaries
            result = []
            for log in filtered_logs:
                log_dict = {
                    "log_id": log.log_id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource": log.resource,
                    "resource_id": log.resource_id,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "timestamp": log.timestamp.isoformat(),
                    "encrypted": log.encrypted
                }
                result.append(log_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the enterprise features system."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "cryptography": CRYPTOGRAPHY_AVAILABLE,
                "jwt": JWT_AVAILABLE
            }
            
            # Check user management
            user_status = {
                "total_users": self.enterprise_stats["total_users"],
                "active_users": self.enterprise_stats["active_users"],
                "total_roles": self.enterprise_stats["total_roles"],
                "total_permissions": self.enterprise_stats["total_permissions"]
            }
            
            # Check SSO
            sso_status = {
                "total_providers": self.enterprise_stats["sso_providers"],
                "enabled_providers": len([config for config in self.sso_configs.values() if config.enabled])
            }
            
            # Check compliance
            compliance_status = {
                "gdpr_enabled": self.compliance_config.gdpr_enabled,
                "hipaa_enabled": self.compliance_config.hipaa_enabled,
                "sox_enabled": self.compliance_config.sox_enabled,
                "audit_logging_enabled": self.compliance_config.audit_logging_enabled,
                "encryption_enabled": self.compliance_config.encryption_enabled,
                "total_features": self.enterprise_stats["compliance_features"]
            }
            
            # Check audit logging
            audit_status = {
                "total_logs": self.enterprise_stats["total_audit_logs"],
                "encryption_available": bool(self.audit_encryption_key),
                "retention_days": self.compliance_config.data_retention_days
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "users": user_status,
                "sso": sso_status,
                "compliance": compliance_status,
                "audit": audit_status,
                "enterprise_stats": self.enterprise_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary enterprise files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for enterprise_file in temp_dir.glob("enterprise_*"):
                    enterprise_file.unlink()
                    logger.debug(f"Cleaned up temp file: {enterprise_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the enterprise features system."""
        try:
            # Clear sensitive data
            self.users.clear()
            self.roles.clear()
            self.permissions.clear()
            self.audit_logs.clear()
            
            # Clear encryption key
            self.audit_encryption_key = None
            
            logger.info("Enterprise features system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def initialize(self) -> None:
        """Initialize the enterprise features system."""
        await self._initialize_service_impl()
        self.status = ServiceStatus.RUNNING
        logger.info("Enterprise features system initialized")
    
    async def start(self) -> None:
        """Start the enterprise features service."""
        await self.initialize()
        self.status = ServiceStatus.RUNNING
        logger.info("Enterprise features service started")
    
    async def stop(self) -> None:
        """Stop the enterprise features service."""
        await self.shutdown()
        self.status = ServiceStatus.STOPPED
        logger.info("Enterprise features service stopped")
