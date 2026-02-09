"""
Enterprise Features for Ultimate Opus Clip

Advanced enterprise-grade features including team management,
role-based access control, billing, analytics, and compliance.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import time
import json
import hashlib
import secrets
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor
import yaml

logger = structlog.get_logger("enterprise_features")

class UserRole(Enum):
    """User roles in the enterprise system."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    """System permissions."""
    CREATE_WORKFLOW = "create_workflow"
    EDIT_WORKFLOW = "edit_workflow"
    DELETE_WORKFLOW = "delete_workflow"
    EXECUTE_WORKFLOW = "execute_workflow"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    MANAGE_BILLING = "manage_billing"
    MANAGE_SETTINGS = "manage_settings"
    EXPORT_DATA = "export_data"
    API_ACCESS = "api_access"

class BillingPlan(Enum):
    """Billing plans."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class OrganizationStatus(Enum):
    """Organization status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    TRIAL = "trial"

@dataclass
class User:
    """Enterprise user."""
    user_id: str
    email: str
    username: str
    full_name: str
    role: UserRole
    organization_id: str
    permissions: List[Permission]
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    profile_data: Dict[str, Any] = None

@dataclass
class Organization:
    """Enterprise organization."""
    organization_id: str
    name: str
    domain: str
    status: OrganizationStatus
    billing_plan: BillingPlan
    created_at: float
    settings: Dict[str, Any] = None
    limits: Dict[str, int] = None
    usage: Dict[str, int] = None

@dataclass
class Team:
    """Team within an organization."""
    team_id: str
    name: str
    description: str
    organization_id: str
    members: List[str]  # User IDs
    created_at: float
    settings: Dict[str, Any] = None

@dataclass
class BillingInfo:
    """Billing information."""
    organization_id: str
    plan: BillingPlan
    billing_cycle: str  # monthly, yearly
    amount: float
    currency: str
    next_billing_date: float
    payment_method: str
    status: str  # active, past_due, cancelled

@dataclass
class UsageMetrics:
    """Usage metrics for billing."""
    organization_id: str
    period_start: float
    period_end: float
    video_processing_minutes: int
    storage_gb: float
    api_calls: int
    workflow_executions: int
    users_active: int

@dataclass
class AuditLog:
    """Audit log entry."""
    log_id: str
    user_id: str
    organization_id: str
    action: str
    resource_type: str
    resource_id: str
    timestamp: float
    ip_address: str
    user_agent: str
    details: Dict[str, Any] = None

class UserManagement:
    """User management system."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.organizations: Dict[str, Organization] = {}
        self.teams: Dict[str, Team] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("User Management initialized")
    
    def create_organization(self, name: str, domain: str, admin_user: Dict[str, Any]) -> Tuple[str, str]:
        """Create a new organization and admin user."""
        try:
            # Create organization
            org_id = str(uuid.uuid4())
            organization = Organization(
                organization_id=org_id,
                name=name,
                domain=domain,
                status=OrganizationStatus.TRIAL,
                billing_plan=BillingPlan.FREE,
                created_at=time.time(),
                limits={
                    "max_users": 5,
                    "max_storage_gb": 10,
                    "max_processing_minutes": 100
                },
                usage={
                    "users": 0,
                    "storage_gb": 0.0,
                    "processing_minutes": 0
                }
            )
            self.organizations[org_id] = organization
            
            # Create admin user
            user_id = str(uuid.uuid4())
            user = User(
                user_id=user_id,
                email=admin_user['email'],
                username=admin_user['username'],
                full_name=admin_user['full_name'],
                role=UserRole.SUPER_ADMIN,
                organization_id=org_id,
                permissions=list(Permission),  # All permissions
                created_at=time.time()
            )
            self.users[user_id] = user
            
            # Update organization usage
            organization.usage['users'] = 1
            
            logger.info(f"Created organization: {name} ({org_id}) with admin user: {user_id}")
            return org_id, user_id
            
        except Exception as e:
            logger.error(f"Error creating organization: {e}")
            raise
    
    def create_user(self, user_data: Dict[str, Any], organization_id: str, role: UserRole = UserRole.VIEWER) -> str:
        """Create a new user."""
        try:
            # Check organization exists
            if organization_id not in self.organizations:
                raise ValueError("Organization not found")
            
            organization = self.organizations[organization_id]
            
            # Check user limits
            if organization.usage['users'] >= organization.limits['max_users']:
                raise ValueError("User limit exceeded")
            
            # Create user
            user_id = str(uuid.uuid4())
            user = User(
                user_id=user_id,
                email=user_data['email'],
                username=user_data['username'],
                full_name=user_data['full_name'],
                role=role,
                organization_id=organization_id,
                permissions=self._get_default_permissions(role),
                created_at=time.time()
            )
            self.users[user_id] = user
            
            # Update organization usage
            organization.usage['users'] += 1
            
            logger.info(f"Created user: {user_data['username']} ({user_id})")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user (simplified - in production, use proper auth)."""
        try:
            # Find user by email
            user = None
            for u in self.users.values():
                if u.email == email:
                    user = u
                    break
            
            if not user or not user.is_active:
                return None
            
            # Update last login
            user.last_login = time.time()
            
            # Create session
            session_id = str(uuid.uuid4())
            self.user_sessions[session_id] = {
                "user_id": user.user_id,
                "created_at": time.time(),
                "expires_at": time.time() + (24 * 60 * 60)  # 24 hours
            }
            
            logger.info(f"User authenticated: {email}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        try:
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return False
            
            return permission in user.permissions
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    def create_team(self, name: str, description: str, organization_id: str, creator_id: str) -> str:
        """Create a new team."""
        try:
            # Check permissions
            if not self.check_permission(creator_id, Permission.MANAGE_USERS):
                raise ValueError("Insufficient permissions")
            
            team_id = str(uuid.uuid4())
            team = Team(
                team_id=team_id,
                name=name,
                description=description,
                organization_id=organization_id,
                members=[creator_id],
                created_at=time.time()
            )
            self.teams[team_id] = team
            
            logger.info(f"Created team: {name} ({team_id})")
            return team_id
            
        except Exception as e:
            logger.error(f"Error creating team: {e}")
            raise
    
    def add_user_to_team(self, team_id: str, user_id: str, requester_id: str) -> bool:
        """Add user to team."""
        try:
            # Check permissions
            if not self.check_permission(requester_id, Permission.MANAGE_USERS):
                return False
            
            if team_id not in self.teams:
                return False
            
            team = self.teams[team_id]
            if user_id not in team.members:
                team.members.append(user_id)
            
            logger.info(f"Added user {user_id} to team {team_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding user to team: {e}")
            return False
    
    def _get_default_permissions(self, role: UserRole) -> List[Permission]:
        """Get default permissions for role."""
        permission_map = {
            UserRole.SUPER_ADMIN: list(Permission),
            UserRole.ADMIN: [
                Permission.CREATE_WORKFLOW,
                Permission.EDIT_WORKFLOW,
                Permission.DELETE_WORKFLOW,
                Permission.EXECUTE_WORKFLOW,
                Permission.VIEW_ANALYTICS,
                Permission.MANAGE_USERS,
                Permission.EXPORT_DATA,
                Permission.API_ACCESS
            ],
            UserRole.MANAGER: [
                Permission.CREATE_WORKFLOW,
                Permission.EDIT_WORKFLOW,
                Permission.EXECUTE_WORKFLOW,
                Permission.VIEW_ANALYTICS,
                Permission.EXPORT_DATA,
                Permission.API_ACCESS
            ],
            UserRole.EDITOR: [
                Permission.CREATE_WORKFLOW,
                Permission.EDIT_WORKFLOW,
                Permission.EXECUTE_WORKFLOW,
                Permission.API_ACCESS
            ],
            UserRole.VIEWER: [
                Permission.VIEW_ANALYTICS
            ],
            UserRole.GUEST: []
        }
        
        return permission_map.get(role, [])

class BillingSystem:
    """Billing and subscription management."""
    
    def __init__(self):
        self.billing_info: Dict[str, BillingInfo] = {}
        self.usage_metrics: Dict[str, List[UsageMetrics]] = {}
        self.plan_limits = {
            BillingPlan.FREE: {
                "max_users": 5,
                "max_storage_gb": 10,
                "max_processing_minutes": 100,
                "max_api_calls": 1000,
                "price": 0.0
            },
            BillingPlan.STARTER: {
                "max_users": 25,
                "max_storage_gb": 100,
                "max_processing_minutes": 1000,
                "max_api_calls": 10000,
                "price": 29.0
            },
            BillingPlan.PROFESSIONAL: {
                "max_users": 100,
                "max_storage_gb": 500,
                "max_processing_minutes": 5000,
                "max_api_calls": 50000,
                "price": 99.0
            },
            BillingPlan.ENTERPRISE: {
                "max_users": -1,  # Unlimited
                "max_storage_gb": -1,
                "max_processing_minutes": -1,
                "max_api_calls": -1,
                "price": 299.0
            }
        }
        
        logger.info("Billing System initialized")
    
    def create_billing_info(self, organization_id: str, plan: BillingPlan, payment_method: str) -> str:
        """Create billing information for organization."""
        try:
            billing_id = str(uuid.uuid4())
            billing_info = BillingInfo(
                organization_id=organization_id,
                plan=plan,
                billing_cycle="monthly",
                amount=self.plan_limits[plan]["price"],
                currency="USD",
                next_billing_date=time.time() + (30 * 24 * 60 * 60),  # 30 days
                payment_method=payment_method,
                status="active"
            )
            
            self.billing_info[organization_id] = billing_info
            
            logger.info(f"Created billing info for organization {organization_id}: {plan.value}")
            return billing_id
            
        except Exception as e:
            logger.error(f"Error creating billing info: {e}")
            raise
    
    def record_usage(self, organization_id: str, usage_type: str, amount: float):
        """Record usage for billing."""
        try:
            if organization_id not in self.usage_metrics:
                self.usage_metrics[organization_id] = []
            
            # Get current period
            current_time = time.time()
            period_start = current_time - (30 * 24 * 60 * 60)  # 30 days ago
            
            # Find or create current period metrics
            current_metrics = None
            for metrics in self.usage_metrics[organization_id]:
                if metrics.period_start <= current_time <= metrics.period_end:
                    current_metrics = metrics
                    break
            
            if not current_metrics:
                current_metrics = UsageMetrics(
                    organization_id=organization_id,
                    period_start=period_start,
                    period_end=current_time + (30 * 24 * 60 * 60),
                    video_processing_minutes=0,
                    storage_gb=0.0,
                    api_calls=0,
                    workflow_executions=0,
                    users_active=0
                )
                self.usage_metrics[organization_id].append(current_metrics)
            
            # Update usage
            if usage_type == "video_processing_minutes":
                current_metrics.video_processing_minutes += int(amount)
            elif usage_type == "storage_gb":
                current_metrics.storage_gb += amount
            elif usage_type == "api_calls":
                current_metrics.api_calls += int(amount)
            elif usage_type == "workflow_executions":
                current_metrics.workflow_executions += int(amount)
            
            logger.info(f"Recorded usage for {organization_id}: {usage_type} = {amount}")
            
        except Exception as e:
            logger.error(f"Error recording usage: {e}")
    
    def check_usage_limits(self, organization_id: str) -> Dict[str, Any]:
        """Check if organization is within usage limits."""
        try:
            if organization_id not in self.billing_info:
                return {"within_limits": True, "warnings": []}
            
            billing_info = self.billing_info[organization_id]
            plan_limits = self.plan_limits[billing_info.plan]
            
            # Get current usage
            current_usage = self._get_current_usage(organization_id)
            
            warnings = []
            within_limits = True
            
            for limit_type, limit_value in plan_limits.items():
                if limit_type == "price":
                    continue
                
                current_value = current_usage.get(limit_type, 0)
                
                if limit_value == -1:  # Unlimited
                    continue
                
                if current_value >= limit_value:
                    warnings.append(f"{limit_type} limit exceeded: {current_value}/{limit_value}")
                    within_limits = False
                elif current_value >= limit_value * 0.8:  # 80% warning
                    warnings.append(f"{limit_type} approaching limit: {current_value}/{limit_value}")
            
            return {
                "within_limits": within_limits,
                "warnings": warnings,
                "current_usage": current_usage,
                "limits": plan_limits
            }
            
        except Exception as e:
            logger.error(f"Error checking usage limits: {e}")
            return {"within_limits": False, "warnings": ["Error checking limits"]}
    
    def _get_current_usage(self, organization_id: str) -> Dict[str, Any]:
        """Get current usage for organization."""
        if organization_id not in self.usage_metrics:
            return {}
        
        # Get most recent metrics
        metrics_list = self.usage_metrics[organization_id]
        if not metrics_list:
            return {}
        
        latest_metrics = max(metrics_list, key=lambda x: x.period_start)
        
        return {
            "max_users": latest_metrics.users_active,
            "max_storage_gb": latest_metrics.storage_gb,
            "max_processing_minutes": latest_metrics.video_processing_minutes,
            "max_api_calls": latest_metrics.api_calls
        }

class AuditLogger:
    """Audit logging system."""
    
    def __init__(self):
        self.audit_logs: List[AuditLog] = []
        self.max_logs = 100000  # Keep last 100k logs
        
        logger.info("Audit Logger initialized")
    
    def log_action(self, user_id: str, organization_id: str, action: str, 
                   resource_type: str, resource_id: str, ip_address: str = "", 
                   user_agent: str = "", details: Dict[str, Any] = None):
        """Log an audit action."""
        try:
            log_entry = AuditLog(
                log_id=str(uuid.uuid4()),
                user_id=user_id,
                organization_id=organization_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                timestamp=time.time(),
                ip_address=ip_address,
                user_agent=user_agent,
                details=details or {}
            )
            
            self.audit_logs.append(log_entry)
            
            # Keep only recent logs
            if len(self.audit_logs) > self.max_logs:
                self.audit_logs = self.audit_logs[-self.max_logs:]
            
            logger.info(f"Audit log: {user_id} performed {action} on {resource_type}:{resource_id}")
            
        except Exception as e:
            logger.error(f"Error logging audit action: {e}")
    
    def get_audit_logs(self, organization_id: str, user_id: str = None, 
                      action: str = None, start_time: float = None, 
                      end_time: float = None) -> List[AuditLog]:
        """Get audit logs with filters."""
        try:
            logs = self.audit_logs.copy()
            
            # Filter by organization
            logs = [log for log in logs if log.organization_id == organization_id]
            
            # Filter by user
            if user_id:
                logs = [log for log in logs if log.user_id == user_id]
            
            # Filter by action
            if action:
                logs = [log for log in logs if log.action == action]
            
            # Filter by time range
            if start_time:
                logs = [log for log in logs if log.timestamp >= start_time]
            
            if end_time:
                logs = [log for log in logs if log.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return []

class EnterpriseFeatures:
    """Main enterprise features orchestrator."""
    
    def __init__(self):
        self.user_management = UserManagement()
        self.billing_system = BillingSystem()
        self.audit_logger = AuditLogger()
        
        logger.info("Enterprise Features initialized")
    
    def create_organization_with_admin(self, org_data: Dict[str, Any], admin_data: Dict[str, Any]) -> Dict[str, str]:
        """Create organization with admin user."""
        try:
            org_id, user_id = self.user_management.create_organization(
                org_data['name'],
                org_data['domain'],
                admin_data
            )
            
            # Create billing info
            self.billing_system.create_billing_info(org_id, BillingPlan.FREE, "none")
            
            # Log creation
            self.audit_logger.log_action(
                user_id, org_id, "create_organization", "organization", org_id
            )
            
            return {
                "organization_id": org_id,
                "admin_user_id": user_id,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Error creating organization: {e}")
            raise
    
    def get_organization_dashboard(self, organization_id: str, user_id: str) -> Dict[str, Any]:
        """Get organization dashboard data."""
        try:
            # Check permissions
            if not self.user_management.check_permission(user_id, Permission.VIEW_ANALYTICS):
                raise ValueError("Insufficient permissions")
            
            # Get organization
            organization = self.user_management.organizations.get(organization_id)
            if not organization:
                raise ValueError("Organization not found")
            
            # Get usage limits
            usage_info = self.billing_system.check_usage_limits(organization_id)
            
            # Get recent audit logs
            recent_logs = self.audit_logger.get_audit_logs(
                organization_id, 
                start_time=time.time() - (7 * 24 * 60 * 60)  # Last 7 days
            )
            
            # Get team information
            teams = [team for team in self.user_management.teams.values() 
                    if team.organization_id == organization_id]
            
            return {
                "organization": asdict(organization),
                "usage_info": usage_info,
                "recent_activity": [asdict(log) for log in recent_logs[:10]],
                "teams": [asdict(team) for team in teams],
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            raise
    
    def upgrade_plan(self, organization_id: str, new_plan: BillingPlan, user_id: str) -> bool:
        """Upgrade organization billing plan."""
        try:
            # Check permissions
            if not self.user_management.check_permission(user_id, Permission.MANAGE_BILLING):
                return False
            
            # Update billing info
            if organization_id in self.billing_system.billing_info:
                billing_info = self.billing_system.billing_info[organization_id]
                billing_info.plan = new_plan
                billing_info.amount = self.billing_system.plan_limits[new_plan]["price"]
                
                # Log upgrade
                self.audit_logger.log_action(
                    user_id, organization_id, "upgrade_plan", 
                    "billing", organization_id, 
                    details={"old_plan": billing_info.plan.value, "new_plan": new_plan.value}
                )
                
                logger.info(f"Upgraded organization {organization_id} to {new_plan.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error upgrading plan: {e}")
            return False

# Global enterprise features instance
_global_enterprise_features: Optional[EnterpriseFeatures] = None

def get_enterprise_features() -> EnterpriseFeatures:
    """Get the global enterprise features instance."""
    global _global_enterprise_features
    if _global_enterprise_features is None:
        _global_enterprise_features = EnterpriseFeatures()
    return _global_enterprise_features

def create_organization(org_name: str, domain: str, admin_email: str, admin_name: str) -> Dict[str, str]:
    """Create a new organization with admin user."""
    enterprise = get_enterprise_features()
    
    org_data = {
        "name": org_name,
        "domain": domain
    }
    
    admin_data = {
        "email": admin_email,
        "username": admin_email.split("@")[0],
        "full_name": admin_name
    }
    
    return enterprise.create_organization_with_admin(org_data, admin_data)


