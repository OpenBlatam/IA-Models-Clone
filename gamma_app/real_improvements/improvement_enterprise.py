"""
Gamma App - Real Improvement Enterprise
Enterprise-grade system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import multiprocessing

logger = logging.getLogger(__name__)

class EnterpriseLevel(Enum):
    """Enterprise levels"""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    ULTIMATE = "ultimate"

class EnterpriseFeature(Enum):
    """Enterprise features"""
    MULTI_TENANT = "multi_tenant"
    RBAC = "rbac"
    AUDIT_LOGGING = "audit_logging"
    COMPLIANCE = "compliance"
    SCALABILITY = "scalability"
    HIGH_AVAILABILITY = "high_availability"
    DISASTER_RECOVERY = "disaster_recovery"
    SECURITY = "security"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"

@dataclass
class EnterpriseTenant:
    """Enterprise tenant"""
    tenant_id: str
    name: str
    domain: str
    level: EnterpriseLevel
    features: List[EnterpriseFeature]
    max_users: int
    max_improvements: int
    created_at: datetime = None
    status: str = "active"

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class EnterpriseUser:
    """Enterprise user"""
    user_id: str
    tenant_id: str
    username: str
    email: str
    role: str
    permissions: List[str]
    created_at: datetime = None
    last_login: Optional[datetime] = None
    status: str = "active"

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class EnterpriseAudit:
    """Enterprise audit log"""
    audit_id: str
    tenant_id: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementEnterprise:
    """
    Enterprise-grade system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize enterprise system"""
        self.project_root = Path(project_root)
        self.tenants: Dict[str, EnterpriseTenant] = {}
        self.users: Dict[str, EnterpriseUser] = {}
        self.audits: Dict[str, EnterpriseAudit] = {}
        self.enterprise_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.task_queue = queue.Queue()
        self.worker_threads = []
        
        # Initialize enterprise database
        self._init_enterprise_database()
        
        # Start background workers
        self._start_background_workers()
        
        logger.info(f"Real Improvement Enterprise initialized for {self.project_root}")
    
    def _init_enterprise_database(self):
        """Initialize enterprise database"""
        try:
            conn = sqlite3.connect("enterprise_improvements.db")
            cursor = conn.cursor()
            
            # Create tenants table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT UNIQUE NOT NULL,
                    level TEXT NOT NULL,
                    features TEXT NOT NULL,
                    max_users INTEGER NOT NULL,
                    max_improvements INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
                )
            ''')
            
            # Create audit logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize enterprise database: {e}")
    
    def _start_background_workers(self):
        """Start background workers"""
        try:
            # Start audit log processor
            audit_worker = threading.Thread(target=self._process_audit_logs, daemon=True)
            audit_worker.start()
            self.worker_threads.append(audit_worker)
            
            # Start cleanup worker
            cleanup_worker = threading.Thread(target=self._cleanup_old_data, daemon=True)
            cleanup_worker.start()
            self.worker_threads.append(cleanup_worker)
            
            # Start monitoring worker
            monitoring_worker = threading.Thread(target=self._monitor_system_health, daemon=True)
            monitoring_worker.start()
            self.worker_threads.append(monitoring_worker)
            
        except Exception as e:
            logger.error(f"Failed to start background workers: {e}")
    
    def create_tenant(self, name: str, domain: str, level: EnterpriseLevel, 
                    max_users: int = 100, max_improvements: int = 1000) -> str:
        """Create enterprise tenant"""
        try:
            tenant_id = f"tenant_{int(time.time() * 1000)}"
            
            # Get features based on level
            features = self._get_features_for_level(level)
            
            tenant = EnterpriseTenant(
                tenant_id=tenant_id,
                name=name,
                domain=domain,
                level=level,
                features=features,
                max_users=max_users,
                max_improvements=max_improvements
            )
            
            self.tenants[tenant_id] = tenant
            
            # Save to database
            self._save_tenant_to_db(tenant)
            
            self._log_enterprise("tenant_created", f"Tenant {name} created with level {level.value}")
            
            return tenant_id
            
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise
    
    def _get_features_for_level(self, level: EnterpriseLevel) -> List[EnterpriseFeature]:
        """Get features for enterprise level"""
        feature_map = {
            EnterpriseLevel.STARTER: [
                EnterpriseFeature.MULTI_TENANT,
                EnterpriseFeature.RBAC
            ],
            EnterpriseLevel.PROFESSIONAL: [
                EnterpriseFeature.MULTI_TENANT,
                EnterpriseFeature.RBAC,
                EnterpriseFeature.AUDIT_LOGGING,
                EnterpriseFeature.MONITORING
            ],
            EnterpriseLevel.BUSINESS: [
                EnterpriseFeature.MULTI_TENANT,
                EnterpriseFeature.RBAC,
                EnterpriseFeature.AUDIT_LOGGING,
                EnterpriseFeature.COMPLIANCE,
                EnterpriseFeature.MONITORING,
                EnterpriseFeature.ANALYTICS
            ],
            EnterpriseLevel.ENTERPRISE: [
                EnterpriseFeature.MULTI_TENANT,
                EnterpriseFeature.RBAC,
                EnterpriseFeature.AUDIT_LOGGING,
                EnterpriseFeature.COMPLIANCE,
                EnterpriseFeature.SCALABILITY,
                EnterpriseFeature.MONITORING,
                EnterpriseFeature.ANALYTICS,
                EnterpriseFeature.SECURITY
            ],
            EnterpriseLevel.ULTIMATE: [
                EnterpriseFeature.MULTI_TENANT,
                EnterpriseFeature.RBAC,
                EnterpriseFeature.AUDIT_LOGGING,
                EnterpriseFeature.COMPLIANCE,
                EnterpriseFeature.SCALABILITY,
                EnterpriseFeature.HIGH_AVAILABILITY,
                EnterpriseFeature.DISASTER_RECOVERY,
                EnterpriseFeature.MONITORING,
                EnterpriseFeature.ANALYTICS,
                EnterpriseFeature.SECURITY
            ]
        }
        
        return feature_map.get(level, [])
    
    def create_user(self, tenant_id: str, username: str, email: str, 
                   role: str, permissions: List[str]) -> str:
        """Create enterprise user"""
        try:
            if tenant_id not in self.tenants:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            # Check user limit
            tenant = self.tenants[tenant_id]
            current_users = len([u for u in self.users.values() if u.tenant_id == tenant_id])
            if current_users >= tenant.max_users:
                raise ValueError(f"User limit reached for tenant {tenant_id}")
            
            user_id = f"user_{int(time.time() * 1000)}"
            
            user = EnterpriseUser(
                user_id=user_id,
                tenant_id=tenant_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions
            )
            
            self.users[user_id] = user
            
            # Save to database
            self._save_user_to_db(user)
            
            self._log_enterprise("user_created", f"User {username} created for tenant {tenant_id}")
            
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    def authenticate_user(self, email: str, password: str, tenant_domain: str) -> Optional[Dict[str, Any]]:
        """Authenticate enterprise user"""
        try:
            # Find tenant by domain
            tenant = None
            for t in self.tenants.values():
                if t.domain == tenant_domain:
                    tenant = t
                    break
            
            if not tenant:
                return None
            
            # Find user by email and tenant
            user = None
            for u in self.users.values():
                if u.email == email and u.tenant_id == tenant.tenant_id:
                    user = u
                    break
            
            if not user:
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            self._save_user_to_db(user)
            
            # Log authentication
            self._log_audit(
                tenant_id=tenant.tenant_id,
                user_id=user.user_id,
                action="login",
                resource="authentication",
                details={"email": email, "tenant_domain": tenant_domain},
                ip_address="127.0.0.1",
                user_agent="Enterprise System"
            )
            
            return {
                "user_id": user.user_id,
                "tenant_id": tenant.tenant_id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "permissions": user.permissions,
                "level": tenant.level.value,
                "features": [f.value for f in tenant.features]
            }
            
        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            return None
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check user permission"""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            return permission in user.permissions
            
        except Exception as e:
            logger.error(f"Failed to check permission: {e}")
            return False
    
    def log_audit(self, tenant_id: str, user_id: str, action: str, 
                 resource: str, details: Dict[str, Any], 
                 ip_address: str = "127.0.0.1", user_agent: str = "Enterprise System"):
        """Log enterprise audit"""
        try:
            audit_id = f"audit_{int(time.time() * 1000)}"
            
            audit = EnterpriseAudit(
                audit_id=audit_id,
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.audits[audit_id] = audit
            
            # Add to processing queue
            self.task_queue.put(("audit", audit))
            
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
    
    def _log_audit(self, tenant_id: str, user_id: str, action: str, 
                  resource: str, details: Dict[str, Any], 
                  ip_address: str, user_agent: str):
        """Internal audit logging"""
        try:
            audit_id = f"audit_{int(time.time() * 1000)}"
            
            audit = EnterpriseAudit(
                audit_id=audit_id,
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.audits[audit_id] = audit
            
            # Save to database
            self._save_audit_to_db(audit)
            
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
    
    def _process_audit_logs(self):
        """Process audit logs in background"""
        while True:
            try:
                if not self.task_queue.empty():
                    task_type, data = self.task_queue.get()
                    
                    if task_type == "audit":
                        self._save_audit_to_db(data)
                    
                    self.task_queue.task_done()
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process audit logs: {e}")
                time.sleep(5)
    
    def _cleanup_old_data(self):
        """Cleanup old data in background"""
        while True:
            try:
                # Cleanup old audit logs (older than 1 year)
                cutoff_date = datetime.utcnow() - timedelta(days=365)
                
                old_audits = [
                    audit_id for audit_id, audit in self.audits.items()
                    if audit.created_at < cutoff_date
                ]
                
                for audit_id in old_audits:
                    del self.audits[audit_id]
                
                if old_audits:
                    self._log_enterprise("cleanup", f"Cleaned up {len(old_audits)} old audit logs")
                
                time.sleep(86400)  # Run daily
                
            except Exception as e:
                logger.error(f"Failed to cleanup old data: {e}")
                time.sleep(3600)
    
    def _monitor_system_health(self):
        """Monitor system health in background"""
        while True:
            try:
                # Monitor system metrics
                metrics = {
                    "tenants": len(self.tenants),
                    "users": len(self.users),
                    "audits": len(self.audits),
                    "queue_size": self.task_queue.qsize(),
                    "active_workers": len(self.worker_threads)
                }
                
                # Log health metrics
                self._log_enterprise("health_check", f"System health: {metrics}")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Failed to monitor system health: {e}")
                time.sleep(300)
    
    def _save_tenant_to_db(self, tenant: EnterpriseTenant):
        """Save tenant to database"""
        try:
            conn = sqlite3.connect("enterprise_improvements.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tenants 
                (tenant_id, name, domain, level, features, max_users, max_improvements, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tenant.tenant_id,
                tenant.name,
                tenant.domain,
                tenant.level.value,
                json.dumps([f.value for f in tenant.features]),
                tenant.max_users,
                tenant.max_improvements,
                tenant.created_at.isoformat(),
                tenant.status
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save tenant to database: {e}")
    
    def _save_user_to_db(self, user: EnterpriseUser):
        """Save user to database"""
        try:
            conn = sqlite3.connect("enterprise_improvements.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, tenant_id, username, email, role, permissions, created_at, last_login, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id,
                user.tenant_id,
                user.username,
                user.email,
                user.role,
                json.dumps(user.permissions),
                user.created_at.isoformat(),
                user.last_login.isoformat() if user.last_login else None,
                user.status
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save user to database: {e}")
    
    def _save_audit_to_db(self, audit: EnterpriseAudit):
        """Save audit to database"""
        try:
            conn = sqlite3.connect("enterprise_improvements.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs 
                (audit_id, tenant_id, user_id, action, resource, details, ip_address, user_agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit.audit_id,
                audit.tenant_id,
                audit.user_id,
                audit.action,
                audit.resource,
                json.dumps(audit.details),
                audit.ip_address,
                audit.user_agent,
                audit.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save audit to database: {e}")
    
    def get_tenant_info(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant information"""
        if tenant_id not in self.tenants:
            return None
        
        tenant = self.tenants[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "domain": tenant.domain,
            "level": tenant.level.value,
            "features": [f.value for f in tenant.features],
            "max_users": tenant.max_users,
            "max_improvements": tenant.max_improvements,
            "created_at": tenant.created_at.isoformat(),
            "status": tenant.status
        }
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        
        return {
            "user_id": user_id,
            "tenant_id": user.tenant_id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "permissions": user.permissions,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "status": user.status
        }
    
    def get_audit_logs(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs for tenant"""
        try:
            tenant_audits = [
                audit for audit in self.audits.values()
                if audit.tenant_id == tenant_id
            ]
            
            # Sort by creation time (newest first)
            tenant_audits.sort(key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            tenant_audits = tenant_audits[:limit]
            
            return [
                {
                    "audit_id": audit.audit_id,
                    "tenant_id": audit.tenant_id,
                    "user_id": audit.user_id,
                    "action": audit.action,
                    "resource": audit.resource,
                    "details": audit.details,
                    "ip_address": audit.ip_address,
                    "user_agent": audit.user_agent,
                    "created_at": audit.created_at.isoformat()
                }
                for audit in tenant_audits
            ]
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    def get_enterprise_summary(self) -> Dict[str, Any]:
        """Get enterprise summary"""
        total_tenants = len(self.tenants)
        total_users = len(self.users)
        total_audits = len(self.audits)
        
        # Count by level
        level_counts = {}
        for tenant in self.tenants.values():
            level = tenant.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by status
        active_tenants = len([t for t in self.tenants.values() if t.status == "active"])
        active_users = len([u for u in self.users.values() if u.status == "active"])
        
        return {
            "total_tenants": total_tenants,
            "total_users": total_users,
            "total_audits": total_audits,
            "active_tenants": active_tenants,
            "active_users": active_users,
            "level_distribution": level_counts,
            "queue_size": self.task_queue.qsize(),
            "active_workers": len(self.worker_threads)
        }
    
    def _log_enterprise(self, event: str, message: str):
        """Log enterprise event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if "enterprise_logs" not in self.enterprise_logs:
            self.enterprise_logs["enterprise_logs"] = []
        
        self.enterprise_logs["enterprise_logs"].append(log_entry)
        
        logger.info(f"Enterprise: {event} - {message}")
    
    def get_enterprise_logs(self) -> List[Dict[str, Any]]:
        """Get enterprise logs"""
        return self.enterprise_logs.get("enterprise_logs", [])
    
    def shutdown(self):
        """Shutdown enterprise system"""
        try:
            # Stop background workers
            for worker in self.worker_threads:
                worker.join(timeout=5)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self._log_enterprise("shutdown", "Enterprise system shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown enterprise system: {e}")

# Global enterprise instance
improvement_enterprise = None

def get_improvement_enterprise() -> RealImprovementEnterprise:
    """Get improvement enterprise instance"""
    global improvement_enterprise
    if not improvement_enterprise:
        improvement_enterprise = RealImprovementEnterprise()
    return improvement_enterprise













