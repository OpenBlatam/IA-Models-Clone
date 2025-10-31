"""
BUL - Business Universal Language (Advanced Security System)
===========================================================

Advanced security system with authentication, authorization, and monitoring.
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path
import json
import time
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_security.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
SECURITY_REQUESTS = Counter('bul_security_requests_total', 'Total security requests', ['method', 'endpoint'])
SECURITY_AUTH_ATTEMPTS = Counter('bul_security_auth_attempts_total', 'Authentication attempts', ['result'])
SECURITY_FAILED_LOGINS = Counter('bul_security_failed_logins_total', 'Failed login attempts', ['username'])
SECURITY_SUSPICIOUS_ACTIVITY = Counter('bul_security_suspicious_activity_total', 'Suspicious activity detected')

class SecurityLevel(str, Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Permission(str, Enum):
    """Permission enumeration."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SECURITY = "security"

# Database Models
class SecurityUser(Base):
    __tablename__ = "security_users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_locked = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    last_login = Column(DateTime)
    last_failed_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    security_level = Column(String, default=SecurityLevel.MEDIUM)
    permissions = Column(Text, default="[]")
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String)

class SecuritySession(Base):
    __tablename__ = "security_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("security_users.id"))
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    user_agent = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("SecurityUser")

class SecurityAuditLog(Base):
    __tablename__ = "security_audit_logs"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("security_users.id"))
    action = Column(String, nullable=False)
    resource = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    success = Column(Boolean, default=True)
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    security_level = Column(String, default=SecurityLevel.MEDIUM)
    
    # Relationships
    user = relationship("SecurityUser")

class SecurityThreat(Base):
    __tablename__ = "security_threats"
    
    id = Column(String, primary_key=True)
    threat_type = Column(String, nullable=False)
    severity = Column(String, default=SecurityLevel.MEDIUM)
    description = Column(Text)
    ip_address = Column(String)
    user_agent = Column(String)
    user_id = Column(String, ForeignKey("security_users.id"))
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    is_resolved = Column(Boolean, default=False)
    mitigation_action = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# Security Configuration
SECURITY_CONFIG = {
    "jwt_secret": "your-super-secret-jwt-key-change-in-production",
    "jwt_algorithm": "HS256",
    "jwt_expiration": 3600,  # 1 hour
    "max_failed_attempts": 5,
    "lockout_duration": 900,  # 15 minutes
    "password_min_length": 8,
    "password_require_special": True,
    "password_require_numbers": True,
    "password_require_uppercase": True,
    "session_timeout": 1800,  # 30 minutes
    "mfa_required": False,
    "ip_whitelist": [],
    "ip_blacklist": [],
    "rate_limit_per_minute": 60
}

class AdvancedSecuritySystem:
    """Advanced security system with comprehensive features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Advanced Security System",
            description="Advanced security system with authentication, authorization, and monitoring",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # Security components
        self.security_bearer = HTTPBearer()
        self.rate_limiter = {}
        self.suspicious_ips = set()
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        
        logger.info("Advanced Security System initialized")
    
    def setup_middleware(self):
        """Setup security middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup security API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with security system information."""
            return {
                "message": "BUL Advanced Security System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Advanced Authentication",
                    "Role-Based Authorization",
                    "Multi-Factor Authentication",
                    "Session Management",
                    "Audit Logging",
                    "Threat Detection",
                    "Rate Limiting",
                    "IP Filtering"
                ],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/auth/register", tags=["Authentication"])
        async def register_user(request: Request, user_data: dict):
            """Register a new user with security validation."""
            try:
                # Rate limiting
                client_ip = request.client.host
                if not self.check_rate_limit(client_ip):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Validate input
                username = user_data.get("username")
                email = user_data.get("email")
                password = user_data.get("password")
                
                if not all([username, email, password]):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                # Validate password strength
                if not self.validate_password_strength(password):
                    raise HTTPException(status_code=400, detail="Password does not meet security requirements")
                
                # Check if user already exists
                existing_user = self.db.query(SecurityUser).filter(
                    (SecurityUser.username == username) | (SecurityUser.email == email)
                ).first()
                
                if existing_user:
                    raise HTTPException(status_code=400, detail="User already exists")
                
                # Create user
                salt = secrets.token_hex(16)
                password_hash = self.hash_password(password, salt)
                
                db_user = SecurityUser(
                    id=f"user_{int(time.time())}",
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    salt=salt,
                    security_level=SecurityLevel.MEDIUM,
                    permissions=json.dumps([Permission.READ])
                )
                
                self.db.add(db_user)
                self.db.commit()
                
                # Log registration
                self.log_security_event(
                    user_id=db_user.id,
                    action="user_registration",
                    success=True,
                    ip_address=client_ip,
                    user_agent=request.headers.get("user-agent")
                )
                
                return {
                    "message": "User registered successfully",
                    "user_id": db_user.id,
                    "username": db_user.username
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Registration error: {e}")
                raise HTTPException(status_code=500, detail="Registration failed")
        
        @self.app.post("/auth/login", tags=["Authentication"])
        async def login_user(request: Request, login_data: dict):
            """Authenticate user with security monitoring."""
            try:
                client_ip = request.client.host
                user_agent = request.headers.get("user-agent")
                
                # Rate limiting
                if not self.check_rate_limit(client_ip):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Check IP blacklist
                if client_ip in SECURITY_CONFIG["ip_blacklist"]:
                    self.log_security_event(
                        action="blocked_ip_login_attempt",
                        success=False,
                        ip_address=client_ip,
                        user_agent=user_agent,
                        security_level=SecurityLevel.HIGH
                    )
                    raise HTTPException(status_code=403, detail="IP address blocked")
                
                username = login_data.get("username")
                password = login_data.get("password")
                
                if not all([username, password]):
                    raise HTTPException(status_code=400, detail="Missing credentials")
                
                # Find user
                user = self.db.query(SecurityUser).filter(SecurityUser.username == username).first()
                
                if not user:
                    self.log_failed_login(username, client_ip, user_agent)
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Check if user is locked
                if user.is_locked:
                    self.log_security_event(
                        user_id=user.id,
                        action="locked_account_login_attempt",
                        success=False,
                        ip_address=client_ip,
                        user_agent=user_agent,
                        security_level=SecurityLevel.MEDIUM
                    )
                    raise HTTPException(status_code=423, detail="Account is locked")
                
                # Verify password
                if not self.verify_password(password, user.password_hash, user.salt):
                    self.log_failed_login(username, client_ip, user_agent)
                    self.increment_failed_attempts(user)
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Reset failed attempts on successful login
                user.failed_login_attempts = 0
                user.last_login = datetime.utcnow()
                user.is_locked = False
                self.db.commit()
                
                # Create session
                session_token = self.create_session(user.id, client_ip, user_agent)
                
                # Log successful login
                self.log_security_event(
                    user_id=user.id,
                    action="user_login",
                    success=True,
                    ip_address=client_ip,
                    user_agent=user_agent
                )
                
                SECURITY_AUTH_ATTEMPTS.labels(result="success").inc()
                
                return {
                    "message": "Login successful",
                    "token": session_token,
                    "user_id": user.id,
                    "username": user.username,
                    "permissions": json.loads(user.permissions),
                    "expires_at": (datetime.utcnow() + timedelta(seconds=SECURITY_CONFIG["jwt_expiration"])).isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Login error: {e}")
                raise HTTPException(status_code=500, detail="Login failed")
        
        @self.app.post("/auth/logout", tags=["Authentication"])
        async def logout_user(request: Request, token: str = Depends(security_bearer)):
            """Logout user and invalidate session."""
            try:
                # Decode token
                payload = self.decode_token(token.credentials)
                user_id = payload.get("user_id")
                
                # Invalidate session
                session = self.db.query(SecuritySession).filter(
                    SecuritySession.user_id == user_id,
                    SecuritySession.token == token.credentials,
                    SecuritySession.is_active == True
                ).first()
                
                if session:
                    session.is_active = False
                    self.db.commit()
                
                # Log logout
                self.log_security_event(
                    user_id=user_id,
                    action="user_logout",
                    success=True,
                    ip_address=request.client.host,
                    user_agent=request.headers.get("user-agent")
                )
                
                return {"message": "Logout successful"}
                
            except Exception as e:
                logger.error(f"Logout error: {e}")
                raise HTTPException(status_code=500, detail="Logout failed")
        
        @self.app.get("/auth/me", tags=["Authentication"])
        async def get_current_user(token: str = Depends(security_bearer)):
            """Get current user information."""
            try:
                payload = self.decode_token(token.credentials)
                user_id = payload.get("user_id")
                
                user = self.db.query(SecurityUser).filter(SecurityUser.id == user_id).first()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                return {
                    "user_id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "security_level": user.security_level,
                    "permissions": json.loads(user.permissions),
                    "mfa_enabled": user.mfa_enabled,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "created_at": user.created_at.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Get user error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get user information")
        
        @self.app.get("/security/audit-logs", tags=["Security"])
        async def get_audit_logs(token: str = Depends(security_bearer), limit: int = 100):
            """Get security audit logs."""
            try:
                payload = self.decode_token(token.credentials)
                user_id = payload.get("user_id")
                
                # Check permissions
                if not self.has_permission(user_id, Permission.SECURITY):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                logs = self.db.query(SecurityAuditLog).order_by(
                    SecurityAuditLog.timestamp.desc()
                ).limit(limit).all()
                
                return {
                    "audit_logs": [
                        {
                            "id": log.id,
                            "user_id": log.user_id,
                            "action": log.action,
                            "resource": log.resource,
                            "ip_address": log.ip_address,
                            "success": log.success,
                            "details": log.details,
                            "timestamp": log.timestamp.isoformat(),
                            "security_level": log.security_level
                        }
                        for log in logs
                    ],
                    "total": len(logs)
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get audit logs error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get audit logs")
        
        @self.app.get("/security/threats", tags=["Security"])
        async def get_security_threats(token: str = Depends(security_bearer)):
            """Get security threats."""
            try:
                payload = self.decode_token(token.credentials)
                user_id = payload.get("user_id")
                
                # Check permissions
                if not self.has_permission(user_id, Permission.SECURITY):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                threats = self.db.query(SecurityThreat).order_by(
                    SecurityThreat.detected_at.desc()
                ).all()
                
                return {
                    "threats": [
                        {
                            "id": threat.id,
                            "threat_type": threat.threat_type,
                            "severity": threat.severity,
                            "description": threat.description,
                            "ip_address": threat.ip_address,
                            "user_id": threat.user_id,
                            "detected_at": threat.detected_at.isoformat(),
                            "resolved_at": threat.resolved_at.isoformat() if threat.resolved_at else None,
                            "is_resolved": threat.is_resolved,
                            "mitigation_action": threat.mitigation_action
                        }
                        for threat in threats
                    ],
                    "total": len(threats)
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get threats error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get security threats")
        
        @self.app.get("/security/dashboard", tags=["Security"])
        async def get_security_dashboard(token: str = Depends(security_bearer)):
            """Get security dashboard data."""
            try:
                payload = self.decode_token(token.credentials)
                user_id = payload.get("user_id")
                
                # Check permissions
                if not self.has_permission(user_id, Permission.SECURITY):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Get statistics
                total_users = self.db.query(SecurityUser).count()
                active_sessions = self.db.query(SecuritySession).filter(SecuritySession.is_active == True).count()
                locked_accounts = self.db.query(SecurityUser).filter(SecurityUser.is_locked == True).count()
                recent_threats = self.db.query(SecurityThreat).filter(
                    SecurityThreat.detected_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                # Get recent audit logs
                recent_logs = self.db.query(SecurityAuditLog).order_by(
                    SecurityAuditLog.timestamp.desc()
                ).limit(10).all()
                
                return {
                    "summary": {
                        "total_users": total_users,
                        "active_sessions": active_sessions,
                        "locked_accounts": locked_accounts,
                        "recent_threats_24h": recent_threats
                    },
                    "recent_activity": [
                        {
                            "action": log.action,
                            "user_id": log.user_id,
                            "success": log.success,
                            "timestamp": log.timestamp.isoformat(),
                            "security_level": log.security_level
                        }
                        for log in recent_logs
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get dashboard error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get security dashboard")
    
    def setup_default_data(self):
        """Setup default security data."""
        try:
            # Create default admin user
            admin_salt = secrets.token_hex(16)
            admin_password_hash = self.hash_password("admin123", admin_salt)
            
            admin_user = SecurityUser(
                id="admin",
                username="admin",
                email="admin@bul-security.com",
                password_hash=admin_password_hash,
                salt=admin_salt,
                security_level=SecurityLevel.HIGH,
                permissions=json.dumps([Permission.ADMIN, Permission.SECURITY]),
                is_active=True
            )
            
            self.db.add(admin_user)
            self.db.commit()
            
            logger.info("Default security data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default security data: {e}")
    
    def hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt."""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password, salt) == password_hash
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < SECURITY_CONFIG["password_min_length"]:
            return False
        
        if SECURITY_CONFIG["password_require_special"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        if SECURITY_CONFIG["password_require_numbers"] and not any(c.isdigit() for c in password):
            return False
        
        if SECURITY_CONFIG["password_require_uppercase"] and not any(c.isupper() for c in password):
            return False
        
        return True
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create user session."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(seconds=SECURITY_CONFIG["jwt_expiration"])
        
        session = SecuritySession(
            id=f"session_{int(time.time())}",
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(session)
        self.db.commit()
        
        return token
    
    def decode_token(self, token: str) -> dict:
        """Decode JWT token."""
        try:
            payload = jwt.decode(token, SECURITY_CONFIG["jwt_secret"], algorithms=[SECURITY_CONFIG["jwt_algorithm"]])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def check_rate_limit(self, ip_address: str) -> bool:
        """Check rate limit for IP address."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self.rate_limiter = {ip: times for ip, times in self.rate_limiter.items() 
                           if any(t > minute_ago for t in times)}
        
        # Check current IP
        if ip_address not in self.rate_limiter:
            self.rate_limiter[ip_address] = []
        
        # Add current request
        self.rate_limiter[ip_address].append(current_time)
        
        # Check limit
        return len(self.rate_limiter[ip_address]) <= SECURITY_CONFIG["rate_limit_per_minute"]
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user = self.db.query(SecurityUser).filter(SecurityUser.id == user_id).first()
        if not user:
            return False
        
        user_permissions = json.loads(user.permissions)
        return permission in user_permissions or Permission.ADMIN in user_permissions
    
    def log_security_event(self, user_id: str = None, action: str = "", success: bool = True,
                          ip_address: str = "", user_agent: str = "", resource: str = "",
                          details: str = "", security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Log security event."""
        try:
            audit_log = SecurityAuditLog(
                id=f"audit_{int(time.time())}",
                user_id=user_id,
                action=action,
                resource=resource,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                details=details,
                security_level=security_level
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
            # Check for suspicious activity
            if not success and security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.detect_suspicious_activity(ip_address, action, user_id)
                
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def log_failed_login(self, username: str, ip_address: str, user_agent: str):
        """Log failed login attempt."""
        SECURITY_FAILED_LOGINS.labels(username=username).inc()
        
        self.log_security_event(
            action="failed_login",
            success=False,
            ip_address=ip_address,
            user_agent=user_agent,
            details=f"Failed login attempt for username: {username}",
            security_level=SecurityLevel.MEDIUM
        )
    
    def increment_failed_attempts(self, user: SecurityUser):
        """Increment failed login attempts."""
        user.failed_login_attempts += 1
        user.last_failed_login = datetime.utcnow()
        
        if user.failed_login_attempts >= SECURITY_CONFIG["max_failed_attempts"]:
            user.is_locked = True
            self.log_security_event(
                user_id=user.id,
                action="account_locked",
                success=False,
                details=f"Account locked after {user.failed_login_attempts} failed attempts",
                security_level=SecurityLevel.HIGH
            )
        
        self.db.commit()
    
    def detect_suspicious_activity(self, ip_address: str, action: str, user_id: str = None):
        """Detect suspicious activity."""
        SECURITY_SUSPICIOUS_ACTIVITY.inc()
        
        threat = SecurityThreat(
            id=f"threat_{int(time.time())}",
            threat_type="suspicious_activity",
            severity=SecurityLevel.HIGH,
            description=f"Suspicious activity detected: {action}",
            ip_address=ip_address,
            user_id=user_id
        )
        
        self.db.add(threat)
        self.db.commit()
        
        logger.warning(f"Suspicious activity detected from IP {ip_address}: {action}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8004, debug: bool = False):
        """Run the security system."""
        logger.info(f"Starting Advanced Security System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Advanced Security System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run security system
    system = AdvancedSecuritySystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
