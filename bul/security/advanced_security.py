"""
BUL Advanced Security System
============================

Advanced security features including encryption, audit trails, access control, and threat detection.
"""

import asyncio
import json
import time
import hashlib
import hmac
import secrets
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
import uuid
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext
import sqlite3
import threading
from pathlib import Path

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALICIOUS_UPLOAD = "malicious_upload"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class AccessLevel(str, Enum):
    """Access levels"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    OWNER = "owner"

class EncryptionType(str, Enum):
    """Encryption types"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"

@dataclass
class SecurityEvent:
    """Security event"""
    event_id: str
    event_type: str
    threat_type: Optional[ThreatType]
    severity: SecurityLevel
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    endpoint: str
    method: str
    request_data: Dict[str, Any]
    response_status: int
    timestamp: datetime
    is_blocked: bool
    block_reason: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class AccessControlEntry:
    """Access control entry"""
    entry_id: str
    resource_id: str
    resource_type: str
    user_id: str
    access_level: AccessLevel
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    conditions: Dict[str, Any]

@dataclass
class EncryptedDocument:
    """Encrypted document"""
    document_id: str
    encrypted_content: bytes
    encryption_key_id: str
    encryption_type: EncryptionType
    iv: bytes
    salt: bytes
    created_at: datetime
    created_by: str
    metadata: Dict[str, Any]

@dataclass
class AuditLog:
    """Audit log entry"""
    log_id: str
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any]
    risk_score: float

class AdvancedSecuritySystem:
    """Advanced Security System"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # Security components
        self.security_events: Dict[str, SecurityEvent] = {}
        self.access_control_entries: Dict[str, AccessControlEntry] = {}
        self.encrypted_documents: Dict[str, EncryptedDocument] = {}
        self.audit_logs: Dict[str, AuditLog] = {}
        
        # Encryption
        self.encryption_keys: Dict[str, bytes] = {}
        self.master_key: bytes = None
        self.fernet_cipher: Fernet = None
        
        # Authentication
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_secret = self._generate_jwt_secret()
        self.jwt_algorithm = "HS256"
        
        # Rate limiting
        self.rate_limit_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Database
        self.db_path = Path("data/security.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.db_lock = threading.Lock()
        
        # Initialize system
        self._initialize_encryption()
        self._initialize_database()
        self._load_data_from_database()
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret"""
        return secrets.token_urlsafe(32)
    
    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            # Generate master key
            self.master_key = Fernet.generate_key()
            self.fernet_cipher = Fernet(self.master_key)
            
            # Generate encryption keys
            self._generate_encryption_keys()
            
            self.logger.info("Encryption system initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption system: {e}")
    
    def _generate_encryption_keys(self):
        """Generate encryption keys"""
        try:
            # Generate symmetric key
            symmetric_key = Fernet.generate_key()
            self.encryption_keys["symmetric"] = symmetric_key
            
            # Generate asymmetric key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
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
            
            self.encryption_keys["private"] = private_pem
            self.encryption_keys["public"] = public_pem
            
            self.logger.info("Encryption keys generated successfully")
        
        except Exception as e:
            self.logger.error(f"Error generating encryption keys: {e}")
    
    def _initialize_database(self):
        """Initialize security database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS security_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        threat_type TEXT,
                        severity TEXT NOT NULL,
                        user_id TEXT,
                        ip_address TEXT NOT NULL,
                        user_agent TEXT,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        request_data TEXT,
                        response_status INTEGER,
                        timestamp TIMESTAMP NOT NULL,
                        is_blocked BOOLEAN DEFAULT FALSE,
                        block_reason TEXT,
                        metadata TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS access_control_entries (
                        entry_id TEXT PRIMARY KEY,
                        resource_id TEXT NOT NULL,
                        resource_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        access_level TEXT NOT NULL,
                        granted_by TEXT NOT NULL,
                        granted_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        conditions TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS encrypted_documents (
                        document_id TEXT PRIMARY KEY,
                        encrypted_content BLOB NOT NULL,
                        encryption_key_id TEXT NOT NULL,
                        encryption_type TEXT NOT NULL,
                        iv BLOB NOT NULL,
                        salt BLOB NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        created_by TEXT NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        log_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        action TEXT NOT NULL,
                        resource_type TEXT NOT NULL,
                        resource_id TEXT NOT NULL,
                        ip_address TEXT NOT NULL,
                        user_agent TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        success BOOLEAN NOT NULL,
                        details TEXT,
                        risk_score REAL DEFAULT 0.0
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_events_ip ON security_events(ip_address)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_control_user ON access_control_entries(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_control_resource ON access_control_entries(resource_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)')
                
                conn.commit()
                conn.close()
                
                self.logger.info("Security database initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize security database: {e}")
    
    def _load_data_from_database(self):
        """Load data from database into memory"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Load security events (last 1000)
                cursor.execute('''
                    SELECT * FROM security_events 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                ''')
                for row in cursor.fetchall():
                    event = SecurityEvent(
                        event_id=row[0],
                        event_type=row[1],
                        threat_type=ThreatType(row[2]) if row[2] else None,
                        severity=SecurityLevel(row[3]),
                        user_id=row[4],
                        ip_address=row[5],
                        user_agent=row[6],
                        endpoint=row[7],
                        method=row[8],
                        request_data=json.loads(row[9]) if row[9] else {},
                        response_status=row[10],
                        timestamp=datetime.fromisoformat(row[11]),
                        is_blocked=bool(row[12]),
                        block_reason=row[13],
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                    self.security_events[event.event_id] = event
                
                # Load access control entries
                cursor.execute('SELECT * FROM access_control_entries')
                for row in cursor.fetchall():
                    entry = AccessControlEntry(
                        entry_id=row[0],
                        resource_id=row[1],
                        resource_type=row[2],
                        user_id=row[3],
                        access_level=AccessLevel(row[4]),
                        granted_by=row[5],
                        granted_at=datetime.fromisoformat(row[6]),
                        expires_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        is_active=bool(row[8]),
                        conditions=json.loads(row[9]) if row[9] else {}
                    )
                    self.access_control_entries[entry.entry_id] = entry
                
                # Load audit logs (last 1000)
                cursor.execute('''
                    SELECT * FROM audit_logs 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                ''')
                for row in cursor.fetchall():
                    log = AuditLog(
                        log_id=row[0],
                        user_id=row[1],
                        action=row[2],
                        resource_type=row[3],
                        resource_id=row[4],
                        ip_address=row[5],
                        user_agent=row[6],
                        timestamp=datetime.fromisoformat(row[7]),
                        success=bool(row[8]),
                        details=json.loads(row[9]) if row[9] else {},
                        risk_score=row[10]
                    )
                    self.audit_logs[log.log_id] = log
                
                conn.close()
                
                self.logger.info(f"Loaded {len(self.security_events)} security events, {len(self.access_control_entries)} access control entries, {len(self.audit_logs)} audit logs")
        
        except Exception as e:
            self.logger.error(f"Error loading data from database: {e}")
    
    def _save_security_event_to_database(self, event: SecurityEvent):
        """Save security event to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO security_events 
                    (event_id, event_type, threat_type, severity, user_id, ip_address, 
                     user_agent, endpoint, method, request_data, response_status, 
                     timestamp, is_blocked, block_reason, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id, event.event_type, event.threat_type.value if event.threat_type else None,
                    event.severity.value, event.user_id, event.ip_address,
                    event.user_agent, event.endpoint, event.method,
                    json.dumps(event.request_data), event.response_status,
                    event.timestamp.isoformat(), event.is_blocked, event.block_reason,
                    json.dumps(event.metadata)
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving security event to database: {e}")
    
    def _save_access_control_entry_to_database(self, entry: AccessControlEntry):
        """Save access control entry to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO access_control_entries 
                    (entry_id, resource_id, resource_type, user_id, access_level,
                     granted_by, granted_at, expires_at, is_active, conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id, entry.resource_id, entry.resource_type,
                    entry.user_id, entry.access_level.value, entry.granted_by,
                    entry.granted_at.isoformat(),
                    entry.expires_at.isoformat() if entry.expires_at else None,
                    entry.is_active, json.dumps(entry.conditions)
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving access control entry to database: {e}")
    
    def _save_audit_log_to_database(self, log: AuditLog):
        """Save audit log to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO audit_logs 
                    (log_id, user_id, action, resource_type, resource_id, ip_address,
                     user_agent, timestamp, success, details, risk_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log.log_id, log.user_id, log.action, log.resource_type,
                    log.resource_id, log.ip_address, log.user_agent,
                    log.timestamp.isoformat(), log.success, json.dumps(log.details),
                    log.risk_score
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving audit log to database: {e}")
    
    async def log_security_event(
        self,
        event_type: str,
        threat_type: Optional[ThreatType],
        severity: SecurityLevel,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        endpoint: str,
        method: str,
        request_data: Dict[str, Any],
        response_status: int,
        is_blocked: bool = False,
        block_reason: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> SecurityEvent:
        """Log a security event"""
        try:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                threat_type=threat_type,
                severity=severity,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                endpoint=endpoint,
                method=method,
                request_data=request_data,
                response_status=response_status,
                timestamp=datetime.now(),
                is_blocked=is_blocked,
                block_reason=block_reason,
                metadata=metadata or {}
            )
            
            # Save to memory and database
            self.security_events[event.event_id] = event
            self._save_security_event_to_database(event)
            
            # Check for threats and take action
            await self._analyze_security_event(event)
            
            self.logger.info(f"Security event logged: {event_type} from {ip_address}")
            return event
        
        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")
            raise
    
    async def _analyze_security_event(self, event: SecurityEvent):
        """Analyze security event for threats"""
        try:
            # Check for brute force attacks
            if await self._detect_brute_force(event):
                await self._handle_brute_force_threat(event)
            
            # Check for suspicious patterns
            if await self._detect_suspicious_activity(event):
                await self._handle_suspicious_activity(event)
            
            # Check for rate limiting violations
            if await self._detect_rate_limit_violation(event):
                await self._handle_rate_limit_violation(event)
        
        except Exception as e:
            self.logger.error(f"Error analyzing security event: {e}")
    
    async def _detect_brute_force(self, event: SecurityEvent) -> bool:
        """Detect brute force attack"""
        try:
            # Check failed login attempts from same IP
            recent_events = [
                e for e in self.security_events.values()
                if (e.ip_address == event.ip_address and 
                    e.event_type == "login_failed" and
                    (datetime.now() - e.timestamp).total_seconds() < 300)  # 5 minutes
            ]
            
            return len(recent_events) >= 5  # 5 failed attempts in 5 minutes
        
        except Exception as e:
            self.logger.error(f"Error detecting brute force: {e}")
            return False
    
    async def _detect_suspicious_activity(self, event: SecurityEvent) -> bool:
        """Detect suspicious activity"""
        try:
            # Check for SQL injection patterns
            sql_patterns = ["'", "union", "select", "drop", "insert", "update", "delete"]
            request_str = json.dumps(event.request_data).lower()
            
            if any(pattern in request_str for pattern in sql_patterns):
                return True
            
            # Check for XSS patterns
            xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
            if any(pattern in request_str for pattern in xss_patterns):
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error detecting suspicious activity: {e}")
            return False
    
    async def _detect_rate_limit_violation(self, event: SecurityEvent) -> bool:
        """Detect rate limit violation"""
        try:
            # Check requests per minute from same IP
            recent_requests = [
                e for e in self.security_events.values()
                if (e.ip_address == event.ip_address and
                    (datetime.now() - e.timestamp).total_seconds() < 60)  # 1 minute
            ]
            
            return len(recent_requests) >= 100  # 100 requests per minute
        
        except Exception as e:
            self.logger.error(f"Error detecting rate limit violation: {e}")
            return False
    
    async def _handle_brute_force_threat(self, event: SecurityEvent):
        """Handle brute force threat"""
        try:
            # Block IP for 1 hour
            self.blocked_ips[event.ip_address] = datetime.now() + timedelta(hours=1)
            
            # Log threat
            await self.log_security_event(
                event_type="brute_force_detected",
                threat_type=ThreatType.BRUTE_FORCE,
                severity=SecurityLevel.HIGH,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                method=event.method,
                request_data=event.request_data,
                response_status=429,
                is_blocked=True,
                block_reason="Brute force attack detected",
                metadata={"block_duration": 3600}
            )
            
            self.logger.warning(f"Brute force attack detected from {event.ip_address}")
        
        except Exception as e:
            self.logger.error(f"Error handling brute force threat: {e}")
    
    async def _handle_suspicious_activity(self, event: SecurityEvent):
        """Handle suspicious activity"""
        try:
            # Log threat
            await self.log_security_event(
                event_type="suspicious_activity_detected",
                threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                severity=SecurityLevel.MEDIUM,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                method=event.method,
                request_data=event.request_data,
                response_status=400,
                is_blocked=True,
                block_reason="Suspicious activity detected",
                metadata={"threat_patterns": ["sql_injection", "xss"]}
            )
            
            self.logger.warning(f"Suspicious activity detected from {event.ip_address}")
        
        except Exception as e:
            self.logger.error(f"Error handling suspicious activity: {e}")
    
    async def _handle_rate_limit_violation(self, event: SecurityEvent):
        """Handle rate limit violation"""
        try:
            # Block IP for 10 minutes
            self.blocked_ips[event.ip_address] = datetime.now() + timedelta(minutes=10)
            
            # Log threat
            await self.log_security_event(
                event_type="rate_limit_violation",
                threat_type=ThreatType.RATE_LIMIT_EXCEEDED,
                severity=SecurityLevel.MEDIUM,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                method=event.method,
                request_data=event.request_data,
                response_status=429,
                is_blocked=True,
                block_reason="Rate limit exceeded",
                metadata={"block_duration": 600}
            )
            
            self.logger.warning(f"Rate limit violation from {event.ip_address}")
        
        except Exception as e:
            self.logger.error(f"Error handling rate limit violation: {e}")
    
    async def check_access_permission(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str,
        required_access_level: AccessLevel
    ) -> bool:
        """Check if user has access permission"""
        try:
            # Find access control entries for this user and resource
            access_entries = [
                entry for entry in self.access_control_entries.values()
                if (entry.user_id == user_id and 
                    entry.resource_id == resource_id and
                    entry.resource_type == resource_type and
                    entry.is_active and
                    (entry.expires_at is None or entry.expires_at > datetime.now()))
            ]
            
            if not access_entries:
                return False
            
            # Check if any entry has sufficient access level
            access_levels = [AccessLevel.READ, AccessLevel.WRITE, AccessLevel.DELETE, AccessLevel.ADMIN, AccessLevel.OWNER]
            required_index = access_levels.index(required_access_level)
            
            for entry in access_entries:
                entry_index = access_levels.index(entry.access_level)
                if entry_index >= required_index:
                    return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error checking access permission: {e}")
            return False
    
    async def grant_access(
        self,
        resource_id: str,
        resource_type: str,
        user_id: str,
        access_level: AccessLevel,
        granted_by: str,
        expires_at: Optional[datetime] = None,
        conditions: Dict[str, Any] = None
    ) -> AccessControlEntry:
        """Grant access to a resource"""
        try:
            entry = AccessControlEntry(
                entry_id=str(uuid.uuid4()),
                resource_id=resource_id,
                resource_type=resource_type,
                user_id=user_id,
                access_level=access_level,
                granted_by=granted_by,
                granted_at=datetime.now(),
                expires_at=expires_at,
                is_active=True,
                conditions=conditions or {}
            )
            
            # Save to memory and database
            self.access_control_entries[entry.entry_id] = entry
            self._save_access_control_entry_to_database(entry)
            
            self.logger.info(f"Access granted: {user_id} -> {resource_id} ({access_level.value})")
            return entry
        
        except Exception as e:
            self.logger.error(f"Error granting access: {e}")
            raise
    
    async def encrypt_document(
        self,
        document_id: str,
        content: str,
        encryption_type: EncryptionType = EncryptionType.SYMMETRIC,
        created_by: str = "system"
    ) -> EncryptedDocument:
        """Encrypt document content"""
        try:
            # Generate salt and IV
            salt = secrets.token_bytes(32)
            iv = secrets.token_bytes(16)
            
            # Encrypt content
            if encryption_type == EncryptionType.SYMMETRIC:
                cipher = Fernet(self.encryption_keys["symmetric"])
                encrypted_content = cipher.encrypt(content.encode('utf-8'))
                key_id = "symmetric"
            else:
                # For asymmetric encryption, we'll use symmetric for content and encrypt the key
                content_key = Fernet.generate_key()
                cipher = Fernet(content_key)
                encrypted_content = cipher.encrypt(content.encode('utf-8'))
                
                # Encrypt the content key with public key
                public_key = serialization.load_pem_public_key(self.encryption_keys["public"])
                encrypted_key = public_key.encrypt(
                    content_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                encrypted_content = encrypted_key + encrypted_content
                key_id = "asymmetric"
            
            encrypted_doc = EncryptedDocument(
                document_id=document_id,
                encrypted_content=encrypted_content,
                encryption_key_id=key_id,
                encryption_type=encryption_type,
                iv=iv,
                salt=salt,
                created_at=datetime.now(),
                created_by=created_by,
                metadata={"original_size": len(content)}
            )
            
            # Save encrypted document
            self.encrypted_documents[document_id] = encrypted_doc
            
            self.logger.info(f"Document {document_id} encrypted using {encryption_type.value}")
            return encrypted_doc
        
        except Exception as e:
            self.logger.error(f"Error encrypting document: {e}")
            raise
    
    async def decrypt_document(self, document_id: str) -> str:
        """Decrypt document content"""
        try:
            if document_id not in self.encrypted_documents:
                raise ValueError(f"Encrypted document {document_id} not found")
            
            encrypted_doc = self.encrypted_documents[document_id]
            
            # Decrypt content
            if encrypted_doc.encryption_type == EncryptionType.SYMMETRIC:
                cipher = Fernet(self.encryption_keys["symmetric"])
                decrypted_content = cipher.decrypt(encrypted_doc.encrypted_content)
            else:
                # For asymmetric encryption, decrypt the key first
                private_key = serialization.load_pem_private_key(
                    self.encryption_keys["private"],
                    password=None
                )
                
                # Extract encrypted key and content
                key_size = 256  # RSA 2048 key size
                encrypted_key = encrypted_doc.encrypted_content[:key_size]
                encrypted_content = encrypted_doc.encrypted_content[key_size:]
                
                # Decrypt the content key
                content_key = private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt the content
                cipher = Fernet(content_key)
                decrypted_content = cipher.decrypt(encrypted_content)
            
            return decrypted_content.decode('utf-8')
        
        except Exception as e:
            self.logger.error(f"Error decrypting document: {e}")
            raise
    
    async def log_audit_event(
        self,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Dict[str, Any] = None,
        risk_score: float = 0.0
    ) -> AuditLog:
        """Log audit event"""
        try:
            log = AuditLog(
                log_id=str(uuid.uuid4()),
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.now(),
                success=success,
                details=details or {},
                risk_score=risk_score
            )
            
            # Save to memory and database
            self.audit_logs[log.log_id] = log
            self._save_audit_log_to_database(log)
            
            return log
        
        except Exception as e:
            self.logger.error(f"Error logging audit event: {e}")
            raise
    
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        try:
            if ip_address in self.blocked_ips:
                if self.blocked_ips[ip_address] > datetime.now():
                    return True
                else:
                    # Remove expired block
                    del self.blocked_ips[ip_address]
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error checking IP block status: {e}")
            return False
    
    async def generate_jwt_token(self, user_id: str, expires_in_hours: int = 24) -> str:
        """Generate JWT token"""
        try:
            payload = {
                "user_id": user_id,
                "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
                "iat": datetime.utcnow(),
                "iss": "bul-system"
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            return token
        
        except Exception as e:
            self.logger.error(f"Error generating JWT token: {e}")
            raise
    
    async def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
        except Exception as e:
            self.logger.error(f"Error verifying JWT token: {e}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.password_context.hash(password)
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.password_context.verify(password, hashed_password)
    
    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        try:
            # Get recent security events
            recent_events = [
                e for e in self.security_events.values()
                if (datetime.now() - e.timestamp).total_seconds() < 86400  # Last 24 hours
            ]
            
            # Count by threat type
            threat_counts = {}
            for event in recent_events:
                if event.threat_type:
                    threat_type = event.threat_type.value
                    threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
            
            # Count by severity
            severity_counts = {}
            for event in recent_events:
                severity = event.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count blocked events
            blocked_count = len([e for e in recent_events if e.is_blocked])
            
            # Get top blocked IPs
            blocked_ips = {}
            for event in recent_events:
                if event.is_blocked:
                    ip = event.ip_address
                    blocked_ips[ip] = blocked_ips.get(ip, 0) + 1
            
            top_blocked_ips = sorted(blocked_ips.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_events_24h": len(recent_events),
                "blocked_events_24h": blocked_count,
                "threat_distribution": threat_counts,
                "severity_distribution": severity_counts,
                "top_blocked_ips": top_blocked_ips,
                "currently_blocked_ips": len(self.blocked_ips),
                "total_access_entries": len(self.access_control_entries),
                "total_encrypted_documents": len(self.encrypted_documents),
                "total_audit_logs": len(self.audit_logs)
            }
        
        except Exception as e:
            self.logger.error(f"Error getting security dashboard data: {e}")
            return {}

# Global security system
_security_system: Optional[AdvancedSecuritySystem] = None

def get_security_system() -> AdvancedSecuritySystem:
    """Get the global security system"""
    global _security_system
    if _security_system is None:
        _security_system = AdvancedSecuritySystem()
    return _security_system

# Security router
security_router = APIRouter(prefix="/security", tags=["Security"])

@security_router.post("/log-event")
async def log_security_event_endpoint(
    event_type: str = Field(..., description="Event type"),
    threat_type: Optional[ThreatType] = Field(None, description="Threat type"),
    severity: SecurityLevel = Field(..., description="Severity level"),
    user_id: Optional[str] = Field(None, description="User ID"),
    ip_address: str = Field(..., description="IP address"),
    user_agent: str = Field(..., description="User agent"),
    endpoint: str = Field(..., description="Endpoint"),
    method: str = Field(..., description="HTTP method"),
    request_data: Dict[str, Any] = Field(..., description="Request data"),
    response_status: int = Field(..., description="Response status"),
    is_blocked: bool = Field(False, description="Is blocked"),
    block_reason: Optional[str] = Field(None, description="Block reason"),
    metadata: Dict[str, Any] = Field({}, description="Metadata")
):
    """Log a security event"""
    try:
        system = get_security_system()
        event = await system.log_security_event(
            event_type, threat_type, severity, user_id, ip_address,
            user_agent, endpoint, method, request_data, response_status,
            is_blocked, block_reason, metadata
        )
        return {"event": asdict(event), "success": True}
    
    except Exception as e:
        logger.error(f"Error logging security event: {e}")
        raise HTTPException(status_code=500, detail="Failed to log security event")

@security_router.post("/grant-access")
async def grant_access_endpoint(
    resource_id: str = Field(..., description="Resource ID"),
    resource_type: str = Field(..., description="Resource type"),
    user_id: str = Field(..., description="User ID"),
    access_level: AccessLevel = Field(..., description="Access level"),
    granted_by: str = Field(..., description="Granted by user ID"),
    expires_at: Optional[datetime] = Field(None, description="Expiration date"),
    conditions: Dict[str, Any] = Field({}, description="Access conditions")
):
    """Grant access to a resource"""
    try:
        system = get_security_system()
        entry = await system.grant_access(
            resource_id, resource_type, user_id, access_level,
            granted_by, expires_at, conditions
        )
        return {"entry": asdict(entry), "success": True}
    
    except Exception as e:
        logger.error(f"Error granting access: {e}")
        raise HTTPException(status_code=500, detail="Failed to grant access")

@security_router.post("/encrypt-document")
async def encrypt_document_endpoint(
    document_id: str = Field(..., description="Document ID"),
    content: str = Field(..., description="Document content"),
    encryption_type: EncryptionType = Field(EncryptionType.SYMMETRIC, description="Encryption type"),
    created_by: str = Field("system", description="Created by user ID")
):
    """Encrypt document content"""
    try:
        system = get_security_system()
        encrypted_doc = await system.encrypt_document(
            document_id, content, encryption_type, created_by
        )
        return {"encrypted_document": asdict(encrypted_doc), "success": True}
    
    except Exception as e:
        logger.error(f"Error encrypting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to encrypt document")

@security_router.post("/decrypt-document")
async def decrypt_document_endpoint(
    document_id: str = Field(..., description="Document ID")
):
    """Decrypt document content"""
    try:
        system = get_security_system()
        content = await system.decrypt_document(document_id)
        return {"content": content, "success": True}
    
    except Exception as e:
        logger.error(f"Error decrypting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to decrypt document")

@security_router.get("/dashboard")
async def get_security_dashboard_endpoint():
    """Get security dashboard data"""
    try:
        system = get_security_system()
        data = await system.get_security_dashboard_data()
        return {"data": data, "success": True}
    
    except Exception as e:
        logger.error(f"Error getting security dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security dashboard data")

@security_router.get("/check-access")
async def check_access_permission_endpoint(
    user_id: str = Field(..., description="User ID"),
    resource_id: str = Field(..., description="Resource ID"),
    resource_type: str = Field(..., description="Resource type"),
    required_access_level: AccessLevel = Field(..., description="Required access level")
):
    """Check access permission"""
    try:
        system = get_security_system()
        has_access = await system.check_access_permission(
            user_id, resource_id, resource_type, required_access_level
        )
        return {"has_access": has_access, "success": True}
    
    except Exception as e:
        logger.error(f"Error checking access permission: {e}")
        raise HTTPException(status_code=500, detail="Failed to check access permission")
