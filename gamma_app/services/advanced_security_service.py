"""
Gamma App - Advanced Security Service
Enterprise-grade security with compliance, threat detection, and audit logging
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict, deque
import jwt
import bcrypt
import secrets
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import re
import ipaddress
import geoip2.database
import geoip2.errors
from user_agents import parse
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Threat types"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDoS = "ddos"
    MALWARE = "malware"
    PHISHING = "phishing"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

class AuditEventType(Enum):
    """Audit event types"""
    LOGIN = "login"
    LOGOUT = "logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    compliance_standards: List[ComplianceStandard]
    severity: SecurityLevel
    enabled: bool = True
    created_at: datetime = None

@dataclass
class ThreatDetection:
    """Threat detection result"""
    threat_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    indicators: List[str]
    confidence: float
    detected_at: datetime
    status: str = "active"

@dataclass
class AuditLog:
    """Audit log entry"""
    log_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    source_ip: str
    user_agent: str
    action: str
    resource: str
    details: Dict[str, Any]
    timestamp: datetime
    success: bool = True

@dataclass
class SecurityIncident:
    """Security incident"""
    incident_id: str
    title: str
    description: str
    threat_type: ThreatType
    severity: SecurityLevel
    status: str = "open"
    assigned_to: Optional[str] = None
    created_at: datetime = None
    resolved_at: Optional[datetime] = None

class AdvancedSecurityService:
    """Advanced Security Service with enterprise features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "security.db")
        self.redis_client = None
        self.security_policies = {}
        self.threat_detections = {}
        self.audit_logs = []
        self.security_incidents = {}
        self.encryption_key = None
        self.geoip_db = None
        self.anomaly_detector = None
        self.rate_limiters = {}
        self.blacklisted_ips = set()
        self.whitelisted_ips = set()
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_encryption()
        self._init_geoip()
        self._init_anomaly_detector()
        self._init_default_policies()
    
    def _init_database(self):
        """Initialize security database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create security policies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    rules TEXT NOT NULL,
                    compliance_standards TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create threat detections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threat_detections (
                    threat_id TEXT PRIMARY KEY,
                    threat_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    user_id TEXT,
                    description TEXT NOT NULL,
                    indicators TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detected_at DATETIME NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Create audit logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    source_ip TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT NOT NULL,
                    success BOOLEAN DEFAULT TRUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create security incidents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_incidents (
                    incident_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    threat_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    assigned_to TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME
                )
            """)
            
            # Create user sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.commit()
        
        logger.info("Security database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching and rate limiting"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for security")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_encryption(self):
        """Initialize encryption for sensitive data"""
        try:
            # Generate or load encryption key
            key_file = Path("data/security_encryption.key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                key_file.parent.mkdir(exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
            
            self.cipher = Fernet(self.encryption_key)
            logger.info("Security encryption initialized")
        except Exception as e:
            logger.error(f"Security encryption initialization failed: {e}")
    
    def _init_geoip(self):
        """Initialize GeoIP database"""
        try:
            geoip_path = self.config.get("geoip_database_path", "data/GeoLite2-City.mmdb")
            if Path(geoip_path).exists():
                self.geoip_db = geoip2.database.Reader(geoip_path)
                logger.info("GeoIP database initialized")
            else:
                logger.warning("GeoIP database not found")
        except Exception as e:
            logger.warning(f"GeoIP initialization failed: {e}")
    
    def _init_anomaly_detector(self):
        """Initialize anomaly detection model"""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            logger.info("Anomaly detector initialized")
        except Exception as e:
            logger.error(f"Anomaly detector initialization failed: {e}")
    
    def _init_default_policies(self):
        """Initialize default security policies"""
        
        default_policies = [
            SecurityPolicy(
                policy_id="brute_force_protection",
                name="Brute Force Protection",
                description="Detect and prevent brute force attacks",
                rules=[
                    {"type": "rate_limit", "max_attempts": 5, "window": 300},
                    {"type": "ip_block", "duration": 3600}
                ],
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.PCI_DSS],
                severity=SecurityLevel.HIGH,
                created_at=datetime.now()
            ),
            SecurityPolicy(
                policy_id="sql_injection_protection",
                name="SQL Injection Protection",
                description="Detect and prevent SQL injection attacks",
                rules=[
                    {"type": "pattern_detection", "patterns": ["'", "union", "select", "drop", "insert"]},
                    {"type": "input_validation", "strict": True}
                ],
                compliance_standards=[ComplianceStandard.OWASP],
                severity=SecurityLevel.CRITICAL,
                created_at=datetime.now()
            ),
            SecurityPolicy(
                policy_id="xss_protection",
                name="XSS Protection",
                description="Detect and prevent cross-site scripting attacks",
                rules=[
                    {"type": "pattern_detection", "patterns": ["<script", "javascript:", "onload="]},
                    {"type": "content_security_policy", "enabled": True}
                ],
                compliance_standards=[ComplianceStandard.OWASP],
                severity=SecurityLevel.HIGH,
                created_at=datetime.now()
            )
        ]
        
        for policy in default_policies:
            self.security_policies[policy.policy_id] = policy
            asyncio.create_task(self._store_security_policy(policy))
    
    async def create_security_policy(
        self,
        name: str,
        description: str,
        rules: List[Dict[str, Any]],
        compliance_standards: List[ComplianceStandard],
        severity: SecurityLevel
    ) -> SecurityPolicy:
        """Create a new security policy"""
        
        policy = SecurityPolicy(
            policy_id=str(uuid.uuid4()),
            name=name,
            description=description,
            rules=rules,
            compliance_standards=compliance_standards,
            severity=severity,
            created_at=datetime.now()
        )
        
        self.security_policies[policy.policy_id] = policy
        await self._store_security_policy(policy)
        
        logger.info(f"Security policy created: {policy.policy_id}")
        return policy
    
    async def detect_threat(
        self,
        source_ip: str,
        user_id: Optional[str],
        request_data: Dict[str, Any],
        user_agent: str = ""
    ) -> List[ThreatDetection]:
        """Detect security threats"""
        
        threats = []
        
        try:
            # Check for brute force attacks
            brute_force_threat = await self._detect_brute_force(source_ip, user_id)
            if brute_force_threat:
                threats.append(brute_force_threat)
            
            # Check for SQL injection
            sql_injection_threat = await self._detect_sql_injection(request_data)
            if sql_injection_threat:
                sql_injection_threat.source_ip = source_ip
                sql_injection_threat.user_id = user_id
                threats.append(sql_injection_threat)
            
            # Check for XSS attacks
            xss_threat = await self._detect_xss(request_data)
            if xss_threat:
                xss_threat.source_ip = source_ip
                xss_threat.user_id = user_id
                threats.append(xss_threat)
            
            # Check for suspicious patterns
            suspicious_threat = await self._detect_suspicious_activity(
                source_ip, user_id, request_data, user_agent
            )
            if suspicious_threat:
                threats.append(suspicious_threat)
            
            # Store detected threats
            for threat in threats:
                self.threat_detections[threat.threat_id] = threat
                await self._store_threat_detection(threat)
                
                # Create security incident if high severity
                if threat.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                    await self._create_security_incident(threat)
            
            return threats
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return []
    
    async def _detect_brute_force(
        self,
        source_ip: str,
        user_id: Optional[str]
    ) -> Optional[ThreatDetection]:
        """Detect brute force attacks"""
        
        try:
            if not self.redis_client:
                return None
            
            # Check rate limiting
            key = f"login_attempts:{source_ip}"
            attempts = self.redis_client.get(key)
            
            if attempts:
                attempts = int(attempts)
                if attempts >= 5:  # Threshold from policy
                    return ThreatDetection(
                        threat_id=str(uuid.uuid4()),
                        threat_type=ThreatType.BRUTE_FORCE,
                        severity=SecurityLevel.HIGH,
                        source_ip=source_ip,
                        user_id=user_id,
                        description=f"Brute force attack detected from {source_ip}",
                        indicators=[f"Failed login attempts: {attempts}"],
                        confidence=0.9,
                        detected_at=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Brute force detection failed: {e}")
            return None
    
    async def _detect_sql_injection(self, request_data: Dict[str, Any]) -> Optional[ThreatDetection]:
        """Detect SQL injection attempts"""
        
        try:
            sql_patterns = ["'", "union", "select", "drop", "insert", "delete", "update", "exec", "script"]
            
            # Check all string values in request data
            for key, value in request_data.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    for pattern in sql_patterns:
                        if pattern in value_lower:
                            return ThreatDetection(
                                threat_id=str(uuid.uuid4()),
                                threat_type=ThreatType.SQL_INJECTION,
                                severity=SecurityLevel.CRITICAL,
                                source_ip="",
                                user_id=None,
                                description=f"SQL injection attempt detected in {key}",
                                indicators=[f"Pattern: {pattern}", f"Value: {value[:100]}"],
                                confidence=0.8,
                                detected_at=datetime.now()
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"SQL injection detection failed: {e}")
            return None
    
    async def _detect_xss(self, request_data: Dict[str, Any]) -> Optional[ThreatDetection]:
        """Detect XSS attempts"""
        
        try:
            xss_patterns = ["<script", "javascript:", "onload=", "onerror=", "onclick=", "onmouseover="]
            
            # Check all string values in request data
            for key, value in request_data.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    for pattern in xss_patterns:
                        if pattern in value_lower:
                            return ThreatDetection(
                                threat_id=str(uuid.uuid4()),
                                threat_type=ThreatType.XSS,
                                severity=SecurityLevel.HIGH,
                                source_ip="",
                                user_id=None,
                                description=f"XSS attempt detected in {key}",
                                indicators=[f"Pattern: {pattern}", f"Value: {value[:100]}"],
                                confidence=0.8,
                                detected_at=datetime.now()
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"XSS detection failed: {e}")
            return None
    
    async def _detect_suspicious_activity(
        self,
        source_ip: str,
        user_id: Optional[str],
        request_data: Dict[str, Any],
        user_agent: str
    ) -> Optional[ThreatDetection]:
        """Detect suspicious activity using anomaly detection"""
        
        try:
            # Prepare features for anomaly detection
            features = []
            
            # IP-based features
            try:
                ip_obj = ipaddress.ip_address(source_ip)
                features.extend([
                    1 if ip_obj.is_private else 0,
                    1 if ip_obj.is_loopback else 0,
                    1 if ip_obj.is_multicast else 0
                ])
            except:
                features.extend([0, 0, 0])
            
            # Geographic features (if GeoIP available)
            if self.geoip_db:
                try:
                    response = self.geoip_db.city(source_ip)
                    features.extend([
                        1 if response.country.iso_code == 'US' else 0,
                        1 if response.country.iso_code in ['CN', 'RU', 'KP'] else 0
                    ])
                except:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            # User agent features
            if user_agent:
                parsed_ua = parse(user_agent)
                features.extend([
                    1 if parsed_ua.is_bot else 0,
                    1 if 'curl' in user_agent.lower() else 0,
                    1 if 'wget' in user_agent.lower() else 0
                ])
            else:
                features.extend([0, 0, 0])
            
            # Request data features
            features.extend([
                len(str(request_data)),
                1 if any(key in str(request_data).lower() for key in ['admin', 'root', 'system']) else 0
            ])
            
            # Ensure we have enough features
            while len(features) < 10:
                features.append(0)
            
            # Use anomaly detection
            if self.anomaly_detector:
                # Fit detector if not already fitted
                if not hasattr(self.anomaly_detector, 'decision_function'):
                    # Use dummy data to fit
                    dummy_data = np.random.random((100, 10))
                    self.anomaly_detector.fit(dummy_data)
                
                # Predict anomaly
                anomaly_score = self.anomaly_detector.decision_function([features])[0]
                is_anomaly = self.anomaly_detector.predict([features])[0] == -1
                
                if is_anomaly and anomaly_score < -0.5:
                    return ThreatDetection(
                        threat_id=str(uuid.uuid4()),
                        threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                        severity=SecurityLevel.MEDIUM,
                        source_ip=source_ip,
                        user_id=user_id,
                        description=f"Suspicious activity detected from {source_ip}",
                        indicators=[f"Anomaly score: {anomaly_score:.3f}"],
                        confidence=abs(anomaly_score),
                        detected_at=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Suspicious activity detection failed: {e}")
            return None
    
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        source_ip: str,
        user_agent: str,
        action: str,
        resource: str,
        details: Dict[str, Any],
        success: bool = True
    ) -> AuditLog:
        """Log security audit event"""
        
        audit_log = AuditLog(
            log_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            source_ip=source_ip,
            user_agent=user_agent,
            action=action,
            resource=resource,
            details=details,
            timestamp=datetime.now(),
            success=success
        )
        
        # Store in memory (with size limit)
        self.audit_logs.append(audit_log)
        if len(self.audit_logs) > 10000:  # Keep last 10k logs in memory
            self.audit_logs = self.audit_logs[-10000:]
        
        # Store in database
        await self._store_audit_log(audit_log)
        
        # Cache in Redis for quick access
        if self.redis_client:
            cache_key = f"audit_log:{audit_log.log_id}"
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(asdict(audit_log), default=str)
            )
        
        return audit_log
    
    async def _create_security_incident(self, threat: ThreatDetection):
        """Create security incident from threat detection"""
        
        incident = SecurityIncident(
            incident_id=str(uuid.uuid4()),
            title=f"{threat.threat_type.value.title()} Threat Detected",
            description=threat.description,
            threat_type=threat.threat_type,
            severity=threat.severity,
            created_at=datetime.now()
        )
        
        self.security_incidents[incident.incident_id] = incident
        await self._store_security_incident(incident)
        
        # Log security event
        await self.log_audit_event(
            event_type=AuditEventType.SECURITY_EVENT,
            user_id=threat.user_id,
            source_ip=threat.source_ip,
            user_agent="",
            action="threat_detected",
            resource="security_system",
            details={
                "threat_id": threat.threat_id,
                "incident_id": incident.incident_id,
                "threat_type": threat.threat_type.value,
                "severity": threat.severity.value
            }
        )
        
        logger.warning(f"Security incident created: {incident.incident_id}")
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if request is within rate limit"""
        
        try:
            if not self.redis_client:
                return True  # Allow if Redis not available
            
            key = f"rate_limit:{identifier}"
            current = self.redis_client.get(key)
            
            if current is None:
                self.redis_client.setex(key, window, 1)
                return True
            elif int(current) < limit:
                self.redis_client.incr(key)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error
    
    async def block_ip(self, ip_address: str, duration: int = 3600):
        """Block IP address"""
        
        try:
            self.blacklisted_ips.add(ip_address)
            
            if self.redis_client:
                key = f"blocked_ip:{ip_address}"
                self.redis_client.setex(key, duration, "blocked")
            
            # Log security event
            await self.log_audit_event(
                event_type=AuditEventType.SECURITY_EVENT,
                user_id=None,
                source_ip=ip_address,
                user_agent="",
                action="ip_blocked",
                resource="security_system",
                details={"duration": duration, "reason": "security_threat"}
            )
            
            logger.warning(f"IP address blocked: {ip_address}")
            
        except Exception as e:
            logger.error(f"IP blocking failed: {e}")
    
    async def unblock_ip(self, ip_address: str):
        """Unblock IP address"""
        
        try:
            self.blacklisted_ips.discard(ip_address)
            
            if self.redis_client:
                key = f"blocked_ip:{ip_address}"
                self.redis_client.delete(key)
            
            # Log security event
            await self.log_audit_event(
                event_type=AuditEventType.SECURITY_EVENT,
                user_id=None,
                source_ip=ip_address,
                user_agent="",
                action="ip_unblocked",
                resource="security_system",
                details={"reason": "manual_unblock"}
            )
            
            logger.info(f"IP address unblocked: {ip_address}")
            
        except Exception as e:
            logger.error(f"IP unblocking failed: {e}")
    
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        
        try:
            if ip_address in self.blacklisted_ips:
                return True
            
            if self.redis_client:
                key = f"blocked_ip:{ip_address}"
                return self.redis_client.exists(key) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"IP block check failed: {e}")
            return False
    
    async def generate_security_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate security report"""
        
        try:
            # Get threats in date range
            threats_in_range = [
                t for t in self.threat_detections.values()
                if start_date <= t.detected_at <= end_date
            ]
            
            # Get incidents in date range
            incidents_in_range = [
                i for i in self.security_incidents.values()
                if start_date <= i.created_at <= end_date
            ]
            
            # Get audit logs in date range
            audit_logs_in_range = [
                a for a in self.audit_logs
                if start_date <= a.timestamp <= end_date
            ]
            
            # Calculate statistics
            threat_stats = defaultdict(int)
            for threat in threats_in_range:
                threat_stats[threat.threat_type.value] += 1
            
            severity_stats = defaultdict(int)
            for threat in threats_in_range:
                severity_stats[threat.severity.value] += 1
            
            # Top source IPs
            ip_stats = defaultdict(int)
            for threat in threats_in_range:
                ip_stats[threat.source_ip] += 1
            top_ips = sorted(ip_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            
            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_threats": len(threats_in_range),
                    "total_incidents": len(incidents_in_range),
                    "total_audit_events": len(audit_logs_in_range)
                },
                "threat_breakdown": dict(threat_stats),
                "severity_breakdown": dict(severity_stats),
                "top_source_ips": [{"ip": ip, "count": count} for ip, count in top_ips],
                "incidents": [
                    {
                        "incident_id": i.incident_id,
                        "title": i.title,
                        "severity": i.severity.value,
                        "status": i.status,
                        "created_at": i.created_at.isoformat()
                    }
                    for i in incidents_in_range
                ],
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Security report generation failed: {e}")
            return {"error": str(e)}
    
    async def _store_security_policy(self, policy: SecurityPolicy):
        """Store security policy in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO security_policies
                (policy_id, name, description, rules, compliance_standards, severity, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                policy.policy_id,
                policy.name,
                policy.description,
                json.dumps(policy.rules),
                json.dumps([s.value for s in policy.compliance_standards]),
                policy.severity.value,
                policy.enabled,
                policy.created_at.isoformat()
            ))
            conn.commit()
    
    async def _store_threat_detection(self, threat: ThreatDetection):
        """Store threat detection in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO threat_detections
                (threat_id, threat_type, severity, source_ip, user_id, description, indicators, confidence, detected_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threat.threat_id,
                threat.threat_type.value,
                threat.severity.value,
                threat.source_ip,
                threat.user_id,
                threat.description,
                json.dumps(threat.indicators),
                threat.confidence,
                threat.detected_at.isoformat(),
                threat.status
            ))
            conn.commit()
    
    async def _store_audit_log(self, audit_log: AuditLog):
        """Store audit log in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs
                (log_id, event_type, user_id, source_ip, user_agent, action, resource, details, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_log.log_id,
                audit_log.event_type.value,
                audit_log.user_id,
                audit_log.source_ip,
                audit_log.user_agent,
                audit_log.action,
                audit_log.resource,
                json.dumps(audit_log.details),
                audit_log.success,
                audit_log.timestamp.isoformat()
            ))
            conn.commit()
    
    async def _store_security_incident(self, incident: SecurityIncident):
        """Store security incident in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO security_incidents
                (incident_id, title, description, threat_type, severity, status, assigned_to, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident.incident_id,
                incident.title,
                incident.description,
                incident.threat_type.value,
                incident.severity.value,
                incident.status,
                incident.assigned_to,
                incident.created_at.isoformat(),
                incident.resolved_at.isoformat() if incident.resolved_at else None
            ))
            conn.commit()
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.redis_client:
            self.redis_client.close()
        
        if self.geoip_db:
            self.geoip_db.close()
        
        logger.info("Advanced security service cleanup completed")

# Global instance
advanced_security_service = None

async def get_advanced_security_service() -> AdvancedSecurityService:
    """Get global advanced security service instance"""
    global advanced_security_service
    if not advanced_security_service:
        config = {
            "database_path": "data/security.db",
            "redis_url": "redis://localhost:6379",
            "geoip_database_path": "data/GeoLite2-City.mmdb"
        }
        advanced_security_service = AdvancedSecurityService(config)
    return advanced_security_service



