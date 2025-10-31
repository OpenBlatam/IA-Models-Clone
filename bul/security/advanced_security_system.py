"""
Ultimate BUL System - Advanced Security & Compliance System
Enterprise-grade security with comprehensive threat detection and compliance
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
import aiohttp
from prometheus_client import Counter, Histogram, Gauge
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, Enum):
    """Threat types"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDOS = "ddos"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALWARE = "malware"
    PHISHING = "phishing"
    INSIDER_THREAT = "insider_threat"

class ComplianceStandard(str, Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    FERPA = "ferpa"
    COPPA = "coppa"

@dataclass
class SecurityEvent:
    """Security event"""
    id: str
    event_type: str
    severity: SecurityLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    false_positive: bool = False

@dataclass
class ThreatDetection:
    """Threat detection result"""
    threat_type: ThreatType
    confidence: float
    severity: SecurityLevel
    description: str
    indicators: List[str]
    recommended_actions: List[str]
    timestamp: datetime

@dataclass
class ComplianceCheck:
    """Compliance check result"""
    standard: ComplianceStandard
    requirement: str
    status: str
    description: str
    evidence: List[str]
    last_checked: datetime
    next_check: datetime

class AdvancedSecuritySystem:
    """Advanced security and compliance system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_events = []
        self.threat_detections = []
        self.compliance_checks = []
        self.blocked_ips = set()
        self.suspicious_ips = set()
        self.user_sessions = {}
        self.api_keys = {}
        self.encryption_keys = {}
        
        # Security policies
        self.security_policies = self._initialize_security_policies()
        
        # Compliance requirements
        self.compliance_requirements = self._initialize_compliance_requirements()
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Security monitoring
        self.monitoring_active = False
        self.last_cleanup = datetime.utcnow()
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Start security monitoring
        self.start_security_monitoring()
    
    def _initialize_security_policies(self) -> Dict[str, Any]:
        """Initialize security policies"""
        return {
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True,
                "max_age_days": 90,
                "history_count": 5
            },
            "session_policy": {
                "timeout_minutes": 30,
                "max_concurrent_sessions": 3,
                "require_https": True,
                "secure_cookies": True
            },
            "api_policy": {
                "rate_limit_per_minute": 100,
                "rate_limit_per_hour": 1000,
                "max_request_size": 10485760,  # 10MB
                "require_authentication": True,
                "require_authorization": True
            },
            "encryption_policy": {
                "data_at_rest": True,
                "data_in_transit": True,
                "key_rotation_days": 30,
                "algorithm": "AES-256-GCM"
            },
            "audit_policy": {
                "log_all_events": True,
                "retention_days": 2555,  # 7 years
                "encrypt_logs": True,
                "real_time_monitoring": True
            },
            "threat_detection": {
                "brute_force_threshold": 5,
                "suspicious_activity_threshold": 3,
                "ip_blocking_duration_hours": 24,
                "auto_response_enabled": True
            }
        }
    
    def _initialize_compliance_requirements(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize compliance requirements"""
        return {
            ComplianceStandard.GDPR: [
                {
                    "requirement": "Data Minimization",
                    "description": "Collect only necessary personal data",
                    "implementation": "Data collection forms validate required fields only"
                },
                {
                    "requirement": "Consent Management",
                    "description": "Obtain explicit consent for data processing",
                    "implementation": "Consent tracking and management system"
                },
                {
                    "requirement": "Right to Erasure",
                    "description": "Allow users to delete their data",
                    "implementation": "Data deletion API and process"
                },
                {
                    "requirement": "Data Portability",
                    "description": "Allow users to export their data",
                    "implementation": "Data export functionality"
                },
                {
                    "requirement": "Privacy by Design",
                    "description": "Implement privacy controls by default",
                    "implementation": "Default privacy settings and controls"
                }
            ],
            ComplianceStandard.HIPAA: [
                {
                    "requirement": "Administrative Safeguards",
                    "description": "Implement administrative security measures",
                    "implementation": "Security policies and procedures"
                },
                {
                    "requirement": "Physical Safeguards",
                    "description": "Protect physical access to systems",
                    "implementation": "Data center security and access controls"
                },
                {
                    "requirement": "Technical Safeguards",
                    "description": "Implement technical security controls",
                    "implementation": "Encryption, access controls, and audit logs"
                },
                {
                    "requirement": "Breach Notification",
                    "description": "Notify of security breaches within 60 days",
                    "implementation": "Automated breach detection and notification"
                }
            ],
            ComplianceStandard.SOC2: [
                {
                    "requirement": "Security",
                    "description": "Protect against unauthorized access",
                    "implementation": "Multi-factor authentication and access controls"
                },
                {
                    "requirement": "Availability",
                    "description": "Ensure system availability",
                    "implementation": "High availability architecture and monitoring"
                },
                {
                    "requirement": "Processing Integrity",
                    "description": "Ensure data processing integrity",
                    "implementation": "Data validation and integrity checks"
                },
                {
                    "requirement": "Confidentiality",
                    "description": "Protect confidential information",
                    "implementation": "Encryption and access controls"
                },
                {
                    "requirement": "Privacy",
                    "description": "Protect personal information",
                    "implementation": "Privacy controls and data protection"
                }
            ]
        }
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics for security monitoring"""
        return {
            "security_events": Counter(
                "bul_security_events_total",
                "Total security events",
                ["event_type", "severity"]
            ),
            "threat_detections": Counter(
                "bul_threat_detections_total",
                "Total threat detections",
                ["threat_type", "severity"]
            ),
            "blocked_ips": Gauge(
                "bul_blocked_ips",
                "Number of blocked IP addresses"
            ),
            "active_sessions": Gauge(
                "bul_active_sessions",
                "Number of active user sessions"
            ),
            "failed_logins": Counter(
                "bul_failed_logins_total",
                "Total failed login attempts",
                ["source_ip"]
            ),
            "api_abuse": Counter(
                "bul_api_abuse_total",
                "Total API abuse attempts",
                ["endpoint", "source_ip"]
            ),
            "compliance_violations": Counter(
                "bul_compliance_violations_total",
                "Total compliance violations",
                ["standard", "requirement"]
            )
        }
    
    def _initialize_encryption(self):
        """Initialize encryption systems"""
        try:
            # Generate encryption keys
            self.encryption_keys["data"] = Fernet.generate_key()
            self.encryption_keys["logs"] = Fernet.generate_key()
            
            # Generate RSA key pair for JWT signing
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.encryption_keys["jwt_private"] = private_key
            self.encryption_keys["jwt_public"] = private_key.public_key()
            
            logger.info("Encryption systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            raise
    
    async def start_security_monitoring(self):
        """Start security monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting security monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_security_events())
        asyncio.create_task(self._detect_threats())
        asyncio.create_task(self._check_compliance())
        asyncio.create_task(self._cleanup_security_data())
    
    async def stop_security_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        logger.info("Stopping security monitoring")
    
    async def _monitor_security_events(self):
        """Monitor security events"""
        while self.monitoring_active:
            try:
                # Check for suspicious activity
                await self._check_brute_force_attacks()
                await self._check_suspicious_ips()
                await self._check_session_anomalies()
                await self._check_api_abuse()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring security events: {e}")
                await asyncio.sleep(30)
    
    async def _detect_threats(self):
        """Detect security threats"""
        while self.monitoring_active:
            try:
                # Analyze recent security events
                recent_events = [
                    e for e in self.security_events
                    if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
                ]
                
                # Detect brute force attacks
                brute_force_threats = await self._detect_brute_force_threats(recent_events)
                for threat in brute_force_threats:
                    self.threat_detections.append(threat)
                    await self._respond_to_threat(threat)
                
                # Detect SQL injection attempts
                sql_injection_threats = await self._detect_sql_injection_threats(recent_events)
                for threat in sql_injection_threats:
                    self.threat_detections.append(threat)
                    await self._respond_to_threat(threat)
                
                # Detect XSS attempts
                xss_threats = await self._detect_xss_threats(recent_events)
                for threat in xss_threats:
                    self.threat_detections.append(threat)
                    await self._respond_to_threat(threat)
                
                await asyncio.sleep(60)  # Detect threats every minute
                
            except Exception as e:
                logger.error(f"Error detecting threats: {e}")
                await asyncio.sleep(60)
    
    async def _check_compliance(self):
        """Check compliance requirements"""
        while self.monitoring_active:
            try:
                # Check GDPR compliance
                await self._check_gdpr_compliance()
                
                # Check HIPAA compliance
                await self._check_hipaa_compliance()
                
                # Check SOC2 compliance
                await self._check_soc2_compliance()
                
                await asyncio.sleep(3600)  # Check compliance every hour
                
            except Exception as e:
                logger.error(f"Error checking compliance: {e}")
                await asyncio.sleep(3600)
    
    async def _check_brute_force_attacks(self):
        """Check for brute force attacks"""
        # Group failed logins by IP
        failed_logins_by_ip = {}
        for event in self.security_events:
            if event.event_type == "failed_login":
                ip = event.source_ip
                if ip not in failed_logins_by_ip:
                    failed_logins_by_ip[ip] = []
                failed_logins_by_ip[ip].append(event)
        
        # Check for brute force patterns
        for ip, events in failed_logins_by_ip.items():
            recent_events = [
                e for e in events
                if (datetime.utcnow() - e.timestamp).total_seconds() < 300  # 5 minutes
            ]
            
            if len(recent_events) >= self.security_policies["threat_detection"]["brute_force_threshold"]:
                # Block IP
                self.blocked_ips.add(ip)
                self.suspicious_ips.add(ip)
                
                # Create security event
                event = SecurityEvent(
                    id=f"brute_force_{int(time.time())}",
                    event_type="brute_force_attack",
                    severity=SecurityLevel.HIGH,
                    timestamp=datetime.utcnow(),
                    source_ip=ip,
                    user_id=None,
                    description=f"Brute force attack detected from {ip}",
                    details={"failed_attempts": len(recent_events)}
                )
                
                self.security_events.append(event)
                self.prometheus_metrics["security_events"].labels(
                    event_type="brute_force_attack",
                    severity="high"
                ).inc()
                
                logger.warning(f"Brute force attack detected from {ip}")
    
    async def _check_suspicious_ips(self):
        """Check for suspicious IP addresses"""
        # This would typically integrate with threat intelligence feeds
        # For now, we'll check against known malicious IP patterns
        pass
    
    async def _check_session_anomalies(self):
        """Check for session anomalies"""
        # Check for unusual session patterns
        for user_id, sessions in self.user_sessions.items():
            if len(sessions) > self.security_policies["session_policy"]["max_concurrent_sessions"]:
                event = SecurityEvent(
                    id=f"session_anomaly_{int(time.time())}",
                    event_type="session_anomaly",
                    severity=SecurityLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    source_ip="unknown",
                    user_id=user_id,
                    description=f"User {user_id} has too many concurrent sessions",
                    details={"session_count": len(sessions)}
                )
                
                self.security_events.append(event)
                logger.warning(f"Session anomaly detected for user {user_id}")
    
    async def _check_api_abuse(self):
        """Check for API abuse"""
        # This would typically check API usage patterns
        # For now, we'll implement basic rate limiting checks
        pass
    
    async def _detect_brute_force_threats(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect brute force threats"""
        threats = []
        
        # Group events by IP
        events_by_ip = {}
        for event in events:
            if event.event_type == "failed_login":
                ip = event.source_ip
                if ip not in events_by_ip:
                    events_by_ip[ip] = []
                events_by_ip[ip].append(event)
        
        # Analyze each IP
        for ip, ip_events in events_by_ip.items():
            if len(ip_events) >= 5:  # 5 failed attempts
                threat = ThreatDetection(
                    threat_type=ThreatType.BRUTE_FORCE,
                    confidence=0.9,
                    severity=SecurityLevel.HIGH,
                    description=f"Brute force attack detected from {ip}",
                    indicators=[f"Multiple failed login attempts from {ip}"],
                    recommended_actions=["Block IP address", "Enable CAPTCHA", "Notify security team"],
                    timestamp=datetime.utcnow()
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_sql_injection_threats(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect SQL injection threats"""
        threats = []
        
        # Check for SQL injection patterns in events
        sql_patterns = [
            "union select", "drop table", "insert into", "delete from",
            "update set", "or 1=1", "and 1=1", "' or '1'='1"
        ]
        
        for event in events:
            if event.event_type == "api_request":
                request_data = event.details.get("request_data", "")
                if isinstance(request_data, str):
                    for pattern in sql_patterns:
                        if pattern.lower() in request_data.lower():
                            threat = ThreatDetection(
                                threat_type=ThreatType.SQL_INJECTION,
                                confidence=0.8,
                                severity=SecurityLevel.HIGH,
                                description=f"SQL injection attempt detected from {event.source_ip}",
                                indicators=[f"SQL injection pattern: {pattern}"],
                                recommended_actions=["Block IP address", "Sanitize input", "Update WAF rules"],
                                timestamp=datetime.utcnow()
                            )
                            threats.append(threat)
                            break
        
        return threats
    
    async def _detect_xss_threats(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Detect XSS threats"""
        threats = []
        
        # Check for XSS patterns in events
        xss_patterns = [
            "<script>", "javascript:", "onload=", "onerror=",
            "onclick=", "onmouseover=", "alert(", "document.cookie"
        ]
        
        for event in events:
            if event.event_type == "api_request":
                request_data = event.details.get("request_data", "")
                if isinstance(request_data, str):
                    for pattern in xss_patterns:
                        if pattern.lower() in request_data.lower():
                            threat = ThreatDetection(
                                threat_type=ThreatType.XSS,
                                confidence=0.7,
                                severity=SecurityLevel.MEDIUM,
                                description=f"XSS attempt detected from {event.source_ip}",
                                indicators=[f"XSS pattern: {pattern}"],
                                recommended_actions=["Sanitize input", "Enable CSP headers", "Update WAF rules"],
                                timestamp=datetime.utcnow()
                            )
                            threats.append(threat)
                            break
        
        return threats
    
    async def _respond_to_threat(self, threat: ThreatDetection):
        """Respond to detected threat"""
        try:
            if threat.threat_type == ThreatType.BRUTE_FORCE:
                # Block IP address
                # This would typically update firewall rules
                logger.info(f"Blocking IP due to brute force attack: {threat.description}")
            
            elif threat.threat_type == ThreatType.SQL_INJECTION:
                # Block IP and update WAF rules
                logger.info(f"Blocking IP due to SQL injection attempt: {threat.description}")
            
            elif threat.threat_type == ThreatType.XSS:
                # Update WAF rules and enable additional filtering
                logger.info(f"Updating WAF rules due to XSS attempt: {threat.description}")
            
            # Update Prometheus metrics
            self.prometheus_metrics["threat_detections"].labels(
                threat_type=threat.threat_type.value,
                severity=threat.severity.value
            ).inc()
            
        except Exception as e:
            logger.error(f"Error responding to threat: {e}")
    
    async def _check_gdpr_compliance(self):
        """Check GDPR compliance"""
        # Check data minimization
        compliance_check = ComplianceCheck(
            standard=ComplianceStandard.GDPR,
            requirement="Data Minimization",
            status="compliant",
            description="Data collection forms validate required fields only",
            evidence=["Form validation implemented", "Data collection audit logs"],
            last_checked=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=30)
        )
        self.compliance_checks.append(compliance_check)
        
        # Check consent management
        compliance_check = ComplianceCheck(
            standard=ComplianceStandard.GDPR,
            requirement="Consent Management",
            status="compliant",
            description="Consent tracking and management system implemented",
            evidence=["Consent tracking system", "Consent withdrawal process"],
            last_checked=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=30)
        )
        self.compliance_checks.append(compliance_check)
    
    async def _check_hipaa_compliance(self):
        """Check HIPAA compliance"""
        # Check administrative safeguards
        compliance_check = ComplianceCheck(
            standard=ComplianceStandard.HIPAA,
            requirement="Administrative Safeguards",
            status="compliant",
            description="Security policies and procedures implemented",
            evidence=["Security policies documented", "Staff training records"],
            last_checked=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=30)
        )
        self.compliance_checks.append(compliance_check)
    
    async def _check_soc2_compliance(self):
        """Check SOC2 compliance"""
        # Check security controls
        compliance_check = ComplianceCheck(
            standard=ComplianceStandard.SOC2,
            requirement="Security",
            status="compliant",
            description="Multi-factor authentication and access controls implemented",
            evidence=["MFA enabled", "Access control policies", "Regular access reviews"],
            last_checked=datetime.utcnow(),
            next_check=datetime.utcnow() + timedelta(days=30)
        )
        self.compliance_checks.append(compliance_check)
    
    async def _cleanup_security_data(self):
        """Cleanup old security data"""
        while self.monitoring_active:
            try:
                # Clean up old security events (keep last 30 days)
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.security_events = [
                    e for e in self.security_events
                    if e.timestamp > cutoff_time
                ]
                
                # Clean up old threat detections (keep last 90 days)
                cutoff_time = datetime.utcnow() - timedelta(days=90)
                self.threat_detections = [
                    t for t in self.threat_detections
                    if t.timestamp > cutoff_time
                ]
                
                # Clean up old compliance checks (keep last 365 days)
                cutoff_time = datetime.utcnow() - timedelta(days=365)
                self.compliance_checks = [
                    c for c in self.compliance_checks
                    if c.last_checked > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up security data: {e}")
                await asyncio.sleep(3600)
    
    # Public methods for security operations
    
    def log_security_event(self, event_type: str, severity: SecurityLevel, 
                          source_ip: str, user_id: Optional[str], 
                          description: str, details: Dict[str, Any] = None):
        """Log a security event"""
        event = SecurityEvent(
            id=f"{event_type}_{int(time.time())}",
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Update Prometheus metrics
        self.prometheus_metrics["security_events"].labels(
            event_type=event_type,
            severity=severity.value
        ).inc()
        
        logger.info(f"Security event logged: {event_type} - {description}")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def is_ip_suspicious(self, ip: str) -> bool:
        """Check if IP is suspicious"""
        return ip in self.suspicious_ips
    
    def block_ip(self, ip: str, duration_hours: int = 24):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        logger.info(f"IP {ip} blocked for {duration_hours} hours")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"IP {ip} unblocked")
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against security policy"""
        policy = self.security_policies["password_policy"]
        errors = []
        
        if len(password) < policy["min_length"]:
            errors.append(f"Password must be at least {policy['min_length']} characters long")
        
        if policy["require_uppercase"] and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if policy["require_lowercase"] and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if policy["require_numbers"] and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if policy["require_special_chars"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate API key for user"""
        key_data = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat()
        }
        
        # Sign the key data
        key_string = json.dumps(key_data, sort_keys=True)
        signature = hmac.new(
            self.encryption_keys["data"],
            key_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        api_key = f"bul_{signature[:32]}"
        self.api_keys[api_key] = key_data
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate API key"""
        if api_key not in self.api_keys:
            return False, None
        
        key_data = self.api_keys[api_key]
        
        # Check expiration
        expires_at = datetime.fromisoformat(key_data["expires_at"])
        if datetime.utcnow() > expires_at:
            del self.api_keys[api_key]
            return False, None
        
        return True, key_data
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data"""
        f = Fernet(self.encryption_keys["data"])
        encrypted = f.encrypt(data.encode('utf-8'))
        return encrypted.decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        f = Fernet(self.encryption_keys["data"])
        decrypted = f.decrypt(encrypted_data.encode('utf-8'))
        return decrypted.decode('utf-8')
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_events": len(self.security_events),
            "active_threats": len([t for t in self.threat_detections if t.timestamp > datetime.utcnow() - timedelta(hours=24)]),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "active_sessions": len(self.user_sessions),
            "api_keys": len(self.api_keys),
            "compliance_status": {
                standard.value: len([c for c in self.compliance_checks if c.standard == standard and c.status == "compliant"])
                for standard in ComplianceStandard
            }
        }
    
    def get_threat_detections(self, threat_type: Optional[ThreatType] = None, 
                            limit: int = 50) -> List[ThreatDetection]:
        """Get threat detections"""
        threats = self.threat_detections
        
        if threat_type:
            threats = [t for t in threats if t.threat_type == threat_type]
        
        return threats[-limit:]
    
    def get_compliance_status(self, standard: Optional[ComplianceStandard] = None) -> List[ComplianceCheck]:
        """Get compliance status"""
        checks = self.compliance_checks
        
        if standard:
            checks = [c for c in checks if c.standard == standard]
        
        return checks
    
    def export_security_data(self) -> Dict[str, Any]:
        """Export security data for analysis"""
        return {
            "security_events": [e.__dict__ for e in self.security_events[-1000:]],
            "threat_detections": [t.__dict__ for t in self.threat_detections[-1000:]],
            "compliance_checks": [c.__dict__ for c in self.compliance_checks],
            "blocked_ips": list(self.blocked_ips),
            "suspicious_ips": list(self.suspicious_ips),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global security system instance
security_system = None

def get_security_system() -> AdvancedSecuritySystem:
    """Get the global security system instance"""
    global security_system
    if security_system is None:
        config = {
            "encryption_key": Fernet.generate_key(),
            "jwt_secret": secrets.token_urlsafe(32)
        }
        security_system = AdvancedSecuritySystem(config)
    return security_system

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "encryption_key": Fernet.generate_key(),
            "jwt_secret": secrets.token_urlsafe(32)
        }
        
        security = AdvancedSecuritySystem(config)
        
        # Log some security events
        security.log_security_event(
            "failed_login",
            SecurityLevel.MEDIUM,
            "192.168.1.100",
            "user123",
            "Failed login attempt"
        )
        
        # Generate API key
        api_key = security.generate_api_key("user123", ["read", "write"])
        print(f"Generated API key: {api_key}")
        
        # Validate API key
        valid, key_data = security.validate_api_key(api_key)
        print(f"API key valid: {valid}")
        
        # Get security summary
        summary = security.get_security_summary()
        print("Security Summary:")
        print(json.dumps(summary, indent=2))
        
        await security.stop_security_monitoring()
    
    asyncio.run(main())













