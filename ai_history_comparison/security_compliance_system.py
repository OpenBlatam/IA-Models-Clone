"""
Security & Compliance System
============================

Advanced security and compliance system for AI model analysis with
comprehensive security controls, compliance monitoring, and audit capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import hashlib as hashlib_module
import hmac
import secrets
import jwt
import bcrypt
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import ssl
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(str, Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"
    CCPA = "ccpa"
    FERPA = "ferpa"


class SecurityEventType(str, Enum):
    """Security event types"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    DATA_BREACH = "data_breach"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_INTRUSION = "system_intrusion"
    DATA_LEAKAGE = "data_leakage"
    MALWARE_DETECTED = "malware_detected"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class ComplianceStatus(str, Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    security_level: SecurityLevel
    compliance_standards: List[ComplianceStandard]
    rules: List[Dict[str, Any]]
    enforcement_actions: List[str]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SecurityEvent:
    """Security event"""
    event_id: str
    event_type: SecurityEventType
    severity: SecurityLevel
    description: str
    source_ip: str
    user_id: str
    resource: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComplianceCheck:
    """Compliance check"""
    check_id: str
    standard: ComplianceStandard
    requirement: str
    description: str
    status: ComplianceStatus
    evidence: List[str]
    last_checked: datetime
    next_check: datetime
    remediation_actions: List[str] = None
    
    def __post_init__(self):
        if self.remediation_actions is None:
            self.remediation_actions = []


@dataclass
class AuditLog:
    """Audit log entry"""
    log_id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SecurityComplianceSystem:
    """Advanced security and compliance system for AI model analysis"""
    
    def __init__(self, max_events: int = 10000, max_logs: int = 100000):
        self.max_events = max_events
        self.max_logs = max_logs
        
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.security_events: List[SecurityEvent] = []
        self.compliance_checks: List[ComplianceCheck] = []
        self.audit_logs: List[AuditLog] = []
        
        # Encryption keys
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Security settings
        self.password_min_length = 12
        self.password_require_special = True
        self.password_require_numbers = True
        self.password_require_uppercase = True
        self.password_require_lowercase = True
        self.session_timeout = 3600  # 1 hour
        self.max_login_attempts = 5
        self.lockout_duration = 1800  # 30 minutes
        
        # Initialize default policies
        self._initialize_default_policies()
        self._initialize_compliance_checks()
    
    async def create_security_policy(self, 
                                   name: str,
                                   description: str,
                                   security_level: SecurityLevel,
                                   compliance_standards: List[ComplianceStandard],
                                   rules: List[Dict[str, Any]],
                                   enforcement_actions: List[str]) -> SecurityPolicy:
        """Create security policy"""
        try:
            policy_id = hashlib.md5(f"{name}_{security_level}_{datetime.now()}".encode()).hexdigest()
            
            policy = SecurityPolicy(
                policy_id=policy_id,
                name=name,
                description=description,
                security_level=security_level,
                compliance_standards=compliance_standards,
                rules=rules,
                enforcement_actions=enforcement_actions
            )
            
            self.security_policies[policy_id] = policy
            
            logger.info(f"Created security policy: {name}")
            
            return policy
            
        except Exception as e:
            logger.error(f"Error creating security policy: {str(e)}")
            raise e
    
    async def log_security_event(self, 
                               event_type: SecurityEventType,
                               severity: SecurityLevel,
                               description: str,
                               source_ip: str,
                               user_id: str,
                               resource: str,
                               metadata: Dict[str, Any] = None) -> SecurityEvent:
        """Log security event"""
        try:
            event_id = hashlib.md5(f"{event_type}_{source_ip}_{datetime.now()}".encode()).hexdigest()
            
            if metadata is None:
                metadata = {}
            
            event = SecurityEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                description=description,
                source_ip=source_ip,
                user_id=user_id,
                resource=resource,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            self.security_events.append(event)
            
            # Check if event triggers any policies
            await self._check_security_policies(event)
            
            logger.warning(f"Security event logged: {event_type.value} - {description}")
            
            return event
            
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")
            raise e
    
    async def log_audit_event(self, 
                            user_id: str,
                            action: str,
                            resource: str,
                            ip_address: str,
                            user_agent: str,
                            success: bool,
                            details: Dict[str, Any] = None) -> AuditLog:
        """Log audit event"""
        try:
            log_id = hashlib.md5(f"{user_id}_{action}_{datetime.now()}".encode()).hexdigest()
            
            if details is None:
                details = {}
            
            audit_log = AuditLog(
                log_id=log_id,
                user_id=user_id,
                action=action,
                resource=resource,
                timestamp=datetime.now(),
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                details=details
            )
            
            self.audit_logs.append(audit_log)
            
            logger.info(f"Audit event logged: {user_id} - {action} - {resource}")
            
            return audit_log
            
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
            raise e
    
    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return encrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise e
    
    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise e
    
    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error hashing password: {str(e)}")
            raise e
    
    async def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False
    
    async def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        try:
            validation_result = {
                "valid": True,
                "score": 0,
                "issues": [],
                "recommendations": []
            }
            
            # Check length
            if len(password) < self.password_min_length:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Password must be at least {self.password_min_length} characters long")
            else:
                validation_result["score"] += 1
            
            # Check for special characters
            if self.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                validation_result["valid"] = False
                validation_result["issues"].append("Password must contain at least one special character")
            else:
                validation_result["score"] += 1
            
            # Check for numbers
            if self.password_require_numbers and not any(c.isdigit() for c in password):
                validation_result["valid"] = False
                validation_result["issues"].append("Password must contain at least one number")
            else:
                validation_result["score"] += 1
            
            # Check for uppercase
            if self.password_require_uppercase and not any(c.isupper() for c in password):
                validation_result["valid"] = False
                validation_result["issues"].append("Password must contain at least one uppercase letter")
            else:
                validation_result["score"] += 1
            
            # Check for lowercase
            if self.password_require_lowercase and not any(c.islower() for c in password):
                validation_result["valid"] = False
                validation_result["issues"].append("Password must contain at least one lowercase letter")
            else:
                validation_result["score"] += 1
            
            # Check for common patterns
            common_patterns = ["password", "123456", "qwerty", "admin", "user"]
            if any(pattern in password.lower() for pattern in common_patterns):
                validation_result["valid"] = False
                validation_result["issues"].append("Password contains common patterns")
                validation_result["score"] -= 1
            
            # Generate recommendations
            if not validation_result["valid"]:
                validation_result["recommendations"] = [
                    "Use a combination of letters, numbers, and special characters",
                    "Avoid common words or patterns",
                    "Make it at least 12 characters long",
                    "Consider using a password manager"
                ]
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating password strength: {str(e)}")
            return {"valid": False, "score": 0, "issues": ["Validation error"], "recommendations": []}
    
    async def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        try:
            return secrets.token_urlsafe(length)
            
        except Exception as e:
            logger.error(f"Error generating secure token: {str(e)}")
            raise e
    
    async def create_jwt_token(self, 
                             payload: Dict[str, Any], 
                             secret_key: str, 
                             expires_in: int = 3600) -> str:
        """Create JWT token"""
        try:
            payload["exp"] = datetime.utcnow() + timedelta(seconds=expires_in)
            payload["iat"] = datetime.utcnow()
            
            token = jwt.encode(payload, secret_key, algorithm="HS256")
            return token
            
        except Exception as e:
            logger.error(f"Error creating JWT token: {str(e)}")
            raise e
    
    async def verify_jwt_token(self, token: str, secret_key: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            return payload
            
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "error": "Invalid token"}
        except Exception as e:
            logger.error(f"Error verifying JWT token: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    async def check_compliance(self, 
                             standard: ComplianceStandard,
                             requirement: str = None) -> Dict[str, Any]:
        """Check compliance with standard"""
        try:
            compliance_result = {
                "standard": standard.value,
                "overall_status": ComplianceStatus.COMPLIANT,
                "checks": [],
                "issues": [],
                "recommendations": []
            }
            
            # Get relevant compliance checks
            relevant_checks = [c for c in self.compliance_checks if c.standard == standard]
            if requirement:
                relevant_checks = [c for c in relevant_checks if requirement.lower() in c.requirement.lower()]
            
            for check in relevant_checks:
                check_result = {
                    "requirement": check.requirement,
                    "status": check.status.value,
                    "last_checked": check.last_checked.isoformat(),
                    "evidence": check.evidence
                }
                compliance_result["checks"].append(check_result)
                
                if check.status != ComplianceStatus.COMPLIANT:
                    compliance_result["overall_status"] = ComplianceStatus.NON_COMPLIANT
                    compliance_result["issues"].append(f"{check.requirement}: {check.status.value}")
                    compliance_result["recommendations"].extend(check.remediation_actions)
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Error checking compliance: {str(e)}")
            return {"error": str(e)}
    
    async def run_security_scan(self, 
                              scan_type: str = "comprehensive") -> Dict[str, Any]:
        """Run security scan"""
        try:
            scan_result = {
                "scan_id": hashlib.md5(f"scan_{datetime.now()}".encode()).hexdigest(),
                "scan_type": scan_type,
                "timestamp": datetime.now().isoformat(),
                "vulnerabilities": [],
                "recommendations": [],
                "overall_risk": "low"
            }
            
            # Simulate security scan based on type
            if scan_type == "comprehensive":
                vulnerabilities = await self._scan_comprehensive()
            elif scan_type == "authentication":
                vulnerabilities = await self._scan_authentication()
            elif scan_type == "data_protection":
                vulnerabilities = await self._scan_data_protection()
            else:
                vulnerabilities = await self._scan_basic()
            
            scan_result["vulnerabilities"] = vulnerabilities
            
            # Calculate overall risk
            if any(v["severity"] == "critical" for v in vulnerabilities):
                scan_result["overall_risk"] = "critical"
            elif any(v["severity"] == "high" for v in vulnerabilities):
                scan_result["overall_risk"] = "high"
            elif any(v["severity"] == "medium" for v in vulnerabilities):
                scan_result["overall_risk"] = "medium"
            
            # Generate recommendations
            scan_result["recommendations"] = await self._generate_security_recommendations(vulnerabilities)
            
            return scan_result
            
        except Exception as e:
            logger.error(f"Error running security scan: {str(e)}")
            return {"error": str(e)}
    
    async def get_security_analytics(self, 
                                   time_range_days: int = 30) -> Dict[str, Any]:
        """Get security analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            
            # Filter recent data
            recent_events = [e for e in self.security_events if e.timestamp >= cutoff_date]
            recent_logs = [l for l in self.audit_logs if l.timestamp >= cutoff_date]
            
            analytics = {
                "total_security_events": len(recent_events),
                "total_audit_logs": len(recent_logs),
                "security_event_types": {},
                "severity_distribution": {},
                "top_source_ips": [],
                "failed_authentication_attempts": 0,
                "suspicious_activities": 0,
                "compliance_status": {},
                "security_trends": {},
                "risk_assessment": {}
            }
            
            # Analyze security events
            for event in recent_events:
                event_type = event.event_type.value
                if event_type not in analytics["security_event_types"]:
                    analytics["security_event_types"][event_type] = 0
                analytics["security_event_types"][event_type] += 1
                
                severity = event.severity.value
                if severity not in analytics["severity_distribution"]:
                    analytics["severity_distribution"][severity] = 0
                analytics["severity_distribution"][severity] += 1
                
                if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
                    analytics["failed_authentication_attempts"] += 1
                elif event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY:
                    analytics["suspicious_activities"] += 1
            
            # Top source IPs
            ip_counts = defaultdict(int)
            for event in recent_events:
                ip_counts[event.source_ip] += 1
            
            top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            analytics["top_source_ips"] = [{"ip": ip, "count": count} for ip, count in top_ips]
            
            # Compliance status
            for check in self.compliance_checks:
                standard = check.standard.value
                if standard not in analytics["compliance_status"]:
                    analytics["compliance_status"][standard] = {
                        "compliant": 0,
                        "non_compliant": 0,
                        "total": 0
                    }
                
                analytics["compliance_status"][standard]["total"] += 1
                if check.status == ComplianceStatus.COMPLIANT:
                    analytics["compliance_status"][standard]["compliant"] += 1
                else:
                    analytics["compliance_status"][standard]["non_compliant"] += 1
            
            # Security trends (daily)
            daily_events = defaultdict(int)
            for event in recent_events:
                date_key = event.timestamp.date()
                daily_events[date_key] += 1
            
            analytics["security_trends"] = {
                date.isoformat(): count for date, count in daily_events.items()
            }
            
            # Risk assessment
            analytics["risk_assessment"] = {
                "overall_risk": "low",
                "authentication_risk": "low",
                "data_protection_risk": "low",
                "network_risk": "low",
                "compliance_risk": "low"
            }
            
            # Calculate risk levels
            if analytics["failed_authentication_attempts"] > 100:
                analytics["risk_assessment"]["authentication_risk"] = "high"
            elif analytics["failed_authentication_attempts"] > 50:
                analytics["risk_assessment"]["authentication_risk"] = "medium"
            
            if analytics["suspicious_activities"] > 10:
                analytics["risk_assessment"]["network_risk"] = "high"
            elif analytics["suspicious_activities"] > 5:
                analytics["risk_assessment"]["network_risk"] = "medium"
            
            # Overall risk
            risk_levels = list(analytics["risk_assessment"].values())
            if "high" in risk_levels:
                analytics["risk_assessment"]["overall_risk"] = "high"
            elif "medium" in risk_levels:
                analytics["risk_assessment"]["overall_risk"] = "medium"
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting security analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    def _initialize_default_policies(self) -> None:
        """Initialize default security policies"""
        try:
            # Password policy
            password_policy = SecurityPolicy(
                policy_id="password_policy",
                name="Password Security Policy",
                description="Enforces strong password requirements",
                security_level=SecurityLevel.HIGH,
                compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001],
                rules=[
                    {"type": "password_length", "min_length": 12},
                    {"type": "password_complexity", "require_special": True},
                    {"type": "password_history", "prevent_reuse": 5}
                ],
                enforcement_actions=["block_weak_passwords", "force_password_reset"]
            )
            self.security_policies["password_policy"] = password_policy
            
            # Data encryption policy
            encryption_policy = SecurityPolicy(
                policy_id="encryption_policy",
                name="Data Encryption Policy",
                description="Enforces encryption of sensitive data",
                security_level=SecurityLevel.CRITICAL,
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA],
                rules=[
                    {"type": "encrypt_at_rest", "algorithm": "AES-256"},
                    {"type": "encrypt_in_transit", "protocol": "TLS 1.3"},
                    {"type": "key_rotation", "interval_days": 90}
                ],
                enforcement_actions=["encrypt_data", "block_unencrypted_access"]
            )
            self.security_policies["encryption_policy"] = encryption_policy
            
            # Access control policy
            access_policy = SecurityPolicy(
                policy_id="access_control_policy",
                name="Access Control Policy",
                description="Enforces proper access controls",
                security_level=SecurityLevel.HIGH,
                compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.NIST],
                rules=[
                    {"type": "principle_of_least_privilege", "enabled": True},
                    {"type": "role_based_access", "enabled": True},
                    {"type": "session_timeout", "timeout_minutes": 60}
                ],
                enforcement_actions=["revoke_excessive_permissions", "force_reauthentication"]
            )
            self.security_policies["access_control_policy"] = access_policy
            
            logger.info(f"Initialized {len(self.security_policies)} default security policies")
            
        except Exception as e:
            logger.error(f"Error initializing default policies: {str(e)}")
    
    def _initialize_compliance_checks(self) -> None:
        """Initialize compliance checks"""
        try:
            # GDPR compliance checks
            gdpr_checks = [
                ComplianceCheck(
                    check_id="gdpr_data_minimization",
                    standard=ComplianceStandard.GDPR,
                    requirement="Data Minimization",
                    description="Ensure only necessary data is collected and processed",
                    status=ComplianceStatus.COMPLIANT,
                    evidence=["Data collection audit", "Processing purpose documentation"],
                    last_checked=datetime.now(),
                    next_check=datetime.now() + timedelta(days=30)
                ),
                ComplianceCheck(
                    check_id="gdpr_consent_management",
                    standard=ComplianceStandard.GDPR,
                    requirement="Consent Management",
                    description="Proper consent collection and management",
                    status=ComplianceStatus.COMPLIANT,
                    evidence=["Consent forms", "Opt-in mechanisms"],
                    last_checked=datetime.now(),
                    next_check=datetime.now() + timedelta(days=30)
                )
            ]
            
            # SOC2 compliance checks
            soc2_checks = [
                ComplianceCheck(
                    check_id="soc2_access_controls",
                    standard=ComplianceStandard.SOC2,
                    requirement="Access Controls",
                    description="Implement proper access controls and authentication",
                    status=ComplianceStatus.COMPLIANT,
                    evidence=["Access control matrix", "Authentication logs"],
                    last_checked=datetime.now(),
                    next_check=datetime.now() + timedelta(days=30)
                ),
                ComplianceCheck(
                    check_id="soc2_data_encryption",
                    standard=ComplianceStandard.SOC2,
                    requirement="Data Encryption",
                    description="Encrypt sensitive data at rest and in transit",
                    status=ComplianceStatus.COMPLIANT,
                    evidence=["Encryption configuration", "Key management"],
                    last_checked=datetime.now(),
                    next_check=datetime.now() + timedelta(days=30)
                )
            ]
            
            self.compliance_checks.extend(gdpr_checks)
            self.compliance_checks.extend(soc2_checks)
            
            logger.info(f"Initialized {len(self.compliance_checks)} compliance checks")
            
        except Exception as e:
            logger.error(f"Error initializing compliance checks: {str(e)}")
    
    async def _check_security_policies(self, event: SecurityEvent) -> None:
        """Check if security event triggers any policies"""
        try:
            for policy in self.security_policies.values():
                for rule in policy.rules:
                    if await self._evaluate_policy_rule(event, rule):
                        # Policy rule triggered
                        await self._enforce_policy_action(policy, event, rule)
                        
        except Exception as e:
            logger.error(f"Error checking security policies: {str(e)}")
    
    async def _evaluate_policy_rule(self, event: SecurityEvent, rule: Dict[str, Any]) -> bool:
        """Evaluate if security event matches policy rule"""
        try:
            rule_type = rule.get("type", "")
            
            if rule_type == "authentication_failure_threshold":
                # Check if authentication failures exceed threshold
                threshold = rule.get("threshold", 5)
                recent_failures = len([
                    e for e in self.security_events 
                    if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE 
                    and e.source_ip == event.source_ip
                    and (datetime.now() - e.timestamp).total_seconds() < 3600
                ])
                return recent_failures >= threshold
            
            elif rule_type == "suspicious_activity_pattern":
                # Check for suspicious activity patterns
                pattern = rule.get("pattern", "")
                return pattern.lower() in event.description.lower()
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating policy rule: {str(e)}")
            return False
    
    async def _enforce_policy_action(self, policy: SecurityPolicy, event: SecurityEvent, rule: Dict[str, Any]) -> None:
        """Enforce policy action"""
        try:
            for action in policy.enforcement_actions:
                if action == "block_ip":
                    await self._block_ip_address(event.source_ip)
                elif action == "alert_admin":
                    await self._send_admin_alert(policy, event)
                elif action == "log_violation":
                    await self._log_policy_violation(policy, event, rule)
                    
        except Exception as e:
            logger.error(f"Error enforcing policy action: {str(e)}")
    
    async def _block_ip_address(self, ip_address: str) -> None:
        """Block IP address"""
        try:
            # Simulate IP blocking
            logger.warning(f"Blocking IP address: {ip_address}")
            
        except Exception as e:
            logger.error(f"Error blocking IP address: {str(e)}")
    
    async def _send_admin_alert(self, policy: SecurityPolicy, event: SecurityEvent) -> None:
        """Send admin alert"""
        try:
            # Simulate admin alert
            logger.warning(f"Admin alert: Policy {policy.name} triggered by event {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error sending admin alert: {str(e)}")
    
    async def _log_policy_violation(self, policy: SecurityPolicy, event: SecurityEvent, rule: Dict[str, Any]) -> None:
        """Log policy violation"""
        try:
            violation_log = {
                "policy_id": policy.policy_id,
                "policy_name": policy.name,
                "event_id": event.event_id,
                "rule": rule,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.warning(f"Policy violation logged: {json.dumps(violation_log)}")
            
        except Exception as e:
            logger.error(f"Error logging policy violation: {str(e)}")
    
    async def _scan_comprehensive(self) -> List[Dict[str, Any]]:
        """Run comprehensive security scan"""
        try:
            vulnerabilities = []
            
            # Check password policies
            if not self.password_require_special:
                vulnerabilities.append({
                    "type": "password_policy",
                    "severity": "medium",
                    "description": "Password policy does not require special characters",
                    "recommendation": "Enable special character requirement"
                })
            
            # Check encryption
            if not self.encryption_key:
                vulnerabilities.append({
                    "type": "encryption",
                    "severity": "critical",
                    "description": "No encryption key configured",
                    "recommendation": "Configure encryption key"
                })
            
            # Check session timeout
            if self.session_timeout > 7200:  # 2 hours
                vulnerabilities.append({
                    "type": "session_management",
                    "severity": "medium",
                    "description": "Session timeout is too long",
                    "recommendation": "Reduce session timeout to 1 hour or less"
                })
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error in comprehensive scan: {str(e)}")
            return []
    
    async def _scan_authentication(self) -> List[Dict[str, Any]]:
        """Run authentication security scan"""
        try:
            vulnerabilities = []
            
            # Check login attempts
            recent_failures = len([
                e for e in self.security_events 
                if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE
                and (datetime.now() - e.timestamp).total_seconds() < 3600
            ])
            
            if recent_failures > 50:
                vulnerabilities.append({
                    "type": "authentication",
                    "severity": "high",
                    "description": f"High number of authentication failures: {recent_failures}",
                    "recommendation": "Implement account lockout and rate limiting"
                })
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error in authentication scan: {str(e)}")
            return []
    
    async def _scan_data_protection(self) -> List[Dict[str, Any]]:
        """Run data protection security scan"""
        try:
            vulnerabilities = []
            
            # Check for data breaches
            recent_breaches = len([
                e for e in self.security_events 
                if e.event_type == SecurityEventType.DATA_BREACH
                and (datetime.now() - e.timestamp).total_seconds() < 86400
            ])
            
            if recent_breaches > 0:
                vulnerabilities.append({
                    "type": "data_protection",
                    "severity": "critical",
                    "description": f"Data breach detected: {recent_breaches} incidents",
                    "recommendation": "Immediate investigation and remediation required"
                })
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error in data protection scan: {str(e)}")
            return []
    
    async def _scan_basic(self) -> List[Dict[str, Any]]:
        """Run basic security scan"""
        try:
            vulnerabilities = []
            
            # Basic checks
            if len(self.security_policies) == 0:
                vulnerabilities.append({
                    "type": "policy",
                    "severity": "medium",
                    "description": "No security policies configured",
                    "recommendation": "Configure security policies"
                })
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error in basic scan: {str(e)}")
            return []
    
    async def _generate_security_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on vulnerabilities"""
        try:
            recommendations = []
            
            for vuln in vulnerabilities:
                if vuln["severity"] == "critical":
                    recommendations.append(f"URGENT: {vuln['recommendation']}")
                elif vuln["severity"] == "high":
                    recommendations.append(f"HIGH PRIORITY: {vuln['recommendation']}")
                else:
                    recommendations.append(vuln['recommendation'])
            
            # Add general recommendations
            if not recommendations:
                recommendations.extend([
                    "Regular security audits and penetration testing",
                    "Employee security awareness training",
                    "Implement multi-factor authentication",
                    "Regular backup and disaster recovery testing",
                    "Keep all systems and software updated"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating security recommendations: {str(e)}")
            return ["Error generating recommendations"]


# Global security compliance system instance
_security_system: Optional[SecurityComplianceSystem] = None


def get_security_compliance_system(max_events: int = 10000, max_logs: int = 100000) -> SecurityComplianceSystem:
    """Get or create global security compliance system instance"""
    global _security_system
    if _security_system is None:
        _security_system = SecurityComplianceSystem(max_events, max_logs)
    return _security_system


# Example usage
async def main():
    """Example usage of the security compliance system"""
    system = get_security_compliance_system()
    
    # Create security policy
    policy = await system.create_security_policy(
        name="AI Model Access Policy",
        description="Controls access to AI model resources",
        security_level=SecurityLevel.HIGH,
        compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.GDPR],
        rules=[
            {"type": "authentication_failure_threshold", "threshold": 5},
            {"type": "suspicious_activity_pattern", "pattern": "unauthorized"}
        ],
        enforcement_actions=["block_ip", "alert_admin", "log_violation"]
    )
    print(f"Created security policy: {policy.policy_id}")
    
    # Log security events
    event1 = await system.log_security_event(
        event_type=SecurityEventType.AUTHENTICATION_FAILURE,
        severity=SecurityLevel.MEDIUM,
        description="Failed login attempt",
        source_ip="192.168.1.100",
        user_id="user123",
        resource="/api/login"
    )
    print(f"Logged security event: {event1.event_id}")
    
    event2 = await system.log_security_event(
        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
        severity=SecurityLevel.HIGH,
        description="Unauthorized access attempt detected",
        source_ip="10.0.0.50",
        user_id="unknown",
        resource="/api/models"
    )
    print(f"Logged security event: {event2.event_id}")
    
    # Log audit events
    audit1 = await system.log_audit_event(
        user_id="user123",
        action="login",
        resource="/api/auth",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0",
        success=True
    )
    print(f"Logged audit event: {audit1.log_id}")
    
    # Encrypt sensitive data
    encrypted_data = await system.encrypt_sensitive_data("sensitive information")
    print(f"Encrypted data: {encrypted_data[:50]}...")
    
    # Decrypt data
    decrypted_data = await system.decrypt_sensitive_data(encrypted_data)
    print(f"Decrypted data: {decrypted_data}")
    
    # Validate password strength
    password_validation = await system.validate_password_strength("MySecure123!")
    print(f"Password validation: {password_validation['valid']}")
    
    # Hash password
    hashed_password = await system.hash_password("MySecure123!")
    print(f"Hashed password: {hashed_password[:50]}...")
    
    # Verify password
    password_valid = await system.verify_password("MySecure123!", hashed_password)
    print(f"Password verification: {password_valid}")
    
    # Generate secure token
    token = await system.generate_secure_token()
    print(f"Secure token: {token}")
    
    # Check compliance
    gdpr_compliance = await system.check_compliance(ComplianceStandard.GDPR)
    print(f"GDPR compliance: {gdpr_compliance['overall_status']}")
    
    # Run security scan
    scan_result = await system.run_security_scan("comprehensive")
    print(f"Security scan: {scan_result['overall_risk']} risk level")
    print(f"Vulnerabilities found: {len(scan_result['vulnerabilities'])}")
    
    # Get security analytics
    analytics = await system.get_security_analytics()
    print(f"Security analytics:")
    print(f"  Total security events: {analytics.get('total_security_events', 0)}")
    print(f"  Failed authentication attempts: {analytics.get('failed_authentication_attempts', 0)}")
    print(f"  Overall risk: {analytics.get('risk_assessment', {}).get('overall_risk', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())

























