"""
Content Security & Compliance Engine - Advanced Content Security and Governance
===========================================================================

This module provides comprehensive content security and compliance capabilities including:
- Content security scanning and threat detection
- Compliance validation and auditing
- Data privacy and GDPR compliance
- Content encryption and access control
- Security monitoring and incident response
- Content backup and disaster recovery
- Audit logging and compliance reporting
- Multi-tenant security isolation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import hmac
import base64
import re
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import redis
import boto3
from google.cloud import secretmanager
import requests
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import yara
import clamav

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Compliance standard enumeration"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"

class ThreatType(Enum):
    """Threat type enumeration"""
    MALWARE = "malware"
    PHISHING = "phishing"
    SPAM = "spam"
    DATA_LEAK = "data_leak"
    SENSITIVE_DATA = "sensitive_data"
    COPYRIGHT_VIOLATION = "copyright_violation"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    ADULT_CONTENT = "adult_content"

class AccessLevel(Enum):
    """Access level enumeration"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

@dataclass
class SecurityScan:
    """Security scan data structure"""
    scan_id: str
    content_id: str
    scan_type: str
    threats_detected: List[ThreatType] = field(default_factory=list)
    security_score: float = 0.0
    compliance_violations: List[ComplianceStandard] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    scan_timestamp: datetime = field(default_factory=datetime.utcnow)
    scan_duration: float = 0.0

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    incident_id: str
    content_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
    affected_users: List[str] = field(default_factory=list)

@dataclass
class ComplianceAudit:
    """Compliance audit data structure"""
    audit_id: str
    content_id: str
    standard: ComplianceStandard
    compliance_score: float = 0.0
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    audit_timestamp: datetime = field(default_factory=datetime.utcnow)
    auditor: str = ""

@dataclass
class AccessControl:
    """Access control data structure"""
    content_id: str
    access_level: AccessLevel
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    encryption_key: str = ""
    expiration_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentSecurityEngine:
    """
    Advanced Content Security & Compliance Engine
    
    Provides comprehensive content security, compliance, and governance capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Security Engine"""
        self.config = config
        self.encryption_keys = {}
        self.access_controls = {}
        self.security_scans = {}
        self.security_incidents = {}
        self.compliance_audits = {}
        self.audit_logs = []
        self.threat_detection_models = {}
        self.compliance_rules = {}
        
        # Initialize security components
        self._initialize_encryption()
        self._initialize_threat_detection()
        self._initialize_compliance_rules()
        self._initialize_audit_logging()
        
        logger.info("Content Security Engine initialized successfully")
    
    def _initialize_encryption(self):
        """Initialize encryption components"""
        try:
            # Generate master encryption key
            self.master_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.master_key)
            
            # Initialize password hashing
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            
            # Initialize JWT settings
            self.jwt_secret = self.config.get("jwt_secret", secrets.token_urlsafe(32))
            self.jwt_algorithm = "HS256"
            
            logger.info("Encryption components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            raise
    
    def _initialize_threat_detection(self):
        """Initialize threat detection models"""
        try:
            # Initialize anomaly detection model
            self.threat_detection_models["anomaly_detector"] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Initialize text analysis model
            self.threat_detection_models["text_analyzer"] = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            
            # Initialize YARA rules for malware detection
            try:
                self.threat_detection_models["yara_rules"] = yara.compile(filepath="malware_rules.yar")
            except:
                logger.warning("YARA rules not found, using basic pattern matching")
                self.threat_detection_models["yara_rules"] = None
            
            # Initialize ClamAV for virus scanning
            try:
                self.threat_detection_models["clamav"] = clamav.ClamAV()
            except:
                logger.warning("ClamAV not available, using basic file scanning")
                self.threat_detection_models["clamav"] = None
            
            logger.info("Threat detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing threat detection: {e}")
            raise
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules and standards"""
        try:
            # GDPR compliance rules
            self.compliance_rules[ComplianceStandard.GDPR] = {
                "personal_data_patterns": [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                    r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'  # Credit card
                ],
                "required_fields": ["privacy_policy", "consent_mechanism"],
                "data_retention_limit": 365,  # days
                "right_to_deletion": True
            }
            
            # HIPAA compliance rules
            self.compliance_rules[ComplianceStandard.HIPAA] = {
                "phi_patterns": [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                    r'\b\d{2}/\d{2}/\d{4}\b'  # Date of birth
                ],
                "required_encryption": True,
                "access_logging": True,
                "audit_trail": True
            }
            
            # PCI DSS compliance rules
            self.compliance_rules[ComplianceStandard.PCI_DSS] = {
                "card_data_patterns": [
                    r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card
                    r'\b\d{3}\b'  # CVV
                ],
                "required_encryption": True,
                "network_security": True,
                "regular_testing": True
            }
            
            logger.info("Compliance rules initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing compliance rules: {e}")
            raise
    
    def _initialize_audit_logging(self):
        """Initialize audit logging system"""
        try:
            # Initialize audit log storage
            self.audit_logs = []
            
            # Initialize audit log patterns
            self.audit_patterns = {
                "access": "User {user_id} accessed content {content_id}",
                "modification": "User {user_id} modified content {content_id}",
                "deletion": "User {user_id} deleted content {content_id}",
                "security_scan": "Security scan performed on content {content_id}",
                "compliance_audit": "Compliance audit performed on content {content_id}"
            }
            
            logger.info("Audit logging system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing audit logging: {e}")
            raise
    
    async def scan_content_security(self, content_id: str, content: str, 
                                  content_type: str = "text") -> SecurityScan:
        """Perform comprehensive security scan on content"""
        try:
            start_time = datetime.utcnow()
            scan_id = str(uuid.uuid4())
            
            threats_detected = []
            compliance_violations = []
            recommendations = []
            
            # Malware detection
            malware_threats = await self._detect_malware(content, content_type)
            threats_detected.extend(malware_threats)
            
            # Sensitive data detection
            sensitive_data_threats = await self._detect_sensitive_data(content)
            threats_detected.extend(sensitive_data_threats)
            
            # Content policy violations
            policy_violations = await self._detect_policy_violations(content)
            threats_detected.extend(policy_violations)
            
            # Compliance violations
            compliance_violations = await self._check_compliance_violations(content)
            
            # Calculate security score
            security_score = await self._calculate_security_score(threats_detected, compliance_violations)
            
            # Generate recommendations
            recommendations = await self._generate_security_recommendations(threats_detected, compliance_violations)
            
            # Create security scan record
            scan_duration = (datetime.utcnow() - start_time).total_seconds()
            security_scan = SecurityScan(
                scan_id=scan_id,
                content_id=content_id,
                scan_type="comprehensive",
                threats_detected=threats_detected,
                security_score=security_score,
                compliance_violations=compliance_violations,
                recommendations=recommendations,
                scan_duration=scan_duration
            )
            
            # Store scan results
            self.security_scans[scan_id] = security_scan
            
            # Log security scan
            await self._log_audit_event("security_scan", {
                "content_id": content_id,
                "scan_id": scan_id,
                "threats_detected": len(threats_detected),
                "security_score": security_score
            })
            
            logger.info(f"Security scan completed for content {content_id}: {security_score:.2f} score")
            
            return security_scan
            
        except Exception as e:
            logger.error(f"Error scanning content security: {e}")
            raise
    
    async def _detect_malware(self, content: str, content_type: str) -> List[ThreatType]:
        """Detect malware in content"""
        try:
            threats = []
            
            # YARA rule scanning
            if self.threat_detection_models["yara_rules"]:
                try:
                    matches = self.threat_detection_models["yara_rules"].match(data=content.encode())
                    if matches:
                        threats.append(ThreatType.MALWARE)
                except:
                    pass
            
            # ClamAV scanning
            if self.threat_detection_models["clamav"]:
                try:
                    result = self.threat_detection_models["clamav"].scan_buffer(content.encode())
                    if result[0] == "FOUND":
                        threats.append(ThreatType.MALWARE)
                except:
                    pass
            
            # Basic pattern matching for common malware signatures
            malware_patterns = [
                r'eval\s*\(',  # JavaScript eval
                r'<script[^>]*>.*?</script>',  # Script tags
                r'javascript:',  # JavaScript protocol
                r'vbscript:',  # VBScript protocol
                r'data:text/html',  # Data URI
                r'<iframe[^>]*>',  # Iframe tags
            ]
            
            for pattern in malware_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(ThreatType.MALWARE)
                    break
            
            return threats
            
        except Exception as e:
            logger.error(f"Error detecting malware: {e}")
            return []
    
    async def _detect_sensitive_data(self, content: str) -> List[ThreatType]:
        """Detect sensitive data in content"""
        try:
            threats = []
            
            # Check for personal identifiable information (PII)
            pii_patterns = {
                ThreatType.SENSITIVE_DATA: [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                    r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card
                    r'\b\d{3}\b'  # CVV
                ]
            }
            
            for threat_type, patterns in pii_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content):
                        threats.append(threat_type)
                        break
            
            return threats
            
        except Exception as e:
            logger.error(f"Error detecting sensitive data: {e}")
            return []
    
    async def _detect_policy_violations(self, content: str) -> List[ThreatType]:
        """Detect content policy violations"""
        try:
            threats = []
            
            # Hate speech detection
            hate_speech_patterns = [
                r'\b(kill|murder|destroy)\s+(all|every)\s+(.*?)\b',
                r'\b(hate|despise)\s+(.*?)\b',
                r'\b(racist|sexist|homophobic)\b'
            ]
            
            for pattern in hate_speech_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(ThreatType.HATE_SPEECH)
                    break
            
            # Violence detection
            violence_patterns = [
                r'\b(violence|violent|attack|assault|bomb|weapon)\b',
                r'\b(kill|murder|destroy|harm|hurt)\b'
            ]
            
            for pattern in violence_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(ThreatType.VIOLENCE)
                    break
            
            # Adult content detection
            adult_content_patterns = [
                r'\b(sex|sexual|porn|adult|nude|naked)\b',
                r'\b(breast|penis|vagina|orgasm)\b'
            ]
            
            for pattern in adult_content_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append(ThreatType.ADULT_CONTENT)
                    break
            
            return threats
            
        except Exception as e:
            logger.error(f"Error detecting policy violations: {e}")
            return []
    
    async def _check_compliance_violations(self, content: str) -> List[ComplianceStandard]:
        """Check for compliance violations"""
        try:
            violations = []
            
            for standard, rules in self.compliance_rules.items():
                if standard == ComplianceStandard.GDPR:
                    # Check for personal data without consent
                    for pattern in rules["personal_data_patterns"]:
                        if re.search(pattern, content):
                            violations.append(standard)
                            break
                
                elif standard == ComplianceStandard.HIPAA:
                    # Check for PHI without proper protection
                    for pattern in rules["phi_patterns"]:
                        if re.search(pattern, content):
                            violations.append(standard)
                            break
                
                elif standard == ComplianceStandard.PCI_DSS:
                    # Check for card data without encryption
                    for pattern in rules["card_data_patterns"]:
                        if re.search(pattern, content):
                            violations.append(standard)
                            break
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking compliance violations: {e}")
            return []
    
    async def _calculate_security_score(self, threats: List[ThreatType], 
                                      violations: List[ComplianceStandard]) -> float:
        """Calculate security score based on threats and violations"""
        try:
            base_score = 100.0
            
            # Deduct points for threats
            threat_penalties = {
                ThreatType.MALWARE: 30.0,
                ThreatType.SENSITIVE_DATA: 25.0,
                ThreatType.HATE_SPEECH: 20.0,
                ThreatType.VIOLENCE: 20.0,
                ThreatType.ADULT_CONTENT: 15.0,
                ThreatType.PHISHING: 25.0,
                ThreatType.SPAM: 10.0,
                ThreatType.DATA_LEAK: 35.0,
                ThreatType.COPYRIGHT_VIOLATION: 15.0
            }
            
            for threat in threats:
                base_score -= threat_penalties.get(threat, 10.0)
            
            # Deduct points for compliance violations
            compliance_penalties = {
                ComplianceStandard.GDPR: 20.0,
                ComplianceStandard.HIPAA: 25.0,
                ComplianceStandard.PCI_DSS: 30.0,
                ComplianceStandard.SOX: 15.0,
                ComplianceStandard.ISO_27001: 10.0,
                ComplianceStandard.SOC2: 15.0,
                ComplianceStandard.CCPA: 15.0
            }
            
            for violation in violations:
                base_score -= compliance_penalties.get(violation, 10.0)
            
            return max(0.0, base_score)
            
        except Exception as e:
            logger.error(f"Error calculating security score: {e}")
            return 0.0
    
    async def _generate_security_recommendations(self, threats: List[ThreatType], 
                                               violations: List[ComplianceStandard]) -> List[str]:
        """Generate security recommendations based on threats and violations"""
        try:
            recommendations = []
            
            # Threat-based recommendations
            threat_recommendations = {
                ThreatType.MALWARE: [
                    "Remove malicious code from content",
                    "Scan content with updated antivirus software",
                    "Review content creation process for security"
                ],
                ThreatType.SENSITIVE_DATA: [
                    "Remove or encrypt sensitive data",
                    "Implement data masking techniques",
                    "Review data handling policies"
                ],
                ThreatType.HATE_SPEECH: [
                    "Remove hateful content",
                    "Implement content moderation",
                    "Review content policies"
                ],
                ThreatType.VIOLENCE: [
                    "Remove violent content",
                    "Add content warnings if necessary",
                    "Review content guidelines"
                ],
                ThreatType.ADULT_CONTENT: [
                    "Add age restrictions",
                    "Implement content filtering",
                    "Review content policies"
                ]
            }
            
            for threat in threats:
                recommendations.extend(threat_recommendations.get(threat, []))
            
            # Compliance-based recommendations
            compliance_recommendations = {
                ComplianceStandard.GDPR: [
                    "Implement privacy policy",
                    "Add consent mechanisms",
                    "Implement data retention policies"
                ],
                ComplianceStandard.HIPAA: [
                    "Encrypt protected health information",
                    "Implement access controls",
                    "Add audit logging"
                ],
                ComplianceStandard.PCI_DSS: [
                    "Encrypt card data",
                    "Implement network security",
                    "Regular security testing"
                ]
            }
            
            for violation in violations:
                recommendations.extend(compliance_recommendations.get(violation, []))
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error generating security recommendations: {e}")
            return []
    
    async def encrypt_content(self, content: str, content_id: str) -> str:
        """Encrypt content with AES encryption"""
        try:
            # Generate content-specific encryption key
            content_key = Fernet.generate_key()
            content_cipher = Fernet(content_key)
            
            # Encrypt content
            encrypted_content = content_cipher.encrypt(content.encode())
            
            # Store encryption key securely
            self.encryption_keys[content_id] = content_key
            
            # Log encryption event
            await self._log_audit_event("encryption", {
                "content_id": content_id,
                "encryption_method": "AES-256"
            })
            
            return base64.b64encode(encrypted_content).decode()
            
        except Exception as e:
            logger.error(f"Error encrypting content: {e}")
            raise
    
    async def decrypt_content(self, encrypted_content: str, content_id: str) -> str:
        """Decrypt content with AES decryption"""
        try:
            if content_id not in self.encryption_keys:
                raise ValueError(f"Encryption key not found for content {content_id}")
            
            # Get encryption key
            content_key = self.encryption_keys[content_id]
            content_cipher = Fernet(content_key)
            
            # Decrypt content
            encrypted_bytes = base64.b64decode(encrypted_content.encode())
            decrypted_content = content_cipher.decrypt(encrypted_bytes)
            
            # Log decryption event
            await self._log_audit_event("decryption", {
                "content_id": content_id,
                "decryption_method": "AES-256"
            })
            
            return decrypted_content.decode()
            
        except Exception as e:
            logger.error(f"Error decrypting content: {e}")
            raise
    
    async def set_access_control(self, content_id: str, access_level: AccessLevel, 
                               allowed_users: List[str] = None, 
                               allowed_roles: List[str] = None) -> AccessControl:
        """Set access control for content"""
        try:
            access_control = AccessControl(
                content_id=content_id,
                access_level=access_level,
                allowed_users=allowed_users or [],
                allowed_roles=allowed_roles or []
            )
            
            # Store access control
            self.access_controls[content_id] = access_control
            
            # Log access control change
            await self._log_audit_event("access_control", {
                "content_id": content_id,
                "access_level": access_level.value,
                "allowed_users": len(allowed_users or []),
                "allowed_roles": len(allowed_roles or [])
            })
            
            logger.info(f"Access control set for content {content_id}: {access_level.value}")
            
            return access_control
            
        except Exception as e:
            logger.error(f"Error setting access control: {e}")
            raise
    
    async def check_access_permission(self, content_id: str, user_id: str, 
                                    user_roles: List[str] = None) -> bool:
        """Check if user has access permission to content"""
        try:
            if content_id not in self.access_controls:
                return True  # No access control set, allow access
            
            access_control = self.access_controls[content_id]
            
            # Check access level
            if access_control.access_level == AccessLevel.PUBLIC:
                return True
            
            # Check if user is in allowed users list
            if user_id in access_control.allowed_users:
                return True
            
            # Check if user has allowed role
            if user_roles:
                for role in user_roles:
                    if role in access_control.allowed_roles:
                        return True
            
            # Check expiration
            if access_control.expiration_date and datetime.utcnow() > access_control.expiration_date:
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking access permission: {e}")
            return False
    
    async def create_security_incident(self, content_id: str, threat_type: ThreatType, 
                                     severity: SecurityLevel, description: str) -> SecurityIncident:
        """Create security incident record"""
        try:
            incident_id = str(uuid.uuid4())
            
            incident = SecurityIncident(
                incident_id=incident_id,
                content_id=content_id,
                threat_type=threat_type,
                severity=severity,
                description=description
            )
            
            # Store incident
            self.security_incidents[incident_id] = incident
            
            # Log incident
            await self._log_audit_event("security_incident", {
                "incident_id": incident_id,
                "content_id": content_id,
                "threat_type": threat_type.value,
                "severity": severity.value
            })
            
            logger.warning(f"Security incident created: {incident_id} - {threat_type.value}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Error creating security incident: {e}")
            raise
    
    async def perform_compliance_audit(self, content_id: str, 
                                     standard: ComplianceStandard) -> ComplianceAudit:
        """Perform compliance audit for specific standard"""
        try:
            audit_id = str(uuid.uuid4())
            
            # Perform compliance check
            violations = []
            recommendations = []
            
            if standard == ComplianceStandard.GDPR:
                violations, recommendations = await self._audit_gdpr_compliance(content_id)
            elif standard == ComplianceStandard.HIPAA:
                violations, recommendations = await self._audit_hipaa_compliance(content_id)
            elif standard == ComplianceStandard.PCI_DSS:
                violations, recommendations = await self._audit_pci_compliance(content_id)
            
            # Calculate compliance score
            compliance_score = max(0.0, 100.0 - len(violations) * 20.0)
            
            audit = ComplianceAudit(
                audit_id=audit_id,
                content_id=content_id,
                standard=standard,
                compliance_score=compliance_score,
                violations=violations,
                recommendations=recommendations,
                auditor="system"
            )
            
            # Store audit
            self.compliance_audits[audit_id] = audit
            
            # Log audit
            await self._log_audit_event("compliance_audit", {
                "audit_id": audit_id,
                "content_id": content_id,
                "standard": standard.value,
                "compliance_score": compliance_score
            })
            
            logger.info(f"Compliance audit completed for {standard.value}: {compliance_score:.2f} score")
            
            return audit
            
        except Exception as e:
            logger.error(f"Error performing compliance audit: {e}")
            raise
    
    async def _audit_gdpr_compliance(self, content_id: str) -> Tuple[List[str], List[str]]:
        """Audit GDPR compliance"""
        try:
            violations = []
            recommendations = []
            
            # Check for privacy policy
            if not self._has_privacy_policy(content_id):
                violations.append("Missing privacy policy")
                recommendations.append("Implement comprehensive privacy policy")
            
            # Check for consent mechanism
            if not self._has_consent_mechanism(content_id):
                violations.append("Missing consent mechanism")
                recommendations.append("Implement consent collection mechanism")
            
            # Check data retention
            if not self._has_data_retention_policy(content_id):
                violations.append("Missing data retention policy")
                recommendations.append("Implement data retention policies")
            
            return violations, recommendations
            
        except Exception as e:
            logger.error(f"Error auditing GDPR compliance: {e}")
            return [], []
    
    async def _audit_hipaa_compliance(self, content_id: str) -> Tuple[List[str], List[str]]:
        """Audit HIPAA compliance"""
        try:
            violations = []
            recommendations = []
            
            # Check encryption
            if not self._is_content_encrypted(content_id):
                violations.append("Content not encrypted")
                recommendations.append("Implement encryption for PHI")
            
            # Check access logging
            if not self._has_access_logging(content_id):
                violations.append("Missing access logging")
                recommendations.append("Implement comprehensive access logging")
            
            # Check audit trail
            if not self._has_audit_trail(content_id):
                violations.append("Missing audit trail")
                recommendations.append("Implement audit trail for all access")
            
            return violations, recommendations
            
        except Exception as e:
            logger.error(f"Error auditing HIPAA compliance: {e}")
            return [], []
    
    async def _audit_pci_compliance(self, content_id: str) -> Tuple[List[str], List[str]]:
        """Audit PCI DSS compliance"""
        try:
            violations = []
            recommendations = []
            
            # Check encryption
            if not self._is_content_encrypted(content_id):
                violations.append("Card data not encrypted")
                recommendations.append("Implement encryption for card data")
            
            # Check network security
            if not self._has_network_security(content_id):
                violations.append("Missing network security")
                recommendations.append("Implement network security measures")
            
            # Check regular testing
            if not self._has_regular_testing(content_id):
                violations.append("Missing regular security testing")
                recommendations.append("Implement regular security testing")
            
            return violations, recommendations
            
        except Exception as e:
            logger.error(f"Error auditing PCI compliance: {e}")
            return [], []
    
    def _has_privacy_policy(self, content_id: str) -> bool:
        """Check if content has privacy policy"""
        # Simplified check - in production, this would check actual content
        return True
    
    def _has_consent_mechanism(self, content_id: str) -> bool:
        """Check if content has consent mechanism"""
        # Simplified check - in production, this would check actual content
        return True
    
    def _has_data_retention_policy(self, content_id: str) -> bool:
        """Check if content has data retention policy"""
        # Simplified check - in production, this would check actual content
        return True
    
    def _is_content_encrypted(self, content_id: str) -> bool:
        """Check if content is encrypted"""
        return content_id in self.encryption_keys
    
    def _has_access_logging(self, content_id: str) -> bool:
        """Check if content has access logging"""
        # Simplified check - in production, this would check actual logging
        return True
    
    def _has_audit_trail(self, content_id: str) -> bool:
        """Check if content has audit trail"""
        # Simplified check - in production, this would check actual audit trail
        return True
    
    def _has_network_security(self, content_id: str) -> bool:
        """Check if content has network security"""
        # Simplified check - in production, this would check actual network security
        return True
    
    def _has_regular_testing(self, content_id: str) -> bool:
        """Check if content has regular testing"""
        # Simplified check - in production, this would check actual testing
        return True
    
    async def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log audit event"""
        try:
            audit_log = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data
            }
            
            self.audit_logs.append(audit_log)
            
            # In production, this would be stored in a secure audit database
            logger.info(f"Audit event logged: {event_type}")
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    async def get_security_report(self, content_id: str) -> Dict[str, Any]:
        """Get comprehensive security report for content"""
        try:
            # Get security scans
            scans = [scan for scan in self.security_scans.values() if scan.content_id == content_id]
            
            # Get security incidents
            incidents = [incident for incident in self.security_incidents.values() if incident.content_id == content_id]
            
            # Get compliance audits
            audits = [audit for audit in self.compliance_audits.values() if audit.content_id == content_id]
            
            # Get access control
            access_control = self.access_controls.get(content_id)
            
            # Calculate overall security score
            overall_score = 0.0
            if scans:
                overall_score = sum(scan.security_score for scan in scans) / len(scans)
            
            return {
                "content_id": content_id,
                "overall_security_score": overall_score,
                "security_scans": [
                    {
                        "scan_id": scan.scan_id,
                        "scan_type": scan.scan_type,
                        "security_score": scan.security_score,
                        "threats_detected": [threat.value for threat in scan.threats_detected],
                        "compliance_violations": [violation.value for violation in scan.compliance_violations],
                        "scan_timestamp": scan.scan_timestamp.isoformat()
                    }
                    for scan in scans
                ],
                "security_incidents": [
                    {
                        "incident_id": incident.incident_id,
                        "threat_type": incident.threat_type.value,
                        "severity": incident.severity.value,
                        "description": incident.description,
                        "detected_at": incident.detected_at.isoformat(),
                        "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None
                    }
                    for incident in incidents
                ],
                "compliance_audits": [
                    {
                        "audit_id": audit.audit_id,
                        "standard": audit.standard.value,
                        "compliance_score": audit.compliance_score,
                        "violations": audit.violations,
                        "audit_timestamp": audit.audit_timestamp.isoformat()
                    }
                    for audit in audits
                ],
                "access_control": {
                    "access_level": access_control.access_level.value if access_control else "public",
                    "allowed_users": access_control.allowed_users if access_control else [],
                    "allowed_roles": access_control.allowed_roles if access_control else [],
                    "expiration_date": access_control.expiration_date.isoformat() if access_control and access_control.expiration_date else None
                },
                "encryption_status": content_id in self.encryption_keys,
                "report_generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Security Engine"""
    try:
        # Initialize engine
        config = {
            "jwt_secret": "your-jwt-secret-key",
            "encryption_key": "your-encryption-key"
        }
        
        engine = ContentSecurityEngine(config)
        
        # Test content
        test_content = "This is a test content with sensitive information like john.doe@example.com and 123-45-6789"
        content_id = "test_content_001"
        
        # Perform security scan
        print("Performing security scan...")
        security_scan = await engine.scan_content_security(content_id, test_content)
        print(f"Security score: {security_scan.security_score:.2f}")
        print(f"Threats detected: {[threat.value for threat in security_scan.threats_detected]}")
        print(f"Compliance violations: {[violation.value for violation in security_scan.compliance_violations]}")
        
        # Encrypt content
        print("\nEncrypting content...")
        encrypted_content = await engine.encrypt_content(test_content, content_id)
        print(f"Content encrypted: {len(encrypted_content)} characters")
        
        # Decrypt content
        print("Decrypting content...")
        decrypted_content = await engine.decrypt_content(encrypted_content, content_id)
        print(f"Content decrypted: {decrypted_content}")
        
        # Set access control
        print("\nSetting access control...")
        access_control = await engine.set_access_control(
            content_id, 
            AccessLevel.CONFIDENTIAL, 
            allowed_users=["user1", "user2"],
            allowed_roles=["admin", "editor"]
        )
        print(f"Access control set: {access_control.access_level.value}")
        
        # Check access permission
        print("Checking access permission...")
        has_access = await engine.check_access_permission(content_id, "user1", ["admin"])
        print(f"User has access: {has_access}")
        
        # Perform compliance audit
        print("\nPerforming GDPR compliance audit...")
        gdpr_audit = await engine.perform_compliance_audit(content_id, ComplianceStandard.GDPR)
        print(f"GDPR compliance score: {gdpr_audit.compliance_score:.2f}")
        print(f"Violations: {gdpr_audit.violations}")
        
        # Create security incident
        print("\nCreating security incident...")
        incident = await engine.create_security_incident(
            content_id, 
            ThreatType.SENSITIVE_DATA, 
            SecurityLevel.HIGH,
            "Sensitive data detected in content"
        )
        print(f"Security incident created: {incident.incident_id}")
        
        # Get security report
        print("\nGenerating security report...")
        security_report = await engine.get_security_report(content_id)
        print(f"Overall security score: {security_report['overall_security_score']:.2f}")
        print(f"Encryption status: {security_report['encryption_status']}")
        
        print("\nContent Security Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























