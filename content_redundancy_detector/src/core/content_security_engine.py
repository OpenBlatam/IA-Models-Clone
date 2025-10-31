"""
Content Security Engine - Advanced content security, threat detection, and protection
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib
import re
import secrets
import base64
from pathlib import Path

import aiofiles
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class SecurityThreat:
    """Security threat detection result"""
    threat_id: str
    threat_type: str
    severity: str
    confidence: float
    description: str
    affected_content: str
    mitigation_suggestions: List[str]
    detection_timestamp: datetime


@dataclass
class ContentSecurity:
    """Content security analysis result"""
    content_id: str
    security_score: float
    threat_count: int
    threats: List[SecurityThreat]
    security_metrics: Dict[str, Any]
    compliance_status: Dict[str, Any]
    encryption_status: Dict[str, Any]
    access_control: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    security_recommendations: List[str]
    analysis_timestamp: datetime


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    policy_name: str
    policy_type: str
    rules: List[Dict[str, Any]]
    severity_levels: Dict[str, str]
    enforcement_actions: List[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


@dataclass
class SecurityAudit:
    """Security audit result"""
    audit_id: str
    audit_timestamp: datetime
    audit_type: str
    content_analyzed: int
    threats_detected: int
    compliance_violations: int
    security_score: float
    recommendations: List[str]
    audit_details: Dict[str, Any]


class ContentSecurityEngine:
    """Advanced content security and threat detection engine"""
    
    def __init__(self):
        self.encryption_key = None
        self.security_policies = {}
        self.threat_patterns = {}
        self.compliance_rules = {}
        self.audit_logs = []
        self.models_loaded = False
        self.security_config = {}
        
    async def initialize(self) -> None:
        """Initialize the security engine"""
        try:
            logger.info("Initializing Content Security Engine...")
            
            # Generate encryption key
            await self._generate_encryption_key()
            
            # Load security policies
            await self._load_security_policies()
            
            # Load threat patterns
            await self._load_threat_patterns()
            
            # Load compliance rules
            await self._load_compliance_rules()
            
            # Load security configuration
            await self._load_security_config()
            
            self.models_loaded = True
            logger.info("Content Security Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Content Security Engine: {e}")
            raise
    
    async def _generate_encryption_key(self) -> None:
        """Generate encryption key for content protection"""
        try:
            # Generate a secure random key
            key = Fernet.generate_key()
            self.encryption_key = key
            logger.info("Encryption key generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate encryption key: {e}")
            self.encryption_key = None
    
    async def _load_security_policies(self) -> None:
        """Load security policies"""
        try:
            self.security_policies = {
                "content_validation": {
                    "max_content_length": 20000,
                    "allowed_file_types": ["txt", "md", "html", "json"],
                    "forbidden_patterns": [
                        r"<script.*?>.*?</script>",
                        r"javascript:",
                        r"data:text/html",
                        r"vbscript:"
                    ]
                },
                "data_protection": {
                    "encrypt_sensitive_data": True,
                    "hash_passwords": True,
                    "sanitize_input": True,
                    "validate_encoding": True
                },
                "access_control": {
                    "require_authentication": True,
                    "rate_limit_requests": True,
                    "log_all_access": True,
                    "session_timeout": 3600
                },
                "threat_detection": {
                    "scan_for_malware": True,
                    "detect_injection_attacks": True,
                    "monitor_suspicious_patterns": True,
                    "block_known_threats": True
                }
            }
            logger.info("Security policies loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load security policies: {e}")
    
    async def _load_threat_patterns(self) -> None:
        """Load threat detection patterns"""
        try:
            self.threat_patterns = {
                "sql_injection": [
                    r"('|(\\')|(;)|(\\;)|(\\|)|(\\|\\|))",
                    r"(union|select|insert|update|delete|drop|create|alter)",
                    r"(or|and)\\s+\\d+\\s*=\\s*\\d+",
                    r"(\\'|\\\")\\s*(or|and)\\s*(\\'|\\\")"
                ],
                "xss_attacks": [
                    r"<script[^>]*>.*?</script>",
                    r"javascript:",
                    r"on\\w+\\s*=",
                    r"<iframe[^>]*>",
                    r"<object[^>]*>",
                    r"<embed[^>]*>"
                ],
                "path_traversal": [
                    r"\\.\\./",
                    r"\\.\\.\\\\",
                    r"%2e%2e%2f",
                    r"%2e%2e%5c",
                    r"\\..\\..\\..\\.."
                ],
                "command_injection": [
                    r"[;&|`$]",
                    r"(cat|ls|dir|type|more|less|head|tail)\\s+",
                    r"(rm|del|mv|cp|chmod|chown)\\s+",
                    r"(wget|curl|nc|netcat)\\s+"
                ],
                "malicious_content": [
                    r"eval\\s*\\(",
                    r"exec\\s*\\(",
                    r"system\\s*\\(",
                    r"shell_exec\\s*\\(",
                    r"passthru\\s*\\("
                ]
            }
            logger.info("Threat patterns loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load threat patterns: {e}")
    
    async def _load_compliance_rules(self) -> None:
        """Load compliance rules"""
        try:
            self.compliance_rules = {
                "gdpr": {
                    "personal_data_patterns": [
                        r"\\b\\d{3}-\\d{2}-\\d{4}\\b",  # SSN
                        r"\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\s?\\d{4}\\b",  # Credit card
                        r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",  # Email
                        r"\\b\\d{3}-\\d{3}-\\d{4}\\b"  # Phone
                    ],
                    "data_retention_period": 365,
                    "consent_required": True,
                    "right_to_erasure": True
                },
                "hipaa": {
                    "phi_patterns": [
                        r"\\b\\d{3}-\\d{2}-\\d{4}\\b",  # SSN
                        r"\\b\\d{3}-\\d{3}-\\d{4}\\b",  # Phone
                        r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",  # Email
                        r"\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b"  # Date of birth
                    ],
                    "encryption_required": True,
                    "access_logging": True,
                    "audit_trail": True
                },
                "pci_dss": {
                    "card_data_patterns": [
                        r"\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\s?\\d{4}\\b",  # Credit card
                        r"\\b\\d{3}\\b",  # CVV
                        r"\\b\\d{1,2}/\\d{2,4}\\b"  # Expiry date
                    ],
                    "encryption_required": True,
                    "secure_transmission": True,
                    "access_restriction": True
                }
            }
            logger.info("Compliance rules loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load compliance rules: {e}")
    
    async def _load_security_config(self) -> None:
        """Load security configuration"""
        try:
            self.security_config = {
                "encryption": {
                    "algorithm": "AES-256-GCM",
                    "key_rotation_days": 90,
                    "encrypt_at_rest": True,
                    "encrypt_in_transit": True
                },
                "authentication": {
                    "jwt_secret": secrets.token_urlsafe(32),
                    "token_expiry": 3600,
                    "refresh_token_expiry": 86400,
                    "max_login_attempts": 5
                },
                "rate_limiting": {
                    "requests_per_minute": 100,
                    "burst_limit": 200,
                    "block_duration": 300
                },
                "monitoring": {
                    "log_security_events": True,
                    "alert_on_threats": True,
                    "audit_frequency": "daily",
                    "retention_days": 90
                }
            }
            logger.info("Security configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load security configuration: {e}")
    
    async def analyze_content_security(
        self,
        content: str,
        content_id: str = "",
        context: Dict[str, Any] = None
    ) -> ContentSecurity:
        """Perform comprehensive content security analysis"""
        
        if not self.models_loaded:
            raise Exception("Security engine not loaded. Call initialize() first.")
        
        if context is None:
            context = {}
        
        try:
            # Run all security analyses in parallel
            results = await asyncio.gather(
                self._detect_security_threats(content),
                self._analyze_security_metrics(content),
                self._check_compliance(content),
                self._analyze_encryption_status(content),
                self._analyze_access_control(content, context),
                self._generate_audit_trail(content, content_id, context),
                return_exceptions=True
            )
            
            # Extract results
            threats = results[0] if not isinstance(results[0], Exception) else []
            security_metrics = results[1] if not isinstance(results[1], Exception) else {}
            compliance_status = results[2] if not isinstance(results[2], Exception) else {}
            encryption_status = results[3] if not isinstance(results[3], Exception) else {}
            access_control = results[4] if not isinstance(results[4], Exception) else {}
            audit_trail = results[5] if not isinstance(results[5], Exception) else []
            
            # Calculate security score
            security_score = await self._calculate_security_score(
                threats, security_metrics, compliance_status
            )
            
            # Generate security recommendations
            security_recommendations = await self._generate_security_recommendations(
                threats, security_metrics, compliance_status, encryption_status
            )
            
            return ContentSecurity(
                content_id=content_id,
                security_score=security_score,
                threat_count=len(threats),
                threats=threats,
                security_metrics=security_metrics,
                compliance_status=compliance_status,
                encryption_status=encryption_status,
                access_control=access_control,
                audit_trail=audit_trail,
                security_recommendations=security_recommendations,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in content security analysis: {e}")
            raise
    
    async def _detect_security_threats(self, content: str) -> List[SecurityThreat]:
        """Detect security threats in content"""
        try:
            threats = []
            content_lower = content.lower()
            
            # Check for SQL injection patterns
            for pattern in self.threat_patterns.get("sql_injection", []):
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        threat_id=f"sql_injection_{len(threats)}",
                        threat_type="sql_injection",
                        severity="high",
                        confidence=0.8,
                        description="Potential SQL injection attack detected",
                        affected_content=content[:100] + "..." if len(content) > 100 else content,
                        mitigation_suggestions=[
                            "Use parameterized queries",
                            "Validate and sanitize input",
                            "Implement input length limits",
                            "Use prepared statements"
                        ],
                        detection_timestamp=datetime.now()
                    ))
            
            # Check for XSS patterns
            for pattern in self.threat_patterns.get("xss_attacks", []):
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        threat_id=f"xss_{len(threats)}",
                        threat_type="xss_attack",
                        severity="high",
                        confidence=0.9,
                        description="Potential XSS attack detected",
                        affected_content=content[:100] + "..." if len(content) > 100 else content,
                        mitigation_suggestions=[
                            "Sanitize HTML content",
                            "Use Content Security Policy",
                            "Validate and escape output",
                            "Implement XSS filters"
                        ],
                        detection_timestamp=datetime.now()
                    ))
            
            # Check for path traversal patterns
            for pattern in self.threat_patterns.get("path_traversal", []):
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        threat_id=f"path_traversal_{len(threats)}",
                        threat_type="path_traversal",
                        severity="medium",
                        confidence=0.7,
                        description="Potential path traversal attack detected",
                        affected_content=content[:100] + "..." if len(content) > 100 else content,
                        mitigation_suggestions=[
                            "Validate file paths",
                            "Use whitelist of allowed paths",
                            "Sanitize file names",
                            "Implement path restrictions"
                        ],
                        detection_timestamp=datetime.now()
                    ))
            
            # Check for command injection patterns
            for pattern in self.threat_patterns.get("command_injection", []):
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        threat_id=f"command_injection_{len(threats)}",
                        threat_type="command_injection",
                        severity="critical",
                        confidence=0.8,
                        description="Potential command injection attack detected",
                        affected_content=content[:100] + "..." if len(content) > 100 else content,
                        mitigation_suggestions=[
                            "Avoid system command execution",
                            "Use safe alternatives to system calls",
                            "Validate and sanitize input",
                            "Implement command whitelisting"
                        ],
                        detection_timestamp=datetime.now()
                    ))
            
            # Check for malicious content patterns
            for pattern in self.threat_patterns.get("malicious_content", []):
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        threat_id=f"malicious_content_{len(threats)}",
                        threat_type="malicious_content",
                        severity="critical",
                        confidence=0.9,
                        description="Potential malicious code execution detected",
                        affected_content=content[:100] + "..." if len(content) > 100 else content,
                        mitigation_suggestions=[
                            "Remove or sanitize dangerous functions",
                            "Implement code execution restrictions",
                            "Use sandboxed environments",
                            "Validate code before execution"
                        ],
                        detection_timestamp=datetime.now()
                    ))
            
            return threats
            
        except Exception as e:
            logger.warning(f"Threat detection failed: {e}")
            return []
    
    async def _analyze_security_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze security metrics"""
        try:
            metrics = {}
            
            # Content length analysis
            content_length = len(content)
            metrics["content_length"] = content_length
            metrics["length_risk"] = "high" if content_length > 10000 else "medium" if content_length > 5000 else "low"
            
            # Character encoding analysis
            try:
                content.encode('utf-8')
                metrics["encoding_valid"] = True
            except UnicodeEncodeError:
                metrics["encoding_valid"] = False
            
            # Suspicious character analysis
            suspicious_chars = ['<', '>', '&', '"', "'", '\\', '/', ';', '|', '`', '$']
            suspicious_count = sum(content.count(char) for char in suspicious_chars)
            metrics["suspicious_char_count"] = suspicious_count
            metrics["suspicious_char_ratio"] = suspicious_count / len(content) if content else 0
            
            # URL analysis
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, content)
            metrics["url_count"] = len(urls)
            metrics["has_urls"] = len(urls) > 0
            
            # Email analysis
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, content)
            metrics["email_count"] = len(emails)
            metrics["has_emails"] = len(emails) > 0
            
            # Phone number analysis
            phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'
            phones = re.findall(phone_pattern, content)
            metrics["phone_count"] = len(phones)
            metrics["has_phones"] = len(phones) > 0
            
            # Overall security score
            security_indicators = [
                metrics.get("encoding_valid", False),
                metrics.get("suspicious_char_ratio", 1) < 0.1,
                metrics.get("length_risk") == "low"
            ]
            metrics["security_indicators"] = sum(security_indicators)
            metrics["security_indicators_total"] = len(security_indicators)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Security metrics analysis failed: {e}")
            return {}
    
    async def _check_compliance(self, content: str) -> Dict[str, Any]:
        """Check compliance with regulations"""
        try:
            compliance_status = {}
            
            # GDPR compliance check
            gdpr_rules = self.compliance_rules.get("gdpr", {})
            personal_data_patterns = gdpr_rules.get("personal_data_patterns", [])
            gdpr_violations = []
            
            for pattern in personal_data_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    gdpr_violations.extend(matches)
            
            compliance_status["gdpr"] = {
                "compliant": len(gdpr_violations) == 0,
                "violations": len(gdpr_violations),
                "violation_details": gdpr_violations[:5],  # Limit to first 5
                "requires_consent": gdpr_rules.get("consent_required", True),
                "data_retention_days": gdpr_rules.get("data_retention_period", 365)
            }
            
            # HIPAA compliance check
            hipaa_rules = self.compliance_rules.get("hipaa", {})
            phi_patterns = hipaa_rules.get("phi_patterns", [])
            hipaa_violations = []
            
            for pattern in phi_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    hipaa_violations.extend(matches)
            
            compliance_status["hipaa"] = {
                "compliant": len(hipaa_violations) == 0,
                "violations": len(hipaa_violations),
                "violation_details": hipaa_violations[:5],  # Limit to first 5
                "encryption_required": hipaa_rules.get("encryption_required", True),
                "access_logging": hipaa_rules.get("access_logging", True)
            }
            
            # PCI DSS compliance check
            pci_rules = self.compliance_rules.get("pci_dss", {})
            card_data_patterns = pci_rules.get("card_data_patterns", [])
            pci_violations = []
            
            for pattern in card_data_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    pci_violations.extend(matches)
            
            compliance_status["pci_dss"] = {
                "compliant": len(pci_violations) == 0,
                "violations": len(pci_violations),
                "violation_details": pci_violations[:5],  # Limit to first 5
                "encryption_required": pci_rules.get("encryption_required", True),
                "secure_transmission": pci_rules.get("secure_transmission", True)
            }
            
            # Overall compliance score
            compliance_scores = [
                compliance_status.get("gdpr", {}).get("compliant", False),
                compliance_status.get("hipaa", {}).get("compliant", False),
                compliance_status.get("pci_dss", {}).get("compliant", False)
            ]
            compliance_status["overall_compliance_score"] = sum(compliance_scores) / len(compliance_scores)
            
            return compliance_status
            
        except Exception as e:
            logger.warning(f"Compliance check failed: {e}")
            return {}
    
    async def _analyze_encryption_status(self, content: str) -> Dict[str, Any]:
        """Analyze encryption status"""
        try:
            encryption_status = {}
            
            # Check if content appears to be encrypted
            encryption_indicators = [
                len(set(content)) / len(content) > 0.8,  # High character diversity
                content.count('=') > len(content) * 0.1,  # Base64 padding
                re.search(r'^[A-Za-z0-9+/]+=*$', content) is not None  # Base64 pattern
            ]
            
            encryption_status["appears_encrypted"] = sum(encryption_indicators) >= 2
            encryption_status["encryption_indicators"] = encryption_indicators
            
            # Check encryption strength
            if encryption_status["appears_encrypted"]:
                encryption_status["encryption_strength"] = "strong"
            else:
                encryption_status["encryption_strength"] = "none"
            
            # Encryption recommendations
            if not encryption_status["appears_encrypted"]:
                encryption_status["recommendations"] = [
                    "Encrypt sensitive content",
                    "Use strong encryption algorithms",
                    "Implement key rotation",
                    "Secure key storage"
                ]
            else:
                encryption_status["recommendations"] = [
                    "Verify encryption implementation",
                    "Check key management",
                    "Monitor encryption performance"
                ]
            
            return encryption_status
            
        except Exception as e:
            logger.warning(f"Encryption status analysis failed: {e}")
            return {}
    
    async def _analyze_access_control(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze access control"""
        try:
            access_control = {}
            
            # Check authentication context
            user_info = context.get("user", {})
            access_control["authenticated"] = bool(user_info.get("user_id"))
            access_control["user_role"] = user_info.get("role", "anonymous")
            access_control["user_permissions"] = user_info.get("permissions", [])
            
            # Check content sensitivity
            sensitive_keywords = ["password", "secret", "private", "confidential", "internal"]
            sensitive_count = sum(content.lower().count(keyword) for keyword in sensitive_keywords)
            access_control["sensitivity_level"] = "high" if sensitive_count > 3 else "medium" if sensitive_count > 0 else "low"
            
            # Access control recommendations
            recommendations = []
            if access_control["sensitivity_level"] == "high" and not access_control["authenticated"]:
                recommendations.append("Require authentication for sensitive content")
            
            if access_control["sensitivity_level"] in ["high", "medium"]:
                recommendations.extend([
                    "Implement role-based access control",
                    "Log all access attempts",
                    "Use encryption for sensitive data",
                    "Implement audit trails"
                ])
            
            access_control["recommendations"] = recommendations
            
            return access_control
            
        except Exception as e:
            logger.warning(f"Access control analysis failed: {e}")
            return {}
    
    async def _generate_audit_trail(self, content: str, content_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate audit trail"""
        try:
            audit_entries = []
            
            # Content access audit
            audit_entries.append({
                "event_type": "content_access",
                "timestamp": datetime.now().isoformat(),
                "content_id": content_id,
                "user_id": context.get("user", {}).get("user_id", "anonymous"),
                "ip_address": context.get("ip_address", "unknown"),
                "user_agent": context.get("user_agent", "unknown"),
                "content_length": len(content),
                "action": "analyze"
            })
            
            # Security analysis audit
            audit_entries.append({
                "event_type": "security_analysis",
                "timestamp": datetime.now().isoformat(),
                "content_id": content_id,
                "analysis_type": "comprehensive",
                "threats_detected": 0,  # Will be updated after threat detection
                "compliance_checked": True,
                "encryption_analyzed": True
            })
            
            return audit_entries
            
        except Exception as e:
            logger.warning(f"Audit trail generation failed: {e}")
            return []
    
    async def _calculate_security_score(
        self,
        threats: List[SecurityThreat],
        security_metrics: Dict[str, Any],
        compliance_status: Dict[str, Any]
    ) -> float:
        """Calculate overall security score"""
        try:
            # Base score
            score = 100.0
            
            # Deduct for threats
            for threat in threats:
                if threat.severity == "critical":
                    score -= 30
                elif threat.severity == "high":
                    score -= 20
                elif threat.severity == "medium":
                    score -= 10
                elif threat.severity == "low":
                    score -= 5
            
            # Deduct for security metrics issues
            if not security_metrics.get("encoding_valid", True):
                score -= 15
            
            if security_metrics.get("suspicious_char_ratio", 0) > 0.1:
                score -= 10
            
            if security_metrics.get("length_risk") == "high":
                score -= 5
            
            # Deduct for compliance violations
            compliance_score = compliance_status.get("overall_compliance_score", 1.0)
            score *= compliance_score
            
            # Ensure score is between 0 and 100
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Security score calculation failed: {e}")
            return 50.0
    
    async def _generate_security_recommendations(
        self,
        threats: List[SecurityThreat],
        security_metrics: Dict[str, Any],
        compliance_status: Dict[str, Any],
        encryption_status: Dict[str, Any]
    ) -> List[str]:
        """Generate security recommendations"""
        try:
            recommendations = []
            
            # Threat-based recommendations
            for threat in threats:
                recommendations.extend(threat.mitigation_suggestions)
            
            # Security metrics recommendations
            if not security_metrics.get("encoding_valid", True):
                recommendations.append("Fix character encoding issues")
            
            if security_metrics.get("suspicious_char_ratio", 0) > 0.1:
                recommendations.append("Reduce suspicious character usage")
            
            if security_metrics.get("length_risk") == "high":
                recommendations.append("Consider content length limits")
            
            # Compliance recommendations
            if not compliance_status.get("gdpr", {}).get("compliant", True):
                recommendations.append("Address GDPR compliance violations")
            
            if not compliance_status.get("hipaa", {}).get("compliant", True):
                recommendations.append("Address HIPAA compliance violations")
            
            if not compliance_status.get("pci_dss", {}).get("compliant", True):
                recommendations.append("Address PCI DSS compliance violations")
            
            # Encryption recommendations
            recommendations.extend(encryption_status.get("recommendations", []))
            
            # Remove duplicates and return
            return list(set(recommendations))
            
        except Exception as e:
            logger.warning(f"Security recommendations generation failed: {e}")
            return []
    
    async def encrypt_content(self, content: str, password: str = None) -> Dict[str, Any]:
        """Encrypt content"""
        try:
            if not self.encryption_key:
                raise Exception("Encryption key not available")
            
            # Use provided password or generate one
            if password:
                # Derive key from password
                salt = secrets.token_bytes(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            else:
                key = self.encryption_key
            
            # Encrypt content
            fernet = Fernet(key)
            encrypted_content = fernet.encrypt(content.encode())
            
            return {
                "encrypted_content": base64.urlsafe_b64encode(encrypted_content).decode(),
                "encryption_method": "AES-256-GCM",
                "key_derivation": "PBKDF2-HMAC-SHA256" if password else "Fernet",
                "salt": base64.urlsafe_b64encode(salt).decode() if password else None,
                "encryption_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Content encryption failed: {e}")
            raise
    
    async def decrypt_content(self, encrypted_data: Dict[str, Any], password: str = None) -> str:
        """Decrypt content"""
        try:
            # Get encryption details
            encrypted_content = base64.urlsafe_b64decode(encrypted_data["encrypted_content"])
            encryption_method = encrypted_data.get("encryption_method", "AES-256-GCM")
            
            # Derive key
            if password and encrypted_data.get("salt"):
                salt = base64.urlsafe_b64decode(encrypted_data["salt"])
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            else:
                key = self.encryption_key
            
            # Decrypt content
            fernet = Fernet(key)
            decrypted_content = fernet.decrypt(encrypted_content)
            
            return decrypted_content.decode()
            
        except Exception as e:
            logger.error(f"Content decryption failed: {e}")
            raise
    
    async def create_security_policy(
        self,
        policy_name: str,
        policy_type: str,
        rules: List[Dict[str, Any]]
    ) -> SecurityPolicy:
        """Create a new security policy"""
        try:
            policy_id = f"policy_{int(datetime.now().timestamp())}"
            
            policy = SecurityPolicy(
                policy_id=policy_id,
                policy_name=policy_name,
                policy_type=policy_type,
                rules=rules,
                severity_levels={
                    "critical": "block",
                    "high": "alert",
                    "medium": "log",
                    "low": "monitor"
                },
                enforcement_actions=["block", "alert", "log", "monitor"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.security_policies[policy_id] = policy
            logger.info(f"Security policy created: {policy_id}")
            
            return policy
            
        except Exception as e:
            logger.error(f"Security policy creation failed: {e}")
            raise
    
    async def perform_security_audit(
        self,
        content_list: List[str],
        audit_type: str = "comprehensive"
    ) -> SecurityAudit:
        """Perform comprehensive security audit"""
        try:
            audit_id = f"audit_{int(datetime.now().timestamp())}"
            total_threats = 0
            total_violations = 0
            security_scores = []
            
            # Analyze each content piece
            for content in content_list:
                security_analysis = await self.analyze_content_security(content)
                total_threats += security_analysis.threat_count
                security_scores.append(security_analysis.security_score)
                
                # Count compliance violations
                for compliance in security_analysis.compliance_status.values():
                    if isinstance(compliance, dict) and not compliance.get("compliant", True):
                        total_violations += compliance.get("violations", 0)
            
            # Calculate average security score
            avg_security_score = sum(security_scores) / len(security_scores) if security_scores else 0
            
            # Generate recommendations
            recommendations = []
            if avg_security_score < 70:
                recommendations.append("Overall security score is low - implement security improvements")
            
            if total_threats > 0:
                recommendations.append(f"Address {total_threats} security threats detected")
            
            if total_violations > 0:
                recommendations.append(f"Resolve {total_violations} compliance violations")
            
            return SecurityAudit(
                audit_id=audit_id,
                audit_timestamp=datetime.now(),
                audit_type=audit_type,
                content_analyzed=len(content_list),
                threats_detected=total_threats,
                compliance_violations=total_violations,
                security_score=avg_security_score,
                recommendations=recommendations,
                audit_details={
                    "individual_scores": security_scores,
                    "threat_distribution": {},
                    "compliance_summary": {}
                }
            )
            
        except Exception as e:
            logger.error(f"Security audit failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of security engine"""
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "models_loaded": self.models_loaded,
            "encryption_available": self.encryption_key is not None,
            "security_policies_loaded": len(self.security_policies),
            "threat_patterns_loaded": len(self.threat_patterns),
            "compliance_rules_loaded": len(self.compliance_rules),
            "audit_logs_count": len(self.audit_logs),
            "timestamp": datetime.now().isoformat()
        }


# Global security engine instance
content_security_engine = ContentSecurityEngine()


async def initialize_content_security_engine() -> None:
    """Initialize the global security engine"""
    await content_security_engine.initialize()


async def analyze_content_security(
    content: str,
    content_id: str = "",
    context: Dict[str, Any] = None
) -> ContentSecurity:
    """Analyze content security"""
    return await content_security_engine.analyze_content_security(content, content_id, context)


async def encrypt_content(content: str, password: str = None) -> Dict[str, Any]:
    """Encrypt content"""
    return await content_security_engine.encrypt_content(content, password)


async def decrypt_content(encrypted_data: Dict[str, Any], password: str = None) -> str:
    """Decrypt content"""
    return await content_security_engine.decrypt_content(encrypted_data, password)


async def create_security_policy(
    policy_name: str,
    policy_type: str,
    rules: List[Dict[str, Any]]
) -> SecurityPolicy:
    """Create security policy"""
    return await content_security_engine.create_security_policy(policy_name, policy_type, rules)


async def perform_security_audit(
    content_list: List[str],
    audit_type: str = "comprehensive"
) -> SecurityAudit:
    """Perform security audit"""
    return await content_security_engine.perform_security_audit(content_list, audit_type)


async def get_security_engine_health() -> Dict[str, Any]:
    """Get security engine health status"""
    return await content_security_engine.health_check()


