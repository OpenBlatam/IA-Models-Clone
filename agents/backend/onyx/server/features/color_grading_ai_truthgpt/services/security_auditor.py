"""
Security Auditor for Color Grading AI
======================================

Advanced security auditing and compliance checking.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityCheck:
    """Security check result."""
    check_name: str
    passed: bool
    level: SecurityLevel
    message: str
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityAudit:
    """Security audit result."""
    overall_score: float  # 0.0 - 1.0
    checks: List[SecurityCheck]
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class SecurityAuditor:
    """
    Security auditor.
    
    Features:
    - Security checks
    - Compliance validation
    - Vulnerability scanning
    - Access control auditing
    - Security recommendations
    """
    
    def __init__(self):
        """Initialize security auditor."""
        self._checks: List[SecurityCheck] = []
        self._audit_history: List[SecurityAudit] = []
    
    def check_input_validation(self, input_data: Any) -> SecurityCheck:
        """
        Check input validation.
        
        Args:
            input_data: Input data to check
            
        Returns:
            Security check result
        """
        issues = []
        recommendations = []
        
        # Check for path traversal
        if isinstance(input_data, str) and (".." in input_data or "//" in input_data):
            issues.append("Potential path traversal detected")
            recommendations.append("Sanitize input paths")
        
        # Check for SQL injection patterns
        if isinstance(input_data, str):
            sql_patterns = ["'", ";", "--", "/*", "*/", "DROP", "DELETE"]
            if any(pattern in input_data.upper() for pattern in sql_patterns):
                issues.append("Potential SQL injection pattern detected")
                recommendations.append("Use parameterized queries")
        
        level = SecurityLevel.CRITICAL if issues else SecurityLevel.LOW
        
        return SecurityCheck(
            check_name="input_validation",
            passed=len(issues) == 0,
            level=level,
            message="Input validation check",
            recommendations=recommendations
        )
    
    def check_authentication(self, auth_config: Dict[str, Any]) -> SecurityCheck:
        """
        Check authentication configuration.
        
        Args:
            auth_config: Authentication configuration
            
        Returns:
            Security check result
        """
        issues = []
        recommendations = []
        
        # Check for weak passwords
        if auth_config.get("min_password_length", 0) < 8:
            issues.append("Password policy too weak")
            recommendations.append("Enforce minimum 8 character passwords")
        
        # Check for missing 2FA
        if not auth_config.get("two_factor_enabled", False):
            issues.append("Two-factor authentication not enabled")
            recommendations.append("Enable 2FA for enhanced security")
        
        level = SecurityLevel.HIGH if issues else SecurityLevel.LOW
        
        return SecurityCheck(
            check_name="authentication",
            passed=len(issues) == 0,
            level=level,
            message="Authentication check",
            recommendations=recommendations
        )
    
    def check_encryption(self, encryption_config: Dict[str, Any]) -> SecurityCheck:
        """
        Check encryption configuration.
        
        Args:
            encryption_config: Encryption configuration
            
        Returns:
            Security check result
        """
        issues = []
        recommendations = []
        
        # Check encryption algorithm
        algorithm = encryption_config.get("algorithm", "")
        if algorithm and "DES" in algorithm.upper():
            issues.append("Weak encryption algorithm")
            recommendations.append("Use AES-256 or stronger")
        
        # Check for unencrypted data
        if not encryption_config.get("encrypt_at_rest", False):
            issues.append("Data at rest not encrypted")
            recommendations.append("Enable encryption at rest")
        
        level = SecurityLevel.HIGH if issues else SecurityLevel.MEDIUM
        
        return SecurityCheck(
            check_name="encryption",
            passed=len(issues) == 0,
            level=level,
            message="Encryption check",
            recommendations=recommendations
        )
    
    def perform_audit(
        self,
        checks: Optional[List[str]] = None
    ) -> SecurityAudit:
        """
        Perform security audit.
        
        Args:
            checks: Optional list of check names to run
            
        Returns:
            Security audit result
        """
        all_checks = []
        
        # Run all checks or specified checks
        if not checks or "input_validation" in checks:
            all_checks.append(self.check_input_validation(""))
        
        if not checks or "authentication" in checks:
            all_checks.append(self.check_authentication({}))
        
        if not checks or "encryption" in checks:
            all_checks.append(self.check_encryption({}))
        
        # Count issues by level
        critical_issues = sum(1 for c in all_checks if not c.passed and c.level == SecurityLevel.CRITICAL)
        high_issues = sum(1 for c in all_checks if not c.passed and c.level == SecurityLevel.HIGH)
        medium_issues = sum(1 for c in all_checks if not c.passed and c.level == SecurityLevel.MEDIUM)
        low_issues = sum(1 for c in all_checks if not c.passed and c.level == SecurityLevel.LOW)
        
        # Calculate overall score
        total_checks = len(all_checks)
        passed_checks = sum(1 for c in all_checks if c.passed)
        overall_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        audit = SecurityAudit(
            overall_score=overall_score,
            checks=all_checks,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues
        )
        
        self._audit_history.append(audit)
        return audit
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security audit statistics."""
        if not self._audit_history:
            return {
                "total_audits": 0,
            }
        
        latest_audit = self._audit_history[-1]
        
        return {
            "total_audits": len(self._audit_history),
            "latest_score": latest_audit.overall_score,
            "critical_issues": latest_audit.critical_issues,
            "high_issues": latest_audit.high_issues,
        }


