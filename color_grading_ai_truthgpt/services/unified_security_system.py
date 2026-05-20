"""
Unified Security System for Color Grading AI
=============================================

Consolidates security services:
- SecurityManager (security management)
- SecurityAuditor (security auditing)
- ConfigValidator (configuration validation)
- ValidationFramework (validation framework)

Features:
- Unified security interface
- Input validation
- Security auditing
- Configuration validation
- Threat detection
- Compliance checking
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .security_manager import SecurityManager
from .security_auditor import SecurityAuditor, SecurityAudit, SecurityLevel
from .config_validator import ConfigValidator, ValidationResult as ConfigValidationResult
from .validation_framework import ValidationFramework, ValidationResult as FrameworkValidationResult, ValidationLevel

logger = logging.getLogger(__name__)


class SecurityMode(Enum):
    """Security modes."""
    BASIC = "basic"  # Basic security checks
    STANDARD = "standard"  # Standard security checks
    STRICT = "strict"  # Strict security checks
    PARANOID = "paranoid"  # Maximum security


@dataclass
class UnifiedSecurityResult:
    """Unified security result."""
    secure: bool
    validation_passed: bool
    audit_score: float
    threats_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedSecuritySystem:
    """
    Unified security system.
    
    Consolidates:
    - SecurityManager: Security management
    - SecurityAuditor: Security auditing
    - ConfigValidator: Configuration validation
    - ValidationFramework: Validation framework
    
    Features:
    - Unified security interface
    - Multi-layer validation
    - Security auditing
    - Threat detection
    """
    
    def __init__(
        self,
        security_mode: SecurityMode = SecurityMode.STANDARD,
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        Initialize unified security system.
        
        Args:
            security_mode: Security mode
            validation_level: Validation level
        """
        self.security_mode = security_mode
        self.validation_level = validation_level
        
        # Initialize components
        self.security_manager = SecurityManager()
        self.security_auditor = SecurityAuditor()
        self.config_validator = ConfigValidator(validation_level=ValidationLevel.STRICT)
        self.validation_framework = ValidationFramework(default_level=validation_level)
        
        logger.info(f"Initialized UnifiedSecuritySystem (mode={security_mode.value})")
    
    def validate_input(
        self,
        input_data: Any,
        input_type: str = "general"
    ) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data
            input_type: Input type
            
        Returns:
            True if valid
        """
        # Use SecurityManager for basic validation
        if not self.security_manager.validate_input(input_data, input_type):
            return False
        
        # Use ValidationFramework for advanced validation
        if input_type == "color_params":
            result = self.validation_framework.validate_color_params(input_data)
            return result.valid
        
        return True
    
    def validate_config(
        self,
        config_data: Dict[str, Any],
        schema_name: str
    ) -> ConfigValidationResult:
        """
        Validate configuration.
        
        Args:
            config_data: Configuration data
            schema_name: Schema name
            
        Returns:
            Validation result
        """
        return self.config_validator.validate_config(config_data, schema_name)
    
    def perform_security_audit(
        self,
        checks: Optional[List[str]] = None
    ) -> SecurityAudit:
        """
        Perform security audit.
        
        Args:
            checks: Optional list of checks to run
            
        Returns:
            Security audit result
        """
        return self.security_auditor.perform_audit(checks)
    
    def comprehensive_check(
        self,
        input_data: Any,
        config_data: Optional[Dict[str, Any]] = None,
        config_schema: Optional[str] = None
    ) -> UnifiedSecurityResult:
        """
        Perform comprehensive security check.
        
        Args:
            input_data: Input data to check
            config_data: Optional configuration data
            config_schema: Optional configuration schema
            
        Returns:
            Unified security result
        """
        threats = []
        recommendations = []
        
        # Validate input
        validation_passed = self.validate_input(input_data)
        if not validation_passed:
            threats.append("Input validation failed")
            recommendations.append("Sanitize and validate all inputs")
        
        # Validate configuration if provided
        config_valid = True
        if config_data and config_schema:
            config_result = self.validate_config(config_data, config_schema)
            config_valid = config_result.status.value == "success"
            if not config_valid:
                threats.append("Configuration validation failed")
                recommendations.extend(config_result.errors)
        
        # Perform security audit
        audit = self.perform_security_audit()
        
        # Collect recommendations from audit
        for check in audit.checks:
            if not check.passed:
                threats.append(f"{check.check_name}: {check.message}")
                recommendations.extend(check.recommendations)
        
        secure = validation_passed and config_valid and audit.overall_score >= 0.8
        
        return UnifiedSecurityResult(
            secure=secure,
            validation_passed=validation_passed and config_valid,
            audit_score=audit.overall_score,
            threats_detected=threats,
            recommendations=recommendations
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "security_mode": self.security_mode.value,
            "validation_level": self.validation_level.value,
            "security_manager": self.security_manager is not None,
            "security_auditor": self.security_auditor.get_statistics(),
            "config_validator": self.config_validator.get_registered_schemas(),
            "validation_framework": self.validation_framework.get_statistics(),
        }


