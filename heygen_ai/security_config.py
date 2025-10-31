"""
Security Configuration Manager for HeyGen AI
============================================

Provides security configuration management, compliance checking,
and security policy enforcement.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
import hashlib
import secrets

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(Enum):
    """Encryption algorithm enumeration."""
    AES_256 = "aes-256"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    CHACHA20 = "chacha20"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    description: str
    level: SecurityLevel
    enabled: bool = True
    rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceConfig:
    """Compliance configuration."""
    standard: str  # e.g., "SOC2", "ISO27001", "GDPR"
    version: str
    requirements: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class SecurityFlags:
    """Security feature flags."""
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    enable_access_control: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_sql_injection_protection: bool = True
    enable_xss_protection: bool = True
    enable_csrf_protection: bool = True
    enable_secure_headers: bool = True
    enable_session_security: bool = True


class SecurityConfigManager:
    """Security configuration manager."""
    
    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        """Initialize security configuration manager."""
        if config_path is None:
            self.config_path = Path("security_config.json")
        elif isinstance(config_path, str):
            self.config_path = Path(config_path)
        else:
            self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Default security configuration
        self._config = {
            "security_level": SecurityLevel.MEDIUM.value,
            "encryption_algorithm": EncryptionAlgorithm.AES_256.value,
            "session_timeout": 3600,  # 1 hour
            "max_login_attempts": 5,
            "password_min_length": 8,
            "password_require_special": True,
            "password_require_numbers": True,
            "password_require_uppercase": True,
            "password_require_lowercase": True,
            "jwt_secret_key": self._generate_secret_key(),
            "jwt_expiration": 3600,
            "rate_limit_requests": 100,
            "rate_limit_window": 3600,
            "audit_log_retention": 90,  # days
            "backup_encryption": True,
            "ssl_required": True,
            "hsts_enabled": True,
            "csp_enabled": True,
            "flags": SecurityFlags().__dict__,
            "policies": {},
            "compliance": {}
        }
        
        self._load_config()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self._config.update(loaded_config)
                self.logger.info(f"Security configuration loaded from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load security configuration: {e}")
        else:
            self.logger.info("No security configuration file found, using defaults")
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            # Create directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            self.logger.info(f"Security configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save security configuration: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """Get current security configuration."""
        return self._config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update security configuration."""
        self._config.update(updates)
        self._save_config()
        self.logger.info("Security configuration updated")
    
    def get_security_level(self) -> SecurityLevel:
        """Get current security level."""
        return SecurityLevel(self._config.get("security_level", SecurityLevel.MEDIUM.value))
    
    def set_security_level(self, level: SecurityLevel):
        """Set security level."""
        self._config["security_level"] = level.value
        self._save_config()
        self.logger.info(f"Security level set to {level.value}")
    
    def get_encryption_algorithm(self) -> EncryptionAlgorithm:
        """Get encryption algorithm."""
        return EncryptionAlgorithm(self._config.get("encryption_algorithm", EncryptionAlgorithm.AES_256.value))
    
    def set_encryption_algorithm(self, algorithm: EncryptionAlgorithm):
        """Set encryption algorithm."""
        self._config["encryption_algorithm"] = algorithm.value
        self._save_config()
        self.logger.info(f"Encryption algorithm set to {algorithm.value}")
    
    def get_security_flags(self) -> SecurityFlags:
        """Get security feature flags."""
        flags_data = self._config.get("flags", {})
        return SecurityFlags(**flags_data)
    
    def set_security_flags(self, flags: SecurityFlags):
        """Set security feature flags."""
        self._config["flags"] = flags.__dict__
        self._save_config()
        self.logger.info("Security flags updated")
    
    def add_policy(self, policy: SecurityPolicy):
        """Add a security policy."""
        self._config["policies"][policy.name] = {
            "description": policy.description,
            "level": policy.level.value,
            "enabled": policy.enabled,
            "rules": policy.rules
        }
        self._save_config()
        self.logger.info(f"Security policy '{policy.name}' added")
    
    def remove_policy(self, policy_name: str):
        """Remove a security policy."""
        if policy_name in self._config["policies"]:
            del self._config["policies"][policy_name]
            self._save_config()
            self.logger.info(f"Security policy '{policy_name}' removed")
    
    def get_policy(self, policy_name: str) -> Optional[SecurityPolicy]:
        """Get a security policy."""
        policy_data = self._config["policies"].get(policy_name)
        if policy_data:
            return SecurityPolicy(
                name=policy_name,
                description=policy_data["description"],
                level=SecurityLevel(policy_data["level"]),
                enabled=policy_data["enabled"],
                rules=policy_data["rules"]
            )
        return None
    
    def list_policies(self) -> List[str]:
        """List all security policy names."""
        return list(self._config["policies"].keys())
    
    def add_compliance_config(self, compliance: ComplianceConfig):
        """Add compliance configuration."""
        self._config["compliance"][compliance.standard] = {
            "version": compliance.version,
            "requirements": compliance.requirements,
            "enabled": compliance.enabled
        }
        self._save_config()
        self.logger.info(f"Compliance configuration for '{compliance.standard}' added")
    
    def get_compliance_config(self, standard: str) -> Optional[ComplianceConfig]:
        """Get compliance configuration."""
        compliance_data = self._config["compliance"].get(standard)
        if compliance_data:
            return ComplianceConfig(
                standard=standard,
                version=compliance_data["version"],
                requirements=compliance_data["requirements"],
                enabled=compliance_data["enabled"]
            )
        return None
    
    def list_compliance_standards(self) -> List[str]:
        """List all compliance standards."""
        return list(self._config["compliance"].keys())
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security requirements."""
        result = {
            "valid": True,
            "errors": [],
            "score": 0
        }
        
        min_length = self._config.get("password_min_length", 8)
        if len(password) < min_length:
            result["valid"] = False
            result["errors"].append(f"Password must be at least {min_length} characters long")
        
        if self._config.get("password_require_uppercase", True):
            if not any(c.isupper() for c in password):
                result["valid"] = False
                result["errors"].append("Password must contain at least one uppercase letter")
        
        if self._config.get("password_require_lowercase", True):
            if not any(c.islower() for c in password):
                result["valid"] = False
                result["errors"].append("Password must contain at least one lowercase letter")
        
        if self._config.get("password_require_numbers", True):
            if not any(c.isdigit() for c in password):
                result["valid"] = False
                result["errors"].append("Password must contain at least one number")
        
        if self._config.get("password_require_special", True):
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                result["valid"] = False
                result["errors"].append("Password must contain at least one special character")
        
        # Calculate password strength score
        score = 0
        if len(password) >= 8:
            score += 1
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        if len(password) >= 12:
            score += 1
        
        result["score"] = score
        
        return result
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a security configuration report."""
        report = {
            "timestamp": str(Path().cwd()),
            "security_level": self.get_security_level().value,
            "encryption_algorithm": self.get_encryption_algorithm().value,
            "security_flags": self.get_security_flags().__dict__,
            "policies_count": len(self._config["policies"]),
            "compliance_standards": len(self._config["compliance"]),
            "recommendations": []
        }
        
        # Generate recommendations based on current configuration
        if self.get_security_level() == SecurityLevel.LOW:
            report["recommendations"].append("Consider increasing security level to MEDIUM or HIGH")
        
        if not self.get_security_flags().enable_encryption:
            report["recommendations"].append("Enable encryption for data protection")
        
        if not self.get_security_flags().enable_audit_logging:
            report["recommendations"].append("Enable audit logging for compliance")
        
        if self._config.get("password_min_length", 8) < 12:
            report["recommendations"].append("Consider increasing minimum password length to 12 characters")
        
        if not self._config.get("ssl_required", True):
            report["recommendations"].append("Enable SSL/TLS requirement for all connections")
        
        return report
    
    def export_config(self, export_path: Path):
        """Export security configuration to a file."""
        try:
            with open(export_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            self.logger.info(f"Security configuration exported to {export_path}")
        except Exception as e:
            self.logger.error(f"Failed to export security configuration: {e}")
            raise
    
    def import_config(self, import_path: Path):
        """Import security configuration from a file."""
        try:
            with open(import_path, 'r') as f:
                imported_config = json.load(f)
                self._config.update(imported_config)
                self._save_config()
            self.logger.info(f"Security configuration imported from {import_path}")
        except Exception as e:
            self.logger.error(f"Failed to import security configuration: {e}")
            raise
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.__init__(self.config_path)
        self._save_config()
        self.logger.info("Security configuration reset to defaults")
    
    def get_compliance_status(self) -> Dict[str, bool]:
        """Get compliance status for all configured standards."""
        status = {}
        for standard in self.list_compliance_standards():
            compliance = self.get_compliance_config(standard)
            if compliance:
                status[standard] = compliance.enabled
        return status
    
    def toggle_compliance_status(self, standard: str):
        """Toggle compliance status for a standard."""
        compliance = self.get_compliance_config(standard)
        if compliance:
            compliance.enabled = not compliance.enabled
            self.add_compliance_config(compliance)
            self.logger.info(f"Compliance status for '{standard}' toggled to {compliance.enabled}")
    
    def get_encryption_settings(self) -> Dict[str, Any]:
        """Get encryption settings."""
        return {
            "is_encryption_at_rest_enabled": self._config.get("backup_encryption", True),
            "algorithm": self._config.get("encryption_algorithm", "aes-256"),
            "key_rotation_days": self._config.get("key_rotation_days", 90)
        }
    
    def get_authentication_settings(self) -> Dict[str, Any]:
        """Get authentication settings."""
        return {
            "is_multi_factor_enabled": self._config.get("mfa_enabled", False),
            "session_timeout": self._config.get("session_timeout", 3600),
            "max_login_attempts": self._config.get("max_login_attempts", 5),
            "password_policy": {
                "min_length": self._config.get("password_min_length", 8),
                "require_special": self._config.get("password_require_special", True),
                "require_numbers": self._config.get("password_require_numbers", True),
                "require_uppercase": self._config.get("password_require_uppercase", True),
                "require_lowercase": self._config.get("password_require_lowercase", True)
            }
        }
    
    def update_authentication_setting(self, key: str, value: Any):
        """Update authentication setting."""
        if key == "is_multi_factor_enabled":
            self._config["mfa_enabled"] = value
        elif key == "session_timeout":
            self._config["session_timeout"] = value
        elif key == "max_login_attempts":
            self._config["max_login_attempts"] = value
        else:
            self._config[key] = value
    
    def save_configuration(self) -> bool:
        """Save configuration to file."""
        try:
            self._save_config()
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        flags = self.get_security_flags()
        
        # Calculate security score
        score = 0
        max_score = 10
        
        if flags.enable_encryption:
            score += 2
        if flags.enable_audit_logging:
            score += 2
        if flags.enable_access_control:
            score += 2
        if flags.enable_rate_limiting:
            score += 1
        if flags.enable_input_validation:
            score += 1
        if flags.enable_sql_injection_protection:
            score += 1
        if flags.enable_xss_protection:
            score += 1
        
        return {
            "security_score": score,
            "max_score": max_score,
            "security_level": self.get_security_level().value,
            "feature_status": {
                "encryption": flags.enable_encryption,
                "audit_logging": flags.enable_audit_logging,
                "access_control": flags.enable_access_control,
                "rate_limiting": flags.enable_rate_limiting,
                "input_validation": flags.enable_input_validation,
                "sql_injection_protection": flags.enable_sql_injection_protection,
                "xss_protection": flags.enable_xss_protection,
                "csrf_protection": flags.enable_csrf_protection,
                "secure_headers": flags.enable_secure_headers,
                "session_security": flags.enable_session_security,
                "firewall": True,  # Default assumption
                "antivirus": True,  # Default assumption
                "intrusion_detection": True  # Default assumption
            },
            "recommendations": self._generate_recommendations(score, max_score)
        }
    
    def _generate_recommendations(self, score: int, max_score: int) -> List[str]:
        """Generate security recommendations based on score."""
        recommendations = []
        
        if score < max_score * 0.7:  # Less than 70%
            recommendations.append("Consider implementing additional security measures")
        
        if score < max_score * 0.5:  # Less than 50%
            recommendations.append("Critical: Security posture needs immediate attention")
        
        flags = self.get_security_flags()
        if not flags.enable_encryption:
            recommendations.append("Enable encryption for data protection")
        if not flags.enable_audit_logging:
            recommendations.append("Enable audit logging for compliance")
        if not flags.enable_access_control:
            recommendations.append("Implement proper access control mechanisms")
        
        return recommendations
