"""
Advanced Security
================

Advanced security modules.
"""

from aws.modules.security_advanced.threat_detector import ThreatDetector, ThreatLevel, Threat
from aws.modules.security_advanced.encryption_manager import EncryptionManager
from aws.modules.security_advanced.audit_logger import AuditLogger, AuditEventType, AuditEvent
from aws.modules.security_advanced.compliance_checker import ComplianceChecker, ComplianceStandard, ComplianceCheck

__all__ = [
    "ThreatDetector",
    "ThreatLevel",
    "Threat",
    "EncryptionManager",
    "AuditLogger",
    "AuditEventType",
    "AuditEvent",
    "ComplianceChecker",
    "ComplianceStandard",
    "ComplianceCheck",
]

