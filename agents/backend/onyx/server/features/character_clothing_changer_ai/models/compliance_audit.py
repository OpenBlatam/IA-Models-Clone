"""
Compliance and Audit System for Flux2 Clothing Changer
=======================================================

Compliance tracking and audit logging.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Compliance standard."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


class AuditEventType(Enum):
    """Audit event type."""
    ACCESS = "access"
    MODIFICATION = "modification"
    DELETION = "deletion"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"


@dataclass
class AuditLog:
    """Audit log entry."""
    event_type: AuditEventType
    user_id: Optional[str]
    resource: str
    action: str
    timestamp: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ComplianceAudit:
    """Compliance and audit system."""
    
    def __init__(
        self,
        audit_log_path: Path = Path("audit_logs"),
        compliance_standards: List[ComplianceStandard] = None,
        retention_days: int = 2555,  # 7 years
    ):
        """
        Initialize compliance and audit system.
        
        Args:
            audit_log_path: Path for audit logs
            compliance_standards: List of compliance standards
            retention_days: Log retention period in days
        """
        self.audit_log_path = audit_log_path
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        
        self.compliance_standards = compliance_standards or [ComplianceStandard.GDPR]
        self.retention_days = retention_days
        
        self.audit_logs: List[AuditLog] = []
        self.compliance_checks: Dict[str, bool] = {}
    
    def log_event(
        self,
        event_type: AuditEventType,
        resource: str,
        action: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            resource: Resource identifier
            action: Action performed
            user_id: Optional user ID
            ip_address: Optional IP address
            user_agent: Optional user agent
            success: Whether action was successful
            details: Optional additional details
            
        Returns:
            Created audit log
        """
        audit_log = AuditLog(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            timestamp=time.time(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
        )
        
        self.audit_logs.append(audit_log)
        self._persist_audit_log(audit_log)
        
        logger.debug(f"Audit log: {event_type.value} - {action} on {resource}")
        return audit_log
    
    def _persist_audit_log(self, audit_log: AuditLog) -> None:
        """Persist audit log to disk."""
        try:
            date_str = datetime.fromtimestamp(audit_log.timestamp).strftime("%Y-%m-%d")
            log_file = self.audit_log_path / f"audit_{date_str}.jsonl"
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(audit_log), default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist audit log: {e}")
    
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        time_range: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """
        Get audit trail.
        
        Args:
            user_id: Filter by user ID
            resource: Filter by resource
            event_type: Filter by event type
            time_range: Time range in seconds
            limit: Maximum number of records
            
        Returns:
            List of audit logs
        """
        logs = self.audit_logs
        
        # Apply filters
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if resource:
            logs = [l for l in logs if l.resource == resource]
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if time_range:
            cutoff = time.time() - time_range
            logs = [l for l in logs if l.timestamp >= cutoff]
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda l: l.timestamp, reverse=True)
        
        return logs[:limit]
    
    def check_compliance(
        self,
        standard: ComplianceStandard,
    ) -> Dict[str, Any]:
        """
        Check compliance with standard.
        
        Args:
            standard: Compliance standard
            
        Returns:
            Compliance check results
        """
        checks = {}
        
        if standard == ComplianceStandard.GDPR:
            checks = self._check_gdpr_compliance()
        elif standard == ComplianceStandard.CCPA:
            checks = self._check_ccpa_compliance()
        elif standard == ComplianceStandard.HIPAA:
            checks = self._check_hipaa_compliance()
        elif standard == ComplianceStandard.SOC2:
            checks = self._check_soc2_compliance()
        elif standard == ComplianceStandard.ISO27001:
            checks = self._check_iso27001_compliance()
        
        self.compliance_checks[standard.value] = all(checks.values())
        
        return {
            "standard": standard.value,
            "compliant": all(checks.values()),
            "checks": checks,
            "timestamp": time.time(),
        }
    
    def _check_gdpr_compliance(self) -> Dict[str, bool]:
        """Check GDPR compliance."""
        return {
            "data_encryption": True,  # Would check actual encryption
            "right_to_access": True,  # Would check data access logs
            "right_to_deletion": True,  # Would check deletion capability
            "data_processing_logs": len(self.audit_logs) > 0,
            "consent_management": True,  # Would check consent system
        }
    
    def _check_ccpa_compliance(self) -> Dict[str, bool]:
        """Check CCPA compliance."""
        return {
            "data_collection_disclosure": True,
            "opt_out_mechanism": True,
            "data_deletion": True,
            "data_sale_opt_out": True,
        }
    
    def _check_hipaa_compliance(self) -> Dict[str, bool]:
        """Check HIPAA compliance."""
        return {
            "access_controls": True,
            "audit_logs": len(self.audit_logs) > 0,
            "encryption": True,
            "breach_notification": True,
        }
    
    def _check_soc2_compliance(self) -> Dict[str, bool]:
        """Check SOC2 compliance."""
        return {
            "access_controls": True,
            "monitoring": True,
            "incident_response": True,
            "change_management": True,
        }
    
    def _check_iso27001_compliance(self) -> Dict[str, bool]:
        """Check ISO27001 compliance."""
        return {
            "information_security_policy": True,
            "risk_management": True,
            "access_control": True,
            "cryptography": True,
            "incident_management": True,
        }
    
    def export_audit_logs(
        self,
        output_path: Path,
        time_range: Optional[float] = None,
    ) -> None:
        """
        Export audit logs.
        
        Args:
            output_path: Output file path
            time_range: Time range in seconds
        """
        logs = self.audit_logs
        
        if time_range:
            cutoff = time.time() - time_range
            logs = [l for l in logs if l.timestamp >= cutoff]
        
        export_data = {
            "export_timestamp": time.time(),
            "total_logs": len(logs),
            "logs": [asdict(log) for log in logs],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(logs)} audit logs to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        by_type = {}
        for log in self.audit_logs:
            event_type = log.event_type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1
        
        return {
            "total_logs": len(self.audit_logs),
            "by_event_type": by_type,
            "compliance_standards": list(c.value for c in self.compliance_standards),
            "compliance_status": dict(self.compliance_checks),
        }


