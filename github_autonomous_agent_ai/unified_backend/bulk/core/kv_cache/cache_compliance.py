"""
Cache compliance and audit utilities.

Provides compliance and audit capabilities.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class CacheCompliance:
    """
    Cache compliance manager.
    
    Ensures cache operations comply with policies.
    """
    
    def __init__(
        self,
        cache: Any,
        compliance_level: ComplianceLevel = ComplianceLevel.STANDARD
    ):
        """
        Initialize compliance manager.
        
        Args:
            cache: Cache instance
            compliance_level: Compliance level
        """
        self.cache = cache
        self.compliance_level = compliance_level
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_log = 10000
    
    def check_compliance(self) -> Dict[str, Any]:
        """
        Check cache compliance.
        
        Returns:
            Dictionary with compliance status
        """
        compliance = {
            "timestamp": time.time(),
            "level": self.compliance_level.value,
            "passed": True,
            "violations": [],
            "warnings": []
        }
        
        stats = self.cache.get_stats()
        config = self.cache.config
        
        # Check based on compliance level
        if self.compliance_level == ComplianceLevel.STRICT:
            # Strict checks
            if stats.get("hit_rate", 0.0) < 0.7:
                compliance["violations"].append({
                    "type": "low_hit_rate",
                    "message": f"Hit rate {stats.get('hit_rate', 0.0):.2%} below strict threshold 0.7"
                })
                compliance["passed"] = False
        
        if self.compliance_level in [ComplianceLevel.STRICT, ComplianceLevel.ENTERPRISE]:
            # Memory compliance
            memory_mb = stats.get("storage_memory_mb", 0.0)
            if memory_mb > 2000:
                compliance["violations"].append({
                    "type": "high_memory",
                    "message": f"Memory usage {memory_mb:.2f} MB exceeds threshold"
                })
                compliance["passed"] = False
        
        if self.compliance_level == ComplianceLevel.ENTERPRISE:
            # Enterprise checks
            if not config.use_compression and stats.get("storage_memory_mb", 0.0) > 1000:
                compliance["warnings"].append({
                    "type": "compression_recommended",
                    "message": "Compression recommended for enterprise compliance"
                })
        
        # Log compliance check
        self.audit_log.append(compliance)
        if len(self.audit_log) > self.max_audit_log:
            self.audit_log = self.audit_log[-self.max_audit_log:]
        
        return compliance
    
    def audit_operation(
        self,
        operation: str,
        position: Optional[int] = None,
        success: bool = True
    ) -> None:
        """
        Audit cache operation.
        
        Args:
            operation: Operation type
            position: Optional cache position
            success: Whether operation succeeded
        """
        audit_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "position": position,
            "success": success
        }
        
        self.audit_log.append(audit_entry)
        
        if len(self.audit_log) > self.max_audit_log:
            self.audit_log = self.audit_log[-self.max_audit_log:]
    
    def get_audit_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate audit report.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary with audit report
        """
        filtered_log = self.audit_log
        
        if start_time:
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] >= start_time]
        
        if end_time:
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] <= end_time]
        
        # Count operations
        operation_counts: Dict[str, int] = {}
        for entry in filtered_log:
            op = entry.get("operation", "unknown")
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        # Count successes/failures
        successes = sum(1 for entry in filtered_log if entry.get("success", False))
        failures = len(filtered_log) - successes
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "total_operations": len(filtered_log),
            "successful_operations": successes,
            "failed_operations": failures,
            "operation_counts": operation_counts,
            "compliance_level": self.compliance_level.value
        }


class CacheAuditor:
    """
    Cache auditor.
    
    Provides comprehensive auditing capabilities.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache auditor.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.audit_records: List[Dict[str, Any]] = []
    
    def audit_cache_state(self) -> Dict[str, Any]:
        """
        Audit current cache state.
        
        Returns:
            Dictionary with audit results
        """
        stats = self.cache.get_stats()
        storage = self.cache.storage
        
        audit = {
            "timestamp": time.time(),
            "cache_size": storage.size(),
            "max_tokens": self.cache.config.max_tokens,
            "utilization": storage.size() / max(self.cache.config.max_tokens, 1),
            "memory_mb": storage.get_total_memory_mb(),
            "hit_rate": stats.get("hit_rate", 0.0),
            "issues": []
        }
        
        # Check for issues
        if audit["utilization"] > 0.9:
            audit["issues"].append({
                "type": "high_utilization",
                "severity": "warning",
                "message": f"Cache utilization {audit['utilization']:.2%} is very high"
            })
        
        if audit["memory_mb"] > 1000:
            audit["issues"].append({
                "type": "high_memory",
                "severity": "warning",
                "message": f"Memory usage {audit['memory_mb']:.2f} MB is high"
            })
        
        self.audit_records.append(audit)
        return audit
    
    def get_audit_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit history.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of audit records
        """
        return self.audit_records[-limit:]

