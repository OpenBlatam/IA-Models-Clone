"""
Compliance and Audit Optimizations

Optimizations for:
- Audit logging
- Compliance checks
- Data retention
- Privacy compliance
- Regulatory compliance
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class AuditLogger:
    """Optimized audit logging."""
    
    def __init__(self, audit_file: str = "./audit.log"):
        """
        Initialize audit logger.
        
        Args:
            audit_file: Audit log file path
        """
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        action: str,
        resource: str,
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log audit event.
        
        Args:
            event_type: Type of event
            user_id: User ID (None for system events)
            action: Action performed
            resource: Resource affected
            result: Result (success, failure)
            metadata: Optional metadata
        """
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'metadata': metadata or {}
        }
        
        # Write to audit log
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query audit events.
        
        Args:
            start_date: Start date
            end_date: End date
            user_id: Filter by user ID
            event_type: Filter by event type
            
        Returns:
            List of matching events
        """
        events = []
        
        if not self.audit_file.exists():
            return events
        
        with open(self.audit_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_time = datetime.fromisoformat(event['timestamp'])
                    
                    # Apply filters
                    if start_date and event_time < start_date:
                        continue
                    if end_date and event_time > end_date:
                        continue
                    if user_id and event.get('user_id') != user_id:
                        continue
                    if event_type and event.get('event_type') != event_type:
                        continue
                    
                    events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing audit log line: {e}")
        
        return events


class ComplianceChecker:
    """Compliance checking optimization."""
    
    def __init__(self):
        """Initialize compliance checker."""
        self.checks: Dict[str, callable] = {}
        self.violations: List[Dict[str, Any]] = []
    
    def register_check(self, name: str, check_func: callable) -> None:
        """
        Register compliance check.
        
        Args:
            name: Check name
            check_func: Check function
        """
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all compliance checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    'status': 'compliant' if result else 'non-compliant',
                    'result': result
                }
                
                if not result:
                    self.violations.append({
                        'check': name,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results


class DataRetentionManager:
    """Data retention management."""
    
    def __init__(self, retention_days: int = 90):
        """
        Initialize data retention manager.
        
        Args:
            retention_days: Number of days to retain data
        """
        self.retention_days = retention_days
    
    def should_retain(self, created_at: datetime) -> bool:
        """
        Check if data should be retained.
        
        Args:
            created_at: Creation timestamp
            
        Returns:
            True if should retain
        """
        age = datetime.utcnow() - created_at
        return age.days < self.retention_days
    
    def get_expired_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get items that should be deleted.
        
        Args:
            items: List of items with 'created_at' field
            
        Returns:
            List of expired items
        """
        expired = []
        
        for item in items:
            if 'created_at' in item:
                created_at = datetime.fromisoformat(item['created_at'])
                if not self.should_retain(created_at):
                    expired.append(item)
        
        return expired


class PrivacyCompliance:
    """Privacy compliance optimization."""
    
    @staticmethod
    def anonymize_data(data: Dict[str, Any], fields_to_anonymize: List[str]) -> Dict[str, Any]:
        """
        Anonymize sensitive data.
        
        Args:
            data: Data dictionary
            fields_to_anonymize: Fields to anonymize
            
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized:
                value = str(anonymized[field])
                # Hash the value
                anonymized[field] = hashlib.sha256(value.encode()).hexdigest()[:16]
        
        return anonymized
    
    @staticmethod
    def check_gdpr_compliance(data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Check GDPR compliance.
        
        Args:
            data: Data to check
            
        Returns:
            (is_compliant, violations)
        """
        violations = []
        
        # Check for required fields
        required_fields = ['consent', 'data_subject_rights']
        for field in required_fields:
            if field not in data:
                violations.append(f"Missing required field: {field}")
        
        # Check data minimization
        if 'unnecessary_data' in data:
            violations.append("Contains unnecessary personal data")
        
        return len(violations) == 0, violations








