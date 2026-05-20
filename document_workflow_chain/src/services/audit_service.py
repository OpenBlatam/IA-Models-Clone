"""
Audit Service - Advanced Implementation
======================================

Advanced audit service with comprehensive event tracking and reporting.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event type enumeration"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    WORKFLOW_EXECUTED = "workflow_executed"
    NODE_ADDED = "node_added"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    NODE_EXECUTED = "node_executed"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"


class AuditSeverity(str, Enum):
    """Audit severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditService:
    """Advanced audit service with comprehensive event tracking"""
    
    def __init__(self):
        self.audit_logs = []
        self.audit_policies = {
            "retention_days": 365,
            "max_log_size_mb": 100,
            "compression_enabled": True,
            "encryption_enabled": True,
            "real_time_alerts": True
        }
        
        # Audit statistics
        self.stats = {
            "total_events": 0,
            "events_by_type": {event_type.value: 0 for event_type in AuditEventType},
            "events_by_severity": {severity.value: 0 for severity in AuditSeverity},
            "events_by_user": {},
            "events_today": 0,
            "events_this_week": 0,
            "events_this_month": 0
        }
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[int] = None,
        details: Dict[str, Any] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log audit event"""
        try:
            # Create audit event
            event = {
                "id": f"audit_{len(self.audit_logs) + 1}",
                "event_type": event_type.value,
                "user_id": user_id,
                "details": details or {},
                "severity": severity.value,
                "resource_id": resource_id,
                "resource_type": resource_type,
                "ip_address": ip_address or "127.0.0.1",
                "user_agent": user_agent or "DocumentWorkflowChain/3.0",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "source": "workflow_chain",
                    "version": "3.0.0",
                    "environment": "production"
                }
            }
            
            # Store audit event
            self.audit_logs.append(event)
            
            # Update statistics
            self._update_statistics(event)
            
            # Check for real-time alerts
            if self.audit_policies["real_time_alerts"]:
                await self._check_real_time_alerts(event)
            
            logger.info(f"Audit event logged: {event_type.value} - {severity.value}")
            return event
        
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return {"error": str(e)}
    
    async def get_audit_logs(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[int] = None,
        severity: Optional[AuditSeverity] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get audit logs with filtering"""
        try:
            # Filter logs
            filtered_logs = self.audit_logs.copy()
            
            # Filter by event type
            if event_type:
                filtered_logs = [log for log in filtered_logs if log["event_type"] == event_type.value]
            
            # Filter by user ID
            if user_id:
                filtered_logs = [log for log in filtered_logs if log["user_id"] == user_id]
            
            # Filter by severity
            if severity:
                filtered_logs = [log for log in filtered_logs if log["severity"] == severity.value]
            
            # Filter by date range
            if start_date:
                filtered_logs = [
                    log for log in filtered_logs
                    if datetime.fromisoformat(log["timestamp"]) >= start_date
                ]
            
            if end_date:
                filtered_logs = [
                    log for log in filtered_logs
                    if datetime.fromisoformat(log["timestamp"]) <= end_date
                ]
            
            # Sort by timestamp (newest first)
            filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Apply pagination
            total_count = len(filtered_logs)
            paginated_logs = filtered_logs[offset:offset + limit]
            
            return {
                "logs": paginated_logs,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return {"error": str(e)}
    
    async def get_audit_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit summary for date range"""
        try:
            # Default to last 30 days if no date range provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Filter logs by date range
            filtered_logs = [
                log for log in self.audit_logs
                if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
            ]
            
            # Calculate summary statistics
            summary = {
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_events": len(filtered_logs),
                "events_by_type": {},
                "events_by_severity": {},
                "events_by_user": {},
                "top_events": [],
                "security_events": 0,
                "system_events": 0,
                "user_events": 0,
                "api_events": 0
            }
            
            # Count events by type
            for log in filtered_logs:
                event_type = log["event_type"]
                if event_type not in summary["events_by_type"]:
                    summary["events_by_type"][event_type] = 0
                summary["events_by_type"][event_type] += 1
                
                # Count by severity
                severity = log["severity"]
                if severity not in summary["events_by_severity"]:
                    summary["events_by_severity"][severity] = 0
                summary["events_by_severity"][severity] += 1
                
                # Count by user
                user_id = log["user_id"]
                if user_id:
                    if user_id not in summary["events_by_user"]:
                        summary["events_by_user"][user_id] = 0
                    summary["events_by_user"][user_id] += 1
                
                # Categorize events
                if "security" in event_type.lower():
                    summary["security_events"] += 1
                elif "system" in event_type.lower():
                    summary["system_events"] += 1
                elif "user" in event_type.lower():
                    summary["user_events"] += 1
                elif "api" in event_type.lower():
                    summary["api_events"] += 1
            
            # Get top events
            summary["top_events"] = sorted(
                summary["events_by_type"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return summary
        
        except Exception as e:
            logger.error(f"Failed to get audit summary: {e}")
            return {"error": str(e)}
    
    async def get_user_activity(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get user activity audit trail"""
        try:
            # Default to last 30 days if no date range provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Filter logs by user and date range
            user_logs = [
                log for log in self.audit_logs
                if log["user_id"] == user_id
                and start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
            ]
            
            # Calculate user activity
            activity = {
                "user_id": user_id,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_events": len(user_logs),
                "events_by_type": {},
                "events_by_severity": {},
                "activity_timeline": [],
                "most_active_hour": 0,
                "most_active_day": "unknown",
                "sessions": {},
                "resources_accessed": set()
            }
            
            # Process user logs
            for log in user_logs:
                event_type = log["event_type"]
                if event_type not in activity["events_by_type"]:
                    activity["events_by_type"][event_type] = 0
                activity["events_by_type"][event_type] += 1
                
                severity = log["severity"]
                if severity not in activity["events_by_severity"]:
                    activity["events_by_severity"][severity] = 0
                activity["events_by_severity"][severity] += 1
                
                # Track resources accessed
                if log["resource_id"]:
                    activity["resources_accessed"].add(log["resource_id"])
                
                # Track sessions
                session_id = log["session_id"]
                if session_id:
                    if session_id not in activity["sessions"]:
                        activity["sessions"][session_id] = {
                            "start_time": log["timestamp"],
                            "end_time": log["timestamp"],
                            "event_count": 0
                        }
                    activity["sessions"][session_id]["event_count"] += 1
                    activity["sessions"][session_id]["end_time"] = log["timestamp"]
            
            # Convert set to list for JSON serialization
            activity["resources_accessed"] = list(activity["resources_accessed"])
            
            return activity
        
        except Exception as e:
            logger.error(f"Failed to get user activity: {e}")
            return {"error": str(e)}
    
    async def get_security_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None
    ) -> Dict[str, Any]:
        """Get security-related audit events"""
        try:
            # Default to last 7 days if no date range provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=7)
            
            # Filter security events
            security_events = [
                log for log in self.audit_logs
                if "security" in log["event_type"].lower()
                and start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
            ]
            
            # Filter by severity if specified
            if severity:
                security_events = [
                    log for log in security_events
                    if log["severity"] == severity.value
                ]
            
            # Calculate security metrics
            security_metrics = {
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_security_events": len(security_events),
                "events_by_severity": {},
                "events_by_type": {},
                "threat_level": "low",
                "recommendations": []
            }
            
            # Count events by severity and type
            for event in security_events:
                severity_level = event["severity"]
                if severity_level not in security_metrics["events_by_severity"]:
                    security_metrics["events_by_severity"][severity_level] = 0
                security_metrics["events_by_severity"][severity_level] += 1
                
                event_type = event["event_type"]
                if event_type not in security_metrics["events_by_type"]:
                    security_metrics["events_by_type"][event_type] = 0
                security_metrics["events_by_type"][event_type] += 1
            
            # Determine threat level
            critical_count = security_metrics["events_by_severity"].get("critical", 0)
            high_count = security_metrics["events_by_severity"].get("high", 0)
            
            if critical_count > 0:
                security_metrics["threat_level"] = "critical"
            elif high_count > 5:
                security_metrics["threat_level"] = "high"
            elif high_count > 0:
                security_metrics["threat_level"] = "medium"
            
            # Generate recommendations
            security_metrics["recommendations"] = self._generate_security_recommendations(security_metrics)
            
            return {
                "events": security_events,
                "metrics": security_metrics
            }
        
        except Exception as e:
            logger.error(f"Failed to get security events: {e}")
            return {"error": str(e)}
    
    async def export_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export audit logs in specified format"""
        try:
            # Default to last 30 days if no date range provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Filter logs by date range
            filtered_logs = [
                log for log in self.audit_logs
                if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
            ]
            
            # Export in specified format
            if format.lower() == "json":
                export_data = {
                    "export_info": {
                        "date_range": {
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat()
                        },
                        "total_records": len(filtered_logs),
                        "export_timestamp": datetime.utcnow().isoformat(),
                        "format": "json"
                    },
                    "audit_logs": filtered_logs
                }
            elif format.lower() == "csv":
                # Convert to CSV format
                csv_data = "timestamp,event_type,user_id,severity,resource_id,details\n"
                for log in filtered_logs:
                    csv_data += f"{log['timestamp']},{log['event_type']},{log['user_id']},{log['severity']},{log['resource_id']},{json.dumps(log['details'])}\n"
                export_data = {"csv_data": csv_data}
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return export_data
        
        except Exception as e:
            logger.error(f"Failed to export audit logs: {e}")
            return {"error": str(e)}
    
    async def _check_real_time_alerts(self, event: Dict[str, Any]) -> None:
        """Check for real-time security alerts"""
        try:
            # Check for critical security events
            if event["severity"] == "critical" and "security" in event["event_type"].lower():
                await self._send_security_alert(event)
            
            # Check for multiple failed login attempts
            if event["event_type"] == "user_login" and not event["details"].get("success", True):
                await self._check_failed_login_pattern(event)
        
        except Exception as e:
            logger.error(f"Failed to check real-time alerts: {e}")
    
    async def _send_security_alert(self, event: Dict[str, Any]) -> None:
        """Send security alert"""
        try:
            alert = {
                "type": "security_alert",
                "severity": "critical",
                "event": event,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Critical security event detected: {event['event_type']}"
            }
            
            logger.critical(f"SECURITY ALERT: {alert['message']}")
            
            # In production, send to security team, SIEM, etc.
        
        except Exception as e:
            logger.error(f"Failed to send security alert: {e}")
    
    async def _check_failed_login_pattern(self, event: Dict[str, Any]) -> None:
        """Check for failed login patterns"""
        try:
            user_id = event["user_id"]
            ip_address = event["ip_address"]
            
            # Count recent failed logins for this user/IP
            recent_failed_logins = [
                log for log in self.audit_logs[-100:]  # Check last 100 logs
                if log["event_type"] == "user_login"
                and not log["details"].get("success", True)
                and (log["user_id"] == user_id or log["ip_address"] == ip_address)
                and datetime.fromisoformat(log["timestamp"]) > datetime.utcnow() - timedelta(minutes=15)
            ]
            
            if len(recent_failed_logins) >= 5:
                await self._send_brute_force_alert(user_id, ip_address, len(recent_failed_logins))
        
        except Exception as e:
            logger.error(f"Failed to check failed login pattern: {e}")
    
    async def _send_brute_force_alert(self, user_id: int, ip_address: str, attempt_count: int) -> None:
        """Send brute force attack alert"""
        try:
            alert = {
                "type": "brute_force_alert",
                "severity": "high",
                "user_id": user_id,
                "ip_address": ip_address,
                "attempt_count": attempt_count,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Potential brute force attack detected: {attempt_count} failed login attempts"
            }
            
            logger.warning(f"BRUTE FORCE ALERT: {alert['message']}")
        
        except Exception as e:
            logger.error(f"Failed to send brute force alert: {e}")
    
    def _generate_security_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on metrics"""
        recommendations = []
        
        critical_count = metrics["events_by_severity"].get("critical", 0)
        high_count = metrics["events_by_severity"].get("high", 0)
        
        if critical_count > 0:
            recommendations.append("Immediate action required: Critical security events detected")
        
        if high_count > 5:
            recommendations.append("Review and strengthen authentication mechanisms")
        
        if metrics["events_by_type"].get("user_login", 0) > 100:
            recommendations.append("Consider implementing additional authentication factors")
        
        if not recommendations:
            recommendations.append("Security posture appears healthy")
        
        return recommendations
    
    def _update_statistics(self, event: Dict[str, Any]) -> None:
        """Update audit statistics"""
        try:
            self.stats["total_events"] += 1
            self.stats["events_by_type"][event["event_type"]] += 1
            self.stats["events_by_severity"][event["severity"]] += 1
            
            # Update user statistics
            user_id = event["user_id"]
            if user_id:
                if user_id not in self.stats["events_by_user"]:
                    self.stats["events_by_user"][user_id] = 0
                self.stats["events_by_user"][user_id] += 1
            
            # Update time-based statistics
            event_time = datetime.fromisoformat(event["timestamp"])
            now = datetime.utcnow()
            
            if event_time.date() == now.date():
                self.stats["events_today"] += 1
            
            if event_time >= now - timedelta(days=7):
                self.stats["events_this_week"] += 1
            
            if event_time >= now - timedelta(days=30):
                self.stats["events_this_month"] += 1
        
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    async def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit service statistics"""
        try:
            return {
                "total_events": self.stats["total_events"],
                "events_by_type": self.stats["events_by_type"],
                "events_by_severity": self.stats["events_by_severity"],
                "events_by_user": self.stats["events_by_user"],
                "events_today": self.stats["events_today"],
                "events_this_week": self.stats["events_this_week"],
                "events_this_month": self.stats["events_this_month"],
                "policies": self.audit_policies,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get audit stats: {e}")
            return {"error": str(e)}


# Global audit service instance
audit_service = AuditService()


