"""
Audit Service
=============

Advanced audit service for tracking and logging system activities.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json

from ...shared.events.event_bus import get_event_bus, DomainEvent, EventMetadata


logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    """Audit log levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditAction(str, Enum):
    """Audit actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    EXPORT = "export"
    IMPORT = "import"
    CONFIGURE = "configure"
    EXECUTE = "execute"


class AuditResource(str, Enum):
    """Audit resources"""
    WORKFLOW = "workflow"
    NODE = "node"
    USER = "user"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    API = "api"
    DATABASE = "database"
    FILE = "file"


@dataclass
class AuditLog:
    """Audit log entry"""
    id: str
    timestamp: datetime
    level: AuditLevel
    action: AuditAction
    resource: AuditResource
    resource_id: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    result: str  # success, failure, error
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class AuditConfig:
    """Audit service configuration"""
    enabled: bool = True
    log_level: AuditLevel = AuditLevel.INFO
    retention_days: int = 365
    batch_size: int = 100
    flush_interval: int = 30  # seconds
    
    # Storage settings
    storage_type: str = "memory"  # memory, database, file
    database_url: Optional[str] = None
    file_path: Optional[str] = None
    
    # Security settings
    mask_sensitive_data: bool = True
    sensitive_fields: List[str] = None
    
    # Compliance settings
    gdpr_compliant: bool = True
    sox_compliant: bool = False
    hipaa_compliant: bool = False


class AuditStorage(ABC):
    """Abstract audit storage interface"""
    
    @abstractmethod
    async def store_log(self, log: AuditLog) -> None:
        """Store audit log entry"""
        pass
    
    @abstractmethod
    async def store_logs(self, logs: List[AuditLog]) -> None:
        """Store multiple audit log entries"""
        pass
    
    @abstractmethod
    async def get_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[AuditResource] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Get audit logs"""
        pass
    
    @abstractmethod
    async def get_log_by_id(self, log_id: str) -> Optional[AuditLog]:
        """Get audit log by ID"""
        pass


class MemoryAuditStorage(AuditStorage):
    """In-memory audit storage"""
    
    def __init__(self, max_logs: int = 50000):
        self._logs: List[AuditLog] = []
        self._max_logs = max_logs
        self._lock = asyncio.Lock()
    
    async def store_log(self, log: AuditLog) -> None:
        """Store audit log entry"""
        async with self._lock:
            self._logs.append(log)
            
            # Maintain max logs limit
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs:]
    
    async def store_logs(self, logs: List[AuditLog]) -> None:
        """Store multiple audit log entries"""
        async with self._lock:
            self._logs.extend(logs)
            
            # Maintain max logs limit
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs:]
    
    async def get_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[AuditResource] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Get audit logs"""
        async with self._lock:
            filtered_logs = self._logs
            
            # Apply filters
            if start_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
            
            if end_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
            
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            
            if action:
                filtered_logs = [log for log in filtered_logs if log.action == action]
            
            if resource:
                filtered_logs = [log for log in filtered_logs if log.resource == resource]
            
            if level:
                filtered_logs = [log for log in filtered_logs if log.level == level]
            
            # Sort by timestamp (newest first)
            filtered_logs.sort(key=lambda log: log.timestamp, reverse=True)
            
            return filtered_logs[:limit]
    
    async def get_log_by_id(self, log_id: str) -> Optional[AuditLog]:
        """Get audit log by ID"""
        async with self._lock:
            for log in self._logs:
                if log.id == log_id:
                    return log
            return None


class AuditService:
    """
    Advanced audit service
    
    Provides comprehensive audit logging with compliance features,
    data masking, and retention policies.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self._storage: Optional[AuditStorage] = None
        self._event_bus = get_event_bus()
        self._log_buffer: List[AuditLog] = []
        self._statistics = {
            "logs_created": 0,
            "logs_stored": 0,
            "logs_failed": 0,
            "by_level": {level.value: 0 for level in AuditLevel},
            "by_action": {action.value: 0 for action in AuditAction},
            "by_resource": {resource.value: 0 for resource in AuditResource}
        }
        self._flush_task: Optional[asyncio.Task] = None
        self._initialize_storage()
        self._start_flush_task()
    
    def _initialize_storage(self):
        """Initialize audit storage"""
        if self.config.storage_type == "memory":
            self._storage = MemoryAuditStorage()
        else:
            logger.warning(f"Storage type {self.config.storage_type} not implemented")
            self._storage = MemoryAuditStorage()
    
    def _start_flush_task(self):
        """Start background flush task"""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_logs_periodically())
    
    async def log_activity(
        self,
        action: AuditAction,
        resource: AuditResource,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        result: str = "success",
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        level: Optional[AuditLevel] = None
    ) -> str:
        """Log audit activity"""
        try:
            if not self.config.enabled:
                return ""
            
            # Determine log level
            if level is None:
                level = AuditLevel.ERROR if result == "error" else AuditLevel.INFO
            
            # Check if we should log this level
            if not self._should_log_level(level):
                return ""
            
            # Create audit log
            log_id = f"audit_{datetime.utcnow().timestamp()}_{id(self)}"
            log = AuditLog(
                id=log_id,
                timestamp=datetime.utcnow(),
                level=level,
                action=action,
                resource=resource,
                resource_id=resource_id,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=self._mask_sensitive_data(details or {}),
                result=result,
                error_message=error_message,
                duration_ms=duration_ms
            )
            
            # Add to buffer
            self._log_buffer.append(log)
            
            # Update statistics
            self._statistics["logs_created"] += 1
            self._statistics["by_level"][level.value] += 1
            self._statistics["by_action"][action.value] += 1
            self._statistics["by_resource"][resource.value] += 1
            
            # Flush if buffer is full
            if len(self._log_buffer) >= self.config.batch_size:
                await self._flush_logs()
            
            return log_id
            
        except Exception as e:
            logger.error(f"Failed to log audit activity: {e}")
            self._statistics["logs_failed"] += 1
            return ""
    
    async def log_workflow_created(
        self,
        workflow_id: str,
        name: str,
        created_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log workflow created activity"""
        return await self.log_activity(
            action=AuditAction.CREATE,
            resource=AuditResource.WORKFLOW,
            resource_id=workflow_id,
            user_id=user_id,
            details={
                "workflow_name": name,
                "created_at": created_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_workflow_updated(
        self,
        workflow_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        updated_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log workflow updated activity"""
        return await self.log_activity(
            action=AuditAction.UPDATE,
            resource=AuditResource.WORKFLOW,
            resource_id=workflow_id,
            user_id=user_id,
            details={
                "field": field,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "updated_at": updated_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_workflow_deleted(
        self,
        workflow_id: str,
        deleted_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log workflow deleted activity"""
        return await self.log_activity(
            action=AuditAction.DELETE,
            resource=AuditResource.WORKFLOW,
            resource_id=workflow_id,
            user_id=user_id,
            details={
                "deleted_at": deleted_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_workflow_status_changed(
        self,
        workflow_id: str,
        old_status: str,
        new_status: str,
        changed_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log workflow status changed activity"""
        return await self.log_activity(
            action=AuditAction.UPDATE,
            resource=AuditResource.WORKFLOW,
            resource_id=workflow_id,
            user_id=user_id,
            details={
                "old_status": old_status,
                "new_status": new_status,
                "changed_at": changed_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_node_created(
        self,
        node_id: str,
        title: str,
        created_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log node created activity"""
        return await self.log_activity(
            action=AuditAction.CREATE,
            resource=AuditResource.NODE,
            resource_id=node_id,
            user_id=user_id,
            details={
                "title": title,
                "created_at": created_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_node_updated(
        self,
        node_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        updated_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log node updated activity"""
        return await self.log_activity(
            action=AuditAction.UPDATE,
            resource=AuditResource.NODE,
            resource_id=node_id,
            user_id=user_id,
            details={
                "field": field,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "updated_at": updated_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_node_deleted(
        self,
        node_id: str,
        deleted_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log node deleted activity"""
        return await self.log_activity(
            action=AuditAction.DELETE,
            resource=AuditResource.NODE,
            resource_id=node_id,
            user_id=user_id,
            details={
                "deleted_at": deleted_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_priority_changed(
        self,
        node_id: str,
        old_priority: int,
        new_priority: int,
        changed_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log priority changed activity"""
        return await self.log_activity(
            action=AuditAction.UPDATE,
            resource=AuditResource.NODE,
            resource_id=node_id,
            user_id=user_id,
            details={
                "old_priority": old_priority,
                "new_priority": new_priority,
                "changed_at": changed_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_tag_added(
        self,
        node_id: str,
        tag: str,
        added_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log tag added activity"""
        return await self.log_activity(
            action=AuditAction.UPDATE,
            resource=AuditResource.NODE,
            resource_id=node_id,
            user_id=user_id,
            details={
                "tag": tag,
                "added_at": added_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def log_tag_removed(
        self,
        node_id: str,
        tag: str,
        removed_at: str,
        event_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log tag removed activity"""
        return await self.log_activity(
            action=AuditAction.UPDATE,
            resource=AuditResource.NODE,
            resource_id=node_id,
            user_id=user_id,
            details={
                "tag": tag,
                "removed_at": removed_at,
                "event_id": event_id
            },
            result="success"
        )
    
    async def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[AuditResource] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Get audit logs"""
        try:
            return await self._storage.get_logs(
                start_date=start_date,
                end_date=end_date,
                user_id=user_id,
                action=action,
                resource=resource,
                level=level,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    async def get_audit_log_by_id(self, log_id: str) -> Optional[AuditLog]:
        """Get audit log by ID"""
        try:
            return await self._storage.get_log_by_id(log_id)
        except Exception as e:
            logger.error(f"Failed to get audit log {log_id}: {e}")
            return None
    
    async def _flush_logs(self) -> None:
        """Flush buffered logs to storage"""
        if not self._log_buffer:
            return
        
        try:
            logs_to_flush = self._log_buffer.copy()
            self._log_buffer.clear()
            
            await self._storage.store_logs(logs_to_flush)
            self._statistics["logs_stored"] += len(logs_to_flush)
            
            logger.debug(f"Flushed {len(logs_to_flush)} audit logs")
            
        except Exception as e:
            logger.error(f"Failed to flush audit logs: {e}")
            self._statistics["logs_failed"] += len(self._log_buffer)
    
    async def _flush_logs_periodically(self) -> None:
        """Periodically flush logs to storage"""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_logs()
            except Exception as e:
                logger.error(f"Error in periodic flush task: {e}")
    
    def _should_log_level(self, level: AuditLevel) -> bool:
        """Check if we should log this level"""
        level_priority = {
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2,
            AuditLevel.ERROR: 3,
            AuditLevel.CRITICAL: 4
        }
        
        config_priority = level_priority.get(self.config.log_level, 1)
        log_priority = level_priority.get(level, 1)
        
        return log_priority >= config_priority
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in audit details"""
        if not self.config.mask_sensitive_data:
            return data
        
        sensitive_fields = self.config.sensitive_fields or [
            "password", "token", "secret", "key", "auth", "credential"
        ]
        
        masked_data = data.copy()
        
        for key, value in masked_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                if isinstance(value, str):
                    masked_data[key] = "***MASKED***"
                else:
                    masked_data[key] = "***MASKED***"
        
        return masked_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit service statistics"""
        return {
            **self._statistics,
            "buffer_size": len(self._log_buffer),
            "config": {
                "enabled": self.config.enabled,
                "log_level": self.config.log_level.value,
                "retention_days": self.config.retention_days,
                "batch_size": self.config.batch_size,
                "flush_interval": self.config.flush_interval,
                "storage_type": self.config.storage_type,
                "mask_sensitive_data": self.config.mask_sensitive_data,
                "gdpr_compliant": self.config.gdpr_compliant,
                "sox_compliant": self.config.sox_compliant,
                "hipaa_compliant": self.config.hipaa_compliant
            }
        }




