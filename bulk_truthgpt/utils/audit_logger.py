"""
Audit Logger
============

Advanced audit logging for compliance and security tracking.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class AuditEvent:
    """Audit event."""
    event_id: str
    event_type: str
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

class AuditLogger:
    """Advanced audit logger."""
    
    def __init__(self, log_path: str = "./logs/audit.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[AuditEvent] = []
        self.max_memory_events = 1000
        self.write_queue = asyncio.Queue()
        self.write_task = None
        self.is_running = False
    
    async def log_event(
        self,
        event_type: str,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log an audit event."""
        import uuid
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        # Add to memory
        self.events.append(event)
        if len(self.events) > self.max_memory_events:
            self.events = self.events[-self.max_memory_events:]
        
        # Queue for writing
        await self.write_queue.put(event)
        
        logger.info(f"Audit event logged: {event_type} - {action} on {resource}")
    
    async def _write_loop(self):
        """Background write loop."""
        while self.is_running:
            try:
                # Get batch of events
                events = []
                try:
                    while len(events) < 100:
                        event = await asyncio.wait_for(
                            self.write_queue.get(),
                            timeout=1.0
                        )
                        events.append(event)
                except asyncio.TimeoutError:
                    pass
                
                if events:
                    await self._write_events(events)
                
            except Exception as e:
                logger.error(f"Audit write loop error: {e}")
                await asyncio.sleep(1)
    
    async def _write_events(self, events: List[AuditEvent]):
        """Write events to file."""
        try:
            async with aiofiles.open(self.log_path, 'a') as f:
                for event in events:
                    event_dict = asdict(event)
                    event_dict['timestamp'] = event.timestamp.isoformat()
                    line = json.dumps(event_dict) + '\n'
                    await f.write(line)
        except Exception as e:
            logger.error(f"Failed to write audit events: {e}")
    
    async def start(self):
        """Start audit logger."""
        if self.is_running:
            return
        
        self.is_running = True
        self.write_task = asyncio.create_task(self._write_loop())
        logger.info("Audit logger started")
    
    async def stop(self):
        """Stop audit logger."""
        self.is_running = False
        
        # Write remaining events
        remaining = []
        while not self.write_queue.empty():
            try:
                event = self.write_queue.get_nowait()
                remaining.append(event)
            except:
                break
        
        if remaining:
            await self._write_events(remaining)
        
        if self.write_task:
            self.write_task.cancel()
            try:
                await self.write_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Audit logger stopped")
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit events."""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if resource:
            events = [e for e in events if e.resource == resource]
        
        events = events[-limit:]
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "user_id": e.user_id,
                "action": e.action,
                "resource": e.resource,
                "timestamp": e.timestamp.isoformat(),
                "success": e.success
            }
            for e in events
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        total_events = len(self.events)
        success_count = sum(1 for e in self.events if e.success)
        failure_count = total_events - success_count
        
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            "total_events": total_events,
            "success_count": success_count,
            "failure_count": failure_count,
            "event_types": event_types,
            "queue_size": self.write_queue.qsize()
        }

# Global instance
audit_logger = AuditLogger()

















