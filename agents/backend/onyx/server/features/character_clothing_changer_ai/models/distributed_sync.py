"""
Distributed Synchronization for Flux2 Clothing Changer
======================================================

Distributed synchronization and state management.
"""

import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Synchronization status."""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class SyncOperation:
    """Synchronization operation."""
    operation_id: str
    resource_type: str
    resource_id: str
    operation: str  # "create", "update", "delete"
    data: Dict[str, Any]
    timestamp: float
    node_id: str
    status: SyncStatus = SyncStatus.PENDING
    conflicts: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []


class DistributedSync:
    """Distributed synchronization system."""
    
    def __init__(
        self,
        node_id: str,
        conflict_resolution: str = "last_write_wins",
    ):
        """
        Initialize distributed sync.
        
        Args:
            node_id: Unique node identifier
            conflict_resolution: Conflict resolution strategy
        """
        self.node_id = node_id
        self.conflict_resolution = conflict_resolution
        
        # Local state
        self.local_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Sync queue
        self.sync_queue: List[SyncOperation] = []
        
        # Sync history
        self.sync_history: List[SyncOperation] = []
        
        # Conflict handlers
        self.conflict_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "sync_operations": 0,
            "conflicts": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
        }
    
    def register_conflict_handler(
        self,
        resource_type: str,
        handler: Callable[[SyncOperation, SyncOperation], Dict[str, Any]],
    ) -> None:
        """
        Register conflict handler.
        
        Args:
            resource_type: Resource type
            handler: Conflict handler function
        """
        self.conflict_handlers[resource_type] = handler
        logger.info(f"Registered conflict handler for {resource_type}")
    
    def create_operation(
        self,
        resource_type: str,
        resource_id: str,
        operation: str,
        data: Dict[str, Any],
    ) -> SyncOperation:
        """
        Create sync operation.
        
        Args:
            resource_type: Resource type
            resource_id: Resource identifier
            operation: Operation type
            data: Operation data
            
        Returns:
            Created sync operation
        """
        operation_id = self._generate_operation_id()
        
        sync_op = SyncOperation(
            operation_id=operation_id,
            resource_type=resource_type,
            resource_id=resource_id,
            operation=operation,
            data=data,
            timestamp=time.time(),
            node_id=self.node_id,
        )
        
        self.sync_queue.append(sync_op)
        self.stats["sync_operations"] += 1
        
        logger.debug(f"Created sync operation: {operation_id}")
        return sync_op
    
    def sync_with_node(
        self,
        remote_operations: List[SyncOperation],
        local_operations: Optional[List[SyncOperation]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronize with remote node.
        
        Args:
            remote_operations: Operations from remote node
            local_operations: Local operations to send
            
        Returns:
            Sync result
        """
        if local_operations is None:
            local_operations = self.sync_queue.copy()
        
        conflicts = []
        synced = []
        errors = []
        
        # Process remote operations
        for remote_op in remote_operations:
            result = self._apply_operation(remote_op)
            
            if result["status"] == "conflict":
                conflicts.append(result)
                self.stats["conflicts"] += 1
            elif result["status"] == "success":
                synced.append(result)
                self.stats["successful_syncs"] += 1
            else:
                errors.append(result)
                self.stats["failed_syncs"] += 1
        
        return {
            "synced": len(synced),
            "conflicts": len(conflicts),
            "errors": len(errors),
            "details": {
                "synced": synced,
                "conflicts": conflicts,
                "errors": errors,
            },
        }
    
    def _apply_operation(self, operation: SyncOperation) -> Dict[str, Any]:
        """Apply sync operation."""
        resource_key = f"{operation.resource_type}:{operation.resource_id}"
        
        # Check for conflicts
        if resource_key in self.local_state:
            local_version = self.local_state[resource_key].get("version", 0)
            remote_version = operation.data.get("version", 0)
            
            if local_version > remote_version:
                # Conflict detected
                return self._resolve_conflict(operation)
        
        # Apply operation
        try:
            if operation.operation == "create":
                self.local_state[resource_key] = {
                    **operation.data,
                    "version": time.time(),
                    "last_sync": time.time(),
                }
            elif operation.operation == "update":
                if resource_key in self.local_state:
                    self.local_state[resource_key].update(operation.data)
                    self.local_state[resource_key]["version"] = time.time()
                    self.local_state[resource_key]["last_sync"] = time.time()
            elif operation.operation == "delete":
                if resource_key in self.local_state:
                    del self.local_state[resource_key]
            
            operation.status = SyncStatus.SYNCED
            self.sync_history.append(operation)
            
            return {
                "status": "success",
                "operation_id": operation.operation_id,
            }
        except Exception as e:
            operation.status = SyncStatus.ERROR
            return {
                "status": "error",
                "operation_id": operation.operation_id,
                "error": str(e),
            }
    
    def _resolve_conflict(
        self,
        remote_operation: SyncOperation,
    ) -> Dict[str, Any]:
        """Resolve conflict."""
        resource_key = f"{remote_operation.resource_type}:{remote_operation.resource_id}"
        local_data = self.local_state[resource_key]
        
        # Use conflict handler if available
        if remote_operation.resource_type in self.conflict_handlers:
            handler = self.conflict_handlers[remote_operation.resource_type]
            resolved = handler(remote_operation, SyncOperation(
                operation_id="local",
                resource_type=remote_operation.resource_type,
                resource_id=remote_operation.resource_id,
                operation="update",
                data=local_data,
                timestamp=local_data.get("last_sync", 0),
                node_id=self.node_id,
            ))
            return {"status": "conflict", "resolved": resolved}
        
        # Default resolution
        if self.conflict_resolution == "last_write_wins":
            # Keep remote (newer)
            return {"status": "conflict", "resolution": "remote_wins"}
        elif self.conflict_resolution == "merge":
            # Merge both
            merged = {**local_data, **remote_operation.data}
            merged["version"] = time.time()
            self.local_state[resource_key] = merged
            return {"status": "conflict", "resolution": "merged"}
        else:
            return {"status": "conflict", "resolution": "manual_required"}
    
    def get_local_state(
        self,
        resource_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get local state.
        
        Args:
            resource_type: Optional resource type filter
            
        Returns:
            Local state
        """
        if resource_type:
            return {
                k: v for k, v in self.local_state.items()
                if k.startswith(f"{resource_type}:")
            }
        return dict(self.local_state)
    
    def _generate_operation_id(self) -> str:
        """Generate operation ID."""
        data = f"{self.node_id}:{time.time()}:{len(self.sync_queue)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sync statistics."""
        return {
            **self.stats,
            "local_resources": len(self.local_state),
            "pending_operations": len(self.sync_queue),
            "sync_history_size": len(self.sync_history),
        }


