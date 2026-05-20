"""
Distributed Sync - Sincronización Distribuida
==============================================

Sistema de sincronización distribuida con conflict resolution y versionado.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Estado de sincronización."""
    PENDING = "pending"
    SYNCING = "syncing"
    COMPLETED = "completed"
    CONFLICT = "conflict"
    FAILED = "failed"


class ConflictResolution(Enum):
    """Resolución de conflicto."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MANUAL = "manual"
    MERGE = "merge"


@dataclass
class SyncOperation:
    """Operación de sincronización."""
    operation_id: str
    resource_id: str
    resource_type: str
    operation_type: str  # "create", "update", "delete"
    data: Dict[str, Any]
    version: int
    node_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: SyncStatus = SyncStatus.PENDING
    conflict_resolution: Optional[ConflictResolution] = None


@dataclass
class SyncConflict:
    """Conflicto de sincronización."""
    conflict_id: str
    resource_id: str
    resource_type: str
    local_version: int
    remote_version: int
    local_data: Dict[str, Any]
    remote_data: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class DistributedSync:
    """Sistema de sincronización distribuida."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.resources: Dict[str, Dict[str, Any]] = {}  # resource_id -> {data, version, node_id}
        self.operations: Dict[str, SyncOperation] = {}
        self.conflicts: Dict[str, SyncConflict] = {}
        self.pending_operations: List[str] = []
        self.conflict_resolution_strategy: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
        self._lock = asyncio.Lock()
    
    def create_resource(
        self,
        resource_id: str,
        resource_type: str,
        data: Dict[str, Any],
    ) -> str:
        """Crear recurso."""
        operation_id = f"op_{resource_id}_{datetime.now().timestamp()}"
        
        operation = SyncOperation(
            operation_id=operation_id,
            resource_id=resource_id,
            resource_type=resource_type,
            operation_type="create",
            data=data,
            version=1,
            node_id=self.node_id,
        )
        
        async def apply_operation():
            async with self._lock:
                self.resources[resource_id] = {
                    "data": data,
                    "version": 1,
                    "node_id": self.node_id,
                    "last_modified": datetime.now(),
                }
                self.operations[operation_id] = operation
                operation.status = SyncStatus.COMPLETED
        
        asyncio.create_task(apply_operation())
        
        logger.info(f"Created resource: {resource_id}")
        return operation_id
    
    def update_resource(
        self,
        resource_id: str,
        data: Dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> str:
        """Actualizar recurso."""
        operation_id = f"op_{resource_id}_{datetime.now().timestamp()}"
        
        async def apply_operation():
            async with self._lock:
                resource = self.resources.get(resource_id)
                if not resource:
                    raise ValueError(f"Resource not found: {resource_id}")
                
                # Verificar versión
                if expected_version is not None:
                    if resource["version"] != expected_version:
                        # Conflicto detectado
                        conflict_id = f"conflict_{resource_id}_{datetime.now().timestamp()}"
                        conflict = SyncConflict(
                            conflict_id=conflict_id,
                            resource_id=resource_id,
                            resource_type=resource.get("type", "unknown"),
                            local_version=resource["version"],
                            remote_version=expected_version,
                            local_data=resource["data"],
                            remote_data=data,
                        )
                        
                        self.conflicts[conflict_id] = conflict
                        
                        operation = SyncOperation(
                            operation_id=operation_id,
                            resource_id=resource_id,
                            resource_type=resource.get("type", "unknown"),
                            operation_type="update",
                            data=data,
                            version=expected_version,
                            node_id=self.node_id,
                            status=SyncStatus.CONFLICT,
                        )
                        
                        self.operations[operation_id] = operation
                        return operation_id
                
                # Aplicar actualización
                resource["data"] = data
                resource["version"] += 1
                resource["node_id"] = self.node_id
                resource["last_modified"] = datetime.now()
                
                operation = SyncOperation(
                    operation_id=operation_id,
                    resource_id=resource_id,
                    resource_type=resource.get("type", "unknown"),
                    operation_type="update",
                    data=data,
                    version=resource["version"],
                    node_id=self.node_id,
                    status=SyncStatus.COMPLETED,
                )
                
                self.operations[operation_id] = operation
                logger.info(f"Updated resource: {resource_id} to version {resource['version']}")
        
        asyncio.create_task(apply_operation())
        return operation_id
    
    async def sync_from_remote(
        self,
        remote_operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Sincronizar desde operaciones remotas."""
        conflicts = []
        synced = 0
        
        for op_data in remote_operations:
            resource_id = op_data.get("resource_id")
            operation_type = op_data.get("operation_type")
            data = op_data.get("data")
            version = op_data.get("version")
            remote_node_id = op_data.get("node_id")
            
            async with self._lock:
                resource = self.resources.get(resource_id)
                
                if operation_type == "create":
                    if resource_id not in self.resources:
                        self.resources[resource_id] = {
                            "data": data,
                            "version": version,
                            "node_id": remote_node_id,
                            "last_modified": datetime.now(),
                        }
                        synced += 1
                    else:
                        # Ya existe, podría ser conflicto
                        conflicts.append(resource_id)
                
                elif operation_type == "update":
                    if resource:
                        if resource["version"] < version:
                            resource["data"] = data
                            resource["version"] = version
                            resource["node_id"] = remote_node_id
                            synced += 1
                        elif resource["version"] > version:
                            # Conflicto: versión local más nueva
                            conflicts.append(resource_id)
                
                elif operation_type == "delete":
                    if resource:
                        del self.resources[resource_id]
                        synced += 1
        
        return {
            "synced": synced,
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        }
    
    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: ConflictResolution,
        resolved_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resolver conflicto."""
        conflict = self.conflicts.get(conflict_id)
        if not conflict or conflict.resolved:
            return False
        
        async def apply_resolution():
            async with self._lock:
                resource = self.resources.get(conflict.resource_id)
                if not resource:
                    return False
                
                if resolution == ConflictResolution.LAST_WRITE_WINS:
                    # Usar datos más recientes
                    if conflict.local_data.get("timestamp", "") > conflict.remote_data.get("timestamp", ""):
                        data = conflict.local_data
                    else:
                        data = conflict.remote_data
                
                elif resolution == ConflictResolution.FIRST_WRITE_WINS:
                    # Usar datos más antiguos
                    if conflict.local_data.get("timestamp", "") < conflict.remote_data.get("timestamp", ""):
                        data = conflict.local_data
                    else:
                        data = conflict.remote_data
                
                elif resolution == ConflictResolution.MANUAL:
                    data = resolved_data or conflict.local_data
                
                elif resolution == ConflictResolution.MERGE:
                    # Merge simple de diccionarios
                    data = {**conflict.local_data, **conflict.remote_data}
                
                resource["data"] = data
                resource["version"] = max(conflict.local_version, conflict.remote_version) + 1
                conflict.resolved = True
        
        asyncio.create_task(apply_resolution())
        return True
    
    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Obtener recurso."""
        resource = self.resources.get(resource_id)
        if not resource:
            return None
        
        return {
            "resource_id": resource_id,
            "data": resource["data"],
            "version": resource["version"],
            "node_id": resource["node_id"],
            "last_modified": resource["last_modified"].isoformat(),
        }
    
    def get_conflicts(self, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Obtener conflictos."""
        conflicts = list(self.conflicts.values())
        
        if resolved is not None:
            conflicts = [c for c in conflicts if c.resolved == resolved]
        
        return [
            {
                "conflict_id": c.conflict_id,
                "resource_id": c.resource_id,
                "resource_type": c.resource_type,
                "local_version": c.local_version,
                "remote_version": c.remote_version,
                "detected_at": c.detected_at.isoformat(),
                "resolved": c.resolved,
            }
            for c in conflicts
        ]
    
    def get_sync_summary(self) -> Dict[str, Any]:
        """Obtener resumen de sincronización."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for op in self.operations.values():
            by_status[op.status.value] += 1
        
        return {
            "node_id": self.node_id,
            "total_resources": len(self.resources),
            "total_operations": len(self.operations),
            "operations_by_status": dict(by_status),
            "total_conflicts": len(self.conflicts),
            "unresolved_conflicts": len([c for c in self.conflicts.values() if not c.resolved]),
        }
















