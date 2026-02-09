"""
Snapshot Manager
================

Snapshot management for quick recovery.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """Snapshot definition."""
    id: str
    resource: str
    location: str
    size_bytes: int
    created_at: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class SnapshotManager:
    """Snapshot manager for quick recovery."""
    
    def __init__(self, max_snapshots: int = 10):
        self.max_snapshots = max_snapshots
        self._snapshots: Dict[str, List[Snapshot]] = {}
    
    def create_snapshot(
        self,
        resource_id: str,
        location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Snapshot:
        """Create snapshot."""
        snapshot_id = f"{resource_id}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot = Snapshot(
            id=snapshot_id,
            resource=resource_id,
            location=location or f"snapshots/{snapshot_id}",
            size_bytes=0,  # In production, get actual size
            created_at=datetime.now(),
            tags=tags or {}
        )
        
        if resource_id not in self._snapshots:
            self._snapshots[resource_id] = []
        
        self._snapshots[resource_id].append(snapshot)
        
        # Keep only max_snapshots
        if len(self._snapshots[resource_id]) > self.max_snapshots:
            oldest = self._snapshots[resource_id].pop(0)
            logger.debug(f"Removed oldest snapshot: {oldest.id}")
        
        logger.info(f"Created snapshot: {snapshot_id}")
        return snapshot
    
    def get_snapshots(self, resource_id: str) -> List[Snapshot]:
        """Get snapshots for resource."""
        return self._snapshots.get(resource_id, []).copy()
    
    def get_latest_snapshot(self, resource_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for resource."""
        snapshots = self.get_snapshots(resource_id)
        if not snapshots:
            return None
        return max(snapshots, key=lambda s: s.created_at)
    
    def delete_snapshot(self, resource_id: str, snapshot_id: str) -> bool:
        """Delete snapshot."""
        if resource_id not in self._snapshots:
            return False
        
        snapshots = self._snapshots[resource_id]
        original_count = len(snapshots)
        self._snapshots[resource_id] = [
            s for s in snapshots if s.id != snapshot_id
        ]
        
        deleted = len(snapshots) < original_count
        if deleted:
            logger.info(f"Deleted snapshot: {snapshot_id}")
        
        return deleted
    
    def get_snapshot_stats(self) -> Dict[str, Any]:
        """Get snapshot statistics."""
        total_snapshots = sum(len(snapshots) for snapshots in self._snapshots.values())
        total_size = sum(
            s.size_bytes
            for snapshots in self._snapshots.values()
            for s in snapshots
        )
        
        return {
            "total_snapshots": total_snapshots,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / 1024 / 1024 / 1024,
            "resources_with_snapshots": len(self._snapshots),
            "by_resource": {
                resource_id: len(snapshots)
                for resource_id, snapshots in self._snapshots.items()
            }
        }















