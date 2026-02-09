"""
State Synchronization for Flux2 Clothing Changer
================================================

Advanced state synchronization and consistency system.
"""

import time
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SyncState(Enum):
    """Synchronization state."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class StateSnapshot:
    """State snapshot."""
    snapshot_id: str
    state_key: str
    state_data: Dict[str, Any]
    checksum: str
    timestamp: float
    version: int = 1


class StateSync:
    """Advanced state synchronization system."""
    
    def __init__(self):
        """Initialize state sync."""
        self.states: Dict[str, StateSnapshot] = {}
        self.sync_queue: List[Dict[str, Any]] = []
        self.conflicts: List[Dict[str, Any]] = []
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate state checksum."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def save_state(
        self,
        state_key: str,
        state_data: Dict[str, Any],
    ) -> StateSnapshot:
        """
        Save state snapshot.
        
        Args:
            state_key: State key
            state_data: State data
            
        Returns:
            Created snapshot
        """
        checksum = self._calculate_checksum(state_data)
        
        # Check if state exists
        if state_key in self.states:
            existing = self.states[state_key]
            if existing.checksum == checksum:
                # No change
                return existing
            version = existing.version + 1
        else:
            version = 1
        
        snapshot_id = f"{state_key}_{version}_{int(time.time())}"
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            state_key=state_key,
            state_data=state_data,
            checksum=checksum,
            timestamp=time.time(),
            version=version,
        )
        
        self.states[state_key] = snapshot
        logger.debug(f"Saved state: {state_key} (v{version})")
        
        return snapshot
    
    def get_state(
        self,
        state_key: str,
    ) -> Optional[StateSnapshot]:
        """Get state snapshot."""
        return self.states.get(state_key)
    
    def sync_states(
        self,
        remote_states: List[StateSnapshot],
    ) -> Dict[str, Any]:
        """
        Synchronize with remote states.
        
        Args:
            remote_states: List of remote state snapshots
            
        Returns:
            Sync result
        """
        synced = []
        conflicts = []
        errors = []
        
        for remote_snapshot in remote_states:
            state_key = remote_snapshot.state_key
            
            if state_key not in self.states:
                # New state, accept
                self.states[state_key] = remote_snapshot
                synced.append(state_key)
            else:
                local_snapshot = self.states[state_key]
                
                if local_snapshot.checksum == remote_snapshot.checksum:
                    # Already synced
                    synced.append(state_key)
                elif local_snapshot.version < remote_snapshot.version:
                    # Remote is newer, accept
                    self.states[state_key] = remote_snapshot
                    synced.append(state_key)
                elif local_snapshot.version > remote_snapshot.version:
                    # Local is newer, conflict
                    conflicts.append({
                        "state_key": state_key,
                        "local_version": local_snapshot.version,
                        "remote_version": remote_snapshot.version,
                    })
                else:
                    # Same version, different checksum - conflict
                    conflicts.append({
                        "state_key": state_key,
                        "local_checksum": local_snapshot.checksum,
                        "remote_checksum": remote_snapshot.checksum,
                    })
        
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
    
    def resolve_conflict(
        self,
        state_key: str,
        resolution: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Resolve state conflict.
        
        Args:
            state_key: State key
            resolution: Resolution strategy ("local", "remote", "merge", "custom")
            data: Optional custom data for "custom" resolution
            
        Returns:
            True if resolved
        """
        if state_key not in self.states:
            return False
        
        if resolution == "custom" and data:
            snapshot = self.states[state_key]
            new_snapshot = StateSnapshot(
                snapshot_id=f"{state_key}_resolved_{int(time.time())}",
                state_key=state_key,
                state_data=data,
                checksum=self._calculate_checksum(data),
                timestamp=time.time(),
                version=snapshot.version + 1,
            )
            self.states[state_key] = new_snapshot
            logger.info(f"Resolved conflict for {state_key} with custom data")
            return True
        
        # Other resolutions would require remote state
        logger.warning(f"Conflict resolution {resolution} not fully implemented")
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get state sync statistics."""
        return {
            "total_states": len(self.states),
            "pending_syncs": len(self.sync_queue),
            "conflicts": len(self.conflicts),
        }


