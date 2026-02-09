"""
Data Versioning for Flux2 Clothing Changer
==========================================

Advanced data versioning and change tracking.
"""

import time
import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataVersion:
    """Data version."""
    version_id: str
    data_id: str
    version_number: int
    data: Dict[str, Any]
    checksum: str
    created_at: float
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataChange:
    """Data change."""
    change_id: str
    data_id: str
    from_version: int
    to_version: int
    changes: Dict[str, Any]
    timestamp: float
    changed_by: Optional[str] = None


class DataVersioning:
    """Advanced data versioning system."""
    
    def __init__(self):
        """Initialize data versioning."""
        self.versions: Dict[str, List[DataVersion]] = defaultdict(list)
        self.changes: Dict[str, List[DataChange]] = defaultdict(list)
    
    def create_version(
        self,
        data_id: str,
        data: Dict[str, Any],
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataVersion:
        """
        Create data version.
        
        Args:
            data_id: Data identifier
            data: Data dictionary
            created_by: Optional creator identifier
            metadata: Optional metadata
            
        Returns:
            Created version
        """
        # Calculate checksum
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Get next version number
        existing_versions = self.versions.get(data_id, [])
        version_number = len(existing_versions) + 1
        
        version_id = f"{data_id}_v{version_number}"
        
        version = DataVersion(
            version_id=version_id,
            data_id=data_id,
            version_number=version_number,
            data=data,
            checksum=checksum,
            created_at=time.time(),
            created_by=created_by,
            metadata=metadata or {},
        )
        
        # Track changes if previous version exists
        if existing_versions:
            previous_version = existing_versions[-1]
            changes = self._calculate_changes(previous_version.data, data)
            
            change = DataChange(
                change_id=f"change_{int(time.time() * 1000)}",
                data_id=data_id,
                from_version=previous_version.version_number,
                to_version=version_number,
                changes=changes,
                timestamp=time.time(),
                changed_by=created_by,
            )
            
            self.changes[data_id].append(change)
        
        self.versions[data_id].append(version)
        logger.info(f"Created version {version_number} for {data_id}")
        
        return version
    
    def get_version(
        self,
        data_id: str,
        version_number: Optional[int] = None,
    ) -> Optional[DataVersion]:
        """
        Get data version.
        
        Args:
            data_id: Data identifier
            version_number: Optional version number (latest if None)
            
        Returns:
            Data version or None
        """
        if data_id not in self.versions:
            return None
        
        versions = self.versions[data_id]
        
        if version_number is None:
            return versions[-1] if versions else None
        
        for version in versions:
            if version.version_number == version_number:
                return version
        
        return None
    
    def get_all_versions(self, data_id: str) -> List[DataVersion]:
        """Get all versions for data."""
        return self.versions.get(data_id, []).copy()
    
    def get_changes(
        self,
        data_id: str,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
    ) -> List[DataChange]:
        """
        Get changes between versions.
        
        Args:
            data_id: Data identifier
            from_version: Optional start version
            to_version: Optional end version
            
        Returns:
            List of changes
        """
        if data_id not in self.changes:
            return []
        
        changes = self.changes[data_id]
        
        if from_version is not None:
            changes = [c for c in changes if c.from_version >= from_version]
        
        if to_version is not None:
            changes = [c for c in changes if c.to_version <= to_version]
        
        return changes
    
    def _calculate_changes(
        self,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate changes between two data dictionaries."""
        changes = {
            "added": {},
            "modified": {},
            "removed": {},
        }
        
        # Find added and modified
        for key, new_value in new_data.items():
            if key not in old_data:
                changes["added"][key] = new_value
            elif old_data[key] != new_value:
                changes["modified"][key] = {
                    "old": old_data[key],
                    "new": new_value,
                }
        
        # Find removed
        for key in old_data:
            if key not in new_data:
                changes["removed"][key] = old_data[key]
        
        return changes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning statistics."""
        return {
            "total_data_items": len(self.versions),
            "total_versions": sum(len(versions) for versions in self.versions.values()),
            "total_changes": sum(len(changes) for changes in self.changes.values()),
        }


