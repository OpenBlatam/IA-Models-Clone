from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import json
import yaml
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import difflib
import logging
from copy import deepcopy
from typing import Any, List, Dict, Optional
import asyncio
"""
Configuration Versioning System
==============================

This module provides comprehensive configuration versioning with:
- Configuration change tracking
- Diff generation and visualization
- Version history and rollback
- Configuration validation
- Automatic backup and restore
- Integration with git version control
"""


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConfigVersion:
    """Represents a configuration version."""
    
    version_id: str
    timestamp: str
    description: str
    author: str
    config_hash: str
    file_path: str
    config_data: Dict[str, Any]
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ConfigDiff:
    """Represents differences between configuration versions."""
    
    old_version: str
    new_version: str
    added_keys: List[str]
    removed_keys: List[str]
    modified_keys: List[str]
    diff_text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigDiff':
        """Create from dictionary."""
        return cls(**data)


class ConfigVersioning:
    """Configuration versioning system."""
    
    def __init__(self, config_dir: str = "config_versions"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Version storage
        self.versions_file = self.config_dir / "versions.json"
        self.versions: Dict[str, ConfigVersion] = {}
        
        # Load existing versions
        self._load_versions()
        
        logger.info(f"Configuration versioning initialized: {self.config_dir}")
    
    def _load_versions(self) -> Any:
        """Load existing versions from storage."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    self.versions = {
                        version_id: ConfigVersion.from_dict(version_data)
                        for version_id, version_data in data.items()
                    }
                logger.info(f"Loaded {len(self.versions)} configuration versions")
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
                self.versions = {}
    
    def _save_versions(self) -> Any:
        """Save versions to storage."""
        try:
            with open(self.versions_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(
                    {vid: version.to_dict() for vid, version in self.versions.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def _generate_version_id(self, config_data: Dict[str, Any], timestamp: str) -> str:
        """Generate unique version ID."""
        content = json.dumps(config_data, sort_keys=True) + timestamp
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate hash of configuration data."""
        content = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_path.suffix.lower() in ['.json']:
                return json.load(f)
            else:
                # Try JSON first, then YAML
                try:
                    f.seek(0)
                    return json.load(f)
                except:
                    f.seek(0)
                    return yaml.safe_load(f)
    
    def _save_config_file(self, file_path: str, config_data: Dict[str, Any]):
        """Save configuration to file."""
        file_path = Path(file_path)
        
        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() in ['.json']:
                json.dump(config_data, f, indent=2)
            else:
                # Default to JSON
                json.dump(config_data, f, indent=2)
    
    def create_version(
        self,
        config_file: str,
        description: str,
        author: str = "system",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new configuration version."""
        try:
            # Load current configuration
            config_data = self._load_config_file(config_file)
            
            # Generate version info
            timestamp = datetime.now().isoformat()
            config_hash = self._calculate_config_hash(config_data)
            version_id = self._generate_version_id(config_data, timestamp)
            
            # Find parent version (most recent)
            parent_version = None
            if self.versions:
                parent_version = max(
                    self.versions.keys(),
                    key=lambda vid: self.versions[vid].timestamp
                )
            
            # Create version object
            version = ConfigVersion(
                version_id=version_id,
                timestamp=timestamp,
                description=description,
                author=author,
                config_hash=config_hash,
                file_path=config_file,
                config_data=deepcopy(config_data),
                parent_version=parent_version,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store version
            self.versions[version_id] = version
            
            # Save version file
            version_file = self.config_dir / f"{version_id}.json"
            with open(version_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(version.to_dict(), f, indent=2)
            
            # Update versions index
            self._save_versions()
            
            logger.info(f"Created configuration version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            raise
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """Get a specific version."""
        return self.versions.get(version_id)
    
    def get_latest_version(self, config_file: str = None) -> Optional[ConfigVersion]:
        """Get the latest version for a configuration file."""
        if not self.versions:
            return None
        
        if config_file:
            # Filter by config file
            file_versions = [
                v for v in self.versions.values()
                if v.file_path == config_file
            ]
            if not file_versions:
                return None
            return max(file_versions, key=lambda v: v.timestamp)
        else:
            # Get overall latest
            return max(self.versions.values(), key=lambda v: v.timestamp)
    
    def list_versions(
        self,
        config_file: str = None,
        limit: int = None,
        tags: List[str] = None
    ) -> List[ConfigVersion]:
        """List configuration versions."""
        versions = list(self.versions.values())
        
        # Filter by config file
        if config_file:
            versions = [v for v in versions if v.file_path == config_file]
        
        # Filter by tags
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def compare_versions(
        self,
        old_version_id: str,
        new_version_id: str
    ) -> Optional[ConfigDiff]:
        """Compare two configuration versions."""
        old_version = self.get_version(old_version_id)
        new_version = self.get_version(new_version_id)
        
        if not old_version or not new_version:
            logger.error("One or both versions not found")
            return None
        
        old_data = old_version.config_data
        new_data = new_version.config_data
        
        # Find differences
        added_keys = []
        removed_keys = []
        modified_keys = []
        
        # Check for added and modified keys
        for key in new_data:
            if key not in old_data:
                added_keys.append(key)
            elif old_data[key] != new_data[key]:
                modified_keys.append(key)
        
        # Check for removed keys
        for key in old_data:
            if key not in new_data:
                removed_keys.append(key)
        
        # Generate diff text
        diff_text = self._generate_diff_text(old_data, new_data)
        
        # Create diff object
        diff = ConfigDiff(
            old_version=old_version_id,
            new_version=new_version_id,
            added_keys=added_keys,
            removed_keys=removed_keys,
            modified_keys=modified_keys,
            diff_text=diff_text
        )
        
        return diff
    
    def _generate_diff_text(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> str:
        """Generate human-readable diff text."""
        old_str = json.dumps(old_data, indent=2, sort_keys=True)
        new_str = json.dumps(new_data, indent=2, sort_keys=True)
        
        old_lines = old_str.splitlines()
        new_lines = new_str.splitlines()
        
        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='old_config',
            tofile='new_config',
            lineterm=''
        )
        
        return '\n'.join(diff_lines)
    
    def restore_version(self, version_id: str, target_file: str = None) -> bool:
        """Restore a configuration version."""
        version = self.get_version(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        try:
            # Use original file path if target not specified
            if target_file is None:
                target_file = version.file_path
            
            # Save configuration to target file
            self._save_config_file(target_file, version.config_data)
            
            logger.info(f"Restored version {version_id} to {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore version: {e}")
            return False
    
    def delete_version(self, version_id: str) -> bool:
        """Delete a configuration version."""
        if version_id not in self.versions:
            logger.error(f"Version not found: {version_id}")
            return False
        
        try:
            # Remove version file
            version_file = self.config_dir / f"{version_id}.json"
            if version_file.exists():
                version_file.unlink()
            
            # Remove from versions dict
            del self.versions[version_id]
            
            # Update versions index
            self._save_versions()
            
            logger.info(f"Deleted version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def get_version_history(self, config_file: str = None) -> List[Tuple[str, str, str]]:
        """Get version history as list of (version_id, timestamp, description)."""
        versions = self.list_versions(config_file)
        return [(v.version_id, v.timestamp, v.description) for v in versions]
    
    def search_versions(
        self,
        query: str,
        config_file: str = None
    ) -> List[ConfigVersion]:
        """Search versions by description or tags."""
        versions = self.list_versions(config_file)
        
        results = []
        query_lower = query.lower()
        
        for version in versions:
            # Search in description
            if query_lower in version.description.lower():
                results.append(version)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in version.tags):
                results.append(version)
                continue
            
            # Search in metadata
            if any(query_lower in str(value).lower() for value in version.metadata.values()):
                results.append(version)
        
        return results
    
    def export_version(self, version_id: str, export_path: str) -> bool:
        """Export a version to a file."""
        version = self.get_version(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(version.to_dict(), f, indent=2)
            
            logger.info(f"Exported version {version_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export version: {e}")
            return False
    
    def import_version(self, import_path: str) -> str:
        """Import a version from a file."""
        try:
            import_path = Path(import_path)
            
            with open(import_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                version_data = json.load(f)
            
            version = ConfigVersion.from_dict(version_data)
            
            # Generate new version ID to avoid conflicts
            timestamp = datetime.now().isoformat()
            version.version_id = self._generate_version_id(version.config_data, timestamp)
            version.timestamp = timestamp
            
            # Store version
            self.versions[version.version_id] = version
            
            # Save version file
            version_file = self.config_dir / f"{version.version_id}.json"
            with open(version_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(version.to_dict(), f, indent=2)
            
            # Update versions index
            self._save_versions()
            
            logger.info(f"Imported version: {version.version_id}")
            return version.version_id
            
        except Exception as e:
            logger.error(f"Failed to import version: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning statistics."""
        if not self.versions:
            return {
                "total_versions": 0,
                "config_files": [],
                "authors": [],
                "tags": [],
                "date_range": None
            }
        
        # Collect statistics
        config_files = list(set(v.file_path for v in self.versions.values()))
        authors = list(set(v.author for v in self.versions.values()))
        tags = list(set(tag for v in self.versions.values() for tag in v.tags))
        
        timestamps = [v.timestamp for v in self.versions.values()]
        date_range = {
            "earliest": min(timestamps),
            "latest": max(timestamps)
        }
        
        return {
            "total_versions": len(self.versions),
            "config_files": config_files,
            "authors": authors,
            "tags": tags,
            "date_range": date_range
        }


# Convenience functions
def create_config_versioning(config_dir: str = "config_versions") -> ConfigVersioning:
    """Create configuration versioning system."""
    return ConfigVersioning(config_dir)


def version_config(
    config_file: str,
    description: str,
    author: str = "system",
    tags: List[str] = None,
    metadata: Dict[str, Any] = None
) -> str:
    """Quick function to version a configuration file."""
    versioning = create_config_versioning()
    return versioning.create_version(config_file, description, author, tags, metadata)


if __name__ == "__main__":
    # Example usage
    print("🔧 Configuration Versioning System")
    print("=" * 40)
    
    # Create versioning system
    versioning = create_config_versioning()
    
    # Example configuration
    config_data = {
        "model": {
            "name": "diffusion_v1",
            "parameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            }
        },
        "data": {
            "path": "/data/videos",
            "format": "mp4"
        }
    }
    
    # Save example config
    config_file = "example_config.json"
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(config_data, f, indent=2)
    
    # Create version
    version_id = versioning.create_version(
        config_file,
        "Initial configuration",
        author="developer",
        tags=["initial", "baseline"],
        metadata={"experiment": "test_run"}
    )
    
    print(f"Created version: {version_id}")
    
    # Modify config
    config_data["model"]["parameters"]["learning_rate"] = 0.0005
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(config_data, f, indent=2)
    
    # Create another version
    version_id2 = versioning.create_version(
        config_file,
        "Reduced learning rate",
        author="developer",
        tags=["optimization", "learning_rate"],
        metadata={"experiment": "test_run", "change": "lr_reduction"}
    )
    
    print(f"Created version: {version_id2}")
    
    # Compare versions
    diff = versioning.compare_versions(version_id, version_id2)
    if diff:
        print(f"Changes: {diff.added_keys} added, {diff.removed_keys} removed, {diff.modified_keys} modified")
    
    # List versions
    versions = versioning.list_versions()
    print(f"Total versions: {len(versions)}")
    
    # Get statistics
    stats = versioning.get_statistics()
    print(f"Statistics: {stats}")
    
    # Cleanup
    if os.path.exists(config_file):
        os.remove(config_file)
    
    print("✅ Configuration versioning example completed!") 