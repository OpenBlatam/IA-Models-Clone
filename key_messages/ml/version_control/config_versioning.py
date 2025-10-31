from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import structlog
import gzip
import shutil
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Versioning System for Key Messages ML Pipeline
Handles configuration snapshots, diffs, and history management
"""


logger = structlog.get_logger(__name__)

@dataclass
class ConfigSnapshot:
    """Represents a configuration snapshot."""
    version: str
    config: Dict[str, Any]
    description: str
    timestamp: float
    author: str
    tags: List[str]
    hash: str
    file_path: str
    
    def __post_init__(self) -> Any:
        if not self.version or not self.config:
            raise ValueError("Version and config are required")

@dataclass
class ConfigChange:
    """Represents a single configuration change."""
    path: str
    old_value: Any
    new_value: Any
    change_type: str  # "added", "removed", "modified"
    
    def __post_init__(self) -> Any:
        if not self.path:
            raise ValueError("Path is required")

@dataclass
class ConfigDiff:
    """Represents differences between two configurations."""
    old_version: str
    new_version: str
    changes: List[ConfigChange]
    summary: str
    
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return len(self.changes) > 0
    
    def get_change_count(self) -> int:
        """Get the number of changes."""
        return len(self.changes)
    
    def get_breaking_changes(self) -> List[ConfigChange]:
        """Get changes that might be breaking."""
        # Define breaking change patterns
        breaking_patterns = [
            "models.",  # Model architecture changes
            "training.batch_size",  # Batch size changes
            "training.learning_rate",  # Learning rate changes
            "data_loading.",  # Data loading changes
            "evaluation.",  # Evaluation changes
        ]
        
        breaking_changes = []
        for change in self.changes:
            for pattern in breaking_patterns:
                if pattern in change.path:
                    breaking_changes.append(change)
                    break
        
        return breaking_changes

@dataclass
class ConfigHistory:
    """Represents configuration history."""
    snapshots: List[ConfigSnapshot]
    total_versions: int
    latest_version: Optional[str]
    first_version: Optional[str]
    
    def get_versions(self) -> List[str]:
        """Get list of all versions."""
        return [snapshot.version for snapshot in self.snapshots]
    
    def get_snapshot(self, version: str) -> Optional[ConfigSnapshot]:
        """Get snapshot by version."""
        for snapshot in self.snapshots:
            if snapshot.version == version:
                return snapshot
        return None

class ConfigVersionManager:
    """Manages configuration versioning and history."""
    
    def __init__(self, config_dir: str = "./config_versions", 
                 auto_snapshot: bool = True, max_history: int = 50,
                 compression: bool = True):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.auto_snapshot = auto_snapshot
        self.max_history = max_history
        self.compression = compression
        
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self.metadata_file = self.config_dir / "metadata.json"
        self._load_metadata()
        
        logger.info("ConfigVersionManager initialized", 
                   config_dir=str(self.config_dir),
                   auto_snapshot=auto_snapshot,
                   max_history=max_history)
    
    def _load_metadata(self) -> Any:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error("Failed to load metadata", error=str(e))
                self.metadata = {"versions": [], "latest": None}
        else:
            self.metadata = {"versions": [], "latest": None}
            self._save_metadata()
    
    def _save_metadata(self) -> Any:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error("Failed to save metadata", error=str(e))
    
    def _generate_version(self, config: Dict[str, Any]) -> str:
        """Generate a version string based on config content."""
        # Create hash of config content
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Add timestamp
        timestamp = int(time.time())
        
        return f"v{timestamp}_{config_hash}"
    
    def _compress_file(self, file_path: str) -> str:
        """Compress a file using gzip."""
        compressed_path = f"{file_path}.gz"
        
        with open(file_path, 'rb') as f_in:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            with gzip.open(compressed_path, 'wb') as f_out:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original file
        os.remove(file_path)
        
        return compressed_path
    
    def _decompress_file(self, file_path: str) -> str:
        """Decompress a gzip file."""
        if file_path.endswith('.gz'):
            decompressed_path = file_path[:-3]
            
            with gzip.open(file_path, 'rb') as f_in:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                with open(decompressed_path, 'wb') as f_out:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    shutil.copyfileobj(f_in, f_out)
            
            return decompressed_path
        
        return file_path
    
    def create_snapshot(self, config: Dict[str, Any], description: str = "",
                       author: str = "ML Pipeline", tags: List[str] = None,
                       version: Optional[str] = None) -> ConfigSnapshot:
        """Create a configuration snapshot."""
        try:
            # Generate version if not provided
            if not version:
                version = self._generate_version(config)
            
            # Create snapshot
            snapshot = ConfigSnapshot(
                version=version,
                config=config,
                description=description,
                timestamp=time.time(),
                author=author,
                tags=tags or [],
                hash=hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                file_path=""
            )
            
            # Save snapshot to file
            file_path = self.config_dir / f"config_{version}.json"
            
            with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump({
                    "version": snapshot.version,
                    "config": snapshot.config,
                    "description": snapshot.description,
                    "timestamp": snapshot.timestamp,
                    "author": snapshot.author,
                    "tags": snapshot.tags,
                    "hash": snapshot.hash
                }, f, indent=2)
            
            # Compress if enabled
            if self.compression:
                compressed_path = self._compress_file(str(file_path))
                snapshot.file_path = compressed_path
            else:
                snapshot.file_path = str(file_path)
            
            # Update metadata
            self.metadata["versions"].append({
                "version": version,
                "timestamp": snapshot.timestamp,
                "description": description,
                "author": author,
                "tags": tags or [],
                "file_path": snapshot.file_path
            })
            
            self.metadata["latest"] = version
            
            # Sort versions by timestamp
            self.metadata["versions"].sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Limit history
            if len(self.metadata["versions"]) > self.max_history:
                # Remove old versions
                old_versions = self.metadata["versions"][self.max_history:]
                for old_version in old_versions:
                    self._remove_version_file(old_version["file_path"])
                
                self.metadata["versions"] = self.metadata["versions"][:self.max_history]
            
            self._save_metadata()
            
            logger.info("Configuration snapshot created", 
                       version=version,
                       description=description,
                       author=author)
            
            return snapshot
            
        except Exception as e:
            logger.error("Failed to create configuration snapshot", error=str(e))
            raise
    
    def _remove_version_file(self, file_path: str):
        """Remove a version file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning("Failed to remove version file", file_path=file_path, error=str(e))
    
    def load_snapshot(self, version: str) -> Optional[ConfigSnapshot]:
        """Load a configuration snapshot by version."""
        try:
            # Find version in metadata
            version_info = None
            for v_info in self.metadata["versions"]:
                if v_info["version"] == version:
                    version_info = v_info
                    break
            
            if not version_info:
                logger.warning("Version not found", version=version)
                return None
            
            # Load config file
            file_path = version_info["file_path"]
            
            # Decompress if needed
            if file_path.endswith('.gz'):
                file_path = self._decompress_file(file_path)
            
            with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = json.load(f)
            
            # Create snapshot object
            snapshot = ConfigSnapshot(
                version=data["version"],
                config=data["config"],
                description=data["description"],
                timestamp=data["timestamp"],
                author=data["author"],
                tags=data["tags"],
                hash=data["hash"],
                file_path=version_info["file_path"]
            )
            
            logger.info("Configuration snapshot loaded", version=version)
            return snapshot
            
        except Exception as e:
            logger.error("Failed to load configuration snapshot", version=version, error=str(e))
            return None
    
    def get_history(self, limit: Optional[int] = None) -> ConfigHistory:
        """Get configuration history."""
        try:
            versions = self.metadata["versions"]
            
            if limit:
                versions = versions[:limit]
            
            snapshots = []
            for version_info in versions:
                snapshot = self.load_snapshot(version_info["version"])
                if snapshot:
                    snapshots.append(snapshot)
            
            return ConfigHistory(
                snapshots=snapshots,
                total_versions=len(self.metadata["versions"]),
                latest_version=self.metadata.get("latest"),
                first_version=versions[-1]["version"] if versions else None
            )
            
        except Exception as e:
            logger.error("Failed to get configuration history", error=str(e))
            return ConfigHistory(snapshots=[], total_versions=0, latest_version=None, first_version=None)
    
    def compare_versions(self, version1: str, version2: str) -> Optional[ConfigDiff]:
        """Compare two configuration versions."""
        try:
            snapshot1 = self.load_snapshot(version1)
            snapshot2 = self.load_snapshot(version2)
            
            if not snapshot1 or not snapshot2:
                logger.error("One or both versions not found", version1=version1, version2=version2)
                return None
            
            changes = self._compute_diff(snapshot1.config, snapshot2.config)
            
            # Generate summary
            summary = f"Configuration changes from {version1} to {version2}: {len(changes)} changes"
            
            diff = ConfigDiff(
                old_version=version1,
                new_version=version2,
                changes=changes,
                summary=summary
            )
            
            logger.info("Configuration versions compared", 
                       version1=version1,
                       version2=version2,
                       changes=len(changes))
            
            return diff
            
        except Exception as e:
            logger.error("Failed to compare configuration versions", 
                        version1=version1,
                        version2=version2,
                        error=str(e))
            return None
    
    def _compute_diff(self, old_config: Dict[str, Any], new_config: Dict[str, Any], 
                     path: str = "") -> List[ConfigChange]:
        """Compute differences between two configurations."""
        changes = []
        
        # Get all keys from both configs
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in old_config:
                # Key was added
                changes.append(ConfigChange(
                    path=current_path,
                    old_value=None,
                    new_value=new_config[key],
                    change_type="added"
                ))
            elif key not in new_config:
                # Key was removed
                changes.append(ConfigChange(
                    path=current_path,
                    old_value=old_config[key],
                    new_value=None,
                    change_type="removed"
                ))
            elif isinstance(old_config[key], dict) and isinstance(new_config[key], dict):
                # Recursively compare nested dictionaries
                nested_changes = self._compute_diff(old_config[key], new_config[key], current_path)
                changes.extend(nested_changes)
            elif old_config[key] != new_config[key]:
                # Value was modified
                changes.append(ConfigChange(
                    path=current_path,
                    old_value=old_config[key],
                    new_value=new_config[key],
                    change_type="modified"
                ))
        
        return changes
    
    def restore_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Restore configuration to a specific version."""
        try:
            snapshot = self.load_snapshot(version)
            
            if not snapshot:
                logger.error("Version not found for restoration", version=version)
                return None
            
            logger.info("Configuration restored", version=version)
            return snapshot.config
            
        except Exception as e:
            logger.error("Failed to restore configuration version", version=version, error=str(e))
            return None
    
    def delete_version(self, version: str) -> bool:
        """Delete a configuration version."""
        try:
            # Find version in metadata
            version_info = None
            for i, v_info in enumerate(self.metadata["versions"]):
                if v_info["version"] == version:
                    version_info = v_info
                    del self.metadata["versions"][i]
                    break
            
            if not version_info:
                logger.warning("Version not found for deletion", version=version)
                return False
            
            # Remove file
            self._remove_version_file(version_info["file_path"])
            
            # Update latest version if needed
            if self.metadata["latest"] == version:
                if self.metadata["versions"]:
                    self.metadata["latest"] = self.metadata["versions"][0]["version"]
                else:
                    self.metadata["latest"] = None
            
            self._save_metadata()
            
            logger.info("Configuration version deleted", version=version)
            return True
            
        except Exception as e:
            logger.error("Failed to delete configuration version", version=version, error=str(e))
            return False
    
    def search_versions(self, query: str) -> List[ConfigSnapshot]:
        """Search for versions by description, author, or tags."""
        try:
            results = []
            
            for version_info in self.metadata["versions"]:
                # Search in description
                if query.lower() in version_info["description"].lower():
                    snapshot = self.load_snapshot(version_info["version"])
                    if snapshot:
                        results.append(snapshot)
                        continue
                
                # Search in author
                if query.lower() in version_info["author"].lower():
                    snapshot = self.load_snapshot(version_info["version"])
                    if snapshot:
                        results.append(snapshot)
                        continue
                
                # Search in tags
                for tag in version_info["tags"]:
                    if query.lower() in tag.lower():
                        snapshot = self.load_snapshot(version_info["version"])
                        if snapshot:
                            results.append(snapshot)
                            break
            
            logger.info("Configuration versions searched", query=query, results=len(results))
            return results
            
        except Exception as e:
            logger.error("Failed to search configuration versions", query=query, error=str(e))
            return []
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version."""
        try:
            for version_info in self.metadata["versions"]:
                if version_info["version"] == version:
                    return version_info
            
            return None
            
        except Exception as e:
            logger.error("Failed to get version info", version=version, error=str(e))
            return None
    
    def export_version(self, version: str, export_path: str) -> bool:
        """Export a configuration version to a file."""
        try:
            snapshot = self.load_snapshot(version)
            
            if not snapshot:
                logger.error("Version not found for export", version=version)
                return False
            
            # Create export data
            export_data = {
                "version": snapshot.version,
                "config": snapshot.config,
                "description": snapshot.description,
                "timestamp": snapshot.timestamp,
                "author": snapshot.author,
                "tags": snapshot.tags,
                "hash": snapshot.hash,
                "export_timestamp": time.time()
            }
            
            # Save to export path
            with open(export_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(export_data, f, indent=2)
            
            logger.info("Configuration version exported", version=version, export_path=export_path)
            return True
            
        except Exception as e:
            logger.error("Failed to export configuration version", 
                        version=version,
                        export_path=export_path,
                        error=str(e))
            return False
    
    def import_version(self, import_path: str) -> Optional[ConfigSnapshot]:
        """Import a configuration version from a file."""
        try:
            with open(import_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                import_data = json.load(f)
            
            # Create snapshot
            snapshot = ConfigSnapshot(
                version=import_data["version"],
                config=import_data["config"],
                description=import_data["description"],
                timestamp=import_data["timestamp"],
                author=import_data["author"],
                tags=import_data["tags"],
                hash=import_data["hash"],
                file_path=""
            )
            
            # Save snapshot
            return self.create_snapshot(
                config=snapshot.config,
                description=snapshot.description,
                author=snapshot.author,
                tags=snapshot.tags,
                version=snapshot.version
            )
            
        except Exception as e:
            logger.error("Failed to import configuration version", 
                        import_path=import_path,
                        error=str(e))
            return None
    
    def cleanup_old_versions(self, keep_count: int = None) -> int:
        """Clean up old configuration versions."""
        try:
            if keep_count is None:
                keep_count = self.max_history
            
            if len(self.metadata["versions"]) <= keep_count:
                return 0
            
            # Remove old versions
            old_versions = self.metadata["versions"][keep_count:]
            removed_count = 0
            
            for version_info in old_versions:
                if self.delete_version(version_info["version"]):
                    removed_count += 1
            
            logger.info("Old configuration versions cleaned up", 
                       removed_count=removed_count,
                       kept_count=keep_count)
            
            return removed_count
            
        except Exception as e:
            logger.error("Failed to cleanup old versions", error=str(e))
            return 0 