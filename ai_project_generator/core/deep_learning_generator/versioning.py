"""
Versioning Module

Configuration versioning and migration utilities.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class ConfigVersion:
    """
    Represents a versioned configuration.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.config = config.copy()
        self.version = version or self._generate_version()
        self.metadata = metadata or {}
        self.metadata.setdefault("created_at", datetime.now().isoformat())
        self.metadata.setdefault("hash", self._generate_hash())
    
    def _generate_version(self) -> str:
        """Generate version from timestamp."""
        return datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _generate_hash(self) -> str:
        """Generate hash of configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "config": self.config,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigVersion":
        """Create from dictionary."""
        return cls(
            config=data["config"],
            version=data.get("version"),
            metadata=data.get("metadata", {})
        )


class ConfigVersionManager:
    """
    Manages configuration versions.
    """
    
    def __init__(self):
        self.versions: Dict[str, List[ConfigVersion]] = {}
    
    def save_version(
        self,
        name: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConfigVersion:
        """
        Save a configuration version.
        
        Args:
            name: Configuration name
            config: Configuration dictionary
            metadata: Additional metadata
            
        Returns:
            ConfigVersion instance
        """
        version = ConfigVersion(config, metadata=metadata)
        
        if name not in self.versions:
            self.versions[name] = []
        
        self.versions[name].append(version)
        logger.info(f"Saved version {version.version} for {name}")
        
        return version
    
    def get_versions(self, name: str) -> List[ConfigVersion]:
        """
        Get all versions for a configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            List of versions
        """
        return self.versions.get(name, []).copy()
    
    def get_latest(self, name: str) -> Optional[ConfigVersion]:
        """
        Get latest version of a configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Latest ConfigVersion or None
        """
        versions = self.get_versions(name)
        if not versions:
            return None
        return max(versions, key=lambda v: v.version)
    
    def get_by_version(self, name: str, version: str) -> Optional[ConfigVersion]:
        """
        Get specific version.
        
        Args:
            name: Configuration name
            version: Version string
            
        Returns:
            ConfigVersion or None
        """
        versions = self.get_versions(name)
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two versions.
        
        Args:
            name: Configuration name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison result or None
        """
        v1 = self.get_by_version(name, version1)
        v2 = self.get_by_version(name, version2)
        
        if not v1 or not v2:
            return None
        
        from .comparator import ConfigComparator
        return ConfigComparator.compare(v1.config, v2.config)
    
    def list_configs(self) -> List[str]:
        """List all configuration names."""
        return list(self.versions.keys())


# Global version manager
_version_manager = ConfigVersionManager()


def get_version_manager() -> ConfigVersionManager:
    """Get the global version manager."""
    return _version_manager


def save_config_version(
    name: str,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> ConfigVersion:
    """Save a configuration version."""
    return _version_manager.save_version(name, config, metadata)


def get_config_versions(name: str) -> List[ConfigVersion]:
    """Get all versions for a configuration."""
    return _version_manager.get_versions(name)


def get_latest_config_version(name: str) -> Optional[ConfigVersion]:
    """Get latest version of a configuration."""
    return _version_manager.get_latest(name)















