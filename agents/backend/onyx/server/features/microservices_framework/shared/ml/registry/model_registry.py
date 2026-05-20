"""
Model Registry
Centralized model versioning and management.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelVersion:
    """Model version information."""
    
    def __init__(
        self,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ):
        self.version = version
        self.model_path = Path(model_path)
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute checksum for model files."""
        if not self.model_path.exists():
            return ""
        
        sha256 = hashlib.sha256()
        
        if self.model_path.is_file():
            with open(self.model_path, "rb") as f:
                sha256.update(f.read())
        elif self.model_path.is_dir():
            # Hash all files in directory
            for file_path in sorted(self.model_path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        sha256.update(f.read())
        
        return sha256.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_path": str(self.model_path),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            model_path=data["model_path"],
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
        )


class ModelRegistry:
    """
    Centralized model registry for versioning and management.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.registry: Dict[str, List[ModelVersion]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    self.registry = {
                        model_name: [ModelVersion.from_dict(v) for v in versions]
                        for model_name, versions in data.items()
                    }
                logger.info(f"Loaded registry with {len(self.registry)} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.registry = {}
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                model_name: [v.to_dict() for v in versions]
                for model_name, versions in self.registry.items()
            }
            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Registry saved")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Register a model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to model files
            version: Version string (auto-generated if None)
            metadata: Additional metadata
            
        Returns:
            ModelVersion instance
        """
        if model_name not in self.registry:
            self.registry[model_name] = []
        
        # Generate version if not provided
        if version is None:
            version = f"v{len(self.registry[model_name]) + 1}"
        
        # Check if version already exists
        existing_versions = [v.version for v in self.registry[model_name]]
        if version in existing_versions:
            raise ValueError(f"Version {version} already exists for model {model_name}")
        
        # Create version
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            metadata=metadata or {},
        )
        
        # Add to registry
        self.registry[model_name].append(model_version)
        self.registry[model_name].sort(key=lambda v: v.created_at, reverse=True)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered {model_name} version {version}")
        return model_version
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """
        Get model version.
        
        Args:
            model_name: Name of the model
            version: Version string (latest if None)
            
        Returns:
            ModelVersion or None
        """
        if model_name not in self.registry:
            return None
        
        versions = self.registry[model_name]
        
        if not versions:
            return None
        
        if version is None:
            return versions[0]  # Latest
        
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.registry.keys())
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        if model_name not in self.registry:
            return []
        return [v.version for v in self.registry[model_name]]
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if model_name not in self.registry:
            return False
        
        versions = self.registry[model_name]
        original_count = len(versions)
        
        self.registry[model_name] = [v for v in versions if v.version != version]
        
        if len(self.registry[model_name]) < original_count:
            self._save_registry()
            logger.info(f"Deleted {model_name} version {version}")
            return True
        
        return False
    
    def get_metadata(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        model_version = self.get_model(model_name, version)
        if model_version:
            return model_version.metadata
        return None
    
    def update_metadata(
        self,
        model_name: str,
        version: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Update model metadata."""
        model_version = self.get_model(model_name, version)
        if model_version:
            model_version.metadata.update(metadata)
            self._save_registry()
            return True
        return False



