"""
Model Versioning and Management
"""

import torch
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)


class ModelVersion:
    """Model version information"""
    
    def __init__(
        self,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model version
        
        Args:
            version: Version string (e.g., "1.0.0")
            model_path: Path to model file
            metadata: Optional metadata
        """
        self.version = version
        self.model_path = model_path
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary"""
        version = cls(
            version=data["version"],
            model_path=data["model_path"],
            metadata=data.get("metadata", {})
        )
        version.created_at = data.get("created_at", datetime.now().isoformat())
        return version


class ModelRegistry:
    """Model registry for version management"""
    
    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize model registry
        
        Args:
            registry_dir: Registry directory
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.versions = self._load_registry()
        
        logger.info(f"ModelRegistry initialized: {registry_dir}")
    
    def _load_registry(self) -> Dict[str, ModelVersion]:
        """Load registry from file"""
        if not self.registry_file.exists():
            return {}
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            versions = {}
            for version_str, version_data in data.items():
                versions[version_str] = ModelVersion.from_dict(version_data)
            
            return versions
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save registry to file"""
        data = {
            version: version_obj.to_dict()
            for version, version_obj in self.versions.items()
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register(
        self,
        model: torch.nn.Module,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_model: bool = True
    ) -> str:
        """
        Register model version
        
        Args:
            model: Model to register
            version: Version string
            metadata: Optional metadata
            save_model: Whether to save model
        
        Returns:
            Model path
        """
        # Create version directory
        version_dir = self.registry_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pth"
        if save_model:
            torch.save(model.state_dict(), model_path)
        
        # Calculate model hash
        model_hash = self._calculate_hash(model)
        
        # Create version
        version_metadata = {
            "model_hash": model_hash,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            **(metadata or {})
        }
        
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            metadata=version_metadata
        )
        
        # Register
        self.versions[version] = model_version
        self._save_registry()
        
        logger.info(f"Model version {version} registered")
        return str(model_path)
    
    def _calculate_hash(self, model: torch.nn.Module) -> str:
        """Calculate model hash"""
        # Get model state
        state = model.state_dict()
        
        # Create hash from state
        state_str = str(sorted(state.items()))
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get model version"""
        return self.versions.get(version)
    
    def list_versions(self) -> List[str]:
        """List all versions"""
        return sorted(self.versions.keys())
    
    def load_model(self, version: str, model_class: torch.nn.Module) -> torch.nn.Module:
        """
        Load model by version
        
        Args:
            version: Version string
            model_class: Model class to instantiate
        
        Returns:
            Loaded model
        """
        model_version = self.get_version(version)
        if model_version is None:
            raise ValueError(f"Version {version} not found")
        
        model = model_class()
        model.load_state_dict(torch.load(model_version.model_path))
        
        logger.info(f"Model version {version} loaded")
        return model
    
    def get_latest(self) -> Optional[ModelVersion]:
        """Get latest version"""
        if not self.versions:
            return None
        
        versions = sorted(self.versions.keys())
        return self.versions[versions[-1]]
    
    def delete_version(self, version: str):
        """Delete model version"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Delete model file
        model_version = self.versions[version]
        if os.path.exists(model_version.model_path):
            os.remove(model_version.model_path)
        
        # Delete version directory
        version_dir = self.registry_dir / version
        if version_dir.exists():
            version_dir.rmdir()
        
        # Remove from registry
        del self.versions[version]
        self._save_registry()
        
        logger.info(f"Model version {version} deleted")


class ModelComparator:
    """Compare different model versions"""
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize model comparator
        
        Args:
            registry: Model registry
        """
        self.registry = registry
    
    def compare_versions(
        self,
        versions: List[str],
        test_loader,
        metrics: List[str] = ["loss", "accuracy"]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model versions
        
        Args:
            versions: List of version strings
            test_loader: Test data loader
            metrics: Metrics to compute
        
        Returns:
            Dictionary of version -> metrics
        """
        results = {}
        
        for version in versions:
            model_version = self.registry.get_version(version)
            if model_version is None:
                continue
            
            # Load and evaluate model
            # This is a simplified version - in practice, you'd load the model
            # and run evaluation
            
            results[version] = {
                "version": version,
                "created_at": model_version.created_at,
                "metadata": model_version.metadata
            }
        
        return results

