"""
Model Versioning

Utilities for versioning models.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelVersioner:
    """Version models and track versions."""
    
    def __init__(self, version_dir: str = "./versions"):
        """
        Initialize model versioner.
        
        Args:
            version_dir: Directory for versions
        """
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.version_index = self.version_dir / "versions.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load version index."""
        if self.version_index.exists():
            with open(self.version_index, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_index(self) -> None:
        """Save version index."""
        with open(self.version_index, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def version(
        self,
        model: nn.Module,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Version a model.
        
        Args:
            model: Model to version
            model_name: Model name
            version: Version string (auto-generated if None)
            metadata: Optional metadata
            
        Returns:
            Version string
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_path = self.version_dir / model_name / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_path / "model.pt"
        torch.save({
            'state_dict': model.state_dict(),
            'metadata': metadata or {}
        }, model_path)
        
        # Update index
        if model_name not in self.versions:
            self.versions[model_name] = []
        
        version_info = {
            'version': version,
            'path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.versions[model_name].append(version_info)
        self._save_index()
        
        logger.info(f"Versioned model: {model_name} v{version}")
        
        return version
    
    def get_version(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get model version.
        
        Args:
            model_name: Model name
            version: Version string (latest if None)
            
        Returns:
            Version information or None
        """
        if model_name not in self.versions:
            return None
        
        versions = self.versions[model_name]
        
        if version is None:
            # Get latest
            return versions[-1] if versions else None
        
        # Get specific version
        for v in versions:
            if v['version'] == version:
                return v
        
        return None
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of versions
        """
        return self.versions.get(model_name, [])


def version_model(
    model: nn.Module,
    model_name: str,
    **kwargs
) -> str:
    """Version model."""
    versioner = ModelVersioner()
    return versioner.version(model, model_name, **kwargs)


def get_model_version(
    model_name: str,
    version: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get model version."""
    versioner = ModelVersioner()
    return versioner.get_version(model_name, version)


def list_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """List model versions."""
    versioner = ModelVersioner()
    return versioner.list_versions(model_name)



