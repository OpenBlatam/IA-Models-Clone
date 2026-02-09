"""
Model Versioning and Management

Provides utilities for:
- Model versioning
- Model registry
- Model comparison
- Model deployment tracking
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import hashlib
import torch

logger = logging.getLogger(__name__)


class ModelVersion:
    """
    Represents a versioned model
    """
    
    def __init__(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model version
        
        Args:
            model_name: Name of the model
            version: Version string (e.g., "1.0.0")
            model_path: Path to model file
            metadata: Additional metadata
        """
        self.model_name = model_name
        self.version = version
        self.model_path = Path(model_path)
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        self._compute_hash()
    
    def _compute_hash(self) -> None:
        """Compute hash of model file"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                self.metadata["file_hash"] = file_hash
        except Exception as e:
            logger.warning(f"Could not compute model hash: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "model_path": str(self.model_path),
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        instance = cls(
            model_name=data["model_name"],
            version=data["version"],
            model_path=data["model_path"],
            metadata=data.get("metadata", {})
        )
        instance.created_at = data.get("created_at", datetime.utcnow().isoformat())
        return instance


class ModelRegistry:
    """
    Registry for managing model versions
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.versions: Dict[str, List[ModelVersion]] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for model_name, versions in data.items():
                        self.versions[model_name] = [
                            ModelVersion.from_dict(v) for v in versions
                        ]
                logger.info(f"Loaded registry with {len(self.versions)} models")
            except Exception as e:
                logger.error(f"Error loading registry: {e}", exc_info=True)
                self.versions = {}
    
    def _save_registry(self) -> None:
        """Save registry to file"""
        try:
            data = {
                model_name: [v.to_dict() for v in versions]
                for model_name, versions in self.versions.items()
            }
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}", exc_info=True)
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model
            version: Version string
            model_path: Path to model file
            metadata: Additional metadata
            
        Returns:
            ModelVersion object
        """
        model_version = ModelVersion(model_name, version, model_path, metadata)
        
        if model_name not in self.versions:
            self.versions[model_name] = []
        
        self.versions[model_name].append(model_version)
        self._save_registry()
        
        logger.info(f"Registered model: {model_name} v{version}")
        return model_version
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get latest version of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest ModelVersion or None
        """
        if model_name not in self.versions or not self.versions[model_name]:
            return None
        
        # Sort by created_at and return latest
        versions = sorted(
            self.versions[model_name],
            key=lambda v: v.created_at,
            reverse=True
        )
        return versions[0]
    
    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """
        Get specific version of a model
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            ModelVersion or None
        """
        if model_name not in self.versions:
            return None
        
        for model_version in self.versions[model_name]:
            if model_version.version == version:
                return model_version
        
        return None
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self.versions.keys())
    
    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """
        List all versions of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of ModelVersion objects
        """
        return self.versions.get(model_name, [])
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a model version
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            True if deleted, False if not found
        """
        if model_name not in self.versions:
            return False
        
        original_count = len(self.versions[model_name])
        self.versions[model_name] = [
            v for v in self.versions[model_name]
            if v.version != version
        ]
        
        if len(self.versions[model_name]) < original_count:
            self._save_registry()
            logger.info(f"Deleted model version: {model_name} v{version}")
            return True
        
        return False


def compare_model_versions(
    version1: ModelVersion,
    version2: ModelVersion,
    test_dataloader: Optional[torch.utils.data.DataLoader] = None
) -> Dict[str, Any]:
    """
    Compare two model versions
    
    Args:
        version1: First model version
        version2: Second model version
        test_dataloader: Optional test data for evaluation
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "version1": version1.to_dict(),
        "version2": version2.to_dict(),
        "metadata_diff": {}
    }
    
    # Compare metadata
    for key in set(version1.metadata.keys()) | set(version2.metadata.keys()):
        val1 = version1.metadata.get(key)
        val2 = version2.metadata.get(key)
        if val1 != val2:
            comparison["metadata_diff"][key] = {
                "version1": val1,
                "version2": val2
            }
    
    # If test data provided, evaluate both
    if test_dataloader:
        from .evaluation_utils import ModelEvaluator
        
        # Load models
        model1 = torch.load(version1.model_path)
        model2 = torch.load(version2.model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        evaluator1 = ModelEvaluator(model1, device)
        evaluator2 = ModelEvaluator(model2, device)
        
        results1 = evaluator1.evaluate(test_dataloader)
        results2 = evaluator2.evaluate(test_dataloader)
        
        comparison["evaluation"] = {
            "version1": results1,
            "version2": results2,
            "difference": {
                "accuracy": results2.get("metrics", {}).get("accuracy", 0) - 
                           results1.get("metrics", {}).get("accuracy", 0),
                "loss": results2.get("loss", 0) - results1.get("loss", 0)
            }
        }
    
    return comparison















