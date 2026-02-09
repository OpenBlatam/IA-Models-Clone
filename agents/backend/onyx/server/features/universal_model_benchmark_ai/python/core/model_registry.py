"""
Model Registry Module - Model versioning and management.

Provides:
- Model registration
- Version control
- Model metadata
- Model discovery
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model status."""
    DRAFT = "draft"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Model metadata."""
    name: str
    version: str
    description: str = ""
    architecture: str = ""
    parameters: int = 0
    size_gb: float = 0.0
    quantization: str = ""
    framework: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelVersion:
    """Model version information."""
    model_name: str
    version: str
    status: ModelStatus
    metadata: ModelMetadata
    path: str
    checksum: Optional[str] = None
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "path": self.path,
            "checksum": self.checksum,
            "benchmark_results": self.benchmark_results,
        }


class ModelRegistry:
    """Model registry for versioning and management."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            storage_path: Path to registry storage
        """
        self.storage_path = Path(storage_path) if storage_path else Path("model_registry")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # {model_name: {version: ModelVersion}}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from storage."""
        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for model_name, versions in data.items():
                        self.models[model_name] = {}
                        for version_str, version_data in versions.items():
                            version = ModelVersion(
                                model_name=version_data["model_name"],
                                version=version_data["version"],
                                status=ModelStatus(version_data["status"]),
                                metadata=ModelMetadata(**version_data["metadata"]),
                                path=version_data["path"],
                                checksum=version_data.get("checksum"),
                                benchmark_results=version_data.get("benchmark_results", {}),
                            )
                            self.models[model_name][version_str] = version
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to storage."""
        registry_file = self.storage_path / "registry.json"
        data = {
            model_name: {
                version: v.to_dict()
                for version, v in versions.items()
            }
            for model_name, versions in self.models.items()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        metadata: ModelMetadata,
        path: str,
        checksum: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            metadata: Model metadata
            path: Path to model files
            checksum: Optional checksum
            
        Returns:
            Registered model version
        """
        model_name = metadata.name
        
        if model_name not in self.models:
            self.models[model_name] = {}
        
        version = ModelVersion(
            model_name=model_name,
            version=metadata.version,
            status=ModelStatus.DRAFT,
            metadata=metadata,
            path=path,
            checksum=checksum,
        )
        
        self.models[model_name][metadata.version] = version
        self._save_registry()
        
        logger.info(f"Registered model: {model_name} v{metadata.version}")
        return version
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get model version.
        
        Args:
            model_name: Model name
            version: Optional version (defaults to latest)
            
        Returns:
            Model version or None
        """
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        
        if not version:
            # Get latest version
            if not versions:
                return None
            version = max(versions.keys(), key=lambda v: versions[v].metadata.created_at)
        
        return versions.get(version)
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        """
        List all model versions.
        
        Args:
            status: Filter by status
            tags: Filter by tags
            
        Returns:
            List of model versions
        """
        all_versions = []
        for versions in self.models.values():
            all_versions.extend(versions.values())
        
        if status:
            all_versions = [v for v in all_versions if v.status == status]
        
        if tags:
            all_versions = [
                v for v in all_versions
                if any(tag in v.metadata.tags for tag in tags)
            ]
        
        return sorted(
            all_versions,
            key=lambda v: v.metadata.created_at,
            reverse=True
        )
    
    def update_status(
        self,
        model_name: str,
        version: str,
        status: ModelStatus,
    ) -> ModelVersion:
        """
        Update model status.
        
        Args:
            model_name: Model name
            version: Version
            status: New status
            
        Returns:
            Updated model version
        """
        if model_name not in self.models or version not in self.models[model_name]:
            raise ValueError(f"Model {model_name} v{version} not found")
        
        version_obj = self.models[model_name][version]
        version_obj.status = status
        version_obj.metadata.updated_at = datetime.now().isoformat()
        
        self._save_registry()
        
        logger.info(f"Updated {model_name} v{version} status to {status.value}")
        return version_obj
    
    def add_benchmark_results(
        self,
        model_name: str,
        version: str,
        benchmark_name: str,
        results: Dict[str, Any],
    ) -> ModelVersion:
        """
        Add benchmark results to model.
        
        Args:
            model_name: Model name
            version: Version
            benchmark_name: Benchmark name
            results: Benchmark results
            
        Returns:
            Updated model version
        """
        if model_name not in self.models or version not in self.models[model_name]:
            raise ValueError(f"Model {model_name} v{version} not found")
        
        version_obj = self.models[model_name][version]
        version_obj.benchmark_results[benchmark_name] = results
        version_obj.metadata.updated_at = datetime.now().isoformat()
        
        self._save_registry()
        
        return version_obj
    
    def get_best_models(
        self,
        benchmark_name: str,
        top_k: int = 5,
    ) -> List[ModelVersion]:
        """
        Get best models for a benchmark.
        
        Args:
            benchmark_name: Benchmark name
            top_k: Number of top models
            
        Returns:
            List of top model versions
        """
        all_versions = []
        for versions in self.models.values():
            for version_obj in versions.values():
                if benchmark_name in version_obj.benchmark_results:
                    all_versions.append(version_obj)
        
        # Sort by accuracy
        all_versions.sort(
            key=lambda v: v.benchmark_results.get(benchmark_name, {}).get("accuracy", 0.0),
            reverse=True
        )
        
        return all_versions[:top_k]












