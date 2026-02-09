"""
Model Versioning System for Flux2 Clothing Changer
===================================================

Version control and management for model checkpoints.
"""

import json
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    timestamp: float
    model_path: Path
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelVersioning:
    """Model versioning and checkpoint management."""
    
    def __init__(
        self,
        versions_dir: Path = Path("models/versions"),
        max_versions: int = 50,
    ):
        """
        Initialize model versioning system.
        
        Args:
            versions_dir: Directory for storing versions
            max_versions: Maximum number of versions to keep
        """
        self.versions_dir = versions_dir
        self.max_versions = max_versions
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None
        
        # Load existing versions
        self._load_versions()
    
    def create_version(
        self,
        model_path: Path,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> str:
        """
        Create a new model version.
        
        Args:
            model_path: Path to model file
            config: Model configuration
            metrics: Model metrics
            description: Version description
            tags: Optional tags
            version: Optional version string (auto-generated if None)
            
        Returns:
            Version string
        """
        if version is None:
            version = self._generate_version()
        
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_filename = model_path.name
        versioned_model_path = version_dir / model_filename
        shutil.copy2(model_path, versioned_model_path)
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().timestamp(),
            "model_path": str(versioned_model_path),
            "config": config,
            "metrics": metrics,
            "description": description,
            "tags": tags or [],
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Create version object
        model_version = ModelVersion(
            version=version,
            timestamp=metadata["timestamp"],
            model_path=versioned_model_path,
            config=config,
            metrics=metrics,
            description=description,
            tags=tags or [],
        )
        
        self.versions[version] = model_version
        self.current_version = version
        
        # Cleanup old versions
        self._cleanup_old_versions()
        
        logger.info(f"Created model version: {version}")
        return version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get version information."""
        return self.versions.get(version)
    
    def list_versions(
        self,
        tags: Optional[List[str]] = None,
        sort_by: str = "timestamp",
        reverse: bool = True,
    ) -> List[ModelVersion]:
        """
        List all versions.
        
        Args:
            tags: Filter by tags
            sort_by: Sort by field (timestamp, version, metrics)
            reverse: Reverse sort order
            
        Returns:
            List of versions
        """
        versions = list(self.versions.values())
        
        # Filter by tags
        if tags:
            versions = [
                v for v in versions
                if any(tag in v.tags for tag in tags)
            ]
        
        # Sort
        if sort_by == "timestamp":
            versions.sort(key=lambda v: v.timestamp, reverse=reverse)
        elif sort_by == "version":
            versions.sort(key=lambda v: v.version, reverse=reverse)
        elif sort_by == "metrics":
            # Sort by a metric (e.g., quality_score)
            versions.sort(
                key=lambda v: v.metrics.get("quality_score", 0.0),
                reverse=reverse
            )
        
        return versions
    
    def restore_version(self, version: str, target_path: Path) -> bool:
        """
        Restore a version to target path.
        
        Args:
            version: Version to restore
            target_path: Target path for restored model
            
        Returns:
            True if successful
        """
        if version not in self.versions:
            logger.error(f"Version {version} not found")
            return False
        
        model_version = self.versions[version]
        
        try:
            shutil.copy2(model_version.model_path, target_path)
            logger.info(f"Restored version {version} to {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore version {version}: {e}")
            return False
    
    def delete_version(self, version: str) -> bool:
        """
        Delete a version.
        
        Args:
            version: Version to delete
            
        Returns:
            True if successful
        """
        if version not in self.versions:
            return False
        
        try:
            version_dir = self.versions_dir / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            del self.versions[version]
            
            if self.current_version == version:
                self.current_version = None
            
            logger.info(f"Deleted version {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete version {version}: {e}")
            return False
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dictionary
        """
        v1 = self.versions.get(version1)
        v2 = self.versions.get(version2)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        comparison = {
            "version1": {
                "version": v1.version,
                "timestamp": v1.timestamp,
                "metrics": v1.metrics,
            },
            "version2": {
                "version": v2.version,
                "timestamp": v2.timestamp,
                "metrics": v2.metrics,
            },
            "differences": {},
        }
        
        # Compare metrics
        for key in set(v1.metrics.keys()) | set(v2.metrics.keys()):
            val1 = v1.metrics.get(key, 0.0)
            val2 = v2.metrics.get(key, 0.0)
            if val1 != val2:
                comparison["differences"][key] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1,
                }
        
        return comparison
    
    def _generate_version(self) -> str:
        """Generate a new version string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"
    
    def _load_versions(self) -> None:
        """Load existing versions from disk."""
        if not self.versions_dir.exists():
            return
        
        for version_dir in self.versions_dir.iterdir():
            if not version_dir.is_dir():
                continue
            
            metadata_path = version_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                model_version = ModelVersion(
                    version=metadata["version"],
                    timestamp=metadata["timestamp"],
                    model_path=Path(metadata["model_path"]),
                    config=metadata["config"],
                    metrics=metadata["metrics"],
                    description=metadata.get("description", ""),
                    tags=metadata.get("tags", []),
                )
                
                self.versions[metadata["version"]] = model_version
            except Exception as e:
                logger.warning(f"Failed to load version from {version_dir}: {e}")
    
    def _cleanup_old_versions(self) -> None:
        """Remove old versions if over limit."""
        if len(self.versions) <= self.max_versions:
            return
        
        # Sort by timestamp and remove oldest
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].timestamp
        )
        
        to_remove = len(self.versions) - self.max_versions
        for version, _ in sorted_versions[:to_remove]:
            self.delete_version(version)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning statistics."""
        return {
            "total_versions": len(self.versions),
            "current_version": self.current_version,
            "max_versions": self.max_versions,
            "versions_dir": str(self.versions_dir),
        }


