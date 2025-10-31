from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import time
import hashlib
import shutil
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import structlog
import gzip
import torch
import torch.nn as nn
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Versioning System for Key Messages ML Pipeline
Handles model registration, metadata management, and version control
"""


logger = structlog.get_logger(__name__)

@dataclass
class ModelMetadata:
    """Represents model metadata."""
    architecture: str
    dataset: str
    accuracy: float
    training_time: str
    framework: str = "pytorch"
    python_version: str = "3.9"
    parameters: Optional[int] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self) -> Any:
        if not self.architecture or not self.dataset:
            raise ValueError("Architecture and dataset are required")

@dataclass
class ModelVersion:
    """Represents a model version."""
    name: str
    version: str
    path: str
    metadata: ModelMetadata
    hash: str
    file_size: int
    created_at: float
    
    def __post_init__(self) -> Any:
        if not self.name or not self.version or not self.path:
            raise ValueError("Name, version, and path are required")

@dataclass
class ModelInfo:
    """Represents information about a model."""
    name: str
    versions: List[str]
    latest_version: str
    total_versions: int
    first_version: str
    last_updated: float
    
    def __post_init__(self) -> Any:
        if not self.name:
            raise ValueError("Name is required")

class ModelVersionManager:
    """Manages model versioning and registration."""
    
    def __init__(self, registry_path: str = "./model_registry", 
                 auto_version: bool = True, version_scheme: str = "semantic",
                 metadata_schema: Dict[str, Any] = None, compression: bool = True):
        
    """__init__ function."""
self.registry_path = Path(registry_path)
        self.auto_version = auto_version
        self.version_scheme = version_scheme
        self.metadata_schema = metadata_schema or {}
        self.compression = compression
        
        # Create registry directory
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self.metadata_file = self.registry_path / "registry.json"
        self._load_metadata()
        
        logger.info("ModelVersionManager initialized", 
                   registry_path=str(self.registry_path),
                   auto_version=auto_version,
                   version_scheme=version_scheme)
    
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
                self.metadata = {"models": {}, "latest_versions": {}}
        else:
            self.metadata = {"models": {}, "latest_versions": {}}
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
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a version string based on the versioning scheme."""
        if self.version_scheme == "semantic":
            # Get current version
            current_version = self.metadata.get("latest_versions", {}).get(model_name, "0.0.0")
            
            # Parse semantic version
            major, minor, patch = map(int, current_version.split("."))
            
            # Increment patch version
            patch += 1
            
            return f"{major}.{minor}.{patch}"
        
        elif self.version_scheme == "timestamp":
            # Use timestamp-based versioning
            timestamp = int(time.time())
            return f"v{timestamp}"
        
        elif self.version_scheme == "hash":
            # Use hash-based versioning
            timestamp = int(time.time())
            hash_suffix = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
            return f"v{timestamp}_{hash_suffix}"
        
        else:
            # Default to timestamp
            timestamp = int(time.time())
            return f"v{timestamp}"
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema."""
        if not self.metadata_schema:
            return True
        
        required_fields = self.metadata_schema.get("required_fields", [])
        optional_fields = self.metadata_schema.get("optional_fields", [])
        
        # Check required fields
        for field in required_fields:
            if field not in metadata:
                logger.error(f"Required field missing: {field}")
                return False
        
        # Check field types if specified
        field_types = self.metadata_schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in metadata:
                if not isinstance(metadata[field], expected_type):
                    logger.error(f"Field {field} has wrong type. Expected {expected_type}, got {type(metadata[field])}")
                    return False
        
        return True
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of model file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for chunk in iter(lambda: f.read(4096), b""):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error("Failed to compute file hash", file_path=file_path, error=str(e))
            return ""
    
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
    
    def register_model(self, model_path: str, model_name: str, 
                      metadata: Dict[str, Any], version: Optional[str] = None,
                      tags: List[str] = None) -> Optional[ModelVersion]:
        """Register a new model version."""
        try:
            # Validate metadata
            if not self._validate_metadata(metadata):
                logger.error("Metadata validation failed")
                return None
            
            # Generate version if not provided
            if not version:
                version = self._generate_version(model_name)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error("Model file not found", model_path=model_path)
                return None
            
            # Compute file hash and size
            file_hash = self._compute_file_hash(model_path)
            file_size = os.path.getsize(model_path)
            
            # Create model metadata
            model_metadata = ModelMetadata(
                architecture=metadata["architecture"],
                dataset=metadata["dataset"],
                accuracy=metadata["accuracy"],
                training_time=metadata.get("training_time", "unknown"),
                framework=metadata.get("framework", "pytorch"),
                python_version=metadata.get("python_version", "3.9"),
                parameters=metadata.get("parameters"),
                description=metadata.get("description"),
                tags=tags or []
            )
            
            # Copy model to registry
            registry_model_path = self.registry_path / f"{model_name}_{version}.pt"
            shutil.copy2(model_path, registry_model_path)
            
            # Compress if enabled
            if self.compression:
                compressed_path = self._compress_file(str(registry_model_path))
                final_path = compressed_path
            else:
                final_path = str(registry_model_path)
            
            # Create model version
            model_version = ModelVersion(
                name=model_name,
                version=version,
                path=final_path,
                metadata=model_metadata,
                hash=file_hash,
                file_size=file_size,
                created_at=time.time()
            )
            
            # Update metadata
            if model_name not in self.metadata["models"]:
                self.metadata["models"][model_name] = {
                    "versions": [],
                    "first_version": version,
                    "last_updated": time.time()
                }
            
            # Add version info
            version_info = {
                "version": version,
                "path": final_path,
                "metadata": {
                    "architecture": model_metadata.architecture,
                    "dataset": model_metadata.dataset,
                    "accuracy": model_metadata.accuracy,
                    "training_time": model_metadata.training_time,
                    "framework": model_metadata.framework,
                    "python_version": model_metadata.python_version,
                    "parameters": model_metadata.parameters,
                    "description": model_metadata.description,
                    "tags": model_metadata.tags,
                    "created_at": model_metadata.created_at,
                    "updated_at": model_metadata.updated_at
                },
                "hash": file_hash,
                "file_size": file_size,
                "created_at": model_version.created_at
            }
            
            self.metadata["models"][model_name]["versions"].append(version_info)
            self.metadata["models"][model_name]["last_updated"] = time.time()
            self.metadata["latest_versions"][model_name] = version
            
            # Sort versions by creation time
            self.metadata["models"][model_name]["versions"].sort(
                key=lambda x: x["created_at"], reverse=True
            )
            
            self._save_metadata()
            
            logger.info("Model registered", 
                       model_name=model_name,
                       version=version,
                       accuracy=model_metadata.accuracy)
            
            return model_version
            
        except Exception as e:
            logger.error("Failed to register model", 
                        model_name=model_name,
                        version=version,
                        error=str(e))
            return None
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        try:
            if model_name not in self.metadata["models"]:
                logger.warning("Model not found", model_name=model_name)
                return None
            
            model_data = self.metadata["models"][model_name]
            versions = [v["version"] for v in model_data["versions"]]
            
            return ModelInfo(
                name=model_name,
                versions=versions,
                latest_version=self.metadata["latest_versions"].get(model_name, ""),
                total_versions=len(versions),
                first_version=model_data["first_version"],
                last_updated=model_data["last_updated"]
            )
            
        except Exception as e:
            logger.error("Failed to get model info", model_name=model_name, error=str(e))
            return None
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        try:
            models = []
            for model_name in self.metadata["models"]:
                model_info = self.get_model_info(model_name)
                if model_info:
                    models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []
    
    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        try:
            if model_name not in self.metadata["models"]:
                logger.warning("Model not found", model_name=model_name)
                return []
            
            versions = []
            for version_info in self.metadata["models"][model_name]["versions"]:
                metadata = ModelMetadata(
                    architecture=version_info["metadata"]["architecture"],
                    dataset=version_info["metadata"]["dataset"],
                    accuracy=version_info["metadata"]["accuracy"],
                    training_time=version_info["metadata"]["training_time"],
                    framework=version_info["metadata"]["framework"],
                    python_version=version_info["metadata"]["python_version"],
                    parameters=version_info["metadata"].get("parameters"),
                    description=version_info["metadata"].get("description"),
                    tags=version_info["metadata"].get("tags", []),
                    created_at=version_info["metadata"]["created_at"],
                    updated_at=version_info["metadata"]["updated_at"]
                )
                
                model_version = ModelVersion(
                    name=model_name,
                    version=version_info["version"],
                    path=version_info["path"],
                    metadata=metadata,
                    hash=version_info["hash"],
                    file_size=version_info["file_size"],
                    created_at=version_info["created_at"]
                )
                
                versions.append(model_version)
            
            return versions
            
        except Exception as e:
            logger.error("Failed to list model versions", model_name=model_name, error=str(e))
            return []
    
    def get_metadata(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version."""
        try:
            if model_name not in self.metadata["models"]:
                logger.warning("Model not found", model_name=model_name)
                return None
            
            for version_info in self.metadata["models"][model_name]["versions"]:
                if version_info["version"] == version:
                    metadata = version_info["metadata"]
                    return ModelMetadata(
                        architecture=metadata["architecture"],
                        dataset=metadata["dataset"],
                        accuracy=metadata["accuracy"],
                        training_time=metadata["training_time"],
                        framework=metadata["framework"],
                        python_version=metadata["python_version"],
                        parameters=metadata.get("parameters"),
                        description=metadata.get("description"),
                        tags=metadata.get("tags", []),
                        created_at=metadata["created_at"],
                        updated_at=metadata["updated_at"]
                    )
            
            logger.warning("Version not found", model_name=model_name, version=version)
            return None
            
        except Exception as e:
            logger.error("Failed to get model metadata", 
                        model_name=model_name,
                        version=version,
                        error=str(e))
            return None
    
    def load_model(self, model_name: str, version: str, 
                  device: str = "cpu") -> Optional[nn.Module]:
        """Load a specific model version."""
        try:
            if model_name not in self.metadata["models"]:
                logger.error("Model not found", model_name=model_name)
                return None
            
            # Find version
            model_path = None
            for version_info in self.metadata["models"][model_name]["versions"]:
                if version_info["version"] == version:
                    model_path = version_info["path"]
                    break
            
            if not model_path:
                logger.error("Version not found", model_name=model_name, version=version)
                return None
            
            # Check if file exists
            if not os.path.exists(model_path):
                logger.error("Model file not found", model_path=model_path)
                return None
            
            # Decompress if needed
            if model_path.endswith('.gz'):
                model_path = self._decompress_file(model_path)
            
            # Load model
            model = torch.load(model_path, map_location=device)
            
            logger.info("Model loaded", 
                       model_name=model_name,
                       version=version,
                       device=device)
            
            return model
            
        except Exception as e:
            logger.error("Failed to load model", 
                        model_name=model_name,
                        version=version,
                        error=str(e))
            return None
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version."""
        try:
            if model_name not in self.metadata["models"]:
                logger.warning("Model not found", model_name=model_name)
                return False
            
            # Find and remove version
            model_data = self.metadata["models"][model_name]
            version_info = None
            
            for i, v_info in enumerate(model_data["versions"]):
                if v_info["version"] == version:
                    version_info = v_info
                    del model_data["versions"][i]
                    break
            
            if not version_info:
                logger.warning("Version not found", model_name=model_name, version=version)
                return False
            
            # Remove file
            if os.path.exists(version_info["path"]):
                os.remove(version_info["path"])
            
            # Update latest version if needed
            if self.metadata["latest_versions"].get(model_name) == version:
                if model_data["versions"]:
                    self.metadata["latest_versions"][model_name] = model_data["versions"][0]["version"]
                else:
                    del self.metadata["latest_versions"][model_name]
                    del self.metadata["models"][model_name]
            
            # Update last_updated
            if model_name in self.metadata["models"]:
                self.metadata["models"][model_name]["last_updated"] = time.time()
            
            self._save_metadata()
            
            logger.info("Model version deleted", model_name=model_name, version=version)
            return True
            
        except Exception as e:
            logger.error("Failed to delete model version", 
                        model_name=model_name,
                        version=version,
                        error=str(e))
            return False
    
    def search_models(self, query: str) -> List[ModelInfo]:
        """Search for models by name, architecture, or tags."""
        try:
            results = []
            
            for model_name in self.metadata["models"]:
                model_info = self.get_model_info(model_name)
                if not model_info:
                    continue
                
                # Search in model name
                if query.lower() in model_name.lower():
                    results.append(model_info)
                    continue
                
                # Search in metadata
                for version_info in self.metadata["models"][model_name]["versions"]:
                    metadata = version_info["metadata"]
                    
                    # Search in architecture
                    if query.lower() in metadata["architecture"].lower():
                        results.append(model_info)
                        break
                    
                    # Search in dataset
                    if query.lower() in metadata["dataset"].lower():
                        results.append(model_info)
                        break
                    
                    # Search in tags
                    for tag in metadata.get("tags", []):
                        if query.lower() in tag.lower():
                            results.append(model_info)
                            break
                    else:
                        continue
                    break
            
            logger.info("Models searched", query=query, results=len(results))
            return results
            
        except Exception as e:
            logger.error("Failed to search models", query=query, error=str(e))
            return []
    
    def export_model(self, model_name: str, version: str, export_path: str) -> bool:
        """Export a model version to a file."""
        try:
            if model_name not in self.metadata["models"]:
                logger.error("Model not found", model_name=model_name)
                return False
            
            # Find version
            model_path = None
            version_metadata = None
            
            for version_info in self.metadata["models"][model_name]["versions"]:
                if version_info["version"] == version:
                    model_path = version_info["path"]
                    version_metadata = version_info
                    break
            
            if not model_path:
                logger.error("Version not found", model_name=model_name, version=version)
                return False
            
            # Check if file exists
            if not os.path.exists(model_path):
                logger.error("Model file not found", model_path=model_path)
                return False
            
            # Decompress if needed
            if model_path.endswith('.gz'):
                model_path = self._decompress_file(model_path)
            
            # Copy to export path
            shutil.copy2(model_path, export_path)
            
            logger.info("Model exported", 
                       model_name=model_name,
                       version=version,
                       export_path=export_path)
            
            return True
            
        except Exception as e:
            logger.error("Failed to export model", 
                        model_name=model_name,
                        version=version,
                        export_path=export_path,
                        error=str(e))
            return False
    
    def import_model(self, model_path: str, model_name: str, 
                    metadata: Dict[str, Any], version: Optional[str] = None) -> Optional[ModelVersion]:
        """Import a model from a file."""
        try:
            return self.register_model(
                model_path=model_path,
                model_name=model_name,
                metadata=metadata,
                version=version
            )
            
        except Exception as e:
            logger.error("Failed to import model", 
                        model_path=model_path,
                        model_name=model_name,
                        error=str(e))
            return None
    
    def cleanup_old_versions(self, model_name: str, keep_count: int = 5) -> int:
        """Clean up old versions of a model."""
        try:
            if model_name not in self.metadata["models"]:
                return 0
            
            model_data = self.metadata["models"][model_name]
            versions = model_data["versions"]
            
            if len(versions) <= keep_count:
                return 0
            
            # Remove old versions
            old_versions = versions[keep_count:]
            removed_count = 0
            
            for version_info in old_versions:
                if self.delete_version(model_name, version_info["version"]):
                    removed_count += 1
            
            logger.info("Old model versions cleaned up", 
                       model_name=model_name,
                       removed_count=removed_count,
                       kept_count=keep_count)
            
            return removed_count
            
        except Exception as e:
            logger.error("Failed to cleanup old versions", model_name=model_name, error=str(e))
            return 0

class ModelRegistry:
    """High-level model registry interface."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        
    """__init__ function."""
self.manager = ModelVersionManager(registry_path=registry_path)
    
    def register(self, name: str, version: str, path: str, 
                metadata: Dict[str, Any], tags: List[str] = None) -> bool:
        """Register a model."""
        model_version = self.manager.register_model(
            model_path=path,
            model_name=name,
            metadata=metadata,
            version=version,
            tags=tags
        )
        return model_version is not None
    
    def list_models(self) -> List[ModelInfo]:
        """List all models."""
        return self.manager.list_models()
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information."""
        return self.manager.get_model_info(name)
    
    def load_model(self, name: str, version: str, device: str = "cpu") -> Optional[nn.Module]:
        """Load a model."""
        return self.manager.load_model(name, version, device)
    
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version."""
        return self.manager.delete_version(name, version)
    
    def search(self, query: str) -> List[ModelInfo]:
        """Search for models."""
        return self.manager.search_models(query) 