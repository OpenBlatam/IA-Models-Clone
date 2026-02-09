#!/usr/bin/env python3
"""
Advanced Model Versioning System for Frontier Model Training
Provides comprehensive model versioning, lineage tracking, and metadata management.
"""

import os
import json
import yaml
import time
import hashlib
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import git
import dvc
import mlflow
import wandb
import sqlite3
from contextlib import contextmanager
import pickle
import joblib
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import boto3
import azure.storage.blob
import google.cloud.storage
import redis
import elasticsearch
from elasticsearch import Elasticsearch

console = Console()

class VersionType(Enum):
    """Version types."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRE_RELEASE = "pre_release"
    BUILD = "build"

class ModelStatus(Enum):
    """Model status."""
    DRAFT = "draft"
    TRAINING = "training"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class LineageType(Enum):
    """Lineage types."""
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    DERIVED = "derived"
    ENSEMBLE = "ensemble"

@dataclass
class ModelMetadata:
    """Model metadata."""
    model_id: str
    version: str
    name: str
    description: str
    tags: List[str]
    created_at: datetime
    created_by: str
    status: ModelStatus
    model_type: str
    framework: str
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    size_mb: float
    checksum: str
    dependencies: List[str]
    lineage: Dict[str, Any]
    artifacts: Dict[str, str]
    environment: Dict[str, Any]
    deployment_info: Dict[str, Any]

@dataclass
class VersionInfo:
    """Version information."""
    version: str
    version_type: VersionType
    created_at: datetime
    created_by: str
    changelog: str
    breaking_changes: bool
    migration_guide: Optional[str]
    deprecation_notice: Optional[str]
    compatibility: Dict[str, Any]

@dataclass
class LineageNode:
    """Lineage node."""
    model_id: str
    version: str
    relationship_type: LineageType
    relationship_data: Dict[str, Any]
    created_at: datetime

class ModelVersionManager:
    """Main model version manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.storage_backend = self._init_storage_backend()
        self.metadata_store = self._init_metadata_store()
        self.lineage_graph = nx.DiGraph()
        self.version_scheme = self._init_version_scheme()
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Model registry
        self.models: Dict[str, ModelMetadata] = {}
        self.versions: Dict[str, List[VersionInfo]] = {}
        
        # Load existing data
        self._load_existing_data()
    
    def _init_storage_backend(self):
        """Initialize storage backend."""
        storage_type = self.config.get("storage_type", "local")
        
        if storage_type == "s3":
            return S3Storage(self.config.get("s3_config", {}))
        elif storage_type == "azure":
            return AzureStorage(self.config.get("azure_config", {}))
        elif storage_type == "gcp":
            return GCPStorage(self.config.get("gcp_config", {}))
        else:
            return LocalStorage(self.config.get("local_config", {}))
    
    def _init_metadata_store(self):
        """Initialize metadata store."""
        store_type = self.config.get("metadata_store", "sqlite")
        
        if store_type == "elasticsearch":
            return ElasticsearchStore(self.config.get("elasticsearch_config", {}))
        elif store_type == "redis":
            return RedisStore(self.config.get("redis_config", {}))
        else:
            return SQLiteStore(self.config.get("sqlite_config", {}))
    
    def _init_version_scheme(self):
        """Initialize version scheme."""
        scheme_type = self.config.get("version_scheme", "semantic")
        
        if scheme_type == "semantic":
            return SemanticVersioning()
        elif scheme_type == "timestamp":
            return TimestampVersioning()
        elif scheme_type == "hash":
            return HashVersioning()
        else:
            return CustomVersioning()
    
    def _init_database(self) -> str:
        """Initialize model versioning database."""
        db_path = Path("./model_versioning.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    training_config TEXT NOT NULL,
                    dataset_info TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    size_mb REAL NOT NULL,
                    checksum TEXT NOT NULL,
                    dependencies TEXT,
                    lineage TEXT,
                    artifacts TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployment_info TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    version_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    changelog TEXT,
                    breaking_changes BOOLEAN DEFAULT FALSE,
                    migration_guide TEXT,
                    deprecation_notice TEXT,
                    compatibility TEXT,
                    PRIMARY KEY (model_id, version),
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage (
                    parent_model_id TEXT NOT NULL,
                    parent_version TEXT NOT NULL,
                    child_model_id TEXT NOT NULL,
                    child_version TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    relationship_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (parent_model_id, parent_version, child_model_id, child_version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT NOT NULL,
                    results TEXT,
                    parent_experiment_id TEXT,
                    FOREIGN KEY (parent_experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
        
        return str(db_path)
    
    def _load_existing_data(self):
        """Load existing model data."""
        with sqlite3.connect(self.db_path) as conn:
            # Load models
            cursor = conn.execute("SELECT * FROM models")
            for row in cursor.fetchall():
                model_metadata = self._row_to_model_metadata(row)
                self.models[model_metadata.model_id] = model_metadata
            
            # Load versions
            cursor = conn.execute("SELECT * FROM versions")
            for row in cursor.fetchall():
                version_info = self._row_to_version_info(row)
                model_id = row[0]
                if model_id not in self.versions:
                    self.versions[model_id] = []
                self.versions[model_id].append(version_info)
            
            # Load lineage
            cursor = conn.execute("SELECT * FROM lineage")
            for row in cursor.fetchall():
                parent_id, parent_version, child_id, child_version, rel_type, rel_data, created_at = row
                
                self.lineage_graph.add_edge(
                    f"{parent_id}:{parent_version}",
                    f"{child_id}:{child_version}",
                    relationship_type=rel_type,
                    relationship_data=json.loads(rel_data),
                    created_at=datetime.fromisoformat(created_at)
                )
    
    def _row_to_model_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            model_id=row[0],
            version=row[1],
            name=row[2],
            description=row[3],
            tags=json.loads(row[4]) if row[4] else [],
            created_at=datetime.fromisoformat(row[5]),
            created_by=row[6],
            status=ModelStatus(row[7]),
            model_type=row[8],
            framework=row[9],
            architecture=json.loads(row[10]),
            hyperparameters=json.loads(row[11]),
            training_config=json.loads(row[12]),
            dataset_info=json.loads(row[13]),
            metrics=json.loads(row[14]),
            performance_metrics=json.loads(row[15]),
            size_mb=row[16],
            checksum=row[17],
            dependencies=json.loads(row[18]) if row[18] else [],
            lineage=json.loads(row[19]) if row[19] else {},
            artifacts=json.loads(row[20]),
            environment=json.loads(row[21]),
            deployment_info=json.loads(row[22]) if row[22] else {}
        )
    
    def _row_to_version_info(self, row) -> VersionInfo:
        """Convert database row to VersionInfo."""
        return VersionInfo(
            version=row[1],
            version_type=VersionType(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            created_by=row[4],
            changelog=row[5],
            breaking_changes=bool(row[6]),
            migration_guide=row[7],
            deprecation_notice=row[8],
            compatibility=json.loads(row[9]) if row[9] else {}
        )
    
    def register_model(self, model_path: str, metadata: Dict[str, Any], 
                      created_by: str, tags: List[str] = None) -> str:
        """Register a new model version."""
        # Generate model ID and version
        model_id = self._generate_model_id(metadata.get("name", "unnamed_model"))
        version = self.version_scheme.generate_version(metadata)
        
        # Calculate model checksum
        checksum = self._calculate_model_checksum(model_path)
        
        # Calculate model size
        size_mb = self._calculate_model_size(model_path)
        
        # Extract model information
        model_info = self._extract_model_info(model_path, metadata)
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=metadata.get("name", "unnamed_model"),
            description=metadata.get("description", ""),
            tags=tags or [],
            created_at=datetime.now(),
            created_by=created_by,
            status=ModelStatus.DRAFT,
            model_type=model_info.get("model_type", "unknown"),
            framework=model_info.get("framework", "unknown"),
            architecture=model_info.get("architecture", {}),
            hyperparameters=metadata.get("hyperparameters", {}),
            training_config=metadata.get("training_config", {}),
            dataset_info=metadata.get("dataset_info", {}),
            metrics=metadata.get("metrics", {}),
            performance_metrics=metadata.get("performance_metrics", {}),
            size_mb=size_mb,
            checksum=checksum,
            dependencies=metadata.get("dependencies", []),
            lineage={},
            artifacts={"model_path": model_path},
            environment=metadata.get("environment", {}),
            deployment_info={}
        )
        
        # Store model
        self._store_model(model_metadata)
        
        # Store artifacts
        self.storage_backend.store_model(model_id, version, model_path, model_metadata)
        
        # Update metadata store
        self.metadata_store.store_metadata(model_id, version, model_metadata)
        
        # Track locally
        self.models[model_id] = model_metadata
        
        console.print(f"[green]Model registered: {model_id} v{version}[/green]")
        return model_id
    
    def create_version(self, model_id: str, metadata: Dict[str, Any], 
                      created_by: str, version_type: VersionType = VersionType.MINOR) -> str:
        """Create a new version of an existing model."""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        # Get current model
        current_model = self.models[model_id]
        
        # Generate new version
        new_version = self.version_scheme.generate_version(
            metadata, current_model.version, version_type
        )
        
        # Create version info
        version_info = VersionInfo(
            version=new_version,
            version_type=version_type,
            created_at=datetime.now(),
            created_by=created_by,
            changelog=metadata.get("changelog", ""),
            breaking_changes=metadata.get("breaking_changes", False),
            migration_guide=metadata.get("migration_guide"),
            deprecation_notice=metadata.get("deprecation_notice"),
            compatibility=metadata.get("compatibility", {})
        )
        
        # Create new model metadata
        new_model_metadata = ModelMetadata(
            model_id=model_id,
            version=new_version,
            name=current_model.name,
            description=metadata.get("description", current_model.description),
            tags=metadata.get("tags", current_model.tags),
            created_at=datetime.now(),
            created_by=created_by,
            status=ModelStatus.DRAFT,
            model_type=current_model.model_type,
            framework=current_model.framework,
            architecture=metadata.get("architecture", current_model.architecture),
            hyperparameters=metadata.get("hyperparameters", current_model.hyperparameters),
            training_config=metadata.get("training_config", current_model.training_config),
            dataset_info=metadata.get("dataset_info", current_model.dataset_info),
            metrics=metadata.get("metrics", current_model.metrics),
            performance_metrics=metadata.get("performance_metrics", current_model.performance_metrics),
            size_mb=current_model.size_mb,
            checksum=current_model.checksum,
            dependencies=metadata.get("dependencies", current_model.dependencies),
            lineage=current_model.lineage,
            artifacts=current_model.artifacts,
            environment=metadata.get("environment", current_model.environment),
            deployment_info=current_model.deployment_info
        )
        
        # Store version
        self._store_version(version_info)
        
        # Store model
        self._store_model(new_model_metadata)
        
        # Update lineage
        self._add_lineage_relationship(
            model_id, current_model.version, model_id, new_version, 
            LineageType.CHILD, {"type": "version_update"}
        )
        
        # Track locally
        if model_id not in self.versions:
            self.versions[model_id] = []
        self.versions[model_id].append(version_info)
        
        console.print(f"[green]New version created: {model_id} v{new_version}[/green]")
        return new_version
    
    def get_model(self, model_id: str, version: str = None) -> Optional[ModelMetadata]:
        """Get model metadata."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        if version and model.version != version:
            # Load specific version from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM models WHERE model_id = ? AND version = ?",
                    (model_id, version)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_model_metadata(row)
            return None
        
        return model
    
    def list_models(self, filters: Dict[str, Any] = None) -> List[ModelMetadata]:
        """List models with optional filters."""
        models = list(self.models.values())
        
        if filters:
            filtered_models = []
            for model in models:
                if self._matches_filters(model, filters):
                    filtered_models.append(model)
            return filtered_models
        
        return models
    
    def list_versions(self, model_id: str) -> List[VersionInfo]:
        """List versions for a model."""
        if model_id not in self.versions:
            return []
        
        return sorted(self.versions[model_id], key=lambda v: v.created_at, reverse=True)
    
    def compare_versions(self, model_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        model1 = self.get_model(model_id, version1)
        model2 = self.get_model(model_id, version2)
        
        if not model1 or not model2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "model_id": model_id,
            "version1": version1,
            "version2": version2,
            "differences": {},
            "metrics_comparison": {},
            "performance_comparison": {}
        }
        
        # Compare metadata
        comparison["differences"] = self._compare_metadata(model1, model2)
        
        # Compare metrics
        comparison["metrics_comparison"] = self._compare_metrics(
            model1.metrics, model2.metrics
        )
        
        # Compare performance
        comparison["performance_comparison"] = self._compare_metrics(
            model1.performance_metrics, model2.performance_metrics
        )
        
        return comparison
    
    def get_lineage(self, model_id: str, version: str = None) -> Dict[str, Any]:
        """Get model lineage."""
        model_key = f"{model_id}:{version}" if version else f"{model_id}:latest"
        
        # Get ancestors (parents)
        ancestors = []
        for predecessor in self.lineage_graph.predecessors(model_key):
            edge_data = self.lineage_graph.get_edge_data(predecessor, model_key)
            ancestors.append({
                "model_id": predecessor.split(":")[0],
                "version": predecessor.split(":")[1],
                "relationship_type": edge_data["relationship_type"],
                "relationship_data": edge_data["relationship_data"]
            })
        
        # Get descendants (children)
        descendants = []
        for successor in self.lineage_graph.successors(model_key):
            edge_data = self.lineage_graph.get_edge_data(model_key, successor)
            descendants.append({
                "model_id": successor.split(":")[0],
                "version": successor.split(":")[1],
                "relationship_type": edge_data["relationship_type"],
                "relationship_data": edge_data["relationship_data"]
            })
        
        return {
            "model_id": model_id,
            "version": version,
            "ancestors": ancestors,
            "descendants": descendants,
            "lineage_graph": self._serialize_lineage_graph()
        }
    
    def visualize_lineage(self, model_id: str, output_path: str = None) -> str:
        """Visualize model lineage."""
        if output_path is None:
            output_path = f"lineage_{model_id}.png"
        
        # Create subgraph for the model
        model_nodes = [node for node in self.lineage_graph.nodes() 
                      if node.startswith(f"{model_id}:")]
        
        subgraph = self.lineage_graph.subgraph(model_nodes)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Draw labels
        labels = {node: node.split(":")[1] for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title(f"Model Lineage: {model_id}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Lineage visualization saved: {output_path}[/green]")
        return output_path
    
    def deprecate_model(self, model_id: str, version: str, 
                       deprecation_notice: str, created_by: str):
        """Deprecate a model version."""
        model = self.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model not found: {model_id} v{version}")
        
        # Update status
        model.status = ModelStatus.DEPRECATED
        
        # Update deprecation notice
        if model_id in self.versions:
            for version_info in self.versions[model_id]:
                if version_info.version == version:
                    version_info.deprecation_notice = deprecation_notice
                    break
        
        # Save to database
        self._store_model(model)
        
        console.print(f"[yellow]Model deprecated: {model_id} v{version}[/yellow]")
    
    def archive_model(self, model_id: str, version: str, reason: str, created_by: str):
        """Archive a model version."""
        model = self.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model not found: {model_id} v{version}")
        
        # Update status
        model.status = ModelStatus.ARCHIVED
        
        # Add archive reason to metadata
        model.description += f"\n\nArchived: {reason}"
        
        # Save to database
        self._store_model(model)
        
        console.print(f"[yellow]Model archived: {model_id} v{version}[/yellow]")
    
    def _generate_model_id(self, name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name}_{name_hash}_{timestamp}"
    
    def _calculate_model_checksum(self, model_path: str) -> str:
        """Calculate model file checksum."""
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model file size in MB."""
        size_bytes = Path(model_path).stat().st_size
        return size_bytes / (1024 * 1024)
    
    def _extract_model_info(self, model_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from file."""
        model_info = {
            "model_type": "unknown",
            "framework": "unknown",
            "architecture": {}
        }
        
        # Try to determine framework and type
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            model_info["framework"] = "pytorch"
            model_info["model_type"] = "neural_network"
        elif model_path.endswith('.pkl') or model_path.endswith('.joblib'):
            model_info["framework"] = "sklearn"
            model_info["model_type"] = "machine_learning"
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            model_info["framework"] = "tensorflow"
            model_info["model_type"] = "neural_network"
        elif model_path.endswith('.onnx'):
            model_info["framework"] = "onnx"
            model_info["model_type"] = "neural_network"
        
        # Try to load and extract architecture info
        try:
            if model_info["framework"] == "pytorch":
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                
                if hasattr(model, 'parameters'):
                    param_count = sum(p.numel() for p in model.parameters())
                    model_info["architecture"]["parameter_count"] = param_count
                
                if hasattr(model, 'modules'):
                    layer_count = len(list(model.modules()))
                    model_info["architecture"]["layer_count"] = layer_count
                    
        except Exception as e:
            self.logger.warning(f"Could not extract model info: {e}")
        
        return model_info
    
    def _matches_filters(self, model: ModelMetadata, filters: Dict[str, Any]) -> bool:
        """Check if model matches filters."""
        for key, value in filters.items():
            if key == "tags" and not any(tag in model.tags for tag in value):
                return False
            elif key == "status" and model.status.value != value:
                return False
            elif key == "framework" and model.framework != value:
                return False
            elif key == "created_by" and model.created_by != value:
                return False
            elif key == "min_accuracy" and model.metrics.get("accuracy", 0) < value:
                return False
        
        return True
    
    def _compare_metadata(self, model1: ModelMetadata, model2: ModelMetadata) -> Dict[str, Any]:
        """Compare model metadata."""
        differences = {}
        
        # Compare basic fields
        basic_fields = ["name", "description", "model_type", "framework"]
        for field in basic_fields:
            if getattr(model1, field) != getattr(model2, field):
                differences[field] = {
                    "version1": getattr(model1, field),
                    "version2": getattr(model2, field)
                }
        
        # Compare hyperparameters
        if model1.hyperparameters != model2.hyperparameters:
            differences["hyperparameters"] = {
                "version1": model1.hyperparameters,
                "version2": model2.hyperparameters
            }
        
        # Compare architecture
        if model1.architecture != model2.architecture:
            differences["architecture"] = {
                "version1": model1.architecture,
                "version2": model2.architecture
            }
        
        return differences
    
    def _compare_metrics(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> Dict[str, Any]:
        """Compare metrics between two models."""
        comparison = {}
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            
            comparison[metric] = {
                "version1": val1,
                "version2": val2,
                "difference": val2 - val1,
                "percentage_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            }
        
        return comparison
    
    def _add_lineage_relationship(self, parent_id: str, parent_version: str,
                                 child_id: str, child_version: str,
                                 relationship_type: LineageType, 
                                 relationship_data: Dict[str, Any]):
        """Add lineage relationship."""
        parent_key = f"{parent_id}:{parent_version}"
        child_key = f"{child_id}:{child_version}"
        
        self.lineage_graph.add_edge(
            parent_key, child_key,
            relationship_type=relationship_type.value,
            relationship_data=relationship_data,
            created_at=datetime.now()
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lineage 
                (parent_model_id, parent_version, child_model_id, child_version, 
                 relationship_type, relationship_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                parent_id, parent_version, child_id, child_version,
                relationship_type.value, json.dumps(relationship_data),
                datetime.now().isoformat()
            ))
    
    def _serialize_lineage_graph(self) -> Dict[str, Any]:
        """Serialize lineage graph for JSON output."""
        return {
            "nodes": list(self.lineage_graph.nodes()),
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "relationship_type": self.lineage_graph[edge[0]][edge[1]]["relationship_type"],
                    "relationship_data": self.lineage_graph[edge[0]][edge[1]]["relationship_data"]
                }
                for edge in self.lineage_graph.edges()
            ]
        }
    
    def _store_model(self, model: ModelMetadata):
        """Store model metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models 
                (model_id, version, name, description, tags, created_at, created_by, 
                 status, model_type, framework, architecture, hyperparameters, 
                 training_config, dataset_info, metrics, performance_metrics, 
                 size_mb, checksum, dependencies, lineage, artifacts, environment, deployment_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_id, model.version, model.name, model.description,
                json.dumps(model.tags), model.created_at.isoformat(), model.created_by,
                model.status.value, model.model_type, model.framework,
                json.dumps(model.architecture), json.dumps(model.hyperparameters),
                json.dumps(model.training_config), json.dumps(model.dataset_info),
                json.dumps(model.metrics), json.dumps(model.performance_metrics),
                model.size_mb, model.checksum, json.dumps(model.dependencies),
                json.dumps(model.lineage), json.dumps(model.artifacts),
                json.dumps(model.environment), json.dumps(model.deployment_info)
            ))
    
    def _store_version(self, version: VersionInfo):
        """Store version info to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO versions 
                (model_id, version, version_type, created_at, created_by, 
                 changelog, breaking_changes, migration_guide, deprecation_notice, compatibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.model_id, version.version, version.version_type.value,
                version.created_at.isoformat(), version.created_by,
                version.changelog, version.breaking_changes,
                version.migration_guide, version.deprecation_notice,
                json.dumps(version.compatibility)
            ))

# Version scheme implementations
class SemanticVersioning:
    def generate_version(self, metadata: Dict[str, Any], 
                        current_version: str = None, 
                        version_type: VersionType = VersionType.MINOR) -> str:
        """Generate semantic version."""
        if not current_version:
            return "1.0.0"
        
        # Parse current version
        parts = current_version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Increment based on type
        if version_type == VersionType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif version_type == VersionType.MINOR:
            minor += 1
            patch = 0
        elif version_type == VersionType.PATCH:
            patch += 1
        
        return f"{major}.{minor}.{patch}"

class TimestampVersioning:
    def generate_version(self, metadata: Dict[str, Any], 
                        current_version: str = None, 
                        version_type: VersionType = None) -> str:
        """Generate timestamp-based version."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"

class HashVersioning:
    def generate_version(self, metadata: Dict[str, Any], 
                        current_version: str = None, 
                        version_type: VersionType = None) -> str:
        """Generate hash-based version."""
        content = json.dumps(metadata, sort_keys=True)
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"hash_{hash_value}"

class CustomVersioning:
    def generate_version(self, metadata: Dict[str, Any], 
                        current_version: str = None, 
                        version_type: VersionType = None) -> str:
        """Generate custom version."""
        return metadata.get("version", "1.0.0")

# Storage backend implementations
class LocalStorage:
    def __init__(self, config: Dict[str, Any]):
        self.base_path = Path(config.get("base_path", "./model_storage"))
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store_model(self, model_id: str, version: str, model_path: str, metadata: ModelMetadata):
        """Store model locally."""
        version_path = self.base_path / model_id / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        shutil.copy2(model_path, version_path / "model")
        
        # Save metadata
        with open(version_path / "metadata.json", 'w') as f:
            json.dump(asdict(metadata), f, default=str)

class S3Storage:
    def __init__(self, config: Dict[str, Any]):
        self.s3_client = boto3.client('s3')
        self.bucket_name = config.get("bucket_name", "model-registry")
    
    def store_model(self, model_id: str, version: str, model_path: str, metadata: ModelMetadata):
        """Store model in S3."""
        key_prefix = f"models/{model_id}/{version}"
        
        # Upload model file
        self.s3_client.upload_file(model_path, self.bucket_name, f"{key_prefix}/model")
        
        # Upload metadata
        metadata_json = json.dumps(asdict(metadata), default=str)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=f"{key_prefix}/metadata.json",
            Body=metadata_json
        )

class AzureStorage:
    def __init__(self, config: Dict[str, Any]):
        self.container_name = config.get("container_name", "model-registry")
        self.blob_service = azure.storage.blob.BlobServiceClient.from_connection_string(
            config.get("connection_string", "")
        )
    
    def store_model(self, model_id: str, version: str, model_path: str, metadata: ModelMetadata):
        """Store model in Azure Blob Storage."""
        blob_prefix = f"models/{model_id}/{version}"
        
        # Upload model file
        with open(model_path, 'rb') as data:
            self.blob_service.upload_blob(
                f"{blob_prefix}/model", data, overwrite=True
            )
        
        # Upload metadata
        metadata_json = json.dumps(asdict(metadata), default=str)
        self.blob_service.upload_blob(
            f"{blob_prefix}/metadata.json", metadata_json, overwrite=True
        )

class GCPStorage:
    def __init__(self, config: Dict[str, Any]):
        self.bucket_name = config.get("bucket_name", "model-registry")
        self.storage_client = google.cloud.storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
    
    def store_model(self, model_id: str, version: str, model_path: str, metadata: ModelMetadata):
        """Store model in GCP Cloud Storage."""
        blob_prefix = f"models/{model_id}/{version}"
        
        # Upload model file
        blob = self.bucket.blob(f"{blob_prefix}/model")
        blob.upload_from_filename(model_path)
        
        # Upload metadata
        metadata_json = json.dumps(asdict(metadata), default=str)
        metadata_blob = self.bucket.blob(f"{blob_prefix}/metadata.json")
        metadata_blob.upload_from_string(metadata_json)

# Metadata store implementations
class SQLiteStore:
    def __init__(self, config: Dict[str, Any]):
        self.db_path = config.get("db_path", "./metadata.db")
    
    def store_metadata(self, model_id: str, version: str, metadata: ModelMetadata):
        """Store metadata in SQLite."""
        # Implementation would store metadata in SQLite
        pass

class RedisStore:
    def __init__(self, config: Dict[str, Any]):
        self.redis_client = redis.Redis(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0)
        )
    
    def store_metadata(self, model_id: str, version: str, metadata: ModelMetadata):
        """Store metadata in Redis."""
        key = f"metadata:{model_id}:{version}"
        self.redis_client.set(key, json.dumps(asdict(metadata), default=str))

class ElasticsearchStore:
    def __init__(self, config: Dict[str, Any]):
        self.es_client = Elasticsearch([config.get("host", "localhost:9200")])
        self.index_name = config.get("index_name", "model_metadata")
    
    def store_metadata(self, model_id: str, version: str, metadata: ModelMetadata):
        """Store metadata in Elasticsearch."""
        doc_id = f"{model_id}_{version}"
        self.es_client.index(
            index=self.index_name,
            id=doc_id,
            body=asdict(metadata)
        )

def main():
    """Main function for model versioning CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Versioning System")
    parser.add_argument("--action", type=str,
                       choices=["register", "create-version", "list", "get", "compare", 
                               "lineage", "visualize", "deprecate", "archive"],
                       required=True, help="Action to perform")
    parser.add_argument("--model-path", type=str, help="Model file path")
    parser.add_argument("--model-id", type=str, help="Model ID")
    parser.add_argument("--version", type=str, help="Model version")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--description", type=str, help="Model description")
    parser.add_argument("--tags", type=str, nargs="+", help="Model tags")
    parser.add_argument("--created-by", type=str, default="user", help="Creator")
    parser.add_argument("--version-type", type=str,
                       choices=["major", "minor", "patch"],
                       default="minor", help="Version type")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Create model version manager
    config = {
        "storage_type": "local",
        "metadata_store": "sqlite",
        "version_scheme": "semantic"
    }
    
    manager = ModelVersionManager(config)
    
    if args.action == "register":
        if not args.model_path or not args.name:
            console.print("[red]Model path and name are required[/red]")
            return
        
        metadata = {
            "name": args.name,
            "description": args.description or "",
            "metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "hyperparameters": {"learning_rate": 0.001, "batch_size": 32}
        }
        
        model_id = manager.register_model(
            args.model_path, metadata, args.created_by, args.tags
        )
        console.print(f"[green]Model registered: {model_id}[/green]")
    
    elif args.action == "create-version":
        if not args.model_id:
            console.print("[red]Model ID is required[/red]")
            return
        
        metadata = {
            "description": args.description or "",
            "changelog": "Updated model with improved performance"
        }
        
        version = manager.create_version(
            args.model_id, metadata, args.created_by, VersionType(args.version_type)
        )
        console.print(f"[green]New version created: {version}[/green]")
    
    elif args.action == "list":
        models = manager.list_models()
        
        table = Table(title="Registered Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Version", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="blue")
        table.add_column("Accuracy", style="red")
        
        for model in models:
            table.add_row(
                model.model_id,
                model.name,
                model.version,
                model.status.value,
                model.created_at.strftime("%Y-%m-%d"),
                f"{model.metrics.get('accuracy', 0):.3f}"
            )
        
        console.print(table)
    
    elif args.action == "get":
        if not args.model_id:
            console.print("[red]Model ID is required[/red]")
            return
        
        model = manager.get_model(args.model_id, args.version)
        if model:
            console.print(f"[blue]Model: {model.name} v{model.version}[/blue]")
            console.print(f"[blue]Description: {model.description}[/blue]")
            console.print(f"[blue]Status: {model.status.value}[/blue]")
            console.print(f"[blue]Metrics: {model.metrics}[/blue]")
        else:
            console.print("[red]Model not found[/red]")
    
    elif args.action == "compare":
        if not args.model_id or not args.version:
            console.print("[red]Model ID and version are required[/red]")
            return
        
        # Compare with previous version
        versions = manager.list_versions(args.model_id)
        if len(versions) < 2:
            console.print("[red]Not enough versions to compare[/red]")
            return
        
        comparison = manager.compare_versions(
            args.model_id, versions[1].version, args.version
        )
        
        console.print(f"[blue]Comparison: {comparison['version1']} vs {comparison['version2']}[/blue]")
        console.print(f"[blue]Performance comparison: {comparison['performance_comparison']}[/blue]")
    
    elif args.action == "lineage":
        if not args.model_id:
            console.print("[red]Model ID is required[/red]")
            return
        
        lineage = manager.get_lineage(args.model_id, args.version)
        
        console.print(f"[blue]Lineage for {args.model_id}:[/blue]")
        console.print(f"[blue]Ancestors: {len(lineage['ancestors'])}[/blue]")
        console.print(f"[blue]Descendants: {len(lineage['descendants'])}[/blue]")
    
    elif args.action == "visualize":
        if not args.model_id:
            console.print("[red]Model ID is required[/red]")
            return
        
        output_path = manager.visualize_lineage(args.model_id, args.output)
        console.print(f"[green]Lineage visualization saved: {output_path}[/green]")
    
    elif args.action == "deprecate":
        if not args.model_id or not args.version:
            console.print("[red]Model ID and version are required[/red]")
            return
        
        manager.deprecate_model(
            args.model_id, args.version, 
            args.description or "Deprecated", args.created_by
        )
        console.print("[yellow]Model deprecated[/yellow]")
    
    elif args.action == "archive":
        if not args.model_id or not args.version:
            console.print("[red]Model ID and version are required[/red]")
            return
        
        manager.archive_model(
            args.model_id, args.version,
            args.description or "Archived", args.created_by
        )
        console.print("[yellow]Model archived[/yellow]")

if __name__ == "__main__":
    main()
