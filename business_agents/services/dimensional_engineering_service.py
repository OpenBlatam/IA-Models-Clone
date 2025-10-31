"""
Dimensional Engineering Service
==============================

Advanced dimensional engineering service for multi-dimensional
space manipulation, dimensional analysis, and space-time engineering.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib
import hmac
from cryptography.fernet import Fernet
import base64
import threading
import time
import math
import random
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

logger = logging.getLogger(__name__)

class DimensionalType(Enum):
    """Dimensional types."""
    EUCLIDEAN = "euclidean"
    NON_EUCLIDEAN = "non_euclidean"
    HYPERBOLIC = "hyperbolic"
    SPHERICAL = "spherical"
    MANIFOLD = "manifold"
    FRACTAL = "fractal"
    QUANTUM = "quantum"
    TRANSCENDENT = "transcendent"

class DimensionalOperation(Enum):
    """Dimensional operations."""
    DIMENSIONAL_REDUCTION = "dimensional_reduction"
    DIMENSIONAL_EXPANSION = "dimensional_expansion"
    DIMENSIONAL_TRANSFORMATION = "dimensional_transformation"
    DIMENSIONAL_PROJECTION = "dimensional_projection"
    DIMENSIONAL_EMBEDDING = "dimensional_embedding"
    DIMENSIONAL_ANALYSIS = "dimensional_analysis"
    DIMENSIONAL_OPTIMIZATION = "dimensional_optimization"
    DIMENSIONAL_SYNTHESIS = "dimensional_synthesis"

class SpaceMetric(Enum):
    """Space metrics."""
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    MANHATTAN_DISTANCE = "manhattan_distance"
    CHEBYSHEV_DISTANCE = "chebyshev_distance"
    COSINE_SIMILARITY = "cosine_similarity"
    MAHALANOBIS_DISTANCE = "mahalanobis_distance"
    HAMMING_DISTANCE = "hamming_distance"
    JACCARD_DISTANCE = "jaccard_distance"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"

@dataclass
class DimensionalSpace:
    """Dimensional space definition."""
    space_id: str
    name: str
    dimensions: int
    space_type: DimensionalType
    metric: SpaceMetric
    coordinates: np.ndarray
    properties: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class DimensionalObject:
    """Dimensional object definition."""
    object_id: str
    name: str
    space_id: str
    position: np.ndarray
    dimensions: int
    properties: Dict[str, Any]
    relationships: List[str]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class DimensionalOperation:
    """Dimensional operation definition."""
    operation_id: str
    name: str
    operation_type: DimensionalOperation
    input_spaces: List[str]
    output_space: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class DimensionalAnalysis:
    """Dimensional analysis definition."""
    analysis_id: str
    name: str
    space_id: str
    analysis_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    accuracy: float
    confidence: float
    created_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any]

class DimensionalEngineeringService:
    """
    Advanced dimensional engineering service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimensional_spaces = {}
        self.dimensional_objects = {}
        self.dimensional_operations = {}
        self.dimensional_analyses = {}
        self.dimensional_algorithms = {}
        self.space_engines = {}
        
        # Dimensional engineering configurations
        self.dimensional_config = config.get("dimensional_engineering", {
            "max_spaces": 100,
            "max_objects": 1000,
            "max_operations": 500,
            "max_analyses": 300,
            "max_dimensions": 1000,
            "default_dimensions": 3,
            "dimensional_analysis_enabled": True,
            "space_manipulation_enabled": True,
            "multi_dimensional_enabled": True,
            "real_time_processing": True
        })
        
    async def initialize(self):
        """Initialize the dimensional engineering service."""
        try:
            await self._initialize_dimensional_algorithms()
            await self._initialize_space_engines()
            await self._load_default_spaces()
            await self._start_dimensional_monitoring()
            logger.info("Dimensional Engineering Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Dimensional Engineering Service: {str(e)}")
            raise
            
    async def _initialize_dimensional_algorithms(self):
        """Initialize dimensional algorithms."""
        try:
            self.dimensional_algorithms = {
                "pca": {
                    "name": "Principal Component Analysis",
                    "description": "Linear dimensionality reduction",
                    "complexity": "O(n^3)",
                    "parameters": {"n_components": 2, "whiten": False},
                    "available": True
                },
                "tsne": {
                    "name": "t-SNE",
                    "description": "Non-linear dimensionality reduction",
                    "complexity": "O(n^2)",
                    "parameters": {"perplexity": 30, "learning_rate": 200},
                    "available": True
                },
                "umap": {
                    "name": "UMAP",
                    "description": "Uniform Manifold Approximation and Projection",
                    "complexity": "O(n log n)",
                    "parameters": {"n_neighbors": 15, "min_dist": 0.1},
                    "available": True
                },
                "isomap": {
                    "name": "Isomap",
                    "description": "Isometric mapping",
                    "complexity": "O(n^3)",
                    "parameters": {"n_neighbors": 5, "n_components": 2},
                    "available": True
                },
                "lle": {
                    "name": "Locally Linear Embedding",
                    "description": "Non-linear dimensionality reduction",
                    "complexity": "O(n^3)",
                    "parameters": {"n_neighbors": 5, "n_components": 2},
                    "available": True
                },
                "mds": {
                    "name": "Multidimensional Scaling",
                    "description": "Distance-based dimensionality reduction",
                    "complexity": "O(n^2)",
                    "parameters": {"n_components": 2, "metric": True},
                    "available": True
                },
                "spectral_embedding": {
                    "name": "Spectral Embedding",
                    "description": "Spectral dimensionality reduction",
                    "complexity": "O(n^3)",
                    "parameters": {"n_components": 2, "affinity": "nearest_neighbors"},
                    "available": True
                },
                "autoencoder": {
                    "name": "Autoencoder",
                    "description": "Neural network dimensionality reduction",
                    "complexity": "O(n)",
                    "parameters": {"hidden_dim": 32, "latent_dim": 2},
                    "available": True
                }
            }
            
            logger.info("Dimensional algorithms initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize dimensional algorithms: {str(e)}")
            
    async def _initialize_space_engines(self):
        """Initialize space engines."""
        try:
            self.space_engines = {
                "euclidean_engine": {
                    "name": "Euclidean Space Engine",
                    "type": "euclidean",
                    "max_dimensions": 1000,
                    "capabilities": ["distance", "angle", "volume", "area"],
                    "available": True
                },
                "non_euclidean_engine": {
                    "name": "Non-Euclidean Space Engine",
                    "type": "non_euclidean",
                    "max_dimensions": 100,
                    "capabilities": ["curvature", "geodesic", "parallel_transport"],
                    "available": True
                },
                "hyperbolic_engine": {
                    "name": "Hyperbolic Space Engine",
                    "type": "hyperbolic",
                    "max_dimensions": 50,
                    "capabilities": ["hyperbolic_distance", "horocycle", "ideal_point"],
                    "available": True
                },
                "spherical_engine": {
                    "name": "Spherical Space Engine",
                    "type": "spherical",
                    "max_dimensions": 20,
                    "capabilities": ["great_circle", "spherical_distance", "spherical_angle"],
                    "available": True
                },
                "manifold_engine": {
                    "name": "Manifold Space Engine",
                    "type": "manifold",
                    "max_dimensions": 10,
                    "capabilities": ["tangent_space", "chart", "atlas"],
                    "available": True
                },
                "fractal_engine": {
                    "name": "Fractal Space Engine",
                    "type": "fractal",
                    "max_dimensions": 5,
                    "capabilities": ["fractal_dimension", "self_similarity", "iteration"],
                    "available": True
                },
                "quantum_engine": {
                    "name": "Quantum Space Engine",
                    "type": "quantum",
                    "max_dimensions": 3,
                    "capabilities": ["superposition", "entanglement", "measurement"],
                    "available": True
                },
                "transcendent_engine": {
                    "name": "Transcendent Space Engine",
                    "type": "transcendent",
                    "max_dimensions": 1,
                    "capabilities": ["transcendence", "infinity", "unity"],
                    "available": True
                }
            }
            
            logger.info("Space engines initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize space engines: {str(e)}")
            
    async def _load_default_spaces(self):
        """Load default dimensional spaces."""
        try:
            # Create sample dimensional spaces
            spaces = [
                DimensionalSpace(
                    space_id="space_001",
                    name="3D Euclidean Space",
                    dimensions=3,
                    space_type=DimensionalType.EUCLIDEAN,
                    metric=SpaceMetric.EUCLIDEAN_DISTANCE,
                    coordinates=np.random.rand(100, 3),
                    properties={"curvature": 0, "topology": "flat"},
                    created_at=datetime.utcnow(),
                    metadata={"type": "default", "description": "Standard 3D space"}
                ),
                DimensionalSpace(
                    space_id="space_002",
                    name="2D Hyperbolic Space",
                    dimensions=2,
                    space_type=DimensionalType.HYPERBOLIC,
                    metric=SpaceMetric.EUCLIDEAN_DISTANCE,
                    coordinates=np.random.rand(50, 2),
                    properties={"curvature": -1, "topology": "hyperbolic"},
                    created_at=datetime.utcnow(),
                    metadata={"type": "default", "description": "Hyperbolic plane"}
                ),
                DimensionalSpace(
                    space_id="space_003",
                    name="4D Spherical Space",
                    dimensions=4,
                    space_type=DimensionalType.SPHERICAL,
                    metric=SpaceMetric.EUCLIDEAN_DISTANCE,
                    coordinates=np.random.rand(30, 4),
                    properties={"curvature": 1, "topology": "spherical"},
                    created_at=datetime.utcnow(),
                    metadata={"type": "default", "description": "4D hypersphere"}
                )
            ]
            
            for space in spaces:
                self.dimensional_spaces[space.space_id] = space
                
            logger.info(f"Loaded {len(spaces)} default dimensional spaces")
            
        except Exception as e:
            logger.error(f"Failed to load default spaces: {str(e)}")
            
    async def _start_dimensional_monitoring(self):
        """Start dimensional monitoring."""
        try:
            # Start background dimensional monitoring
            asyncio.create_task(self._monitor_dimensional_systems())
            logger.info("Started dimensional monitoring")
            
        except Exception as e:
            logger.error(f"Failed to start dimensional monitoring: {str(e)}")
            
    async def _monitor_dimensional_systems(self):
        """Monitor dimensional systems."""
        while True:
            try:
                # Update dimensional spaces
                await self._update_dimensional_spaces()
                
                # Update dimensional objects
                await self._update_dimensional_objects()
                
                # Update dimensional operations
                await self._update_dimensional_operations()
                
                # Update dimensional analyses
                await self._update_dimensional_analyses()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in dimensional monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _update_dimensional_spaces(self):
        """Update dimensional spaces."""
        try:
            # Update space properties
            for space_id, space in self.dimensional_spaces.items():
                # Simulate dynamic space properties
                if "curvature" in space.properties:
                    space.properties["curvature"] += random.uniform(-0.01, 0.01)
                    space.properties["curvature"] = max(-1, min(1, space.properties["curvature"]))
                    
        except Exception as e:
            logger.error(f"Failed to update dimensional spaces: {str(e)}")
            
    async def _update_dimensional_objects(self):
        """Update dimensional objects."""
        try:
            # Update object positions
            for object_id, obj in self.dimensional_objects.items():
                if obj.space_id in self.dimensional_spaces:
                    space = self.dimensional_spaces[obj.space_id]
                    # Simulate object movement
                    movement = np.random.normal(0, 0.01, obj.position.shape)
                    obj.position += movement
                    
        except Exception as e:
            logger.error(f"Failed to update dimensional objects: {str(e)}")
            
    async def _update_dimensional_operations(self):
        """Update dimensional operations."""
        try:
            # Update running operations
            for operation_id, operation in self.dimensional_operations.items():
                if operation.status == "running":
                    # Simulate operation progress
                    if not operation.result:
                        operation.result = {
                            "progress": 0.0,
                            "intermediate_results": []
                        }
                    else:
                        operation.result["progress"] = min(1.0, 
                            operation.result["progress"] + random.uniform(0.01, 0.05))
                        
                        if operation.result["progress"] >= 1.0:
                            operation.status = "completed"
                            operation.completed_at = datetime.utcnow()
                            
        except Exception as e:
            logger.error(f"Failed to update dimensional operations: {str(e)}")
            
    async def _update_dimensional_analyses(self):
        """Update dimensional analyses."""
        try:
            # Update analysis accuracy
            for analysis_id, analysis in self.dimensional_analyses.items():
                if analysis.accuracy < 0.95:
                    analysis.accuracy = min(0.95, analysis.accuracy + 0.01)
                    analysis.confidence = min(0.99, analysis.confidence + 0.005)
                    
        except Exception as e:
            logger.error(f"Failed to update dimensional analyses: {str(e)}")
            
    async def _cleanup_old_data(self):
        """Clean up old dimensional data."""
        try:
            # Remove analyses older than 1 hour
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            old_analyses = [aid for aid, analysis in self.dimensional_analyses.items() 
                          if analysis.created_at < cutoff_time]
            
            for aid in old_analyses:
                del self.dimensional_analyses[aid]
                
            if old_analyses:
                logger.info(f"Cleaned up {len(old_analyses)} old dimensional analyses")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            
    async def create_dimensional_space(self, space: DimensionalSpace) -> str:
        """Create dimensional space."""
        try:
            # Generate space ID if not provided
            if not space.space_id:
                space.space_id = f"space_{uuid.uuid4().hex[:8]}"
                
            # Set timestamp
            space.created_at = datetime.utcnow()
            
            # Validate dimensions
            if space.dimensions <= 0 or space.dimensions > 1000:
                raise ValueError("Invalid number of dimensions")
                
            # Create dimensional space
            self.dimensional_spaces[space.space_id] = space
            
            logger.info(f"Created dimensional space: {space.space_id}")
            
            return space.space_id
            
        except Exception as e:
            logger.error(f"Failed to create dimensional space: {str(e)}")
            raise
            
    async def create_dimensional_object(self, obj: DimensionalObject) -> str:
        """Create dimensional object."""
        try:
            # Generate object ID if not provided
            if not obj.object_id:
                obj.object_id = f"object_{uuid.uuid4().hex[:8]}"
                
            # Set timestamp
            obj.created_at = datetime.utcnow()
            
            # Validate space exists
            if obj.space_id not in self.dimensional_spaces:
                raise ValueError(f"Space {obj.space_id} not found")
                
            # Create dimensional object
            self.dimensional_objects[obj.object_id] = obj
            
            logger.info(f"Created dimensional object: {obj.object_id}")
            
            return obj.object_id
            
        except Exception as e:
            logger.error(f"Failed to create dimensional object: {str(e)}")
            raise
            
    async def perform_dimensional_operation(self, operation: DimensionalOperation) -> str:
        """Perform dimensional operation."""
        try:
            # Generate operation ID if not provided
            if not operation.operation_id:
                operation.operation_id = f"operation_{uuid.uuid4().hex[:8]}"
                
            # Set timestamp
            operation.created_at = datetime.utcnow()
            operation.status = "running"
            
            # Create dimensional operation
            self.dimensional_operations[operation.operation_id] = operation
            
            # Run operation in background
            asyncio.create_task(self._run_dimensional_operation(operation))
            
            logger.info(f"Started dimensional operation: {operation.operation_id}")
            
            return operation.operation_id
            
        except Exception as e:
            logger.error(f"Failed to perform dimensional operation: {str(e)}")
            raise
            
    async def _run_dimensional_operation(self, operation: DimensionalOperation):
        """Run dimensional operation."""
        try:
            operation_type = operation.operation_type
            
            # Simulate dimensional operation based on type
            if operation_type == DimensionalOperation.DIMENSIONAL_REDUCTION:
                operation.result = {
                    "reduced_dimensions": random.randint(2, 10),
                    "variance_explained": random.uniform(0.7, 0.95),
                    "compression_ratio": random.uniform(0.1, 0.5)
                }
            elif operation_type == DimensionalOperation.DIMENSIONAL_EXPANSION:
                operation.result = {
                    "expanded_dimensions": random.randint(10, 100),
                    "expansion_factor": random.uniform(2.0, 10.0),
                    "information_preserved": random.uniform(0.8, 0.99)
                }
            elif operation_type == DimensionalOperation.DIMENSIONAL_TRANSFORMATION:
                operation.result = {
                    "transformation_matrix": np.random.rand(3, 3).tolist(),
                    "determinant": random.uniform(0.1, 2.0),
                    "condition_number": random.uniform(1.0, 100.0)
                }
            elif operation_type == DimensionalOperation.DIMENSIONAL_PROJECTION:
                operation.result = {
                    "projection_plane": np.random.rand(3).tolist(),
                    "projection_distance": random.uniform(0.1, 10.0),
                    "distortion_factor": random.uniform(0.01, 0.5)
                }
            else:
                operation.result = {
                    "operation_completed": True,
                    "result_quality": random.uniform(0.7, 0.95)
                }
                
            # Complete operation
            operation.status = "completed"
            operation.completed_at = datetime.utcnow()
            
            logger.info(f"Completed dimensional operation: {operation.operation_id}")
            
        except Exception as e:
            logger.error(f"Failed to run dimensional operation: {str(e)}")
            operation.status = "failed"
            
    async def analyze_dimensional_space(self, analysis: DimensionalAnalysis) -> str:
        """Analyze dimensional space."""
        try:
            # Generate analysis ID if not provided
            if not analysis.analysis_id:
                analysis.analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
                
            # Set timestamp
            analysis.created_at = datetime.utcnow()
            
            # Perform dimensional analysis
            await self._perform_dimensional_analysis(analysis)
            
            # Store analysis
            self.dimensional_analyses[analysis.analysis_id] = analysis
            
            logger.info(f"Created dimensional analysis: {analysis.analysis_id}")
            
            return analysis.analysis_id
            
        except Exception as e:
            logger.error(f"Failed to analyze dimensional space: {str(e)}")
            raise
            
    async def _perform_dimensional_analysis(self, analysis: DimensionalAnalysis):
        """Perform dimensional analysis."""
        try:
            if analysis.space_id not in self.dimensional_spaces:
                raise ValueError(f"Space {analysis.space_id} not found")
                
            space = self.dimensional_spaces[analysis.space_id]
            analysis_type = analysis.analysis_type
            
            # Simulate dimensional analysis based on type
            if analysis_type == "dimensionality_analysis":
                analysis.results = {
                    "intrinsic_dimension": random.randint(2, space.dimensions),
                    "fractal_dimension": random.uniform(1.0, space.dimensions),
                    "topological_dimension": space.dimensions
                }
                analysis.accuracy = random.uniform(0.8, 0.95)
                analysis.confidence = random.uniform(0.7, 0.9)
                
            elif analysis_type == "curvature_analysis":
                analysis.results = {
                    "gaussian_curvature": random.uniform(-1, 1),
                    "mean_curvature": random.uniform(-0.5, 0.5),
                    "ricci_curvature": random.uniform(-0.3, 0.3)
                }
                analysis.accuracy = random.uniform(0.85, 0.98)
                analysis.confidence = random.uniform(0.8, 0.95)
                
            elif analysis_type == "topology_analysis":
                analysis.results = {
                    "euler_characteristic": random.randint(-10, 10),
                    "genus": random.randint(0, 5),
                    "homology_groups": [random.randint(0, 10) for _ in range(space.dimensions)]
                }
                analysis.accuracy = random.uniform(0.9, 0.99)
                analysis.confidence = random.uniform(0.85, 0.95)
                
            else:
                analysis.results = {
                    "general_analysis": "completed",
                    "complexity": random.uniform(0.5, 0.9)
                }
                analysis.accuracy = random.uniform(0.7, 0.9)
                analysis.confidence = random.uniform(0.6, 0.8)
                
            analysis.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to perform dimensional analysis: {str(e)}")
            analysis.accuracy = 0.0
            analysis.confidence = 0.0
            
    async def get_dimensional_space(self, space_id: str) -> Optional[DimensionalSpace]:
        """Get dimensional space by ID."""
        return self.dimensional_spaces.get(space_id)
        
    async def get_dimensional_object(self, object_id: str) -> Optional[DimensionalObject]:
        """Get dimensional object by ID."""
        return self.dimensional_objects.get(object_id)
        
    async def get_dimensional_operation(self, operation_id: str) -> Optional[DimensionalOperation]:
        """Get dimensional operation by ID."""
        return self.dimensional_operations.get(operation_id)
        
    async def get_dimensional_analysis(self, analysis_id: str) -> Optional[DimensionalAnalysis]:
        """Get dimensional analysis by ID."""
        return self.dimensional_analyses.get(analysis_id)
        
    async def list_dimensional_spaces(self, space_type: Optional[DimensionalType] = None) -> List[DimensionalSpace]:
        """List dimensional spaces."""
        spaces = list(self.dimensional_spaces.values())
        
        if space_type:
            spaces = [space for space in spaces if space.space_type == space_type]
            
        return spaces
        
    async def list_dimensional_objects(self, space_id: Optional[str] = None) -> List[DimensionalObject]:
        """List dimensional objects."""
        objects = list(self.dimensional_objects.values())
        
        if space_id:
            objects = [obj for obj in objects if obj.space_id == space_id]
            
        return objects
        
    async def list_dimensional_operations(self, status: Optional[str] = None) -> List[DimensionalOperation]:
        """List dimensional operations."""
        operations = list(self.dimensional_operations.values())
        
        if status:
            operations = [op for op in operations if op.status == status]
            
        return operations
        
    async def list_dimensional_analyses(self, analysis_type: Optional[str] = None) -> List[DimensionalAnalysis]:
        """List dimensional analyses."""
        analyses = list(self.dimensional_analyses.values())
        
        if analysis_type:
            analyses = [analysis for analysis in analyses if analysis.analysis_type == analysis_type]
            
        return analyses
        
    async def get_service_status(self) -> Dict[str, Any]:
        """Get dimensional engineering service status."""
        try:
            total_spaces = len(self.dimensional_spaces)
            total_objects = len(self.dimensional_objects)
            total_operations = len(self.dimensional_operations)
            total_analyses = len(self.dimensional_analyses)
            running_operations = len([op for op in self.dimensional_operations.values() if op.status == "running"])
            
            return {
                "service_status": "active",
                "total_spaces": total_spaces,
                "total_objects": total_objects,
                "total_operations": total_operations,
                "total_analyses": total_analyses,
                "running_operations": running_operations,
                "dimensional_algorithms": len(self.dimensional_algorithms),
                "space_engines": len(self.space_engines),
                "dimensional_analysis_enabled": self.dimensional_config.get("dimensional_analysis_enabled", True),
                "space_manipulation_enabled": self.dimensional_config.get("space_manipulation_enabled", True),
                "multi_dimensional_enabled": self.dimensional_config.get("multi_dimensional_enabled", True),
                "real_time_processing": self.dimensional_config.get("real_time_processing", True),
                "max_dimensions": self.dimensional_config.get("max_dimensions", 1000),
                "max_spaces": self.dimensional_config.get("max_spaces", 100),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            return {"service_status": "error", "error": str(e)}
























