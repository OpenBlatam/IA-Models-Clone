"""
Advanced Hyperdimensional Computing System

This module provides comprehensive hyperdimensional computing capabilities
for the refactored HeyGen AI system with high-dimensional vector operations,
symbolic reasoning, and brain-inspired computing paradigms.
"""

import asyncio
import json
import logging
import uuid
import time
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import redis
import threading
from collections import defaultdict, deque
import yaml
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import requests
import websockets
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class HyperdimensionalOperation(str, Enum):
    """Hyperdimensional operations."""
    BINDING = "binding"
    BUNDLING = "bundling"
    UNBINDING = "unbinding"
    SIMILARITY = "similarity"
    ROTATION = "rotation"
    PERMUTATION = "permutation"
    THRESHOLD = "threshold"
    CLEANUP = "cleanup"


class SymbolicReasoning(str, Enum):
    """Symbolic reasoning types."""
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    LOGICAL = "logical"
    PROBABILISTIC = "probabilistic"
    FUZZY = "fuzzy"
    NEURAL_SYMBOLIC = "neural_symbolic"


@dataclass
class HyperdimensionalVector:
    """Hyperdimensional vector structure."""
    vector_id: str
    dimension: int
    values: np.ndarray
    norm: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolicConcept:
    """Symbolic concept structure."""
    concept_id: str
    name: str
    vector: HyperdimensionalVector
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HyperdimensionalMemory:
    """Hyperdimensional memory structure."""
    memory_id: str
    name: str
    capacity: int
    vectors: List[HyperdimensionalVector] = field(default_factory=list)
    similarity_threshold: float = 0.8
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HyperdimensionalOperations:
    """Hyperdimensional computing operations."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.random_state = np.random.RandomState(42)
    
    def generate_random_vector(self) -> np.ndarray:
        """Generate random hyperdimensional vector."""
        return self.random_state.choice([-1, 1], size=self.dimension)
    
    def binding(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Binding operation (element-wise multiplication)."""
        return v1 * v2
    
    def bundling(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundling operation (sum and threshold)."""
        if not vectors:
            return np.zeros(self.dimension)
        
        result = np.sum(vectors, axis=0)
        return np.sign(result)
    
    def unbinding(self, bound_vector: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbinding operation (inverse of binding)."""
        return bound_vector * key
    
    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def rotation(self, vector: np.ndarray, positions: int = 1) -> np.ndarray:
        """Rotation operation (circular shift)."""
        return np.roll(vector, positions)
    
    def permutation(self, vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        """Permutation operation."""
        return vector[permutation]
    
    def threshold(self, vector: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Threshold operation."""
        return np.where(vector > threshold, 1, -1)
    
    def cleanup(self, noisy_vector: np.ndarray, memory: List[np.ndarray]) -> np.ndarray:
        """Cleanup operation (find most similar vector in memory)."""
        if not memory:
            return noisy_vector
        
        similarities = [self.similarity(noisy_vector, mem_vec) for mem_vec in memory]
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] > 0.5:  # Threshold for cleanup
            return memory[best_match_idx]
        else:
            return noisy_vector


class SymbolicReasoningEngine:
    """Symbolic reasoning engine using hyperdimensional vectors."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.hd_ops = HyperdimensionalOperations(dimension)
        self.concepts: Dict[str, SymbolicConcept] = {}
        self.relationships: Dict[str, List[str]] = defaultdict(list)
    
    def create_concept(self, name: str, properties: Dict[str, Any] = None) -> SymbolicConcept:
        """Create a new symbolic concept."""
        concept_id = str(uuid.uuid4())
        vector = HyperdimensionalVector(
            vector_id=str(uuid.uuid4()),
            dimension=self.dimension,
            values=self.hd_ops.generate_random_vector()
        )
        
        concept = SymbolicConcept(
            concept_id=concept_id,
            name=name,
            vector=vector,
            properties=properties or {}
        )
        
        self.concepts[concept_id] = concept
        return concept
    
    def bind_concepts(self, concept1_id: str, concept2_id: str) -> SymbolicConcept:
        """Bind two concepts together."""
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            raise ValueError("One or both concepts not found")
        
        concept1 = self.concepts[concept1_id]
        concept2 = self.concepts[concept2_id]
        
        # Create bound concept
        bound_vector = self.hd_ops.binding(concept1.vector.values, concept2.vector.values)
        bound_concept = SymbolicConcept(
            concept_id=str(uuid.uuid4()),
            name=f"{concept1.name}_bound_{concept2.name}",
            vector=HyperdimensionalVector(
                vector_id=str(uuid.uuid4()),
                dimension=self.dimension,
                values=bound_vector
            ),
            properties={
                'bound_from': [concept1_id, concept2_id],
                'operation': 'binding'
            }
        )
        
        self.concepts[bound_concept.concept_id] = bound_concept
        return bound_concept
    
    def find_similar_concepts(self, query_concept_id: str, threshold: float = 0.7) -> List[SymbolicConcept]:
        """Find concepts similar to the query concept."""
        if query_concept_id not in self.concepts:
            return []
        
        query_concept = self.concepts[query_concept_id]
        similar_concepts = []
        
        for concept in self.concepts.values():
            if concept.concept_id != query_concept_id:
                similarity = self.hd_ops.similarity(
                    query_concept.vector.values,
                    concept.vector.values
                )
                if similarity >= threshold:
                    similar_concepts.append((concept, similarity))
        
        # Sort by similarity
        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in similar_concepts]
    
    def analogical_reasoning(self, source_concept_id: str, target_concept_id: str) -> Dict[str, Any]:
        """Perform analogical reasoning between concepts."""
        if source_concept_id not in self.concepts or target_concept_id not in self.concepts:
            return {}
        
        source_concept = self.concepts[source_concept_id]
        target_concept = self.concepts[target_concept_id]
        
        # Calculate analogical mapping
        source_properties = source_concept.properties
        target_properties = target_concept.properties
        
        analogical_mapping = {}
        for prop in source_properties:
            if prop in target_properties:
                analogical_mapping[prop] = {
                    'source_value': source_properties[prop],
                    'target_value': target_properties[prop],
                    'similarity': self.hd_ops.similarity(
                        np.array(source_properties[prop]) if isinstance(source_properties[prop], (list, np.ndarray)) else np.array([source_properties[prop]]),
                        np.array(target_properties[prop]) if isinstance(target_properties[prop], (list, np.ndarray)) else np.array([target_properties[prop]])
                    )
                }
        
        return {
            'source_concept': source_concept.name,
            'target_concept': target_concept.name,
            'analogical_mapping': analogical_mapping,
            'overall_similarity': self.hd_ops.similarity(
                source_concept.vector.values,
                target_concept.vector.values
            )
        }
    
    def causal_reasoning(self, cause_concept_id: str, effect_concept_id: str) -> Dict[str, Any]:
        """Perform causal reasoning between concepts."""
        if cause_concept_id not in self.concepts or effect_concept_id not in self.concepts:
            return {}
        
        cause_concept = self.concepts[cause_concept_id]
        effect_concept = self.concepts[effect_concept_id]
        
        # Calculate causal strength
        causal_strength = self.hd_ops.similarity(
            cause_concept.vector.values,
            effect_concept.vector.values
        )
        
        return {
            'cause_concept': cause_concept.name,
            'effect_concept': effect_concept.name,
            'causal_strength': causal_strength,
            'is_causal': causal_strength > 0.5
        }


class HyperdimensionalMemorySystem:
    """Hyperdimensional memory system for storing and retrieving information."""
    
    def __init__(self, dimension: int = 10000, capacity: int = 1000):
        self.dimension = dimension
        self.capacity = capacity
        self.hd_ops = HyperdimensionalOperations(dimension)
        self.memories: List[HyperdimensionalVector] = []
        self.memory_index: Dict[str, int] = {}
    
    def store_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store a memory in hyperdimensional space."""
        # Convert content to hyperdimensional vector
        content_vector = self._text_to_hd_vector(content)
        
        memory_id = str(uuid.uuid4())
        memory = HyperdimensionalVector(
            vector_id=memory_id,
            dimension=self.dimension,
            values=content_vector,
            metadata=metadata or {}
        )
        
        # Store memory
        if len(self.memories) >= self.capacity:
            # Remove oldest memory
            oldest_memory = self.memories.pop(0)
            if oldest_memory.vector_id in self.memory_index:
                del self.memory_index[oldest_memory.vector_id]
        
        self.memories.append(memory)
        self.memory_index[memory_id] = len(self.memories) - 1
        
        return memory_id
    
    def retrieve_memory(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve memories similar to the query."""
        query_vector = self._text_to_hd_vector(query)
        
        similar_memories = []
        for memory in self.memories:
            similarity = self.hd_ops.similarity(query_vector, memory.values)
            if similarity >= threshold:
                similar_memories.append({
                    'memory_id': memory.vector_id,
                    'similarity': similarity,
                    'metadata': memory.metadata
                })
        
        # Sort by similarity
        similar_memories.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_memories
    
    def _text_to_hd_vector(self, text: str) -> np.ndarray:
        """Convert text to hyperdimensional vector."""
        # Simple text-to-vector conversion
        # In practice, this would use more sophisticated methods
        words = text.lower().split()
        vector = np.zeros(self.dimension)
        
        for word in words:
            # Create word vector using hash
            word_hash = hash(word) % self.dimension
            vector[word_hash] = 1
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class AdvancedHyperdimensionalComputingSystem:
    """
    Advanced hyperdimensional computing system with comprehensive capabilities.
    
    Features:
    - Hyperdimensional vector operations
    - Symbolic reasoning and concept manipulation
    - Memory systems and information retrieval
    - Analogical and causal reasoning
    - High-dimensional data processing
    - Brain-inspired computing paradigms
    - Scalable vector operations
    - Real-time reasoning
    """
    
    def __init__(
        self,
        dimension: int = 10000,
        database_path: str = "hyperdimensional_computing.db",
        redis_url: str = None
    ):
        """
        Initialize the advanced hyperdimensional computing system.
        
        Args:
            dimension: Dimension of hyperdimensional vectors
            database_path: SQLite database path
            redis_url: Redis URL for caching
        """
        self.dimension = dimension
        self.database_path = database_path
        self.redis_url = redis_url
        
        # Initialize components
        self.hd_ops = HyperdimensionalOperations(dimension)
        self.reasoning_engine = SymbolicReasoningEngine(dimension)
        self.memory_system = HyperdimensionalMemorySystem(dimension)
        
        # Initialize Redis client
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
        
        # Initialize database
        self._initialize_database()
        
        # System state
        self.active_vectors: Dict[str, HyperdimensionalVector] = {}
        self.active_concepts: Dict[str, SymbolicConcept] = {}
        self.active_memories: Dict[str, HyperdimensionalMemory] = {}
        
        # Initialize metrics
        self.metrics = {
            'vectors_created': Counter('hd_vectors_created_total', 'Total hyperdimensional vectors created'),
            'concepts_created': Counter('hd_concepts_created_total', 'Total symbolic concepts created'),
            'memories_stored': Counter('hd_memories_stored_total', 'Total memories stored'),
            'operations_performed': Counter('hd_operations_performed_total', 'Total operations performed', ['operation']),
            'reasoning_queries': Counter('hd_reasoning_queries_total', 'Total reasoning queries', ['reasoning_type']),
            'memory_retrievals': Counter('hd_memory_retrievals_total', 'Total memory retrievals'),
            'similarity_calculations': Histogram('hd_similarity_calculations', 'Similarity calculation time'),
            'active_vectors': Gauge('active_hd_vectors', 'Currently active vectors'),
            'active_concepts': Gauge('active_hd_concepts', 'Currently active concepts')
        }
        
        logger.info("Advanced hyperdimensional computing system initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hyperdimensional_vectors (
                    vector_id TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    values BLOB NOT NULL,
                    norm REAL DEFAULT 0.0,
                    metadata TEXT,
                    created_at DATETIME NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS symbolic_concepts (
                    concept_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    vector_id TEXT NOT NULL,
                    properties TEXT,
                    relationships TEXT,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY (vector_id) REFERENCES hyperdimensional_vectors (vector_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hyperdimensional_memories (
                    memory_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    similarity_threshold REAL DEFAULT 0.8,
                    created_at DATETIME NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def create_vector(self, values: np.ndarray = None, metadata: Dict[str, Any] = None) -> HyperdimensionalVector:
        """Create a new hyperdimensional vector."""
        try:
            if values is None:
                values = self.hd_ops.generate_random_vector()
            
            vector_id = str(uuid.uuid4())
            vector = HyperdimensionalVector(
                vector_id=vector_id,
                dimension=self.dimension,
                values=values,
                norm=np.linalg.norm(values),
                metadata=metadata or {}
            )
            
            # Store vector
            self.active_vectors[vector_id] = vector
            await self._store_vector(vector)
            
            # Update metrics
            self.metrics['vectors_created'].inc()
            self.metrics['active_vectors'].inc()
            
            logger.info(f"Hyperdimensional vector {vector_id} created successfully")
            return vector
            
        except Exception as e:
            logger.error(f"Vector creation error: {e}")
            raise
    
    async def create_concept(self, name: str, properties: Dict[str, Any] = None) -> SymbolicConcept:
        """Create a new symbolic concept."""
        try:
            concept = self.reasoning_engine.create_concept(name, properties)
            
            # Store concept
            self.active_concepts[concept.concept_id] = concept
            await self._store_concept(concept)
            
            # Update metrics
            self.metrics['concepts_created'].inc()
            self.metrics['active_concepts'].inc()
            
            logger.info(f"Symbolic concept {concept.concept_id} created successfully")
            return concept
            
        except Exception as e:
            logger.error(f"Concept creation error: {e}")
            raise
    
    async def perform_operation(
        self, 
        operation: HyperdimensionalOperation, 
        vectors: List[np.ndarray], 
        **kwargs
    ) -> np.ndarray:
        """Perform hyperdimensional operation."""
        try:
            start_time = time.time()
            
            if operation == HyperdimensionalOperation.BINDING:
                if len(vectors) != 2:
                    raise ValueError("Binding requires exactly 2 vectors")
                result = self.hd_ops.binding(vectors[0], vectors[1])
            
            elif operation == HyperdimensionalOperation.BUNDLING:
                result = self.hd_ops.bundling(vectors)
            
            elif operation == HyperdimensionalOperation.UNBINDING:
                if len(vectors) != 2:
                    raise ValueError("Unbinding requires exactly 2 vectors")
                result = self.hd_ops.unbinding(vectors[0], vectors[1])
            
            elif operation == HyperdimensionalOperation.SIMILARITY:
                if len(vectors) != 2:
                    raise ValueError("Similarity requires exactly 2 vectors")
                result = np.array([self.hd_ops.similarity(vectors[0], vectors[1])])
            
            elif operation == HyperdimensionalOperation.ROTATION:
                if len(vectors) != 1:
                    raise ValueError("Rotation requires exactly 1 vector")
                positions = kwargs.get('positions', 1)
                result = self.hd_ops.rotation(vectors[0], positions)
            
            elif operation == HyperdimensionalOperation.PERMUTATION:
                if len(vectors) != 1:
                    raise ValueError("Permutation requires exactly 1 vector")
                permutation = kwargs.get('permutation', np.arange(self.dimension))
                result = self.hd_ops.permutation(vectors[0], permutation)
            
            elif operation == HyperdimensionalOperation.THRESHOLD:
                if len(vectors) != 1:
                    raise ValueError("Threshold requires exactly 1 vector")
                threshold = kwargs.get('threshold', 0.0)
                result = self.hd_ops.threshold(vectors[0], threshold)
            
            elif operation == HyperdimensionalOperation.CLEANUP:
                if len(vectors) != 1:
                    raise ValueError("Cleanup requires exactly 1 vector")
                memory = kwargs.get('memory', [])
                result = self.hd_ops.cleanup(vectors[0], memory)
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            operation_time = time.time() - start_time
            
            # Update metrics
            self.metrics['operations_performed'].labels(operation=operation.value).inc()
            if operation == HyperdimensionalOperation.SIMILARITY:
                self.metrics['similarity_calculations'].observe(operation_time)
            
            logger.info(f"Operation {operation.value} completed in {operation_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Operation {operation.value} error: {e}")
            raise
    
    async def store_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store memory in hyperdimensional space."""
        try:
            memory_id = self.memory_system.store_memory(content, metadata)
            
            # Update metrics
            self.metrics['memories_stored'].inc()
            
            logger.info(f"Memory {memory_id} stored successfully")
            return memory_id
            
        except Exception as e:
            logger.error(f"Memory storage error: {e}")
            raise
    
    async def retrieve_memory(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve memories similar to query."""
        try:
            memories = self.memory_system.retrieve_memory(query, threshold)
            
            # Update metrics
            self.metrics['memory_retrievals'].inc()
            
            logger.info(f"Retrieved {len(memories)} memories for query: {query}")
            return memories
            
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []
    
    async def perform_reasoning(
        self, 
        reasoning_type: SymbolicReasoning, 
        concept_ids: List[str], 
        **kwargs
    ) -> Dict[str, Any]:
        """Perform symbolic reasoning."""
        try:
            start_time = time.time()
            
            if reasoning_type == SymbolicReasoning.ANALOGICAL:
                if len(concept_ids) != 2:
                    raise ValueError("Analogical reasoning requires exactly 2 concepts")
                result = self.reasoning_engine.analogical_reasoning(concept_ids[0], concept_ids[1])
            
            elif reasoning_type == SymbolicReasoning.CAUSAL:
                if len(concept_ids) != 2:
                    raise ValueError("Causal reasoning requires exactly 2 concepts")
                result = self.reasoning_engine.causal_reasoning(concept_ids[0], concept_ids[1])
            
            elif reasoning_type == SymbolicReasoning.TEMPORAL:
                # Temporal reasoning implementation
                result = {'reasoning_type': 'temporal', 'concepts': concept_ids}
            
            elif reasoning_type == SymbolicReasoning.SPATIAL:
                # Spatial reasoning implementation
                result = {'reasoning_type': 'spatial', 'concepts': concept_ids}
            
            elif reasoning_type == SymbolicReasoning.LOGICAL:
                # Logical reasoning implementation
                result = {'reasoning_type': 'logical', 'concepts': concept_ids}
            
            elif reasoning_type == SymbolicReasoning.PROBABILISTIC:
                # Probabilistic reasoning implementation
                result = {'reasoning_type': 'probabilistic', 'concepts': concept_ids}
            
            elif reasoning_type == SymbolicReasoning.FUZZY:
                # Fuzzy reasoning implementation
                result = {'reasoning_type': 'fuzzy', 'concepts': concept_ids}
            
            elif reasoning_type == SymbolicReasoning.NEURAL_SYMBOLIC:
                # Neural-symbolic reasoning implementation
                result = {'reasoning_type': 'neural_symbolic', 'concepts': concept_ids}
            
            else:
                raise ValueError(f"Unsupported reasoning type: {reasoning_type}")
            
            reasoning_time = time.time() - start_time
            
            # Update metrics
            self.metrics['reasoning_queries'].labels(reasoning_type=reasoning_type.value).inc()
            
            logger.info(f"Reasoning {reasoning_type.value} completed in {reasoning_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Reasoning {reasoning_type.value} error: {e}")
            return {}
    
    async def _store_vector(self, vector: HyperdimensionalVector):
        """Store vector in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO hyperdimensional_vectors
                (vector_id, dimension, values, norm, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                vector.vector_id,
                vector.dimension,
                vector.values.tobytes(),
                vector.norm,
                json.dumps(vector.metadata),
                vector.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
    
    async def _store_concept(self, concept: SymbolicConcept):
        """Store concept in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO symbolic_concepts
                (concept_id, name, vector_id, properties, relationships, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                concept.concept_id,
                concept.name,
                concept.vector.vector_id,
                json.dumps(concept.properties),
                json.dumps(concept.relationships),
                concept.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing concept: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'total_vectors': len(self.active_vectors),
            'total_concepts': len(self.active_concepts),
            'total_memories': len(self.memory_system.memories),
            'dimension': self.dimension,
            'memory_capacity': self.memory_system.capacity,
            'memory_usage': len(self.memory_system.memories) / self.memory_system.capacity
        }


# Example usage and demonstration
async def main():
    """Demonstrate the advanced hyperdimensional computing system."""
    print("🔮 HeyGen AI - Advanced Hyperdimensional Computing System Demo")
    print("=" * 70)
    
    # Initialize hyperdimensional computing system
    hd_system = AdvancedHyperdimensionalComputingSystem(
        dimension=10000,
        database_path="hyperdimensional_computing.db",
        redis_url="redis://localhost:6379/0"
    )
    
    try:
        # Create hyperdimensional vectors
        print("\n📐 Creating Hyperdimensional Vectors...")
        
        vectors = []
        for i in range(5):
            vector = await hd_system.create_vector(
                metadata={'index': i, 'type': 'test_vector'}
            )
            vectors.append(vector)
            print(f"  Vector {i+1}: {vector.vector_id} (norm: {vector.norm:.4f})")
        
        # Test hyperdimensional operations
        print("\n⚡ Testing Hyperdimensional Operations...")
        
        # Binding operation
        if len(vectors) >= 2:
            binding_result = await hd_system.perform_operation(
                HyperdimensionalOperation.BINDING,
                [vectors[0].values, vectors[1].values]
            )
            print(f"  Binding result shape: {binding_result.shape}")
        
        # Bundling operation
        bundling_result = await hd_system.perform_operation(
            HyperdimensionalOperation.BUNDLING,
            [v.values for v in vectors]
        )
        print(f"  Bundling result shape: {bundling_result.shape}")
        
        # Similarity operation
        if len(vectors) >= 2:
            similarity_result = await hd_system.perform_operation(
                HyperdimensionalOperation.SIMILARITY,
                [vectors[0].values, vectors[1].values]
            )
            print(f"  Similarity result: {similarity_result[0]:.4f}")
        
        # Rotation operation
        rotation_result = await hd_system.perform_operation(
            HyperdimensionalOperation.ROTATION,
            [vectors[0].values],
            positions=5
        )
        print(f"  Rotation result shape: {rotation_result.shape}")
        
        # Create symbolic concepts
        print("\n🧠 Creating Symbolic Concepts...")
        
        concepts = []
        concept_names = ['cat', 'dog', 'bird', 'fish', 'mammal']
        
        for name in concept_names:
            concept = await hd_system.create_concept(
                name=name,
                properties={
                    'type': 'animal',
                    'legs': 4 if name in ['cat', 'dog', 'mammal'] else 2,
                    'habitat': 'land' if name in ['cat', 'dog', 'mammal'] else 'air' if name == 'bird' else 'water'
                }
            )
            concepts.append(concept)
            print(f"  Concept: {concept.name} ({concept.concept_id})")
        
        # Test symbolic reasoning
        print("\n🔍 Testing Symbolic Reasoning...")
        
        # Analogical reasoning
        if len(concepts) >= 2:
            analogical_result = await hd_system.perform_reasoning(
                SymbolicReasoning.ANALOGICAL,
                [concepts[0].concept_id, concepts[1].concept_id]
            )
            print(f"  Analogical reasoning: {analogical_result}")
        
        # Causal reasoning
        if len(concepts) >= 2:
            causal_result = await hd_system.perform_reasoning(
                SymbolicReasoning.CAUSAL,
                [concepts[0].concept_id, concepts[1].concept_id]
            )
            print(f"  Causal reasoning: {causal_result}")
        
        # Test memory system
        print("\n💾 Testing Memory System...")
        
        # Store memories
        memories = [
            "The cat is sleeping on the couch",
            "The dog is playing in the garden",
            "The bird is singing in the tree",
            "The fish is swimming in the pond",
            "Mammals are warm-blooded animals"
        ]
        
        memory_ids = []
        for memory in memories:
            memory_id = await hd_system.store_memory(
                memory,
                metadata={'type': 'example_memory'}
            )
            memory_ids.append(memory_id)
            print(f"  Stored memory: {memory}")
        
        # Retrieve memories
        print("\n🔍 Testing Memory Retrieval...")
        
        queries = ["cat", "dog", "animals", "water"]
        for query in queries:
            retrieved_memories = await hd_system.retrieve_memory(query, threshold=0.3)
            print(f"  Query '{query}': {len(retrieved_memories)} memories found")
            for memory in retrieved_memories[:3]:  # Show top 3
                print(f"    Similarity: {memory['similarity']:.4f}")
        
        # Test concept binding
        print("\n🔗 Testing Concept Binding...")
        
        if len(concepts) >= 2:
            bound_concept = hd_system.reasoning_engine.bind_concepts(
                concepts[0].concept_id,
                concepts[1].concept_id
            )
            print(f"  Bound concept: {bound_concept.name}")
            print(f"  Properties: {bound_concept.properties}")
        
        # Test similarity search
        print("\n🔍 Testing Similarity Search...")
        
        if len(concepts) >= 1:
            similar_concepts = hd_system.reasoning_engine.find_similar_concepts(
                concepts[0].concept_id,
                threshold=0.1
            )
            print(f"  Similar to '{concepts[0].name}': {len(similar_concepts)} concepts")
            for concept in similar_concepts[:3]:  # Show top 3
                print(f"    {concept.name}")
        
        # Get system metrics
        print("\n📊 System Metrics:")
        metrics = hd_system.get_system_metrics()
        print(f"  Total Vectors: {metrics['total_vectors']}")
        print(f"  Total Concepts: {metrics['total_concepts']}")
        print(f"  Total Memories: {metrics['total_memories']}")
        print(f"  Dimension: {metrics['dimension']}")
        print(f"  Memory Capacity: {metrics['memory_capacity']}")
        print(f"  Memory Usage: {metrics['memory_usage']:.2%}")
        
        # Test high-dimensional operations
        print("\n🚀 Testing High-Dimensional Operations...")
        
        # Create large vectors
        large_vectors = []
        for i in range(3):
            large_vector = await hd_system.create_vector(
                metadata={'size': 'large', 'index': i}
            )
            large_vectors.append(large_vector)
        
        # Test operations on large vectors
        start_time = time.time()
        large_bundling = await hd_system.perform_operation(
            HyperdimensionalOperation.BUNDLING,
            [v.values for v in large_vectors]
        )
        operation_time = time.time() - start_time
        print(f"  Large vector bundling: {operation_time:.4f}s")
        print(f"  Result shape: {large_bundling.shape}")
        
        print(f"\n🌐 Hyperdimensional Computing Dashboard available at: http://localhost:8080/hyperdimensional")
        print(f"📊 Hyperdimensional Computing API available at: http://localhost:8080/api/v1/hyperdimensional")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
