#!/usr/bin/env python3
"""
Advanced Neural Architecture Search (NAS) System for Frontier Model Training
Provides automated architecture discovery, optimization, and deployment.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import ray
from ray import tune
import dask
from dask.distributed import Client
import networkx as nx
import graphviz
from graphviz import Digraph
import random
import copy
from collections import defaultdict
import pickle
import joblib

console = Console()

class SearchStrategy(Enum):
    """NAS search strategies."""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    GRADIENT_BASED = "gradient_based"
    PROGRESSIVE = "progressive"
    DIFFERENTIABLE = "differentiable"

class ArchitectureType(Enum):
    """Architecture types."""
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    RESNET = "resnet"
    DENSENET = "densenet"
    EFFICIENTNET = "efficientnet"
    MOBILENET = "mobilenet"
    VISION_TRANSFORMER = "vision_transformer"
    HYBRID = "hybrid"

class OptimizationObjective(Enum):
    """Optimization objectives."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    MEMORY = "memory"
    FLOPS = "flops"
    PARAMETERS = "parameters"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class NASConfig:
    """NAS configuration."""
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    architecture_type: ArchitectureType = ArchitectureType.CNN
    optimization_objective: OptimizationObjective = OptimizationObjective.ACCURACY
    max_trials: int = 100
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    early_stopping_patience: int = 10
    max_epochs_per_trial: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    device: str = "auto"
    parallel_trials: int = 4
    enable_pruning: bool = True
    enable_quantization: bool = False
    enable_distillation: bool = False

@dataclass
class Architecture:
    """Neural architecture representation."""
    architecture_id: str
    name: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    parameters: Dict[str, Any]
    created_at: datetime
    performance_metrics: Dict[str, float] = None
    complexity_metrics: Dict[str, float] = None

@dataclass
class SearchResult:
    """NAS search result."""
    result_id: str
    best_architecture: Architecture
    search_history: List[Architecture]
    optimization_curve: List[float]
    search_time: float
    total_trials: int
    success_rate: float
    created_at: datetime

class ArchitectureBuilder:
    """Neural architecture builder."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Architecture components
        self.layer_types = self._init_layer_types()
        self.activation_functions = ['relu', 'leaky_relu', 'gelu', 'swish', 'mish']
        self.optimizers = ['adam', 'sgd', 'adamw', 'rmsprop']
        
        # Search space
        self.search_space = self._init_search_space()
    
    def _init_layer_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available layer types."""
        return {
            'conv2d': {
                'params': ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding'],
                'ranges': {
                    'out_channels': (16, 512),
                    'kernel_size': [1, 3, 5, 7],
                    'stride': [1, 2],
                    'padding': [0, 1, 2]
                }
            },
            'maxpool2d': {
                'params': ['kernel_size', 'stride'],
                'ranges': {
                    'kernel_size': [2, 3, 4],
                    'stride': [1, 2]
                }
            },
            'avgpool2d': {
                'params': ['kernel_size', 'stride'],
                'ranges': {
                    'kernel_size': [2, 3, 4],
                    'stride': [1, 2]
                }
            },
            'linear': {
                'params': ['in_features', 'out_features'],
                'ranges': {
                    'out_features': (64, 2048)
                }
            },
            'dropout': {
                'params': ['p'],
                'ranges': {
                    'p': (0.1, 0.5)
                }
            },
            'batchnorm2d': {
                'params': ['num_features'],
                'ranges': {}
            },
            'layernorm': {
                'params': ['normalized_shape'],
                'ranges': {}
            }
        }
    
    def _init_search_space(self) -> Dict[str, Any]:
        """Initialize search space."""
        return {
            'max_layers': 20,
            'min_layers': 3,
            'layer_types': list(self.layer_types.keys()),
            'activation_functions': self.activation_functions,
            'optimizers': self.optimizers,
            'learning_rates': [0.001, 0.01, 0.1],
            'batch_sizes': [16, 32, 64, 128],
            'weight_decays': [1e-5, 1e-4, 1e-3]
        }
    
    def generate_random_architecture(self, num_layers: int = None) -> Architecture:
        """Generate random architecture."""
        if num_layers is None:
            num_layers = random.randint(
                self.search_space['min_layers'],
                self.search_space['max_layers']
            )
        
        architecture_id = f"arch_{int(time.time())}_{random.randint(1000, 9999)}"
        
        layers = []
        connections = []
        
        # Generate layers
        for i in range(num_layers):
            layer_type = random.choice(self.search_space['layer_types'])
            layer_config = self._generate_layer_config(layer_type, i, layers)
            layers.append(layer_config)
        
        # Generate connections
        connections = self._generate_connections(layers)
        
        # Generate parameters
        parameters = self._generate_parameters()
        
        return Architecture(
            architecture_id=architecture_id,
            name=f"Random_Arch_{num_layers}L",
            layers=layers,
            connections=connections,
            parameters=parameters,
            created_at=datetime.now()
        )
    
    def _generate_layer_config(self, layer_type: str, layer_idx: int, 
                              existing_layers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate configuration for a layer."""
        layer_config = {
            'type': layer_type,
            'id': layer_idx,
            'activation': random.choice(self.search_space['activation_functions'])
        }
        
        # Add layer-specific parameters
        if layer_type in self.layer_types:
            layer_spec = self.layer_types[layer_type]
            for param in layer_spec['params']:
                if param in layer_spec['ranges']:
                    ranges = layer_spec['ranges'][param]
                    if isinstance(ranges, tuple):
                        layer_config[param] = random.randint(ranges[0], ranges[1])
                    elif isinstance(ranges, list):
                        layer_config[param] = random.choice(ranges)
                    else:
                        layer_config[param] = random.uniform(ranges[0], ranges[1])
        
        return layer_config
    
    def _generate_connections(self, layers: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Generate connections between layers."""
        connections = []
        
        # Create sequential connections
        for i in range(len(layers) - 1):
            connections.append((i, i + 1))
        
        # Add some random skip connections
        num_skip_connections = random.randint(0, len(layers) // 3)
        for _ in range(num_skip_connections):
            from_layer = random.randint(0, len(layers) - 2)
            to_layer = random.randint(from_layer + 1, len(layers) - 1)
            if (from_layer, to_layer) not in connections:
                connections.append((from_layer, to_layer))
        
        return connections
    
    def _generate_parameters(self) -> Dict[str, Any]:
        """Generate training parameters."""
        return {
            'optimizer': random.choice(self.search_space['optimizers']),
            'learning_rate': random.choice(self.search_space['learning_rates']),
            'batch_size': random.choice(self.search_space['batch_sizes']),
            'weight_decay': random.choice(self.search_space['weight_decays']),
            'epochs': self.config.max_epochs_per_trial
        }
    
    def mutate_architecture(self, architecture: Architecture) -> Architecture:
        """Mutate an architecture."""
        mutated = copy.deepcopy(architecture)
        mutated.architecture_id = f"mut_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Random mutation operations
        mutation_ops = [
            self._add_layer,
            self._remove_layer,
            self._modify_layer,
            self._add_connection,
            self._remove_connection,
            self._modify_parameters
        ]
        
        # Apply random mutations
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            mutation_op = random.choice(mutation_ops)
            try:
                mutation_op(mutated)
            except Exception as e:
                self.logger.warning(f"Mutation failed: {e}")
        
        return mutated
    
    def _add_layer(self, architecture: Architecture):
        """Add a random layer to architecture."""
        if len(architecture.layers) >= self.search_space['max_layers']:
            return
        
        # Choose insertion point
        insert_idx = random.randint(0, len(architecture.layers))
        
        # Generate new layer
        layer_type = random.choice(self.search_space['layer_types'])
        new_layer = self._generate_layer_config(layer_type, len(architecture.layers), architecture.layers)
        new_layer['id'] = len(architecture.layers)
        
        # Insert layer
        architecture.layers.insert(insert_idx, new_layer)
        
        # Update layer IDs
        for i, layer in enumerate(architecture.layers):
            layer['id'] = i
        
        # Update connections
        architecture.connections = self._update_connections_after_insertion(
            architecture.connections, insert_idx
        )
    
    def _remove_layer(self, architecture: Architecture):
        """Remove a random layer from architecture."""
        if len(architecture.layers) <= self.search_space['min_layers']:
            return
        
        # Choose layer to remove
        remove_idx = random.randint(0, len(architecture.layers) - 1)
        
        # Remove layer
        architecture.layers.pop(remove_idx)
        
        # Update layer IDs
        for i, layer in enumerate(architecture.layers):
            layer['id'] = i
        
        # Update connections
        architecture.connections = self._update_connections_after_removal(
            architecture.connections, remove_idx
        )
    
    def _modify_layer(self, architecture: Architecture):
        """Modify a random layer in architecture."""
        if not architecture.layers:
            return
        
        layer_idx = random.randint(0, len(architecture.layers) - 1)
        layer = architecture.layers[layer_idx]
        
        # Modify layer parameters
        if layer['type'] in self.layer_types:
            layer_spec = self.layer_types[layer['type']]
            for param in layer_spec['params']:
                if param in layer_spec['ranges'] and random.random() < 0.3:
                    ranges = layer_spec['ranges'][param]
                    if isinstance(ranges, tuple):
                        layer[param] = random.randint(ranges[0], ranges[1])
                    elif isinstance(ranges, list):
                        layer[param] = random.choice(ranges)
                    else:
                        layer[param] = random.uniform(ranges[0], ranges[1])
    
    def _add_connection(self, architecture: Architecture):
        """Add a random connection to architecture."""
        if len(architecture.layers) < 2:
            return
        
        # Find valid connection
        max_attempts = 10
        for _ in range(max_attempts):
            from_layer = random.randint(0, len(architecture.layers) - 2)
            to_layer = random.randint(from_layer + 1, len(architecture.layers) - 1)
            
            if (from_layer, to_layer) not in architecture.connections:
                architecture.connections.append((from_layer, to_layer))
                break
    
    def _remove_connection(self, architecture: Architecture):
        """Remove a random connection from architecture."""
        if not architecture.connections:
            return
        
        # Don't remove sequential connections
        removable_connections = [
            conn for conn in architecture.connections
            if conn[1] - conn[0] > 1
        ]
        
        if removable_connections:
            conn_to_remove = random.choice(removable_connections)
            architecture.connections.remove(conn_to_remove)
    
    def _modify_parameters(self, architecture: Architecture):
        """Modify training parameters."""
        if random.random() < 0.3:
            architecture.parameters['learning_rate'] = random.choice(self.search_space['learning_rates'])
        if random.random() < 0.3:
            architecture.parameters['batch_size'] = random.choice(self.search_space['batch_sizes'])
        if random.random() < 0.3:
            architecture.parameters['optimizer'] = random.choice(self.search_space['optimizers'])
    
    def _update_connections_after_insertion(self, connections: List[Tuple[int, int]], 
                                          insert_idx: int) -> List[Tuple[int, int]]:
        """Update connections after layer insertion."""
        updated_connections = []
        for from_layer, to_layer in connections:
            if from_layer >= insert_idx:
                from_layer += 1
            if to_layer >= insert_idx:
                to_layer += 1
            updated_connections.append((from_layer, to_layer))
        return updated_connections
    
    def _update_connections_after_removal(self, connections: List[Tuple[int, int]], 
                                        remove_idx: int) -> List[Tuple[int, int]]:
        """Update connections after layer removal."""
        updated_connections = []
        for from_layer, to_layer in connections:
            if from_layer > remove_idx:
                from_layer -= 1
            if to_layer > remove_idx:
                to_layer -= 1
            if from_layer != remove_idx and to_layer != remove_idx:
                updated_connections.append((from_layer, to_layer))
        return updated_connections
    
    def crossover_architectures(self, parent1: Architecture, parent2: Architecture) -> Tuple[Architecture, Architecture]:
        """Perform crossover between two architectures."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        child1.architecture_id = f"cross_{int(time.time())}_{random.randint(1000, 9999)}"
        child2.architecture_id = f"cross_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Crossover layers
        if len(parent1.layers) > 1 and len(parent2.layers) > 1:
            crossover_point = random.randint(1, min(len(parent1.layers), len(parent2.layers)) - 1)
            
            # Swap layer configurations
            child1.layers[crossover_point:] = parent2.layers[crossover_point:]
            child2.layers[crossover_point:] = parent1.layers[crossover_point:]
        
        # Crossover connections
        if parent1.connections and parent2.connections:
            crossover_point = random.randint(0, min(len(parent1.connections), len(parent2.connections)) - 1)
            
            child1.connections = parent1.connections[:crossover_point] + parent2.connections[crossover_point:]
            child2.connections = parent2.connections[:crossover_point] + parent1.connections[crossover_point:]
        
        # Crossover parameters
        if random.random() < 0.5:
            child1.parameters, child2.parameters = child2.parameters, child1.parameters
        
        return child1, child2

class ArchitectureEvaluator:
    """Architecture evaluator."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Evaluation cache
        self.evaluation_cache = {}
    
    def evaluate_architecture(self, architecture: Architecture, 
                           train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate an architecture."""
        # Check cache
        cache_key = self._get_cache_key(architecture)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        try:
            # Build model
            model = self._build_model(architecture)
            model = model.to(self.device)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(model)
            
            # Train and evaluate model
            performance_metrics = self._train_and_evaluate_model(
                model, architecture, train_loader, val_loader
            )
            
            # Combine metrics
            all_metrics = {**performance_metrics, **complexity_metrics}
            
            # Cache results
            self.evaluation_cache[cache_key] = all_metrics
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Architecture evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'loss': float('inf'),
                'parameters': 0,
                'flops': 0,
                'latency': float('inf'),
                'memory': 0
            }
    
    def _get_cache_key(self, architecture: Architecture) -> str:
        """Get cache key for architecture."""
        return hashlib.md5(
            json.dumps(asdict(architecture), sort_keys=True).encode()
        ).hexdigest()
    
    def _build_model(self, architecture: Architecture) -> nn.Module:
        """Build PyTorch model from architecture."""
        class NASModel(nn.Module):
            def __init__(self, arch):
                super().__init__()
                self.arch = arch
                self.layers = nn.ModuleList()
                self.connections = arch.connections
                
                # Build layers
                for layer_config in arch.layers:
                    layer = self._create_layer(layer_config)
                    self.layers.append(layer)
                
                # Add classifier
                self.classifier = nn.Linear(512, 10)  # Assuming 10 classes
            
            def _create_layer(self, layer_config):
                """Create a layer from configuration."""
                layer_type = layer_config['type']
                
                if layer_type == 'conv2d':
                    return nn.Conv2d(
                        layer_config.get('in_channels', 3),
                        layer_config.get('out_channels', 64),
                        kernel_size=layer_config.get('kernel_size', 3),
                        stride=layer_config.get('stride', 1),
                        padding=layer_config.get('padding', 1)
                    )
                elif layer_type == 'maxpool2d':
                    return nn.MaxPool2d(
                        kernel_size=layer_config.get('kernel_size', 2),
                        stride=layer_config.get('stride', 2)
                    )
                elif layer_type == 'avgpool2d':
                    return nn.AvgPool2d(
                        kernel_size=layer_config.get('kernel_size', 2),
                        stride=layer_config.get('stride', 2)
                    )
                elif layer_type == 'linear':
                    return nn.Linear(
                        layer_config.get('in_features', 512),
                        layer_config.get('out_features', 256)
                    )
                elif layer_type == 'dropout':
                    return nn.Dropout(layer_config.get('p', 0.5))
                elif layer_type == 'batchnorm2d':
                    return nn.BatchNorm2d(layer_config.get('num_features', 64))
                elif layer_type == 'layernorm':
                    return nn.LayerNorm(layer_config.get('normalized_shape', 512))
                else:
                    return nn.Identity()
            
            def forward(self, x):
                """Forward pass with skip connections."""
                layer_outputs = {}
                
                for i, layer in enumerate(self.layers):
                    # Get input for this layer
                    if i == 0:
                        layer_input = x
                    else:
                        # Collect inputs from connected layers
                        layer_inputs = []
                        for from_layer, to_layer in self.connections:
                            if to_layer == i:
                                layer_inputs.append(layer_outputs[from_layer])
                        
                        if layer_inputs:
                            layer_input = torch.cat(layer_inputs, dim=1)
                        else:
                            layer_input = layer_outputs[i-1]
                    
                    # Apply layer
                    layer_output = layer(layer_input)
                    
                    # Apply activation
                    activation = self.arch.layers[i].get('activation', 'relu')
                    if activation == 'relu':
                        layer_output = torch.relu(layer_output)
                    elif activation == 'leaky_relu':
                        layer_output = torch.leaky_relu(layer_output)
                    elif activation == 'gelu':
                        layer_output = torch.gelu(layer_output)
                    
                    layer_outputs[i] = layer_output
                
                # Global average pooling and classifier
                x = torch.adaptive_avg_pool2d(layer_outputs[len(self.layers)-1], (1, 1))
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                
                return x
        
        return NASModel(architecture)
    
    def _calculate_complexity_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Calculate model complexity metrics."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (simplified)
        flops = self._estimate_flops(model)
        
        # Estimate memory usage
        memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Estimate latency (simplified)
        latency_ms = self._estimate_latency(model)
        
        return {
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'flops': flops,
            'memory_mb': memory_mb,
            'latency_ms': latency_ms
        }
    
    def _estimate_flops(self, model: nn.Module) -> int:
        """Estimate FLOPs for the model."""
        flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPs = output_size * kernel_size^2 * in_channels * out_channels
                output_size = 32 * 32  # Assuming 32x32 input
                kernel_size = module.kernel_size[0] * module.kernel_size[1]
                flops += output_size * kernel_size * module.in_channels * module.out_channels
            elif isinstance(module, nn.Linear):
                # Linear FLOPs = in_features * out_features
                flops += module.in_features * module.out_features
        
        return flops
    
    def _estimate_latency(self, model: nn.Module) -> float:
        """Estimate inference latency."""
        # Simplified latency estimation
        total_params = sum(p.numel() for p in model.parameters())
        
        # Rough estimation: 1ms per 100K parameters
        latency_ms = total_params / 100000
        
        return min(latency_ms, 1000)  # Cap at 1 second
    
    def _train_and_evaluate_model(self, model: nn.Module, architecture: Architecture,
                                 train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train and evaluate model."""
        # Setup optimizer
        optimizer_name = architecture.parameters.get('optimizer', 'adam')
        learning_rate = architecture.parameters.get('learning_rate', 0.001)
        weight_decay = architecture.parameters.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(min(self.config.max_epochs_per_trial, 10)):  # Limit epochs for speed
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:  # Limit batches for speed
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 20:  # Limit batches for speed
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / max(batch_idx + 1, 1)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }

class NeuralArchitectureSearch:
    """Main NAS system."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.architecture_builder = ArchitectureBuilder(config)
        self.evaluator = ArchitectureEvaluator(config)
        
        # Search state
        self.population: List[Architecture] = []
        self.search_history: List[Architecture] = []
        self.best_architecture: Optional[Architecture] = None
        
        # Initialize database
        self.db_path = self._init_database()
    
    def _init_database(self) -> str:
        """Initialize NAS database."""
        db_path = Path("./nas.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS architectures (
                    architecture_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    layers TEXT NOT NULL,
                    connections TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    performance_metrics TEXT,
                    complexity_metrics TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    result_id TEXT PRIMARY KEY,
                    best_architecture_id TEXT NOT NULL,
                    search_history TEXT NOT NULL,
                    optimization_curve TEXT NOT NULL,
                    search_time REAL NOT NULL,
                    total_trials INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (best_architecture_id) REFERENCES architectures (architecture_id)
                )
            """)
        
        return str(db_path)
    
    def search(self, train_loader: DataLoader, val_loader: DataLoader) -> SearchResult:
        """Perform neural architecture search."""
        console.print(f"[blue]Starting NAS with {self.config.search_strategy.value} strategy[/blue]")
        
        start_time = time.time()
        result_id = f"nas_{int(time.time())}"
        
        # Initialize population
        self._initialize_population()
        
        # Search loop
        for generation in range(self.config.generations):
            console.print(f"[blue]Generation {generation + 1}/{self.config.generations}[/blue]")
            
            # Evaluate population
            self._evaluate_population(train_loader, val_loader)
            
            # Update best architecture
            self._update_best_architecture()
            
            # Selection and reproduction
            if self.config.search_strategy == SearchStrategy.EVOLUTIONARY:
                self._evolutionary_step()
            elif self.config.search_strategy == SearchStrategy.RANDOM:
                self._random_step()
            elif self.config.search_strategy == SearchStrategy.BAYESIAN:
                self._bayesian_step()
            
            # Log progress
            if self.best_architecture:
                best_acc = self.best_architecture.performance_metrics.get('accuracy', 0)
                console.print(f"[green]Best accuracy: {best_acc:.4f}[/green]")
        
        # Final evaluation
        search_time = time.time() - start_time
        
        # Create search result
        search_result = SearchResult(
            result_id=result_id,
            best_architecture=self.best_architecture,
            search_history=self.search_history.copy(),
            optimization_curve=self._get_optimization_curve(),
            search_time=search_time,
            total_trials=len(self.search_history),
            success_rate=self._calculate_success_rate(),
            created_at=datetime.now()
        )
        
        # Save results
        self._save_search_result(search_result)
        
        console.print(f"[green]NAS completed in {search_time:.2f} seconds[/green]")
        console.print(f"[blue]Best architecture: {self.best_architecture.architecture_id}[/blue]")
        
        return search_result
    
    def _initialize_population(self):
        """Initialize population of architectures."""
        self.population = []
        
        for i in range(self.config.population_size):
            architecture = self.architecture_builder.generate_random_architecture()
            self.population.append(architecture)
        
        console.print(f"[blue]Initialized population of {len(self.population)} architectures[/blue]")
    
    def _evaluate_population(self, train_loader: DataLoader, val_loader: DataLoader):
        """Evaluate all architectures in population."""
        for architecture in self.population:
            if architecture.performance_metrics is None:
                metrics = self.evaluator.evaluate_architecture(
                    architecture, train_loader, val_loader
                )
                architecture.performance_metrics = metrics
                self.search_history.append(architecture)
    
    def _update_best_architecture(self):
        """Update best architecture based on objective."""
        if not self.population:
            return
        
        # Sort by objective
        if self.config.optimization_objective == OptimizationObjective.ACCURACY:
            self.population.sort(
                key=lambda x: x.performance_metrics.get('accuracy', 0), 
                reverse=True
            )
        elif self.config.optimization_objective == OptimizationObjective.LATENCY:
            self.population.sort(
                key=lambda x: x.performance_metrics.get('latency_ms', float('inf'))
            )
        elif self.config.optimization_objective == OptimizationObjective.PARAMETERS:
            self.population.sort(
                key=lambda x: x.performance_metrics.get('parameters', float('inf'))
            )
        
        # Update best
        if self.best_architecture is None:
            self.best_architecture = self.population[0]
        else:
            current_best = self.population[0]
            if self._is_better(current_best, self.best_architecture):
                self.best_architecture = current_best
    
    def _is_better(self, arch1: Architecture, arch2: Architecture) -> bool:
        """Check if architecture 1 is better than architecture 2."""
        if self.config.optimization_objective == OptimizationObjective.ACCURACY:
            return arch1.performance_metrics.get('accuracy', 0) > arch2.performance_metrics.get('accuracy', 0)
        elif self.config.optimization_objective == OptimizationObjective.LATENCY:
            return arch1.performance_metrics.get('latency_ms', float('inf')) < arch2.performance_metrics.get('latency_ms', float('inf'))
        elif self.config.optimization_objective == OptimizationObjective.PARAMETERS:
            return arch1.performance_metrics.get('parameters', float('inf')) < arch2.performance_metrics.get('parameters', float('inf'))
        else:
            return arch1.performance_metrics.get('accuracy', 0) > arch2.performance_metrics.get('accuracy', 0)
    
    def _evolutionary_step(self):
        """Perform one evolutionary step."""
        # Sort population by fitness
        self.population.sort(
            key=lambda x: x.performance_metrics.get('accuracy', 0), 
            reverse=True
        )
        
        # Keep elite
        elite = self.population[:self.config.elite_size]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.architecture_builder.crossover_architectures(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
        
        # Mutation
        for i in range(self.config.elite_size, len(new_population)):
            if random.random() < self.config.mutation_rate:
                new_population[i] = self.architecture_builder.mutate_architecture(new_population[i])
        
        # Trim to population size
        self.population = new_population[:self.config.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> Architecture:
        """Tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.performance_metrics.get('accuracy', 0))
    
    def _random_step(self):
        """Perform random search step."""
        # Replace worst architectures with random ones
        self.population.sort(
            key=lambda x: x.performance_metrics.get('accuracy', 0), 
            reverse=True
        )
        
        num_replace = len(self.population) // 4
        for i in range(num_replace):
            new_arch = self.architecture_builder.generate_random_architecture()
            self.population[-(i+1)] = new_arch
    
    def _bayesian_step(self):
        """Perform Bayesian optimization step."""
        # Simplified Bayesian step - in practice, you'd use a proper Bayesian optimization library
        self._random_step()
    
    def _get_optimization_curve(self) -> List[float]:
        """Get optimization curve."""
        curve = []
        for arch in self.search_history:
            if arch.performance_metrics:
                curve.append(arch.performance_metrics.get('accuracy', 0))
        return curve
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate."""
        if not self.search_history:
            return 0.0
        
        successful = sum(1 for arch in self.search_history 
                        if arch.performance_metrics and 
                        arch.performance_metrics.get('accuracy', 0) > 0.5)
        
        return successful / len(self.search_history)
    
    def visualize_architecture(self, architecture: Architecture, output_path: str = None) -> str:
        """Visualize neural architecture."""
        if output_path is None:
            output_path = f"architecture_{architecture.architecture_id}.png"
        
        # Create graph
        dot = Digraph(comment='Neural Architecture')
        dot.attr(rankdir='TB')
        
        # Add nodes
        for i, layer in enumerate(architecture.layers):
            label = f"{layer['type']}\n{layer.get('activation', '')}"
            dot.node(str(i), label)
        
        # Add edges
        for from_layer, to_layer in architecture.connections:
            dot.edge(str(from_layer), str(to_layer))
        
        # Render
        dot.render(output_path, format='png', cleanup=True)
        
        console.print(f"[green]Architecture visualization saved: {output_path}[/green]")
        return output_path
    
    def _save_search_result(self, search_result: SearchResult):
        """Save search result to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save best architecture
            if search_result.best_architecture:
                arch = search_result.best_architecture
                conn.execute("""
                    INSERT OR REPLACE INTO architectures 
                    (architecture_id, name, layers, connections, parameters, 
                     performance_metrics, complexity_metrics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    arch.architecture_id, arch.name,
                    json.dumps(arch.layers), json.dumps(arch.connections),
                    json.dumps(arch.parameters),
                    json.dumps(arch.performance_metrics) if arch.performance_metrics else None,
                    json.dumps(arch.complexity_metrics) if arch.complexity_metrics else None,
                    arch.created_at.isoformat()
                ))
            
            # Save search result
            conn.execute("""
                INSERT OR REPLACE INTO search_results 
                (result_id, best_architecture_id, search_history, optimization_curve,
                 search_time, total_trials, success_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                search_result.result_id,
                search_result.best_architecture.architecture_id if search_result.best_architecture else None,
                json.dumps([asdict(arch) for arch in search_result.search_history]),
                json.dumps(search_result.optimization_curve),
                search_result.search_time,
                search_result.total_trials,
                search_result.success_rate,
                search_result.created_at.isoformat()
            ))

def main():
    """Main function for NAS CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Architecture Search System")
    parser.add_argument("--strategy", type=str,
                       choices=["random", "evolutionary", "bayesian"],
                       default="evolutionary", help="Search strategy")
    parser.add_argument("--architecture-type", type=str,
                       choices=["cnn", "rnn", "transformer"],
                       default="cnn", help="Architecture type")
    parser.add_argument("--objective", type=str,
                       choices=["accuracy", "latency", "parameters"],
                       default="accuracy", help="Optimization objective")
    parser.add_argument("--max-trials", type=int, default=100, help="Maximum trials")
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--max-epochs", type=int, default=10, help="Max epochs per trial")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    
    args = parser.parse_args()
    
    # Create NAS configuration
    config = NASConfig(
        search_strategy=SearchStrategy(args.strategy),
        architecture_type=ArchitectureType(args.architecture_type),
        optimization_objective=OptimizationObjective(args.objective),
        max_trials=args.max_trials,
        population_size=args.population_size,
        generations=args.generations,
        max_epochs_per_trial=args.max_epochs,
        device=args.device
    )
    
    # Create NAS system
    nas = NeuralArchitectureSearch(config)
    
    # Load dataset
    if args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    else:
        console.print(f"[red]Unsupported dataset: {args.dataset}[/red]")
        return
    
    # Run NAS
    search_result = nas.search(train_loader, val_loader)
    
    # Show results
    console.print(f"[green]NAS completed successfully[/green]")
    console.print(f"[blue]Best architecture ID: {search_result.best_architecture.architecture_id}[/blue]")
    console.print(f"[blue]Best accuracy: {search_result.best_architecture.performance_metrics.get('accuracy', 0):.4f}[/blue]")
    console.print(f"[blue]Total trials: {search_result.total_trials}[/blue]")
    console.print(f"[blue]Search time: {search_result.search_time:.2f} seconds[/blue]")
    
    # Visualize best architecture
    nas.visualize_architecture(search_result.best_architecture)

if __name__ == "__main__":
    main()
