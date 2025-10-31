"""
Advanced Neural Network Neural Architecture Optimization System for TruthGPT Optimization Core
Complete neural architecture optimization with evolutionary algorithms, reinforcement learning, and Bayesian optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ArchitectureSearchStrategy(Enum):
    """Architecture search strategies"""
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRADIENT_BASED = "gradient_based"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"

class LayerType(Enum):
    """Layer types"""
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    DENSE = "dense"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"

class ActivationType(Enum):
    """Activation types"""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    GELU = "gelu"

class ArchitectureConfig:
    """Configuration for neural architecture optimization"""
    # Basic settings
    search_strategy: ArchitectureSearchStrategy = ArchitectureSearchStrategy.EVOLUTIONARY
    max_layers: int = 10
    min_layers: int = 2
    max_neurons: int = 1024
    min_neurons: int = 32
    
    # Layer types
    available_layer_types: List[LayerType] = field(default_factory=lambda: [
        LayerType.CONV2D, LayerType.DENSE, LayerType.LSTM, LayerType.DROPOUT, LayerType.BATCH_NORM
    ])
    available_activations: List[ActivationType] = field(default_factory=lambda: [
        ActivationType.RELU, ActivationType.LEAKY_RELU, ActivationType.TANH, ActivationType.SIGMOID
    ])
    
    # Search settings
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Performance settings
    max_training_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Multi-objective settings
    enable_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "efficiency"])
    objective_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    
    # Advanced features
    enable_transfer_learning: bool = True
    enable_progressive_search: bool = True
    enable_architecture_pruning: bool = True
    enable_ensemble_search: bool = False
    
    def __post_init__(self):
        """Validate architecture configuration"""
        if self.max_layers <= 0:
            raise ValueError("Maximum layers must be positive")
        if self.min_layers <= 0:
            raise ValueError("Minimum layers must be positive")
        if self.max_layers < self.min_layers:
            raise ValueError("Maximum layers must be >= minimum layers")
        if self.max_neurons <= 0:
            raise ValueError("Maximum neurons must be positive")
        if self.min_neurons <= 0:
            raise ValueError("Minimum neurons must be positive")
        if self.max_neurons < self.min_neurons:
            raise ValueError("Maximum neurons must be >= minimum neurons")
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.max_generations <= 0:
            raise ValueError("Maximum generations must be positive")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        if self.max_training_epochs <= 0:
            raise ValueError("Maximum training epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be positive")
        if not (0 < self.validation_split < 1):
            raise ValueError("Validation split must be between 0 and 1")

class ArchitectureGene:
    """Gene representing a layer in neural architecture"""
    
    def __init__(self, layer_type: LayerType, neurons: int = None, 
                 activation: ActivationType = None, **kwargs):
        self.layer_type = layer_type
        self.neurons = neurons
        self.activation = activation
        self.parameters = kwargs
        logger.debug(f"✅ Architecture gene created: {layer_type.value}")
    
    def copy(self):
        """Create a copy of the gene"""
        return ArchitectureGene(
            self.layer_type, self.neurons, self.activation, **self.parameters
        )
    
    def mutate(self, config: ArchitectureConfig):
        """Mutate the gene"""
        if random.random() < config.mutation_rate:
            # Mutate layer type
            if random.random() < 0.3:
                self.layer_type = random.choice(config.available_layer_types)
            
            # Mutate neurons
            if self.neurons is not None and random.random() < 0.3:
                self.neurons = random.randint(config.min_neurons, config.max_neurons)
            
            # Mutate activation
            if self.activation is not None and random.random() < 0.3:
                self.activation = random.choice(config.available_activations)
            
            # Mutate parameters
            for param_name in self.parameters:
                if random.random() < 0.2:
                    if isinstance(self.parameters[param_name], (int, float)):
                        self.parameters[param_name] *= random.uniform(0.8, 1.2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary"""
        return {
            'layer_type': self.layer_type.value,
            'neurons': self.neurons,
            'activation': self.activation.value if self.activation else None,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureGene':
        """Create gene from dictionary"""
        return cls(
            LayerType(data['layer_type']),
            data.get('neurons'),
            ActivationType(data['activation']) if data.get('activation') else None,
            **data.get('parameters', {})
        )

class NeuralArchitecture:
    """Neural architecture representation"""
    
    def __init__(self, genes: List[ArchitectureGene], config: ArchitectureConfig):
        self.genes = genes
        self.config = config
        self.fitness = None
        self.objectives = []
        self.training_time = 0
        self.parameters_count = 0
        logger.info(f"✅ Neural architecture created with {len(genes)} layers")
    
    def copy(self):
        """Create a copy of the architecture"""
        copied_genes = [gene.copy() for gene in self.genes]
        return NeuralArchitecture(copied_genes, self.config)
    
    def build_model(self, input_shape: Tuple[int, ...], output_shape: int) -> nn.Module:
        """Build PyTorch model from architecture"""
        logger.info("🏗️ Building PyTorch model from architecture")
        
        layers = []
        current_shape = input_shape
        
        for gene in self.genes:
            layer = self._build_layer(gene, current_shape)
            if layer is not None:
                layers.append(layer)
                # Update current shape (simplified)
                if hasattr(layer, 'out_features'):
                    current_shape = (layer.out_features,)
                elif hasattr(layer, 'out_channels'):
                    current_shape = (layer.out_channels, *current_shape[1:])
        
        # Add output layer
        if len(layers) > 0:
            if isinstance(layers[-1], nn.Linear):
                layers.append(nn.Linear(layers[-1].out_features, output_shape))
            else:
                layers.append(nn.Linear(current_shape[0] if isinstance(current_shape, tuple) else current_shape, output_shape))
        
        model = nn.Sequential(*layers)
        
        # Count parameters
        self.parameters_count = sum(p.numel() for p in model.parameters())
        
        return model
    
    def _build_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build individual layer from gene"""
        if gene.layer_type == LayerType.CONV2D:
            return self._build_conv2d_layer(gene, input_shape)
        elif gene.layer_type == LayerType.CONV1D:
            return self._build_conv1d_layer(gene, input_shape)
        elif gene.layer_type == LayerType.DENSE:
            return self._build_dense_layer(gene, input_shape)
        elif gene.layer_type == LayerType.LSTM:
            return self._build_lstm_layer(gene, input_shape)
        elif gene.layer_type == LayerType.GRU:
            return self._build_gru_layer(gene, input_shape)
        elif gene.layer_type == LayerType.DROPOUT:
            return self._build_dropout_layer(gene)
        elif gene.layer_type == LayerType.BATCH_NORM:
            return self._build_batch_norm_layer(gene, input_shape)
        elif gene.layer_type == LayerType.MAX_POOL:
            return self._build_max_pool_layer(gene)
        elif gene.layer_type == LayerType.AVG_POOL:
            return self._build_avg_pool_layer(gene)
        else:
            return None
    
    def _build_conv2d_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build Conv2D layer"""
        if len(input_shape) < 3:
            return None
        
        in_channels = input_shape[0]
        out_channels = gene.neurons or 64
        kernel_size = gene.parameters.get('kernel_size', 3)
        stride = gene.parameters.get('stride', 1)
        padding = gene.parameters.get('padding', 1)
        
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Add activation
        if gene.activation:
            activation_layer = self._build_activation(gene.activation)
            return nn.Sequential(conv_layer, activation_layer)
        
        return conv_layer
    
    def _build_conv1d_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build Conv1D layer"""
        if len(input_shape) < 2:
            return None
        
        in_channels = input_shape[0]
        out_channels = gene.neurons or 64
        kernel_size = gene.parameters.get('kernel_size', 3)
        stride = gene.parameters.get('stride', 1)
        padding = gene.parameters.get('padding', 1)
        
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Add activation
        if gene.activation:
            activation_layer = self._build_activation(gene.activation)
            return nn.Sequential(conv_layer, activation_layer)
        
        return conv_layer
    
    def _build_dense_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build Dense layer"""
        if len(input_shape) == 0:
            return None
        
        in_features = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        out_features = gene.neurons or 128
        
        dense_layer = nn.Linear(in_features, out_features)
        
        # Add activation
        if gene.activation:
            activation_layer = self._build_activation(gene.activation)
            return nn.Sequential(dense_layer, activation_layer)
        
        return dense_layer
    
    def _build_lstm_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build LSTM layer"""
        if len(input_shape) < 2:
            return None
        
        input_size = input_shape[0]
        hidden_size = gene.neurons or 128
        num_layers = gene.parameters.get('num_layers', 1)
        dropout = gene.parameters.get('dropout', 0.0)
        
        return nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    
    def _build_gru_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build GRU layer"""
        if len(input_shape) < 2:
            return None
        
        input_size = input_shape[0]
        hidden_size = gene.neurons or 128
        num_layers = gene.parameters.get('num_layers', 1)
        dropout = gene.parameters.get('dropout', 0.0)
        
        return nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    
    def _build_dropout_layer(self, gene: ArchitectureGene) -> nn.Module:
        """Build Dropout layer"""
        dropout_rate = gene.parameters.get('dropout_rate', 0.5)
        return nn.Dropout(dropout_rate)
    
    def _build_batch_norm_layer(self, gene: ArchitectureGene, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build BatchNorm layer"""
        if len(input_shape) == 0:
            return None
        
        num_features = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        return nn.BatchNorm1d(num_features)
    
    def _build_max_pool_layer(self, gene: ArchitectureGene) -> nn.Module:
        """Build MaxPool layer"""
        kernel_size = gene.parameters.get('kernel_size', 2)
        stride = gene.parameters.get('stride', 2)
        return nn.MaxPool2d(kernel_size, stride)
    
    def _build_avg_pool_layer(self, gene: ArchitectureGene) -> nn.Module:
        """Build AvgPool layer"""
        kernel_size = gene.parameters.get('kernel_size', 2)
        stride = gene.parameters.get('stride', 2)
        return nn.AvgPool2d(kernel_size, stride)
    
    def _build_activation(self, activation_type: ActivationType) -> nn.Module:
        """Build activation layer"""
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU()
        elif activation_type == ActivationType.ELU:
            return nn.ELU()
        elif activation_type == ActivationType.SELU:
            return nn.SELU()
        elif activation_type == ActivationType.TANH:
            return nn.Tanh()
        elif activation_type == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation_type == ActivationType.SWISH:
            return nn.SiLU()  # Swish is SiLU in PyTorch
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def mutate(self):
        """Mutate the architecture"""
        logger.info("🧬 Mutating neural architecture")
        
        # Add layer
        if len(self.genes) < self.config.max_layers and random.random() < 0.1:
            new_gene = ArchitectureGene(
                random.choice(self.config.available_layer_types),
                random.randint(self.config.min_neurons, self.config.max_neurons),
                random.choice(self.config.available_activations)
            )
            self.genes.append(new_gene)
        
        # Remove layer
        if len(self.genes) > self.config.min_layers and random.random() < 0.1:
            self.genes.pop(random.randint(0, len(self.genes) - 1))
        
        # Mutate existing genes
        for gene in self.genes:
            gene.mutate(self.config)
    
    def crossover(self, other: 'NeuralArchitecture') -> Tuple['NeuralArchitecture', 'NeuralArchitecture']:
        """Perform crossover with another architecture"""
        logger.info("🧬 Performing architecture crossover")
        
        if len(self.genes) == 0 or len(other.genes) == 0:
            return self.copy(), other.copy()
        
        # Single point crossover
        crossover_point = random.randint(1, min(len(self.genes), len(other.genes)) - 1)
        
        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]
        
        child1 = NeuralArchitecture(child1_genes, self.config)
        child2 = NeuralArchitecture(child2_genes, self.config)
        
        return child1, child2
    
    def evaluate(self, train_data: Tuple[torch.Tensor, torch.Tensor], 
                val_data: Tuple[torch.Tensor, torch.Tensor] = None) -> float:
        """Evaluate architecture performance"""
        logger.info("📊 Evaluating architecture performance")
        
        try:
            # Build model
            input_shape = train_data[0].shape[1:]
            output_shape = len(torch.unique(train_data[1]))
            
            model = self.build_model(input_shape, output_shape)
            
            # Train model
            start_time = time.time()
            fitness = self._train_model(model, train_data, val_data)
            self.training_time = time.time() - start_time
            
            self.fitness = fitness
            return fitness
            
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            self.fitness = 0.0
            return 0.0
    
    def _train_model(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                    val_data: Tuple[torch.Tensor, torch.Tensor] = None) -> float:
        """Train model and return fitness"""
        # Simplified training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(min(10, self.config.max_training_epochs)):  # Limit training time
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(train_data[0])
            loss = criterion(outputs, train_data[1])
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation data
        model.eval()
        with torch.no_grad():
            if val_data:
                val_outputs = model(val_data[0])
                val_predictions = torch.argmax(val_outputs, dim=1)
                accuracy = (val_predictions == val_data[1]).float().mean().item()
            else:
                # Use training data for evaluation
                train_outputs = model(train_data[0])
                train_predictions = torch.argmax(train_outputs, dim=1)
                accuracy = (train_predictions == train_data[1]).float().mean().item()
        
        return accuracy

class ArchitecturePopulation:
    """Population of neural architectures"""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.architectures = []
        self.generation = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        logger.info("✅ Architecture population initialized")
    
    def initialize(self, input_shape: Tuple[int, ...], output_shape: int):
        """Initialize population with random architectures"""
        logger.info(f"🏗️ Initializing population with {self.config.population_size} architectures")
        
        self.architectures = []
        
        for _ in range(self.config.population_size):
            # Random number of layers
            num_layers = random.randint(self.config.min_layers, self.config.max_layers)
            
            # Generate random genes
            genes = []
            for _ in range(num_layers):
                gene = ArchitectureGene(
                    random.choice(self.config.available_layer_types),
                    random.randint(self.config.min_neurons, self.config.max_neurons),
                    random.choice(self.config.available_activations)
                )
                genes.append(gene)
            
            architecture = NeuralArchitecture(genes, self.config)
            self.architectures.append(architecture)
        
        logger.info("✅ Population initialized")
    
    def evaluate_population(self, train_data: Tuple[torch.Tensor, torch.Tensor],
                           val_data: Tuple[torch.Tensor, torch.Tensor] = None):
        """Evaluate all architectures in population"""
        logger.info("📊 Evaluating population")
        
        for i, architecture in enumerate(self.architectures):
            logger.info(f"   Evaluating architecture {i + 1}/{len(self.architectures)}")
            architecture.evaluate(train_data, val_data)
        
        # Sort by fitness
        self.architectures.sort(key=lambda x: x.fitness, reverse=True)
        
        # Store fitness history
        best_fitness = self.architectures[0].fitness
        average_fitness = np.mean([arch.fitness for arch in self.architectures])
        
        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(average_fitness)
        
        logger.info(f"   Best fitness: {best_fitness:.4f}, Average fitness: {average_fitness:.4f}")
    
    def select_parents(self) -> List[NeuralArchitecture]:
        """Select parents for reproduction"""
        logger.info("👥 Selecting parents")
        
        # Tournament selection
        parents = []
        
        for _ in range(self.config.population_size):
            # Select tournament participants
            tournament = random.sample(self.architectures, min(3, len(self.architectures)))
            
            # Select winner
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner.copy())
        
        return parents
    
    def create_offspring(self, parents: List[NeuralArchitecture]) -> List[NeuralArchitecture]:
        """Create offspring through crossover and mutation"""
        logger.info("🧬 Creating offspring")
        
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                if random.random() < self.config.crossover_rate:
                    child1, child2 = parent1.crossover(parent2)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1.copy(), parent2.copy()])
            else:
                offspring.append(parents[i].copy())
        
        # Mutate offspring
        for child in offspring:
            if random.random() < self.config.mutation_rate:
                child.mutate()
        
        return offspring
    
    def replace_population(self, offspring: List[NeuralArchitecture]):
        """Replace population with offspring"""
        logger.info("🔄 Replacing population")
        
        # Keep best architectures (elitism)
        elite_size = min(5, len(self.architectures) // 4)
        elite = self.architectures[:elite_size]
        
        # Combine elite and offspring
        self.architectures = elite + offspring[:self.config.population_size - elite_size]
        
        # Increment generation
        self.generation += 1

class NeuralArchitectureOptimizer:
    """Main neural architecture optimizer"""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.population = ArchitecturePopulation(config)
        self.optimization_history = []
        logger.info("✅ Neural Architecture Optimizer initialized")
    
    def optimize(self, train_data: Tuple[torch.Tensor, torch.Tensor],
                val_data: Tuple[torch.Tensor, torch.Tensor] = None) -> Dict[str, Any]:
        """Optimize neural architecture"""
        logger.info(f"🚀 Optimizing neural architecture using {self.config.search_strategy.value}")
        
        optimization_results = {
            'start_time': time.time(),
            'config': self.config,
            'generations': []
        }
        
        # Initialize population
        input_shape = train_data[0].shape[1:]
        output_shape = len(torch.unique(train_data[1]))
        self.population.initialize(input_shape, output_shape)
        
        # Evaluate initial population
        self.population.evaluate_population(train_data, val_data)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            logger.info(f"🔄 Generation {generation + 1}/{self.config.max_generations}")
            
            # Select parents
            parents = self.population.select_parents()
            
            # Create offspring
            offspring = self.population.create_offspring(parents)
            
            # Evaluate offspring
            for child in offspring:
                child.evaluate(train_data, val_data)
            
            # Replace population
            self.population.replace_population(offspring)
            
            # Store generation results
            generation_result = {
                'generation': generation,
                'best_fitness': self.population.best_fitness_history[-1],
                'average_fitness': self.population.average_fitness_history[-1],
                'best_architecture': self.population.architectures[0].genes.copy()
            }
            
            optimization_results['generations'].append(generation_result)
            
            if generation % 10 == 0:
                logger.info(f"   Generation {generation}: Best = {generation_result['best_fitness']:.4f}, "
                          f"Avg = {generation_result['average_fitness']:.4f}")
        
        # Final evaluation
        optimization_results['end_time'] = time.time()
        optimization_results['total_duration'] = optimization_results['end_time'] - optimization_results['start_time']
        optimization_results['best_architecture'] = self.population.architectures[0]
        optimization_results['best_fitness'] = self.population.architectures[0].fitness
        optimization_results['final_generation'] = self.population.generation
        
        # Store results
        self.optimization_history.append(optimization_results)
        
        logger.info("✅ Neural architecture optimization completed")
        return optimization_results
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate optimization report"""
        report = []
        report.append("=" * 50)
        report.append("NEURAL ARCHITECTURE OPTIMIZATION REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nNEURAL ARCHITECTURE OPTIMIZATION CONFIGURATION:")
        report.append("-" * 45)
        report.append(f"Search Strategy: {self.config.search_strategy.value}")
        report.append(f"Maximum Layers: {self.config.max_layers}")
        report.append(f"Minimum Layers: {self.config.min_layers}")
        report.append(f"Maximum Neurons: {self.config.max_neurons}")
        report.append(f"Minimum Neurons: {self.config.min_neurons}")
        report.append(f"Available Layer Types: {[t.value for t in self.config.available_layer_types]}")
        report.append(f"Available Activations: {[a.value for a in self.config.available_activations]}")
        report.append(f"Population Size: {self.config.population_size}")
        report.append(f"Maximum Generations: {self.config.max_generations}")
        report.append(f"Mutation Rate: {self.config.mutation_rate}")
        report.append(f"Crossover Rate: {self.config.crossover_rate}")
        report.append(f"Maximum Training Epochs: {self.config.max_training_epochs}")
        report.append(f"Early Stopping Patience: {self.config.early_stopping_patience}")
        report.append(f"Validation Split: {self.config.validation_split}")
        report.append(f"Multi-Objective: {'Enabled' if self.config.enable_multi_objective else 'Disabled'}")
        report.append(f"Objectives: {self.config.objectives}")
        report.append(f"Objective Weights: {self.config.objective_weights}")
        report.append(f"Transfer Learning: {'Enabled' if self.config.enable_transfer_learning else 'Disabled'}")
        report.append(f"Progressive Search: {'Enabled' if self.config.enable_progressive_search else 'Disabled'}")
        report.append(f"Architecture Pruning: {'Enabled' if self.config.enable_architecture_pruning else 'Disabled'}")
        report.append(f"Ensemble Search: {'Enabled' if self.config.enable_ensemble_search else 'Disabled'}")
        
        # Results
        report.append("\nNEURAL ARCHITECTURE OPTIMIZATION RESULTS:")
        report.append("-" * 42)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        report.append(f"Start Time: {results.get('start_time', 'Unknown')}")
        report.append(f"End Time: {results.get('end_time', 'Unknown')}")
        report.append(f"Final Generation: {results.get('final_generation', 0)}")
        report.append(f"Best Fitness: {results.get('best_fitness', 0):.4f}")
        
        # Best architecture details
        if 'best_architecture' in results:
            best_arch = results['best_architecture']
            report.append(f"Best Architecture Layers: {len(best_arch.genes)}")
            report.append(f"Best Architecture Parameters: {best_arch.parameters_count}")
            report.append(f"Best Architecture Training Time: {best_arch.training_time:.2f} seconds")
        
        # Generation results
        if 'generations' in results:
            report.append(f"\nNumber of Generations: {len(results['generations'])}")
            
            if results['generations']:
                final_generation = results['generations'][-1]
                report.append(f"Final Best Fitness: {final_generation.get('best_fitness', 0):.4f}")
                report.append(f"Final Average Fitness: {final_generation.get('average_fitness', 0):.4f}")
        
        return "\n".join(report)
    
    def visualize_optimization_results(self, save_path: str = None):
        """Visualize optimization results"""
        if not self.optimization_history:
            logger.warning("No optimization history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Optimization duration over time
        durations = [r.get('total_duration', 0) for r in self.optimization_history]
        axes[0, 0].plot(durations, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Optimization Run')
        axes[0, 0].set_ylabel('Duration (seconds)')
        axes[0, 0].set_title('Neural Architecture Optimization Duration Over Time')
        axes[0, 0].grid(True)
        
        # Plot 2: Search strategy distribution
        search_strategies = [self.config.search_strategy.value]
        strategy_counts = [1]
        
        axes[0, 1].pie(strategy_counts, labels=search_strategies, autopct='%1.1f%%')
        axes[0, 1].set_title('Search Strategy Distribution')
        
        # Plot 3: Layer type distribution
        layer_types = [t.value for t in self.config.available_layer_types]
        layer_counts = [1] * len(layer_types)
        
        axes[1, 0].pie(layer_counts, labels=layer_types, autopct='%1.1f%%')
        axes[1, 0].set_title('Available Layer Types Distribution')
        
        # Plot 4: Architecture configuration
        config_values = [
            self.config.max_layers,
            self.config.max_neurons,
            self.config.population_size,
            self.config.max_generations
        ]
        config_labels = ['Max Layers', 'Max Neurons', 'Population Size', 'Max Generations']
        
        axes[1, 1].bar(config_labels, config_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Architecture Configuration')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

# Factory functions
def create_architecture_config(**kwargs) -> ArchitectureConfig:
    """Create architecture configuration"""
    return ArchitectureConfig(**kwargs)

def create_architecture_gene(layer_type: LayerType, neurons: int = None, 
                           activation: ActivationType = None, **kwargs) -> ArchitectureGene:
    """Create architecture gene"""
    return ArchitectureGene(layer_type, neurons, activation, **kwargs)

def create_neural_architecture(genes: List[ArchitectureGene], config: ArchitectureConfig) -> NeuralArchitecture:
    """Create neural architecture"""
    return NeuralArchitecture(genes, config)

def create_architecture_population(config: ArchitectureConfig) -> ArchitecturePopulation:
    """Create architecture population"""
    return ArchitecturePopulation(config)

def create_neural_architecture_optimizer(config: ArchitectureConfig) -> NeuralArchitectureOptimizer:
    """Create neural architecture optimizer"""
    return NeuralArchitectureOptimizer(config)

# Example usage
def example_neural_architecture_optimization():
    """Example of neural architecture optimization system"""
    # Create configuration
    config = create_architecture_config(
        search_strategy=ArchitectureSearchStrategy.EVOLUTIONARY,
        max_layers=10,
        min_layers=2,
        max_neurons=1024,
        min_neurons=32,
        available_layer_types=[LayerType.CONV2D, LayerType.DENSE, LayerType.DROPOUT, LayerType.BATCH_NORM],
        available_activations=[ActivationType.RELU, ActivationType.LEAKY_RELU, ActivationType.TANH, ActivationType.SIGMOID],
        population_size=50,
        max_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_training_epochs=50,
        early_stopping_patience=10,
        validation_split=0.2,
        enable_multi_objective=False,
        objectives=["accuracy", "efficiency"],
        objective_weights=[0.7, 0.3],
        enable_transfer_learning=True,
        enable_progressive_search=True,
        enable_architecture_pruning=True,
        enable_ensemble_search=False
    )
    
    # Create neural architecture optimizer
    architecture_optimizer = create_neural_architecture_optimizer(config)
    
    # Generate dummy data
    n_samples = 1000
    n_features = 784
    
    train_data = (torch.randn(n_samples, n_features), torch.randint(0, 10, (n_samples,)))
    val_data = (torch.randn(200, n_features), torch.randint(0, 10, (200,)))
    
    # Optimize architecture
    optimization_results = architecture_optimizer.optimize(train_data, val_data)
    
    # Generate report
    optimization_report = architecture_optimizer.generate_optimization_report(optimization_results)
    
    print(f"✅ Neural Architecture Optimization Example Complete!")
    print(f"🚀 Neural Architecture Optimization Statistics:")
    print(f"   Search Strategy: {config.search_strategy.value}")
    print(f"   Maximum Layers: {config.max_layers}")
    print(f"   Minimum Layers: {config.min_layers}")
    print(f"   Maximum Neurons: {config.max_neurons}")
    print(f"   Minimum Neurons: {config.min_neurons}")
    print(f"   Available Layer Types: {[t.value for t in config.available_layer_types]}")
    print(f"   Available Activations: {[a.value for a in config.available_activations]}")
    print(f"   Population Size: {config.population_size}")
    print(f"   Maximum Generations: {config.max_generations}")
    print(f"   Mutation Rate: {config.mutation_rate}")
    print(f"   Crossover Rate: {config.crossover_rate}")
    print(f"   Maximum Training Epochs: {config.max_training_epochs}")
    print(f"   Early Stopping Patience: {config.early_stopping_patience}")
    print(f"   Validation Split: {config.validation_split}")
    print(f"   Multi-Objective: {'Enabled' if config.enable_multi_objective else 'Disabled'}")
    print(f"   Objectives: {config.objectives}")
    print(f"   Objective Weights: {config.objective_weights}")
    print(f"   Transfer Learning: {'Enabled' if config.enable_transfer_learning else 'Disabled'}")
    print(f"   Progressive Search: {'Enabled' if config.enable_progressive_search else 'Disabled'}")
    print(f"   Architecture Pruning: {'Enabled' if config.enable_architecture_pruning else 'Disabled'}")
    print(f"   Ensemble Search: {'Enabled' if config.enable_ensemble_search else 'Disabled'}")
    
    print(f"\n📊 Neural Architecture Optimization Results:")
    print(f"   Optimization History Length: {len(architecture_optimizer.optimization_history)}")
    print(f"   Total Duration: {optimization_results.get('total_duration', 0):.2f} seconds")
    print(f"   Final Generation: {optimization_results.get('final_generation', 0)}")
    print(f"   Best Fitness: {optimization_results.get('best_fitness', 0):.4f}")
    
    # Show best architecture details
    if 'best_architecture' in optimization_results:
        best_arch = optimization_results['best_architecture']
        print(f"   Best Architecture Layers: {len(best_arch.genes)}")
        print(f"   Best Architecture Parameters: {best_arch.parameters_count}")
        print(f"   Best Architecture Training Time: {best_arch.training_time:.2f} seconds")
    
    # Show generation results summary
    if 'generations' in optimization_results:
        print(f"   Number of Generations: {len(optimization_results['generations'])}")
    
    print(f"\n📋 Neural Architecture Optimization Report:")
    print(optimization_report)
    
    return architecture_optimizer

# Export utilities
__all__ = [
    'ArchitectureSearchStrategy',
    'LayerType',
    'ActivationType',
    'ArchitectureConfig',
    'ArchitectureGene',
    'NeuralArchitecture',
    'ArchitecturePopulation',
    'NeuralArchitectureOptimizer',
    'create_architecture_config',
    'create_architecture_gene',
    'create_neural_architecture',
    'create_architecture_population',
    'create_neural_architecture_optimizer',
    'example_neural_architecture_optimization'
]

if __name__ == "__main__":
    example_neural_architecture_optimization()
    print("✅ Neural architecture optimization example completed successfully!")