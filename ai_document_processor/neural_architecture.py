#!/usr/bin/env python3
"""
Neural Architecture Search - Advanced AI Document Processor
=========================================================

Next-generation neural architecture search and optimization for document processing.
"""

import asyncio
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class ArchitectureConfig:
    """Neural architecture configuration."""
    search_space: Dict[str, List[Any]] = field(default_factory=lambda: {
        'num_layers': [2, 3, 4, 5, 6],
        'hidden_sizes': [64, 128, 256, 512, 1024],
        'activation_functions': ['relu', 'gelu', 'swish', 'mish'],
        'dropout_rates': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rates': [1e-4, 1e-3, 1e-2],
        'optimizers': ['adam', 'adamw', 'sgd', 'rmsprop']
    })
    max_trials: int = 100
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    max_epochs: int = 50
    batch_size: int = 32

@dataclass
class ArchitectureCandidate:
    """Neural architecture candidate."""
    architecture: Dict[str, Any]
    fitness_score: float = 0.0
    training_time: float = 0.0
    validation_accuracy: float = 0.0
    parameters_count: int = 0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

class DocumentProcessingModel(nn.Module):
    """Base document processing neural network."""
    
    def __init__(self, config: Dict[str, Any], input_size: int, output_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Build architecture based on config
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        current_size = input_size
        
        # Hidden layers
        num_layers = config.get('num_layers', 3)
        hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
        dropout_rates = config.get('dropout_rates', [0.2, 0.3, 0.2])
        activation = config.get('activation_functions', 'relu')
        
        for i in range(num_layers):
            if i < len(hidden_sizes):
                hidden_size = hidden_sizes[i]
            else:
                hidden_size = hidden_sizes[-1]  # Use last size for additional layers
            
            self.layers.append(nn.Linear(current_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]))
            current_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(current_size, output_size)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass."""
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)
        
        x = self.output_layer(x)
        return x
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

class NeuralArchitectureSearch:
    """Neural Architecture Search for document processing."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.population: List[ArchitectureCandidate] = []
        self.generation = 0
        self.best_architecture = None
        self.fitness_history: List[float] = []
        self.search_history: List[Dict[str, Any]] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population of architectures."""
        self.population = []
        
        for i in range(self.config.population_size):
            architecture = self._generate_random_architecture()
            candidate = ArchitectureCandidate(
                architecture=architecture,
                generation=0
            )
            self.population.append(candidate)
        
        logger.info(f"Initialized population of {len(self.population)} architectures")
    
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """Generate random architecture from search space."""
        architecture = {}
        
        for param, values in self.config.search_space.items():
            if param == 'hidden_sizes':
                # Generate list of hidden sizes
                num_layers = np.random.choice(self.config.search_space['num_layers'])
                architecture[param] = np.random.choice(values, size=num_layers, replace=True).tolist()
            else:
                architecture[param] = np.random.choice(values)
        
        return architecture
    
    async def search(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                    val_data: torch.Tensor, val_labels: torch.Tensor) -> ArchitectureCandidate:
        """Perform neural architecture search."""
        logger.info("Starting Neural Architecture Search...")
        
        start_time = time.time()
        
        for generation in range(self.config.max_trials // self.config.population_size):
            self.generation = generation
            
            # Evaluate population
            await self._evaluate_population(train_data, train_labels, val_data, val_labels)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Update best architecture
            if self.best_architecture is None or self.population[0].fitness_score > self.best_architecture.fitness_score:
                self.best_architecture = self.population[0]
            
            # Record fitness history
            self.fitness_history.append(self.population[0].fitness_score)
            
            # Log progress
            logger.info(f"Generation {generation}: Best fitness = {self.population[0].fitness_score:.4f}")
            
            # Check early stopping
            if self._check_early_stopping():
                logger.info("Early stopping triggered")
                break
            
            # Create next generation
            self._create_next_generation()
        
        search_time = time.time() - start_time
        logger.info(f"NAS completed in {search_time:.2f}s. Best fitness: {self.best_architecture.fitness_score:.4f}")
        
        return self.best_architecture
    
    async def _evaluate_population(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                                 val_data: torch.Tensor, val_labels: torch.Tensor):
        """Evaluate all architectures in population."""
        tasks = []
        
        for candidate in self.population:
            if candidate.fitness_score == 0.0:  # Only evaluate unevaluated candidates
                task = asyncio.create_task(
                    self._evaluate_architecture(candidate, train_data, train_labels, val_data, val_labels)
                )
                tasks.append(task)
        
        # Wait for all evaluations to complete
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _evaluate_architecture(self, candidate: ArchitectureCandidate,
                                   train_data: torch.Tensor, train_labels: torch.Tensor,
                                   val_data: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Evaluate a single architecture."""
        try:
            start_time = time.time()
            
            # Create model
            input_size = train_data.shape[1]
            output_size = len(torch.unique(train_labels))
            model = DocumentProcessingModel(candidate.architecture, input_size, output_size).to(self.device)
            
            # Get optimizer
            optimizer = self._get_optimizer(model, candidate.architecture)
            
            # Training loop
            model.train()
            best_val_accuracy = 0.0
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Training
                train_loss = await self._train_epoch(model, optimizer, train_data, train_labels)
                
                # Validation
                val_accuracy = await self._validate_epoch(model, val_data, val_labels)
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    break
            
            # Calculate fitness
            training_time = time.time() - start_time
            inference_time = await self._measure_inference_time(model, val_data)
            memory_usage = self._measure_memory_usage(model)
            
            fitness = self._calculate_fitness(
                best_val_accuracy, training_time, inference_time, 
                memory_usage, model.get_parameter_count()
            )
            
            # Update candidate
            candidate.fitness_score = fitness
            candidate.training_time = training_time
            candidate.validation_accuracy = best_val_accuracy
            candidate.parameters_count = model.get_parameter_count()
            candidate.inference_time = inference_time
            candidate.memory_usage = memory_usage
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return fitness
            
        except Exception as e:
            logger.error(f"Architecture evaluation failed: {e}")
            candidate.fitness_score = 0.0
            return 0.0
    
    async def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                          train_data: torch.Tensor, train_labels: torch.Tensor) -> float:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    async def _validate_epoch(self, model: nn.Module, val_data: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Validate model for one epoch."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            dataset = torch.utils.data.TensorDataset(val_data, val_labels)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
            
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    async def _measure_inference_time(self, model: nn.Module, test_data: torch.Tensor) -> float:
        """Measure inference time."""
        model.eval()
        test_data = test_data.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data[:1])
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_data[:1])
        end_time = time.time()
        
        return (end_time - start_time) / 100  # Average time per inference
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Measure model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024  # Convert to MB
    
    def _get_optimizer(self, model: nn.Module, architecture: Dict[str, Any]) -> torch.optim.Optimizer:
        """Get optimizer based on architecture config."""
        optimizer_name = architecture.get('optimizers', 'adam')
        learning_rate = architecture.get('learning_rates', 1e-3)
        
        optimizers = {
            'adam': optim.Adam(model.parameters(), lr=learning_rate),
            'adamw': optim.AdamW(model.parameters(), lr=learning_rate),
            'sgd': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
            'rmsprop': optim.RMSprop(model.parameters(), lr=learning_rate)
        }
        
        return optimizers.get(optimizer_name, optim.Adam(model.parameters(), lr=learning_rate))
    
    def _calculate_fitness(self, accuracy: float, training_time: float, inference_time: float,
                          memory_usage: float, parameter_count: int) -> float:
        """Calculate fitness score for architecture."""
        # Multi-objective fitness function
        accuracy_weight = 0.5
        speed_weight = 0.2
        efficiency_weight = 0.2
        size_weight = 0.1
        
        # Normalize metrics (higher is better)
        accuracy_score = accuracy
        
        # Speed score (lower training time is better)
        speed_score = 1.0 / (1.0 + training_time)
        
        # Efficiency score (lower inference time is better)
        efficiency_score = 1.0 / (1.0 + inference_time)
        
        # Size score (fewer parameters is better)
        size_score = 1.0 / (1.0 + parameter_count / 1000000)  # Normalize by 1M parameters
        
        fitness = (accuracy_weight * accuracy_score +
                  speed_weight * speed_score +
                  efficiency_weight * efficiency_score +
                  size_weight * size_score)
        
        return fitness
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria is met."""
        if len(self.fitness_history) < self.config.early_stopping_patience:
            return False
        
        recent_fitness = self.fitness_history[-self.config.early_stopping_patience:]
        return max(recent_fitness) - min(recent_fitness) < 0.001  # No improvement
    
    def _create_next_generation(self):
        """Create next generation using genetic operators."""
        new_population = []
        
        # Keep elite
        elite = self.population[:self.config.elite_size]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._tournament_selection()
                child = self._mutate(parent)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def _tournament_selection(self, tournament_size: int = 3) -> ArchitectureCandidate:
        """Tournament selection for parent selection."""
        tournament = np.random.choice(self.population, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _crossover(self, parent1: ArchitectureCandidate, parent2: ArchitectureCandidate) -> ArchitectureCandidate:
        """Crossover two parent architectures."""
        child_architecture = {}
        
        for param in self.config.search_space.keys():
            if np.random.random() < 0.5:
                child_architecture[param] = parent1.architecture[param]
            else:
                child_architecture[param] = parent2.architecture[param]
        
        child = ArchitectureCandidate(
            architecture=child_architecture,
            generation=self.generation,
            parent_ids=[id(parent1), id(parent2)]
        )
        
        return child
    
    def _mutate(self, parent: ArchitectureCandidate) -> ArchitectureCandidate:
        """Mutate parent architecture."""
        child_architecture = parent.architecture.copy()
        
        for param, values in self.config.search_space.items():
            if np.random.random() < self.config.mutation_rate:
                if param == 'hidden_sizes':
                    # Mutate hidden sizes list
                    num_layers = len(child_architecture[param])
                    child_architecture[param] = np.random.choice(values, size=num_layers, replace=True).tolist()
                else:
                    # Mutate single value
                    child_architecture[param] = np.random.choice(values)
        
        child = ArchitectureCandidate(
            architecture=child_architecture,
            generation=self.generation,
            parent_ids=[id(parent)],
            mutation_history=parent.mutation_history + [f"mutated_{param}" for param in self.config.search_space.keys()]
        )
        
        return child
    
    def get_search_results(self) -> Dict[str, Any]:
        """Get comprehensive search results."""
        return {
            'best_architecture': self.best_architecture.architecture if self.best_architecture else None,
            'best_fitness': self.best_architecture.fitness_score if self.best_architecture else 0.0,
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'population_size': len(self.population),
            'search_space': self.config.search_space,
            'total_evaluations': len(self.search_history)
        }
    
    def display_search_dashboard(self):
        """Display neural architecture search dashboard."""
        results = self.get_search_results()
        
        # Best architecture table
        if self.best_architecture:
            best_table = Table(title="Best Architecture")
            best_table.add_column("Metric", style="cyan")
            best_table.add_column("Value", style="green")
            
            best_table.add_row("Fitness Score", f"{self.best_architecture.fitness_score:.4f}")
            best_table.add_row("Validation Accuracy", f"{self.best_architecture.validation_accuracy:.4f}")
            best_table.add_row("Training Time", f"{self.best_architecture.training_time:.2f}s")
            best_table.add_row("Inference Time", f"{self.best_architecture.inference_time:.4f}s")
            best_table.add_row("Memory Usage", f"{self.best_architecture.memory_usage:.2f}MB")
            best_table.add_row("Parameters", f"{self.best_architecture.parameters_count:,}")
            
            console.print(best_table)
        
        # Architecture details
        if self.best_architecture:
            arch_table = Table(title="Architecture Configuration")
            arch_table.add_column("Parameter", style="cyan")
            arch_table.add_column("Value", style="green")
            
            for param, value in self.best_architecture.architecture.items():
                arch_table.add_row(param, str(value))
            
            console.print(arch_table)
        
        # Search statistics
        stats_table = Table(title="Search Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Generation", str(results['generation']))
        stats_table.add_row("Population Size", str(results['population_size']))
        stats_table.add_row("Total Evaluations", str(results['total_evaluations']))
        stats_table.add_row("Best Fitness", f"{results['best_fitness']:.4f}")
        
        console.print(stats_table)

# Global NAS instance
nas_engine = NeuralArchitectureSearch(ArchitectureConfig())

# Utility functions
async def search_architecture(train_data: torch.Tensor, train_labels: torch.Tensor,
                            val_data: torch.Tensor, val_labels: torch.Tensor) -> ArchitectureCandidate:
    """Search for optimal neural architecture."""
    return await nas_engine.search(train_data, train_labels, val_data, val_labels)

def get_search_results() -> Dict[str, Any]:
    """Get neural architecture search results."""
    return nas_engine.get_search_results()

def display_search_dashboard():
    """Display neural architecture search dashboard."""
    nas_engine.display_search_dashboard()

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create sample data
        train_data = torch.randn(1000, 100)
        train_labels = torch.randint(0, 10, (1000,))
        val_data = torch.randn(200, 100)
        val_labels = torch.randint(0, 10, (200,))
        
        # Search for optimal architecture
        best_arch = await search_architecture(train_data, train_labels, val_data, val_labels)
        print(f"Best architecture: {best_arch.architecture}")
        print(f"Best fitness: {best_arch.fitness_score:.4f}")
        
        # Display dashboard
        display_search_dashboard()
    
    asyncio.run(main())














