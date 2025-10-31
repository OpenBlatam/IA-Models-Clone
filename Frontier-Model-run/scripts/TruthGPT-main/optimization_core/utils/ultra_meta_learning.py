"""
Ultra-Advanced Meta-Learning Module
===================================

This module provides meta-learning capabilities for TruthGPT models,
including model-agnostic meta-learning, gradient-based meta-learning, and few-shot learning.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings
import threading
import queue
import asyncio
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class MetaLearningAlgorithm(Enum):
    """Meta-learning algorithms."""
    MAML = "maml"
    REPTILE = "reptile"
    PROTONET = "protonet"
    META_SGD = "meta_sgd"
    META_LSTM = "meta_lstm"
    CAVIA = "cavia"
    LEARNED_INITIALIZATION = "learned_initialization"
    GRADIENT_BASED_META = "gradient_based_meta"

class MetaLearningTask(Enum):
    """Meta-learning task types."""
    FEW_SHOT_CLASSIFICATION = "few_shot_classification"
    FEW_SHOT_REGRESSION = "few_shot_regression"
    FEW_SHOT_SEGMENTATION = "few_shot_segmentation"
    FEW_SHOT_DETECTION = "few_shot_detection"
    MULTI_TASK_LEARNING = "multi_task_learning"
    DOMAIN_ADAPTATION = "domain_adaptation"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"

class MetaLearningMode(Enum):
    """Meta-learning modes."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    MULTI_MODAL = "multi_modal"

class AdaptationStrategy(Enum):
    """Adaptation strategies."""
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON_METHOD = "newton_method"
    ADAPTIVE_MOMENTUM = "adaptive_momentum"
    LEARNING_TO_LEARN = "learning_to_learn"
    META_GRADIENT = "meta_gradient"
    IMPLICIT_DIFFERENTIATION = "implicit_differentiation"

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML
    task_type: MetaLearningTask = MetaLearningTask.FEW_SHOT_CLASSIFICATION
    mode: MetaLearningMode = MetaLearningMode.SUPERVISED
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.GRADIENT_DESCENT
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    inner_steps: int = 5
    outer_steps: int = 1000
    support_shots: int = 5
    query_shots: int = 15
    meta_batch_size: int = 4
    task_batch_size: int = 8
    gradient_clip: float = 10.0
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./meta_learning_results"

class MetaTask:
    """Represents a meta-learning task."""
    
    def __init__(self, task_id: str, support_set: List[Tuple[torch.Tensor, torch.Tensor]], 
                 query_set: List[Tuple[torch.Tensor, torch.Tensor]], task_type: MetaLearningTask):
        self.task_id = task_id
        self.support_set = support_set
        self.query_set = query_set
        self.task_type = task_type
        self.created_at = time.time()
        self.metadata = {}
        
    def get_support_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get support set data."""
        if not self.support_set:
            return torch.empty(0), torch.empty(0)
        
        inputs, targets = zip(*self.support_set)
        return torch.stack(inputs), torch.stack(targets)
    
    def get_query_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get query set data."""
        if not self.query_set:
            return torch.empty(0), torch.empty(0)
        
        inputs, targets = zip(*self.query_set)
        return torch.stack(inputs), torch.stack(targets)
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get task information."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'support_size': len(self.support_set),
            'query_size': len(self.query_set),
            'created_at': self.created_at,
            'metadata': self.metadata
        }

class MAML:
    """Model-Agnostic Meta-Learning implementation."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.model = None
        self.meta_optimizer = None
        self.inner_optimizer = None
        self.training_history = deque(maxlen=1000)
        
    def set_model(self, model: nn.Module):
        """Set the model for meta-learning."""
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.outer_lr)
        
    def meta_train_step(self, meta_batch: List[MetaTask]) -> Dict[str, float]:
        """Perform one meta-training step."""
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        meta_loss = 0.0
        meta_accuracies = []
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for task in meta_batch:
            # Inner loop: adapt to task
            adapted_params = self._inner_loop(task)
            
            # Outer loop: evaluate on query set
            query_inputs, query_targets = task.get_query_data()
            
            # Temporarily update model parameters
            self._update_model_parameters(adapted_params)
            
            # Forward pass on query set
            query_outputs = self.model(query_inputs)
            task_loss = self._compute_loss(query_outputs, query_targets, task.task_type)
            
            # Compute accuracy
            accuracy = self._compute_accuracy(query_outputs, query_targets, task.task_type)
            meta_accuracies.append(accuracy)
            
            meta_loss += task_loss
        
        # Restore original parameters
        self._update_model_parameters(original_params)
        
        # Meta-optimization step
        meta_loss /= len(meta_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'meta_accuracy': statistics.mean(meta_accuracies),
            'num_tasks': len(meta_batch)
        }
    
    def _inner_loop(self, task: MetaTask) -> Dict[str, torch.Tensor]:
        """Inner loop adaptation."""
        support_inputs, support_targets = task.get_support_data()
        
        # Create inner optimizer
        inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.inner_lr)
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner gradient steps
        for _ in range(self.config.inner_steps):
            inner_optimizer.zero_grad()
            
            outputs = self.model(support_inputs)
            loss = self._compute_loss(outputs, support_targets, task.task_type)
            
            loss.backward()
            inner_optimizer.step()
        
        # Return adapted parameters
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Restore original parameters
        self._update_model_parameters(original_params)
        
        return adapted_params
    
    def _update_model_parameters(self, params: Dict[str, torch.Tensor]):
        """Update model parameters."""
        for name, param in self.model.named_parameters():
            if name in params:
                param.data = params[name].data
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                     task_type: MetaLearningTask) -> torch.Tensor:
        """Compute loss based on task type."""
        if task_type in [MetaLearningTask.FEW_SHOT_CLASSIFICATION, MetaLearningTask.MULTI_TASK_LEARNING]:
            return F.cross_entropy(outputs, targets)
        elif task_type in [MetaLearningTask.FEW_SHOT_REGRESSION]:
            return F.mse_loss(outputs, targets)
        else:
            return F.cross_entropy(outputs, targets)  # Default
    
    def _compute_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor, 
                         task_type: MetaLearningTask) -> float:
        """Compute accuracy based on task type."""
        if task_type in [MetaLearningTask.FEW_SHOT_CLASSIFICATION, MetaLearningTask.MULTI_TASK_LEARNING]:
            predictions = outputs.argmax(dim=1)
            return (predictions == targets).float().mean().item()
        else:
            return 0.0  # Not applicable for regression tasks
    
    def meta_test(self, test_task: MetaTask) -> Dict[str, float]:
        """Meta-test on a new task."""
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Adapt to test task
        adapted_params = self._inner_loop(test_task)
        
        # Evaluate on query set
        query_inputs, query_targets = test_task.get_query_data()
        
        # Temporarily update model parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        self._update_model_parameters(adapted_params)
        
        # Forward pass
        query_outputs = self.model(query_inputs)
        test_loss = self._compute_loss(query_outputs, query_targets, test_task.task_type)
        test_accuracy = self._compute_accuracy(query_outputs, query_targets, test_task.task_type)
        
        # Restore original parameters
        self._update_model_parameters(original_params)
        
        return {
            'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy
        }

class Reptile:
    """Reptile meta-learning implementation."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.model = None
        self.meta_optimizer = None
        self.training_history = deque(maxlen=1000)
        
    def set_model(self, model: nn.Module):
        """Set the model for meta-learning."""
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.outer_lr)
        
    def meta_train_step(self, meta_batch: List[MetaTask]) -> Dict[str, float]:
        """Perform one meta-training step."""
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        meta_gradients = []
        
        for task in meta_batch:
            # Inner loop: adapt to task
            adapted_params = self._inner_loop(task)
            
            # Compute gradient as difference from original parameters
            task_gradient = {}
            for name, param in self.model.named_parameters():
                if name in adapted_params:
                    task_gradient[name] = adapted_params[name] - original_params[name]
            
            meta_gradients.append(task_gradient)
        
        # Average gradients
        avg_gradient = {}
        for name in original_params.keys():
            gradients = [grad[name] for grad in meta_gradients if name in grad]
            if gradients:
                avg_gradient[name] = torch.stack(gradients).mean(dim=0)
        
        # Update model parameters
        for name, param in self.model.named_parameters():
            if name in avg_gradient:
                param.data = original_params[name] + avg_gradient[name]
        
        return {
            'meta_loss': 0.0,  # Reptile doesn't compute explicit loss
            'num_tasks': len(meta_batch)
        }
    
    def _inner_loop(self, task: MetaTask) -> Dict[str, torch.Tensor]:
        """Inner loop adaptation."""
        support_inputs, support_targets = task.get_support_data()
        
        # Create inner optimizer
        inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.inner_lr)
        
        # Inner gradient steps
        for _ in range(self.config.inner_steps):
            inner_optimizer.zero_grad()
            
            outputs = self.model(support_inputs)
            loss = F.cross_entropy(outputs, support_targets)
            
            loss.backward()
            inner_optimizer.step()
        
        # Return adapted parameters
        return {name: param.clone() for name, param in self.model.named_parameters()}

class ProtoNet:
    """Prototypical Networks implementation."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.encoder = None
        self.training_history = deque(maxlen=1000)
        
    def set_encoder(self, encoder: nn.Module):
        """Set the encoder for prototypical networks."""
        self.encoder = encoder
        
    def compute_prototypes(self, support_set: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[int, torch.Tensor]:
        """Compute class prototypes from support set."""
        if self.encoder is None:
            raise ValueError("Encoder not set. Call set_encoder() first.")
        
        prototypes = {}
        
        # Group by class
        class_samples = defaultdict(list)
        for inputs, targets in support_set:
            for i, target in enumerate(targets):
                class_id = target.item()
                class_samples[class_id].append(inputs[i])
        
        # Compute prototypes
        for class_id, samples in class_samples.items():
            if samples:
                samples_tensor = torch.stack(samples)
                embeddings = self.encoder(samples_tensor)
                prototypes[class_id] = embeddings.mean(dim=0)
        
        return prototypes
    
    def meta_train_step(self, meta_batch: List[MetaTask]) -> Dict[str, float]:
        """Perform one meta-training step."""
        if self.encoder is None:
            raise ValueError("Encoder not set. Call set_encoder() first.")
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        for task in meta_batch:
            support_inputs, support_targets = task.get_support_data()
            query_inputs, query_targets = task.get_query_data()
            
            # Compute prototypes
            prototypes = self.compute_prototypes(list(zip(support_inputs, support_targets)))
            
            # Encode query samples
            query_embeddings = self.encoder(query_inputs)
            
            # Compute distances to prototypes
            distances = []
            for query_embedding in query_embeddings:
                query_distances = []
                for class_id, prototype in prototypes.items():
                    distance = F.pairwise_distance(query_embedding.unsqueeze(0), prototype.unsqueeze(0))
                    query_distances.append(distance)
                distances.append(torch.stack(query_distances))
            
            distances = torch.stack(distances)
            
            # Convert distances to probabilities (negative distances)
            logits = -distances
            
            # Compute loss and accuracy
            loss = F.cross_entropy(logits, query_targets)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == query_targets).float().mean().item()
            
            total_loss += loss.item()
            total_accuracy += accuracy
        
        return {
            'meta_loss': total_loss / len(meta_batch),
            'meta_accuracy': total_accuracy / len(meta_batch),
            'num_tasks': len(meta_batch)
        }
    
    def meta_test(self, test_task: MetaTask) -> Dict[str, float]:
        """Meta-test on a new task."""
        if self.encoder is None:
            raise ValueError("Encoder not set. Call set_encoder() first.")
        
        support_inputs, support_targets = test_task.get_support_data()
        query_inputs, query_targets = test_task.get_query_data()
        
        # Compute prototypes
        prototypes = self.compute_prototypes(list(zip(support_inputs, support_targets)))
        
        # Encode query samples
        query_embeddings = self.encoder(query_inputs)
        
        # Compute distances to prototypes
        distances = []
        for query_embedding in query_embeddings:
            query_distances = []
            for class_id, prototype in prototypes.items():
                distance = F.pairwise_distance(query_embedding.unsqueeze(0), prototype.unsqueeze(0))
                query_distances.append(distance)
            distances.append(torch.stack(query_distances))
        
        distances = torch.stack(distances)
        
        # Convert distances to probabilities
        logits = -distances
        
        # Compute loss and accuracy
        loss = F.cross_entropy(logits, query_targets)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_targets).float().mean().item()
        
        return {
            'test_loss': loss.item(),
            'test_accuracy': accuracy
        }

class MetaLearningManager:
    """Main manager for meta-learning."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.algorithms = {
            MetaLearningAlgorithm.MAML: MAML(config),
            MetaLearningAlgorithm.REPTILE: Reptile(config),
            MetaLearningAlgorithm.PROTONET: ProtoNet(config)
        }
        self.training_history = deque(maxlen=1000)
        self.test_history = deque(maxlen=1000)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def set_model(self, model: nn.Module):
        """Set model for meta-learning."""
        for algorithm in self.algorithms.values():
            if hasattr(algorithm, 'set_model'):
                algorithm.set_model(model)
            elif hasattr(algorithm, 'set_encoder'):
                algorithm.set_encoder(model)
    
    def meta_train(self, meta_tasks: List[MetaTask], num_epochs: int = 100) -> Dict[str, Any]:
        """Meta-train the model."""
        logger.info(f"Starting meta-training with {len(meta_tasks)} tasks")
        
        algorithm = self.algorithms.get(self.config.algorithm)
        if not algorithm:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Create meta-batches
            meta_batches = self._create_meta_batches(meta_tasks)
            
            epoch_losses = []
            epoch_accuracies = []
            
            for meta_batch in meta_batches:
                result = algorithm.meta_train_step(meta_batch)
                epoch_losses.append(result['meta_loss'])
                epoch_accuracies.append(result.get('meta_accuracy', 0.0))
            
            # Record training progress
            avg_loss = statistics.mean(epoch_losses)
            avg_accuracy = statistics.mean(epoch_accuracies)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'timestamp': time.time()
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'final_loss': avg_loss,
            'final_accuracy': avg_accuracy,
            'epochs': num_epochs,
            'algorithm': self.config.algorithm.value
        }
    
    def meta_test(self, test_tasks: List[MetaTask]) -> Dict[str, Any]:
        """Meta-test the model."""
        logger.info(f"Meta-testing on {len(test_tasks)} tasks")
        
        algorithm = self.algorithms.get(self.config.algorithm)
        if not algorithm:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        test_losses = []
        test_accuracies = []
        
        for test_task in test_tasks:
            result = algorithm.meta_test(test_task)
            test_losses.append(result['test_loss'])
            test_accuracies.append(result['test_accuracy'])
            
            self.test_history.append({
                'task_id': test_task.task_id,
                'loss': result['test_loss'],
                'accuracy': result['test_accuracy'],
                'timestamp': time.time()
            })
        
        return {
            'avg_loss': statistics.mean(test_losses),
            'avg_accuracy': statistics.mean(test_accuracies),
            'std_loss': statistics.stdev(test_losses) if len(test_losses) > 1 else 0.0,
            'std_accuracy': statistics.stdev(test_accuracies) if len(test_accuracies) > 1 else 0.0,
            'num_tasks': len(test_tasks)
        }
    
    def _create_meta_batches(self, meta_tasks: List[MetaTask]) -> List[List[MetaTask]]:
        """Create meta-batches from tasks."""
        batches = []
        
        for i in range(0, len(meta_tasks), self.config.meta_batch_size):
            batch = meta_tasks[i:i + self.config.meta_batch_size]
            batches.append(batch)
        
        return batches
    
    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        return {
            'algorithm': self.config.algorithm.value,
            'task_type': self.config.task_type.value,
            'mode': self.config.mode.value,
            'training_history_size': len(self.training_history),
            'test_history_size': len(self.test_history),
            'config': {
                'inner_lr': self.config.inner_lr,
                'outer_lr': self.config.outer_lr,
                'inner_steps': self.config.inner_steps,
                'outer_steps': self.config.outer_steps,
                'support_shots': self.config.support_shots,
                'query_shots': self.config.query_shots
            }
        }

# Factory functions
def create_meta_learning_config(algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML,
                               task_type: MetaLearningTask = MetaLearningTask.FEW_SHOT_CLASSIFICATION,
                               **kwargs) -> MetaLearningConfig:
    """Create meta-learning configuration."""
    return MetaLearningConfig(
        algorithm=algorithm,
        task_type=task_type,
        **kwargs
    )

def create_meta_task(task_id: str, support_set: List[Tuple[torch.Tensor, torch.Tensor]], 
                    query_set: List[Tuple[torch.Tensor, torch.Tensor]], 
                    task_type: MetaLearningTask) -> MetaTask:
    """Create a meta-learning task."""
    return MetaTask(task_id, support_set, query_set, task_type)

def create_maml(config: MetaLearningConfig) -> MAML:
    """Create MAML algorithm."""
    return MAML(config)

def create_reptile(config: MetaLearningConfig) -> Reptile:
    """Create Reptile algorithm."""
    return Reptile(config)

def create_protonet(config: MetaLearningConfig) -> ProtoNet:
    """Create ProtoNet algorithm."""
    return ProtoNet(config)

def create_meta_learning_manager(config: Optional[MetaLearningConfig] = None) -> MetaLearningManager:
    """Create meta-learning manager."""
    if config is None:
        config = create_meta_learning_config()
    return MetaLearningManager(config)

# Example usage
def example_meta_learning():
    """Example of meta-learning."""
    # Create configuration
    config = create_meta_learning_config(
        algorithm=MetaLearningAlgorithm.MAML,
        task_type=MetaLearningTask.FEW_SHOT_CLASSIFICATION,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5
    )
    
    # Create manager
    manager = create_meta_learning_manager(config)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Set model
    manager.set_model(model)
    
    # Create sample tasks
    tasks = []
    for i in range(10):
        # Generate random support and query sets
        support_inputs = torch.randn(5, 10)
        support_targets = torch.randint(0, 2, (5,))
        query_inputs = torch.randn(15, 10)
        query_targets = torch.randint(0, 2, (15,))
        
        task = create_meta_task(
            f"task_{i}",
            list(zip(support_inputs, support_targets)),
            list(zip(query_inputs, query_targets)),
            MetaLearningTask.FEW_SHOT_CLASSIFICATION
        )
        tasks.append(task)
    
    # Meta-train
    training_result = manager.meta_train(tasks, num_epochs=50)
    print(f"Training result: {training_result}")
    
    # Meta-test
    test_tasks = tasks[:5]  # Use first 5 tasks for testing
    test_result = manager.meta_test(test_tasks)
    print(f"Test result: {test_result}")
    
    # Get statistics
    stats = manager.get_meta_learning_statistics()
    print(f"Statistics: {stats}")
    
    return training_result, test_result

if __name__ == "__main__":
    # Run example
    example_meta_learning()
