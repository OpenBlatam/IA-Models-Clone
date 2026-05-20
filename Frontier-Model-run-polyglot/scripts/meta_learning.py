#!/usr/bin/env python3
"""
Advanced Meta-Learning Framework for Frontier Model Training
Provides comprehensive few-shot learning, meta-optimization, and learning-to-learn capabilities.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import higher
import learn2learn
from learn2learn.algorithms import MAML, MetaSGD, Reptile, FOMAML
from learn2learn.data import MetaDataset, TaskDataset
import torchmeta
from torchmeta.datasets import Omniglot, MiniImagenet, CIFARFS, FC100
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class MetaLearningAlgorithm(Enum):
    """Meta-learning algorithms."""
    MAML = "maml"
    META_SGD = "meta_sgd"
    REPTILE = "reptile"
    FOMAML = "fomaml"
    PROTONET = "protonet"
    MATCHING_NETWORKS = "matching_networks"
    RELATION_NETWORKS = "relation_networks"
    META_LSTM = "meta_lstm"
    META_LEARNER_LSTM = "meta_learner_lstm"
    GRADIENT_BASED_META_LEARNING = "gradient_based_meta_learning"

class TaskDistribution(Enum):
    """Task distribution types."""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    MULTIMODAL = "multimodal"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"

class MetaOptimizer(Enum):
    """Meta-optimizers."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    META_ADAM = "meta_adam"
    META_SGD = "meta_sgd"

class FewShotSetting(Enum):
    """Few-shot learning settings."""
    N_WAY_K_SHOT = "n_way_k_shot"
    FEW_SHOT_CLASSIFICATION = "few_shot_classification"
    FEW_SHOT_REGRESSION = "few_shot_regression"
    ONE_SHOT_LEARNING = "one_shot_learning"
    ZERO_SHOT_LEARNING = "zero_shot_learning"
    CONTINUAL_LEARNING = "continual_learning"

@dataclass
class MetaLearningConfig:
    """Meta-learning configuration."""
    algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML
    task_distribution: TaskDistribution = TaskDistribution.UNIFORM
    meta_optimizer: MetaOptimizer = MetaOptimizer.ADAM
    few_shot_setting: FewShotSetting = FewShotSetting.N_WAY_K_SHOT
    n_way: int = 5
    k_shot: int = 5
    meta_lr: float = 0.01
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_steps: int = 100
    batch_size: int = 32
    num_tasks: int = 1000
    enable_second_order: bool = True
    enable_meta_regularization: bool = True
    enable_task_augmentation: bool = True
    enable_adaptive_inner_lr: bool = True
    enable_meta_curriculum: bool = True
    enable_uncertainty_quantification: bool = True
    enable_meta_interpretability: bool = True
    device: str = "auto"

@dataclass
class MetaTask:
    """Meta-learning task."""
    task_id: str
    support_set: Tuple[np.ndarray, np.ndarray]
    query_set: Tuple[np.ndarray, np.ndarray]
    task_metadata: Dict[str, Any]
    difficulty: float
    created_at: datetime

@dataclass
class MetaLearningResult:
    """Meta-learning result."""
    result_id: str
    algorithm: MetaLearningAlgorithm
    performance_metrics: Dict[str, float]
    adaptation_metrics: Dict[str, float]
    meta_learning_curve: List[float]
    task_performance: Dict[str, float]
    meta_model_state: Dict[str, Any]
    created_at: datetime

class MetaDatasetGenerator:
    """Meta-dataset generator."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_meta_tasks(self, dataset: str = "synthetic", num_tasks: int = None) -> List[MetaTask]:
        """Generate meta-learning tasks."""
        if num_tasks is None:
            num_tasks = self.config.num_tasks
        
        console.print(f"[blue]Generating {num_tasks} meta-learning tasks...[/blue]")
        
        tasks = []
        
        if dataset == "synthetic":
            tasks = self._generate_synthetic_tasks(num_tasks)
        elif dataset == "omniglot":
            tasks = self._generate_omniglot_tasks(num_tasks)
        elif dataset == "miniimagenet":
            tasks = self._generate_miniimagenet_tasks(num_tasks)
        else:
            tasks = self._generate_synthetic_tasks(num_tasks)
        
        console.print(f"[green]Generated {len(tasks)} meta-learning tasks[/green]")
        return tasks
    
    def _generate_synthetic_tasks(self, num_tasks: int) -> List[MetaTask]:
        """Generate synthetic meta-learning tasks."""
        tasks = []
        
        for i in range(num_tasks):
            # Generate task parameters
            task_params = self._sample_task_parameters()
            
            # Generate support set
            support_x, support_y = self._generate_task_data(task_params, self.config.k_shot * self.config.n_way)
            
            # Generate query set
            query_x, query_y = self._generate_task_data(task_params, self.config.k_shot * self.config.n_way)
            
            # Create meta task
            task = MetaTask(
                task_id=f"synthetic_task_{i}",
                support_set=(support_x, support_y),
                query_set=(query_x, query_y),
                task_metadata={
                    'task_params': task_params,
                    'n_way': self.config.n_way,
                    'k_shot': self.config.k_shot
                },
                difficulty=self._calculate_task_difficulty(task_params),
                created_at=datetime.now()
            )
            
            tasks.append(task)
        
        return tasks
    
    def _sample_task_parameters(self) -> Dict[str, Any]:
        """Sample task parameters."""
        if self.config.task_distribution == TaskDistribution.UNIFORM:
            return {
                'center': np.random.uniform(-2, 2, 2),
                'scale': np.random.uniform(0.5, 2.0),
                'rotation': np.random.uniform(0, 2 * np.pi)
            }
        elif self.config.task_distribution == TaskDistribution.GAUSSIAN:
            return {
                'center': np.random.normal(0, 1, 2),
                'scale': np.random.normal(1, 0.3),
                'rotation': np.random.normal(0, np.pi/4)
            }
        else:
            return {
                'center': np.random.uniform(-1, 1, 2),
                'scale': 1.0,
                'rotation': 0.0
            }
    
    def _generate_task_data(self, task_params: Dict[str, Any], num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data for a specific task."""
        center = task_params['center']
        scale = task_params['scale']
        rotation = task_params['rotation']
        
        # Generate class centers
        class_centers = []
        for i in range(self.config.n_way):
            angle = 2 * np.pi * i / self.config.n_way + rotation
            class_center = center + scale * np.array([np.cos(angle), np.sin(angle)])
            class_centers.append(class_center)
        
        # Generate samples for each class
        X = []
        y = []
        
        samples_per_class = num_samples // self.config.n_way
        
        for class_idx, class_center in enumerate(class_centers):
            # Generate samples around class center
            class_samples = np.random.normal(class_center, 0.1, (samples_per_class, 2))
            X.extend(class_samples)
            y.extend([class_idx] * samples_per_class)
        
        return np.array(X), np.array(y)
    
    def _calculate_task_difficulty(self, task_params: Dict[str, Any]) -> float:
        """Calculate task difficulty."""
        # Simple difficulty metric based on scale and separation
        scale = task_params['scale']
        center_norm = np.linalg.norm(task_params['center'])
        
        # Higher scale and center distance = easier task
        difficulty = 1.0 / (scale + center_norm + 1e-8)
        return min(difficulty, 1.0)
    
    def _generate_omniglot_tasks(self, num_tasks: int) -> List[MetaTask]:
        """Generate Omniglot meta-learning tasks."""
        # Simplified Omniglot task generation
        tasks = []
        
        for i in range(num_tasks):
            # Generate synthetic Omniglot-like tasks
            support_x, support_y = self._generate_omniglot_data(self.config.k_shot * self.config.n_way)
            query_x, query_y = self._generate_omniglot_data(self.config.k_shot * self.config.n_way)
            
            task = MetaTask(
                task_id=f"omniglot_task_{i}",
                support_set=(support_x, support_y),
                query_set=(query_x, query_y),
                task_metadata={
                    'dataset': 'omniglot',
                    'n_way': self.config.n_way,
                    'k_shot': self.config.k_shot
                },
                difficulty=0.5,  # Fixed difficulty for now
                created_at=datetime.now()
            )
            
            tasks.append(task)
        
        return tasks
    
    def _generate_omniglot_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Omniglot-like data."""
        # Generate synthetic character-like data
        X = np.random.randn(num_samples, 28, 28)  # 28x28 images
        y = np.random.randint(0, self.config.n_way, num_samples)
        
        return X, y
    
    def _generate_miniimagenet_tasks(self, num_tasks: int) -> List[MetaTask]:
        """Generate Mini-ImageNet meta-learning tasks."""
        # Simplified Mini-ImageNet task generation
        tasks = []
        
        for i in range(num_tasks):
            # Generate synthetic Mini-ImageNet-like tasks
            support_x, support_y = self._generate_miniimagenet_data(self.config.k_shot * self.config.n_way)
            query_x, query_y = self._generate_miniimagenet_data(self.config.k_shot * self.config.n_way)
            
            task = MetaTask(
                task_id=f"miniimagenet_task_{i}",
                support_set=(support_x, support_y),
                query_set=(query_x, query_y),
                task_metadata={
                    'dataset': 'miniimagenet',
                    'n_way': self.config.n_way,
                    'k_shot': self.config.k_shot
                },
                difficulty=0.7,  # Fixed difficulty for now
                created_at=datetime.now()
            )
            
            tasks.append(task)
        
        return tasks
    
    def _generate_miniimagenet_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Mini-ImageNet-like data."""
        # Generate synthetic image-like data
        X = np.random.randn(num_samples, 3, 84, 84)  # 3x84x84 images
        y = np.random.randint(0, self.config.n_way, num_samples)
        
        return X, y

class MAMLLearner:
    """Model-Agnostic Meta-Learning (MAML) implementation."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_meta_model(self, input_size: int, output_size: int) -> nn.Module:
        """Create meta-learning model."""
        model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        ).to(self.device)
        
        return model
    
    def meta_train(self, model: nn.Module, meta_tasks: List[MetaTask]) -> Dict[str, Any]:
        """Meta-train the model using MAML."""
        console.print("[blue]Starting MAML meta-training...[/blue]")
        
        # Initialize meta-optimizer
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.meta_lr)
        criterion = nn.CrossEntropyLoss()
        
        meta_losses = []
        meta_accuracies = []
        
        model.train()
        
        for meta_step in range(self.config.meta_steps):
            meta_loss = 0.0
            meta_accuracy = 0.0
            
            # Sample batch of tasks
            task_batch = np.random.choice(meta_tasks, size=min(self.config.batch_size, len(meta_tasks)), replace=False)
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in model.named_parameters()}
            
            for task in task_batch:
                # Inner loop: adapt to task
                adapted_model = self._adapt_to_task(model, task, criterion)
                
                # Outer loop: compute meta-loss on query set
                query_x, query_y = task.query_set
                query_x_tensor = torch.FloatTensor(query_x).to(self.device)
                query_y_tensor = torch.LongTensor(query_y).to(self.device)
                
                query_output = adapted_model(query_x_tensor)
                task_loss = criterion(query_output, query_y_tensor)
                
                # Compute gradients w.r.t. original parameters
                task_loss.backward()
                
                meta_loss += task_loss.item()
                
                # Compute accuracy
                with torch.no_grad():
                    predictions = torch.argmax(query_output, dim=1)
                    accuracy = (predictions == query_y_tensor).float().mean().item()
                    meta_accuracy += accuracy
            
            # Average meta-loss
            meta_loss /= len(task_batch)
            meta_accuracy /= len(task_batch)
            
            # Update meta-parameters
            meta_optimizer.step()
            meta_optimizer.zero_grad()
            
            # Restore original parameters for next iteration
            for name, param in model.named_parameters():
                param.data = original_params[name].data
            
            meta_losses.append(meta_loss)
            meta_accuracies.append(meta_accuracy)
            
            if meta_step % 10 == 0:
                console.print(f"[blue]Meta-step {meta_step}: Loss = {meta_loss:.4f}, Accuracy = {meta_accuracy:.4f}[/blue]")
        
        return {
            'meta_losses': meta_losses,
            'meta_accuracies': meta_accuracies,
            'final_meta_loss': meta_losses[-1] if meta_losses else 0.0,
            'final_meta_accuracy': meta_accuracies[-1] if meta_accuracies else 0.0
        }
    
    def _adapt_to_task(self, model: nn.Module, task: MetaTask, criterion: nn.CrossEntropyLoss) -> nn.Module:
        """Adapt model to a specific task."""
        # Create a copy of the model for adaptation
        adapted_model = type(model)(*[layer for layer in model.children()]).to(self.device)
        adapted_model.load_state_dict(model.state_dict())
        
        # Inner loop optimization
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)
        
        support_x, support_y = task.support_set
        support_x_tensor = torch.FloatTensor(support_x).to(self.device)
        support_y_tensor = torch.LongTensor(support_y).to(self.device)
        
        for inner_step in range(self.config.inner_steps):
            inner_optimizer.zero_grad()
            
            support_output = adapted_model(support_x_tensor)
            inner_loss = criterion(support_output, support_y_tensor)
            
            inner_loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def evaluate_meta_model(self, model: nn.Module, test_tasks: List[MetaTask]) -> Dict[str, float]:
        """Evaluate meta-learning model on test tasks."""
        console.print("[blue]Evaluating meta-learning model...[/blue]")
        
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_accuracies = []
        test_losses = []
        adaptation_times = []
        
        for task in test_tasks:
            start_time = time.time()
            
            # Adapt to task
            adapted_model = self._adapt_to_task(model, task, criterion)
            
            adaptation_time = time.time() - start_time
            adaptation_times.append(adaptation_time)
            
            # Evaluate on query set
            query_x, query_y = task.query_set
            query_x_tensor = torch.FloatTensor(query_x).to(self.device)
            query_y_tensor = torch.LongTensor(query_y).to(self.device)
            
            with torch.no_grad():
                query_output = adapted_model(query_x_tensor)
                query_loss = criterion(query_output, query_y_tensor)
                
                predictions = torch.argmax(query_output, dim=1)
                accuracy = (predictions == query_y_tensor).float().mean().item()
                
                test_accuracies.append(accuracy)
                test_losses.append(query_loss.item())
        
        return {
            'test_accuracy': np.mean(test_accuracies),
            'test_accuracy_std': np.std(test_accuracies),
            'test_loss': np.mean(test_losses),
            'test_loss_std': np.std(test_losses),
            'adaptation_time': np.mean(adaptation_times),
            'adaptation_time_std': np.std(adaptation_times)
        }

class ReptileLearner:
    """Reptile meta-learning implementation."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def meta_train(self, model: nn.Module, meta_tasks: List[MetaTask]) -> Dict[str, Any]:
        """Meta-train the model using Reptile."""
        console.print("[blue]Starting Reptile meta-training...[/blue]")
        
        meta_losses = []
        meta_accuracies = []
        
        model.train()
        
        for meta_step in range(self.config.meta_steps):
            meta_loss = 0.0
            meta_accuracy = 0.0
            
            # Sample batch of tasks
            task_batch = np.random.choice(meta_tasks, size=min(self.config.batch_size, len(meta_tasks)), replace=False)
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in model.named_parameters()}
            
            for task in task_batch:
                # Inner loop: adapt to task
                adapted_model = self._adapt_to_task(model, task)
                
                # Compute meta-loss on query set
                query_x, query_y = task.query_set
                query_x_tensor = torch.FloatTensor(query_x).to(self.device)
                query_y_tensor = torch.LongTensor(query_y).to(self.device)
                
                query_output = adapted_model(query_x_tensor)
                task_loss = F.cross_entropy(query_output, query_y_tensor)
                
                meta_loss += task_loss.item()
                
                # Compute accuracy
                with torch.no_grad():
                    predictions = torch.argmax(query_output, dim=1)
                    accuracy = (predictions == query_y_tensor).float().mean().item()
                    meta_accuracy += accuracy
            
            # Average meta-loss
            meta_loss /= len(task_batch)
            meta_accuracy /= len(task_batch)
            
            # Reptile update: move towards adapted parameters
            for name, param in model.named_parameters():
                adapted_param = next(adapted_model.named_parameters())[1]
                param.data = param.data + self.config.meta_lr * (adapted_param.data - param.data)
            
            meta_losses.append(meta_loss)
            meta_accuracies.append(meta_accuracy)
            
            if meta_step % 10 == 0:
                console.print(f"[blue]Meta-step {meta_step}: Loss = {meta_loss:.4f}, Accuracy = {meta_accuracy:.4f}[/blue]")
        
        return {
            'meta_losses': meta_losses,
            'meta_accuracies': meta_accuracies,
            'final_meta_loss': meta_losses[-1] if meta_losses else 0.0,
            'final_meta_accuracy': meta_accuracies[-1] if meta_accuracies else 0.0
        }
    
    def _adapt_to_task(self, model: nn.Module, task: MetaTask) -> nn.Module:
        """Adapt model to a specific task."""
        # Create a copy of the model for adaptation
        adapted_model = type(model)(*[layer for layer in model.children()]).to(self.device)
        adapted_model.load_state_dict(model.state_dict())
        
        # Inner loop optimization
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)
        
        support_x, support_y = task.support_set
        support_x_tensor = torch.FloatTensor(support_x).to(self.device)
        support_y_tensor = torch.LongTensor(support_y).to(self.device)
        
        for inner_step in range(self.config.inner_steps):
            inner_optimizer.zero_grad()
            
            support_output = adapted_model(support_x_tensor)
            inner_loss = F.cross_entropy(support_output, support_y_tensor)
            
            inner_loss.backward()
            inner_optimizer.step()
        
        return adapted_model

class MetaLearningSystem:
    """Main meta-learning system."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dataset_generator = MetaDatasetGenerator(config)
        
        # Initialize learners
        self.maml_learner = MAMLLearner(config)
        self.reptile_learner = ReptileLearner(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.meta_results: Dict[str, MetaLearningResult] = {}
    
    def _init_database(self) -> str:
        """Initialize meta-learning database."""
        db_path = Path("./meta_learning.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_tasks (
                    task_id TEXT PRIMARY KEY,
                    support_set TEXT NOT NULL,
                    query_set TEXT NOT NULL,
                    task_metadata TEXT NOT NULL,
                    difficulty REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_results (
                    result_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    adaptation_metrics TEXT NOT NULL,
                    meta_learning_curve TEXT NOT NULL,
                    task_performance TEXT NOT NULL,
                    meta_model_state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_meta_learning_experiment(self, dataset: str = "synthetic", 
                                    num_tasks: int = None) -> MetaLearningResult:
        """Run complete meta-learning experiment."""
        console.print("[blue]Starting meta-learning experiment...[/blue]")
        
        start_time = time.time()
        result_id = f"meta_exp_{int(time.time())}"
        
        # Generate meta-tasks
        meta_tasks = self.dataset_generator.generate_meta_tasks(dataset, num_tasks)
        
        # Split into train and test tasks
        train_tasks = meta_tasks[:int(0.8 * len(meta_tasks))]
        test_tasks = meta_tasks[int(0.8 * len(meta_tasks)):]
        
        # Create meta-model
        input_size = 2  # For synthetic 2D data
        output_size = self.config.n_way
        meta_model = self.maml_learner.create_meta_model(input_size, output_size)
        
        # Meta-train
        if self.config.algorithm == MetaLearningAlgorithm.MAML:
            training_result = self.maml_learner.meta_train(meta_model, train_tasks)
            evaluation_result = self.maml_learner.evaluate_meta_model(meta_model, test_tasks)
        elif self.config.algorithm == MetaLearningAlgorithm.REPTILE:
            training_result = self.reptile_learner.meta_train(meta_model, train_tasks)
            evaluation_result = self.maml_learner.evaluate_meta_model(meta_model, test_tasks)
        else:
            training_result = self.maml_learner.meta_train(meta_model, train_tasks)
            evaluation_result = self.maml_learner.evaluate_meta_model(meta_model, test_tasks)
        
        # Create meta-learning result
        meta_result = MetaLearningResult(
            result_id=result_id,
            algorithm=self.config.algorithm,
            performance_metrics=evaluation_result,
            adaptation_metrics={
                'adaptation_time': evaluation_result.get('adaptation_time', 0),
                'adaptation_stability': 1.0 - evaluation_result.get('test_accuracy_std', 0)
            },
            meta_learning_curve=training_result.get('meta_accuracies', []),
            task_performance={
                'train_tasks': len(train_tasks),
                'test_tasks': len(test_tasks),
                'avg_task_difficulty': np.mean([task.difficulty for task in meta_tasks])
            },
            meta_model_state={
                'input_size': input_size,
                'output_size': output_size,
                'num_parameters': sum(p.numel() for p in meta_model.parameters())
            },
            created_at=datetime.now()
        )
        
        # Store result
        self.meta_results[result_id] = meta_result
        
        # Save to database
        self._save_meta_result(meta_result)
        
        experiment_time = time.time() - start_time
        console.print(f"[green]Meta-learning experiment completed in {experiment_time:.2f} seconds[/green]")
        console.print(f"[blue]Test accuracy: {evaluation_result.get('test_accuracy', 0):.4f}[/blue]")
        console.print(f"[blue]Adaptation time: {evaluation_result.get('adaptation_time', 0):.4f}s[/blue]")
        
        return meta_result
    
    def _save_meta_result(self, result: MetaLearningResult):
        """Save meta-learning result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO meta_results 
                (result_id, algorithm, performance_metrics, adaptation_metrics,
                 meta_learning_curve, task_performance, meta_model_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.algorithm.value,
                json.dumps(result.performance_metrics),
                json.dumps(result.adaptation_metrics),
                json.dumps(result.meta_learning_curve),
                json.dumps(result.task_performance),
                json.dumps(result.meta_model_state),
                result.created_at.isoformat()
            ))
    
    def visualize_meta_learning_results(self, result: MetaLearningResult, 
                                      output_path: str = None) -> str:
        """Visualize meta-learning results."""
        if output_path is None:
            output_path = f"meta_learning_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Meta-learning curve
        if result.meta_learning_curve:
            axes[0, 0].plot(result.meta_learning_curve)
            axes[0, 0].set_title('Meta-Learning Curve')
            axes[0, 0].set_xlabel('Meta-Step')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        performance_metrics = result.performance_metrics
        metric_names = list(performance_metrics.keys())
        metric_values = list(performance_metrics.values())
        
        axes[0, 1].bar(metric_names, metric_values)
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Adaptation metrics
        adaptation_metrics = result.adaptation_metrics
        adapt_names = list(adaptation_metrics.keys())
        adapt_values = list(adaptation_metrics.values())
        
        axes[1, 0].bar(adapt_names, adapt_values)
        axes[1, 0].set_title('Adaptation Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Task performance
        task_performance = result.task_performance
        task_names = list(task_performance.keys())
        task_values = list(task_performance.values())
        
        axes[1, 1].bar(task_names, task_values)
        axes[1, 1].set_title('Task Performance')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Meta-learning visualization saved: {output_path}[/green]")
        return output_path
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get meta-learning system summary."""
        if not self.meta_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.meta_results)
        
        # Calculate average performance
        test_accuracies = [result.performance_metrics.get('test_accuracy', 0) 
                          for result in self.meta_results.values()]
        adaptation_times = [result.performance_metrics.get('adaptation_time', 0) 
                          for result in self.meta_results.values()]
        
        avg_test_accuracy = np.mean(test_accuracies)
        avg_adaptation_time = np.mean(adaptation_times)
        
        # Best performing experiment
        best_result = max(self.meta_results.values(), 
                         key=lambda x: x.performance_metrics.get('test_accuracy', 0))
        
        return {
            'total_experiments': total_experiments,
            'average_test_accuracy': avg_test_accuracy,
            'average_adaptation_time': avg_adaptation_time,
            'best_test_accuracy': best_result.performance_metrics.get('test_accuracy', 0),
            'best_experiment_id': best_result.result_id,
            'algorithms_used': list(set(result.algorithm.value for result in self.meta_results.values()))
        }

def main():
    """Main function for meta-learning CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Learning Framework")
    parser.add_argument("--algorithm", type=str,
                       choices=["maml", "reptile", "meta_sgd"],
                       default="maml", help="Meta-learning algorithm")
    parser.add_argument("--task-distribution", type=str,
                       choices=["uniform", "gaussian", "multimodal"],
                       default="uniform", help="Task distribution")
    parser.add_argument("--meta-optimizer", type=str,
                       choices=["adam", "sgd", "rmsprop"],
                       default="adam", help="Meta-optimizer")
    parser.add_argument("--few-shot-setting", type=str,
                       choices=["n_way_k_shot", "few_shot_classification", "one_shot_learning"],
                       default="n_way_k_shot", help="Few-shot setting")
    parser.add_argument("--n-way", type=int, default=5,
                       help="Number of classes per task")
    parser.add_argument("--k-shot", type=int, default=5,
                       help="Number of support examples per class")
    parser.add_argument("--meta-lr", type=float, default=0.01,
                       help="Meta-learning rate")
    parser.add_argument("--inner-lr", type=float, default=0.01,
                       help="Inner loop learning rate")
    parser.add_argument("--inner-steps", type=int, default=5,
                       help="Inner loop steps")
    parser.add_argument("--meta-steps", type=int, default=100,
                       help="Meta-training steps")
    parser.add_argument("--num-tasks", type=int, default=1000,
                       help="Number of meta-tasks")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create meta-learning configuration
    config = MetaLearningConfig(
        algorithm=MetaLearningAlgorithm(args.algorithm),
        task_distribution=TaskDistribution(args.task_distribution),
        meta_optimizer=MetaOptimizer(args.meta_optimizer),
        few_shot_setting=FewShotSetting(args.few_shot_setting),
        n_way=args.n_way,
        k_shot=args.k_shot,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_steps=args.meta_steps,
        num_tasks=args.num_tasks,
        device=args.device
    )
    
    # Create meta-learning system
    meta_system = MetaLearningSystem(config)
    
    # Run meta-learning experiment
    result = meta_system.run_meta_learning_experiment(dataset="synthetic", num_tasks=args.num_tasks)
    
    # Show results
    console.print(f"[green]Meta-learning experiment completed[/green]")
    console.print(f"[blue]Algorithm: {result.algorithm.value}[/blue]")
    console.print(f"[blue]Test accuracy: {result.performance_metrics.get('test_accuracy', 0):.4f}[/blue]")
    console.print(f"[blue]Adaptation time: {result.performance_metrics.get('adaptation_time', 0):.4f}s[/blue]")
    
    # Create visualization
    meta_system.visualize_meta_learning_results(result)
    
    # Show summary
    summary = meta_system.get_meta_learning_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
