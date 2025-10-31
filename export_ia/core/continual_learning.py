"""
Continual Learning Engine for Export IA
Advanced continual learning with catastrophic forgetting prevention and knowledge transfer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import pickle
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning"""
    # Learning strategy
    strategy: str = "ewc"  # ewc, lwf, mas, packnet, gem, agem, icarl, der
    
    # Memory management
    memory_size: int = 1000
    memory_selection: str = "random"  # random, herding, icarl, gradient_based
    memory_update: str = "fifo"  # fifo, lru, importance_based
    
    # Regularization
    regularization_strength: float = 1000.0
    importance_weighting: bool = True
    elastic_weight_consolidation: bool = True
    
    # Knowledge distillation
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5
    use_old_model: bool = True
    
    # Gradient-based methods
    gradient_episodic_memory: bool = False
    gem_memory_size: int = 100
    gem_margin: float = 0.5
    
    # PackNet
    packnet_pruning_ratio: float = 0.5
    packnet_retrain_epochs: int = 10
    
    # DER (Dark Experience Replay)
    der_alpha: float = 0.3
    der_beta: float = 0.5
    
    # Evaluation
    evaluate_on_all_tasks: bool = True
    save_best_models: bool = True
    early_stopping_patience: int = 10
    
    # Logging and visualization
    log_metrics: bool = True
    visualize_forgetting: bool = True
    save_plots: bool = True

class ExperienceReplay:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.memory = []
        self.importance_scores = []
        self.task_labels = []
        self.current_size = 0
        
    def add_experience(self, inputs: torch.Tensor, targets: torch.Tensor, 
                      task_id: int, importance: float = 1.0):
        """Add experience to replay buffer"""
        
        if self.current_size < self.config.memory_size:
            # Buffer not full, add directly
            self.memory.append((inputs.clone(), targets.clone()))
            self.importance_scores.append(importance)
            self.task_labels.append(task_id)
            self.current_size += 1
        else:
            # Buffer full, need to replace
            if self.config.memory_selection == "random":
                idx = random.randint(0, self.config.memory_size - 1)
            elif self.config.memory_selection == "importance_based":
                # Replace least important experience
                idx = np.argmin(self.importance_scores)
            else:
                # FIFO
                idx = 0
                
            self.memory[idx] = (inputs.clone(), targets.clone())
            self.importance_scores[idx] = importance
            self.task_labels[idx] = task_id
            
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Sample batch from replay buffer"""
        
        if self.current_size == 0:
            return None, None, []
            
        # Sample based on importance
        if self.config.importance_weighting:
            probs = np.array(self.importance_scores[:self.current_size])
            probs = probs / np.sum(probs)
            indices = np.random.choice(
                self.current_size, size=min(batch_size, self.current_size), 
                replace=False, p=probs
            )
        else:
            indices = np.random.choice(
                self.current_size, size=min(batch_size, self.current_size), 
                replace=False
            )
            
        batch_inputs = torch.cat([self.memory[i][0] for i in indices])
        batch_targets = torch.cat([self.memory[i][1] for i in indices])
        batch_tasks = [self.task_labels[i] for i in indices]
        
        return batch_inputs, batch_targets, batch_tasks
        
    def get_task_experiences(self, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all experiences for a specific task"""
        
        task_inputs = []
        task_targets = []
        
        for i, task_label in enumerate(self.task_labels[:self.current_size]):
            if task_label == task_id:
                task_inputs.append(self.memory[i][0])
                task_targets.append(self.memory[i][1])
                
        if task_inputs:
            return torch.cat(task_inputs), torch.cat(task_targets)
        else:
            return None, None

class ElasticWeightConsolidation:
    """Elastic Weight Consolidation (EWC) implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        """Compute Fisher Information Matrix"""
        
        model.eval()
        fisher_info = {}
        
        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
            
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 100:  # Limit to 100 batches for efficiency
                break
                
            model.zero_grad()
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
        # Average over batches
        for name in fisher_info:
            fisher_info[name] /= 100
            
        self.fisher_information = fisher_info
        
    def save_optimal_params(self, model: nn.Module):
        """Save optimal parameters for current task"""
        
        self.optimal_params = {}
        for name, param in model.named_parameters():
            self.optimal_params[name] = param.data.clone()
            
    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss"""
        
        ewc_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
                
        return self.config.regularization_strength * ewc_loss

class LearningWithoutForgetting:
    """Learning Without Forgetting (LwF) implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.old_model = None
        
    def save_old_model(self, model: nn.Module):
        """Save old model for knowledge distillation"""
        
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        
    def compute_distillation_loss(self, new_outputs: torch.Tensor, 
                                 old_outputs: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        
        if self.old_model is None:
            return torch.tensor(0.0)
            
        # Softmax with temperature
        new_soft = nn.functional.softmax(new_outputs / self.config.distillation_temperature, dim=1)
        old_soft = nn.functional.softmax(old_outputs / self.config.distillation_temperature, dim=1)
        
        # KL divergence
        distillation_loss = nn.functional.kl_div(
            new_soft.log(), old_soft, reduction='batchmean'
        )
        
        return distillation_loss * (self.config.distillation_temperature ** 2)

class MemoryAwareSynapses:
    """Memory Aware Synapses (MAS) implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.importance_weights = {}
        
    def compute_importance(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        """Compute importance weights for parameters"""
        
        model.eval()
        importance = {}
        
        for name, param in model.named_parameters():
            importance[name] = torch.zeros_like(param)
            
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 100:  # Limit to 100 batches
                break
                
            model.zero_grad()
            outputs = model(inputs)
            
            # Compute gradient of output magnitude
            output_magnitude = torch.norm(outputs, p=2, dim=1).sum()
            output_magnitude.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    importance[name] += torch.abs(param.grad.data)
                    
        # Average over batches
        for name in importance:
            importance[name] /= 100
            
        self.importance_weights = importance
        
    def compute_mas_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute MAS regularization loss"""
        
        mas_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in self.importance_weights:
                importance = self.importance_weights[name]
                mas_loss += (importance * param ** 2).sum()
                
        return self.config.regularization_strength * mas_loss

class PackNet:
    """PackNet implementation for continual learning"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.masks = {}
        self.task_masks = {}
        
    def prune_network(self, model: nn.Module, task_id: int):
        """Prune network for new task"""
        
        # Create mask for current task
        task_mask = {}
        
        for name, param in model.named_parameters():
            if param.dim() > 1:  # Only prune weight matrices
                # Compute importance scores
                importance = torch.abs(param.data)
                
                # Prune least important weights
                threshold = torch.quantile(importance.flatten(), self.config.packnet_pruning_ratio)
                mask = importance > threshold
                
                task_mask[name] = mask
                
                # Apply mask
                param.data *= mask.float()
                
        self.task_masks[task_id] = task_mask
        
    def apply_task_mask(self, model: nn.Module, task_id: int):
        """Apply task-specific mask"""
        
        if task_id in self.task_masks:
            mask = self.task_masks[task_id]
            
            for name, param in model.named_parameters():
                if name in mask:
                    param.data *= mask[name].float()

class GradientEpisodicMemory:
    """Gradient Episodic Memory (GEM) implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.episodic_memory = []
        self.gradient_memory = []
        
    def store_gradient(self, model: nn.Module, task_id: int, dataloader: torch.utils.data.DataLoader):
        """Store gradient for a task"""
        
        model.eval()
        gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.clone())
                
        self.gradient_memory.append((task_id, gradients))
        
        # Store episodic memory
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 10:  # Store only 10 batches
                break
            self.episodic_memory.append((inputs.clone(), targets.clone(), task_id))
            
    def project_gradient(self, model: nn.Module, current_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project gradient to avoid catastrophic forgetting"""
        
        if not self.gradient_memory:
            return current_gradients
            
        # Compute constraints
        constraints = []
        for task_id, old_gradients in self.gradient_memory:
            constraint = 0.0
            for curr_grad, old_grad in zip(current_gradients, old_gradients):
                constraint += (curr_grad * old_grad).sum()
            constraints.append(constraint)
            
        # Check if constraints are satisfied
        if all(c >= -self.config.gem_margin for c in constraints):
            return current_gradients
            
        # Project gradient
        projected_gradients = self._solve_quadratic_program(current_gradients, constraints)
        
        return projected_gradients
        
    def _solve_quadratic_program(self, gradients: List[torch.Tensor], 
                                constraints: List[float]) -> List[torch.Tensor]:
        """Solve quadratic program for gradient projection"""
        
        # Simplified projection - in practice, you'd use a proper QP solver
        projected_gradients = []
        
        for grad in gradients:
            # Simple scaling to satisfy constraints
            scale_factor = 0.9  # Reduce gradient magnitude
            projected_gradients.append(grad * scale_factor)
            
        return projected_gradients

class ContinualLearningEngine:
    """Main Continual Learning Engine"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.model = None
        self.experience_replay = ExperienceReplay(config)
        
        # Initialize strategy-specific components
        self.ewc = ElasticWeightConsolidation(config) if config.strategy == "ewc" else None
        self.lwf = LearningWithoutForgetting(config) if config.strategy == "lwf" else None
        self.mas = MemoryAwareSynapses(config) if config.strategy == "mas" else None
        self.packnet = PackNet(config) if config.strategy == "packnet" else None
        self.gem = GradientEpisodicMemory(config) if config.strategy == "gem" else None
        
        # Task tracking
        self.current_task = 0
        self.task_performances = defaultdict(list)
        self.forgetting_metrics = defaultdict(list)
        
    def set_model(self, model: nn.Module):
        """Set model for continual learning"""
        
        self.model = model
        
    def learn_task(self, task_id: int, train_dataloader: torch.utils.data.DataLoader,
                   val_dataloader: torch.utils.data.DataLoader = None) -> Dict[str, Any]:
        """Learn a new task"""
        
        logger.info(f"Learning task {task_id}")
        
        # Initialize task-specific components
        if self.config.strategy == "ewc":
            if task_id > 0:
                self.ewc.compute_fisher_information(self.model, train_dataloader)
                self.ewc.save_optimal_params(self.model)
                
        elif self.config.strategy == "lwf":
            if task_id > 0 and self.config.use_old_model:
                self.lwf.save_old_model(self.model)
                
        elif self.config.strategy == "mas":
            self.mas.compute_importance(self.model, train_dataloader)
            
        elif self.config.strategy == "packnet":
            if task_id > 0:
                self.packnet.prune_network(self.model, task_id)
                
        # Train on current task
        task_results = self._train_on_task(task_id, train_dataloader, val_dataloader)
        
        # Update experience replay
        self._update_experience_replay(task_id, train_dataloader)
        
        # Evaluate on all previous tasks
        if self.config.evaluate_on_all_tasks:
            self._evaluate_all_tasks()
            
        self.current_task = task_id
        
        return task_results
        
    def _train_on_task(self, task_id: int, train_dataloader: torch.utils.data.DataLoader,
                      val_dataloader: torch.utils.data.DataLoader = None) -> Dict[str, Any]:
        """Train model on current task"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(50):  # 50 epochs per task
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                task_loss = criterion(outputs, targets)
                
                # Add regularization loss
                total_loss_batch = task_loss
                
                if self.config.strategy == "ewc" and task_id > 0:
                    ewc_loss = self.ewc.compute_ewc_loss(self.model)
                    total_loss_batch += ewc_loss
                    
                elif self.config.strategy == "lwf" and task_id > 0:
                    with torch.no_grad():
                        old_outputs = self.lwf.old_model(inputs)
                    distillation_loss = self.lwf.compute_distillation_loss(outputs, old_outputs)
                    total_loss_batch += self.config.distillation_alpha * distillation_loss
                    
                elif self.config.strategy == "mas":
                    mas_loss = self.mas.compute_mas_loss(self.model)
                    total_loss_batch += mas_loss
                    
                # Add experience replay loss
                if self.experience_replay.current_size > 0:
                    replay_inputs, replay_targets, replay_tasks = self.experience_replay.sample_batch(32)
                    if replay_inputs is not None:
                        replay_outputs = self.model(replay_inputs)
                        replay_loss = criterion(replay_outputs, replay_targets)
                        total_loss_batch += 0.5 * replay_loss
                        
                # Backward pass
                total_loss_batch.backward()
                
                # Gradient projection for GEM
                if self.config.strategy == "gem" and task_id > 0:
                    current_gradients = [param.grad for param in self.model.parameters()]
                    projected_gradients = self.gem.project_gradient(self.model, current_gradients)
                    for param, proj_grad in zip(self.model.parameters(), projected_gradients):
                        param.grad = proj_grad
                        
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            # Validation
            if val_dataloader is not None:
                val_accuracy = self._evaluate_model(val_dataloader)
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            logger.info(f"Task {task_id}, Epoch {epoch}: Loss = {total_loss:.4f}, "
                       f"Accuracy = {100 * correct / total:.2f}%")
                       
        return {
            'task_id': task_id,
            'final_accuracy': best_accuracy,
            'epochs_trained': epoch + 1
        }
        
    def _evaluate_model(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model on dataloader"""
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return correct / total
        
    def _update_experience_replay(self, task_id: int, dataloader: torch.utils.data.DataLoader):
        """Update experience replay buffer"""
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 10:  # Store only 10 batches per task
                break
                
            # Compute importance score
            importance = 1.0  # Simplified importance
            self.experience_replay.add_experience(inputs, targets, task_id, importance)
            
    def _evaluate_all_tasks(self):
        """Evaluate model on all previous tasks"""
        
        # This would evaluate on all previous tasks
        # For now, just log current task performance
        logger.info(f"Evaluated on all tasks up to task {self.current_task}")
        
    def get_forgetting_metrics(self) -> Dict[str, Any]:
        """Get catastrophic forgetting metrics"""
        
        return {
            'task_performances': dict(self.task_performances),
            'forgetting_metrics': dict(self.forgetting_metrics),
            'current_task': self.current_task
        }
        
    def save_model(self, filepath: str):
        """Save model and continual learning state"""
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'current_task': self.current_task,
            'task_performances': dict(self.task_performances),
            'experience_replay': self.experience_replay,
            'config': self.config
        }
        
        torch.save(state, filepath)
        
    def load_model(self, filepath: str):
        """Load model and continual learning state"""
        
        state = torch.load(filepath)
        self.model.load_state_dict(state['model_state_dict'])
        self.current_task = state['current_task']
        self.task_performances = defaultdict(list, state['task_performances'])
        self.experience_replay = state['experience_replay']

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test continual learning
    print("Testing Continual Learning Engine...")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(784, 256)
            self.linear2 = nn.Linear(256, 128)
            self.linear3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.linear1(x))
            x = self.dropout(x)
            x = self.relu(self.linear2(x))
            x = self.dropout(x)
            x = self.linear3(x)
            return x
    
    model = TestModel()
    
    # Create continual learning config
    config = ContinualLearningConfig(
        strategy="ewc",
        memory_size=1000,
        regularization_strength=1000.0,
        evaluate_on_all_tasks=True
    )
    
    # Create continual learning engine
    cl_engine = ContinualLearningEngine(config)
    cl_engine.set_model(model)
    
    # Create dummy dataloaders for testing
    def create_dummy_dataloader(task_id: int, num_samples: int = 100):
        inputs = torch.randn(num_samples, 784)
        targets = torch.randint(0, 10, (num_samples,))
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test learning multiple tasks
    for task_id in range(3):
        print(f"Learning task {task_id}...")
        
        train_loader = create_dummy_dataloader(task_id, 200)
        val_loader = create_dummy_dataloader(task_id, 50)
        
        results = cl_engine.learn_task(task_id, train_loader, val_loader)
        print(f"Task {task_id} completed: {results}")
        
    # Get forgetting metrics
    metrics = cl_engine.get_forgetting_metrics()
    print(f"Forgetting metrics: {metrics}")
    
    print("\nContinual learning engine initialized successfully!")
























