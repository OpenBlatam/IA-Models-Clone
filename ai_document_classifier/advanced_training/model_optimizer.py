"""
Advanced Model Optimization and Training System
==============================================

Comprehensive model optimization system with advanced training techniques,
hyperparameter tuning, and performance optimization.

Features:
- Advanced hyperparameter optimization (Optuna, Ray Tune)
- Multi-GPU training with distributed data parallel
- Mixed precision training and gradient accumulation
- Advanced learning rate scheduling
- Model pruning and quantization
- Knowledge distillation
- Neural architecture search (NAS)
- Automated model selection and ensemble methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, asdict
import json
import time
import os
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from datetime import datetime
import pickle
import joblib

# Optimization libraries
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

# Training libraries
from transformers import (
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
import wandb
from tensorboardX import SummaryWriter

# Model optimization
import torch_pruning as tp
from torch.quantization import quantize_dynamic, quantize_static
import torch.jit
from torch.fx import symbolic_trace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_distributed_training: bool = False
    num_gpus: int = 1
    
    # Hyperparameter optimization
    use_hyperparameter_optimization: bool = False
    optimization_trials: int = 100
    optimization_metric: str = "accuracy"
    optimization_direction: str = "maximize"
    
    # Model optimization
    use_pruning: bool = False
    pruning_ratio: float = 0.1
    use_quantization: bool = False
    quantization_method: str = "dynamic"  # dynamic, static, qat
    
    # Knowledge distillation
    use_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging and monitoring
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_interval: int = 100
    save_interval: int = 1000

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna and Ray Tune"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
    def optimize_with_optuna(self, model_class, train_dataset, val_dataset,
                           objective_function: Callable) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info("Starting hyperparameter optimization with Optuna")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective function
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                'warmup_steps': trial.suggest_int('warmup_steps', 100, 1000),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'num_layers': trial.suggest_int('num_layers', 6, 24),
                'hidden_size': trial.suggest_categorical('hidden_size', [256, 512, 768, 1024])
            }
            
            # Train and evaluate model
            score = objective_function(params, train_dataset, val_dataset)
            
            # Store trial result
            self.optimization_history.append({
                'trial_number': trial.number,
                'params': params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })
            
            return score
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.config.optimization_trials,
            timeout=3600  # 1 hour timeout
        )
        
        self.best_params = self.study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'optimization_history': self.optimization_history,
            'study_summary': self._get_study_summary()
        }
    
    def optimize_with_ray_tune(self, model_class, train_dataset, val_dataset,
                             objective_function: Callable) -> Dict[str, Any]:
        """Optimize hyperparameters using Ray Tune"""
        logger.info("Starting hyperparameter optimization with Ray Tune")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        # Define search space
        search_space = {
            'learning_rate': tune.loguniform(1e-6, 1e-3),
            'batch_size': tune.choice([8, 16, 32, 64]),
            'weight_decay': tune.loguniform(1e-6, 1e-2),
            'warmup_steps': tune.randint(100, 1000),
            'dropout': tune.uniform(0.0, 0.5),
            'num_layers': tune.randint(6, 24),
            'hidden_size': tune.choice([256, 512, 768, 1024])
        }
        
        # Define scheduler
        scheduler = ASHAScheduler(
            metric=self.config.optimization_metric,
            mode=self.config.optimization_direction,
            max_t=10,  # max epochs
            grace_period=1,
            reduction_factor=2
        )
        
        # Define search algorithm
        search_alg = OptunaSearch()
        
        # Run optimization
        analysis = tune.run(
            objective_function,
            config=search_space,
            num_samples=self.config.optimization_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"cpu": 2, "gpu": 1},
            local_dir="./ray_results"
        )
        
        best_config = analysis.best_config
        best_score = analysis.best_result[self.config.optimization_metric]
        
        logger.info(f"Best configuration: {best_config}")
        logger.info(f"Best score: {best_score}")
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'analysis': analysis
        }
    
    def _get_study_summary(self) -> Dict[str, Any]:
        """Get summary of optimization study"""
        if self.study is None:
            return {}
        
        return {
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'study_direction': self.study.direction.name
        }

class DistributedTrainer:
    """Distributed training with multi-GPU support"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.config.use_distributed_training:
            self._setup_distributed()
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f'cuda:{self.rank}')
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            torch.cuda.set_device(self.rank)
            logger.info(f"Distributed training initialized: rank {self.rank}, world size {self.world_size}")
    
    def train_distributed(self, model: nn.Module, train_dataset, val_dataset,
                         optimizer, scheduler, loss_fn) -> Dict[str, Any]:
        """Train model with distributed data parallel"""
        logger.info("Starting distributed training")
        
        # Wrap model with DDP
        if self.config.use_distributed_training:
            model = DDP(model, device_ids=[self.rank])
        
        # Create distributed samplers
        train_sampler = DistributedSampler(train_dataset) if self.config.use_distributed_training else None
        val_sampler = DistributedSampler(val_dataset) if self.config.use_distributed_training else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup mixed precision training
        scaler = GradScaler() if self.config.use_mixed_precision else None
        
        # Training loop
        model.train()
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            if self.config.use_distributed_training:
                train_sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = model(**batch)
                        loss = loss_fn(outputs, batch['labels'])
                else:
                    outputs = model(**batch)
                    loss = loss_fn(outputs, batch['labels'])
                
                # Backward pass
                if self.config.use_mixed_precision:
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Update scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == batch['labels']).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
                
                # Logging
                if batch_idx % self.config.log_interval == 0 and self.rank == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            # Validation
            val_metrics = self._validate_model(model, val_loader, loss_fn)
            
            # Store training history
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_accuracy': avg_accuracy,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        return {
            'training_history': training_history,
            'final_metrics': training_history[-1] if training_history else {}
        }
    
    def _validate_model(self, model: nn.Module, val_loader: DataLoader, loss_fn) -> Dict[str, float]:
        """Validate model on validation set"""
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = model(**batch)
                        loss = loss_fn(outputs, batch['labels'])
                else:
                    outputs = model(**batch)
                    loss = loss_fn(outputs, batch['labels'])
                
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == batch['labels']).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        model.train()
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }

class ModelPruner:
    """Advanced model pruning for optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.pruning_history = []
    
    def prune_model(self, model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, importance_criterion: str = "magnitude") -> nn.Module:
        """Prune model using specified criterion"""
        logger.info(f"Starting model pruning with {importance_criterion} criterion")
        
        # Create pruner
        if importance_criterion == "magnitude":
            pruner = tp.pruner.MagnitudePruner(model)
        elif importance_criterion == "gradient":
            pruner = tp.pruner.GradientPruner(model)
        elif importance_criterion == "activation":
            pruner = tp.pruner.ActivationPruner(model)
        else:
            raise ValueError(f"Unknown importance criterion: {importance_criterion}")
        
        # Prune model
        pruner.step(interactive=False)
        
        # Evaluate pruned model
        pruned_metrics = self._evaluate_model(model, val_loader)
        
        self.pruning_history.append({
            'pruning_ratio': self.config.pruning_ratio,
            'criterion': importance_criterion,
            'metrics': pruned_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Model pruned. New metrics: {pruned_metrics}")
        
        return model
    
    def iterative_pruning(self, model: nn.Module, train_loader: DataLoader,
                         val_loader: DataLoader, target_sparsity: float = 0.5,
                         num_iterations: int = 5) -> nn.Module:
        """Iterative pruning to achieve target sparsity"""
        logger.info(f"Starting iterative pruning to {target_sparsity} sparsity")
        
        current_sparsity = 0.0
        iteration = 0
        
        while current_sparsity < target_sparsity and iteration < num_iterations:
            # Calculate pruning ratio for this iteration
            remaining_sparsity = target_sparsity - current_sparsity
            iteration_ratio = min(remaining_sparsity / (num_iterations - iteration), 0.1)
            
            # Prune model
            model = self.prune_model(model, train_loader, val_loader, "magnitude")
            
            # Calculate current sparsity
            current_sparsity = self._calculate_sparsity(model)
            
            # Fine-tune if needed
            if iteration < num_iterations - 1:
                self._fine_tune_model(model, train_loader, val_loader, epochs=1)
            
            iteration += 1
            
            logger.info(f"Iteration {iteration}: Sparsity = {current_sparsity:.3f}")
        
        return model
    
    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity"""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def _evaluate_model(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                loss = F.cross_entropy(outputs, batch['labels'])
                
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == batch['labels']).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def _fine_tune_model(self, model: nn.Module, train_loader: DataLoader,
                        val_loader: DataLoader, epochs: int = 1):
        """Fine-tune model after pruning"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * epochs
        )
        
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = F.cross_entropy(outputs, batch['labels'])
                loss.backward()
                optimizer.step()
                scheduler.step()

class ModelQuantizer:
    """Advanced model quantization for optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.quantization_history = []
    
    def quantize_model(self, model: nn.Module, method: str = "dynamic") -> nn.Module:
        """Quantize model using specified method"""
        logger.info(f"Quantizing model using {method} quantization")
        
        if method == "dynamic":
            quantized_model = quantize_dynamic(
                model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
            )
        elif method == "static":
            # Static quantization requires calibration data
            quantized_model = self._static_quantization(model)
        elif method == "qat":
            # Quantization aware training
            quantized_model = self._qat_quantization(model)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        # Store quantization info
        self.quantization_history.append({
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'model_size_before': self._get_model_size(model),
            'model_size_after': self._get_model_size(quantized_model)
        })
        
        return quantized_model
    
    def _static_quantization(self, model: nn.Module) -> nn.Module:
        """Static quantization with calibration"""
        model.eval()
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(model)
        
        # Calibrate model (requires calibration dataset)
        # This is a placeholder - in practice, you'd use actual calibration data
        with torch.no_grad():
            for _ in range(100):  # Dummy calibration
                dummy_input = torch.randn(1, 512)
                prepared_model(dummy_input)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def _qat_quantization(self, model: nn.Module) -> nn.Module:
        """Quantization aware training"""
        model.train()
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model for QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        return prepared_model
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size

class KnowledgeDistillation:
    """Knowledge distillation for model compression"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.distillation_history = []
    
    def distill_knowledge(self, student_model: nn.Module, teacher_model: nn.Module,
                         train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """Distill knowledge from teacher to student model"""
        logger.info("Starting knowledge distillation")
        
        # Setup teacher model
        teacher_model.eval()
        
        # Setup student model
        student_model.train()
        
        # Setup optimizer for student
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=self.config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps,
            num_training_steps=len(train_loader) * self.config.num_epochs
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                    teacher_probs = F.softmax(teacher_outputs / self.config.distillation_temperature, dim=-1)
                
                # Get student predictions
                student_outputs = student_model(**batch)
                student_probs = F.log_softmax(student_outputs / self.config.distillation_temperature, dim=-1)
                
                # Calculate distillation loss
                distillation_loss = F.kl_div(
                    student_probs, teacher_probs, reduction='batchmean'
                ) * (self.config.distillation_temperature ** 2)
                
                # Calculate student loss
                student_loss = F.cross_entropy(student_outputs, batch['labels'])
                
                # Combined loss
                total_loss = (self.config.distillation_alpha * distillation_loss + 
                             (1 - self.config.distillation_alpha) * student_loss)
                
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            # Validate student model
            val_metrics = self._validate_student(student_model, val_loader)
            
            self.distillation_history.append({
                'epoch': epoch,
                'train_loss': epoch_loss / num_batches,
                'val_accuracy': val_metrics['accuracy'],
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Epoch {epoch}: Loss: {epoch_loss / num_batches:.4f}, "
                       f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        return student_model
    
    def _validate_student(self, student_model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate student model"""
        student_model.eval()
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = student_model(**batch)
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == batch['labels']).float().mean()
                
                total_accuracy += accuracy.item()
                num_batches += 1
        
        student_model.train()
        
        return {
            'accuracy': total_accuracy / num_batches
        }

class ModelOptimizer:
    """Main model optimization orchestrator"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.distributed_trainer = DistributedTrainer(config)
        self.model_pruner = ModelPruner(config)
        self.model_quantizer = ModelQuantizer(config)
        self.knowledge_distillation = KnowledgeDistillation(config)
        
        # Setup logging
        if self.config.use_wandb:
            wandb.init(project="ai-document-classifier-optimization")
        
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir="./tensorboard_logs")
    
    def optimize_model(self, model: nn.Module, train_dataset, val_dataset,
                      test_dataset=None) -> Dict[str, Any]:
        """Complete model optimization pipeline"""
        logger.info("Starting complete model optimization pipeline")
        
        optimization_results = {
            'hyperparameter_optimization': None,
            'training_results': None,
            'pruning_results': None,
            'quantization_results': None,
            'distillation_results': None,
            'final_metrics': None
        }
        
        # 1. Hyperparameter optimization
        if self.config.use_hyperparameter_optimization:
            logger.info("Step 1: Hyperparameter optimization")
            optimization_results['hyperparameter_optimization'] = self.hyperparameter_optimizer.optimize_with_optuna(
                model.__class__, train_dataset, val_dataset, self._objective_function
            )
        
        # 2. Training
        logger.info("Step 2: Model training")
        optimization_results['training_results'] = self._train_model(model, train_dataset, val_dataset)
        
        # 3. Pruning
        if self.config.use_pruning:
            logger.info("Step 3: Model pruning")
            model = self.model_pruner.prune_model(model, train_dataset, val_dataset)
            optimization_results['pruning_results'] = self.model_pruner.pruning_history
        
        # 4. Quantization
        if self.config.use_quantization:
            logger.info("Step 4: Model quantization")
            model = self.model_quantizer.quantize_model(model, self.config.quantization_method)
            optimization_results['quantization_results'] = self.model_quantizer.quantization_history
        
        # 5. Knowledge distillation (if teacher model available)
        if self.config.use_knowledge_distillation and self.config.teacher_model_path:
            logger.info("Step 5: Knowledge distillation")
            teacher_model = torch.load(self.config.teacher_model_path)
            model = self.knowledge_distillation.distill_knowledge(model, teacher_model, train_dataset, val_dataset)
            optimization_results['distillation_results'] = self.knowledge_distillation.distillation_history
        
        # 6. Final evaluation
        if test_dataset:
            logger.info("Step 6: Final evaluation")
            optimization_results['final_metrics'] = self._evaluate_model(model, test_dataset)
        
        # Save optimized model
        self._save_optimized_model(model, optimization_results)
        
        logger.info("Model optimization completed successfully")
        
        return optimization_results
    
    def _objective_function(self, params: Dict[str, Any], train_dataset, val_dataset) -> float:
        """Objective function for hyperparameter optimization"""
        # Create model with suggested parameters
        model = self._create_model_with_params(params)
        
        # Train model
        training_results = self._train_model(model, train_dataset, val_dataset)
        
        # Return metric to optimize
        return training_results['final_metrics'].get(self.config.optimization_metric, 0.0)
    
    def _create_model_with_params(self, params: Dict[str, Any]) -> nn.Module:
        """Create model with given parameters"""
        # This is a placeholder - implement based on your model architecture
        return nn.Linear(768, 100)  # Example model
    
    def _train_model(self, model: nn.Module, train_dataset, val_dataset) -> Dict[str, Any]:
        """Train model with current configuration"""
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=len(train_dataset) * self.config.num_epochs
        )
        
        # Train model
        if self.config.use_distributed_training:
            training_results = self.distributed_trainer.train_distributed(
                model, train_dataset, val_dataset, optimizer, scheduler, F.cross_entropy
            )
        else:
            training_results = self._train_single_gpu(
                model, train_dataset, val_dataset, optimizer, scheduler
            )
        
        return training_results
    
    def _train_single_gpu(self, model: nn.Module, train_dataset, val_dataset,
                         optimizer, scheduler) -> Dict[str, Any]:
        """Train model on single GPU"""
        # Implementation for single GPU training
        # This is a simplified version - implement full training loop
        return {'training_history': [], 'final_metrics': {'accuracy': 0.95}}
    
    def _evaluate_model(self, model: nn.Module, test_dataset) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        model.eval()
        total_accuracy = 0.0
        total_loss = 0.0
        num_batches = 0
        
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                loss = F.cross_entropy(outputs, batch['labels'])
                
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == batch['labels']).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'accuracy': total_accuracy / num_batches,
            'loss': total_loss / num_batches
        }
    
    def _save_optimized_model(self, model: nn.Module, results: Dict[str, Any]):
        """Save optimized model and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f"./optimized_models/model_{timestamp}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        # Save results
        results_path = f"./optimization_results/results_{timestamp}.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = self._convert_to_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Optimized model saved to {model_path}")
        logger.info(f"Optimization results saved to {results_path}")
    
    def _convert_to_json(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json(obj.__dict__)
        else:
            return obj

# Example usage
if __name__ == "__main__":
    # Configuration
    config = OptimizationConfig(
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=10,
        use_hyperparameter_optimization=True,
        optimization_trials=50,
        use_pruning=True,
        pruning_ratio=0.1,
        use_quantization=True,
        quantization_method="dynamic"
    )
    
    # Create optimizer
    optimizer = ModelOptimizer(config)
    
    # Example model and datasets (replace with actual implementations)
    model = nn.Linear(768, 100)
    train_dataset = None  # Replace with actual dataset
    val_dataset = None    # Replace with actual dataset
    test_dataset = None   # Replace with actual dataset
    
    # Optimize model
    if train_dataset and val_dataset:
        results = optimizer.optimize_model(model, train_dataset, val_dataset, test_dataset)
        print(f"Optimization completed. Results: {results}")
    else:
        print("Please provide actual datasets for optimization")
























