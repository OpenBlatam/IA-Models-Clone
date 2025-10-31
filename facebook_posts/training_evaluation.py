from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import json
import pickle
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from data_loader_utils import make_loader
from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
import random
from diffusion_models import DiffusionModel, DiffusionConfig, DiffusionTrainer
from deep_learning_framework import DeepLearningFramework, FrameworkConfig, TaskType
from advanced_transformers import AdvancedTransformerModel
from llm_training import AdvancedLLMTrainer
from training_optimization import AdvancedTrainingManager
from loss_functions import LossFunctionFactory
from optimization_algorithms import OptimizationManager
from normalization_techniques import AdvancedLayerNorm
from weight_initialization import AdvancedWeightInitializer
from gradient_analysis import GradientMonitor
from framework_utils import MetricsTracker, ModelAnalyzer, PerformanceMonitor
from sklearn.model_selection import KFold
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Comprehensive Training and Evaluation System
Advanced training and evaluation framework for deep learning models.
"""


# Import our components


class TrainingMode(Enum):
    """Training modes."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    FEDERATED = "federated"


class EvaluationMode(Enum):
    """Evaluation modes."""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    CROSS_VALIDATION = "cross_validation"
    ENSEMBLE = "ensemble"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    lr_scheduler_patience: int = 5
    
    # Advanced features
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    checkpoint_save_frequency: int = 5
    
    # Monitoring
    log_frequency: int = 100
    eval_frequency: int = 500
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    
    # Evaluation
    eval_batch_size: int = 64
    num_eval_steps: int = None  # None for full evaluation
    save_best_model: bool = True
    save_last_model: bool = True
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    random_seed: Optional[int] = 42
    
    # Custom callbacks
    custom_callbacks: List[Callable] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Evaluation parameters
    eval_batch_size: int = 64
    num_eval_steps: Optional[int] = None
    compute_metrics: bool = True
    save_predictions: bool = False
    prediction_format: str = "numpy"  # "numpy", "json", "csv"
    
    # Metrics
    classification_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "confusion_matrix"
    ])
    regression_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "rmse", "r2_score"
    ])
    generation_metrics: List[str] = field(default_factory=lambda: [
        "bleu", "rouge", "perplexity"
    ])
    
    # Visualization
    plot_metrics: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    
    # Advanced
    ensemble_evaluation: bool = False
    cross_validation_folds: int = 5
    statistical_significance: bool = False


class TrainingManager:
    """Comprehensive training manager."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.performance_monitor = PerformanceMonitor()
        
        # Logging
        self.writer = None
        if self.config.tensorboard_logging:
            self.writer = SummaryWriter(log_dir=f"runs/training_{int(time.time())}")
        
        if self.config.wandb_logging:
            wandb.init(project="deep-learning-training", config=config.__dict__)
        
        # Optional reproducibility
        try:
            if self.config.random_seed is not None:
                seed = int(self.config.random_seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    
    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for training."""
        # Enable TF32 and matmul precision when available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
        except Exception:
            pass

        model = model.to(self.device)
        
        # Initialize weights
        initializer = AdvancedWeightInitializer()
        initializer.initialize_model(model)
        
        # Setup gradient monitoring
        self.gradient_monitor = GradientMonitor(model)
        
        return model
    
    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: torch.optim.Optimizer, 
                       total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        if self.config.scheduler_type.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif self.config.scheduler_type.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler_type.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.config.lr_scheduler_patience, factor=0.5
            )
        elif self.config.scheduler_type.lower() == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.config.learning_rate, total_steps=total_steps
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        
        return scheduler
    
    def setup_loss_function(self, task_type: TaskType) -> nn.Module:
        """Setup loss function based on task type."""
        loss_factory = LossFunctionFactory()
        
        if task_type == TaskType.CLASSIFICATION:
            return loss_factory.create_loss("cross_entropy", num_classes=10)
        elif task_type == TaskType.REGRESSION:
            return loss_factory.create_loss("mse")
        elif task_type == TaskType.GENERATION:
            return loss_factory.create_loss("cross_entropy")
        else:
            return nn.MSELoss()
    
    def train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   loss_function: nn.Module, train_dataloader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                inputs, targets = batch, None
            
            inputs = inputs.to(self.device, non_blocking=True)
            if targets is not None:
                targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.config.mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if targets is not None:
                        loss = loss_function(outputs, targets)
                    else:
                        loss = loss_function(outputs)
            else:
                outputs = model(inputs)
                if targets is not None:
                    loss = loss_function(outputs, targets)
                else:
                    loss = loss_function(outputs)
            
            # Backward pass
            if self.config.mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    if self.config.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            # Update scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Logging
            if self.global_step % self.config.log_frequency == 0:
                self._log_training_step(loss.item(), optimizer.param_groups[0]['lr'])
            
            # Gradient monitoring
            if hasattr(self, 'gradient_monitor'):
                self.gradient_monitor.update()
        
        # Update scheduler (for non-OneCycleLR schedulers)
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        return {
            'epoch_loss': epoch_loss / num_batches,
            'num_batches': num_batches
        }
    
    def evaluate(self, model: nn.Module, loss_function: nn.Module,
                eval_dataloader: DataLoader, evaluator: 'Evaluator') -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        eval_loss = 0.0
        predictions = []
        targets = []
        
        with torch.inference_mode():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                if isinstance(batch, (list, tuple)):
                    inputs, batch_targets = batch
                else:
                    inputs, batch_targets = batch, None
                
                inputs = inputs.to(self.device, non_blocking=True)
                if batch_targets is not None:
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                
                if batch_targets is not None:
                    loss = loss_function(outputs, batch_targets)
                    eval_loss += loss.item()
                    
                    predictions.append(outputs.cpu())
                    targets.append(batch_targets.cpu())
                else:
                    predictions.append(outputs.cpu())
        
        # Compute metrics
        if predictions and targets:
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            metrics = evaluator.compute_metrics(predictions, targets)
        else:
            metrics = {}
        
        metrics['eval_loss'] = eval_loss / len(eval_dataloader)
        
        return metrics
    
    def train(self, model: nn.Module, train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              task_type: TaskType = TaskType.CLASSIFICATION,
              evaluator: Optional['Evaluator'] = None) -> Dict[str, Any]:
        """Complete training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Setup components
        model = self.setup_model(model)
        optimizer = self.setup_optimizer(model)
        
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = self.setup_scheduler(optimizer, total_steps)
        loss_function = self.setup_loss_function(task_type)
        
        # Training history
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Training
            train_results = self.train_epoch(
                model, optimizer, scheduler, loss_function, train_dataloader, epoch
            )
            
            # Validation
            val_results = {}
            if val_dataloader is not None and evaluator is not None:
                val_results = self.evaluate(model, loss_function, val_dataloader, evaluator)
            
            # Update history
            training_history['train_losses'].append(train_results['epoch_loss'])
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if val_results:
                training_history['val_losses'].append(val_results.get('eval_loss', 0))
                training_history['val_metrics'].append(val_results)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_results['epoch_loss']:.6f}"
            )
            
            if val_results:
                self.logger.info(f"Val Loss: {val_results.get('eval_loss', 0):.6f}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_results['epoch_loss'], epoch)
                if val_results:
                    self.writer.add_scalar('Loss/Val', val_results.get('eval_loss', 0), epoch)
                self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # WandB logging
            if self.config.wandb_logging:
                log_dict = {
                    'train_loss': train_results['epoch_loss'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                }
                if val_results:
                    log_dict.update(val_results)
                wandb.log(log_dict)
            
            # Early stopping
            if val_results and self.config.early_stopping_patience > 0:
                current_metric = val_results.get('eval_loss', float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    if self.config.save_best_model:
                        self.save_checkpoint(model, optimizer, epoch, "best")
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_save_frequency == 0:
                self.save_checkpoint(model, optimizer, epoch, f"epoch_{epoch + 1}")
            
            self.current_epoch = epoch + 1
        
        # Save final model
        if self.config.save_last_model:
            self.save_checkpoint(model, optimizer, self.current_epoch, "last")
        
        # Close logging
        if self.writer:
            self.writer.close()
        
        if self.config.wandb_logging:
            wandb.finish()
        
        return {
            'model': model,
            'training_history': training_history,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric
        }
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, name: str):
        """Save model checkpoint."""
        cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': cpu_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _log_training_step(self, loss: float, learning_rate: float):
        """Log training step."""
        if self.writer:
            self.writer.add_scalar('Loss/Step', loss, self.global_step)
            self.writer.add_scalar('Learning_Rate/Step', learning_rate, self.global_step)
        
        if self.config.wandb_logging:
            wandb.log({
                'loss_step': loss,
                'lr_step': learning_rate,
                'step': self.global_step
            })


class Evaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics_tracker = MetricsTracker()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def compute_metrics(self, predictions: torch.Tensor, 
                       targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Determine task type based on predictions shape
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            # Classification task
            return self._compute_classification_metrics(predictions_np, targets_np)
        elif len(predictions.shape) == 1 or predictions.shape[1] == 1:
            # Regression task
            return self._compute_regression_metrics(predictions_np, targets_np)
        else:
            # Generation task
            return self._compute_generation_metrics(predictions_np, targets_np)
    
    def _compute_classification_metrics(self, predictions: np.ndarray, 
                                      targets: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        # Convert to class predictions
        if len(predictions.shape) == 2:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        metrics = {}
        
        # Accuracy
        if 'accuracy' in self.config.classification_metrics:
            metrics['accuracy'] = accuracy_score(targets, pred_classes)
        
        # Precision, Recall, F1
        if any(metric in self.config.classification_metrics for metric in ['precision', 'recall', 'f1']):
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, pred_classes, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        # Confusion Matrix
        if 'confusion_matrix' in self.config.classification_metrics:
            cm = confusion_matrix(targets, pred_classes)
            metrics['confusion_matrix'] = cm
            
            # Plot confusion matrix
            if self.config.plot_metrics:
                self._plot_confusion_matrix(cm)
        
        return metrics
    
    def _compute_regression_metrics(self, predictions: np.ndarray, 
                                   targets: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        # MSE
        if 'mse' in self.config.regression_metrics:
            metrics['mse'] = np.mean((predictions - targets) ** 2)
        
        # MAE
        if 'mae' in self.config.regression_metrics:
            metrics['mae'] = np.mean(np.abs(predictions - targets))
        
        # RMSE
        if 'rmse' in self.config.regression_metrics:
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # R² Score
        if 'r2_score' in self.config.regression_metrics:
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            metrics['r2_score'] = 1 - (ss_res / ss_tot)
        
        return metrics
    
    def _compute_generation_metrics(self, predictions: np.ndarray, 
                                   targets: np.ndarray) -> Dict[str, float]:
        """Compute generation metrics."""
        metrics = {}
        
        # Perplexity
        if 'perplexity' in self.config.generation_metrics:
            # Simplified perplexity calculation
            log_probs = np.log(predictions + 1e-8)
            perplexity = np.exp(-np.mean(log_probs))
            metrics['perplexity'] = perplexity
        
        # BLEU and ROUGE would require additional libraries
        # For now, we'll use simplified metrics
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.config.save_plots:
            plt.savefig(f'confusion_matrix.{self.config.plot_format}')
        plt.show()
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader,
                      device: torch.device, loss_function: nn.Module) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        model.eval()
        eval_loss = 0.0
        predictions = []
        targets = []
        
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if isinstance(batch, (list, tuple)):
                    inputs, batch_targets = batch
                else:
                    inputs, batch_targets = batch, None
                
                inputs = inputs.to(device)
                if batch_targets is not None:
                    batch_targets = batch_targets.to(device)
                
                outputs = model(inputs)
                
                if batch_targets is not None:
                    loss = loss_function(outputs, batch_targets)
                    eval_loss += loss.item()
                    
                    predictions.append(outputs.cpu())
                    targets.append(batch_targets.cpu())
                else:
                    predictions.append(outputs.cpu())
        
        # Compute metrics
        if predictions and targets:
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            metrics = self.compute_metrics(predictions, targets)
        else:
            metrics = {}
        
        metrics['eval_loss'] = eval_loss / len(dataloader)
        
        # Save predictions if requested
        if self.config.save_predictions:
            self._save_predictions(predictions, targets)
        
        return metrics
    
    def _save_predictions(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Save predictions to file."""
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        if self.config.prediction_format == "numpy":
            np.save("predictions.npy", predictions_np)
            np.save("targets.npy", targets_np)
        elif self.config.prediction_format == "json":
            results = {
                'predictions': predictions_np.tolist(),
                'targets': targets_np.tolist()
            }
            with open("predictions.json", 'w') as f:
                json.dump(results, f)
        elif self.config.prediction_format == "csv":
            df = pd.DataFrame({
                'predictions': predictions_np.flatten(),
                'targets': targets_np.flatten()
            })
            df.to_csv("predictions.csv", index=False)
    
    def cross_validate(self, model_class: type, train_dataset: Dataset,
                       config: TrainingConfig, n_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Create fold-specific datasets
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = make_loader(train_dataset, batch_size=config.batch_size, sampler=train_subsampler, generator_seed=config.random_seed)
            val_loader = make_loader(train_dataset, batch_size=config.eval_batch_size, sampler=val_subsampler, generator_seed=config.random_seed)
            
            # Train model
            model = model_class()
            trainer = TrainingManager(config)
            evaluator = Evaluator(EvaluationConfig())
            
            training_results = trainer.train(
                model, train_loader, val_loader, evaluator=evaluator
            )
            
            # Get best validation metrics
            best_metrics = training_results['training_history']['val_metrics'][-1]
            fold_metrics.append(best_metrics)
        
        # Aggregate results
        aggregated_metrics = {}
        for metric_name in fold_metrics[0].keys():
            values = [fold[metric_name] for fold in fold_metrics]
            aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregated_metrics[f"{metric_name}_std"] = np.std(values)
        
        return aggregated_metrics


def demonstrate_training_evaluation():
    """Demonstrate training and evaluation capabilities."""
    print("Training and Evaluation Demonstration")
    print("=" * 50)
    
    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, num_samples=1000, num_classes=10) -> Any:
            self.data = torch.randn(num_samples, 3, 32, 32)
            self.targets = torch.randint(0, num_classes, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=10) -> Any:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x) -> Any:
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # Setup
    dataset = DummyDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = make_loader(train_dataset, batch_size=32, shuffle=True, generator_seed=training_config.random_seed)
    val_loader = make_loader(val_dataset, batch_size=64, shuffle=False, generator_seed=training_config.random_seed)
    
    # Training configuration
    training_config = TrainingConfig(
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        mixed_precision=True,
        tensorboard_logging=True
    )
    
    # Evaluation configuration
    eval_config = EvaluationConfig(
        compute_metrics=True,
        save_predictions=True,
        plot_metrics=True
    )
    
    # Create model and components
    model = SimpleModel()
    trainer = TrainingManager(training_config)
    evaluator = Evaluator(eval_config)
    
    # Train model
    print("Starting training...")
    training_results = trainer.train(
        model, train_loader, val_loader, 
        task_type=TaskType.CLASSIFICATION,
        evaluator=evaluator
    )
    
    # Evaluate final model
    print("\nEvaluating final model...")
    final_metrics = evaluator.evaluate_model(
        model, val_loader, trainer.device, nn.CrossEntropyLoss()
    )
    
    print(f"Final evaluation metrics: {final_metrics}")
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = evaluator.cross_validate(
        SimpleModel, dataset, training_config, n_folds=3
    )
    
    print(f"Cross-validation results: {cv_results}")
    
    return {
        'training_results': training_results,
        'final_metrics': final_metrics,
        'cv_results': cv_results
    }


if __name__ == "__main__":
    # Demonstrate training and evaluation
    results = demonstrate_training_evaluation()
    print("\nTraining and evaluation demonstration completed!") 