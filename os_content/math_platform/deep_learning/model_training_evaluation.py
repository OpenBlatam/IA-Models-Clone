from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import warnings
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import copy
        from sklearn.metrics import roc_curve
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Model Training and Evaluation System
Production-ready implementation of comprehensive training and evaluation pipelines.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model parameters
    model_name: str = "custom_model"
    model_type: str = "transformer"  # transformer, diffusion, cnn, rnn
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    save_best_model: bool = True
    
    # Optimization parameters
    optimizer: str = "adamw"  # adam, adamw, sgd, lion
    scheduler: str = "cosine"  # cosine, linear, step, exponential
    lr_scheduler_warmup_steps: int = 1000
    lr_scheduler_total_steps: int = 10000
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Evaluation parameters
    eval_steps: int = 500
    eval_strategy: str = "steps"  # steps, epoch
    eval_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    eval_batch_size: int = 32
    
    # Logging and monitoring
    logging_steps: int = 100
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Output parameters
    output_dir: str = "./training_outputs"
    experiment_name: str = "experiment"
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    # Data parameters
    train_data_path: str = ""
    eval_data_path: str = ""
    test_data_path: str = ""
    max_seq_length: int = 512
    num_classes: int = 2
    
    # Advanced parameters
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Evaluation parameters
    eval_batch_size: int = 32
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"])
    eval_strategy: str = "epoch"  # epoch, steps
    
    # Model parameters
    model_path: str = ""
    model_type: str = "transformer"
    
    # Data parameters
    test_data_path: str = ""
    max_seq_length: int = 512
    num_classes: int = 2
    
    # Output parameters
    output_dir: str = "./evaluation_outputs"
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_roc_curve: bool = True
    
    # Advanced parameters
    device: str = "cuda"
    dtype: str = "float16"


class BaseTrainer(ABC):
    """Base class for model trainers."""
    
    def __init__(self, config: TrainingConfig, model: nn.Module, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        
    """__init__ function."""
self.config = config
        self.model = model.to(config.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup components
        self._setup_components()
        self._setup_logging()
        
        logger.info(f"Initialized {self.__class__.__name__} with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    @abstractmethod
    def _setup_components(self) -> Any:
        """Setup training components."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        pass
    
    @abstractmethod
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        pass
    
    def _setup_logging(self) -> Any:
        """Setup logging and monitoring."""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup TensorBoard
        if self.config.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(self.config.output_dir, "tensorboard")
            )
        
        # Setup WandB
        if self.config.use_wandb:
            wandb.init(
                project=self.config.experiment_name,
                config=self.config.__dict__,
                name=f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training loop."""
        # Setup data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory
        )
        
        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.pin_memory
            )
        
        # Training history
        history = {
            'train_loss': [],
            'train_metrics': [],
            'eval_loss': [],
            'eval_metrics': []
        }
        
        # Early stopping
        best_eval_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            
            # Evaluation phase
            if eval_loader and (epoch + 1) % self.config.eval_strategy == 0:
                eval_loss, eval_metrics = self._eval_epoch(eval_loader, epoch)
                history['eval_loss'].append(eval_loss)
                history['eval_metrics'].append(eval_metrics)
                
                # Early stopping
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    if self.config.save_best_model:
                        self._save_model("best_model.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Logging
            self._log_epoch(epoch, train_loss, train_metrics, eval_loss if eval_loader else None, eval_metrics if eval_loader else None)
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            step_loss, step_metrics = self.train_step(batch)
            
            total_loss += step_loss
            for key, value in step_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_loss:.4f}",
                'avg_loss': f"{total_loss / (step + 1):.4f}"
            })
            
            # Logging
            if step % self.config.logging_steps == 0:
                self._log_step(epoch, step, step_loss, step_metrics, "train")
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {key: value / len(train_loader) for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _eval_epoch(self, eval_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        
        progress_bar = tqdm(eval_loader, desc=f"Evaluation Epoch {epoch + 1}")
        
        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                # Evaluation step
                step_loss, step_metrics = self.eval_step(batch)
                
                total_loss += step_loss
                for key, value in step_metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_loss:.4f}",
                    'avg_loss': f"{total_loss / (step + 1):.4f}"
                })
        
        # Calculate averages
        avg_loss = total_loss / len(eval_loader)
        avg_metrics = {key: value / len(eval_loader) for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _log_step(self, epoch: int, step: int, loss: float, metrics: Dict[str, float], phase: str):
        """Log training/evaluation step."""
        if self.config.use_tensorboard:
            self.tensorboard_writer.add_scalar(f"{phase}/loss", loss, epoch * len(self.train_dataset) + step)
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{phase}/{key}", value, epoch * len(self.train_dataset) + step)
        
        if self.config.use_wandb:
            log_dict = {f"{phase}/loss": loss}
            for key, value in metrics.items():
                log_dict[f"{phase}/{key}"] = value
            wandb.log(log_dict, step=epoch * len(self.train_dataset) + step)
    
    def _log_epoch(self, epoch: int, train_loss: float, train_metrics: Dict[str, float], 
                   eval_loss: Optional[float], eval_metrics: Optional[Dict[str, float]]):
        """Log epoch results."""
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}")
        for key, value in train_metrics.items():
            logger.info(f"  Train {key}: {value:.4f}")
        
        if eval_loss is not None:
            logger.info(f"  Eval Loss: {eval_loss:.4f}")
            for key, value in eval_metrics.items():
                logger.info(f"  Eval {key}: {value:.4f}")
    
    def _save_model(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.__dict__,
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0
        }
        
        torch.save(checkpoint, os.path.join(self.config.output_dir, filename))
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, model_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {model_path}")


class TransformerTrainer(BaseTrainer):
    """Trainer for transformer models."""
    
    def _setup_components(self) -> Any:
        """Setup transformer training components."""
        # Optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_scheduler_total_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.lr_scheduler_total_steps
            )
        elif self.config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_scheduler_warmup_steps,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single training step for transformer."""
        # Move batch to device
        batch = {key: value.to(self.config.device) for key, value in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            outputs = self.model(**batch)
            loss = self.criterion(outputs.logits, batch['labels'])
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Calculate metrics
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = (predictions == batch['labels']).float().mean().item()
        
        metrics = {
            'accuracy': accuracy
        }
        
        return loss.item(), metrics
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single evaluation step for transformer."""
        # Move batch to device
        batch = {key: value.to(self.config.device) for key, value in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            outputs = self.model(**batch)
            loss = self.criterion(outputs.logits, batch['labels'])
        
        # Calculate metrics
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = (predictions == batch['labels']).float().mean().item()
        
        metrics = {
            'accuracy': accuracy
        }
        
        return loss.item(), metrics


class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion models."""
    
    def _setup_components(self) -> Any:
        """Setup diffusion training components."""
        # Optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_scheduler_total_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        else:
            self.scheduler = None
        
        # Loss function (MSE for noise prediction)
        self.criterion = nn.MSELoss()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single training step for diffusion."""
        # Move batch to device
        batch = {key: value.to(self.config.device) for key, value in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            noise_pred = self.model(batch['noisy_images'], batch['timesteps'])
            loss = self.criterion(noise_pred, batch['noise'])
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        metrics = {
            'mse_loss': loss.item()
        }
        
        return loss.item(), metrics
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single evaluation step for diffusion."""
        # Move batch to device
        batch = {key: value.to(self.config.device) for key, value in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            noise_pred = self.model(batch['noisy_images'], batch['timesteps'])
            loss = self.criterion(noise_pred, batch['noise'])
        
        metrics = {
            'mse_loss': loss.item()
        }
        
        return loss.item(), metrics


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config: EvaluationConfig, model: nn.Module, test_dataset: Dataset):
        
    """__init__ function."""
self.config = config
        self.model = model.to(config.device)
        self.test_dataset = test_dataset
        
        # Setup components
        self._setup_components()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _setup_components(self) -> Any:
        """Setup evaluation components."""
        # Data loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Metrics
        self.metrics = {}
        self.predictions = []
        self.targets = []
    
    def evaluate(self) -> Dict[str, float]:
        """Complete evaluation."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        progress_bar = tqdm(self.test_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = {key: value.to(self.config.device) for key, value in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                if hasattr(outputs, 'loss'):
                    total_loss += outputs.loss.item()
                
                # Get predictions
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets, total_loss / len(self.test_loader))
        
        # Save results
        if self.config.save_predictions:
            self._save_predictions(all_predictions, all_targets)
        
        if self.config.save_confusion_matrix:
            self._save_confusion_matrix(all_predictions, all_targets)
        
        if self.config.save_roc_curve:
            self._save_roc_curve(all_predictions, all_targets)
        
        return metrics
    
    def _calculate_metrics(self, predictions: List[int], targets: List[int], loss: float) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {'loss': loss}
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Accuracy
        if 'accuracy' in self.config.eval_metrics:
            metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Precision, Recall, F1
        if any(metric in self.config.eval_metrics for metric in ['precision', 'recall', 'f1']):
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        # AUC (for binary classification)
        if 'auc' in self.config.eval_metrics and self.config.num_classes == 2:
            try:
                metrics['auc'] = roc_auc_score(targets, predictions)
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    def _save_predictions(self, predictions: List[int], targets: List[int]):
        """Save predictions to file."""
        results = {
            'predictions': predictions,
            'targets': targets,
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.config.output_dir, 'predictions.json')
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2)
        
        logger.info(f"Predictions saved to {output_path}")
    
    def _save_confusion_matrix(self, predictions: List[int], targets: List[int]):
        """Save confusion matrix plot."""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def _save_roc_curve(self, predictions: List[int], targets: List[int]):
        """Save ROC curve plot."""
        if self.config.num_classes != 2:
            return
        
        
        fpr, tpr, _ = roc_curve(targets, predictions)
        auc = roc_auc_score(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        output_path = os.path.join(self.config.output_dir, 'roc_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")


class TrainingManager:
    """High-level training manager."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.trainers = {}
        self.evaluators = {}
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _setup_logging(self) -> Any:
        """Setup logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.config.__dict__, f, indent=2)
    
    def add_trainer(self, name: str, trainer: BaseTrainer):
        """Add a trainer."""
        self.trainers[name] = trainer
        logger.info(f"Added trainer: {name}")
    
    def add_evaluator(self, name: str, evaluator: ModelEvaluator):
        """Add an evaluator."""
        self.evaluators[name] = evaluator
        logger.info(f"Added evaluator: {name}")
    
    def train_all(self) -> Dict[str, Dict[str, List[float]]]:
        """Train all models."""
        results = {}
        
        for name, trainer in self.trainers.items():
            logger.info(f"Training {name}")
            try:
                history = trainer.train()
                results[name] = history
                logger.info(f"Training completed for {name}")
            except Exception as e:
                logger.error(f"Training failed for {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all models."""
        results = {}
        
        for name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating {name}")
            try:
                metrics = evaluator.evaluate()
                results[name] = metrics
                logger.info(f"Evaluation completed for {name}")
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save training/evaluation results."""
        output_path = os.path.join(self.config.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def create_trainer(config: TrainingConfig, model: nn.Module, train_dataset: Dataset, 
                  eval_dataset: Optional[Dataset] = None) -> BaseTrainer:
    """Create a trainer based on model type."""
    if config.model_type == "transformer":
        return TransformerTrainer(config, model, train_dataset, eval_dataset)
    elif config.model_type == "diffusion":
        return DiffusionTrainer(config, model, train_dataset, eval_dataset)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def create_evaluator(config: EvaluationConfig, model: nn.Module, test_dataset: Dataset) -> ModelEvaluator:
    """Create an evaluator."""
    return ModelEvaluator(config, model, test_dataset)


# Example usage
if __name__ == "__main__":
    # Create training configuration
    training_config = TrainingConfig(
        model_name="transformer_classifier",
        model_type="transformer",
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10,
        output_dir="./training_outputs",
        experiment_name="transformer_experiment"
    )
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        eval_batch_size=32,
        eval_metrics=["accuracy", "precision", "recall", "f1"],
        output_dir="./evaluation_outputs"
    )
    
    # Example model (placeholder)
    class SimpleTransformer(nn.Module):
        def __init__(self, config) -> Any:
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(128, 8, batch_first=True),
                num_layers=6
            )
            self.classifier = nn.Linear(128, config.num_classes)
        
        def forward(self, input_ids, labels=None) -> Any:
            x = self.embedding(input_ids)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            logits = self.classifier(x)
            
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()
            else:
                return type('Output', (), {'logits': logits})()
    
    # Create model
    model = SimpleTransformer(training_config)
    
    # Create datasets (placeholder)
    class SimpleDataset(Dataset):
        def __init__(self, size=1000) -> Any:
            self.data = torch.randint(0, 1000, (size, 512))
            self.labels = torch.randint(0, 2, (size,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return {
                'input_ids': self.data[idx],
                'labels': self.labels[idx]
            }
    
    train_dataset = SimpleDataset(1000)
    eval_dataset = SimpleDataset(200)
    test_dataset = SimpleDataset(100)
    
    # Create trainer
    trainer = create_trainer(training_config, model, train_dataset, eval_dataset)
    
    # Train model
    history = trainer.train()
    
    # Create evaluator
    evaluator = create_evaluator(eval_config, model, test_dataset)
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    print("Training completed!")
    print(f"Final metrics: {metrics}") 