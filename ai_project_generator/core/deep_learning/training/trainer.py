"""
Trainer - Main Training Loop with Best Practices
==================================================

Implements training with:
- Mixed precision (AMP)
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Gradient clipping
- NaN/Inf detection
- Experiment tracking
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_weights(model)
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _save_weights(self, model: nn.Module) -> None:
        """Save model weights."""
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def _restore_weights(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class TrainingConfig:
    """Configuration class for training."""
    
    def __init__(
        self,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True,
        use_multi_gpu: bool = False,
        device: Optional[torch.device] = None,
        save_dir: Optional[Path] = None,
        log_interval: int = 10,
        eval_interval: int = 1,
        early_stopping: Optional[EarlyStopping] = None
    ):
        """
        Initialize training configuration.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            use_mixed_precision: Use mixed precision training
            use_multi_gpu: Use DataParallel/DistributedDataParallel
            device: Target device
            save_dir: Directory to save checkpoints
            log_interval: Logging interval in batches
            eval_interval: Evaluation interval in epochs
            early_stopping: Early stopping callback
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.use_multi_gpu = use_multi_gpu
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.early_stopping = early_stopping
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)


class Trainer:
    """
    Main trainer class implementing best practices for deep learning training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Training configuration
            optimizer: Optimizer (created from config if None)
            scheduler: Learning rate scheduler (optional)
            loss_fn: Loss function (optional)
        """
        self.model = model
        self.config = config
        self.scheduler = scheduler
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Setup multi-GPU if requested
        if config.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Loss function
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast() if self.config.use_mixed_precision else torch.no_grad():
                outputs = self._forward_pass(batch)
                loss = self._compute_loss(outputs, batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            metrics = self._compute_metrics(outputs, batch)
            total_correct += metrics.get('correct', 0)
            total_samples += metrics.get('total', 0)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}")
                raise ValueError("Training loss became NaN/Inf")
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                batch = self._move_to_device(batch)
                
                with autocast() if self.config.use_mixed_precision else torch.no_grad():
                    outputs = self._forward_pass(batch)
                    loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                metrics = self._compute_metrics(outputs, batch)
                total_correct += metrics.get('correct', 0)
                total_samples += metrics.get('total', 0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics.get('accuracy', 0.0))
            
            # Validate
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics.get('accuracy', 0.0))
                
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics.get('accuracy', 0.0):.4f}"
                )
                
                # Early stopping
                if self.config.early_stopping:
                    if self.config.early_stopping(val_metrics['loss'], self.model):
                        logger.info("Early stopping triggered")
                        break
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if self.config.save_dir:
                        self._save_checkpoint(epoch, val_metrics)
            else:
                logger.info(
                    f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}"
                )
        
        return self.history
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through model."""
        if isinstance(batch, dict):
            return self.model(**batch)
        elif isinstance(batch, (tuple, list)):
            return self.model(*batch)
        else:
            return self.model(batch)
    
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss."""
        if isinstance(batch, dict) and 'labels' in batch:
            return self.loss_fn(outputs, batch['labels'])
        elif isinstance(batch, tuple) and len(batch) == 2:
            return self.loss_fn(outputs, batch[1])
        else:
            raise ValueError("Cannot determine labels from batch")
    
    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        """Compute accuracy metrics."""
        if isinstance(batch, dict) and 'labels' in batch:
            labels = batch['labels']
        elif isinstance(batch, tuple) and len(batch) == 2:
            labels = batch[1]
        else:
            return {'correct': 0, 'total': 0}
        
        if outputs.dim() > 1:
            predictions = outputs.argmax(dim=-1)
        else:
            predictions = (outputs > 0.5).long()
        
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        
        return {'correct': correct, 'total': total}
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return tuple(v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for v in batch)
        else:
            return batch.to(self.config.device) if isinstance(batch, torch.Tensor) else batch
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        if self.config.save_dir:
            checkpoint_path = self.config.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_checkpoint(
                checkpoint_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics
            )



