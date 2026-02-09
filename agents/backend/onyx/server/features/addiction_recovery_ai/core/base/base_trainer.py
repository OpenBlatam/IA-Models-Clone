"""
Base Trainer Interface
Abstract base classes for training with complete implementation
Enhanced with callbacks, early stopping, LR scheduling, gradient accumulation, and multi-GPU support
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Callable
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for trainers with complete training loop implementation
    Supports callbacks, early stopping, LR scheduling, gradient accumulation, and multi-GPU
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        use_multi_gpu: bool = False,
        use_ddp: bool = False
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            device: Device to use
            use_mixed_precision: Use mixed precision (FP16)
            gradient_clip_val: Gradient clipping value
            accumulate_grad_batches: Gradient accumulation steps
            use_multi_gpu: Use DataParallel for multi-GPU
            use_ddp: Use DistributedDataParallel (for distributed training)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.use_ddp = use_ddp
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU setup
        if self.use_ddp:
            self.model = DistributedDataParallel(self.model)
            logger.info("Using DistributedDataParallel")
        elif self.use_multi_gpu:
            self.model = DataParallel(self.model)
            logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def add_callback(self, callback: Callable):
        """Add training callback"""
        self.callbacks.append(callback)
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin(self)
    
    def remove_callback(self, callback: Callable):
        """Remove training callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _call_callbacks(self, method_name: str, **kwargs):
        """Call callback method on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    getattr(callback, method_name)(self, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback {callback} failed: {e}")
    
    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        if self.gradient_clip_val > 0:
            if self.use_multi_gpu or self.use_ddp:
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.gradient_clip_val)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
    
    def _check_nan_inf(self, loss: torch.Tensor) -> bool:
        """Check for NaN/Inf in loss"""
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf detected in loss: {loss.item()}")
            return True
        return False
    
    @abstractmethod
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step - must be implemented by subclasses
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Dictionary with 'loss' and optionally other metrics
        """
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation and mixed precision
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch
            
        Returns:
            Training metrics
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Call batch begin callback
            self._call_callbacks('on_batch_begin', batch=batch, batch_idx=batch_idx)
            
            # Training step with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    step_output = self.training_step(batch, batch_idx)
                    loss = step_output['loss'] / self.accumulate_grad_batches
                
                # Check for NaN/Inf
                if self._check_nan_inf(loss):
                    optimizer.zero_grad()
                    continue
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                step_output = self.training_step(batch, batch_idx)
                loss = step_output['loss'] / self.accumulate_grad_batches
                
                if self._check_nan_inf(loss):
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Clip gradients
                if self.use_mixed_precision:
                    self.scaler.unscale_(optimizer)
                    self._clip_gradients()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    self._clip_gradients()
                    optimizer.step()
                
                optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += step_output['loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * self.accumulate_grad_batches})
            
            # Call batch end callback
            self._call_callbacks('on_batch_end', batch=batch, loss=loss.item(), batch_idx=batch_idx)
        
        # Final gradient step if needed
        if num_batches % self.accumulate_grad_batches != 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(optimizer)
                self._clip_gradients()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                self._clip_gradients()
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    @abstractmethod
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single validation step - must be implemented by subclasses
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        pass
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = self._move_batch_to_device(batch)
                
                if self.use_mixed_precision:
                    with autocast():
                        step_output = self.validation_step(batch, batch_idx)
                else:
                    step_output = self.validation_step(batch, batch_idx)
                
                total_loss += step_output.get('loss', torch.tensor(0.0)).item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
        else:
            return batch.to(self.device) if isinstance(batch, torch.Tensor) else batch
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        scheduler: Optional[Any] = None,
        experiment_tracker: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full training loop with callbacks, early stopping, and experiment tracking
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of epochs
            scheduler: Learning rate scheduler
            experiment_tracker: Experiment tracker (TensorBoard/WandB)
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        # Call train begin callbacks
        self._call_callbacks('on_train_begin')
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        try:
            for epoch in range(num_epochs):
                # Call epoch begin callback
                self._call_callbacks('on_epoch_begin', epoch=epoch)
                
                # Training
                train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)
                history['train_loss'].append(train_metrics['loss'])
                
                # Validation
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self.validate(val_loader, criterion)
                    history['val_loss'].append(val_metrics['loss'])
                    
                    # Track best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        if self.use_multi_gpu or self.use_ddp:
                            best_model_state = self.model.module.state_dict().copy()
                        else:
                            best_model_state = self.model.state_dict().copy()
                
                history['epochs'].append(epoch)
                
                # Learning rate scheduling
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                    else:
                        scheduler.step()
                
                # Log metrics
                if experiment_tracker:
                    experiment_tracker.log_metrics({
                        'train_loss': train_metrics['loss'],
                        **val_metrics
                    }, step=epoch, prefix='epoch')
                
                # Combine metrics for callbacks
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Call epoch end callback
                self._call_callbacks('on_epoch_end', epoch=epoch, metrics=epoch_metrics)
                
                # Check for early stopping
                early_stop = False
                for callback in self.callbacks:
                    if hasattr(callback, 'should_stop') and callback.should_stop:
                        early_stop = True
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                if early_stop:
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            raise
        
        finally:
            # Restore best model
            if best_model_state is not None:
                if self.use_multi_gpu or self.use_ddp:
                    self.model.module.load_state_dict(best_model_state)
                else:
                    self.model.load_state_dict(best_model_state)
                logger.info("Restored best model state")
            
            # Call train end callbacks
            self._call_callbacks('on_train_end')
        
        return {
            'history': history,
            'best_val_loss': best_val_loss,
            'num_epochs_completed': len(history['epochs'])
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators
    """
    
    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            device: Device to use
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Metrics dictionary
        """
        pass






