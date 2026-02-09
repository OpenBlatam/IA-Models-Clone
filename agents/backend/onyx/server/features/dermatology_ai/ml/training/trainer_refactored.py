"""
Refactored Trainer Class
Improved structure with callbacks, better error handling, and cleaner code
Enhanced with validation and common utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Callable, Any
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import warnings

from .losses import MultiTaskLoss
from .metrics import MetricCalculator
from .distributed import (
    wrap_model_for_distributed,
    is_main_process,
    reduce_tensor,
    synchronize
)
from .callbacks import (
    TrainingCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    MetricsLoggingCallback
)
from .validation import TrainingValidator

# Import common utilities
import sys
from pathlib import Path as PathLib
_parent = PathLib(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from ml.common.utils import get_device, format_time, set_seed
    from ml.common.errors import TrainingError, ValidationError
    from ml.common.validators import InputValidator
    COMMON_AVAILABLE = True
except ImportError:
    COMMON_AVAILABLE = False
    import warnings
    warnings.warn("Common utilities not available, using fallbacks")

logger = logging.getLogger(__name__)


class RefactoredTrainer:
    """
    Refactored trainer with improved structure:
    - Callback system for extensibility
    - Better error handling
    - Cleaner code organization
    - Improved logging
    - Integrated validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = None,
        use_mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        gradient_accumulation_steps: int = 1,
        callbacks: Optional[List[TrainingCallback]] = None,
        loss_fn: Optional[nn.Module] = None,
        use_ddp: bool = False,
        find_unused_parameters: bool = False,
        enable_profiling: bool = False,
        enable_anomaly_detection: bool = False,
        validate_setup: bool = True
    ):
        # Get device
        if COMMON_AVAILABLE:
            self.device = get_device(device)
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = model.to(self.device)
        
        # Wrap for distributed training
        if use_ddp or torch.cuda.device_count() > 1:
            self.model = wrap_model_for_distributed(
                self.model,
                self.device,
                use_ddp=use_ddp,
                find_unused_parameters=find_unused_parameters
            )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_mixed_precision = use_mixed_precision and self.device.type != "cpu"
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.loss_fn = loss_fn or MultiTaskLoss()
        self.enable_profiling = enable_profiling
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Profiling
        self.profiler = None
        if self.enable_profiling:
            self._setup_profiler()
        
        # Anomaly detection
        if self.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - this will slow down training")
        
        # Training state
        self.current_epoch = 0
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.should_stop = False
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Metrics calculator
        self.metric_calculator = MetricCalculator()
        
        # Gradient accumulation state
        self.accumulation_step = 0
        
        # Validation
        if validate_setup and COMMON_AVAILABLE:
            try:
                TrainingValidator.validate_before_training(
                    model=self.model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    device=str(self.device)
                )
                logger.info("Training setup validated successfully")
            except Exception as e:
                logger.warning(f"Training setup validation failed: {e}")
    
    def _setup_profiler(self):
        """Setup PyTorch profiler"""
        try:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
                record_shapes=True,
                with_stack=True
            )
        except Exception as e:
            logger.warning(f"Could not setup profiler: {e}")
            self.profiler = None
    
    def add_callback(self, callback: TrainingCallback):
        """Add a training callback"""
        self.callbacks.append(callback)
        logger.debug(f"Added callback: {callback.__class__.__name__}")
    
    def _trigger_callbacks(self, event: str, **kwargs):
        """Trigger callbacks for an event"""
        for callback in self.callbacks:
            try:
                if hasattr(callback, event):
                    getattr(callback, event)(**kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback.__class__.__name__}: {e}", exc_info=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        self.accumulation_step = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            disable=not is_main_process()
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Trigger batch start callback
            self._trigger_callbacks('on_batch_start', batch_idx=batch_idx, trainer=self)
            
            try:
                # Validate batch if common utilities available
                if COMMON_AVAILABLE:
                    if not InputValidator.validate_batch(batch, required_keys=['image']):
                        logger.warning(f"Invalid batch at index {batch_idx}, skipping")
                        continue
                
                # Move to device
                images = batch['image'].to(self.device, non_blocking=True)
                targets = self._prepare_targets(batch)
                
                # Forward pass
                loss = self._forward_pass(images, targets)
                
                # Backward pass
                self._backward_pass(loss)
                
                # Update metrics
                with torch.no_grad():
                    outputs = self.model(images)
                    running_loss += loss.item() * self.gradient_accumulation_steps
                    all_predictions.append(self._extract_predictions(outputs))
                    all_targets.append(self._extract_targets(targets))
                
                # Update progress bar
                if is_main_process():
                    pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
                
                # Trigger batch end callback
                self._trigger_callbacks(
                    'on_batch_end',
                    batch_idx=batch_idx,
                    loss=loss.item() * self.gradient_accumulation_steps,
                    trainer=self
                )
                
                # Profiling
                if self.profiler:
                    self.profiler.step()
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}")
                    if self.enable_anomaly_detection:
                        raise RuntimeError("NaN/Inf loss detected with anomaly detection enabled")
                    break
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}", exc_info=True)
                if self.enable_anomaly_detection:
                    raise
                continue
        
        # Synchronize for distributed training
        synchronize()
        
        # Reduce metrics
        running_loss = reduce_tensor(
            torch.tensor(running_loss, device=self.device)
        ).item()
        
        # Calculate metrics
        metrics = self.metric_calculator.calculate_metrics(
            all_predictions,
            all_targets
        )
        
        avg_loss = running_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            **metrics
        }
    
    def _forward_pass(
        self,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with mixed precision"""
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(images)
                loss_dict = self.loss_fn(outputs, targets)
                loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                loss = loss / self.gradient_accumulation_steps
        else:
            outputs = self.model(images)
            loss_dict = self.loss_fn(outputs, targets)
            loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
            loss = loss / self.gradient_accumulation_steps
        
        return loss
    
    def _backward_pass(self, loss: torch.Tensor):
        """Backward pass with gradient accumulation"""
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.accumulation_step += 1
        
        if self.accumulation_step % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.gradient_clip_val > 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
            
            # Optimizer step
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Validating",
                disable=not is_main_process()
            )
            
            for batch in pbar:
                # Validate batch if common utilities available
                if COMMON_AVAILABLE:
                    if not InputValidator.validate_batch(batch, required_keys=['image']):
                        logger.warning("Invalid validation batch, skipping")
                        continue
                
                images = batch['image'].to(self.device, non_blocking=True)
                targets = self._prepare_targets(batch)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss_dict = self.loss_fn(outputs, targets)
                        loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                else:
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, targets)
                    loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                
                running_loss += loss.item()
                all_predictions.append(self._extract_predictions(outputs))
                all_targets.append(self._extract_targets(targets))
        
        # Synchronize
        synchronize()
        
        # Reduce metrics
        running_loss = reduce_tensor(
            torch.tensor(running_loss, device=self.device)
        ).item()
        
        # Calculate metrics
        metrics = self.metric_calculator.calculate_metrics(
            all_predictions,
            all_targets
        )
        
        avg_loss = running_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            **metrics
        }
    
    def _prepare_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare targets from batch"""
        targets = {}
        for key in ['conditions', 'metrics', 'labels']:
            if key in batch:
                targets[key] = batch[key].to(self.device, non_blocking=True)
        return targets
    
    def _extract_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract predictions from model outputs"""
        if isinstance(outputs, dict):
            return {k: v.cpu() for k, v in outputs.items()}
        else:
            return {'output': outputs.cpu()}
    
    def _extract_targets(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract targets"""
        if isinstance(targets, dict):
            return {k: v.cpu() for k, v in targets.items()}
        else:
            return {'target': targets.cpu()}
    
    def fit(
        self,
        optimizer: optim.Optimizer,
        num_epochs: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None
    ):
        """Train the model"""
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if criterion:
            self.loss_fn = criterion
        
        # Add scheduler callback if scheduler provided
        if scheduler and not any(isinstance(cb, LearningRateSchedulerCallback) for cb in self.callbacks):
            self.add_callback(LearningRateSchedulerCallback(scheduler))
        
        # Trigger training start
        self._trigger_callbacks('on_training_start', trainer=self)
        
        # Start profiler
        if self.profiler:
            self.profiler.start()
        
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                if self.should_stop:
                    break
                
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Trigger epoch start
                self._trigger_callbacks('on_epoch_start', epoch=epoch, trainer=self)
                
                if self.should_stop:
                    break
                
                # Train
                train_metrics = self.train_epoch()
                self.training_history['train_loss'].append(train_metrics['loss'])
                self.training_history['train_metrics'].append(train_metrics)
                
                # Validate
                val_metrics = {}
                if self.val_loader:
                    val_metrics = self.validate()
                    self.training_history['val_loss'].append(val_metrics.get('loss', 0))
                    self.training_history['val_metrics'].append(val_metrics)
                
                # Track learning rate
                if self.optimizer:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.training_history['learning_rates'].append(current_lr)
                
                # Trigger epoch end
                all_metrics = {**train_metrics, **val_metrics}
                self._trigger_callbacks(
                    'on_epoch_end',
                    epoch=epoch,
                    metrics=all_metrics,
                    trainer=self
                )
                
                # Log metrics (only on main process)
                if is_main_process():
                    epoch_time = time.time() - epoch_start_time
                    epoch_time_str = format_time(epoch_time) if COMMON_AVAILABLE else f"{epoch_time:.2f}s"
                    
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f} - "
                        f"Val Loss: {val_metrics.get('loss', 'N/A')} - "
                        f"LR: {current_lr:.6f} - "
                        f"Time: {epoch_time_str}"
                    )
                
                # Synchronize
                synchronize()
        
        finally:
            # Stop profiler
            if self.profiler:
                self.profiler.stop()
                if is_main_process():
                    logger.info("Profiling completed. Check ./logs/profiler for results")
            
            # Trigger training end
            self._trigger_callbacks('on_training_end', trainer=self)
            
            if is_main_process():
                total_time = time.time() - start_time
                total_time_str = format_time(total_time) if COMMON_AVAILABLE else f"{total_time:.2f}s"
                logger.info(f"Training completed in {total_time_str}")
    
    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[optim.Optimizer] = None,
        epoch: Optional[int] = None
    ):
        """Save model checkpoint"""
        optimizer = optimizer or self.optimizer
        epoch = epoch if epoch is not None else self.current_epoch
        
        # Get underlying model if wrapped
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'training_history': self.training_history,
            'model_config': {
                'name': getattr(model_to_save, 'name', 'Unknown'),
                'device': str(self.device)
            }
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        path: Path,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        model_to_load = self.model
        if hasattr(self.model, 'module'):
            model_to_load = self.model.module
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        })
        
        logger.info(f"Checkpoint loaded: {path}")
