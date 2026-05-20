"""
Trainer class for model training
Enhanced with distributed training, gradient accumulation, and profiling
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

logger = logging.getLogger(__name__)


class Trainer:
    """
    Enhanced training class with:
    - Distributed training support (DataParallel/DistributedDataParallel)
    - Gradient accumulation
    - Mixed precision training
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling
    - Profiling and debugging
    - Proper evaluation metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int = 10,
        experiment_tracker: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        use_ddp: bool = False,
        find_unused_parameters: bool = False,
        enable_profiling: bool = False,
        enable_anomaly_detection: bool = False
    ):
        self.device = torch.device(device)
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
        self.early_stopping_patience = early_stopping_patience
        self.experiment_tracker = experiment_tracker
        self.loss_fn = loss_fn or MultiTaskLoss()
        self.enable_profiling = enable_profiling
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Profiling
        self.profiler = None
        if self.enable_profiling:
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
        
        # Anomaly detection
        if self.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - this will slow down training")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Metrics calculator
        self.metric_calculator = MetricCalculator()
        
        # Gradient accumulation state
        self.accumulation_step = 0
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        criterion = criterion or self.loss_fn
        
        # Reset accumulation
        self.accumulation_step = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            disable=not is_main_process()
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            
            # Prepare targets
            targets = self._prepare_targets(batch)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self._compute_loss(criterion, outputs, targets)
                    loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss_dict = self._compute_loss(criterion, outputs, targets)
                loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            self.accumulation_step += 1
            if self.accumulation_step % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Update metrics (unscale loss for logging)
            if self.use_mixed_precision:
                # Unscale for logging
                unscaled_loss = loss.item() * self.gradient_accumulation_steps
            else:
                unscaled_loss = loss.item() * self.gradient_accumulation_steps
            
            running_loss += unscaled_loss
            all_predictions.append(self._extract_predictions(outputs))
            all_targets.append(self._extract_targets(targets))
            
            # Update progress bar
            if is_main_process():
                pbar.set_postfix({'loss': unscaled_loss})
            
            # Profiling
            if self.profiler:
                self.profiler.step()
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}")
                if self.enable_anomaly_detection:
                    raise RuntimeError("NaN/Inf loss detected with anomaly detection enabled")
                break
        
        # Synchronize for distributed training
        synchronize()
        
        # Reduce metrics across processes
        running_loss = reduce_tensor(torch.tensor(running_loss, device=self.device)).item()
        
        # Calculate metrics
        metrics = self.metric_calculator.calculate_metrics(
            all_predictions,
            all_targets
        )
        
        # Average loss
        avg_loss = running_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            **metrics
        }
    
    def validate(
        self,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        criterion = criterion or self.loss_fn
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Validating",
                disable=not is_main_process()
            )
            
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = self._prepare_targets(batch)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss_dict = self._compute_loss(criterion, outputs, targets)
                        loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                else:
                    outputs = self.model(images)
                    loss_dict = self._compute_loss(criterion, outputs, targets)
                    loss = loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                
                running_loss += loss.item()
                all_predictions.append(self._extract_predictions(outputs))
                all_targets.append(self._extract_targets(targets))
        
        # Synchronize for distributed training
        synchronize()
        
        # Reduce metrics across processes
        running_loss = reduce_tensor(torch.tensor(running_loss, device=self.device)).item()
        
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
    
    def _compute_loss(
        self,
        criterion: nn.Module,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss from outputs and targets"""
        if isinstance(criterion, MultiTaskLoss):
            return criterion(outputs, targets)
        else:
            # Single loss function
            if isinstance(outputs, dict):
                output = next(iter(outputs.values()))
            else:
                output = outputs
            
            if isinstance(targets, dict):
                target = next(iter(targets.values()))
            else:
                target = targets
            
            loss = criterion(output, target)
            return {'loss': loss}
    
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
        criterion: Optional[nn.Module] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """Train the model"""
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir and is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_main_process():
            logger.info(f"Starting training for {num_epochs} epochs")
            logger.info(f"Device: {self.device}, Mixed Precision: {self.use_mixed_precision}")
            logger.info(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        
        # Start profiler
        if self.profiler:
            self.profiler.start()
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                start_time = time.time()
                
                # Train
                train_metrics = self.train_epoch(optimizer, criterion)
                self.training_history['train_loss'].append(train_metrics['loss'])
                self.training_history['train_metrics'].append(train_metrics)
                
                # Validate
                val_metrics = {}
                if self.val_loader:
                    val_metrics = self.validate(criterion)
                    self.training_history['val_loss'].append(val_metrics.get('loss', 0))
                    self.training_history['val_metrics'].append(val_metrics)
                
                # Learning rate scheduling
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                    else:
                        scheduler.step()
                
                # Log metrics (only on main process)
                if is_main_process():
                    epoch_time = time.time() - start_time
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f} - "
                        f"Val Loss: {val_metrics.get('loss', 'N/A')} - "
                        f"Time: {epoch_time:.2f}s"
                    )
                    
                    # Experiment tracking
                    if self.experiment_tracker:
                        from core.experiment_tracker import ExperimentMetrics
                        self.experiment_tracker.log_metrics(ExperimentMetrics(
                            experiment_id=self.experiment_tracker.current_experiment,
                            epoch=epoch + 1,
                            train_loss=train_metrics['loss'],
                            val_loss=val_metrics.get('loss'),
                            learning_rate=optimizer.param_groups[0]['lr']
                        ))
                    
                    # Early stopping
                    if self.val_loader:
                        val_loss = val_metrics.get('loss', float('inf'))
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            
                            # Save best model
                            if checkpoint_dir:
                                self.save_checkpoint(
                                    checkpoint_dir / f"best_model_epoch_{epoch + 1}.pt",
                                    optimizer,
                                    epoch
                                )
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.early_stopping_patience:
                                logger.info(f"Early stopping at epoch {epoch + 1}")
                                break
                    
                    # Save checkpoint
                    if checkpoint_dir and (epoch + 1) % 10 == 0:
                        self.save_checkpoint(
                            checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                            optimizer,
                            epoch
                        )
                
                # Synchronize all processes
                synchronize()
        
        finally:
            # Stop profiler
            if self.profiler:
                self.profiler.stop()
                if is_main_process():
                    logger.info("Profiling completed. Check ./logs/profiler for results")
        
        if is_main_process():
            logger.info("Training completed")
    
    def save_checkpoint(
        self,
        path: Path,
        optimizer: optim.Optimizer,
        epoch: int
    ):
        """Save model checkpoint"""
        # Get underlying model if wrapped
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path, optimizer: Optional[optim.Optimizer] = None):
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
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        })
        
        logger.info(f"Checkpoint loaded: {path}")


def create_data_loaders(
    train_dataset,
    val_dataset=None,
    test_dataset=None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_distributed: bool = False
) -> Dict[str, DataLoader]:
    """Create data loaders with optimal settings"""
    from torch.utils.data import DataLoader
    from .distributed import DistributedSampler
    
    loaders = {}
    
    # Sampler for distributed training
    train_sampler = None
    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    if val_dataset:
        val_sampler = None
        if use_distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    if test_dataset:
        test_sampler = None
        if use_distributed:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    return loaders
