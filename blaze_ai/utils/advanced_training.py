"""
Advanced training system following official best practices.
PyTorch, Transformers, Diffusers, and Gradio training techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import warnings

class TrainingMode(Enum):
    """Training modes following best practices."""
    SINGLE_GPU = "single_gpu"
    DATA_PARALLEL = "data_parallel"
    DISTRIBUTED = "distributed"
    MIXED_PRECISION = "mixed_precision"

@dataclass
class TrainingConfig:
    """Training configuration following best practices."""
    # Basic training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Advanced features
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    
    # Multi-GPU
    training_mode: TrainingMode = TrainingMode.SINGLE_GPU
    gpu_ids: List[int] = None
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Monitoring
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Advanced optimizations
    enable_amp: bool = True
    enable_channels_last: bool = True
    enable_compile: bool = True
    enable_gradient_scaling: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    enable_lr_finding: bool = False
    enable_swa: bool = False  # Stochastic Weight Averaging
    
    # Library optimizations
    enable_xformers: bool = True
    enable_flash_attn: bool = True
    enable_apex: bool = True
    enable_triton: bool = True
    enable_deepspeed: bool = False

class AdvancedTrainer:
    """Advanced trainer following best practices."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        self._setup_multi_gpu()
        self._setup_gradient_checkpointing()
        self._setup_advanced_optimizations()
        self._setup_library_optimizations()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.training_losses = []
        self.validation_losses = []
        self.learning_rates = []
        self.training_metrics = []
        self.validation_metrics = []
        
        # Early stopping
        if self.config.enable_early_stopping:
            self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        # SWA setup
        if self.config.enable_swa:
            self._setup_swa()
    
    def _setup_optimizer(self):
        """Setup optimizer following best practices."""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        elif self.config.optimizer.lower() == "lion":
            # Lion optimizer for better performance
            try:
                from lion_pytorch import Lion
                self.optimizer = Lion(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            except ImportError:
                warnings.warn("Lion optimizer not available, falling back to AdamW")
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler following best practices."""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif self.config.scheduler.lower() == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True,
                min_lr=1e-7
            )
        elif self.config.scheduler.lower() == "onecycle":
            # OneCycleLR for better convergence
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=100,  # Will be updated during training
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training following best practices."""
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            self.autocast = autocast
        else:
            self.scaler = None
            self.autocast = self._noop_context
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU training following best practices."""
        if not torch.cuda.is_available():
            return
        
        if self.config.gpu_ids and len(self.config.gpu_ids) > 1:
            if self.config.training_mode == TrainingMode.DATA_PARALLEL:
                self.model = DataParallel(
                    self.model,
                    device_ids=self.config.gpu_ids,
                    dim=0,
                    output_device=self.config.gpu_ids[0]
                )
            elif self.config.training_mode == TrainingMode.DISTRIBUTED:
                # Distributed training setup
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend='nccl',
                        init_method='env://'
                    )
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.device],
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                    broadcast_buffers=False
                )
        
        self.model = self.model.to(self.device)
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing following best practices."""
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
    
    def _setup_advanced_optimizations(self):
        """Setup advanced optimizations."""
        # Enable channels last for better performance
        if self.config.enable_channels_last and hasattr(torch, 'channels_last'):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                warnings.warn("Failed to enable channels last memory format")
        
        # Enable model compilation
        if self.config.enable_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune', fullgraph=True)
            except Exception as e:
                warnings.warn(f"Model compilation failed: {e}")
    
    def _setup_library_optimizations(self):
        """Setup library-specific optimizations."""
        if torch.cuda.is_available():
            # Setup xformers
            if self.config.enable_xformers:
                self._setup_xformers()
            
            # Setup flash attention
            if self.config.enable_flash_attn:
                self._setup_flash_attention()
            
            # Setup apex
            if self.config.enable_apex:
                self._setup_apex()
            
            # Setup triton
            if self.config.enable_triton:
                self._setup_triton()
            
            # Setup deepspeed
            if self.config.enable_deepspeed:
                self._setup_deepspeed()
    
    def _setup_xformers(self):
        """Setup xformers optimizations."""
        try:
            import xformers
            import xformers.ops as xops
            
            # Enable xformers memory efficient attention
            if hasattr(xops, 'memory_efficient_attention'):
                self.xformers_available = True
                self.xformers_ops = xops
                
                # Apply xformers optimizations to model
                if hasattr(self.model, 'config'):
                    if hasattr(self.model.config, 'attention_mode'):
                        self.model.config.attention_mode = "xformers"
                    if hasattr(self.model.config, 'use_memory_efficient_attention'):
                        self.model.config.use_memory_efficient_attention = True
                        
            else:
                self.xformers_available = False
                
        except ImportError:
            warnings.warn("xformers not available, falling back to standard attention")
            self.xformers_available = False
    
    def _setup_flash_attention(self):
        """Setup flash attention optimizations."""
        try:
            import flash_attn
            
            # Enable flash attention
            self.flash_attn_available = True
            self.flash_attn = flash_attn
            
            # Apply flash attention optimizations to model
            if hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'attention_mode'):
                    self.model.config.attention_mode = "flash_attention_2"
                if hasattr(self.model.config, 'use_flash_attention'):
                    self.model.config.use_flash_attention = True
                    
        except ImportError:
            warnings.warn("flash-attn not available, falling back to standard attention")
            self.flash_attn_available = False
    
    def _setup_apex(self):
        """Setup NVIDIA Apex optimizations."""
        try:
            import apex
            from apex import amp as apex_amp
            
            # Enable apex mixed precision
            self.apex_available = True
            self.apex_amp = apex_amp
            
            # Apply apex optimizations to model
            if self.config.get('enable_apex_amp', True):
                self.model, self.optimizer = self.apex_amp.initialize(
                    self.model, self.optimizer, opt_level='O2'
                )
                
        except ImportError:
            warnings.warn("NVIDIA Apex not available, using PyTorch AMP")
            self.apex_available = False
    
    def _setup_triton(self):
        """Setup Triton optimizations."""
        try:
            import triton
            
            # Enable triton optimizations
            self.triton_available = True
            self.triton = triton
            
        except ImportError:
            warnings.warn("Triton not available")
            self.triton_available = False
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed optimizations."""
        try:
            import deepspeed
            
            # Enable deepspeed optimizations
            self.deepspeed_available = True
            self.deepspeed = deepspeed
            
            # DeepSpeed configuration
            self.deepspeed_config = {
                "train_batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.config.learning_rate,
                        "warmup_num_steps": 100
                    }
                },
                "fp16": {
                    "enabled": self.config.mixed_precision
                },
                "zero_optimization": {
                    "stage": 2
                }
            }
            
        except ImportError:
            warnings.warn("DeepSpeed not available")
            self.deepspeed_available = False
    
    def _setup_swa(self):
        """Setup Stochastic Weight Averaging."""
        try:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, 
                swa_lr=0.05
            )
            self.swa_start = 5
            self.swa_interval = 1
        except Exception as e:
            warnings.warn(f"SWA setup failed: {e}")
            self.config.enable_swa = False
    
    @contextmanager
    def _noop_context(self):
        """No-op context manager for when autocast is disabled."""
        yield
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch following best practices."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = []
        
        # Update scheduler steps for OneCycleLR
        if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move data to device
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Convert to channels last if enabled
            if self.config.enable_channels_last and hasattr(torch, 'channels_last'):
                try:
                    data = data.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            
            # Training step
            loss, metrics = self._training_step(data, targets)
            epoch_loss += loss
            if metrics:
                epoch_metrics.append(metrics)
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_training_step(batch_idx, loss, metrics)
            
            # Update step counter
            self.current_step += 1
        
        # Update epoch counter
        self.current_epoch += 1
        
        # Update scheduler
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(epoch_loss / len(dataloader))
        elif isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            pass  # Already updated each step
        else:
            self.scheduler.step()
        
        # SWA update
        if self.config.enable_swa and self.current_epoch >= self.swa_start:
            if (self.current_epoch - self.swa_start) % self.swa_interval == 0:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
        
        # Record metrics
        avg_loss = epoch_loss / len(dataloader)
        avg_metrics = self._average_metrics(epoch_metrics) if epoch_metrics else {}
        
        self.training_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        if avg_metrics:
            self.training_metrics.append(avg_metrics)
        
        return {'epoch_loss': avg_loss, 'learning_rate': self.optimizer.param_groups[0]['lr'], **avg_metrics}
    
    def _training_step(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Optional[Dict[str, float]]]:
        """Single training step following best practices."""
        # Zero gradients efficiently
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with self.autocast():
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler and self.config.enable_gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping:
            if self.scaler and self.config.enable_gradient_scaling:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )
        
        # Optimizer step
        if self.scaler and self.config.enable_gradient_scaling:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(outputs, targets)
        
        return loss.item(), metrics
    
    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate additional training metrics."""
        metrics = {}
        
        # Accuracy
        if outputs.dim() > 1:
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            accuracy = 100 * correct / targets.size(0)
            metrics['accuracy'] = accuracy
        
        # Loss components
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        metrics['ce_loss'] = ce_loss.mean().item()
        
        return metrics
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    def _log_training_step(self, batch_idx: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log training step information."""
        log_msg = f'Epoch: {self.current_epoch}, Batch: {batch_idx}, Loss: {loss:.6f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
        
        if metrics:
            for key, value in metrics.items():
                log_msg += f', {key}: {value:.4f}'
        
        print(log_msg)
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model following best practices."""
        # Use SWA model for validation if enabled
        if self.config.enable_swa and self.current_epoch >= self.swa_start:
            model_to_validate = self.swa_model
        else:
            model_to_validate = self.model
        
        model_to_validate.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_metrics = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Convert to channels last if enabled
                if self.config.enable_channels_last and hasattr(torch, 'channels_last'):
                    try:
                        data = data.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                
                with self.autocast():
                    outputs = model_to_validate(data)
                    loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if outputs.dim() > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                # Calculate additional metrics
                metrics = self._calculate_metrics(outputs, targets)
                all_metrics.append(metrics)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0
        avg_metrics = self._average_metrics(all_metrics) if all_metrics else {}
        
        # Record validation metrics
        self.validation_losses.append(avg_loss)
        if avg_metrics:
            self.validation_metrics.append(avg_metrics)
        
        # Early stopping check
        if self.config.enable_early_stopping:
            should_stop = self.early_stopping(avg_loss)
            if should_stop:
                print(f"Early stopping triggered at epoch {self.current_epoch}")
        
        return {'validation_loss': avg_loss, 'accuracy': accuracy, **avg_metrics}
    
    def find_lr(self, dataloader: DataLoader, start_lr: float = 1e-7, end_lr: float = 10, num_iter: int = 100) -> Tuple[List[float], List[float]]:
        """Find optimal learning rate using learning rate finder."""
        if not self.config.enable_lr_finding:
            return [], []
        
        # Save original learning rate
        original_lr = self.optimizer.param_groups[0]['lr']
        
        # Setup LR finder
        try:
            import torch_lr_finder
            lr_finder = torch_lr_finder.LRFinder(self.model, self.optimizer, F.cross_entropy, device=self.device)
            
            # Run LR finder
            lr_finder.range_test(dataloader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
            
            # Get results
            lrs, losses = lr_finder.plot()
            
            # Reset to original learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = original_lr
            
            # Clean up
            lr_finder.reset()
            
            return lrs, losses
            
        except ImportError:
            warnings.warn("torch-lr-finder not available")
            return [], []
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint following best practices."""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_metric': self.best_metric,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'learning_rates': self.learning_rates,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.config.enable_swa:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint following best practices."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'swa_model_state_dict' in checkpoint and self.config.enable_swa:
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.training_losses = checkpoint.get('training_losses', [])
        self.validation_losses = checkpoint.get('validation_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.training_metrics = checkpoint.get('training_metrics', [])
        self.validation_metrics = checkpoint.get('validation_metrics', [])
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'best_metric': self.best_metric,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'learning_rates': self.learning_rates,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'xformers_available': getattr(self, 'xformers_available', False),
            'flash_attn_available': getattr(self, 'flash_attn_available', False),
            'apex_available': getattr(self, 'apex_available', False),
            'triton_available': getattr(self, 'triton_available', False),
            'deepspeed_available': getattr(self, 'deepspeed_available', False)
        }

class EarlyStopping:
    """Early stopping implementation following best practices."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class TransformersTrainer(AdvancedTrainer):
    """Transformers-specific trainer following best practices."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        super().__init__(model, config)
        
        # Transformers-specific setup
        self._setup_transformers_optimizations()
    
    def _setup_transformers_optimizations(self):
        """Setup Transformers-specific optimizations."""
        # Enable gradient checkpointing for Transformers models
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Set attention mode for memory efficiency
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'attention_mode'):
            if self.xformers_available:
                self.model.config.attention_mode = "xformers"
            elif self.flash_attn_available:
                self.model.config.attention_mode = "flash_attention_2"
            else:
                self.model.config.attention_mode = "flash_attention_2"
        
        # Enable memory efficient attention
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_memory_efficient_attention'):
            self.model.config.use_memory_efficient_attention = True
    
    def _training_step(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Optional[Dict[str, float]]]:
        """Transformers-specific training step."""
        # Handle different input formats
        if isinstance(data, dict):
            # Move all inputs to device
            inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in data.items()}
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            with self.autocast():
                outputs = self.model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else F.cross_entropy(outputs.logits, targets)
        else:
            # Standard tensor input
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with self.autocast():
                outputs = self.model(data)
                loss = F.cross_entropy(outputs.logits, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler and self.config.enable_gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping:
            if self.scaler and self.config.enable_gradient_scaling:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )
        
        # Optimizer step
        if self.scaler and self.config.enable_gradient_scaling:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Calculate metrics
        metrics = self._calculate_metrics(outputs, targets)
        
        return loss.item(), metrics

class DiffusionTrainer(AdvancedTrainer):
    """Diffusion model trainer following best practices."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        super().__init__(model, config)
        
        # Diffusion-specific setup
        self._setup_diffusion_optimizations()
    
    def _setup_diffusion_optimizations(self):
        """Setup diffusion-specific optimizations."""
        # Enable gradient checkpointing for diffusion models
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def _training_step(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Optional[Dict[str, float]]]:
        """Diffusion-specific training step."""
        # Handle diffusion model inputs
        if isinstance(data, dict):
            # Move all inputs to device
            inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in data.items()}
            
            # Forward pass
            with self.autocast():
                outputs = self.model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        else:
            # Standard tensor input
            data = data.to(self.device, non_blocking=True)
            
            with self.autocast():
                outputs = self.model(data)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler and self.config.enable_gradient_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping:
            if self.scaler and self.config.enable_gradient_scaling:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )
        
        # Optimizer step
        if self.scaler and self.config.enable_gradient_scaling:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Calculate metrics
        metrics = self._calculate_metrics(outputs, targets)
        
        return loss.item(), metrics

# Factory functions
def create_trainer(model: nn.Module, config: TrainingConfig, trainer_type: str = "advanced") -> AdvancedTrainer:
    """Create appropriate trainer instance."""
    if trainer_type == "transformers":
        return TransformersTrainer(model, config)
    elif trainer_type == "diffusion":
        return DiffusionTrainer(model, config)
    else:
        return AdvancedTrainer(model, config)

__all__ = [
    "TrainingMode",
    "TrainingConfig", 
    "AdvancedTrainer",
    "TransformersTrainer",
    "DiffusionTrainer",
    "EarlyStopping",
    "create_trainer"
]
