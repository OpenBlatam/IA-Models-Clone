"""
PyTorch Primary Framework System
Implements comprehensive PyTorch-based deep learning workflows and best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from torch.jit import script, trace
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from enum import Enum
import json
import yaml
import time
import os
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PYTORCH PRIMARY FRAMEWORK SYSTEM
# =============================================================================

@dataclass
class PyTorchFrameworkConfig:
    """Configuration for PyTorch primary framework system"""
    # Framework configuration
    use_cuda: bool = True
    use_mixed_precision: bool = True
    use_distributed: bool = False
    use_data_parallel: bool = False
    use_gradient_checkpointing: bool = True
    
    # Performance optimization
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True
    memory_efficient_attention: bool = True
    
    # Training configuration
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Model optimization
    compile_model: bool = True  # PyTorch 2.0+ compilation
    use_channels_last: bool = False
    use_amp: bool = True
    
    # Monitoring
    use_tensorboard: bool = True
    use_profiler: bool = False
    log_memory_usage: bool = True

class PyTorchPrimaryFrameworkSystem:
    """Comprehensive system using PyTorch as primary framework"""
    
    def __init__(self, config: PyTorchFrameworkConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup PyTorch framework
        self.device = self._setup_device()
        self._setup_pytorch_optimizations()
        
        # Framework components
        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.tensorboard_writer = None
        self.profiler = None
        
        # Setup monitoring
        self._setup_monitoring()
        
        self.logger.info("PyTorch Primary Framework System initialized")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device for PyTorch operations"""
        if self.config.use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.logger.info(f"Device name: {torch.cuda.get_device_name()}")
            
            # Set default device
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU device")
        
        return device
    
    def _setup_pytorch_optimizations(self):
        """Setup PyTorch performance optimizations"""
        if not torch.cuda.is_available():
            return
        
        # CUDNN optimizations
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            self.logger.info("CUDNN benchmark enabled")
        
        if self.config.enable_cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            self.logger.info("CUDNN deterministic enabled")
        
        # TensorFloat-32 for Ampere GPUs
        if self.config.enable_tf32 and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("TensorFloat-32 enabled for Ampere GPU")
        
        # Memory optimizations
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.empty_cache()
            self.logger.info("CUDA memory cache cleared")
    
    def _setup_monitoring(self):
        """Setup monitoring and logging"""
        if self.config.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir='./runs/pytorch_framework')
            self.logger.info("TensorBoard writer initialized")
        
        if self.config.use_profiler:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/profiler'),
                record_shapes=True,
                with_stack=True
            )
            self.logger.info("PyTorch profiler initialized")
    
    def create_optimized_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        """Create PyTorch DataLoader with optimizations"""
        dataloader_kwargs = {
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.persistent_workers,
            'prefetch_factor': self.config.prefetch_factor,
            **kwargs
        }
        
        # Override with provided kwargs
        dataloader_kwargs.update(kwargs)
        
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        self.logger.info(f"DataLoader created with {dataloader_kwargs}")
        
        return dataloader
    
    def optimize_model_for_pytorch(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch-specific optimizations to model"""
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        # Use channels last memory format for better performance
        if self.config.use_channels_last and hasattr(model, 'to'):
            model = model.to(memory_format=torch.channels_last)
            self.logger.info("Channels last memory format enabled")
        
        # Compile model (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Apply DataParallel if specified
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            model = DataParallel(model)
            self.logger.info(f"DataParallel enabled on {torch.cuda.device_count()} GPUs")
        
        return model
    
    def create_pytorch_optimizer(self, model: nn.Module, optimizer_type: str = "adamw",
                                **kwargs) -> optim.Optimizer:
        """Create PyTorch optimizer with best practices"""
        if optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=kwargs.get('lr', 1e-4),
                weight_decay=kwargs.get('weight_decay', 1e-5),
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=kwargs.get('lr', 1e-4),
                weight_decay=kwargs.get('weight_decay', 1e-5),
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=kwargs.get('lr', 1e-3),
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 1e-5),
                nesterov=kwargs.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        self.logger.info(f"PyTorch optimizer created: {optimizer_type}")
        return optimizer
    
    def create_pytorch_scheduler(self, optimizer: optim.Optimizer, scheduler_type: str = "cosine",
                                **kwargs) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create PyTorch learning rate scheduler"""
        if scheduler_type.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 0)
            )
        elif scheduler_type.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                verbose=True
            )
        elif scheduler_type.lower() == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 1e-3),
                epochs=kwargs.get('epochs', 100),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100)
            )
        else:
            return None
        
        self.logger.info(f"PyTorch scheduler created: {scheduler_type}")
        return scheduler
    
    def train_step_pytorch(self, model: nn.Module, data: torch.Tensor, targets: torch.Tensor,
                           criterion: nn.Module, optimizer: optim.Optimizer,
                           scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                           **kwargs) -> Dict[str, float]:
        """Execute single PyTorch training step with optimizations"""
        # Move data to device
        data = data.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Use channels last if enabled
        if self.config.use_channels_last and data.dim() == 4:
            data = data.to(memory_format=torch.channels_last)
        
        # Forward pass with mixed precision
        if self.config.use_mixed_precision and self.scaler is not None:
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # Backward pass
        if self.config.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if kwargs.get('grad_clip_norm'):
            if self.config.use_mixed_precision and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                kwargs['grad_clip_norm']
            )
        
        # Optimizer step
        if self.config.use_mixed_precision and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Log metrics
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Loss/Train_Step', loss.item(), 
                                            kwargs.get('global_step', 0))
            self.tensorboard_writer.add_scalar('Learning_Rate', 
                                            optimizer.param_groups[0]['lr'],
                                            kwargs.get('global_step', 0))
        
        return {
            'loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    
    def validate_step_pytorch(self, model: nn.Module, data: torch.Tensor, targets: torch.Tensor,
                             criterion: nn.Module) -> Dict[str, float]:
        """Execute single PyTorch validation step"""
        model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Use channels last if enabled
            if self.config.use_channels_last and data.dim() == 4:
                data = data.to(memory_format=torch.channels_last)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
        
        model.train()
        
        return {
            'loss': loss.item(),
            'outputs': outputs
        }
    
    def save_pytorch_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                               scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                               epoch: int = 0, loss: float = 0.0,
                               filename: str = "checkpoint.pth") -> str:
        """Save PyTorch model checkpoint with best practices"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'pytorch_version': torch.__version__,
            'device': str(self.device)
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Create checkpoint directory
        checkpoint_dir = Path('./checkpoints/pytorch_framework')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"PyTorch checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_pytorch_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                               checkpoint_path: str,
                               scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """Load PyTorch model checkpoint with best practices"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"PyTorch checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Checkpoint info: Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.4f}")
        
        return checkpoint
    
    def profile_pytorch_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                             num_runs: int = 100) -> Dict[str, Any]:
        """Profile PyTorch model performance"""
        if not self.config.use_profiler:
            self.logger.warning("Profiler not enabled")
            return {}
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        if self.config.use_channels_last and dummy_input.dim() == 4:
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Profile
        if self.profiler:
            with self.profiler:
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = model(dummy_input)
        
        # Measure memory usage
        if torch.cuda.is_available():
            memory_stats = torch.cuda.memory_stats()
            memory_usage = {
                'allocated_gb': memory_stats['allocated_bytes.all.current'] / (1024**3),
                'reserved_gb': memory_stats['reserved_bytes.all.current'] / (1024**3),
                'max_allocated_gb': memory_stats['allocated_bytes.all.peak'] / (1024**3)
            }
        else:
            memory_usage = {}
        
        self.logger.info(f"PyTorch model profiling completed for {num_runs} runs")
        
        return {
            'memory_usage': memory_usage,
            'profiler_active': self.profiler is not None
        }
    
    def export_pytorch_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                            export_format: str = "torchscript",
                            filename: str = "model") -> str:
        """Export PyTorch model to different formats"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        if self.config.use_channels_last and dummy_input.dim() == 4:
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
        
        export_path = ""
        
        if export_format.lower() == "torchscript":
            # Export to TorchScript
            try:
                traced_model = trace(model, dummy_input)
                export_path = f"./exports/{filename}.pt"
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                traced_model.save(export_path)
                self.logger.info(f"Model exported to TorchScript: {export_path}")
            except Exception as e:
                self.logger.error(f"TorchScript export failed: {e}")
        
        elif export_format.lower() == "onnx":
            # Export to ONNX
            try:
                export_path = f"./exports/{filename}.onnx"
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                self.logger.info(f"Model exported to ONNX: {export_path}")
            except Exception as e:
                self.logger.error(f"ONNX export failed: {e}")
        
        return export_path
    
    def get_pytorch_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive PyTorch memory information"""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        device = torch.cuda.current_device()
        memory_info = {
            "cuda_available": True,
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
            "memory_cached_gb": torch.cuda.memory_reserved(device) / (1024**3),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3)
        }
        
        return memory_info
    
    def clear_pytorch_memory(self):
        """Clear PyTorch memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("PyTorch memory cache cleared")
    
    def cleanup(self):
        """Cleanup PyTorch framework resources"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.profiler:
            self.profiler.stop()
        
        self.clear_pytorch_memory()
        self.logger.info("PyTorch framework cleanup completed")

# =============================================================================
# PYTORCH MODEL ARCHITECTURES
# =============================================================================

class PyTorchTransformerModel(nn.Module):
    """PyTorch-native Transformer model implementation"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PyTorchPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Use PyTorch's native TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',  # Better than ReLU for transformers
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using PyTorch best practices
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using PyTorch best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Create attention mask for causal language modeling
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer encoding
        x = self.transformer(x, mask=mask)
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal attention mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PyTorchCNNModel(nn.Module):
    """PyTorch-native CNN model implementation"""
    
    def __init__(self, input_channels: int, num_classes: int, base_channels: int, num_layers: int):
        super().__init__()
        
        # Use PyTorch's modern CNN building blocks
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate feature size
        feature_size = base_channels * (2 ** (num_layers - 1))
        
        # Use PyTorch's modern classifier design
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using PyTorch best practices"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class PyTorchPositionalEncoding(nn.Module):
    """PyTorch-native positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Use PyTorch's efficient tensor operations
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer for efficient GPU transfer
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# =============================================================================
# PYTORCH TRAINING UTILITIES
# =============================================================================

class PyTorchTrainingUtilities:
    """Utility functions for PyTorch training"""
    
    @staticmethod
    def create_learning_rate_finder(model: nn.Module, optimizer: optim.Optimizer,
                                   criterion: nn.Module, dataloader: DataLoader,
                                   device: torch.device, start_lr: float = 1e-7,
                                   end_lr: float = 10, num_iter: int = 100) -> Dict[str, List[float]]:
        """Create learning rate finder using PyTorch best practices"""
        from torch_lr_finder import LRFinder
        
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda" if device.type == "cuda" else "cpu")
        lr_finder.range_test(dataloader, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
        
        # Get results
        lr_finder.plot()
        lr_finder.reset()
        
        return {
            'learning_rates': lr_finder.history['lr'],
            'losses': lr_finder.history['loss']
        }
    
    @staticmethod
    def create_gradcam_visualization(model: nn.Module, target_layer: nn.Module,
                                   input_tensor: torch.Tensor, target_class: int,
                                   device: torch.device) -> torch.Tensor:
        """Create GradCAM visualization using PyTorch"""
        model.eval()
        
        # Register hooks for gradient and activation
        gradients = []
        activations = []
        
        def save_gradient(grad):
            gradients.append(grad)
        
        def save_activation(module, input, output):
            activations.append(output)
        
        # Register hooks
        target_layer.register_forward_hook(save_activation)
        
        # Forward pass
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = gradients[0]
        activations = activations[0]
        
        # Calculate weights
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Generate CAM
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0), size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_pytorch_framework_example():
    """Example of using PyTorch Primary Framework System"""
    
    print("=== PyTorch Primary Framework System Example ===")
    
    # Configuration
    config = PyTorchFrameworkConfig(
        use_cuda=True,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        compile_model=True,
        use_tensorboard=True
    )
    
    # Create system
    pytorch_system = PyTorchPrimaryFrameworkSystem(config)
    
    # Create model
    model = PyTorchTransformerModel(
        vocab_size=30000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1
    )
    
    # Optimize model for PyTorch
    model = pytorch_system.optimize_model_for_pytorch(model)
    
    # Create optimizer and scheduler
    optimizer = pytorch_system.create_pytorch_optimizer(model, "adamw", lr=1e-4)
    scheduler = pytorch_system.create_pytorch_scheduler(optimizer, "cosine", T_max=100)
    
    # Create sample data
    batch_size = 16
    seq_len = 100
    vocab_size = 30000
    
    sample_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    sample_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create dataset and dataloader
    dataset = TensorDataset(sample_data, sample_targets)
    dataloader = pytorch_system.create_optimized_dataloader(dataset, shuffle=True)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting PyTorch training...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # Training step
            step_result = pytorch_system.train_step_pytorch(
                model=model,
                data=data,
                targets=targets,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip_norm=1.0,
                global_step=epoch * len(dataloader) + batch_idx
            )
            
            total_loss += step_result['loss']
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            pytorch_system.save_pytorch_checkpoint(
                model, optimizer, scheduler, epoch + 1, avg_loss,
                f"pytorch_checkpoint_epoch_{epoch + 1}.pth"
            )
    
    # Profile model
    print("Profiling PyTorch model...")
    profile_result = pytorch_system.profile_pytorch_model(
        model, (1, 100), num_runs=50
    )
    
    # Export model
    print("Exporting PyTorch model...")
    export_path = pytorch_system.export_pytorch_model(
        model, (1, 100), "torchscript", "pytorch_transformer"
    )
    
    # Get memory info
    memory_info = pytorch_system.get_pytorch_memory_info()
    print(f"Memory Info: {memory_info}")
    
    # Cleanup
    pytorch_system.cleanup()
    
    print("=== PyTorch Framework Example completed successfully ===")

def main():
    """Main function demonstrating PyTorch primary framework"""
    create_pytorch_framework_example()

if __name__ == "__main__":
    main()


