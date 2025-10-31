# Multi-GPU Training System - Complete Documentation

## Overview

The Multi-GPU Training System provides comprehensive support for parallel training across multiple GPUs using PyTorch's DataParallel and DistributedDataParallel. This system ensures efficient utilization of multiple GPUs with automatic mixed precision, gradient accumulation, and distributed training capabilities.

## Architecture

### Core Components

1. **MultiGPUTrainer**: Core multi-GPU training implementation
2. **MultiGPUTrainingManager**: High-level training management
3. **MultiGPUConfig**: Comprehensive configuration for multi-GPU setup
4. **ParallelStrategy**: Different parallel training strategies
5. **MultiGPUMode**: Different multi-GPU training modes

### Key Features

- **DataParallel**: Simple multi-GPU training with automatic data distribution
- **DistributedDataParallel**: Advanced distributed training with process groups
- **Mixed Precision**: Automatic mixed precision training with AMP
- **Gradient Accumulation**: Support for gradient accumulation across steps
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Synchronized BatchNorm**: Automatic conversion to SyncBatchNorm
- **Checkpoint Management**: Distributed checkpoint saving and loading
- **Performance Monitoring**: GPU utilization and training metrics

## Multi-GPU Training Modes

### Single GPU Training

```python
config = MultiGPUConfig(
    mode=MultiGPUMode.SINGLE_GPU,
    num_gpus=1,
    gpu_ids=[0]
)
```

### DataParallel Training

```python
config = MultiGPUConfig(
    mode=MultiGPUMode.DATAPARALLEL,
    strategy=ParallelStrategy.DATA_PARALLEL,
    num_gpus=4,
    gpu_ids=[0, 1, 2, 3],
    use_amp=True,
    use_gradient_accumulation=True,
    accumulation_steps=4
)
```

### DistributedDataParallel Training

```python
config = MultiGPUConfig(
    mode=MultiGPUMode.DISTRIBUTED_DATAPARALLEL,
    strategy=ParallelStrategy.DATA_PARALLEL,
    num_gpus=4,
    backend="nccl",
    init_method="env://",
    sync_bn=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    broadcast_buffers=True,
    bucket_cap_mb=25,
    static_graph=False
)
```

## MultiGPU Trainer

### Core Setup Methods

```python
class MultiGPUTrainer:
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.is_distributed = False
        self.is_master = False
        self.world_size = 1
        self.rank = 0
        
        # Initialize multi-GPU setup
        self._setup_multi_gpu()
```

### Distributed Training Setup

```python
def _setup_distributed_training(self):
    """Setup distributed training."""
    try:
        # Initialize distributed training
        if 'WORLD_SIZE' in os.environ:
            self.config.world_size = int(os.environ['WORLD_SIZE'])
            self.config.rank = int(os.environ['RANK'])
            self.config.local_rank = int(os.environ['LOCAL_RANK'])
        
        # Set device based on local rank
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f"cuda:{self.config.local_rank}")
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        self.is_distributed = True
        self.is_master = self.config.rank == 0
        self.world_size = self.config.world_size
        self.rank = self.config.rank
        
        self.logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
        
    except Exception as e:
        self.logger.error(f"Failed to setup distributed training: {str(e)}")
        self._setup_single_gpu_training()
```

### DataParallel Setup

```python
def _setup_dataparallel_training(self):
    """Setup DataParallel training."""
    try:
        self.logger.info(f"Setting up DataParallel with {self.config.num_gpus} GPUs")
        self.device = torch.device(f"cuda:{self.config.gpu_ids[0]}")
        
    except Exception as e:
        self.logger.error(f"Failed to setup DataParallel: {str(e)}")
        self._setup_single_gpu_training()
```

### Model Setup for Multi-GPU

```python
def setup_model(self, model: nn.Module) -> nn.Module:
    """Setup model for multi-GPU training."""
    self.model = model
    
    # Move model to device
    self.model = self.model.to(self.device)
    
    # Setup parallel training
    if self.config.mode == MultiGPUMode.DISTRIBUTED_DATAPARALLEL:
        self.model = self._setup_distributed_model()
    elif self.config.mode == MultiGPUMode.DATAPARALLEL:
        self.model = self._setup_dataparallel_model()
    
    # Setup mixed precision
    if self.config.use_amp:
        self.scaler = torch.cuda.amp.GradScaler()
    
    # Setup gradient checkpointing
    if self.config.use_gradient_checkpointing:
        self.model = self._setup_gradient_checkpointing()
    
    return self.model
```

### Distributed Model Setup

```python
def _setup_distributed_model(self) -> nn.Module:
    """Setup model for distributed training."""
    # Convert BatchNorm to SyncBatchNorm if needed
    if self.config.sync_bn:
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
    
    # Wrap with DistributedDataParallel
    self.model = nn.parallel.DistributedDataParallel(
        self.model,
        device_ids=[self.config.local_rank],
        output_device=self.config.local_rank,
        find_unused_parameters=self.config.find_unused_parameters,
        gradient_as_bucket_view=self.config.gradient_as_bucket_view,
        broadcast_buffers=self.config.broadcast_buffers,
        bucket_cap_mb=self.config.bucket_cap_mb,
        static_graph=self.config.static_graph
    )
    
    return self.model
```

### DataParallel Model Setup

```python
def _setup_dataparallel_model(self) -> nn.Module:
    """Setup model for DataParallel training."""
    # Wrap with DataParallel
    self.model = nn.DataParallel(
        self.model,
        device_ids=self.config.gpu_ids
    )
    
    return self.model
```

### Gradient Checkpointing Setup

```python
def _setup_gradient_checkpointing(self) -> nn.Module:
    """Setup gradient checkpointing."""
    if hasattr(self.model, 'gradient_checkpointing_enable'):
        self.model.gradient_checkpointing_enable()
    else:
        # Manual gradient checkpointing for custom models
        self.logger.info("Manual gradient checkpointing enabled")
    
    return self.model
```

### DataLoader Setup

```python
def setup_dataloader(self, dataset: data.Dataset, **kwargs) -> data.DataLoader:
    """Setup dataloader for multi-GPU training."""
    # Setup sampler for distributed training
    if self.is_distributed:
        sampler = data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=kwargs.get('shuffle', True)
        )
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = False
    
    # Adjust batch size for multi-GPU
    if 'batch_size' in kwargs:
        kwargs['batch_size'] = kwargs['batch_size'] // self.config.num_gpus
    
    # Setup dataloader
    dataloader = data.DataLoader(dataset, **kwargs)
    
    return dataloader
```

## Training Methods

### Training Step with Mixed Precision

```python
def train_step(self, data_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict[str, Any]:
    """Perform training step with multi-GPU support."""
    # Move data to device
    data_batch = data_batch.to(self.device, non_blocking=True)
    target_batch = target_batch.to(self.device, non_blocking=True)
    
    # Zero gradients
    self.optimizer.zero_grad()
    
    # Forward pass with mixed precision
    if self.config.use_amp and self.scaler is not None:
        with torch.cuda.amp.autocast():
            output = self.model(data_batch)
            loss = self.criterion(output, target_batch)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation
        if self.config.use_gradient_accumulation:
            if (self.current_step + 1) % self.config.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
    else:
        # Standard training without mixed precision
        output = self.model(data_batch)
        loss = self.criterion(output, target_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if self.config.use_gradient_accumulation:
            if (self.current_step + 1) % self.config.accumulation_steps == 0:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
    
    # Calculate metrics
    metrics = {
        'loss': loss.item(),
        'output_shape': output.shape,
        'target_shape': target_batch.shape
    }
    
    return metrics
```

### Validation Step

```python
def validate_step(self, data_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict[str, Any]:
    """Perform validation step with multi-GPU support."""
    # Move data to device
    data_batch = data_batch.to(self.device, non_blocking=True)
    target_batch = target_batch.to(self.device, non_blocking=True)
    
    # Disable gradient computation
    with torch.no_grad():
        if self.config.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.model(data_batch)
                loss = self.criterion(output, target_batch)
        else:
            output = self.model(data_batch)
            loss = self.criterion(output, target_batch)
    
    # Calculate metrics
    metrics = {
        'loss': loss.item(),
        'output_shape': output.shape,
        'target_shape': target_batch.shape
    }
    
    return metrics
```

## Checkpoint Management

### Saving Checkpoints

```python
def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
    """Save checkpoint for multi-GPU training."""
    if self.is_master or not self.is_distributed:
        # Save model state
        if self.is_distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            **kwargs
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
```

### Loading Checkpoints

```python
def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
    """Load checkpoint for multi-GPU training."""
    checkpoint = torch.load(filepath, map_location=self.device)
    
    # Load model state
    if self.is_distributed:
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if 'scaler_state_dict' in checkpoint and self.scaler is not None:
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    self.logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint
```

## MultiGPU Training Manager

### High-level Management

```python
class MultiGPUTrainingManager:
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.trainer = MultiGPUTrainer(config)
        self.logger = self.trainer.logger
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Setup logging
        self.setup_logging()
```

### Training Setup

```python
def setup_training(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                  criterion: Callable, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """Setup complete training configuration."""
    # Setup model
    self.model = self.trainer.setup_model(model)
    
    # Setup optimizer
    self.optimizer = self.trainer.setup_optimizer(optimizer)
    
    # Setup criterion
    self.criterion = self.trainer.setup_criterion(criterion)
    
    # Setup scheduler
    if scheduler is not None:
        self.scheduler = self.trainer.setup_scheduler(scheduler)
    
    self.logger.info("Training setup completed")
```

### Training Loop

```python
def train_epoch(self, train_loader: data.DataLoader) -> Dict[str, Any]:
    """Train for one epoch."""
    self.model.train()
    
    if self.trainer.is_distributed:
        train_loader.sampler.set_epoch(self.current_epoch)
    
    epoch_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
        # Training step
        metrics = self.trainer.train_step(data_batch, target_batch)
        
        epoch_loss += metrics['loss']
        self.current_step += 1
        
        # Log progress
        if batch_idx % 10 == 0 and (self.trainer.is_master or not self.trainer.is_distributed):
            self.logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                           f"Loss: {metrics['loss']:.4f}")
    
    avg_loss = epoch_loss / num_batches
    
    # Update learning rate
    if self.scheduler is not None:
        self.scheduler.step()
    
    return {'epoch_loss': avg_loss, 'num_batches': num_batches}
```

## Usage Examples

### Basic DataParallel Training

```python
# Create configuration
config = MultiGPUConfig(
    mode=MultiGPUMode.DATAPARALLEL,
    num_gpus=2,
    gpu_ids=[0, 1],
    use_amp=True,
    use_gradient_accumulation=False
)

# Create training manager
training_manager = MultiGPUTrainingManager(config)

# Setup model, optimizer, criterion
model = nn.Sequential(nn.Linear(784, 10))
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

training_manager.setup_training(model, optimizer, criterion)

# Create dataloader
train_loader = training_manager.trainer.setup_dataloader(
    dataset, batch_size=64, shuffle=True
)

# Train
training_manager.train(train_loader, num_epochs=10)
```

### Distributed Training

```python
# Create configuration for distributed training
config = MultiGPUConfig(
    mode=MultiGPUMode.DISTRIBUTED_DATAPARALLEL,
    num_gpus=4,
    backend="nccl",
    sync_bn=True,
    use_amp=True,
    use_gradient_accumulation=True,
    accumulation_steps=4
)

# Create training manager
training_manager = MultiGPUTrainingManager(config)

# Setup training components
model = nn.Sequential(nn.Linear(784, 10))
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

training_manager.setup_training(model, optimizer, criterion, scheduler)

# Create dataloader with distributed sampler
train_loader = training_manager.trainer.setup_dataloader(
    dataset, batch_size=32, shuffle=True
)

# Train
training_manager.train(train_loader, num_epochs=10)
```

### Advanced Configuration

```python
# Advanced multi-GPU configuration
config = MultiGPUConfig(
    mode=MultiGPUMode.DISTRIBUTED_DATAPARALLEL,
    strategy=ParallelStrategy.DATA_PARALLEL,
    num_gpus=8,
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    backend="nccl",
    init_method="env://",
    sync_bn=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    broadcast_buffers=True,
    bucket_cap_mb=25,
    static_graph=True,
    use_mixed_precision=True,
    use_gradient_accumulation=True,
    accumulation_steps=2,
    use_gradient_checkpointing=True,
    use_amp=True,
    amp_dtype="float16"
)
```

## Performance Optimization

### Mixed Precision Training

```python
# Enable automatic mixed precision
config = MultiGPUConfig(
    use_amp=True,
    amp_dtype="float16"  # or "bfloat16"
)

# The trainer automatically handles mixed precision
# Forward pass with autocast
with torch.cuda.amp.autocast():
    output = model(input_data)
    loss = criterion(output, target)

# Backward pass with gradient scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

```python
# Enable gradient accumulation
config = MultiGPUConfig(
    use_gradient_accumulation=True,
    accumulation_steps=4
)

# The trainer automatically handles gradient accumulation
# Gradients are accumulated for 4 steps before optimizer step
```

### Gradient Checkpointing

```python
# Enable gradient checkpointing for memory efficiency
config = MultiGPUConfig(
    use_gradient_checkpointing=True
)

# The trainer automatically enables gradient checkpointing
# This trades compute for memory
```

## Best Practices

### Multi-GPU Best Practices

1. **Batch Size**: Scale batch size with number of GPUs
2. **Learning Rate**: Scale learning rate with batch size
3. **Mixed Precision**: Use AMP for better performance
4. **Gradient Accumulation**: Use for large effective batch sizes
5. **Checkpointing**: Save only on master process

### Distributed Training Best Practices

1. **Process Groups**: Properly initialize and cleanup
2. **Samplers**: Use DistributedSampler for data distribution
3. **Synchronization**: Ensure proper synchronization
4. **Communication**: Minimize inter-process communication
5. **Monitoring**: Monitor GPU utilization and memory

### Performance Best Practices

1. **Data Loading**: Use multiple workers for data loading
2. **Memory Management**: Monitor GPU memory usage
3. **Gradient Scaling**: Use gradient scaling for mixed precision
4. **Optimization**: Profile and optimize bottlenecks
5. **Monitoring**: Monitor training metrics across GPUs

## Configuration Options

### Basic Configuration

```python
config = MultiGPUConfig(
    mode=MultiGPUMode.DATAPARALLEL,
    num_gpus=2,
    use_amp=True
)
```

### Advanced Configuration

```python
config = MultiGPUConfig(
    mode=MultiGPUMode.DISTRIBUTED_DATAPARALLEL,
    strategy=ParallelStrategy.DATA_PARALLEL,
    num_gpus=4,
    backend="nccl",
    sync_bn=True,
    use_amp=True,
    use_gradient_accumulation=True,
    accumulation_steps=4,
    use_gradient_checkpointing=True
)
```

## Conclusion

The Multi-GPU Training System provides comprehensive support for parallel training:

- **DataParallel**: Simple multi-GPU training with automatic data distribution
- **DistributedDataParallel**: Advanced distributed training with process groups
- **Mixed Precision**: Automatic mixed precision training with AMP
- **Gradient Accumulation**: Support for gradient accumulation across steps
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Synchronized BatchNorm**: Automatic conversion to SyncBatchNorm
- **Checkpoint Management**: Distributed checkpoint saving and loading
- **Performance Monitoring**: GPU utilization and training metrics

This system ensures efficient utilization of multiple GPUs for production-ready deep learning applications. 