# PyTorch Optimization Guide

A comprehensive guide for optimizing PyTorch performance in the Onyx Ads Backend profiling and optimization system.

## 🚀 Overview

This guide covers PyTorch-specific optimizations including:
- **Memory optimization** - Efficient tensor operations and memory management
- **Performance optimization** - CUDA optimization, mixed precision, and parallel processing
- **Model optimization** - Model compilation, quantization, and optimization techniques
- **Data loading optimization** - Efficient DataLoader configuration and custom datasets
- **Training optimization** - Gradient accumulation, distributed training, and optimization strategies

## 🔧 PyTorch Configuration

### Basic PyTorch Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Check PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Set default tensor type for optimization
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

### Memory Management

```python
# Memory optimization utilities
class TorchMemoryOptimizer:
    """PyTorch memory optimization utilities."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            return {
                'allocated_mb': allocated / 1024**2,
                'reserved_mb': reserved / 1024**2,
                'max_allocated_mb': max_allocated / 1024**2
            }
        return {}
    
    @staticmethod
    def optimize_tensor(tensor):
        """Optimize tensor for memory efficiency."""
        # Use appropriate dtype
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        elif tensor.dtype == torch.int64:
            tensor = tensor.int()
        
        # Use contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pin memory if on CPU and moving to GPU
        if tensor.device.type == 'cpu' and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        
        return tensor
    
    @staticmethod
    def gradient_checkpointing(model, enable=True):
        """Enable gradient checkpointing for memory efficiency."""
        if enable:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
    
    @staticmethod
    def mixed_precision_setup():
        """Setup mixed precision training."""
        scaler = torch.cuda.amp.GradScaler()
        return scaler
```

## ⚡ Performance Optimization

### CUDA Optimization

```python
class TorchPerformanceOptimizer:
    """PyTorch performance optimization utilities."""
    
    @staticmethod
    def optimize_cuda_settings():
        """Optimize CUDA settings for performance."""
        if torch.cuda.is_available():
            # Enable cuDNN benchmark for optimal performance
            torch.backends.cudnn.benchmark = True
            
            # Disable deterministic mode for better performance
            torch.backends.cudnn.deterministic = False
            
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.enabled = True
            
            # Set memory fraction (optional)
            # torch.cuda.set_per_process_memory_fraction(0.8)
    
    @staticmethod
    def compile_model(model, mode='default'):
        """Compile model for better performance (PyTorch 2.0+)."""
        try:
            if hasattr(torch, 'compile'):
                return torch.compile(model, mode=mode)
            else:
                print("torch.compile not available (requires PyTorch 2.0+)")
                return model
        except Exception as e:
            print(f"Model compilation failed: {e}")
            return model
    
    @staticmethod
    def optimize_data_loader(dataloader, num_workers=None, pin_memory=True):
        """Optimize DataLoader for better performance."""
        if num_workers is None:
            num_workers = min(4, os.cpu_count())
        
        dataloader.num_workers = num_workers
        dataloader.pin_memory = pin_memory and torch.cuda.is_available()
        dataloader.persistent_workers = num_workers > 0
        
        return dataloader
```

### Mixed Precision Training

```python
class TorchMixedPrecisionTrainer:
    """Mixed precision training utilities."""
    
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target, loss_fn):
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = loss_fn(output, target)
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def update_learning_rate(self, new_lr):
        """Update learning rate with scaling."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def get_scale(self):
        """Get current gradient scaler scale."""
        return self.scaler.get_scale()
```

## 🧠 Model Optimization

### Model Quantization

```python
class TorchModelOptimizer:
    """PyTorch model optimization utilities."""
    
    @staticmethod
    def quantize_model(model, quantization_type='dynamic'):
        """Quantize model for reduced memory usage and faster inference."""
        if quantization_type == 'dynamic':
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # Requires calibration data
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Calibrate with sample data
            # torch.quantization.convert(model, inplace=True)
            return model
        else:
            return model
    
    @staticmethod
    def optimize_model_for_inference(model):
        """Optimize model for inference."""
        model.eval()
        
        # Use TorchScript for optimization
        try:
            scripted_model = torch.jit.script(model)
            return scripted_model
        except Exception as e:
            print(f"TorchScript optimization failed: {e}")
            return model
    
    @staticmethod
    def fuse_model_layers(model):
        """Fuse model layers for better performance."""
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
        return model
```

### Custom Optimized Layers

```python
class OptimizedLinear(nn.Module):
    """Optimized linear layer with memory efficiency."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class OptimizedConv2d(nn.Module):
    """Optimized 2D convolution layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x):
        return self.conv(x)
```

## 📊 Data Loading Optimization

### Optimized Dataset

```python
class OptimizedTorchDataset(Dataset):
    """Optimized PyTorch dataset with memory efficiency."""
    
    def __init__(self, data, targets=None, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
        # Optimize data storage
        if isinstance(self.data, torch.Tensor):
            self.data = TorchMemoryOptimizer.optimize_tensor(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx] if self.targets is not None else None
        
        if self.transform is not None:
            data = self.transform(data)
        
        if target is not None and self.target_transform is not None:
            target = self.target_transform(target)
        
        return data, target

class StreamingTorchDataset(IterableDataset):
    """Streaming dataset for large datasets."""
    
    def __init__(self, data_source, batch_size=32, transform=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.transform = transform
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            return self._stream_data()
        else:
            # Multi-process data loading
            return self._stream_data_worker(worker_info)
    
    def _stream_data(self):
        # Implementation depends on data source
        pass
    
    def _stream_data_worker(self, worker_info):
        # Implementation for multi-worker streaming
        pass
```

### DataLoader Optimization

```python
class OptimizedDataLoader:
    """Optimized DataLoader configuration."""
    
    @staticmethod
    def create_optimized_loader(dataset, batch_size=32, shuffle=True, **kwargs):
        """Create an optimized DataLoader."""
        # Calculate optimal number of workers
        num_workers = min(4, os.cpu_count())
        
        # Determine pin memory based on CUDA availability
        pin_memory = torch.cuda.is_available()
        
        # Create DataLoader with optimized settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
            **kwargs
        )
        
        return dataloader
    
    @staticmethod
    def profile_dataloader(dataloader, num_batches=10):
        """Profile DataLoader performance."""
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            batch_start = time.time()
            _ = batch  # Consume batch
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if i >= num_batches:
                break
        
        total_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        return {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'throughput': len(batch_times) / total_time
        }
```

## 🎯 Training Optimization

### Optimized Training Loop

```python
class TorchOptimizedTrainer:
    """Optimized PyTorch training utilities."""
    
    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def train_epoch(self, dataloader, epoch=0):
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate model with optimizations."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path, epoch, loss):
        """Save optimized checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load optimized checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
```

### Gradient Accumulation

```python
class TorchGradientAccumulator:
    """Gradient accumulation for large effective batch sizes."""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    def train_step(self, data, target, criterion, step):
        """Training step with gradient accumulation."""
        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = criterion(output, target) / self.accumulation_steps
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
        else:
            output = self.model(data)
            loss = criterion(output, target) / self.accumulation_steps
            loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps
```

## 🔍 Profiling and Monitoring

### PyTorch Profiler Integration

```python
class TorchProfiler:
    """PyTorch-specific profiling utilities."""
    
    def __init__(self, config):
        self.config = config
        self.profiler = None
    
    def start_profiling(self):
        """Start PyTorch profiling."""
        if torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                use_cuda=True
            )
            self.profiler.start()
    
    def stop_profiling(self):
        """Stop PyTorch profiling and get results."""
        if self.profiler:
            self.profiler.stop()
            return self.profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            )
        return None
    
    def profile_model(self, model, sample_input):
        """Profile model performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Profile
        self.start_profiling()
        with torch.no_grad():
            for _ in range(100):
                _ = model(sample_input)
        results = self.stop_profiling()
        
        return results
    
    def get_memory_stats(self):
        """Get detailed memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved(),
                'memory_fragmentation': torch.cuda.memory_fragmentation()
            }
        return {}
```

## 🚀 Advanced Optimizations

### Custom CUDA Kernels

```python
# Example of custom CUDA kernel (requires CUDA toolkit)
try:
    import torch.utils.cpp_extension as cpp_extension
    
    # Compile custom CUDA kernel
    custom_kernel = cpp_extension.load(
        name="custom_kernel",
        sources=["custom_kernel.cu"],
        extra_cuda_cflags=["-O3"]
    )
except ImportError:
    print("CUDA toolkit not available for custom kernels")

# Custom kernel implementation would go here
```

### Model Parallelism

```python
class TorchModelParallel:
    """Model parallelism utilities."""
    
    @staticmethod
    def split_model_across_gpus(model, num_gpus):
        """Split model across multiple GPUs."""
        if num_gpus <= 1:
            return model
        
        # Split model layers across GPUs
        layers = list(model.children())
        layers_per_gpu = len(layers) // num_gpus
        
        for i, layer in enumerate(layers):
            gpu_id = i // layers_per_gpu
            if gpu_id < num_gpus:
                layer.to(f'cuda:{gpu_id}')
        
        return model
    
    @staticmethod
    def data_parallel_model(model, device_ids=None):
        """Wrap model in DataParallel."""
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) > 1:
            return nn.DataParallel(model, device_ids=device_ids)
        return model
```

## 📈 Performance Benchmarks

### Benchmarking Utilities

```python
class TorchBenchmarker:
    """PyTorch benchmarking utilities."""
    
    @staticmethod
    def benchmark_model(model, input_shape, num_runs=100, warmup_runs=10):
        """Benchmark model performance."""
        model.eval()
        device = next(model.parameters()).device
        
        # Create sample input
        sample_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = num_runs / (end_time - start_time)
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2
        }
    
    @staticmethod
    def benchmark_training(model, dataloader, num_epochs=1):
        """Benchmark training performance."""
        model.train()
        device = next(model.parameters()).device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return {
            'total_training_time': end_time - start_time,
            'avg_epoch_time': (end_time - start_time) / num_epochs,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2
        }
```

## 🔧 Configuration Examples

### Production Configuration

```python
# Production PyTorch configuration
def setup_production_config():
    """Setup PyTorch for production use."""
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set memory management
    torch.cuda.empty_cache()
    
    # Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    return scaler

# Training configuration
def create_optimized_training_config():
    """Create optimized training configuration."""
    config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'gradient_checkpointing': True,
        'model_compilation': True
    }
    return config
```

This comprehensive PyTorch optimization guide provides the tools and techniques needed to maximize performance and efficiency in the Onyx Ads Backend profiling and optimization system. 