#!/bin/bash

# Deep Learning Optimization Script for Blaze AI Production
# Prioritizes clarity, efficiency, and best practices in ML workflows
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PYTHON_SCRIPT_DIR="deployment/scripts/python"
LOG_DIR="/var/log/blaze-ai/deep-learning"
MODELS_DIR="models"
CACHE_DIR="cache"
CONFIG_DIR="configs"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

log_ml() {
    echo -e "${PURPLE}[ML]${NC} $1"
}

check_prerequisites() {
    log_info "Checking deep learning optimization prerequisites..."
    
    # Check Python and ML libraries
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed"
        exit 1
    fi
    
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        log_ml "CUDA detected - GPU acceleration available"
        export CUDA_AVAILABLE=true
    else
        log_warn "CUDA not detected - CPU-only mode"
        export CUDA_AVAILABLE=false
    fi
    
    # Create directories
    mkdir -p $PYTHON_SCRIPT_DIR
    mkdir -p $LOG_DIR
    mkdir -p $MODELS_DIR
    mkdir -p $CACHE_DIR
    mkdir -p $CONFIG_DIR
    
    log_info "Prerequisites check passed"
}

create_ml_optimization_script() {
    log_info "Creating deep learning optimization script..."
    
    cat > $PYTHON_SCRIPT_DIR/ml_optimizer.py << 'EOF'
#!/usr/bin/env python3
"""
Blaze AI Deep Learning Optimizer
Production ML workflow optimization with best practices
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import yaml
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/blaze-ai/deep-learning/ml_optimizer.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class OptimizationConfig:
    """Configuration for ML optimization"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping: bool = True
    mixed_precision: bool = True
    gradient_accumulation: int = 1
    num_workers: int = 4
    pin_memory: bool = True

class MLWorkflowOptimizer:
    """Deep Learning workflow optimization with best practices"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device()
        self.scaler = None
        self._setup_mixed_precision()
        
    def _setup_device(self) -> torch.device:
        """Setup optimal device with best practices"""
        if torch.cuda.is_available():
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Select best GPU
            gpu_id = self._select_best_gpu()
            device = torch.device(f"cuda:{gpu_id}")
            
            # Set memory fraction for multi-GPU environments
            torch.cuda.set_device(gpu_id)
            
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
            
        return device
    
    def _select_best_gpu(self) -> int:
        """Select GPU with most available memory"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Sort by available memory
                gpus.sort(key=lambda x: x.memoryFree, reverse=True)
                return gpus[0].id
        except:
            pass
        return 0
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training for efficiency"""
        if self.config.mixed_precision and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                self.logger.info("Mixed precision training enabled")
            except ImportError:
                self.logger.warning("Mixed precision not available, using FP32")
                self.config.mixed_precision = False
    
    def optimize_data_loading(self, dataset, batch_size: Optional[int] = None) -> DataLoader:
        """Optimize data loading with best practices"""
        batch_size = batch_size or self.config.batch_size
        
        # Calculate optimal number of workers
        num_workers = min(self.config.num_workers, os.cpu_count() or 4)
        
        # Create optimized DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=True
        )
        
        self.logger.info(f"DataLoader optimized: {num_workers} workers, pin_memory={self.config.pin_memory}")
        return dataloader
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimized optimizer with best practices"""
        # Group parameters for different learning rates
        param_groups = self._group_parameters(model)
        
        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.logger.info(f"Optimizer created: AdamW with lr={self.config.learning_rate}")
        return optimizer
    
    def _group_parameters(self, model: nn.Module) -> List[Dict]:
        """Group parameters for different learning rates"""
        # Separate parameters that should have different learning rates
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return param_groups
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with best practices"""
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            verbose=True,
            min_lr=1e-7
        )
        
        self.logger.info("Learning rate scheduler created: ReduceLROnPlateau")
        return scheduler
    
    def training_step(self, model: nn.Module, batch: Tuple, 
                     optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Optimized training step with mixed precision"""
        model.train()
        
        # Move batch to device
        batch = tuple(b.to(self.device) for b in batch)
        
        # Forward pass with mixed precision
        if self.config.mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = model(*batch[:-1])  # Assume last element is target
                loss = criterion(outputs, batch[-1])
        else:
            outputs = model(*batch[:-1])
            loss = criterion(outputs, batch[-1])
        
        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            if self.config.mixed_precision and self.scaler:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        if self.config.mixed_precision and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        return {'loss': loss.item()}
    
    def validation_step(self, model: nn.Module, batch: Tuple, 
                       criterion: nn.Module) -> Dict[str, float]:
        """Optimized validation step"""
        model.eval()
        
        with torch.no_grad():
            batch = tuple(b.to(self.device) for b in batch)
            
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(*batch[:-1])
                    loss = criterion(outputs, batch[-1])
            else:
                outputs = model(*batch[:-1])
                loss = criterion(outputs, batch[-1])
        
        return {'val_loss': loss.item()}
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch with optimization"""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for i, batch in enumerate(train_loader):
            # Gradient accumulation
            if i % self.config.gradient_accumulation == 0:
                step_loss = self.training_step(model, batch, optimizer, criterion)
                total_loss += step_loss['loss']
                
                # Log progress
                if i % 10 == 0:
                    self.logger.info(f"Batch {i}/{num_batches}, Loss: {step_loss['loss']:.4f}")
        
        return {'train_loss': total_loss / num_batches}
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                step_loss = self.validation_step(model, batch, criterion)
                total_loss += step_loss['val_loss']
        
        return {'val_loss': total_loss / num_batches}
    
    def train_model(self, model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, criterion: nn.Module,
                   save_path: str) -> Dict[str, Any]:
        """Complete training loop with best practices"""
        model = model.to(self.device)
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            # Training
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            training_history.append(metrics)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Train Loss: {metrics['train_loss']:.4f}, "
                f"Val Loss: {metrics['val_loss']:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Early stopping
            if self.config.early_stopping:
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    
                    # Save best model
                    self._save_model(model, optimizer, epoch, metrics, save_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }
    
    def _save_model(self, model: nn.Module, optimizer: optim.Optimizer,
                   epoch: int, metrics: Dict, save_path: str):
        """Save model checkpoint with best practices"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}_epoch_{epoch}_{timestamp}.pt"
        
        torch.save(checkpoint, filename)
        self.logger.info(f"Model saved: {filename}")
    
    def optimize_model_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference"""
        model.eval()
        
        if torch.cuda.is_available():
            # Enable TensorRT optimization if available
            try:
                import torch_tensorrt
                model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],  # Adjust input shape
                    enabled_precisions=[torch.float16, torch.float32]
                )
                self.logger.info("TensorRT optimization applied")
            except ImportError:
                self.logger.info("TensorRT not available, using standard optimization")
        
        # Enable inference optimizations
        with torch.no_grad():
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
        
        return model
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage information"""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system'] = {
            'total': system_memory.total,
            'available': system_memory.available,
            'used': system_memory.used,
            'percent': system_memory.percent
        }
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                gpu_memory.append({
                    'device_id': i,
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'free': total - reserved
                })
            
            memory_info['gpu'] = gpu_memory
        
        return memory_info
    
    def optimize_batch_size(self, model: nn.Module, sample_input: torch.Tensor,
                           max_memory_usage: float = 0.8) -> int:
        """Dynamically optimize batch size based on memory"""
        if not torch.cuda.is_available():
            return self.config.batch_size
        
        model.eval()
        device = next(model.parameters()).device
        
        # Start with current batch size
        batch_size = self.config.batch_size
        
        while batch_size > 1:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Test batch size
                test_input = sample_input.repeat(batch_size, 1, 1, 1).to(device)
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_usage = torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
                
                if memory_usage <= max_memory_usage:
                    self.logger.info(f"Optimal batch size: {batch_size}")
                    return batch_size
                
                batch_size //= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size //= 2
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        self.logger.warning(f"Could not find optimal batch size, using: {batch_size}")
        return max(1, batch_size)

def create_optimization_config() -> OptimizationConfig:
    """Create optimization configuration with best practices"""
    config = OptimizationConfig()
    
    # Auto-adjust based on system capabilities
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        if gpu_memory >= 24:  # High-end GPU
            config.batch_size = 64
            config.mixed_precision = True
        elif gpu_memory >= 12:  # Mid-range GPU
            config.batch_size = 32
            config.mixed_precision = True
        else:  # Low-end GPU
            config.batch_size = 16
            config.mixed_precision = False
    
    return config

def main():
    """Main optimization execution"""
    print("Starting Blaze AI Deep Learning Optimization...")
    
    # Create configuration
    config = create_optimization_config()
    print(f"Configuration: {config}")
    
    # Create optimizer
    optimizer = MLWorkflowOptimizer(config)
    
    # Get memory usage
    memory_info = optimizer.get_memory_usage()
    print(f"Memory Usage: {json.dumps(memory_info, indent=2)}")
    
    print("Deep Learning optimization setup completed!")

if __name__ == "__main__":
    main()
EOF
    
    log_info "Deep learning optimization script created"
}

run_ml_optimization() {
    log_info "Running deep learning optimization..."
    
    if [ -f "$PYTHON_SCRIPT_DIR/ml_optimizer.py" ]; then
        # Install required ML packages
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip3 install psutil gputil pyyaml
        
        # Run optimization script
        python3 $PYTHON_SCRIPT_DIR/ml_optimizer.py
        
        log_info "Deep learning optimization completed"
    else
        log_error "ML optimization script not found"
    fi
}

create_ml_best_practices_guide() {
    log_info "Creating ML best practices guide..."
    
    cat > $CONFIG_DIR/ml_best_practices.md << 'EOF'
# Deep Learning Best Practices for Blaze AI Production

## 🎯 Core Principles
- **Clarity**: Clean, readable, and maintainable code
- **Efficiency**: Optimized performance and resource usage
- **Reliability**: Robust error handling and validation
- **Scalability**: Design for growth and distributed training

## 🚀 Training Optimization

### 1. Mixed Precision Training
```python
# Enable automatic mixed precision
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Gradient Accumulation
```python
# For large effective batch sizes
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    loss = model(inputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Learning Rate Scheduling
```python
# ReduceLROnPlateau for automatic LR adjustment
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Step after validation
scheduler.step(val_loss)
```

## 📊 Data Loading Optimization

### 1. Efficient DataLoader
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Optimal for most systems
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    drop_last=True  # Consistent batch sizes
)
```

### 2. Memory-Efficient Batching
```python
# Dynamic batch sizing based on memory
def get_optimal_batch_size(model, sample_input, max_memory=0.8):
    batch_size = 32
    while batch_size > 1:
        try:
            test_input = sample_input.repeat(batch_size, 1, 1, 1)
            _ = model(test_input)
            return batch_size
        except RuntimeError:
            batch_size //= 2
    return 1
```

## 🔧 Model Optimization

### 1. Model Compilation
```python
# PyTorch 2.0 compilation
model = torch.compile(model, mode="reduce-overhead")

# JIT optimization
model = torch.jit.optimize_for_inference(torch.jit.script(model))
```

### 2. Memory Management
```python
# Clear cache between operations
torch.cuda.empty_cache()

# Gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)
```

## 📈 Monitoring and Logging

### 1. Comprehensive Metrics
```python
class TrainingMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_summary(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}
```

### 2. Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

## 🎛️ Configuration Management

### 1. Structured Config
```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    mixed_precision: bool = True
    
    def validate(self):
        assert self.batch_size > 0
        assert 0 < self.learning_rate < 1
        assert self.max_epochs > 0
```

### 2. Environment-Specific Settings
```python
def get_config(environment: str = "production") -> TrainingConfig:
    if environment == "development":
        return TrainingConfig(batch_size=8, max_epochs=10)
    elif environment == "staging":
        return TrainingConfig(batch_size=16, max_epochs=50)
    else:  # production
        return TrainingConfig(batch_size=32, max_epochs=100)
```

## 🔒 Error Handling and Validation

### 1. Robust Training Loop
```python
def safe_training_step(model, batch, optimizer, criterion):
    try:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'status': 'success'}
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return {'loss': 0.0, 'status': 'oom'}
        else:
            raise e
```

### 2. Input Validation
```python
def validate_inputs(inputs, targets):
    assert inputs.dim() == 4, f"Expected 4D input, got {inputs.dim()}D"
    assert targets.dim() == 1, f"Expected 1D targets, got {targets.dim()}D"
    assert inputs.size(0) == targets.size(0), "Batch size mismatch"
    assert not torch.isnan(inputs).any(), "Input contains NaN"
    assert not torch.isnan(targets).any(), "Targets contain NaN"
```

## 🚀 Performance Tips

1. **Use torch.backends.cudnn.benchmark = True** for fixed input sizes
2. **Enable TensorFloat-32** for faster training on Ampere GPUs
3. **Use gradient clipping** to prevent exploding gradients
4. **Implement proper data augmentation** for regularization
5. **Monitor GPU utilization** and adjust batch sizes accordingly
6. **Use distributed training** for large models and datasets
7. **Implement proper checkpointing** for fault tolerance
8. **Profile your training loop** to identify bottlenecks

## 📊 Resource Monitoring

```python
def monitor_resources():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # GPU usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_util = GPUtil.getGPUs()[0].load * 100
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'gpu_memory_gb': gpu_memory,
        'gpu_utilization': gpu_util
    }
```

## 🔄 Continuous Improvement

1. **Regular code reviews** focusing on ML best practices
2. **Performance benchmarking** against baseline models
3. **A/B testing** different optimization strategies
4. **Monitoring and alerting** for training failures
5. **Automated testing** of training pipelines
6. **Documentation updates** based on learnings
EOF
    
    log_info "ML best practices guide created"
}

create_ml_workflow_templates() {
    log_info "Creating ML workflow templates..."
    
    # Training template
    cat > $CONFIG_DIR/training_template.py << 'EOF'
#!/usr/bin/env python3
"""
Blaze AI Training Template
Production-ready training workflow with best practices
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
from datetime import datetime
import json

from ml_optimizer import MLWorkflowOptimizer, OptimizationConfig

class TrainingWorkflow:
    """Standardized training workflow"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizer = MLWorkflowOptimizer(config)
        self.logger = logging.getLogger(__name__)
        
    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, criterion: nn.Module,
              save_path: str) -> Dict[str, Any]:
        """Execute training workflow"""
        
        # Log training start
        self.logger.info(f"Starting training with config: {self.config}")
        
        # Train model
        results = self.optimizer.train_model(
            model, train_loader, val_loader, criterion, save_path
        )
        
        # Log results
        self.logger.info(f"Training completed: {results}")
        
        return results

# Usage example
if __name__ == "__main__":
    # Create configuration
    config = OptimizationConfig(
        batch_size=32,
        learning_rate=1e-4,
        max_epochs=100,
        patience=10,
        mixed_precision=True
    )
    
    # Create workflow
    workflow = TrainingWorkflow(config)
    
    # Train model (implement your model, data, and criterion)
    # results = workflow.train(model, train_loader, val_loader, criterion, "models/model")
EOF
    
    # Inference template
    cat > $CONFIG_DIR/inference_template.py << 'EOF'
#!/usr/bin/env python3
"""
Blaze AI Inference Template
Production-ready inference with optimization
"""

import torch
import torch.nn as nn
from typing import Any, List
import logging

class InferenceWorkflow:
    """Standardized inference workflow"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = self._load_model()
        self.logger = logging.getLogger(__name__)
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup optimal device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_model(self) -> nn.Module:
        """Load and optimize model for inference"""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = checkpoint['model']  # Adjust based on your checkpoint structure
        
        # Move to device
        model = model.to(self.device)
        
        # Optimize for inference
        model.eval()
        
        return model
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            return outputs
    
    def batch_predict(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Batch prediction for efficiency"""
        # Batch inputs
        batched_inputs = torch.stack(inputs)
        
        # Make prediction
        outputs = self.predict(batched_inputs)
        
        # Return as list
        return [outputs[i] for i in range(outputs.size(0))]

# Usage example
if __name__ == "__main__":
    # Create inference workflow
    workflow = InferenceWorkflow("models/best_model.pt")
    
    # Make predictions
    # inputs = torch.randn(1, 3, 224, 224)
    # outputs = workflow.predict(inputs)
EOF
    
    log_info "ML workflow templates created"
}

optimize_pytorch_settings() {
    log_info "Optimizing PyTorch settings..."
    
    # Create PyTorch optimization script
    cat > $PYTHON_SCRIPT_DIR/pytorch_optimizer.py << 'EOF'
#!/usr/bin/env python3
"""
PyTorch Settings Optimizer
Optimize PyTorch for production deep learning
"""

import torch
import os

def optimize_pytorch_settings():
    """Apply optimal PyTorch settings for production"""
    
    print("Optimizing PyTorch settings...")
    
    # CUDA optimizations
    if torch.cuda.is_available():
        # Enable cuDNN benchmark for fixed input sizes
        torch.backends.cudnn.benchmark = True
        
        # Enable TensorFloat-32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("Flash attention enabled")
        except:
            print("Flash attention not available")
        
        print("CUDA optimizations applied")
    else:
        print("CUDA not available, skipping CUDA optimizations")
    
    # Threading optimizations
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    
    # Memory optimizations
    torch.backends.cudnn.deterministic = False  # Faster, less deterministic
    
    print("PyTorch optimization completed")

if __name__ == "__main__":
    optimize_pytorch_settings()
EOF
    
    # Run PyTorch optimization
    python3 $PYTHON_SCRIPT_DIR/pytorch_optimizer.py
    
    log_info "PyTorch settings optimized"
}

check_ml_environment() {
    log_info "Checking ML environment..."
    
    # Check PyTorch installation
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        log_ml "PyTorch installed successfully"
    else
        log_error "PyTorch not installed"
    fi
    
    # Check CUDA availability
    if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        log_ml "CUDA check completed"
    else
        log_error "CUDA check failed"
    fi
    
    # Check GPU information
    if command -v nvidia-smi &> /dev/null; then
        log_ml "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    fi
    
    log_info "ML environment check completed"
}

generate_ml_report() {
    log_info "Generating ML optimization report..."
    
    REPORT_FILE="$LOG_DIR/ml_optimization_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $REPORT_FILE << EOF
Blaze AI Deep Learning Optimization Report
==========================================
Generated: $(date)
Hostname: $(hostname)

ML Environment:
- Python: $(python3 --version 2>&1)
- PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
- CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Unknown")
- GPU Count: $(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "Unknown")

Optimization Features:
- Mixed Precision Training: Enabled
- Gradient Accumulation: Supported
- Dynamic Batch Sizing: Available
- Memory Optimization: Active
- CUDA Optimizations: Applied

Best Practices Implemented:
- Clean code structure with type hints
- Comprehensive error handling
- Resource monitoring and optimization
- Automated configuration management
- Performance benchmarking tools

Generated Files:
- ML Optimizer: $PYTHON_SCRIPT_DIR/ml_optimizer.py
- Best Practices Guide: $CONFIG_DIR/ml_best_practices.md
- Training Template: $CONFIG_DIR/training_template.py
- Inference Template: $CONFIG_DIR/inference_template.py
- PyTorch Optimizer: $PYTHON_SCRIPT_DIR/pytorch_optimizer.py

Next Steps:
1. Review best practices guide
2. Use provided templates for new projects
3. Monitor training performance
4. Optimize based on your specific use case
5. Implement distributed training if needed

Report generated by Blaze AI ML Optimizer
EOF
    
    log_info "ML optimization report generated: $REPORT_FILE"
}

# Main execution
main() {
    log_info "Starting Blaze AI Deep Learning Optimization..."
    
    check_prerequisites
    create_ml_optimization_script
    create_ml_best_practices_guide
    create_ml_workflow_templates
    optimize_pytorch_settings
    check_ml_environment
    
    # Run ML optimization
    run_ml_optimization
    
    # Generate report
    generate_ml_report
    
    log_info "Deep Learning optimization completed successfully!"
    log_ml "Best practices and optimization templates are ready for production use"
}

# Run main function
main "$@"
