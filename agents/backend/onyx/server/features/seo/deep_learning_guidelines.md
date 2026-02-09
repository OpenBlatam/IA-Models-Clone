# Deep Learning Guidelines for SEO Service

## Core Principles

### 1. Clarity and Readability
- **Descriptive variable names**: Use `is_model_trained`, `has_valid_predictions`, `should_use_gpu`
- **Type hints**: Always include type annotations for function signatures
- **Documentation**: Comprehensive docstrings for all functions and classes
- **Modular design**: Separate concerns into distinct modules (data, models, training, inference)

### 2. Efficiency and Performance
- **GPU utilization**: Prioritize CUDA operations for compute-intensive tasks
- **Mixed precision**: Use `torch.cuda.amp` for faster training with minimal accuracy loss
- **Memory management**: Implement proper cleanup and memory monitoring
- **Batch processing**: Process data in optimal batch sizes for your hardware

### 3. Best Practices
- **Reproducibility**: Set random seeds and document environment
- **Error handling**: Comprehensive exception handling with meaningful messages
- **Logging**: Structured logging for monitoring and debugging
- **Testing**: Unit tests for all critical functions

## Code Structure

### Module Organization
```
deep_learning/
├── models/
│   ├── __init__.py
│   ├── seo_classifier.py
│   └── text_encoder.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── dataloader.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── optimizer.py
├── inference/
│   ├── __init__.py
│   └── pipeline.py
└── utils/
    ├── __init__.py
    ├── metrics.py
    └── visualization.py
```

### Function Design Patterns

#### RORO Pattern Implementation
```python
def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train model with RORO pattern"""
    # Input validation
    if not config.get('model_name'):
        raise ValueError("model_name is required")
    
    # Process
    model = create_model(config)
    metrics = train_loop(model, config)
    
    # Return structured output
    return {
        'model': model,
        'metrics': metrics,
        'config': config,
        'status': 'completed'
    }
```

#### Async Functions for I/O Operations
```python
async def load_dataset_async(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load dataset asynchronously"""
    dataset_path = config.get('dataset_path')
    if not dataset_path:
        raise ValueError("dataset_path is required")
    
    # Async file operations
    data = await load_data_from_file(dataset_path)
    return {'dataset': data, 'size': len(data)}
```

## Performance Optimization

### 1. Model Optimization
```python
# Enable mixed precision
scaler = GradScaler()

# Model compilation (PyTorch 2.0+)
model = torch.compile(model, mode='reduce-overhead')

# Quantization for inference
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 2. Data Loading Optimization
```python
# Optimized DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)
```

### 3. Memory Management
```python
# Monitor memory usage
def monitor_memory():
    if torch.cuda.is_available():
        return {
            'gpu_allocated': torch.cuda.memory_allocated(),
            'gpu_cached': torch.cuda.memory_reserved()
        }
    return {'cpu_ram': psutil.virtual_memory().used}

# Clear memory
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Training Best Practices

### 1. Training Loop Structure
```python
async def train_with_optimizations(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any]
) -> Dict[str, List[float]]:
    """Optimized training loop"""
    
    # Setup
    device = setup_device(config.get('device', 'auto'))
    optimizer, scheduler = create_optimizer_with_scheduler(model, config)
    scaler = GradScaler() if config.get('use_mixed_precision', True) else None
    
    # Training history
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    model.to(device)
    model.train()
    
    for epoch in range(config['num_epochs']):
        # Training phase
        train_metrics = await train_epoch(
            model, train_loader, optimizer, scaler, device
        )
        
        # Validation phase
        val_metrics = await validate_epoch(
            model, val_loader, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        for key in metrics:
            metrics[key].append(train_metrics[key] if 'train' in key else val_metrics[key])
        
        # Log progress
        log_epoch_results(epoch, train_metrics, val_metrics)
        
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return metrics
```

### 2. Early Stopping and Checkpointing
```python
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

## Inference Optimization

### 1. Batch Inference
```python
async def batch_inference(
    model: nn.Module,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """Efficient batch inference"""
    
    model.eval()
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # Format results
        for text, pred, prob in zip(batch_texts, predictions, probabilities):
            results.append({
                'text': text,
                'prediction': pred.item(),
                'confidence': prob.max().item(),
                'probabilities': prob.cpu().numpy().tolist()
            })
    
    return results
```

### 2. Model Serving
```python
class ModelServer:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(model_path)
        
        # Optimize for inference
        self.model.eval()
        self.model = torch.compile(self.model)
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """Single prediction"""
        return (await self.batch_predict([text]))[0]
    
    async def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction"""
        return await batch_inference(
            self.model, texts, self.tokenizer, self.device
        )
```

## Error Handling and Validation

### 1. Input Validation
```python
from pydantic import BaseModel, Field, validator

class TrainingConfig(BaseModel):
    model_name: str = Field(..., description="Model identifier")
    batch_size: int = Field(gt=0, le=128, description="Batch size")
    learning_rate: float = Field(gt=0, le=1, description="Learning rate")
    num_epochs: int = Field(gt=0, le=1000, description="Number of epochs")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("model_name cannot be empty")
        return v.strip()
```

### 2. Exception Handling
```python
class ModelTrainingError(Exception):
    """Custom exception for training errors"""
    pass

async def safe_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Training with comprehensive error handling"""
    try:
        # Validate config
        validated_config = TrainingConfig(**config)
        
        # Setup environment
        device = setup_device(validated_config.device)
        
        # Training
        metrics = await train_model(validated_config, device)
        
        return {
            'status': 'success',
            'metrics': metrics,
            'device_used': str(device)
        }
        
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ModelTrainingError(f"Invalid configuration: {e}")
        
    except RuntimeError as e:
        logger.error(f"Training runtime error: {e}")
        raise ModelTrainingError(f"Training failed: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise ModelTrainingError(f"Unexpected error: {e}")
```

## Monitoring and Logging

### 1. Structured Logging
```python
import structlog

logger = structlog.get_logger()

def log_training_metrics(epoch: int, metrics: Dict[str, float]):
    """Log training metrics in structured format"""
    logger.info(
        "Training metrics",
        epoch=epoch,
        train_loss=metrics['train_loss'],
        val_loss=metrics['val_loss'],
        train_accuracy=metrics['train_accuracy'],
        val_accuracy=metrics['val_accuracy'],
        learning_rate=metrics['learning_rate']
    )
```

### 2. Performance Monitoring
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} completed in {elapsed:.2f}s")

# Usage
with timer("model_training"):
    metrics = await train_model(config)
```

## Testing Guidelines

### 1. Unit Tests
```python
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

@pytest_asyncio.asyncio
async def test_model_training():
    """Test model training functionality"""
    config = {
        'model_name': 'bert-base-uncased',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 1
    }
    
    with patch('torch.cuda.is_available', return_value=False):
        result = await train_model(config)
        
        assert result['status'] == 'success'
        assert 'metrics' in result
        assert len(result['metrics']['train_losses']) == 1

@pytest.mark.parametrize("invalid_config", [
    {'model_name': ''},
    {'batch_size': -1},
    {'learning_rate': 2.0}
])
def test_config_validation(invalid_config):
    """Test configuration validation"""
    with pytest.raises(ValidationError):
        TrainingConfig(**invalid_config)
```

### 2. Integration Tests
```python
@pytest_asyncio.asyncio
async def test_end_to_end_pipeline():
    """Test complete training and inference pipeline"""
    # Setup
    config = create_test_config()
    
    # Training
    training_result = await train_model(config)
    assert training_result['status'] == 'success'
    
    # Inference
    model = training_result['model']
    predictions = await batch_inference(model, ["test text"], tokenizer, device)
    assert len(predictions) == 1
    assert 'prediction' in predictions[0]
```

## Deployment Considerations

### 1. Model Serialization
```python
def save_model_for_production(model: nn.Module, path: str):
    """Save model for production deployment"""
    # Save model state
    torch.save(model.state_dict(), f"{path}/model.pth")
    
    # Save model config
    config = {
        'model_type': model.__class__.__name__,
        'model_config': model.config if hasattr(model, 'config') else None,
        'version': '1.0.0',
        'created_at': datetime.now().isoformat()
    }
    
    with open(f"{path}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
```

### 2. Docker Configuration
```dockerfile
# Use PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
COPY requirements.deep_learning.txt .
RUN pip install -r requirements.deep_learning.txt

# Copy application
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Run application
CMD ["python", "deep_learning_workflow.py"]
```

## Performance Benchmarks

### 1. Training Performance
- **Throughput**: Measure samples/second during training
- **Memory usage**: Monitor GPU and CPU memory consumption
- **Convergence**: Track loss and accuracy over epochs

### 2. Inference Performance
- **Latency**: Measure time per prediction
- **Throughput**: Measure predictions/second
- **Memory efficiency**: Monitor memory usage during inference

### 3. Optimization Metrics
```python
def benchmark_model_performance(model: nn.Module, test_data: List[str]):
    """Benchmark model performance"""
    device = next(model.parameters()).device
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(torch.randn(1, 512).to(device))
    
    # Benchmark
    start_time = time.time()
    predictions = []
    
    for text in test_data:
        with timer("single_prediction"):
            pred = model(text)
            predictions.append(pred)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(test_data)
    
    return {
        'total_predictions': len(predictions),
        'total_time': total_time,
        'avg_time_per_prediction': avg_time,
        'predictions_per_second': len(predictions) / total_time
    }
```

## Security Considerations

### 1. Input Sanitization
```python
def sanitize_text_input(text: str) -> str:
    """Sanitize text input for model inference"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove potentially dangerous characters
    sanitized = text.strip()
    if len(sanitized) > 10000:  # Limit input size
        sanitized = sanitized[:10000]
    
    return sanitized
```

### 2. Model Security
```python
def validate_model_integrity(model_path: str) -> bool:
    """Validate model file integrity"""
    try:
        # Check file exists and is readable
        if not os.path.exists(model_path):
            return False
        
        # Load model and verify structure
        model = torch.load(model_path, map_location='cpu')
        required_keys = ['model_state_dict', 'config']
        
        return all(key in model for key in required_keys)
        
    except Exception:
        return False
```

## Conclusion

These guidelines ensure that deep learning components in the SEO service are:
- **Clear and maintainable**: Well-documented, type-hinted code
- **Efficient**: Optimized for performance and resource usage
- **Robust**: Comprehensive error handling and validation
- **Testable**: Proper unit and integration tests
- **Deployable**: Production-ready with monitoring and security

Follow these guidelines to create high-quality, production-ready deep learning solutions for the SEO service. 