# Frontier Model Training - Enhanced Documentation

## Overview

The Frontier Model Training system is a comprehensive framework for training large language models with advanced optimization techniques, monitoring, and deployment capabilities. This enhanced version includes improved configuration management, error handling, performance monitoring, testing, and deployment automation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Training](#training)
4. [Monitoring](#monitoring)
5. [Testing](#testing)
6. [Deployment](#deployment)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for cluster deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd frontier-model-run

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### Basic Usage

```bash
# Create default configuration
python config_manager.py --create-default config/default.yaml

# Run training with configuration
python run_training.py --config config/default.yaml

# Monitor training
python performance_monitor.py --log-dir ./logs
```

## Configuration

### Configuration Management

The system uses a hierarchical configuration system with environment-specific settings.

#### Creating Configuration

```python
from config_manager import ConfigManager, Environment

# Create default configuration
manager = ConfigManager()
manager.create_default_config("config/development.yaml", Environment.DEVELOPMENT)

# Load and modify configuration
config = manager.load_config("config/development.yaml")
config.training.batch_size = 16
config.training.learning_rate = 1e-4

# Save modified configuration
manager.save_config(config, "config/custom.yaml")
```

#### Configuration Structure

```yaml
# Dataset Configuration
dataset:
  name: "your_dataset"
  config: "your_config"
  train_split: "train"
  test_split: "test"

# Model Configuration
model:
  name: "deepseek-ai/deepseek-r1"
  use_deepspeed: false
  fp16: true
  bf16: false

# Training Parameters
training:
  batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  max_steps: 1000
  warmup_ratio: 0.1
  weight_decay: 0.01

# Optimization Settings
optimization:
  use_amp: true
  use_gradient_checkpointing: true
  use_flash_attention: true
  use_8bit_optimizer: false

# Performance Settings
performance:
  use_cudnn_benchmark: true
  use_tf32: true
  use_channels_last: true
  use_compile: true
```

### Environment-Specific Configurations

```bash
# Development environment
python config_manager.py --create-default config/dev.yaml --environment development

# Staging environment
python config_manager.py --create-default config/staging.yaml --environment staging

# Production environment
python config_manager.py --create-default config/prod.yaml --environment production
```

## Training

### Basic Training

```python
from run_training import main_with_config
from config_manager import ConfigManager

# Load configuration
manager = ConfigManager()
config = manager.load_config("config/development.yaml")

# Run training
main_with_config(config)
```

### Advanced Training with Monitoring

```python
from run_training import main_with_config
from config_manager import ConfigManager
from performance_monitor import create_metrics_collector
from error_handler import setup_logging

# Setup logging and monitoring
logger = setup_logging(log_dir="./logs", enable_sentry=True)
collector = create_metrics_collector(
    log_dir="./metrics",
    enable_tensorboard=True,
    enable_wandb=True,
    wandb_project="frontier-model"
)

# Start monitoring
collector.start_monitoring(interval=5.0)

# Load and run training
manager = ConfigManager()
config = manager.load_config("config/development.yaml")

try:
    main_with_config(config)
except Exception as e:
    logger.log_error(e, ErrorType.TRAINING, "main", "training")
finally:
    collector.stop_monitoring()
    collector.generate_report()
```

### Custom Training Loop

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from performance_monitor import MetricsCollector, TrainingMetrics
from error_handler import error_handler, ErrorType

@error_handler(ErrorType.TRAINING, "training", "custom_loop")
def custom_training_loop(model, dataloader, optimizer, scheduler, collector):
    model.train()
    
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Log metrics
        metrics = TrainingMetrics(
            step=step,
            epoch=0,
            training_loss=loss.item(),
            learning_rate=scheduler.get_last_lr()[0],
            batch_time=time.time() - batch_start_time
        )
        collector.log_training_metrics(metrics)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
```

## Monitoring

### Performance Monitoring

```python
from performance_monitor import create_metrics_collector, TrainingMetrics, SystemMetrics

# Create metrics collector
collector = create_metrics_collector(
    log_dir="./metrics",
    enable_tensorboard=True,
    enable_wandb=True,
    wandb_project="frontier-model"
)

# Start system monitoring
collector.start_monitoring(interval=5.0)

# Log training metrics
training_metrics = TrainingMetrics(
    step=100,
    epoch=1,
    training_loss=0.5,
    validation_loss=0.6,
    learning_rate=1e-4,
    batch_time=0.8,
    throughput=125.0
)
collector.log_training_metrics(training_metrics)

# Generate performance report
report_path = collector.generate_report()
print(f"Performance report generated: {report_path}")
```

### Error Handling and Logging

```python
from error_handler import setup_logging, error_context, ErrorType
from performance_monitor import create_metrics_collector

# Setup logging
logger = setup_logging(
    log_dir="./logs",
    log_level=LogLevel.INFO,
    enable_sentry=True,
    sentry_dsn="your-sentry-dsn"
)

# Use error context for operations
with error_context(ErrorType.MODEL_LOADING, "model", "load") as (logger, handler):
    model = load_model("path/to/model")
    logger.info("Model loaded successfully")

# Automatic error handling with decorator
@error_handler(ErrorType.TRAINING, "training", "step")
def training_step(model, batch):
    return model(**batch)
```

### Alerting

```python
from performance_monitor import MetricsCollector, AlertLevel

collector = create_metrics_collector()

# Custom alert thresholds
collector.alert_thresholds["memory_usage"]["warning"] = 70.0
collector.alert_thresholds["memory_usage"]["error"] = 85.0
collector.alert_thresholds["memory_usage"]["critical"] = 95.0

# Check alerts
for alert in collector.alerts:
    if alert.level == AlertLevel.CRITICAL:
        print(f"CRITICAL ALERT: {alert.message}")
        # Send notification, scale resources, etc.
```

## Testing

### Running Tests

```bash
# Generate test files
python test_framework.py --generate-tests

# Run all tests
python test_framework.py --run-tests --coverage --parallel

# Run specific test suite
python test_framework.py --run-tests --test-suite unit_tests

# Run with HTML report
python test_framework.py --run-tests --html-report
```

### Writing Tests

```python
import unittest
from unittest.mock import Mock, patch
from config_manager import ConfigManager, FrontierConfig
from error_handler import StructuredLogger, ErrorType

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        manager = ConfigManager()
        manager.create_default_config(str(self.config_file))
        
        self.assertTrue(self.config_file.exists())
    
    def test_load_config(self):
        """Test loading configuration from file."""
        # Create test config
        test_config = {
            'environment': 'development',
            'model': {'name': 'test-model'},
            'training': {'batch_size': 16}
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        manager = ConfigManager()
        config = manager.load_config(str(self.config_file))
        
        self.assertEqual(config.model.name, 'test-model')
        self.assertEqual(config.training.batch_size, 16)

if __name__ == '__main__':
    unittest.main()
```

### Performance Testing

```python
import time
import psutil
from performance_monitor import create_metrics_collector, TrainingMetrics

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.collector = create_metrics_collector()
    
    def test_training_performance(self):
        """Test training performance benchmarks."""
        start_time = time.time()
        
        # Simulate training
        for step in range(100):
            metrics = TrainingMetrics(
                step=step,
                epoch=0,
                training_loss=1.0 - step * 0.01,
                batch_time=0.1
            )
            self.collector.log_training_metrics(metrics)
        
        duration = time.time() - start_time
        
        # Assert performance requirements
        self.assertLess(duration, 10.0)  # Should complete in under 10 seconds
        self.assertEqual(len(self.collector.training_metrics), 100)
    
    def test_memory_usage(self):
        """Test memory usage during training."""
        initial_memory = psutil.virtual_memory().used
        
        # Simulate memory-intensive operation
        data = [torch.randn(1000, 1000) for _ in range(10)]
        
        final_memory = psutil.virtual_memory().used
        memory_increase = (final_memory - initial_memory) / 1024**2  # MB
        
        # Assert memory usage is reasonable
        self.assertLess(memory_increase, 1000)  # Less than 1GB increase
```

## Deployment

### Docker Deployment

```bash
# Create Docker configuration
python deployment_manager.py --create-docker

# Build Docker image
docker build -t frontier-model:latest .

# Run container
docker run -d \
  --name frontier-model \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e ENVIRONMENT=production \
  frontier-model:latest

# Use Docker Compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Create deployment configuration
cat > deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontier-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontier-model
  template:
    metadata:
      labels:
        app: frontier-model
    spec:
      containers:
      - name: frontier-model
        image: frontier-model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: ENVIRONMENT
          value: "production"
EOF

# Deploy to Kubernetes
kubectl apply -f deployment.yaml
```

### CI/CD Pipeline

```bash
# Create GitHub Actions workflow
python deployment_manager.py --create-ci --ci-type github

# Create GitLab CI configuration
python deployment_manager.py --create-ci --ci-type gitlab

# Create Jenkins pipeline
python deployment_manager.py --create-ci --ci-type jenkins
```

## API Reference

### ConfigManager

```python
class ConfigManager:
    def __init__(self, config_path: Optional[str] = None)
    def load_config(self, config_path: Optional[str] = None) -> FrontierConfig
    def save_config(self, config: FrontierConfig, output_path: str, format: str = 'yaml') -> None
    def create_default_config(self, output_path: str, environment: Environment = Environment.DEVELOPMENT) -> None
    def validate_config(self, config: FrontierConfig) -> List[str]
    def display_config(self, config: FrontierConfig) -> None
    def get_environment_config(self, environment: Environment) -> FrontierConfig
```

### StructuredLogger

```python
class StructuredLogger:
    def __init__(self, log_dir: str = "./logs", log_level: LogLevel = LogLevel.INFO, ...)
    def log_error(self, error: Exception, error_type: ErrorType = ErrorType.UNKNOWN, ...)
    def log_performance(self, metrics: PerformanceMetrics) -> None
    def start_performance_monitoring(self, interval: float = 10.0) -> None
    def stop_performance_monitoring(self) -> None
    def get_performance_summary(self) -> Dict[str, Any]
```

### MetricsCollector

```python
class MetricsCollector:
    def __init__(self, log_dir: str = "./metrics", enable_tensorboard: bool = True, ...)
    def log_training_metrics(self, metrics: TrainingMetrics) -> None
    def start_monitoring(self, interval: float = 5.0) -> None
    def stop_monitoring(self) -> None
    def start_profiling(self, activities: List[torch.profiler.ProfilerActivity] = None) -> None
    def stop_profiling(self) -> None
    def generate_report(self, output_path: Optional[str] = None) -> str
    def get_summary(self) -> Dict[str, Any]
```

## Examples

### Complete Training Pipeline

```python
#!/usr/bin/env python3
"""
Complete training pipeline example with all features.
"""

import os
import time
from pathlib import Path

from config_manager import ConfigManager, Environment
from error_handler import setup_logging, error_context, ErrorType
from performance_monitor import create_metrics_collector, TrainingMetrics
from test_framework import TestRunner

def main():
    """Main training pipeline."""
    
    # Setup logging
    logger = setup_logging(
        log_dir="./logs",
        enable_sentry=True,
        sentry_dsn=os.getenv("SENTRY_DSN")
    )
    
    # Setup monitoring
    collector = create_metrics_collector(
        log_dir="./metrics",
        enable_tensorboard=True,
        enable_wandb=True,
        wandb_project="frontier-model-training"
    )
    
    # Start monitoring
    collector.start_monitoring(interval=5.0)
    
    try:
        with error_context(ErrorType.CONFIGURATION, "main", "setup") as (logger, handler):
            # Load configuration
            manager = ConfigManager()
            config = manager.load_config("config/production.yaml")
            
            # Validate configuration
            issues = manager.validate_config(config)
            if issues:
                logger.warning(f"Configuration issues: {issues}")
            
            logger.info("Configuration loaded successfully")
        
        with error_context(ErrorType.TRAINING, "main", "training") as (logger, handler):
            # Run training
            logger.info("Starting training...")
            
            # Simulate training loop
            for step in range(1000):
                # Simulate training step
                loss = 1.0 - step * 0.001
                
                # Log metrics
                metrics = TrainingMetrics(
                    step=step,
                    epoch=step // 100,
                    training_loss=loss,
                    learning_rate=config.training.learning_rate * (0.9 ** (step // 100)),
                    batch_time=0.1,
                    throughput=100.0
                )
                collector.log_training_metrics(metrics)
                
                if step % 100 == 0:
                    logger.info(f"Step {step}, Loss: {loss:.4f}")
                
                time.sleep(0.01)  # Simulate processing time
            
            logger.info("Training completed successfully")
    
    except Exception as e:
        logger.log_error(e, ErrorType.TRAINING, "main", "training")
        raise
    
    finally:
        # Stop monitoring and generate report
        collector.stop_monitoring()
        report_path = collector.generate_report()
        logger.info(f"Performance report generated: {report_path}")
        
        # Get summary
        summary = collector.get_summary()
        logger.info(f"Training summary: {summary}")

if __name__ == "__main__":
    main()
```

### Custom Model Training

```python
#!/usr/bin/env python3
"""
Custom model training with advanced features.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from performance_monitor import create_metrics_collector, TrainingMetrics
from error_handler import error_handler, ErrorType

class CustomModelTrainer:
    def __init__(self, config):
        self.config = config
        self.collector = create_metrics_collector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @error_handler(ErrorType.MODEL_LOADING, "trainer", "load_model")
    def load_model(self):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            torch_dtype=torch.float16 if self.config.model.fp16 else torch.float32
        )
        self.model.to(self.device)
        
    @error_handler(ErrorType.TRAINING, "trainer", "train")
    def train(self, dataloader, optimizer, scheduler):
        """Training loop with monitoring."""
        self.model.train()
        
        for step, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Log metrics
            batch_time = time.time() - batch_start_time
            metrics = TrainingMetrics(
                step=step,
                epoch=step // len(dataloader),
                training_loss=loss.item(),
                learning_rate=scheduler.get_last_lr()[0],
                batch_time=batch_time,
                throughput=batch['input_ids'].size(0) / batch_time
            )
            self.collector.log_training_metrics(metrics)
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Usage
if __name__ == "__main__":
    from config_manager import ConfigManager
    
    # Load configuration
    manager = ConfigManager()
    config = manager.load_config("config/development.yaml")
    
    # Create trainer
    trainer = CustomModelTrainer(config)
    trainer.load_model()
    
    # Setup training
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Create dummy dataloader for example
    dummy_data = [{"input_ids": torch.randint(0, 1000, (8, 128))} for _ in range(100)]
    dataloader = DataLoader(dummy_data, batch_size=1)
    
    # Train
    trainer.train(dataloader, optimizer, scheduler)
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Solution: Enable gradient checkpointing and reduce batch size
config.optimization.use_gradient_checkpointing = True
config.training.batch_size = 4
config.training.gradient_accumulation_steps = 4
```

#### Slow Training

```python
# Solution: Enable performance optimizations
config.performance.use_cudnn_benchmark = True
config.performance.use_tf32 = True
config.performance.use_compile = True
config.optimization.use_flash_attention = True
```

#### Configuration Errors

```python
# Solution: Validate configuration
manager = ConfigManager()
config = manager.load_config("config.yaml")
issues = manager.validate_config(config)
if issues:
    print(f"Configuration issues: {issues}")
```

### Debug Mode

```python
# Enable debug logging
from error_handler import setup_logging, LogLevel

logger = setup_logging(
    log_dir="./logs",
    log_level=LogLevel.DEBUG,
    enable_console_logging=True
)
```

### Performance Debugging

```python
# Enable profiling
from performance_monitor import create_metrics_collector

collector = create_metrics_collector()
collector.start_profiling()

# Your training code here

collector.stop_profiling()
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd frontier-model-run

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
python test_framework.py --run-tests --coverage

# Run specific test suite
python test_framework.py --run-tests --test-suite unit_tests

# Run with verbose output
python test_framework.py --run-tests --verbose
```

### Code Style

```bash
# Format code
black scripts/
isort scripts/

# Lint code
flake8 scripts/
mypy scripts/
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the examples and API reference
- Contact the development team

---

*Last updated: 2025-01-22*
