# Training Logging System Guide

## Overview

This guide covers the comprehensive training logging system that provides proper logging for training progress and errors. The system includes real-time monitoring, detailed error tracking, performance metrics, and multiple output formats for complete training visibility.

## 📊 Available Training Logging Systems

### 1. Training Logging System (`training_logging_system.py`)
**Port**: 7870
**Description**: Comprehensive logging for training progress and errors

**Features**:
- **Training Progress Logging**: Real-time metrics tracking and visualization
- **Error Logging**: Detailed error context and automatic recovery strategies
- **Performance Monitoring**: Resource usage tracking and optimization
- **Model Checkpoint Logging**: Checkpoint management and validation
- **Multiple Output Formats**: Console, file, CSV, JSON, and TensorBoard logging
- **Resource Monitoring**: CPU, memory, and GPU usage tracking

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_gradio_demos.txt
pip install tensorboard psutil
```

2. **Launch Training Logging System**:
```bash
# Launch training logging system
python demo_launcher.py --demo training-logging

# Launch all logging systems
python demo_launcher.py --all
```

### Direct Launch

```bash
# Training logging system
python training_logging_system.py
```

## 📊 Training Logging Features

### Training Session Management

**Session Start Logging**:
```python
def log_training_start(self, config: Dict[str, Any]):
    """Log training session start"""
    self.logger.info("=" * 80)
    self.logger.info(f"TRAINING SESSION STARTED - {self.experiment_name}")
    self.logger.info("=" * 80)
    self.logger.info(f"Start time: {self.start_time}")
    self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Log system information
    self._log_system_info()
    
    # Start resource monitoring
    self.resource_monitor.start()
```

**Session End Logging**:
```python
def log_training_end(self, final_metrics: Dict[str, Any]):
    """Log training session end"""
    end_time = datetime.now()
    duration = (end_time - self.start_time).total_seconds()
    
    self.logger.info("=" * 80)
    self.logger.info(f"TRAINING SESSION COMPLETED - {self.experiment_name}")
    self.logger.info("=" * 80)
    self.logger.info(f"End time: {end_time}")
    self.logger.info(f"Total duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    # Stop resource monitoring
    self.resource_monitor.stop()
    
    # Generate training summary
    self._generate_training_summary()
```

### Training Progress Logging

**Step-by-Step Logging**:
```python
def log_training_step(self, step: int, loss: float, accuracy: Optional[float] = None,
                     learning_rate: Optional[float] = None, gradient_norm: Optional[float] = None):
    """Log individual training step"""
    self.current_step = step
    
    # Get resource usage
    memory_usage = self.resource_monitor.get_memory_usage()
    gpu_memory = self.resource_monitor.get_gpu_memory_usage()
    
    # Create metrics object
    metrics = TrainingMetrics(
        epoch=self.current_epoch,
        step=step,
        loss=loss,
        accuracy=accuracy,
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        memory_usage=memory_usage,
        gpu_memory=gpu_memory
    )
    
    # Store metrics
    self.training_metrics.append(metrics)
    
    # Log to console (every 100 steps)
    if step % 100 == 0:
        self.logger.info(
            f"Step {step:6d} | Loss: {loss:.4f} | "
            f"Acc: {accuracy:.4f if accuracy else 'N/A'} | "
            f"LR: {learning_rate:.6f if learning_rate else 'N/A'} | "
            f"Memory: {memory_usage['used_mb']:.1f}MB"
        )
    
    # Log to TensorBoard
    if self.writer:
        self.writer.add_scalar('Training/Loss', loss, step)
        if accuracy:
            self.writer.add_scalar('Training/Accuracy', accuracy, step)
        if learning_rate:
            self.writer.add_scalar('Training/LearningRate', learning_rate, step)
        if gradient_norm:
            self.writer.add_scalar('Training/GradientNorm', gradient_norm, step)
        
        # Log resource usage
        self.writer.add_scalar('System/Memory_Usage_MB', memory_usage['used_mb'], step)
        if gpu_memory:
            self.writer.add_scalar('System/GPU_Memory_MB', gpu_memory['allocated_mb'], step)
    
    # Log to CSV
    self._log_metrics_to_csv(metrics)
```

**Epoch Logging**:
```python
def log_epoch_start(self, epoch: int, total_epochs: int):
    """Log epoch start"""
    self.current_epoch = epoch
    epoch_start_time = datetime.now()
    
    self.logger.info("-" * 60)
    self.logger.info(f"EPOCH {epoch}/{total_epochs} STARTED")
    self.logger.info(f"Start time: {epoch_start_time}")
    
    if self.writer:
        self.writer.add_scalar('Training/Epoch', epoch, epoch)

def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
    """Log epoch end with metrics"""
    epoch_end_time = datetime.now()
    epoch_duration = (epoch_end_time - self.start_time).total_seconds()
    
    self.logger.info(f"EPOCH {epoch} COMPLETED")
    self.logger.info(f"Duration: {epoch_duration:.2f} seconds")
    self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    self.logger.info("-" * 60)
    
    # Log to TensorBoard
    if self.writer:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Epoch/{key}', value, epoch)
```

### Validation Logging

**Validation Results Logging**:
```python
def log_validation(self, validation_loss: float, validation_accuracy: Optional[float] = None,
                  additional_metrics: Optional[Dict[str, float]] = None):
    """Log validation results"""
    self.logger.info(
        f"VALIDATION | Loss: {validation_loss:.4f} | "
        f"Accuracy: {validation_accuracy:.4f if validation_accuracy else 'N/A'}"
    )
    
    if additional_metrics:
        self.logger.info(f"Additional metrics: {json.dumps(additional_metrics, indent=2)}")
    
    # Log to TensorBoard
    if self.writer:
        self.writer.add_scalar('Validation/Loss', validation_loss, self.current_step)
        if validation_accuracy:
            self.writer.add_scalar('Validation/Accuracy', validation_accuracy, self.current_step)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Validation/{key}', value, self.current_step)
```

### Error Logging

**Comprehensive Error Logging**:
```python
def log_error(self, error: Exception, context: Dict[str, Any] = None, 
              severity: str = "ERROR", recovery_action: str = None):
    """Log training error with detailed context"""
    error_log = ErrorLog(
        timestamp=datetime.now(),
        error_type=type(error).__name__,
        error_message=str(error),
        stack_trace=traceback.format_exc(),
        context=context or {},
        severity=severity,
        epoch=self.current_epoch,
        step=self.current_step,
        recovery_action=recovery_action
    )
    
    self.error_logs.append(error_log)
    
    # Log error details
    self.logger.error(f"TRAINING ERROR: {error_log.error_type}")
    self.logger.error(f"Message: {error_log.error_message}")
    self.logger.error(f"Context: {json.dumps(error_log.context, indent=2)}")
    self.logger.error(f"Stack trace: {error_log.stack_trace}")
    
    if recovery_action:
        self.logger.info(f"Recovery action: {recovery_action}")
    
    # Log to TensorBoard
    if self.writer:
        self.writer.add_text('Errors/Error_Log', 
                           f"Type: {error_log.error_type}\nMessage: {error_log.error_message}\nContext: {error_log.context}",
                           self.current_step)
    
    # Try automatic recovery
    self._attempt_error_recovery(error_log)
```

**Error Recovery Strategies**:
```python
def _attempt_error_recovery(self, error_log: ErrorLog):
    """Attempt automatic error recovery"""
    recovery_func = self.error_recovery_strategies.get(error_log.error_type)
    if recovery_func:
        try:
            recovery_action = recovery_func(error_log)
            error_log.recovery_action = recovery_action
            error_log.resolved = True
            self.logger.info(f"Error recovery attempted: {recovery_action}")
        except Exception as e:
            self.logger.error(f"Error recovery failed: {e}")

def _handle_memory_error(self, error_log: ErrorLog) -> str:
    """Handle memory errors"""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    return "Cleared GPU cache and forced garbage collection"

def _handle_cuda_error(self, error_log: ErrorLog) -> str:
    """Handle CUDA errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "Cleared CUDA cache"
    return "CUDA not available"
```

### Checkpoint Logging

**Model Checkpoint Logging**:
```python
def log_checkpoint(self, epoch: int, step: int, file_path: str, metrics: Dict[str, float],
                  is_best: bool = False):
    """Log model checkpoint"""
    checkpoint_log = CheckpointLog(
        timestamp=datetime.now(),
        epoch=epoch,
        step=step,
        file_path=file_path,
        metrics=metrics,
        model_size=self._get_file_size(file_path),
        validation_score=metrics.get('validation_loss'),
        is_best=is_best
    )
    
    self.checkpoint_logs.append(checkpoint_log)
    
    self.logger.info(
        f"CHECKPOINT SAVED | Epoch {epoch} | Step {step} | "
        f"File: {file_path} | Size: {checkpoint_log.model_size:.2f}MB | "
        f"Best: {is_best}"
    )
    
    # Log to TensorBoard
    if self.writer:
        self.writer.add_scalar('Checkpoint/Model_Size_MB', checkpoint_log.model_size, step)
        if is_best:
            self.writer.add_scalar('Checkpoint/Best_Model', 1, step)
    
    # Save checkpoint log
    self._save_checkpoint_log(checkpoint_log)
```

## 🔧 Logging Configuration

### Multiple Output Formats

**File Logging Setup**:
```python
def setup_logging(self):
    """Setup comprehensive logging configuration"""
    # Create log files
    self.log_file = self.experiment_dir / "training.log"
    self.error_log_file = self.experiment_dir / "errors.log"
    self.metrics_file = self.experiment_dir / "metrics.csv"
    self.checkpoint_log_file = self.experiment_dir / "checkpoints.json"
    
    # Configure main logger
    self.logger = logging.getLogger(f"training_{self.experiment_name}")
    self.logger.setLevel(logging.INFO)
    
    # File handler for training logs
    file_handler = logging.FileHandler(self.log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    self.logger.addHandler(file_handler)
    
    # File handler for error logs
    error_handler = logging.FileHandler(self.error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
    )
    error_handler.setFormatter(error_formatter)
    self.logger.addHandler(error_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    self.logger.addHandler(console_handler)
```

**TensorBoard Integration**:
```python
def setup_tensorboard(self):
    """Setup TensorBoard logging"""
    try:
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        self.logger.info(f"TensorBoard logging enabled: {self.tensorboard_dir}")
    except Exception as e:
        self.logger.warning(f"TensorBoard setup failed: {e}")
        self.writer = None
```

**CSV Metrics Logging**:
```python
def setup_metrics_tracking(self):
    """Setup metrics tracking and CSV logging"""
    # Initialize metrics CSV file
    metrics_headers = [
        'timestamp', 'epoch', 'step', 'loss', 'accuracy', 'learning_rate',
        'gradient_norm', 'validation_loss', 'validation_accuracy',
        'duration', 'memory_usage_mb', 'gpu_memory_mb'
    ]
    
    with open(self.metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics_headers)

def _log_metrics_to_csv(self, metrics: TrainingMetrics):
    """Log metrics to CSV file"""
    try:
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.timestamp.isoformat(),
                metrics.epoch,
                metrics.step,
                metrics.loss,
                metrics.accuracy or '',
                metrics.learning_rate or '',
                metrics.gradient_norm or '',
                metrics.validation_loss or '',
                metrics.validation_accuracy or '',
                metrics.duration or '',
                metrics.memory_usage['used_mb'] if metrics.memory_usage else '',
                metrics.gpu_memory['allocated_mb'] if metrics.gpu_memory else ''
            ])
    except Exception as e:
        self.logger.error(f"Failed to write metrics to CSV: {e}")
```

## 📊 Resource Monitoring

### System Resource Tracking

**Resource Monitor**:
```python
class ResourceMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        self.metrics_history = deque(maxlen=10000)
    
    def start(self):
        """Start resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
                    'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                    'gpu_memory': self.get_gpu_memory_usage()
                }
                
                self.metrics_queue.put(metrics)
                self.metrics_history.append(metrics)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(5)  # Wait longer on error
```

**Memory Usage Tracking**:
```python
def get_memory_usage(self) -> Dict[str, float]:
    """Get current memory usage"""
    memory = psutil.virtual_memory()
    return {
        'total_mb': memory.total / (1024 * 1024),
        'used_mb': memory.used / (1024 * 1024),
        'available_mb': memory.available / (1024 * 1024),
        'percent': memory.percent
    }

def get_gpu_memory_usage(self) -> Optional[Dict[str, float]]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None
    
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': total,
            'free_mb': total - reserved
        }
    except Exception as e:
        logger.error(f"GPU memory monitoring error: {e}")
        return None
```

## 📈 Training Summary Generation

### Comprehensive Summary

**Training Summary Generation**:
```python
def _generate_training_summary(self):
    """Generate training summary report"""
    summary_file = self.experiment_dir / "training_summary.json"
    
    summary = {
        "experiment_name": self.experiment_name,
        "start_time": self.start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_epochs": self.current_epoch,
        "total_steps": self.current_step,
        "total_errors": len(self.error_logs),
        "total_checkpoints": len(self.checkpoint_logs),
        "final_metrics": self.training_metrics[-1].__dict__ if self.training_metrics else None,
        "error_summary": self._get_error_summary(),
        "performance_summary": self.resource_monitor.get_summary()
    }
    
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Training summary saved to: {summary_file}")
    except Exception as e:
        self.logger.error(f"Failed to save training summary: {e}")
```

**Error Summary**:
```python
def _get_error_summary(self) -> Dict[str, Any]:
    """Get error summary statistics"""
    if not self.error_logs:
        return {"message": "No errors recorded"}
    
    error_types = defaultdict(int)
    severities = defaultdict(int)
    resolved_count = sum(1 for log in self.error_logs if log.resolved)
    
    for log in self.error_logs:
        error_types[log.error_type] += 1
        severities[log.severity] += 1
    
    return {
        "total_errors": len(self.error_logs),
        "resolved_errors": resolved_count,
        "resolution_rate": resolved_count / len(self.error_logs) if self.error_logs else 0,
        "error_types": dict(error_types),
        "severities": dict(severities)
    }
```

## 🎯 Usage Examples

### Basic Training Logging

```python
from training_logging_system import TrainingLogger

# Create training logger
training_logger = TrainingLogger("logs", "my_experiment")

# Start training session
config = {
    "model": "resnet50",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100
}
training_logger.log_training_start(config)

# Log training steps
for epoch in range(10):
    training_logger.log_epoch_start(epoch, 10)
    
    for step in range(100):
        loss = 0.5 - (step * 0.001)  # Simulated loss
        accuracy = 0.8 + (step * 0.001)  # Simulated accuracy
        
        training_logger.log_training_step(step, loss, accuracy, 0.001)
    
    # Log validation
    training_logger.log_validation(0.3, 0.85)
    
    # Log checkpoint
    training_logger.log_checkpoint(epoch, step, f"checkpoint_{epoch}.pth", 
                                  {"loss": loss, "accuracy": accuracy})
    
    training_logger.log_epoch_end(epoch, {"loss": loss, "accuracy": accuracy})

# End training session
final_metrics = {"final_loss": 0.1, "final_accuracy": 0.95}
training_logger.log_training_end(final_metrics)
training_logger.close()
```

### Error Handling Integration

```python
# Log errors with automatic recovery
try:
    # Training operation
    result = model(input_data)
except Exception as e:
    context = {
        "input_shape": input_data.shape,
        "model_type": type(model).__name__,
        "device": str(input_data.device)
    }
    training_logger.log_error(e, context, "ERROR")
```

### Progress Monitoring

```python
# Get current training progress
progress = training_logger.get_training_progress()
print(f"Current epoch: {progress['current_epoch']}")
print(f"Current step: {progress['current_step']}")
print(f"Latest loss: {progress['latest_loss']}")
print(f"Memory usage: {progress['memory_usage']}")
```

## 📁 Generated Log Files

### Log File Structure

**Generated Files**:
```
logs/
└── my_experiment/
    ├── training.log              # Main training log
    ├── errors.log                # Error logs with stack traces
    ├── metrics.csv               # Training metrics in CSV format
    ├── checkpoints.json          # Checkpoint information
    ├── training_summary.json     # Final training summary
    └── tensorboard/              # TensorBoard logs
        ├── events.out.tfevents.*
        └── ...
```

### Log File Contents

**Training Log Format**:
```
2024-01-15 10:30:00 - training_my_experiment - INFO - ================================================================================
2024-01-15 10:30:00 - training_my_experiment - INFO - TRAINING SESSION STARTED - my_experiment
2024-01-15 10:30:00 - training_my_experiment - INFO - ================================================================================
2024-01-15 10:30:00 - training_my_experiment - INFO - Start time: 2024-01-15 10:30:00
2024-01-15 10:30:00 - training_my_experiment - INFO - Configuration: {"model": "resnet50", "batch_size": 32}
2024-01-15 10:30:00 - training_my_experiment - INFO - SYSTEM INFORMATION:
2024-01-15 10:30:00 - training_my_experiment - INFO - Python version: 3.9.0
2024-01-15 10:30:00 - training_my_experiment - INFO - PyTorch version: 1.12.0
2024-01-15 10:30:00 - training_my_experiment - INFO - CUDA available: True
2024-01-15 10:30:00 - training_my_experiment - INFO - ------------------------------------------------------------
2024-01-15 10:30:00 - training_my_experiment - INFO - EPOCH 1/10 STARTED
2024-01-15 10:30:00 - training_my_experiment - INFO - Start time: 2024-01-15 10:30:00
2024-01-15 10:30:00 - training_my_experiment - INFO - Step    100 | Loss: 0.4000 | Acc: 0.8500 | LR: 0.001000 | Memory: 2048.0MB
2024-01-15 10:30:00 - training_my_experiment - INFO - VALIDATION | Loss: 0.3000 | Accuracy: 0.9000
2024-01-15 10:30:00 - training_my_experiment - INFO - CHECKPOINT SAVED | Epoch 1 | Step 100 | File: checkpoint_1.pth | Size: 25.50MB | Best: True
2024-01-15 10:30:00 - training_my_experiment - INFO - EPOCH 1 COMPLETED
2024-01-15 10:30:00 - training_my_experiment - INFO - Duration: 3600.00 seconds
2024-01-15 10:30:00 - training_my_experiment - INFO - Metrics: {"loss": 0.4, "accuracy": 0.85}
2024-01-15 10:30:00 - training_my_experiment - INFO - ------------------------------------------------------------
```

**Error Log Format**:
```
2024-01-15 10:30:00 - training_my_experiment - ERROR - TRAINING ERROR: OutOfMemoryError
2024-01-15 10:30:00 - training_my_experiment - ERROR - Message: CUDA out of memory
2024-01-15 10:30:00 - training_my_experiment - ERROR - Context: {"batch_size": 32, "input_shape": [32, 3, 224, 224]}
2024-01-15 10:30:00 - training_my_experiment - ERROR - Stack trace: Traceback (most recent call last):
  File "training.py", line 100, in <module>
    result = model(input_data)
RuntimeError: CUDA out of memory
2024-01-15 10:30:00 - training_my_experiment - INFO - Recovery action: Cleared GPU cache and forced garbage collection
```

**Metrics CSV Format**:
```csv
timestamp,epoch,step,loss,accuracy,learning_rate,gradient_norm,validation_loss,validation_accuracy,duration,memory_usage_mb,gpu_memory_mb
2024-01-15T10:30:00,1,100,0.4000,0.8500,0.001000,,0.3000,0.9000,,2048.0,1024.0
2024-01-15T10:30:01,1,101,0.3990,0.8510,0.001000,,,2048.0,1024.0
```

## 🔍 TensorBoard Integration

### TensorBoard Usage

**Launch TensorBoard**:
```bash
# Navigate to experiment directory
cd logs/my_experiment

# Launch TensorBoard
tensorboard --logdir=tensorboard --port=6006
```

**Available Visualizations**:
- **Training Metrics**: Loss, accuracy, learning rate curves
- **Validation Metrics**: Validation loss and accuracy
- **System Metrics**: Memory usage, GPU memory, CPU usage
- **Checkpoint Information**: Model size, best model tracking
- **Error Logs**: Error types and messages
- **Text Logs**: Detailed error and warning messages

## 🎯 Best Practices

### Logging Best Practices

1. **Comprehensive Context**: Include relevant context with every log entry
2. **Structured Logging**: Use consistent log formats and levels
3. **Error Recovery**: Implement automatic error recovery strategies
4. **Performance Monitoring**: Track resource usage continuously
5. **Checkpoint Management**: Log all checkpoint operations with metadata

### Training Logging Best Practices

1. **Regular Logging**: Log training steps at regular intervals
2. **Validation Logging**: Log validation results after each epoch
3. **Error Handling**: Log all errors with detailed context
4. **Resource Monitoring**: Monitor system resources during training
5. **Summary Generation**: Generate comprehensive training summaries

### Performance Optimization

1. **Asynchronous Logging**: Use background threads for logging
2. **Batch Logging**: Batch multiple log entries when possible
3. **Log Rotation**: Implement log rotation for long training sessions
4. **Selective Logging**: Log only essential information at high frequencies
5. **Compression**: Compress old log files to save space

## 🔧 Configuration Options

### Logger Configuration

**Custom Configuration**:
```python
# Custom log directory and experiment name
training_logger = TrainingLogger(
    log_dir="custom_logs",
    experiment_name="my_custom_experiment"
)

# Custom logging levels
training_logger.logger.setLevel(logging.DEBUG)

# Custom formatters
custom_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
```

### TensorBoard Configuration

**Custom TensorBoard Setup**:
```python
# Custom TensorBoard directory
training_logger.tensorboard_dir = Path("custom_tensorboard")

# Custom TensorBoard writer
from torch.utils.tensorboard import SummaryWriter
custom_writer = SummaryWriter(
    log_dir=str(training_logger.tensorboard_dir),
    comment="custom_experiment"
)
```

## 📚 API Reference

### TrainingLogger Methods

**Core Methods**:
- `log_training_start(config)` → Log training session start
- `log_training_end(final_metrics)` → Log training session end
- `log_epoch_start(epoch, total_epochs)` → Log epoch start
- `log_epoch_end(epoch, metrics)` → Log epoch end
- `log_training_step(step, loss, accuracy, lr, gradient_norm)` → Log training step
- `log_validation(validation_loss, validation_accuracy, additional_metrics)` → Log validation
- `log_checkpoint(epoch, step, file_path, metrics, is_best)` → Log checkpoint
- `log_error(error, context, severity, recovery_action)` → Log error
- `log_warning(message, context)` → Log warning
- `log_info(message, context)` → Log information

**Utility Methods**:
- `get_training_progress()` → Get current training progress
- `close()` → Close logging resources
- `_generate_training_summary()` → Generate training summary
- `_get_error_summary()` → Get error summary statistics

### Data Structures

**TrainingMetrics**:
```python
@dataclass
class TrainingMetrics:
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None
    gpu_memory: Optional[Dict[str, float]] = None
```

**ErrorLog**:
```python
@dataclass
class ErrorLog:
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    severity: str
    epoch: Optional[int] = None
    step: Optional[int] = None
    recovery_action: Optional[str] = None
    resolved: bool = False
```

**CheckpointLog**:
```python
@dataclass
class CheckpointLog:
    timestamp: datetime
    epoch: int
    step: int
    file_path: str
    metrics: Dict[str, float]
    model_size: float
    validation_score: Optional[float] = None
    is_best: bool = False
```

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Analytics**: ML-based training analysis and insights
2. **Real-time Alerts**: Automated alerting for training issues
3. **Distributed Logging**: Multi-node training logging coordination
4. **Cloud Integration**: Cloud-based logging and monitoring
5. **Custom Dashboards**: Custom web dashboards for training visualization

### Technology Integration

1. **MLflow Integration**: MLflow experiment tracking
2. **Weights & Biases**: W&B logging integration
3. **Prometheus**: Prometheus metrics integration
4. **Grafana**: Grafana dashboard integration
5. **Elasticsearch**: Log aggregation and search

---

**Comprehensive Training Logging for Reliable AI Training! 📊**

For more information, see the main documentation or run:
```bash
python demo_launcher.py --help
``` 