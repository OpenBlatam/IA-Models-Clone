# Training Logging System

## Overview

The Training Logging System provides comprehensive, structured logging for machine learning training processes with particular focus on cybersecurity applications. It offers real-time progress tracking, error handling, security event logging, performance monitoring, and rich visualization capabilities.

## Key Features

### 🎯 **Structured Training Logging**
- **Event-based logging**: Track training events (epochs, batches, validation)
- **Metrics tracking**: Loss, accuracy, precision, recall, F1-score
- **Performance monitoring**: CPU, memory, GPU usage
- **Rich console output**: Beautiful progress bars and real-time updates

### 🛡️ **Security Event Logging**
- **Threat detection**: Log security anomalies and threats
- **Network monitoring**: Track source/destination IPs, ports, protocols
- **Confidence scoring**: Record threat confidence levels
- **Event categorization**: High, medium, low severity levels

### 📊 **Performance Monitoring**
- **Resource tracking**: CPU, memory, GPU, disk I/O
- **Performance alerts**: Automatic alerts for high resource usage
- **Batch timing**: Track training and inference times
- **Memory management**: Monitor memory leaks and usage patterns

### 🔧 **Error Handling & Recovery**
- **Comprehensive error logging**: Detailed error context and stack traces
- **Error categorization**: Classify errors by type and severity
- **Recovery mechanisms**: Automatic recovery strategies
- **Error tracking**: Persistent error history and analysis

### 📈 **Visualization & Analysis**
- **Training curves**: Loss, accuracy, learning rate plots
- **Performance dashboards**: Resource usage over time
- **Security event analysis**: Threat patterns and trends
- **CSV export**: Structured data for external analysis

## Architecture

### Core Components

1. **TrainingLogger Class**
   - Main logging coordinator
   - Event tracking and categorization
   - Metrics storage and retrieval
   - Log file management

2. **PerformanceMonitor Class**
   - System resource monitoring
   - Performance alert generation
   - Resource usage tracking
   - Alert threshold management

3. **Data Structures**
   - `TrainingMetrics`: Training progress and performance
   - `SecurityEvent`: Security incidents and threats
   - `PerformanceMetrics`: System resource usage

4. **Event Types**
   - `TrainingEventType`: Epoch, batch, validation events
   - `LogLevel`: Debug, info, warning, error, critical

### Logging Flow

```python
# Training event flow
1. Start training session
2. Log epoch start
3. For each batch:
   - Log batch start
   - Perform training
   - Log metrics and performance
   - Log batch end
4. Log validation results
5. Log epoch end
6. Repeat for all epochs
7. End training session
```

## Usage Guide

### Basic Setup

```python
# Create training logger
training_logger = create_training_logger({
    "log_dir": "training_logs",
    "log_level": LogLevel.INFO,
    "enable_console": True,
    "enable_file": True,
    "enable_rich": True
})
```

### Training Loop Integration

```python
# Start training
training_logger.start_training(total_epochs=10, total_batches=1000)

for epoch in range(10):
    training_logger.start_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        training_logger.start_batch(batch_idx, epoch)
        
        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch_idx,
            loss=loss.item(),
            accuracy=calculate_accuracy(output, target),
            learning_rate=optimizer.param_groups[0]['lr'],
            timestamp=datetime.now().isoformat()
        )
        
        training_logger.end_batch(batch_idx, epoch, metrics)
        
        # Log performance metrics
        perf_metrics = training_logger.performance_monitor.get_current_metrics()
        training_logger.log_performance_metrics(perf_metrics)
    
    # Validation
    val_loss, val_accuracy = validate_model(model, val_loader)
    training_logger.log_validation(epoch, val_loss, val_accuracy)
    training_logger.end_epoch(epoch)

# End training
training_logger.end_training()
```

### Security Event Logging

```python
# Log security anomalies
training_logger.log_security_anomaly(
    source_ip="192.168.1.100",
    destination_ip="10.0.0.50",
    threat_level="high",
    confidence=0.95,
    description="DDoS attack detected"
)

# Log security events
security_event = SecurityEvent(
    event_type="threat_detection",
    severity="high",
    description="Suspicious network activity",
    source_ip="192.168.1.100",
    destination_ip="10.0.0.50",
    port=80,
    protocol="TCP",
    threat_level="high",
    confidence=0.95
)

training_logger.log_security_event(security_event, level=LogLevel.WARNING)
```

### Error Handling

```python
try:
    # Training operation
    loss = model(data)
except Exception as e:
    training_logger.log_error(
        e,
        context={
            "epoch": epoch,
            "batch": batch_idx,
            "data_shape": data.shape,
            "model_state": "training"
        }
    )
    # Continue training or handle error
```

### Performance Monitoring

```python
# Get current performance metrics
perf_metrics = training_logger.performance_monitor.get_current_metrics()

# Log performance metrics
training_logger.log_performance_metrics(perf_metrics)

# Check for performance alerts
alerts = training_logger.performance_monitor.check_performance_alerts(perf_metrics)
for alert in alerts:
    training_logger.log_performance_alert("system", alert, perf_metrics)
```

## Configuration Options

### Logger Configuration

```python
config = {
    "log_dir": "logs",                    # Log directory
    "log_level": LogLevel.INFO,           # Minimum log level
    "enable_console": True,               # Console output
    "enable_file": True,                  # File logging
    "enable_rich": True,                  # Rich console output
    "max_log_files": 10,                  # Max log files to keep
    "log_rotation_size": 100 * 1024 * 1024  # Log rotation size (100MB)
}
```

### Performance Alert Thresholds

```python
# Default thresholds
CPU_USAGE_THRESHOLD = 90.0        # 90% CPU usage
MEMORY_USAGE_THRESHOLD = 90.0     # 90% memory usage
GPU_MEMORY_THRESHOLD = 8000.0     # 8GB GPU memory
```

## Data Structures

### TrainingMetrics

```python
@dataclass
class TrainingMetrics:
    epoch: int
    batch: int
    loss: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    timestamp: Optional[str] = None
```

### SecurityEvent

```python
@dataclass
class SecurityEvent:
    event_type: str
    severity: str
    description: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None
    threat_level: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    disk_io: Optional[float] = None
    network_io: Optional[float] = None
    batch_time: Optional[float] = None
    epoch_time: Optional[float] = None
    timestamp: Optional[str] = None
```

## Log Analysis and Reporting

### Training Summary

```python
# Get comprehensive training summary
summary = training_logger.get_training_summary()

# Summary includes:
# - Session information
# - Training duration
# - Metrics statistics
# - Security events count
# - Performance metrics
# - Error statistics
```

### Metrics Export

```python
# Save training metrics to CSV
training_logger.save_metrics_to_csv("training_metrics.csv")

# Generate training curves
training_logger.plot_training_curves(
    save_path="training_curves.png",
    show_plot=True
)
```

### Log Analysis

```python
# Analyze security events
security_events = training_logger.security_events
high_severity_events = [e for e in security_events if e.severity == "high"]

# Analyze performance metrics
performance_metrics = training_logger.performance_metrics
avg_cpu_usage = np.mean([m.cpu_usage for m in performance_metrics])
max_memory_usage = max([m.memory_usage for m in performance_metrics])
```

## Security Considerations

### Input Validation

```python
# Validate IP addresses
def validate_ip_address(ip: str) -> bool:
    try:
        import ipaddress
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

# Validate threat levels
VALID_THREAT_LEVELS = ["low", "medium", "high"]
def validate_threat_level(level: str) -> bool:
    return level in VALID_THREAT_LEVELS
```

### Log Security

```python
# Sanitize sensitive data
def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_fields = ["password", "token", "key", "secret"]
    sanitized = {}
    
    for key, value in data.items():
        if any(field in key.lower() for field in sensitive_fields):
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    
    return sanitized
```

### Access Control

```python
# Implement log access control
def check_log_access(user: str, log_file: str) -> bool:
    # Implement access control logic
    authorized_users = ["admin", "analyst"]
    return user in authorized_users
```

## Performance Optimization

### Memory Management

```python
# Efficient metrics storage
def optimize_metrics_storage(metrics: List[TrainingMetrics]) -> List[TrainingMetrics]:
    # Keep only essential metrics for long training runs
    if len(metrics) > 10000:
        # Sample metrics to reduce memory usage
        return metrics[::10]  # Keep every 10th metric
    return metrics
```

### Log Rotation

```python
# Automatic log rotation
def rotate_logs(log_dir: Path, max_size: int = 100 * 1024 * 1024):
    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_size > max_size:
            # Create backup and start new log
            backup_file = log_file.with_suffix(f".{int(time.time())}.log")
            log_file.rename(backup_file)
            log_file.touch()
```

### Async Logging

```python
# Async logging for high-frequency events
async def async_log_metrics(logger: TrainingLogger, metrics: TrainingMetrics):
    await asyncio.get_event_loop().run_in_executor(
        None, logger.log_training_event, 
        TrainingEventType.METRIC_UPDATE, 
        "Metrics update", 
        metrics
    )
```

## Integration with Existing Systems

### Robust Operations Integration

```python
# Integrate with robust operations
robust_ops = RobustOperations(config)
training_logger = create_training_logger(config)

# Use robust operations for error handling
@robust_ops.operation_context("training", OperationType.MODEL_INFERENCE)
def train_with_robust_ops():
    try:
        # Training code
        pass
    except Exception as e:
        training_logger.log_error(e)
        raise
```

### FastAPI Integration

```python
# FastAPI endpoint for training monitoring
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

@app.post("/start_training")
async def start_training(background_tasks: BackgroundTasks):
    training_logger = create_training_logger()
    background_tasks.add_task(train_model, training_logger)
    return {"message": "Training started", "session_id": training_logger.session_id}

@app.get("/training_status/{session_id}")
async def get_training_status(session_id: str):
    # Retrieve training status from logs
    return {"status": "training", "progress": 75}
```

### Database Integration

```python
# Store logs in database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///training_logs.db")
Session = sessionmaker(bind=engine)

def store_metrics_in_db(metrics: TrainingMetrics):
    session = Session()
    # Store metrics in database
    session.commit()
    session.close()
```

## Best Practices

### Logging Best Practices

1. **Structured Logging**
   ```python
   # Good: Structured logging
   training_logger.log_training_event(
       TrainingEventType.BATCH_END,
       "Batch completed",
       metrics=metrics,
       level=LogLevel.INFO
   )
   
   # Bad: Unstructured logging
   print(f"Batch {batch} completed with loss {loss}")
   ```

2. **Error Context**
   ```python
   # Good: Rich error context
   training_logger.log_error(
       error,
       context={
           "epoch": epoch,
           "batch": batch,
           "data_shape": data.shape,
           "model_state": model.training
       }
   )
   ```

3. **Performance Monitoring**
   ```python
   # Good: Regular performance monitoring
   perf_metrics = training_logger.performance_monitor.get_current_metrics()
   training_logger.log_performance_metrics(perf_metrics)
   
   alerts = training_logger.performance_monitor.check_performance_alerts(perf_metrics)
   for alert in alerts:
       training_logger.log_performance_alert("system", alert, perf_metrics)
   ```

### Security Best Practices

1. **Input Validation**
   ```python
   # Validate all inputs
   if not validate_ip_address(source_ip):
       raise ValueError("Invalid source IP address")
   
   if not validate_threat_level(threat_level):
       raise ValueError("Invalid threat level")
   ```

2. **Data Sanitization**
   ```python
   # Sanitize sensitive data before logging
   sanitized_data = sanitize_log_data(log_data)
   training_logger.log_training_event(event_type, message, **sanitized_data)
   ```

3. **Access Control**
   ```python
   # Implement proper access control
   if not check_log_access(user, log_file):
       raise PermissionError("Access denied")
   ```

### Performance Best Practices

1. **Efficient Storage**
   ```python
   # Use efficient data structures
   metrics = TrainingMetrics(
       epoch=epoch,
       batch=batch,
       loss=loss.item(),
       accuracy=accuracy,
       timestamp=datetime.now().isoformat()
   )
   ```

2. **Log Rotation**
   ```python
   # Implement log rotation
   training_logger = create_training_logger({
       "max_log_files": 10,
       "log_rotation_size": 100 * 1024 * 1024
   })
   ```

3. **Async Operations**
   ```python
   # Use async for high-frequency logging
   await async_log_metrics(training_logger, metrics)
   ```

## Monitoring and Alerting

### Performance Alerts

```python
# Configure performance alerts
PERFORMANCE_ALERTS = {
    "cpu_usage": 90.0,
    "memory_usage": 90.0,
    "gpu_memory": 8000.0,
    "batch_time": 5.0
}

# Check alerts
alerts = training_logger.performance_monitor.check_performance_alerts(perf_metrics)
for alert in alerts:
    training_logger.log_performance_alert("performance", alert, perf_metrics)
```

### Security Alerts

```python
# Configure security alerts
SECURITY_ALERTS = {
    "high_threat_confidence": 0.9,
    "multiple_high_severity_events": 5,
    "unusual_network_patterns": True
}

# Monitor security events
high_confidence_events = [
    e for e in training_logger.security_events 
    if e.confidence and e.confidence > SECURITY_ALERTS["high_threat_confidence"]
]
```

### Training Alerts

```python
# Configure training alerts
TRAINING_ALERTS = {
    "loss_increase_threshold": 0.1,
    "accuracy_decrease_threshold": 0.05,
    "gradient_explosion_threshold": 10.0
}

# Monitor training metrics
for metrics in training_logger.training_metrics[-10:]:  # Last 10 batches
    if metrics.loss > TRAINING_ALERTS["loss_increase_threshold"]:
        training_logger.log_training_event(
            TrainingEventType.WARNING,
            "Loss increase detected",
            level=LogLevel.WARNING
        )
```

## Conclusion

The Training Logging System provides a comprehensive solution for logging machine learning training processes with particular emphasis on cybersecurity applications. It offers structured logging, security event tracking, performance monitoring, and rich visualization capabilities.

The system is designed to be production-ready with proper error handling, security considerations, and performance optimization. It integrates seamlessly with existing robust operations frameworks and provides the tools needed for comprehensive training monitoring and analysis.

For questions, issues, or contributions, please refer to the project documentation or contact the development team. 