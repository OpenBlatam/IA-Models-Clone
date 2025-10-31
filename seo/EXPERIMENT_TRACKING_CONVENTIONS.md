# Experiment Tracking Integration - Key Conventions

## 📋 **Core Conventions for TensorBoard & Weights & Biases Integration**

This document outlines the essential conventions and standards for integrating TensorBoard and Weights & Biases with our Advanced LLM SEO Engine code profiling system.

## 🔧 **1. Import and Setup Conventions**

### **Required Imports**
```python
# Standard imports for experiment tracking
from torch.utils.tensorboard import SummaryWriter
import wandb
import time
from typing import Dict, Any, List
import torch
import logging
```

### **Class Naming Conventions**
```python
# Use descriptive class names with "Tracker" suffix
class TensorBoardTracker:      # ✅ Correct
class WandBTracker:            # ✅ Correct
class ExperimentTracker:       # ✅ Correct

# Avoid generic names
class Tracker:                 # ❌ Too generic
class Logger:                  # ❌ Confusing with logging module
class Monitor:                 # ❌ Unclear purpose
```

### **Method Naming Conventions**
```python
# Use descriptive method names with clear purpose
def setup_tensorboard(self):           # ✅ Clear setup method
def log_training_metrics(self):        # ✅ Clear logging method
def log_performance_metrics(self):     # ✅ Clear performance logging
def log_profiling_data(self):         # ✅ Clear profiling logging

# Avoid abbreviations
def setup_tb(self):                    # ❌ Abbreviated
def log_train(self):                   # ❌ Abbreviated
def log_perf(self):                    # ❌ Abbreviated
```

## 📊 **2. Profiling Integration Conventions**

### **Profiling Context Manager Usage**
```python
# Always wrap experiment tracking operations with profiling context
def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
    with self.code_profiler.profile_operation("tensorboard_training_logging", "experiment_tracking"):
        # Logging logic here
        pass

# Use descriptive operation names
"tensorboard_setup"           # ✅ Clear operation name
"wandb_training_logging"      # ✅ Clear operation name
"unified_profiling_logging"   # ✅ Clear operation name

# Avoid generic names
"logging"                     # ❌ Too generic
"tracking"                    # ❌ Too generic
"metrics"                     # ❌ Too generic
```

### **Profiling Categories**
```python
# Use consistent profiling categories
"experiment_tracking"         # ✅ Main category for experiment tracking
"progress_monitoring"         # ✅ For progress tracking operations
"performance_benchmarking"    # ✅ For benchmarking operations
"memory_usage"               # ✅ For memory-related operations
"parallel_computation"       # ✅ For multi-threaded operations

# Avoid inconsistent categories
"tracking"                   # ❌ Inconsistent with main category
"monitoring"                 # ❌ Inconsistent with main category
"benchmark"                  # ❌ Inconsistent with main category
```

## 🏗️ **3. Configuration Conventions**

### **Configuration Parameter Naming**
```python
# Use descriptive configuration parameter names
@dataclass
class SEOConfig:
    # Experiment tracking settings
    tensorboard_logging: bool = True                    # ✅ Clear boolean flag
    wandb_logging: bool = True                         # ✅ Clear boolean flag
    experiment_tracking_enabled: bool = True            # ✅ Clear boolean flag
    
    # TensorBoard settings
    tensorboard_log_dir: str = "runs"                  # ✅ Clear directory setting
    tensorboard_flush_secs: int = 30                   # ✅ Clear time setting
    tensorboard_max_queue: int = 10                    # ✅ Clear queue setting
    
    # Weights & Biases settings
    wandb_project: str = "seo-engine-experiments"      # ✅ Clear project name
    wandb_entity: str = None                           # ✅ Clear entity setting
    wandb_tags: List[str] = None                       # ✅ Clear tags setting
    
    # Performance optimization
    experiment_logging_interval: int = 10               # ✅ Clear interval setting
    asynchronous_logging: bool = True                  # ✅ Clear async setting
    selective_logging: bool = True                     # ✅ Clear selective setting
```

### **Configuration Validation**
```python
# Always validate configuration parameters
def validate_config(self):
    """Validate experiment tracking configuration."""
    if not self.config.experiment_tracking_enabled:
        return
    
    # Validate TensorBoard settings
    if self.config.tensorboard_logging:
        if not self.config.tensorboard_log_dir:
            raise ValueError("tensorboard_log_dir must be specified when tensorboard_logging is True")
        if self.config.tensorboard_flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be positive")
        if self.config.tensorboard_max_queue <= 0:
            raise ValueError("tensorboard_max_queue must be positive")
    
    # Validate Weights & Biases settings
    if self.config.wandb_logging:
        if not self.config.wandb_project:
            raise ValueError("wandb_project must be specified when wandb_logging is True")
        if self.config.experiment_logging_interval <= 0:
            raise ValueError("experiment_logging_interval must be positive")
```

## 📝 **4. Logging Conventions**

### **Log Message Formatting**
```python
# Use consistent log message formatting with emojis and clear status
self.logger.info(f"✅ TensorBoard initialized: {log_dir}")           # ✅ Success
self.logger.info(f"✅ Weights & Biases initialized: {self.run.name}") # ✅ Success
self.logger.warning(f"⚠️ Model architecture logging failed: {e}")    # ✅ Warning
self.logger.error(f"❌ TensorBoard setup failed: {e}")               # ✅ Error

# Avoid inconsistent formatting
self.logger.info("TensorBoard initialized")                          # ❌ No emoji
self.logger.info("TensorBoard initialized successfully")             # ❌ No emoji
self.logger.warning("Model architecture logging failed")             # ❌ No emoji
```

### **Error Handling Conventions**
```python
# Always wrap experiment tracking operations in try-except blocks
def setup_tensorboard(self):
    """Setup TensorBoard with comprehensive profiling."""
    with self.code_profiler.profile_operation("tensorboard_setup", "experiment_tracking"):
        try:
            # Setup logic here
            self.logger.info(f"✅ TensorBoard initialized: {log_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ TensorBoard setup failed: {e}")
            self.writer = None  # Set to None on failure

# Provide fallback behavior
def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
    """Log training metrics to TensorBoard with profiling."""
    with self.code_profiler.profile_operation("tensorboard_training_logging", "experiment_tracking"):
        if self.writer is None:  # Check if writer is available
            return
        
        try:
            # Logging logic here
            pass
            
        except Exception as e:
            self.logger.error(f"❌ TensorBoard training logging failed: {e}")
            # Continue execution without failing the entire operation
```

## 🎯 **5. Metric Logging Conventions**

### **Metric Naming Conventions**
```python
# Use hierarchical metric naming with forward slashes
# Training metrics
"training/loss"               # ✅ Clear training loss metric
"training/accuracy"           # ✅ Clear training accuracy metric
"training/learning_rate"      # ✅ Clear learning rate metric

# Performance metrics
"performance/gpu_memory_allocated_mb"    # ✅ Clear GPU memory metric
"performance/gpu_memory_reserved_mb"     # ✅ Clear GPU memory metric
"performance/cpu_usage_percent"          # ✅ Clear CPU usage metric

# Profiling metrics
"profiling/timing/operation_name"        # ✅ Clear timing metric
"profiling/memory/operation_name"        # ✅ Clear memory metric
"profiling/bottlenecks/bottleneck_0"     # ✅ Clear bottleneck metric

# Avoid flat naming
"loss"                       # ❌ No hierarchy
"gpu_memory"                 # ❌ No hierarchy
"operation_timing"           # ❌ No hierarchy
```

### **Metric Value Validation**
```python
# Always validate metric values before logging
def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
    with self.code_profiler.profile_operation("tensorboard_training_logging", "experiment_tracking"):
        if self.writer is None:
            return
        
        try:
            # Log scalar metrics with validation
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):  # ✅ Type validation
                    if not (math.isnan(metric_value) or math.isinf(metric_value)):  # ✅ Value validation
                        self.writer.add_scalar(
                            f"training/{metric_name}",
                            metric_value,
                            global_step=step
                        )
                    else:
                        self.logger.warning(f"⚠️ Skipping invalid metric {metric_name}: {metric_value}")
                else:
                    self.logger.warning(f"⚠️ Skipping non-numeric metric {metric_name}: {type(metric_value)}")
```

## 🔄 **6. Asynchronous Logging Conventions**

### **Queue and Thread Naming**
```python
# Use descriptive names for async components
self.logging_queue = queue.Queue()                    # ✅ Clear queue name
self.logging_thread = threading.Thread(...)           # ✅ Clear thread name

# Avoid generic names
self.queue = queue.Queue()                            # ❌ Too generic
self.thread = threading.Thread(...)                   # ❌ Too generic
```

### **Worker Function Conventions**
```python
# Use descriptive worker function names
def _logging_worker(self):                            # ✅ Clear worker name
    """Background worker for experiment logging."""
    while True:
        try:
            # Get logging task from queue
            task = self.logging_queue.get(timeout=1.0)
            
            if task is None:  # Shutdown signal
                break
            
            # Execute logging task
            task_type, data = task
            if task_type == "training":
                self.experiment_tracker.log_training_metrics(**data)
            elif task_type == "performance":
                self.experiment_tracker.log_performance_metrics(**data)
            elif task_type == "profiling":
                self.experiment_tracker.log_profiling_data(**data)
            
            self.logging_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            self.logger.error(f"❌ Logging worker error: {e}")

# Avoid generic names
def worker(self):                                      # ❌ Too generic
def background_task(self):                             # ❌ Unclear purpose
```

## 📁 **7. File and Directory Conventions**

### **Log Directory Structure**
```python
# Use consistent log directory structure
log_dir = f"runs/{experiment_name}"                   # ✅ Clear structure
experiment_name = f"seo_engine_{int(time.time())}"    # ✅ Descriptive naming

# Avoid inconsistent structures
log_dir = "logs"                                       # ❌ Too generic
log_dir = "experiments"                                # ❌ Too generic
```

### **File Naming Conventions**
```python
# Use descriptive file names with timestamps
f"seo_engine_{timestamp}.log"                         # ✅ Clear log file name
f"training_progress_{timestamp}.log"                  # ✅ Clear training log name
f"errors_{timestamp}.log"                             # ✅ Clear error log name
f"performance_metrics_{timestamp}.log"                # ✅ Clear performance log name

# Avoid generic names
"log.txt"                                              # ❌ Too generic
"output.log"                                           # ❌ Too generic
"data.log"                                             # ❌ Too generic
```

## 🚀 **8. Performance Optimization Conventions**

### **Logging Interval Conventions**
```python
# Use configurable logging intervals
if step % self.config.logging_steps == 0:              # ✅ Configurable interval
    self.experiment_tracker.log_training_metrics(...)

# Avoid hardcoded intervals
if step % 10 == 0:                                     # ❌ Hardcoded interval
    self.experiment_tracker.log_training_metrics(...)
```

### **Selective Logging Conventions**
```python
# Use priority-based logging
high_priority_metrics = ['loss', 'accuracy', 'learning_rate']        # ✅ Clear priority
medium_priority_metrics = ['gradient_norm', 'weight_norm']           # ✅ Clear priority
low_priority_metrics = ['memory_usage', 'cpu_usage']                # ✅ Clear priority

def log_metrics_with_priority(self, metrics, priority='medium'):
    """Log metrics based on priority level."""
    if priority == 'high':
        # Log all high priority metrics
        self.experiment_tracker.log_training_metrics(
            epoch=self.current_epoch,
            step=self.global_step,
            metrics={k: v for k, v in metrics.items() if k in high_priority_metrics}
        )
    elif priority == 'medium':
        # Log medium and high priority metrics
        medium_high_metrics = high_priority_metrics + medium_priority_metrics
        self.experiment_tracker.log_training_metrics(
            epoch=self.current_epoch,
            step=self.global_step,
            metrics={k: v for k, v in metrics.items() if k in medium_high_metrics}
        )
    else:
        # Log all metrics
        self.experiment_tracker.log_training_metrics(
            epoch=self.current_epoch,
            step=self.global_step,
            metrics=metrics
        )
```

## 🔧 **9. Integration Conventions**

### **Unified Tracker Interface**
```python
# Use consistent interface for all trackers
class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.code_profiler = config.code_profiler
        self.tensorboard_tracker = None
        self.wandb_tracker = None
        self.setup_trackers()
    
    def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log training metrics to all enabled trackers."""
        with self.code_profiler.profile_operation("unified_training_logging", "experiment_tracking"):
            # Log to TensorBoard
            if self.tensorboard_tracker:
                self.tensorboard_tracker.log_training_metrics(epoch, step, metrics)
            
            # Log to Weights & Biases
            if self.wandb_tracker:
                self.wandb_tracker.log_training_metrics(epoch, step, metrics)
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics to all enabled trackers."""
        with self.code_profiler.profile_operation("unified_performance_logging", "experiment_tracking"):
            # Log to TensorBoard
            if self.tensorboard_tracker:
                self.tensorboard_tracker.log_performance_metrics(metrics)
            
            # Log to Weights & Biases
            if self.wandb_tracker:
                self.wandb_tracker.log_performance_metrics(metrics)
    
    def log_profiling_data(self, profiling_data: Dict[str, Any]):
        """Log profiling data to all enabled trackers."""
        with self.code_profiler.profile_operation("unified_profiling_logging", "experiment_tracking"):
            # Log to TensorBoard
            if self.tensorboard_tracker:
                self.tensorboard_tracker.log_profiling_data(profiling_data)
            
            # Log to Weights & Biases
            if self.wandb_tracker:
                self.wandb_tracker.log_profiling_data(profiling_data)
```

### **Cleanup Conventions**
```python
# Always implement proper cleanup methods
def close(self):
    """Close all experiment trackers."""
    with self.code_profiler.profile_operation("experiment_tracker_cleanup", "experiment_tracking"):
        # Close TensorBoard
        if self.tensorboard_tracker:
            self.tensorboard_tracker.close()
        
        # Close Weights & Biases
        if self.wandb_tracker:
            self.wandb_tracker.close()
        
        self.logger.info("✅ All experiment trackers closed")

# Use context manager pattern when possible
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
```

## 📚 **10. Documentation Conventions**

### **Docstring Formatting**
```python
def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
    """Log training metrics to TensorBoard with profiling.
    
    Args:
        epoch (int): Current training epoch
        step (int): Current training step
        metrics (Dict[str, float]): Dictionary of training metrics to log
        
    Returns:
        None
        
    Raises:
        None (errors are logged but don't raise exceptions)
        
    Example:
        >>> tracker.log_training_metrics(
        ...     epoch=1,
        ...     step=100,
        ...     metrics={'loss': 0.123, 'accuracy': 0.95}
        ... )
    """
    with self.code_profiler.profile_operation("tensorboard_training_logging", "experiment_tracking"):
        # Implementation here
        pass
```

### **Inline Comments**
```python
# Use clear inline comments for complex logic
def setup_tensorboard(self):
    with self.code_profiler.profile_operation("tensorboard_setup", "experiment_tracking"):
        try:
            # Create TensorBoard writer with experiment-specific log directory
            experiment_name = f"seo_engine_{int(time.time())}"
            log_dir = f"runs/{experiment_name}"
            
            self.writer = SummaryWriter(
                log_dir=log_dir,
                comment=f"SEO_Engine_{self.config.model_name}",
                max_queue=10,      # Buffer size for efficient logging
                flush_secs=30      # Flush logs every 30 seconds
            )
            
            # Log experiment configuration and model architecture
            self._log_experiment_config()
            self._log_model_architecture()
            
            self.logger.info(f"✅ TensorBoard initialized: {log_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ TensorBoard setup failed: {e}")
            self.writer = None  # Set to None to indicate failure
```

## 🎯 **11. Testing Conventions**

### **Test Naming Conventions**
```python
# Use descriptive test names that indicate what is being tested
def test_tensorboard_tracker_setup_success():          # ✅ Clear test purpose
def test_wandb_tracker_logging_failure_handling():     # ✅ Clear test purpose
def test_unified_tracker_metrics_logging():            # ✅ Clear test purpose

# Avoid generic test names
def test_setup():                                      # ❌ Too generic
def test_logging():                                    # ❌ Too generic
def test_metrics():                                    # ❌ Too generic
```

### **Mock and Fixture Conventions**
```python
# Use descriptive mock names
@patch('torch.utils.tensorboard.SummaryWriter')
def test_tensorboard_setup_success(mock_summary_writer):
    # Test implementation here
    pass

@patch('wandb.init')
def test_wandb_setup_success(mock_wandb_init):
    # Test implementation here
    pass

# Avoid generic mock names
@patch('SummaryWriter')
def test_setup(mock_writer):                           # ❌ Too generic
    pass
```

## 🔍 **12. Debugging Conventions**

### **Debug Logging**
```python
# Use debug logging for detailed troubleshooting
def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
    with self.code_profiler.profile_operation("tensorboard_training_logging", "experiment_tracking"):
        if self.writer is None:
            self.logger.debug("TensorBoard writer not available, skipping logging")
            return
        
        try:
            # Log scalar metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.logger.debug(f"Logging metric: {metric_name} = {metric_value}")
                    self.writer.add_scalar(
                        f"training/{metric_name}",
                        metric_value,
                        global_step=step
                    )
            
        except Exception as e:
            self.logger.error(f"❌ TensorBoard training logging failed: {e}")
            self.logger.debug(f"Failed metrics: {metrics}")
            self.logger.debug(f"Failed at epoch {epoch}, step {step}")
```

### **Performance Debugging**
```python
# Use timing for performance debugging
def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
    start_time = time.time()
    
    with self.code_profiler.profile_operation("tensorboard_training_logging", "experiment_tracking"):
        # Logging logic here
        pass
    
    logging_time = time.time() - start_time
    if logging_time > 0.1:  # Log if logging takes more than 100ms
        self.logger.warning(f"⚠️ Slow logging detected: {logging_time:.3f}s for {len(metrics)} metrics")
```

## 📋 **13. Summary of Key Conventions**

### **✅ DO:**
- Use descriptive class and method names with clear purpose
- Always wrap operations with profiling context managers
- Use consistent profiling categories and operation names
- Implement proper error handling with fallback behavior
- Use hierarchical metric naming with forward slashes
- Validate metric values before logging
- Use configurable parameters instead of hardcoded values
- Implement proper cleanup methods
- Use clear log messages with emojis and status indicators
- Follow consistent file and directory naming conventions

### **❌ DON'T:**
- Use abbreviations or generic names
- Skip error handling or validation
- Use hardcoded values or intervals
- Mix different naming conventions
- Skip proper cleanup
- Use inconsistent log message formatting
- Use flat metric naming without hierarchy
- Skip configuration validation

### **🔧 ALWAYS:**
- Profile all experiment tracking operations
- Handle errors gracefully without failing the entire operation
- Validate configuration parameters
- Use consistent naming conventions
- Implement proper cleanup
- Document complex logic with clear comments
- Test error handling and edge cases
- Monitor performance and log slow operations

## 📚 **Related Documentation**

- **Detailed Integration**: See `EXPERIMENT_TRACKING_PROFILING_INTEGRATION.md`
- **Summary**: See `EXPERIMENT_TRACKING_SUMMARY.md`
- **Configuration Guide**: See `README.md` - Experiment Tracking section
- **Dependencies Overview**: See `DEPENDENCIES.md`
- **Performance Analysis**: See `code_profiling_summary.md`






