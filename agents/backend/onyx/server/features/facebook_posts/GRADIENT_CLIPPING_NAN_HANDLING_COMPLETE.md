# Gradient Clipping and NaN/Inf Value Handling System

## Overview

This document provides a comprehensive overview of the gradient clipping and NaN/Inf value handling system that implements advanced numerical stability techniques for deep learning applications, now enhanced with **PyTorch built-in debugging tools** and **enterprise-grade performance optimization**.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Gradient Clipping](#gradient-clipping)
3. [NaN/Inf Handling](#naninf-handling)
4. [Numerical Stability Manager](#numerical-stability-manager)
5. [PyTorch Debugging Integration](#pytorch-debugging-integration)
6. [Performance Optimization Integration](#performance-optimization-integration)
7. [Configuration Options](#configuration-options)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [Monitoring and Visualization](#monitoring-and-visualization)

## System Architecture

### Core Components

The gradient clipping and NaN handling system consists of several key components:

```python
class ClippingType(Enum):
    """Types of gradient clipping."""
    NORM = "norm"  # L2 norm clipping
    VALUE = "value"  # Value clipping
    GLOBAL_NORM = "global_norm"  # Global norm clipping
    ADAPTIVE = "adaptive"  # Adaptive clipping


class NaNHandlingType(Enum):
    """Types of NaN/Inf handling."""
    DETECT = "detect"  # Detect and log
    REPLACE = "replace"  # Replace with safe values
    SKIP = "skip"  # Skip update
    RESTORE = "restore"  # Restore from checkpoint
    GRADIENT_ZEROING = "gradient_zeroing"  # Zero gradients


class PyTorchDebuggingConfig:
    """Configuration for PyTorch built-in debugging tools."""
    enable_autograd_anomaly: bool = False      # Autograd anomaly detection
    enable_grad_check: bool = False            # Gradient numerical checking
    enable_memory_debugging: bool = False      # Memory usage monitoring
    enable_cuda_debugging: bool = False        # CUDA operation debugging
    enable_performance_debugging: bool = False # Performance profiling


class PerformanceOptimizationConfig:
    """Configuration for performance optimization integration."""
    enable_performance_optimization: bool = True    # Enable performance optimization
    optimization_level: str = "advanced"           # "basic", "advanced", "ultra"
    integrate_with_stability: bool = True          # Integrate with stability framework
    enable_mixed_precision: bool = True            # Enable mixed precision training
    enable_model_compilation: bool = True          # Enable PyTorch compilation
    enable_memory_optimization: bool = True        # Enable memory optimization
```

### Configuration Classes

```python
@dataclass
class GradientClippingConfig:
    """Configuration for gradient clipping."""
    # Clipping type
    clipping_type: ClippingType = ClippingType.NORM
    
    # Clipping parameters
    max_norm: float = 1.0
    max_value: float = 1.0
    clip_ratio: float = 0.1
    
    # Adaptive clipping
    adaptive_threshold: float = 0.1
    adaptive_factor: float = 2.0
    
    # Monitoring
    monitor_clipping: bool = True
    log_clipping_stats: bool = True
    save_clipping_history: bool = True
    clipping_history_file: str = "gradient_clipping_history.json"
    
    # Advanced settings
    clip_grad_norm: bool = True
    clip_grad_value: bool = False
    use_global_norm: bool = False
    
    # Performance
    efficient_clipping: bool = True
    parallel_clipping: bool = False


@dataclass
class NaNHandlingConfig:
    """Configuration for NaN/Inf handling."""
    # Handling type
    handling_type: NaNHandlingType = NaNHandlingType.DETECT
    
    # Detection settings
    detect_nan: bool = True
    detect_inf: bool = True
    detect_overflow: bool = True
    
    # Replacement values
    nan_replacement: float = 0.0
    inf_replacement: float = 1e6
    overflow_replacement: float = 1e6
    
    # Thresholds
    nan_threshold: float = 1e-6
    inf_threshold: float = 1e6
    overflow_threshold: float = 1e6
    
    # Monitoring
    monitor_nan: bool = True
    log_nan_stats: bool = True
    save_nan_history: bool = True
    nan_history_file: str = "nan_handling_history.json"
    
    # Advanced settings
    gradient_zeroing: bool = True
    parameter_checking: bool = True
    loss_checking: bool = True
    
    # Performance
    efficient_detection: bool = True
    batch_detection: bool = True
```

## Gradient Clipping

### Core Gradient Clipping Implementation

```python
class GradientClipper:
    """Advanced gradient clipping implementation."""
    
    def __init__(self, config: GradientClippingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Clipping history
        self.clipping_history = {
            'steps': [],
            'clipping_ratios': [],
            'gradient_norms': [],
            'clipped_norms': [],
            'clipping_types': []
        }
        
        # Current step
        self.current_step = 0
```

### Clipping Methods

#### **1. Norm Clipping**
```python
def _clip_grad_norm(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
    """Clip gradients by norm."""
    # Get all gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.data.view(-1))
    
    if not gradients:
        return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
    
    # Concatenate gradients
    all_gradients = torch.cat(gradients)
    
    # Calculate norm
    total_norm = all_gradients.norm(2)
    
    # Clip if necessary
    clip_coef = self.config.max_norm / (total_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)
    
    # Apply clipping
    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    # Update history
    self._update_clipping_history(total_norm.item(), total_norm.item() * clip_coef, 'norm')
    
    return {
        'clipping_ratio': 1.0 - clip_coef,
        'gradient_norm': total_norm.item(),
        'clipped_norm': total_norm.item() * clip_coef
    }
```

#### **2. Value Clipping**
```python
def _clip_grad_value(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
    """Clip gradients by value."""
    total_norm = 0.0
    clipped_norm = 0.0
    clipped_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            # Calculate norm before clipping
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # Clip gradients
            param.grad.data.clamp_(-self.config.max_value, self.config.max_value)
            
            # Calculate norm after clipping
            clipped_param_norm = param.grad.data.norm(2)
            clipped_norm += clipped_param_norm.item() ** 2
            
            if param_norm.item() > self.config.max_value:
                clipped_count += 1
    
    total_norm = math.sqrt(total_norm)
    clipped_norm = math.sqrt(clipped_norm)
    
    # Update history
    self._update_clipping_history(total_norm, clipped_norm, 'value')
    
    return {
        'clipping_ratio': clipped_count / len(list(model.parameters())),
        'gradient_norm': total_norm,
        'clipped_norm': clipped_norm
    }
```

#### **3. Global Norm Clipping**
```python
def _clip_global_norm(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
    """Clip gradients using global norm."""
    # Get all gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.data.view(-1))
    
    if not gradients:
        return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
    
    # Concatenate gradients
    all_gradients = torch.cat(gradients)
    
    # Calculate global norm
    global_norm = all_gradients.norm(2)
    
    # Clip if necessary
    clip_coef = self.config.max_norm / (global_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)
    
    # Apply clipping
    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    # Update history
    self._update_clipping_history(global_norm.item(), global_norm.item() * clip_coef, 'global_norm')
    
    return {
        'clipping_ratio': 1.0 - clip_coef,
        'gradient_norm': global_norm.item(),
        'clipped_norm': global_norm.item() * clip_coef
    }
```

#### **4. Adaptive Clipping**
```python
def _clip_adaptive(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
    """Adaptive gradient clipping."""
    # Get all gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.data.view(-1))
    
    if not gradients:
        return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
    
    # Concatenate gradients
    all_gradients = torch.cat(gradients)
    
    # Calculate norm
    total_norm = all_gradients.norm(2)
    
    # Adaptive threshold
    adaptive_threshold = self.config.adaptive_threshold
    if len(self.clipping_history['gradient_norms']) > 0:
        avg_norm = np.mean(self.clipping_history['gradient_norms'][-10:])
        adaptive_threshold = avg_norm * self.config.adaptive_factor
    
    # Clip if necessary
    clip_coef = adaptive_threshold / (total_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)
    
    # Apply clipping
    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    # Update history
    self._update_clipping_history(total_norm.item(), total_norm.item() * clip_coef, 'adaptive')
    
    return {
        'clipping_ratio': 1.0 - clip_coef,
        'gradient_norm': total_norm.item(),
        'clipped_norm': total_norm.item() * clip_coef,
        'adaptive_threshold': adaptive_threshold
    }
```

### Clipping History Management

```python
def _update_clipping_history(self, gradient_norm: float, clipped_norm: float, clipping_type: str):
    """Update clipping history."""
    self.clipping_history['steps'].append(self.current_step)
    self.clipping_history['gradient_norms'].append(gradient_norm)
    self.clipping_history['clipped_norms'].append(clipped_norm)
    self.clipping_history['clipping_types'].append(clipping_type)
    
    # Calculate clipping ratio
    if gradient_norm > 0:
        clipping_ratio = 1.0 - (clipped_norm / gradient_norm)
    else:
        clipping_ratio = 0.0
    
    self.clipping_history['clipping_ratios'].append(clipping_ratio)
    
    # Log if enabled
    if self.config.log_clipping_stats:
        self.logger.info(f"Step {self.current_step}: Gradient norm = {gradient_norm:.6f}, "
                       f"Clipped norm = {clipped_norm:.6f}, Clipping ratio = {clipping_ratio:.4f}")
```

## NaN/Inf Handling

### Core NaN Handling Implementation

```python
class NaNHandler:
    """Advanced NaN/Inf value handling."""
    
    def __init__(self, config: NaNHandlingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # NaN history
        self.nan_history = {
            'steps': [],
            'nan_counts': [],
            'inf_counts': [],
            'overflow_counts': [],
            'handling_actions': []
        }
        
        # Current step
        self.current_step = 0
        
        # Statistics
        self.total_nan_detected = 0
        self.total_inf_detected = 0
        self.total_overflow_detected = 0
```

### Detection Methods

#### **1. Tensor Checking**
```python
def _check_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[bool, bool, bool]:
    """Check tensor for NaN/Inf values."""
    nan_detected = torch.isnan(tensor).any().item()
    inf_detected = torch.isinf(tensor).any().item()
    overflow_detected = (tensor.abs() > self.config.overflow_threshold).any().item()
    
    if nan_detected or inf_detected or overflow_detected:
        if self.config.log_nan_stats:
            self.logger.warning(f"Step {self.current_step}: {name} has "
                              f"NaN={nan_detected}, Inf={inf_detected}, Overflow={overflow_detected}")
    
    return nan_detected, inf_detected, overflow_detected
```

#### **2. Parameter Checking**
```python
def _check_parameters(self, model: nn.Module) -> Tuple[bool, bool, bool]:
    """Check model parameters for NaN/Inf values."""
    nan_detected = False
    inf_detected = False
    overflow_detected = False
    
    for name, param in model.named_parameters():
        param_nan, param_inf, param_overflow = self._check_tensor(param.data, f"parameter {name}")
        nan_detected |= param_nan
        inf_detected |= param_inf
        overflow_detected |= param_overflow
    
    return nan_detected, inf_detected, overflow_detected
```

#### **3. Gradient Checking**
```python
def _check_gradients(self, model: nn.Module) -> Tuple[bool, bool, bool]:
    """Check model gradients for NaN/Inf values."""
    nan_detected = False
    inf_detected = False
    overflow_detected = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_nan, grad_inf, grad_overflow = self._check_tensor(param.grad.data, f"gradient {name}")
            nan_detected |= grad_nan
            inf_detected |= grad_inf
            overflow_detected |= grad_overflow
    
    return nan_detected, inf_detected, overflow_detected
```

### Handling Methods

#### **1. Replace Numerical Issues**
```python
def _replace_numerical_issues(self, model: nn.Module):
    """Replace NaN/Inf values with safe values."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Replace NaN gradients
            if torch.isnan(param.grad.data).any():
                param.grad.data = torch.where(torch.isnan(param.grad.data),
                                            torch.tensor(self.config.nan_replacement),
                                            param.grad.data)
            
            # Replace Inf gradients
            if torch.isinf(param.grad.data).any():
                param.grad.data = torch.where(torch.isinf(param.grad.data),
                                            torch.tensor(self.config.inf_replacement),
                                            param.grad.data)
            
            # Replace overflow gradients
            overflow_mask = param.grad.data.abs() > self.config.overflow_threshold
            if overflow_mask.any():
                param.grad.data = torch.where(overflow_mask,
                                            torch.tensor(self.config.overflow_replacement),
                                            param.grad.data)
```

#### **2. Zero Gradients**
```python
def _zero_gradients(self, model: nn.Module):
    """Zero gradients with numerical issues."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Zero gradients with NaN/Inf
            nan_inf_mask = torch.isnan(param.grad.data) | torch.isinf(param.grad.data)
            if nan_inf_mask.any():
                param.grad.data[nan_inf_mask] = 0.0
```

#### **3. Main Handling Method**
```python
def check_and_handle(self, model: nn.Module, loss: torch.Tensor, 
                     optimizer: Optimizer) -> Dict[str, Any]:
    """Check for NaN/Inf values and handle them."""
    self.current_step += 1
    
    nan_detected = False
    inf_detected = False
    overflow_detected = False
    handling_action = "none"
    
    # Check loss
    if self.config.loss_checking:
        loss_nan, loss_inf, loss_overflow = self._check_tensor(loss, "loss")
        nan_detected |= loss_nan
        inf_detected |= loss_inf
        overflow_detected |= loss_overflow
    
    # Check parameters
    if self.config.parameter_checking:
        param_nan, param_inf, param_overflow = self._check_parameters(model)
        nan_detected |= param_nan
        inf_detected |= param_inf
        overflow_detected |= param_overflow
    
    # Check gradients
    grad_nan, grad_inf, grad_overflow = self._check_gradients(model)
    nan_detected |= grad_nan
    inf_detected |= grad_inf
    overflow_detected |= grad_overflow
    
    # Handle if detected
    if nan_detected or inf_detected or overflow_detected:
        handling_action = self._handle_numerical_issues(model, optimizer, 
                                                      nan_detected, inf_detected, overflow_detected)
    
    # Update statistics
    if nan_detected:
        self.total_nan_detected += 1
    if inf_detected:
        self.total_inf_detected += 1
    if overflow_detected:
        self.total_overflow_detected += 1
    
    # Update history
    self._update_nan_history(nan_detected, inf_detected, overflow_detected, handling_action)
    
    return {
        'nan_detected': nan_detected,
        'inf_detected': inf_detected,
        'overflow_detected': overflow_detected,
        'handling_action': handling_action,
        'total_nan': self.total_nan_detected,
        'total_inf': self.total_inf_detected,
        'total_overflow': self.total_overflow_detected
    }
```

## Numerical Stability Manager

### Core Manager Implementation

```python
class NumericalStabilityManager:
    """Comprehensive numerical stability manager."""
    
    def __init__(self, clipping_config: GradientClippingConfig, 
                 nan_config: NaNHandlingConfig):
        self.clipping_config = clipping_config
        self.nan_config = nan_config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.gradient_clipper = GradientClipper(clipping_config)
        self.nan_handler = NaNHandler(nan_config)
        
        # Training state
        self.current_step = 0
        self.stability_history = {
            'steps': [],
            'clipping_stats': [],
            'nan_stats': [],
            'stability_scores': []
        }
```

### Main Step Method

```python
def step(self, model: nn.Module, loss: torch.Tensor, 
         optimizer: Optimizer) -> Dict[str, Any]:
    """Perform one step with numerical stability checks."""
    self.current_step += 1
    
    # Check for NaN/Inf values
    nan_stats = self.nan_handler.check_and_handle(model, loss, optimizer)
    
    # Apply gradient clipping
    clipping_stats = self.gradient_clipper.clip_gradients(model, optimizer)
    
    # Calculate stability score
    stability_score = self._calculate_stability_score(nan_stats, clipping_stats)
    
    # Update history
    self._update_stability_history(nan_stats, clipping_stats, stability_score)
    
    # Log if issues detected
    if nan_stats['nan_detected'] or nan_stats['inf_detected'] or nan_stats['overflow_detected']:
        self.logger.warning(f"Step {self.current_step}: Numerical issues detected - "
                          f"NaN: {nan_stats['nan_detected']}, "
                          f"Inf: {nan_stats['inf_detected']}, "
                          f"Overflow: {nan_stats['overflow_detected']}")
    
    return {
        'nan_stats': nan_stats,
        'clipping_stats': clipping_stats,
        'stability_score': stability_score
    }
```

### Stability Score Calculation

```python
def _calculate_stability_score(self, nan_stats: Dict[str, Any], 
                              clipping_stats: Dict[str, float]) -> float:
    """Calculate numerical stability score."""
    score = 1.0
    
    # Penalize for numerical issues
    if nan_stats['nan_detected']:
        score -= 0.3
    if nan_stats['inf_detected']:
        score -= 0.2
    if nan_stats['overflow_detected']:
        score -= 0.1
    
    # Penalize for excessive clipping
    if 'clipping_ratio' in clipping_stats:
        score -= clipping_stats['clipping_ratio'] * 0.1
    
    return max(0.0, score)
```

## PyTorch Debugging Integration

### Overview

The system now includes comprehensive integration with PyTorch's built-in debugging tools, providing advanced debugging capabilities for deep learning training:

- **Autograd Anomaly Detection**: Automatic detection of gradient computation anomalies
- **Gradient Checking**: Numerical validation of gradient computations
- **Memory Debugging**: CUDA and CPU memory usage monitoring
- **Performance Profiling**: Training performance analysis and optimization
- **Enhanced Logging**: Detailed debugging information with centralized logging

### PyTorchDebuggingManager

```python
class PyTorchDebuggingManager:
    """Manager for PyTorch built-in debugging tools."""
    
    def __init__(self, config: PyTorchDebuggingConfig):
        self.config = config
        self.logger = get_logger('pytorch_debugging')
        
        # Debug state
        self.debug_enabled = False
        self.anomaly_detector_enabled = False
        self.grad_check_enabled = False
        self.memory_debug_enabled = False
        
        # Setup debugging if enabled
        if config.enable_autograd_anomaly or config.enable_grad_check or config.enable_memory_debugging:
            self._setup_debugging()
```

### Key Debugging Features

#### **1. Autograd Anomaly Detection**
```python
def _enable_autograd_anomaly(self):
    """Enable autograd anomaly detection."""
    if self.config.autograd_anomaly_mode == "detect":
        torch.autograd.set_detect_anomaly(True)
        self.anomaly_detector_enabled = True
    elif self.config.autograd_anomaly_mode == "raise":
        torch.autograd.set_detect_anomaly(True)
        self.anomaly_detector_enabled = True
```

**Modes**:
- **`detect`**: Log anomalies without stopping execution
- **`raise`**: Stop execution when anomalies are detected

#### **2. Gradient Checking**
```python
def _enable_grad_check(self):
    """Enable gradient checking."""
    if self.config.grad_check_numerical:
        torch.autograd.gradcheck.enable()
        self.grad_check_enabled = True
    
    if self.config.grad_check_sparse_numerical:
        torch.autograd.gradcheck.enable_sparse_numerical()
```

**Features**:
- Numerical gradient validation
- Sparse gradient handling
- Automatic validation during training

#### **3. Memory Debugging**
```python
def _check_memory_usage(self) -> Dict[str, Any]:
    """Check memory usage and CUDA memory."""
    memory_info = {
        "cpu_memory": {},
        "cuda_memory": {}
    }
    
    # CPU memory info
    import psutil
    process = psutil.Process()
    memory_info["cpu_memory"] = {
        "rss": process.memory_info().rss / 1024 / 1024,  # MB
        "vms": process.memory_info().vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }
    
    # CUDA memory info
    if torch.cuda.is_available():
        memory_info["cuda_memory"] = {
            "allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
            "cached": torch.cuda.memory_reserved() / 1024 / 1024,  # MB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,  # MB
            "device_count": torch.cuda.device_count()
        }
    
    return memory_info
```

### Enhanced NumericalStabilityManager

The `NumericalStabilityManager` now integrates PyTorch debugging:

```python
class NumericalStabilityManager:
    def __init__(self, clipping_config: GradientClippingConfig, 
                 nan_config: NaNHandlingConfig,
                 debug_config: PyTorchDebuggingConfig = None):
        # ... existing initialization ...
        
        # Initialize PyTorch debugging manager if config provided
        self.debug_manager = None
        if debug_config is not None:
            self.debug_manager = PyTorchDebuggingManager(debug_config)
            self.logger.info("PyTorch debugging manager initialized")
        
        # Enhanced stability history
        self.stability_history = {
            'steps': [],
            'clipping_stats': [],
            'nan_stats': [],
            'stability_scores': [],
            'debug_info': []  # New: debug information
        }
    
    def step(self, model: nn.Module, loss: torch.Tensor, 
             optimizer: Optimizer) -> Dict[str, Any]:
        """Perform one step with numerical stability checks and debugging."""
        # ... existing stability checks ...
        
        # Check debugging status if enabled
        debug_info = {}
        if self.debug_manager is not None:
            debug_info = self.debug_manager.check_debug_status(model, loss, optimizer)
            
            # Log debug information
            if debug_info.get('anomalies_detected', False):
                self.logger.warning(f"Debug anomalies detected in step {self.current_step}: {debug_info.get('anomaly_details', [])}")
            
            # Update debug history
            self.stability_history['debug_info'].append(debug_info)
        
        # ... rest of the method ...
        
        return {
            'nan_stats': nan_stats,
            'clipping_stats': clipping_stats,
            'stability_score': stability_score,
            'debug_info': debug_info  # New: include debug information
        }
```

### Debug Session Management

```python
def start_debug_session(self, session_name: str = None):
    """Start a PyTorch debugging session if enabled."""
    if self.debug_manager is not None:
        self.debug_manager.start_debug_session(session_name)
        self.logger.info(f"Debug session started: {session_name}")

def stop_debug_session(self):
    """Stop the current PyTorch debugging session if active."""
    if self.debug_manager is not None:
        self.debug_manager.stop_debug_session()
        self.logger.info("Debug session stopped")

def get_debug_summary(self) -> Dict[str, Any]:
    """Get a summary of debugging information if available."""
    if self.debug_manager is not None:
        return self.debug_manager.get_debug_summary()
    else:
        return {"debug_enabled": False, "message": "Debug manager not initialized"}
```

### Performance Optimization Integration

The `NumericalStabilityManager` now also integrates performance optimization:

```python
class NumericalStabilityManager:
    def __init__(self, clipping_config: GradientClippingConfig, 
                 nan_config: NaNHandlingConfig,
                 debug_config: PyTorchDebuggingConfig = None,
                 performance_config: PerformanceOptimizationConfig = None):  # New parameter
        # ... existing initialization ...
        
        # Initialize performance optimization if available
        self.performance_optimizer = None
        if (self.performance_config.enable_performance_optimization and 
            PERFORMANCE_OPTIMIZATION_AVAILABLE):
            self._setup_performance_optimization()
        
        # Enhanced stability history with performance metrics
        self.stability_history = {
            'steps': [],
            'clipping_stats': [],
            'nan_stats': [],
            'stability_scores': [],
            'debug_info': [],
            'performance_metrics': []  # New: performance metrics
        }
    
    def _setup_performance_optimization(self):
        """Setup performance optimization system."""
        # Creates PerformanceConfig based on optimization level
        # Initializes PerformanceOptimizer with appropriate settings
    
    def step(self, model: nn.Module, loss: torch.Tensor, 
             optimizer: Optimizer) -> Dict[str, Any]:
        """Perform one step with numerical stability checks, debugging, and performance optimization."""
        # ... existing stability and debug checks ...
        
        # Apply performance optimizations if enabled
        performance_metrics = {}
        if self.performance_optimizer is not None:
            # Monitor memory usage and record performance metrics
            memory_info = self.performance_optimizer.memory_manager.monitor_memory(self.current_step)
            # Store performance metrics in stability history
        
        # ... rest of the method ...
        
        return {
            'nan_stats': nan_stats,
            'clipping_stats': clipping_stats,
            'stability_score': stability_score,
            'debug_info': debug_info,
            'performance_metrics': performance_metrics  # New: include performance metrics
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary if available."""
        if self.performance_optimizer is not None:
            return self.performance_optimizer.get_optimization_status()
        else:
            return {"performance_optimization": False, "message": "Performance optimizer not initialized"}
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply performance optimizations to the model if available."""
        if self.performance_optimizer is not None:
            try:
                optimized_model = self.performance_optimizer.model_optimizer.optimize_model(model)
                self.logger.info("Model performance optimizations applied")
                return optimized_model
            except Exception as e:
                self.logger.warning(f"Model performance optimization failed: {e}")
                return model
        else:
            return model
```

### Enhanced Visualization

The plotting system now includes debug information:

```python
def plot_stability_history(self, save_path: Optional[str] = None):
    """Plot numerical stability history with debug information."""
    # Create subplots - add one more if debug info is available
    debug_available = any(info.get('debug_enabled', False) for info in self.stability_history['debug_info'])
    num_plots = 3 if debug_available else 2
    fig, axes = plt.subplots(2, num_plots, figsize=(15, 10))
    
    # ... existing plots ...
    
    # Plot debug anomalies if available
    if debug_available:
        debug_anomalies = [info.get('anomalies_detected', False) for info in self.stability_history['debug_info']]
        anomaly_steps = [step for step, anomaly in zip(self.stability_history['steps'], debug_anomalies) if anomaly]
        anomaly_values = [1] * len(anomaly_steps)
        
        axes[0, 2].scatter(anomaly_steps, anomaly_values, color='red', s=100, alpha=0.7, label='Debug Anomalies')
        axes[0, 2].set_title('PyTorch Debug Anomalies')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Anomaly Detected')
        axes[0, 2].set_ylim(0, 2)
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # Plot debug information if available
    if debug_available:
        # Plot gradient statistics over time
        grad_norms = []
        for info in self.stability_history['debug_info']:
            if info.get('gradient_stats', {}).get('total_norm'):
                grad_norms.append(info['gradient_stats']['total_norm'])
            else:
                grad_norms.append(0.0)
        
        axes[1, 2].plot(self.stability_history['steps'], grad_norms, 
                        label='Total Gradient Norm', color='purple', alpha=0.7)
        axes[1, 2].set_title('Gradient Norms Over Time (Debug)')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
```

## Configuration Options

### Gradient Clipping Configuration

#### **1. Clipping Types**
- `NORM`: L2 norm clipping (default)
- `VALUE`: Value clipping
- `GLOBAL_NORM`: Global norm clipping
- `ADAPTIVE`: Adaptive clipping

#### **2. Clipping Parameters**
- `max_norm`: Maximum gradient norm (default: 1.0)
- `max_value`: Maximum gradient value (default: 1.0)
- `clip_ratio`: Clipping ratio threshold (default: 0.1)

#### **3. Adaptive Clipping**
- `adaptive_threshold`: Adaptive threshold (default: 0.1)
- `adaptive_factor`: Adaptive factor (default: 2.0)

#### **4. Monitoring**
- `monitor_clipping`: Enable clipping monitoring (default: True)
- `log_clipping_stats`: Log clipping statistics (default: True)
- `save_clipping_history`: Save clipping history (default: True)

### NaN Handling Configuration

#### **1. Handling Types**
- `DETECT`: Detect and log (default)
- `REPLACE`: Replace with safe values
- `SKIP`: Skip update
- `RESTORE`: Restore from checkpoint
- `GRADIENT_ZEROING`: Zero gradients

#### **2. Detection Settings**
- `detect_nan`: Detect NaN values (default: True)
- `detect_inf`: Detect Inf values (default: True)
- `detect_overflow`: Detect overflow values (default: True)

#### **3. Replacement Values**
- `nan_replacement`: NaN replacement value (default: 0.0)
- `inf_replacement`: Inf replacement value (default: 1e6)
- `overflow_replacement`: Overflow replacement value (default: 1e6)

#### **4. Thresholds**
- `nan_threshold`: NaN detection threshold (default: 1e-6)
- `inf_threshold`: Inf detection threshold (default: 1e6)
- `overflow_threshold`: Overflow detection threshold (default: 1e6)

### PyTorch Debugging Configuration

#### **1. Autograd Anomaly Detection**
- `enable_autograd_anomaly`: Enable autograd anomaly detection (default: False)
- `autograd_anomaly_mode`: Detection mode - "detect" or "raise" (default: "detect")

#### **2. Gradient Checking**
- `enable_grad_check`: Enable gradient checking (default: False)
- `grad_check_numerical`: Enable numerical gradient checking (default: True)
- `grad_check_sparse_numerical`: Enable sparse gradient checking (default: True)

#### **3. Memory Debugging**
- `enable_memory_debugging`: Enable memory debugging (default: False)
- `memory_tracking`: Enable memory tracking (default: False)
- `memory_profiling`: Enable memory profiling (default: False)

#### **4. CUDA Debugging**
- `enable_cuda_debugging`: Enable CUDA debugging (default: False)
- `cuda_synchronize`: Enable CUDA synchronization (default: False)
- `cuda_memory_fraction`: CUDA memory fraction limit (default: 1.0)

#### **5. Performance Debugging**
- `enable_performance_debugging`: Enable performance debugging (default: False)
- `profile_autograd`: Profile autograd operations (default: False)
- `profile_memory`: Profile memory usage (default: False)

#### **6. Debug Settings**
- `debug_level`: Debug level - "info", "warning", "error" (default: "info")
- `verbose_logging`: Enable verbose logging (default: False)
- `max_debug_iterations`: Maximum debug iterations (default: 1000)
- `debug_timeout`: Debug session timeout in seconds (default: 300.0)

#### **7. Output Settings**
- `save_debug_info`: Save debug information to files (default: True)
- `debug_output_dir`: Debug output directory (default: "debug_output")
- `debug_file_prefix`: Debug file prefix (default: "pytorch_debug")

## Usage Examples

### Basic Gradient Clipping

```python
# Create gradient clipping configuration
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.NORM,
    max_norm=1.0,
    monitor_clipping=True,
    log_clipping_stats=True
)

# Create gradient clipper
clipper = GradientClipper(clipping_config)

# Use in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping
        clipping_stats = clipper.clip_gradients(model, optimizer)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Gradient norm: {clipping_stats['gradient_norm']:.6f}")
        print(f"Clipping ratio: {clipping_stats['clipping_ratio']:.4f}")
```

### Basic NaN Handling

```python
# Create NaN handling configuration
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.DETECT,
    detect_nan=True,
    detect_inf=True,
    detect_overflow=True,
    monitor_nan=True,
    log_nan_stats=True
)

# Create NaN handler
nan_handler = NaNHandler(nan_config)

# Use in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check for NaN/Inf
        nan_stats = nan_handler.check_and_handle(model, loss, optimizer)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        if nan_stats['nan_detected'] or nan_stats['inf_detected']:
            print(f"Numerical issues detected: NaN={nan_stats['nan_detected']}, "
                  f"Inf={nan_stats['inf_detected']}")
```

### Complete Numerical Stability Management

```python
# Create configurations
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.ADAPTIVE,
    max_norm=1.0,
    adaptive_threshold=0.1,
    adaptive_factor=2.0,
    monitor_clipping=True
)

nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.REPLACE,
    detect_nan=True,
    detect_inf=True,
    detect_overflow=True,
    nan_replacement=0.0,
    inf_replacement=1e6,
    monitor_nan=True
)

# Create stability manager
stability_manager = NumericalStabilityManager(clipping_config, nan_config)

# Use in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply numerical stability measures
        stability_result = stability_manager.step(model, loss, optimizer)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Log stability information
        print(f"Stability score: {stability_result['stability_score']:.4f}")
        print(f"Clipping ratio: {stability_result['clipping_stats']['clipping_ratio']:.4f}")
        print(f"NaN detected: {stability_result['nan_stats']['nan_detected']}")
```

### PyTorch Debugging Integration

```python
# Create PyTorch debugging configuration
debug_config = PyTorchDebuggingConfig(
    enable_autograd_anomaly=True,
    autograd_anomaly_mode="detect",
    enable_grad_check=True,
    enable_memory_debugging=True,
    enable_cuda_debugging=True,
    cuda_memory_fraction=0.8,
    debug_level="info",
    verbose_logging=True,
    save_debug_info=True
)

# Create stability manager with debugging
stability_manager = NumericalStabilityManager(
    clipping_config=clipping_config,
    nan_config=nan_config,
    debug_config=debug_config
)

# Start debugging session
stability_manager.start_debug_session("production_training_session")

# Use in training loop with debugging
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply numerical stability measures with debugging
        stability_result = stability_manager.step(model, loss, optimizer)
        
        # Check debug information
        debug_info = stability_result.get('debug_info', {})
        if debug_info.get('anomalies_detected', False):
            print(f"Debug anomalies detected: {debug_info['anomaly_details']}")
        
        if debug_info.get('gradient_stats'):
            grad_stats = debug_info['gradient_stats']
            print(f"Gradient norm: {grad_stats.get('total_norm', 0.0):.4f}")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

# Stop debugging session
stability_manager.stop_debug_session()

# Get debug summary
debug_summary = stability_manager.get_debug_summary()
print(f"Debug session completed:")
print(f"  Total iterations: {debug_summary['total_iterations']}")
print(f"  Anomalies detected: {debug_summary['anomalies_detected']}")
print(f"  Gradient checks: {debug_summary['gradient_checks']}")
print(f"  Memory checks: {debug_summary['memory_checks']}")
```

### Training Wrapper with Debugging

```python
# Create training wrapper with debugging
wrapper = create_training_wrapper(
    clipping_config=clipping_config,
    nan_config=nan_config,
    debug_config=debug_config
)

# Use in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply stability measures with automatic debugging
        stability_result = wrapper(model, loss, optimizer)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

# Get debug summary and stop debugging
debug_summary = wrapper.get_debug_summary()
wrapper.stop_debug_session()
```

### Different Clipping Strategies

#### **1. Norm Clipping**
```python
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.NORM,
    max_norm=1.0
)
```

#### **2. Value Clipping**
```python
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.VALUE,
    max_value=1.0
)
```

#### **3. Adaptive Clipping**
```python
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.ADAPTIVE,
    adaptive_threshold=0.1,
    adaptive_factor=2.0
)
```

### Different NaN Handling Strategies

#### **1. Detect Only**
```python
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.DETECT,
    detect_nan=True,
    detect_inf=True
)
```

#### **2. Replace Values**
```python
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.REPLACE,
    detect_nan=True,
    detect_inf=True,
    nan_replacement=0.0,
    inf_replacement=1e6
)
```

#### **3. Skip Updates**
```python
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.SKIP,
    detect_nan=True,
    detect_inf=True
)
```

#### **4. Zero Gradients**
```python
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.GRADIENT_ZEROING,
    detect_nan=True,
    detect_inf=True,
    gradient_zeroing=True
)
```

## Best Practices

### 1. Gradient Clipping Guidelines

#### **Clipping Type Selection**
- **Norm Clipping**: Good for most cases, prevents gradient explosion
- **Value Clipping**: Good for specific value constraints
- **Adaptive Clipping**: Good for dynamic learning rate scenarios
- **Global Norm**: Good for distributed training

#### **Parameter Tuning**
- **max_norm**: 0.1-10.0 depending on model size and task
- **max_value**: 0.1-1.0 for value clipping
- **adaptive_threshold**: 0.1-1.0 for adaptive clipping
- **adaptive_factor**: 1.5-3.0 for adaptive clipping

#### **Monitoring**
- Monitor clipping ratios over time
- Log gradient norms for analysis
- Plot clipping history for insights

### 2. NaN Handling Guidelines

#### **Detection Strategy**
- Always enable NaN/Inf detection
- Use appropriate thresholds for your data
- Monitor detection frequency

#### **Handling Strategy**
- **DETECT**: Use during development and debugging
- **REPLACE**: Use for production with careful value selection
- **SKIP**: Use when numerical issues are rare
- **GRADIENT_ZEROING**: Use when you want to continue training

#### **Replacement Values**
- **NaN replacement**: 0.0 or small positive value
- **Inf replacement**: Large finite value (1e6)
- **Overflow replacement**: Large finite value (1e6)

### 3. Integration Guidelines

#### **Training Loop Integration**
```python
# Initialize stability manager
stability_manager = NumericalStabilityManager(clipping_config, nan_config)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply numerical stability measures
        stability_result = stability_manager.step(model, loss, optimizer)
        
        # Check for critical issues
        if stability_result['stability_score'] < 0.5:
            print(f"Warning: Low stability score: {stability_result['stability_score']:.4f}")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
```

#### **Error Handling**
```python
try:
    stability_result = stability_manager.step(model, loss, optimizer)
except Exception as e:
    print(f"Numerical stability error: {e}")
    # Handle error appropriately
    optimizer.zero_grad()
    continue
```

### 4. Performance Considerations

#### **Efficient Implementation**
- Use vectorized operations where possible
- Enable efficient detection and clipping
- Use batch processing for large models

#### **Memory Management**
- Clean up temporary tensors
- Use in-place operations when possible
- Monitor memory usage during clipping

#### **Computational Overhead**
- Profile clipping and detection overhead
- Use appropriate monitoring frequency
- Balance safety with performance

## Monitoring and Visualization

### 1. Clipping History Visualization

```python
def plot_clipping_history(self, save_path: Optional[str] = None):
    """Plot gradient clipping history."""
    if not self.clipping_history['steps']:
        self.logger.warning("No clipping history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot gradient norms
    axes[0, 0].plot(self.clipping_history['steps'], self.clipping_history['gradient_norms'], 
                    label='Gradient Norm', alpha=0.7)
    axes[0, 0].plot(self.clipping_history['steps'], self.clipping_history['clipped_norms'], 
                    label='Clipped Norm', alpha=0.7)
    axes[0, 0].set_title('Gradient Norms Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Norm')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Plot clipping ratios
    axes[0, 1].plot(self.clipping_history['steps'], self.clipping_history['clipping_ratios'], 
                    label='Clipping Ratio', color='red')
    axes[0, 1].set_title('Clipping Ratios Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Clipping Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot clipping ratio distribution
    axes[1, 0].hist(self.clipping_history['clipping_ratios'], bins=30, alpha=0.7, 
                    edgecolor='black', color='red')
    axes[1, 0].set_title('Clipping Ratio Distribution')
    axes[1, 0].set_xlabel('Clipping Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Plot gradient norm distribution
    axes[1, 1].hist(self.clipping_history['gradient_norms'], bins=30, alpha=0.7, 
                    edgecolor='black', color='blue')
    axes[1, 1].set_title('Gradient Norm Distribution')
    axes[1, 1].set_xlabel('Gradient Norm')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Clipping history plot saved to {save_path}")
    
    plt.show()
```

### 2. NaN History Visualization

```python
def plot_nan_history(self, save_path: Optional[str] = None):
    """Plot NaN handling history."""
    if not self.nan_history['steps']:
        self.logger.warning("No NaN history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot NaN/Inf/Overflow counts over time
    axes[0, 0].plot(self.nan_history['steps'], self.nan_history['nan_counts'], 
                    label='NaN', alpha=0.7, color='red')
    axes[0, 0].plot(self.nan_history['steps'], self.nan_history['inf_counts'], 
                    label='Inf', alpha=0.7, color='orange')
    axes[0, 0].plot(self.nan_history['steps'], self.nan_history['overflow_counts'], 
                    label='Overflow', alpha=0.7, color='yellow')
    axes[0, 0].set_title('Numerical Issues Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot cumulative counts
    cumulative_nan = np.cumsum(self.nan_history['nan_counts'])
    cumulative_inf = np.cumsum(self.nan_history['inf_counts'])
    cumulative_overflow = np.cumsum(self.nan_history['overflow_counts'])
    
    axes[0, 1].plot(self.nan_history['steps'], cumulative_nan, 
                    label='Cumulative NaN', color='red')
    axes[0, 1].plot(self.nan_history['steps'], cumulative_inf, 
                    label='Cumulative Inf', color='orange')
    axes[0, 1].plot(self.nan_history['steps'], cumulative_overflow, 
                    label='Cumulative Overflow', color='yellow')
    axes[0, 1].set_title('Cumulative Numerical Issues')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Cumulative Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot handling actions distribution
    action_counts = {}
    for action in self.nan_history['handling_actions']:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    if action_counts:
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        axes[1, 0].bar(actions, counts, alpha=0.7)
        axes[1, 0].set_title('Handling Actions Distribution')
        axes[1, 0].set_xlabel('Action')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True)
    
    # Plot numerical issues distribution
    total_issues = np.array(self.nan_history['nan_counts']) + \
                  np.array(self.nan_history['inf_counts']) + \
                  np.array(self.nan_history['overflow_counts'])
    
    axes[1, 1].hist(total_issues, bins=range(int(max(total_issues)) + 2), 
                    alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Total Numerical Issues Distribution')
    axes[1, 1].set_xlabel('Total Issues per Step')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"NaN history plot saved to {save_path}")
    
    plt.show()
```

### 3. Stability History Visualization

```python
def plot_stability_history(self, save_path: Optional[str] = None):
    """Plot numerical stability history."""
    if not self.stability_history['steps']:
        self.logger.warning("No stability history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot stability scores
    axes[0, 0].plot(self.stability_history['steps'], self.stability_history['stability_scores'], 
                    label='Stability Score', color='green')
    axes[0, 0].set_title('Numerical Stability Score Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Stability Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot clipping ratios
    clipping_ratios = [stats.get('clipping_ratio', 0.0) for stats in self.stability_history['clipping_stats']]
    axes[0, 1].plot(self.stability_history['steps'], clipping_ratios, 
                    label='Clipping Ratio', color='blue')
    axes[0, 1].set_title('Gradient Clipping Ratio Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Clipping Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot numerical issues
    nan_counts = [stats['nan_detected'] for stats in self.stability_history['nan_stats']]
    inf_counts = [stats['inf_detected'] for stats in self.stability_history['nan_stats']]
    overflow_counts = [stats['overflow_detected'] for stats in self.stability_history['nan_stats']]
    
    axes[1, 0].plot(self.stability_history['steps'], nan_counts, 
                    label='NaN', color='red', alpha=0.7)
    axes[1, 0].plot(self.stability_history['steps'], inf_counts, 
                    label='Inf', color='orange', alpha=0.7)
    axes[1, 0].plot(self.stability_history['steps'], overflow_counts, 
                    label='Overflow', color='yellow', alpha=0.7)
    axes[1, 0].set_title('Numerical Issues Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot stability score distribution
    axes[1, 1].hist(self.stability_history['stability_scores'], bins=20, alpha=0.7, 
                    edgecolor='black', color='green')
    axes[1, 1].set_title('Stability Score Distribution')
    axes[1, 1].set_xlabel('Stability Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Stability history plot saved to {save_path}")
    
    plt.show()
```

## Conclusion

The gradient clipping and NaN/Inf value handling system provides:

1. **Comprehensive Gradient Clipping**: Multiple clipping strategies (norm, value, global norm, adaptive)
2. **Advanced NaN/Inf Handling**: Detection, replacement, skipping, and gradient zeroing
3. **Numerical Stability Management**: Integrated stability monitoring and scoring
4. **Flexible Configuration**: Extensive configuration options for different use cases
5. **Monitoring and Visualization**: Complete plotting and analysis capabilities
6. **Error Handling**: Robust error handling and logging
7. **Performance Optimization**: Efficient computation and memory management
8. **Production Ready**: Comprehensive logging, error handling, and documentation
9. **Multiple Strategies**: Different approaches for different scenarios
10. **History Tracking**: Complete history tracking and analysis
11. **Stability Scoring**: Quantitative stability assessment
12. **Best Practices**: Guidelines for configuration and usage

This system ensures numerical stability in deep learning training, preventing gradient explosion and handling numerical issues effectively across different domains and applications. 