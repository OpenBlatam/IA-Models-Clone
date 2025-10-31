"""
Interface Definitions
Abstract interfaces for modular architecture
"""

import abc
from typing import Dict, Any, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass
import torch
import torch.nn as nn

class IConfigurable(Protocol):
    """Interface for configurable components"""
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the component"""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        ...

class ILoggable(Protocol):
    """Interface for loggable components"""
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message"""
        ...
    
    def get_logger(self) -> Any:
        """Get logger instance"""
        ...

class IMeasurable(Protocol):
    """Interface for measurable components"""
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        ...
    
    def reset_metrics(self) -> None:
        """Reset metrics"""
        ...

class IPlugin(Protocol):
    """Interface for plugin components"""
    
    def get_name(self) -> str:
        """Get plugin name"""
        ...
    
    def get_version(self) -> str:
        """Get plugin version"""
        ...
    
    def get_dependencies(self) -> List[str]:
        """Get plugin dependencies"""
        ...
    
    def is_compatible(self, version: str) -> bool:
        """Check compatibility"""
        ...

class IOptimizer(Protocol):
    """Interface for optimization components"""
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Optimize a model"""
        ...
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information"""
        ...
    
    def can_optimize(self, model: nn.Module) -> bool:
        """Check if model can be optimized"""
        ...

class IModel(Protocol):
    """Interface for model components"""
    
    def load(self, path: Optional[str] = None) -> nn.Module:
        """Load a model"""
        ...
    
    def save(self, model: nn.Module, path: str) -> bool:
        """Save a model"""
        ...
    
    def create(self, config: Dict[str, Any]) -> nn.Module:
        """Create a new model"""
        ...
    
    def get_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information"""
        ...

class ITrainer(Protocol):
    """Interface for training components"""
    
    def setup(self, model: nn.Module, train_data: Any, val_data: Optional[Any] = None) -> bool:
        """Setup training"""
        ...
    
    def train(self, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        ...
    
    def validate(self, model: nn.Module, data: Any) -> Dict[str, Any]:
        """Validate the model"""
        ...
    
    def save_checkpoint(self, path: str) -> bool:
        """Save training checkpoint"""
        ...
    
    def load_checkpoint(self, path: str) -> bool:
        """Load training checkpoint"""
        ...

class IInferencer(Protocol):
    """Interface for inference components"""
    
    def load_model(self, model: nn.Module, tokenizer: Optional[Any] = None) -> None:
        """Load model for inference"""
        ...
    
    def generate(self, input_data: Union[str, List[int]], **kwargs) -> Dict[str, Any]:
        """Generate output"""
        ...
    
    def batch_generate(self, inputs: List[Union[str, List[int]]], **kwargs) -> List[Dict[str, Any]]:
        """Generate for multiple inputs"""
        ...
    
    def optimize_for_inference(self) -> None:
        """Optimize for inference"""
        ...

class IMonitor(Protocol):
    """Interface for monitoring components"""
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring"""
        ...
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        ...
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric"""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        ...
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive report"""
        ...

class IBenchmarker(Protocol):
    """Interface for benchmarking components"""
    
    def benchmark_model(self, model: nn.Module, test_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Benchmark a model"""
        ...
    
    def compare_models(self, models: Dict[str, nn.Module], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple models"""
        ...
    
    def benchmark_optimization(self, model: nn.Module, optimizers: List[IOptimizer], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark different optimizations"""
        ...

class IDataProcessor(Protocol):
    """Interface for data processing components"""
    
    def process(self, data: Any, **kwargs) -> Any:
        """Process data"""
        ...
    
    def preprocess(self, data: Any) -> Any:
        """Preprocess data"""
        ...
    
    def postprocess(self, data: Any) -> Any:
        """Postprocess data"""
        ...

class IMemoryManager(Protocol):
    """Interface for memory management components"""
    
    def allocate(self, size: int) -> bool:
        """Allocate memory"""
        ...
    
    def deallocate(self, size: int) -> bool:
        """Deallocate memory"""
        ...
    
    def get_usage(self) -> Dict[str, Any]:
        """Get memory usage"""
        ...
    
    def cleanup(self) -> None:
        """Cleanup memory"""
        ...

class IDeviceManager(Protocol):
    """Interface for device management components"""
    
    def get_available_devices(self) -> List[str]:
        """Get available devices"""
        ...
    
    def select_device(self, preference: Optional[str] = None) -> str:
        """Select best device"""
        ...
    
    def move_to_device(self, model: nn.Module, device: str) -> nn.Module:
        """Move model to device"""
        ...

class IConfigValidator(Protocol):
    """Interface for configuration validation"""
    
    def validate(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        ...
    
    def get_errors(self) -> List[str]:
        """Get validation errors"""
        ...
    
    def get_schema(self, component_type: str) -> Dict[str, Any]:
        """Get schema for component type"""
        ...

class IEventEmitter(Protocol):
    """Interface for event emission"""
    
    def emit(self, event: str, data: Any) -> None:
        """Emit an event"""
        ...
    
    def on(self, event: str, callback: Callable) -> None:
        """Register event callback"""
        ...
    
    def off(self, event: str, callback: Callable) -> None:
        """Unregister event callback"""
        ...

class ISerializer(Protocol):
    """Interface for serialization"""
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object"""
        ...
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data"""
        ...
    
    def get_format(self) -> str:
        """Get serialization format"""
        ...

class ILoader(Protocol):
    """Interface for loading components"""
    
    def load(self, path: str) -> Any:
        """Load from path"""
        ...
    
    def can_load(self, path: str) -> bool:
        """Check if path can be loaded"""
        ...
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats"""
        ...

class ISaver(Protocol):
    """Interface for saving components"""
    
    def save(self, obj: Any, path: str) -> bool:
        """Save to path"""
        ...
    
    def can_save(self, obj: Any) -> bool:
        """Check if object can be saved"""
        ...
    
    def get_supported_formats(self) -> List[str]:
        """Get supported formats"""
        ...

# Composite interfaces
class IOptimizationModule(IOptimizer, IConfigurable, ILoggable, IMeasurable, Protocol):
    """Complete optimization module interface"""
    pass

class IModelModule(IModel, IConfigurable, ILoggable, IMeasurable, Protocol):
    """Complete model module interface"""
    pass

class ITrainingModule(ITrainer, IConfigurable, ILoggable, IMeasurable, Protocol):
    """Complete training module interface"""
    pass

class IInferenceModule(IInferencer, IConfigurable, ILoggable, IMeasurable, Protocol):
    """Complete inference module interface"""
    pass

class IMonitoringModule(IMonitor, IConfigurable, ILoggable, IMeasurable, Protocol):
    """Complete monitoring module interface"""
    pass

class IBenchmarkingModule(IBenchmarker, IConfigurable, ILoggable, IMeasurable, Protocol):
    """Complete benchmarking module interface"""
    pass

